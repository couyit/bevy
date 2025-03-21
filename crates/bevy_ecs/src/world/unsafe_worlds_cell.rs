//! Contains types that allow disjoint mutable access to a [`World`].

use super::{Mut, Ref, Storage, World, WorldId, Worlds};
use crate::{
    archetype::{Archetype, Archetypes},
    bundle::Bundles,
    change_detection::{MaybeLocation, MutUntyped, Ticks, TicksMut},
    component::{ComponentId, ComponentTicks, Components, Mutable, StorageType, Tick, TickCells},
    entity::{Entities, Entity, EntityBorrow, EntityDoesNotExistError, EntityLocation},
    observer::Observers,
    prelude::Component,
    query::{DebugCheckedUnwrap, ReadOnlyQueryData},
    removal_detection::RemovedComponentEvents,
    resource::Resource,
    storage::{ComponentSparseSet, Table},
    world::RawCommandQueue,
};
use bevy_platform_support::sync::atomic::Ordering;
use bevy_ptr::{Ptr, UnsafeCellDeref};
use core::{any::TypeId, cell::UnsafeCell, fmt::Debug, marker::PhantomData, panic::Location, ptr};
use thiserror::Error;

/// Variant of the [`World`] where resource and component accesses take `&self`, and the responsibility to avoid
/// aliasing violations are given to the caller instead of being checked at compile-time by rust's unique XOR shared rule.
///
/// ### Rationale
/// In rust, having a `&mut World` means that there are absolutely no other references to the safe world alive at the same time,
/// without exceptions. Not even unsafe code can change this.
///
/// But there are situations where careful shared mutable access through a type is possible and safe. For this, rust provides the [`UnsafeCell`]
/// escape hatch, which allows you to get a `*mut T` from a `&UnsafeCell<T>` and around which safe abstractions can be built.
///
/// Access to resources and components can be done uniquely using [`World::resource_mut`] and [`World::entity_mut`], and shared using [`World::resource`] and [`World::entity`].
/// These methods use lifetimes to check at compile time that no aliasing rules are being broken.
///
/// This alone is not enough to implement bevy systems where multiple systems can access *disjoint* parts of the world concurrently. For this, bevy stores all values of
/// resources and components (and [`ComponentTicks`]) in [`UnsafeCell`]s, and carefully validates disjoint access patterns using
/// APIs like [`System::component_access`](crate::system::System::component_access).
///
/// A system then can be executed using [`System::run_unsafe`](crate::system::System::run_unsafe) with a `&World` and use methods with interior mutability to access resource values.
///
/// ### Example Usage
///
/// [`UnsafeWorldCell`] can be used as a building block for writing APIs that safely allow disjoint access into the world.
/// In the following example, the world is split into a resource access half and a component access half, where each one can
/// safely hand out mutable references.
///
/// ```
/// use bevy_ecs::world::World;
/// use bevy_ecs::change_detection::Mut;
/// use bevy_ecs::resource::Resource;
/// use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
///
/// // INVARIANT: existence of this struct means that users of it are the only ones being able to access resources in the world
/// struct OnlyResourceAccessWorld<'w>(UnsafeWorldCell<'w>);
/// // INVARIANT: existence of this struct means that users of it are the only ones being able to access components in the world
/// struct OnlyComponentAccessWorld<'w>(UnsafeWorldCell<'w>);
///
/// impl<'w> OnlyResourceAccessWorld<'w> {
///     fn get_resource_mut<T: Resource>(&mut self) -> Option<Mut<'_, T>> {
///         // SAFETY: resource access is allowed through this UnsafeWorldCell
///         unsafe { self.0.get_resource_mut::<T>() }
///     }
/// }
/// // impl<'w> OnlyComponentAccessWorld<'w> {
/// //     ...
/// // }
///
/// // the two `UnsafeWorldCell`s borrow from the `&mut World`, so it cannot be accessed while they are live
/// fn split_world_access(world: &mut World) -> (OnlyResourceAccessWorld<'_>, OnlyComponentAccessWorld<'_>) {
///     let unsafe_world_cell = world.as_unsafe_world_cell();
///     let resource_access = OnlyResourceAccessWorld(unsafe_world_cell);
///     let component_access = OnlyComponentAccessWorld(unsafe_world_cell);
///     (resource_access, component_access)
/// }
/// ```
#[derive(Copy, Clone)]
pub struct UnsafeWorldsCell<'w> {
    ptr: *mut Worlds,
    #[cfg(debug_assertions)]
    allows_mutable_access: bool,
    _marker: PhantomData<(&'w Worlds, &'w UnsafeCell<Worlds>)>,
}

// SAFETY: `&World` and `&mut World` are both `Send`
unsafe impl Send for UnsafeWorldsCell<'_> {}
// SAFETY: `&World` and `&mut World` are both `Sync`
unsafe impl Sync for UnsafeWorldsCell<'_> {}

impl<'w> From<&'w mut Worlds> for UnsafeWorldsCell<'w> {
    fn from(value: &'w mut Worlds) -> Self {
        value.as_unsafe_world_cell()
    }
}

impl<'w> From<&'w Worlds> for UnsafeWorldsCell<'w> {
    fn from(value: &'w Worlds) -> Self {
        value.as_unsafe_world_cell_readonly()
    }
}

impl<'w> UnsafeWorldCell<'w> {
    /// Creates a [`UnsafeWorldCell`] that can be used to access everything immutably
    #[inline]
    pub(crate) fn new_readonly(world: &'w World) -> Self {
        Self {
            ptr: ptr::from_ref(world).cast_mut(),
            #[cfg(debug_assertions)]
            allows_mutable_access: false,
            _marker: PhantomData,
        }
    }

    /// Creates [`UnsafeWorldCell`] that can be used to access everything mutably
    #[inline]
    pub(crate) fn new_mutable(world: &'w mut World) -> Self {
        Self {
            ptr: ptr::from_mut(world),
            #[cfg(debug_assertions)]
            allows_mutable_access: true,
            _marker: PhantomData,
        }
    }

    #[cfg_attr(debug_assertions, inline(never), track_caller)]
    #[cfg_attr(not(debug_assertions), inline(always))]
    pub(crate) fn assert_allows_mutable_access(self) {
        // This annotation is needed because the
        // allows_mutable_access field doesn't exist otherwise.
        // Kinda weird, since debug_assert would never be called,
        // but CI complained in https://github.com/bevyengine/bevy/pull/17393
        #[cfg(debug_assertions)]
        debug_assert!(
            self.allows_mutable_access,
            "mutating world data via `World::as_unsafe_world_cell_readonly` is forbidden"
        );
    }

    /// Gets a mutable reference to the [`World`] this [`UnsafeWorldCell`] belongs to.
    /// This is an incredibly error-prone operation and is only valid in a small number of circumstances.
    ///
    /// Calling this method implies mutable access to the *whole* world (see first point on safety section
    /// below), which includes all entities, components, and resources. Notably, calling this on
    /// [`WorldQuery::init_fetch`](crate::query::WorldQuery::init_fetch) and
    /// [`SystemParam::get_param`](crate::system::SystemParam::get_param) are most likely *unsound* unless
    /// you can prove that the underlying [`World`] is exclusive, which in normal circumstances is not.
    ///
    /// # Safety
    /// - `self` must have been obtained from a call to [`World::as_unsafe_world_cell`]
    ///   (*not* `as_unsafe_world_cell_readonly` or any other method of construction that
    ///   does not provide mutable access to the entire world).
    ///   - This means that if you have an `UnsafeWorldCell` that you didn't create yourself,
    ///     it is likely *unsound* to call this method.
    /// - The returned `&mut World` *must* be unique: it must never be allowed to exist
    ///   at the same time as any other borrows of the world or any accesses to its data.
    ///   This includes safe ways of accessing world data, such as [`UnsafeWorldCell::archetypes`].
    ///   - Note that the `&mut World` *may* exist at the same time as instances of `UnsafeWorldCell`,
    ///     so long as none of those instances are used to access world data in any way
    ///     while the mutable borrow is active.
    ///
    /// [//]: # (This test fails miri.)
    /// ```no_run
    /// # use bevy_ecs::prelude::*;
    /// # #[derive(Component)] struct Player;
    /// # fn store_but_dont_use<T>(_: T) {}
    /// # let mut world = World::new();
    /// // Make an UnsafeWorldCell.
    /// let world_cell = world.as_unsafe_world_cell();
    ///
    /// // SAFETY: `world_cell` was originally created from `&mut World`.
    /// // We must be sure not to access any world data while `world_mut` is active.
    /// let world_mut = unsafe { world_cell.world_mut() };
    ///
    /// // We can still use `world_cell` so long as we don't access the world with it.
    /// store_but_dont_use(world_cell);
    ///
    /// // !!This is unsound!! Even though this method is safe, we cannot call it until
    /// // `world_mut` is no longer active.
    /// let tick = world_cell.change_tick();
    ///
    /// // Use mutable access to spawn an entity.
    /// world_mut.spawn(Player);
    ///
    /// // Since we never use `world_mut` after this, the borrow is released
    /// // and we are once again allowed to access the world using `world_cell`.
    /// let archetypes = world_cell.archetypes();
    /// ```
    #[inline]
    pub unsafe fn worlds_mut(self) -> &'w mut Worlds {
        self.assert_allows_mutable_access();
        // SAFETY:
        // - caller ensures the created `&mut World` is the only borrow of world
        unsafe { &mut *self.ptr }
    }

    /// Gets a reference to the [`&World`](World) this [`UnsafeWorldCell`] belongs to.
    /// This can be used for arbitrary shared/readonly access.
    ///
    /// # Safety
    /// - must have permission to access the whole world immutably
    /// - there must be no live exclusive borrows on world data
    /// - there must be no live exclusive borrow of world
    #[inline]
    pub unsafe fn worlds(self) -> &'w Worlds {
        // SAFETY:
        // - caller ensures there is no `&mut World` this makes it okay to make a `&World`
        // - caller ensures there is no mutable borrows of world data, this means the caller cannot
        //   misuse the returned `&World`
        unsafe { self.unsafe_worlds() }
    }

    /// Variant on [`UnsafeWorldCell::world`] solely used for implementing this type's methods.
    /// It allows having an `&World` even with live mutable borrows of components and resources
    /// so the returned `&World` should not be handed out to safe code and care should be taken
    /// when working with it.
    ///
    /// Deliberately private as the correct way to access data in a [`World`] that may have existing
    /// mutable borrows of data inside it, is to use [`UnsafeWorldCell`].
    ///
    /// # Safety
    /// - must not be used in a way that would conflict with any
    ///   live exclusive borrows on world data
    #[inline]
    unsafe fn unsafe_worlds(self) -> &'w Worlds {
        // SAFETY:
        // - caller ensures that the returned `&World` is not used in a way that would conflict
        //   with any existing mutable borrows of world data
        unsafe { &*self.ptr }
    }

    /// Retrieves this world's unique [ID](WorldId).
    #[inline]
    pub fn id(self) -> WorldsId {
        // SAFETY:
        // - we only access world metadata
        unsafe { self.unsafe_worlds() }.id()
    }
}

impl Debug for UnsafeWorldCell<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        // SAFETY: World's Debug implementation only accesses metadata.
        Debug::fmt(unsafe { self.world_metadata() }, f)
    }
}

/// A interior-mutable reference to a particular [`Entity`] and all of its components
#[derive(Copy, Clone)]
pub struct UnsafeEntityCell<'w> {
    world: UnsafeWorldCell<'w>,
    entity: Entity,
    location: EntityLocation,
}

impl<'w> UnsafeEntityCell<'w> {
    #[inline]
    pub(crate) fn new(
        world: UnsafeWorldCell<'w>,
        entity: Entity,
        location: EntityLocation,
    ) -> Self {
        UnsafeEntityCell {
            world,
            entity,
            location,
        }
    }

    /// Returns the [ID](Entity) of the current entity.
    #[inline]
    #[must_use = "Omit the .id() call if you do not need to store the `Entity` identifier."]
    pub fn id(self) -> Entity {
        self.entity
    }

    /// Gets metadata indicating the location where the current entity is stored.
    #[inline]
    pub fn location(self) -> EntityLocation {
        self.location
    }

    /// Returns the archetype that the current entity belongs to.
    #[inline]
    pub fn archetype(self) -> &'w Archetype {
        &self.world.archetypes()[self.location.archetype_id]
    }

    /// Gets the world that the current entity belongs to.
    #[inline]
    pub fn world(self) -> UnsafeWorldCell<'w> {
        self.world
    }

    /// Returns `true` if the current entity has a component of type `T`.
    /// Otherwise, this returns `false`.
    ///
    /// ## Notes
    ///
    /// If you do not know the concrete type of a component, consider using
    /// [`Self::contains_id`] or [`Self::contains_type_id`].
    #[inline]
    pub fn contains<T: Component>(self) -> bool {
        self.contains_type_id(TypeId::of::<T>())
    }

    /// Returns `true` if the current entity has a component identified by `component_id`.
    /// Otherwise, this returns false.
    ///
    /// ## Notes
    ///
    /// - If you know the concrete type of the component, you should prefer [`Self::contains`].
    /// - If you know the component's [`TypeId`] but not its [`ComponentId`], consider using
    ///   [`Self::contains_type_id`].
    #[inline]
    pub fn contains_id(self, component_id: ComponentId) -> bool {
        self.archetype().contains(component_id)
    }

    /// Returns `true` if the current entity has a component with the type identified by `type_id`.
    /// Otherwise, this returns false.
    ///
    /// ## Notes
    ///
    /// - If you know the concrete type of the component, you should prefer [`Self::contains`].
    /// - If you have a [`ComponentId`] instead of a [`TypeId`], consider using [`Self::contains_id`].
    #[inline]
    pub fn contains_type_id(self, type_id: TypeId) -> bool {
        let Some(id) = self.world.components().get_id(type_id) else {
            return false;
        };
        self.contains_id(id)
    }

    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component
    /// - no other mutable references to the component exist at the same time
    #[inline]
    pub unsafe fn get<T: Component>(self) -> Option<&'w T> {
        let component_id = self.world.components().get_id(TypeId::of::<T>())?;
        // SAFETY:
        // - `storage_type` is correct (T component_id + T::STORAGE_TYPE)
        // - `location` is valid
        // - proper aliasing is promised by caller
        unsafe {
            get_component(
                self.world,
                component_id,
                T::STORAGE_TYPE,
                self.entity,
                self.location,
            )
            // SAFETY: returned component is of type T
            .map(|value| value.deref::<T>())
        }
    }

    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component
    /// - no other mutable references to the component exist at the same time
    #[inline]
    pub unsafe fn get_ref<T: Component>(self) -> Option<Ref<'w, T>> {
        let last_change_tick = self.world.last_change_tick();
        let change_tick = self.world.change_tick();
        let component_id = self.world.components().get_id(TypeId::of::<T>())?;

        // SAFETY:
        // - `storage_type` is correct (T component_id + T::STORAGE_TYPE)
        // - `location` is valid
        // - proper aliasing is promised by caller
        unsafe {
            get_component_and_ticks(
                self.world,
                component_id,
                T::STORAGE_TYPE,
                self.entity,
                self.location,
            )
            .map(|(value, cells, caller)| Ref {
                // SAFETY: returned component is of type T
                value: value.deref::<T>(),
                ticks: Ticks::from_tick_cells(cells, last_change_tick, change_tick),
                changed_by: caller.map(|caller| caller.deref()),
            })
        }
    }

    /// Retrieves the change ticks for the given component. This can be useful for implementing change
    /// detection in custom runtimes.
    ///
    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component
    /// - no other mutable references to the component exist at the same time
    #[inline]
    pub unsafe fn get_change_ticks<T: Component>(self) -> Option<ComponentTicks> {
        let component_id = self.world.components().get_id(TypeId::of::<T>())?;

        // SAFETY:
        // - entity location is valid
        // - proper world access is promised by caller
        unsafe {
            get_ticks(
                self.world,
                component_id,
                T::STORAGE_TYPE,
                self.entity,
                self.location,
            )
        }
    }

    /// Retrieves the change ticks for the given [`ComponentId`]. This can be useful for implementing change
    /// detection in custom runtimes.
    ///
    /// **You should prefer to use the typed API [`UnsafeEntityCell::get_change_ticks`] where possible and only
    /// use this in cases where the actual component types are not known at
    /// compile time.**
    ///
    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component
    /// - no other mutable references to the component exist at the same time
    #[inline]
    pub unsafe fn get_change_ticks_by_id(
        &self,
        component_id: ComponentId,
    ) -> Option<ComponentTicks> {
        let info = self.world.components().get_info(component_id)?;
        // SAFETY:
        // - entity location and entity is valid
        // - world access is immutable, lifetime tied to `&self`
        // - the storage type provided is correct for T
        unsafe {
            get_ticks(
                self.world,
                component_id,
                info.storage_type(),
                self.entity,
                self.location,
            )
        }
    }

    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component mutably
    /// - no other references to the component exist at the same time
    #[inline]
    pub unsafe fn get_mut<T: Component<Mutability = Mutable>>(self) -> Option<Mut<'w, T>> {
        // SAFETY:
        // - trait bound `T: Component<Mutability = Mutable>` ensures component is mutable
        // - same safety requirements
        unsafe { self.get_mut_assume_mutable() }
    }

    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component mutably
    /// - no other references to the component exist at the same time
    /// - the component `T` is mutable
    #[inline]
    pub unsafe fn get_mut_assume_mutable<T: Component>(self) -> Option<Mut<'w, T>> {
        // SAFETY: same safety requirements
        unsafe {
            self.get_mut_using_ticks_assume_mutable(
                self.world.last_change_tick(),
                self.world.change_tick(),
            )
        }
    }

    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component mutably
    /// - no other references to the component exist at the same time
    /// - The component `T` is mutable
    #[inline]
    pub(crate) unsafe fn get_mut_using_ticks_assume_mutable<T: Component>(
        &self,
        last_change_tick: Tick,
        change_tick: Tick,
    ) -> Option<Mut<'w, T>> {
        self.world.assert_allows_mutable_access();

        let component_id = self.world.components().get_id(TypeId::of::<T>())?;

        // SAFETY:
        // - `storage_type` is correct
        // - `location` is valid
        // - aliasing rules are ensured by caller
        unsafe {
            get_component_and_ticks(
                self.world,
                component_id,
                T::STORAGE_TYPE,
                self.entity,
                self.location,
            )
            .map(|(value, cells, caller)| Mut {
                // SAFETY: returned component is of type T
                value: value.assert_unique().deref_mut::<T>(),
                ticks: TicksMut::from_tick_cells(cells, last_change_tick, change_tick),
                changed_by: caller.map(|caller| caller.deref_mut()),
            })
        }
    }

    /// Returns read-only components for the current entity that match the query `Q`,
    /// or `None` if the entity does not have the components required by the query `Q`.
    ///
    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the queried data immutably
    /// - no mutable references to the queried data exist at the same time
    pub(crate) unsafe fn get_components<Q: ReadOnlyQueryData>(&self) -> Option<Q::Item<'w>> {
        // SAFETY: World is only used to access query data and initialize query state
        let state = unsafe {
            let world = self.world().world();
            Q::get_state(world.components())?
        };
        let location = self.location();
        // SAFETY: Location is guaranteed to exist
        let archetype = unsafe {
            self.world
                .archetypes()
                .get(location.archetype_id)
                .debug_checked_unwrap()
        };
        if Q::matches_component_set(&state, &|id| archetype.contains(id)) {
            // SAFETY: state was initialized above using the world passed into this function
            let mut fetch = unsafe {
                Q::init_fetch(
                    self.world,
                    &state,
                    self.world.last_change_tick(),
                    self.world.change_tick(),
                )
            };
            // SAFETY: Table is guaranteed to exist
            let table = unsafe {
                self.world
                    .unsafe_world()
                    .tables()
                    .get(location.table_id)
                    .debug_checked_unwrap()
            };
            // SAFETY: Archetype and table are from the same world used to initialize state and fetch.
            // Table corresponds to archetype. State is the same state used to init fetch above.
            unsafe { Q::set_archetype(&mut fetch, &state, archetype, table) }
            // SAFETY: Called after set_archetype above. Entity and location are guaranteed to exist.
            unsafe { Some(Q::fetch(&mut fetch, self.id(), location.table_row)) }
        } else {
            None
        }
    }

    /// Gets the component of the given [`ComponentId`] from the entity.
    ///
    /// **You should prefer to use the typed API where possible and only
    /// use this in cases where the actual component types are not known at
    /// compile time.**
    ///
    /// Unlike [`UnsafeEntityCell::get`], this returns a raw pointer to the component,
    /// which is only valid while the `'w` borrow of the lifetime is active.
    ///
    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component
    /// - no other mutable references to the component exist at the same time
    #[inline]
    pub unsafe fn get_by_id(self, component_id: ComponentId) -> Option<Ptr<'w>> {
        let info = self.world.components().get_info(component_id)?;
        // SAFETY: entity_location is valid, component_id is valid as checked by the line above
        unsafe {
            get_component(
                self.world,
                component_id,
                info.storage_type(),
                self.entity,
                self.location,
            )
        }
    }

    /// Retrieves a mutable untyped reference to the given `entity`'s [`Component`] of the given [`ComponentId`].
    /// Returns `None` if the `entity` does not have a [`Component`] of the given type.
    ///
    /// **You should prefer to use the typed API [`UnsafeEntityCell::get_mut`] where possible and only
    /// use this in cases where the actual types are not known at compile time.**
    ///
    /// # Safety
    /// It is the callers responsibility to ensure that
    /// - the [`UnsafeEntityCell`] has permission to access the component mutably
    /// - no other references to the component exist at the same time
    #[inline]
    pub unsafe fn get_mut_by_id(
        self,
        component_id: ComponentId,
    ) -> Result<MutUntyped<'w>, GetEntityMutByIdError> {
        self.world.assert_allows_mutable_access();

        let info = self
            .world
            .components()
            .get_info(component_id)
            .ok_or(GetEntityMutByIdError::InfoNotFound)?;

        // If a component is immutable then a mutable reference to it doesn't exist
        if !info.mutable() {
            return Err(GetEntityMutByIdError::ComponentIsImmutable);
        }

        // SAFETY: entity_location is valid, component_id is valid as checked by the line above
        unsafe {
            get_component_and_ticks(
                self.world,
                component_id,
                info.storage_type(),
                self.entity,
                self.location,
            )
            .map(|(value, cells, caller)| MutUntyped {
                // SAFETY: world access validated by caller and ties world lifetime to `MutUntyped` lifetime
                value: value.assert_unique(),
                ticks: TicksMut::from_tick_cells(
                    cells,
                    self.world.last_change_tick(),
                    self.world.change_tick(),
                ),
                changed_by: caller.map(|caller| caller.deref_mut()),
            })
            .ok_or(GetEntityMutByIdError::ComponentNotFound)
        }
    }

    /// Returns the source code location from which this entity has been spawned.
    pub fn spawned_by(self) -> MaybeLocation {
        self.world()
            .entities()
            .entity_get_spawned_or_despawned_by(self.entity)
            .map(|o| o.unwrap())
    }
}

/// Error that may be returned when calling [`UnsafeEntityCell::get_mut_by_id`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum GetEntityMutByIdError {
    /// The [`ComponentInfo`](crate::component::ComponentInfo) could not be found.
    #[error("the `ComponentInfo` could not be found")]
    InfoNotFound,
    /// The [`Component`] is immutable. Creating a mutable reference violates its
    /// invariants.
    #[error("the `Component` is immutable")]
    ComponentIsImmutable,
    /// This [`Entity`] does not have the desired [`Component`].
    #[error("the `Component` could not be found")]
    ComponentNotFound,
}

impl<'w> UnsafeWorldCell<'w> {
    #[inline]
    /// # Safety
    /// - the returned `Table` is only used in ways that this [`UnsafeWorldCell`] has permission for.
    /// - the returned `Table` is only used in ways that would not conflict with any existing borrows of world data.
    unsafe fn fetch_table(self, location: EntityLocation) -> Option<&'w Table> {
        // SAFETY:
        // - caller ensures returned data is not misused and we have not created any borrows of component/resource data
        // - `location` contains a valid `TableId`, so getting the table won't fail
        unsafe { self.unsafe_world().tables().get(location.table_id) }
    }

    #[inline]
    /// # Safety
    /// - the returned `ComponentSparseSet` is only used in ways that this [`UnsafeWorldCell`] has permission for.
    /// - the returned `ComponentSparseSet` is only used in ways that would not conflict with any existing
    ///   borrows of world data.
    unsafe fn fetch_sparse_set(self, component_id: ComponentId) -> Option<&'w ComponentSparseSet> {
        // SAFETY: caller ensures returned data is not misused and we have not created any borrows
        // of component/resource data
        unsafe { self.unsafe_world() }
            .sparse_sets()
            .get(component_id)
    }
}

/// Get an untyped pointer to a particular [`Component`] on a particular [`Entity`] in the provided [`World`].
///
/// # Safety
/// - `location` must refer to an archetype that contains `entity`
///   the archetype
/// - `component_id` must be valid
/// - `storage_type` must accurately reflect where the components for `component_id` are stored.
/// - the caller must ensure that no aliasing rules are violated
#[inline]
unsafe fn get_component(
    world: UnsafeWorldCell<'_>,
    component_id: ComponentId,
    storage_type: StorageType,
    entity: Entity,
    location: EntityLocation,
) -> Option<Ptr<'_>> {
    // SAFETY: component_id exists and is therefore valid
    match storage_type {
        StorageType::Table => {
            let table = world.fetch_table(location)?;
            // SAFETY: archetypes only store valid table_rows and caller ensure aliasing rules
            table.get_component(component_id, location.table_row)
        }
        StorageType::SparseSet => world.fetch_sparse_set(component_id)?.get(entity),
    }
}

/// Get an untyped pointer to a particular [`Component`] and its [`ComponentTicks`]
///
/// # Safety
/// - `location` must refer to an archetype that contains `entity`
/// - `component_id` must be valid
/// - `storage_type` must accurately reflect where the components for `component_id` are stored.
/// - the caller must ensure that no aliasing rules are violated
#[inline]
unsafe fn get_component_and_ticks(
    world: UnsafeWorldCell<'_>,
    component_id: ComponentId,
    storage_type: StorageType,
    entity: Entity,
    location: EntityLocation,
) -> Option<(
    Ptr<'_>,
    TickCells<'_>,
    MaybeLocation<&UnsafeCell<&'static Location<'static>>>,
)> {
    match storage_type {
        StorageType::Table => {
            let table = world.fetch_table(location)?;

            // SAFETY: archetypes only store valid table_rows and caller ensure aliasing rules
            Some((
                table.get_component(component_id, location.table_row)?,
                TickCells {
                    added: table
                        .get_added_tick(component_id, location.table_row)
                        .debug_checked_unwrap(),
                    changed: table
                        .get_changed_tick(component_id, location.table_row)
                        .debug_checked_unwrap(),
                },
                table
                    .get_changed_by(component_id, location.table_row)
                    .map(|changed_by| changed_by.debug_checked_unwrap()),
            ))
        }
        StorageType::SparseSet => world.fetch_sparse_set(component_id)?.get_with_ticks(entity),
    }
}

/// Get an untyped pointer to the [`ComponentTicks`] on a particular [`Entity`]
///
/// # Safety
/// - `location` must refer to an archetype that contains `entity`
///   the archetype
/// - `component_id` must be valid
/// - `storage_type` must accurately reflect where the components for `component_id` are stored.
/// - the caller must ensure that no aliasing rules are violated
#[inline]
unsafe fn get_ticks(
    world: UnsafeWorldCell<'_>,
    component_id: ComponentId,
    storage_type: StorageType,
    entity: Entity,
    location: EntityLocation,
) -> Option<ComponentTicks> {
    match storage_type {
        StorageType::Table => {
            let table = world.fetch_table(location)?;
            // SAFETY: archetypes only store valid table_rows and caller ensure aliasing rules
            table.get_ticks_unchecked(component_id, location.table_row)
        }
        StorageType::SparseSet => world.fetch_sparse_set(component_id)?.get_ticks(entity),
    }
}

impl EntityBorrow for UnsafeEntityCell<'_> {
    fn entity(&self) -> Entity {
        self.id()
    }
}

#[cfg(test)]
mod tests {
    use crate::world::Worlds;

    use super::*;

    #[test]
    #[should_panic = "is forbidden"]
    fn as_unsafe_world_cell_readonly_world_mut_forbidden() {
        let worlds = Worlds::new();
        let world = worlds.get_main_world();
        let world_cell = world.as_unsafe_world_cell_readonly();
        // SAFETY: this invalid usage will be caught by a runtime panic.
        let _ = unsafe { world_cell.world_mut() };
    }

    #[derive(Resource)]
    struct R;

    #[test]
    #[should_panic = "is forbidden"]
    fn as_unsafe_world_cell_readonly_resource_mut_forbidden() {
        let mut worlds = Worlds::new();
        let world = worlds.get_resource_world_mut();
        world.insert_resource(R);
        let world_cell = world.as_unsafe_world_cell_readonly();
        // SAFETY: this invalid usage will be caught by a runtime panic.
        let _ = unsafe { world_cell.get_resource_mut::<R>() };
    }

    #[derive(Component)]
    struct C;

    #[test]
    #[should_panic = "is forbidden"]
    fn as_unsafe_world_cell_readonly_component_mut_forbidden() {
        let mut worlds = Worlds::new();
        let world = worlds.get_main_world_mut();
        let entity = world.spawn(C).id();
        let world_cell = world.as_unsafe_world_cell_readonly();
        let entity_cell = world_cell.get_entity(entity).unwrap();
        // SAFETY: this invalid usage will be caught by a runtime panic.
        let _ = unsafe { entity_cell.get_mut::<C>() };
    }
}
