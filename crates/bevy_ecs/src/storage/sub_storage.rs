use core::{
    any::TypeId,
    ops::{Index, IndexMut},
};
use std::vec::Vec;

use bevy_utils::TypeIdMap;
use log::warn;

use crate::{bundle::BundleSpawner, prelude::QueryState};

use crate::{
    archetype::{ArchetypeRow, Archetypes},
    bundle::{Bundle, BundleInfo, Bundles},
    component::{
        ComponentDescriptor, ComponentHooks, ComponentId, ComponentInfo, Components, Mutable,
        RequiredComponents, RequiredComponentsError, StorageType,
    },
    entity::{Entities, Entity, EntityLocation},
    prelude::Component,
    query::{DebugCheckedUnwrap, QueryData, QueryFilter},
    system::Commands,
    world::{
        error::{EntityFetchError, TryDespawnError},
        DeferredWorld, EntityMut, EntityRef, EntityWorldMut, Mut, SpawnBatchIter, WorldEntityFetch,
    },
};

use super::{SparseSets, Tables};

#[derive(Default)]
pub struct SubWorlds {
    pub sub_storages: Vec<SubWorldStorage>,
    pub indices: TypeIdMap<SubWorldId>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SubWorldId(pub u32);

pub struct SubWorldStorage {
    pub(crate) id: SubWorldId,
    pub(crate) archetypes: Archetypes,
    pub(crate) bundles: Bundles,
    pub(crate) components: Components,
    pub(crate) entities: Entities,
    pub(crate) tables: Tables,
    pub(crate) sparse_sets: SparseSets,
}

pub trait SubWorld: Send + Sync + 'static {}

pub struct MainSubWorld;
pub struct InvalidSubWorld;

impl SubWorld for MainSubWorld {}
impl SubWorld for InvalidSubWorld {}

impl SubWorlds {
    pub const MAIN_STORAGE: SubWorldId = SubWorldId(0);

    pub(crate) fn new() -> Self {
        let mut sub_storages = Self {
            sub_storages: Vec::with_capacity(1),
            indices: TypeIdMap::default(),
        };

        sub_storages.sub_storages.push(SubWorldStorage {
            id: SubWorldId(0),
            archetypes: Archetypes::new(),
            bundles: Default::default(),
            components: Components::default(),
            entities: Entities::new(),
            tables: Default::default(),
            sparse_sets: Default::default(),
        });

        sub_storages
            .indices
            .insert(TypeId::of::<MainSubWorld>(), SubWorldId(0));

        sub_storages
    }

    pub fn create_sub_storage<'w, T: SubWorld>(&mut self) -> SubWorldId {
        let sub_storage = SubWorldId(self.sub_storages.len() as u32);

        self.sub_storages.push(SubWorldStorage {
            id: sub_storage,
            archetypes: Archetypes::new(),
            bundles: Default::default(),
            components: Components::default(),
            entities: Entities::new(),
            tables: Default::default(),
            sparse_sets: Default::default(),
        });

        self.indices.insert(TypeId::of::<T>(), sub_storage);

        sub_storage
    }

    #[inline]
    pub(crate) fn get_2_mut(
        &mut self,
        a: SubWorldId,
        b: SubWorldId,
    ) -> (&mut SubWorldStorage, &mut SubWorldStorage) {
        if a.as_usize() > b.as_usize() {
            let (b_slice, a_slice) = self.sub_storages.split_at_mut(a.as_usize());
            (&mut a_slice[0], &mut b_slice[b.as_usize()])
        } else {
            let (a_slice, b_slice) = self.sub_storages.split_at_mut(b.as_usize());
            (&mut a_slice[a.as_usize()], &mut b_slice[0])
        }
    }
}

impl SubWorldStorage {
    /// Retrieves this world's [`Entities`] collection.
    #[inline]
    pub fn entities(&self) -> &Entities {
        &self.entities
    }

    /// Retrieves this world's [`Entities`] collection mutably.
    ///
    /// # Safety
    /// Mutable reference must not be used to put the [`Entities`] data
    /// in an invalid state for this [`World`]
    #[inline]
    pub unsafe fn entities_mut(&mut self) -> &mut Entities {
        &mut self.entities
    }

    /// Retrieves this world's [`Archetypes`] collection.
    #[inline]
    pub fn archetypes(&self) -> &Archetypes {
        &self.archetypes
    }

    /// Retrieves this world's [`Bundles`] collection.
    #[inline]
    pub fn bundles(&self) -> &Bundles {
        &self.bundles
    }

    /// Retrieves this world's [`Components`] collection.
    #[inline]
    pub fn components(&self) -> &Components {
        &self.components
    }

    /// ensures that the components in the bundle have its necessary storage initialized.
    pub fn prepare_bundle(&mut self, components: &Components, bundle: &BundleInfo) {
        for component_id in bundle.iter_contributed_components() {
            // Safety: These ids came out of the passed `components`, so they must be valid.
            let info = unsafe { components.get_info_unchecked(component_id) };
            self.prepare_component(info);
        }
    }

    /// ensures that the component has its necessary storage initialize.
    pub fn prepare_component(&mut self, component: &ComponentInfo) {
        match component.storage_type() {
            StorageType::Table => {
                // table needs no preparation
            }
            StorageType::SparseSet => {
                self.sparse_sets.get_or_insert(component);
            }
        }
    }
    /// Creates a new [`Commands`] instance that writes to the world's command queue
    /// Use [`World::flush`] to apply all queued commands
    #[inline]
    pub fn commands(&mut self) -> Commands {
        // SAFETY: command_queue is stored on world and always valid while the world exists
        unsafe { Commands::new_raw_from_entities(self.command_queue.clone(), &self.entities) }
    }

    /// Registers a new [`Component`] type and returns the [`ComponentId`] created for it.
    pub fn register_component<T: Component>(&mut self) -> ComponentId {
        self.components.register_component::<T>()
    }

    /// Returns a mutable reference to the [`ComponentHooks`] for a [`Component`] type.
    ///
    /// Will panic if `T` exists in any archetypes.
    pub fn register_component_hooks<T: Component>(&mut self) -> &mut ComponentHooks {
        let index = self.register_component::<T>();
        assert!(!self.archetypes.archetypes.iter().any(|a| a.contains(index)), "Components hooks cannot be modified if the component already exists in an archetype, use register_component if {} may already be in use", core::any::type_name::<T>());
        // SAFETY: We just created this component
        unsafe { self.components.get_hooks_mut(index).debug_checked_unwrap() }
    }

    /// Returns a mutable reference to the [`ComponentHooks`] for a [`Component`] with the given id if it exists.
    ///
    /// Will panic if `id` exists in any archetypes.
    pub fn register_component_hooks_by_id(
        &mut self,
        id: ComponentId,
    ) -> Option<&mut ComponentHooks> {
        assert!(!self.archetypes.archetypes.iter().any(|a| a.contains(id)), "Components hooks cannot be modified if the component already exists in an archetype, use register_component if the component with id {:?} may already be in use", id);
        self.components.get_hooks_mut(id)
    }

    /// Registers the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The [`Default`] `constructor` will be used for the creation of `R`.
    /// If a custom constructor is desired, use [`World::register_required_components_with`] instead.
    ///
    /// For the non-panicking version, see [`World::try_register_required_components`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Panics
    ///
    /// Panics if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut world = World::default();
    /// // Register B as required by A and C as required by B.
    /// world.register_required_components::<A, B>();
    /// world.register_required_components::<B, C>();
    ///
    /// // This will implicitly also insert B and C with their Default constructors.
    /// let id = world.spawn(A).id();
    /// assert_eq!(&B(0), world.entity(id).get::<B>().unwrap());
    /// assert_eq!(&C(0), world.entity(id).get::<C>().unwrap());
    /// ```
    pub fn register_required_components<T: Component, R: Component + Default>(&mut self) {
        self.try_register_required_components::<T, R>().unwrap();
    }

    /// Registers the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The given `constructor` will be used for the creation of `R`.
    /// If a [`Default`] constructor is desired, use [`World::register_required_components`] instead.
    ///
    /// For the non-panicking version, see [`World::try_register_required_components_with`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Panics
    ///
    /// Panics if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut world = World::default();
    /// // Register B and C as required by A and C as required by B.
    /// // A requiring C directly will overwrite the indirect requirement through B.
    /// world.register_required_components::<A, B>();
    /// world.register_required_components_with::<B, C>(|| C(1));
    /// world.register_required_components_with::<A, C>(|| C(2));
    ///
    /// // This will implicitly also insert B with its Default constructor and C
    /// // with the custom constructor defined by A.
    /// let id = world.spawn(A).id();
    /// assert_eq!(&B(0), world.entity(id).get::<B>().unwrap());
    /// assert_eq!(&C(2), world.entity(id).get::<C>().unwrap());
    /// ```
    pub fn register_required_components_with<T: Component, R: Component>(
        &mut self,
        constructor: fn() -> R,
    ) {
        self.try_register_required_components_with::<T, R>(constructor)
            .unwrap();
    }

    /// Tries to register the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The [`Default`] `constructor` will be used for the creation of `R`.
    /// If a custom constructor is desired, use [`World::register_required_components_with`] instead.
    ///
    /// For the panicking version, see [`World::register_required_components`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Errors
    ///
    /// Returns a [`RequiredComponentsError`] if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut world = World::default();
    /// // Register B as required by A and C as required by B.
    /// world.register_required_components::<A, B>();
    /// world.register_required_components::<B, C>();
    ///
    /// // Duplicate registration! This will fail.
    /// assert!(world.try_register_required_components::<A, B>().is_err());
    ///
    /// // This will implicitly also insert B and C with their Default constructors.
    /// let id = world.spawn(A).id();
    /// assert_eq!(&B(0), world.entity(id).get::<B>().unwrap());
    /// assert_eq!(&C(0), world.entity(id).get::<C>().unwrap());
    /// ```
    pub fn try_register_required_components<T: Component, R: Component + Default>(
        &mut self,
    ) -> Result<(), RequiredComponentsError> {
        self.try_register_required_components_with::<T, R>(R::default)
    }

    /// Tries to register the given component `R` as a [required component] for `T`.
    ///
    /// When `T` is added to an entity, `R` and its own required components will also be added
    /// if `R` was not already provided. The given `constructor` will be used for the creation of `R`.
    /// If a [`Default`] constructor is desired, use [`World::register_required_components`] instead.
    ///
    /// For the panicking version, see [`World::register_required_components_with`].
    ///
    /// Note that requirements must currently be registered before `T` is inserted into the world
    /// for the first time. This limitation may be fixed in the future.
    ///
    /// [required component]: Component#required-components
    ///
    /// # Errors
    ///
    /// Returns a [`RequiredComponentsError`] if `R` is already a directly required component for `T`, or if `T` has ever been added
    /// on an entity before the registration.
    ///
    /// Indirect requirements through other components are allowed. In those cases, any existing requirements
    /// will only be overwritten if the new requirement is more specific.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct A;
    ///
    /// #[derive(Component, Default, PartialEq, Eq, Debug)]
    /// struct B(usize);
    ///
    /// #[derive(Component, PartialEq, Eq, Debug)]
    /// struct C(u32);
    ///
    /// # let mut world = World::default();
    /// // Register B and C as required by A and C as required by B.
    /// // A requiring C directly will overwrite the indirect requirement through B.
    /// world.register_required_components::<A, B>();
    /// world.register_required_components_with::<B, C>(|| C(1));
    /// world.register_required_components_with::<A, C>(|| C(2));
    ///
    /// // Duplicate registration! Even if the constructors were different, this would fail.
    /// assert!(world.try_register_required_components_with::<B, C>(|| C(1)).is_err());
    ///
    /// // This will implicitly also insert B with its Default constructor and C
    /// // with the custom constructor defined by A.
    /// let id = world.spawn(A).id();
    /// assert_eq!(&B(0), world.entity(id).get::<B>().unwrap());
    /// assert_eq!(&C(2), world.entity(id).get::<C>().unwrap());
    /// ```
    pub fn try_register_required_components_with<T: Component, R: Component>(
        &mut self,
        constructor: fn() -> R,
    ) -> Result<(), RequiredComponentsError> {
        let requiree = self.register_component::<T>();

        // TODO: Remove this panic and update archetype edges accordingly when required components are added
        if self.archetypes().component_index().contains_key(&requiree) {
            return Err(RequiredComponentsError::ArchetypeExists(requiree));
        }

        let required = self.register_component::<R>();

        // SAFETY: We just created the `required` and `requiree` components.
        unsafe {
            self.components
                .register_required_components::<R>(requiree, required, constructor)
        }
    }

    /// Retrieves the [required components](RequiredComponents) for the given component type, if it exists.
    pub fn get_required_components<C: Component>(&self) -> Option<&RequiredComponents> {
        let id = self.components().component_id::<C>()?;
        let component_info = self.components().get_info(id)?;
        Some(component_info.required_components())
    }

    /// Retrieves the [required components](RequiredComponents) for the component of the given [`ComponentId`], if it exists.
    pub fn get_required_components_by_id(&self, id: ComponentId) -> Option<&RequiredComponents> {
        let component_info = self.components().get_info(id)?;
        Some(component_info.required_components())
    }

    /// Registers a new [`Component`] type and returns the [`ComponentId`] created for it.
    ///
    /// This method differs from [`World::register_component`] in that it uses a [`ComponentDescriptor`]
    /// to register the new component type instead of statically available type information. This
    /// enables the dynamic registration of new component definitions at runtime for advanced use cases.
    ///
    /// While the option to register a component from a descriptor is useful in type-erased
    /// contexts, the standard [`World::register_component`] function should always be used instead
    /// when type information is available at compile time.
    pub fn register_component_with_descriptor(
        &mut self,
        descriptor: ComponentDescriptor,
    ) -> ComponentId {
        self.components
            .register_component_with_descriptor(descriptor)
    }

    /// Returns the [`ComponentId`] of the given [`Component`] type `T`.
    ///
    /// The returned `ComponentId` is specific to the `World` instance
    /// it was retrieved from and should not be used with another `World` instance.
    ///
    /// Returns [`None`] if the `Component` type has not yet been initialized within
    /// the `World` using [`World::register_component`].
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    ///
    /// let mut world = World::new();
    ///
    /// #[derive(Component)]
    /// struct ComponentA;
    ///
    /// let component_a_id = world.register_component::<ComponentA>();
    ///
    /// assert_eq!(component_a_id, world.component_id::<ComponentA>().unwrap())
    /// ```
    ///
    /// # See also
    ///
    /// * [`Components::component_id()`]
    /// * [`Components::get_id()`]
    #[inline]
    pub fn component_id<T: Component>(&self) -> Option<ComponentId> {
        self.components.component_id::<T>()
    }

    /// Returns [`EntityRef`]s that expose read-only operations for the given
    /// `entities`. This will panic if any of the given entities do not exist. Use
    /// [`World::get_entity`] if you want to check for entity existence instead
    /// of implicitly panicking.
    ///
    /// This function supports fetching a single entity or multiple entities:
    /// - Pass an [`Entity`] to receive a single [`EntityRef`].
    /// - Pass a slice of [`Entity`]s to receive a [`Vec<EntityRef>`].
    /// - Pass an array of [`Entity`]s to receive an equally-sized array of [`EntityRef`]s.
    ///
    /// # Panics
    ///
    /// If any of the given `entities` do not exist in the world.
    ///
    /// # Examples
    ///
    /// ## Single [`Entity`]
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    ///
    /// let position = world.entity(entity).get::<Position>().unwrap();
    /// assert_eq!(position.x, 0.0);
    /// ```
    ///
    /// ## Array of [`Entity`]s
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    /// let e2 = world.spawn(Position { x: 1.0, y: 1.0 }).id();
    ///
    /// let [e1_ref, e2_ref] = world.entity([e1, e2]);
    /// let e1_position = e1_ref.get::<Position>().unwrap();
    /// assert_eq!(e1_position.x, 0.0);
    /// let e2_position = e2_ref.get::<Position>().unwrap();
    /// assert_eq!(e2_position.x, 1.0);
    /// ```
    ///
    /// ## Slice of [`Entity`]s
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e2 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e3 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    ///
    /// let ids = vec![e1, e2, e3];
    /// for eref in world.entity(&ids[..]) {
    ///     assert_eq!(eref.get::<Position>().unwrap().y, 1.0);
    /// }
    /// ```
    ///
    /// ## [`EntityHashSet`](crate::entity::hash_map::EntityHashMap)
    ///
    /// ```
    /// # use bevy_ecs::{prelude::*, entity::hash_set::EntityHashSet};
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e2 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e3 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    ///
    /// let ids = EntityHashSet::from_iter([e1, e2, e3]);
    /// for (_id, eref) in world.entity(&ids) {
    ///     assert_eq!(eref.get::<Position>().unwrap().y, 1.0);
    /// }
    /// ```
    ///
    /// [`EntityHashSet`]: crate::entity::hash_set::EntityHashSet
    #[inline]
    #[track_caller]
    pub fn entity<F: WorldEntityFetch>(&self, entities: F) -> F::Ref<'_> {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn panic_no_entity(world: &SubWorldStorage, entity: Entity) -> ! {
            panic!(
                "Entity {entity} {}",
                world.entities.entity_does_not_exist_error_details(entity)
            );
        }

        match self.get_entity(entities) {
            Ok(fetched) => fetched,
            Err(entity) => panic_no_entity(self, entity),
        }
    }

    /// Returns [`EntityMut`]s that expose read and write operations for the
    /// given `entities`. This will panic if any of the given entities do not
    /// exist. Use [`World::get_entity_mut`] if you want to check for entity
    /// existence instead of implicitly panicking.
    ///
    /// This function supports fetching a single entity or multiple entities:
    /// - Pass an [`Entity`] to receive a single [`EntityWorldMut`].
    ///    - This reference type allows for structural changes to the entity,
    ///      such as adding or removing components, or despawning the entity.
    /// - Pass a slice of [`Entity`]s to receive a [`Vec<EntityMut>`].
    /// - Pass an array of [`Entity`]s to receive an equally-sized array of [`EntityMut`]s.
    /// - Pass a reference to a [`EntityHashSet`](crate::entity::hash_map::EntityHashMap) to receive an
    ///   [`EntityHashMap<EntityMut>`](crate::entity::hash_map::EntityHashMap).
    ///
    /// In order to perform structural changes on the returned entity reference,
    /// such as adding or removing components, or despawning the entity, only a
    /// single [`Entity`] can be passed to this function. Allowing multiple
    /// entities at the same time with structural access would lead to undefined
    /// behavior, so [`EntityMut`] is returned when requesting multiple entities.
    ///
    /// # Panics
    ///
    /// If any of the given `entities` do not exist in the world.
    ///
    /// # Examples
    ///
    /// ## Single [`Entity`]
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    ///
    /// let mut entity_mut = world.entity_mut(entity);
    /// let mut position = entity_mut.get_mut::<Position>().unwrap();
    /// position.y = 1.0;
    /// assert_eq!(position.x, 0.0);
    /// entity_mut.despawn();
    /// # assert!(world.get_entity_mut(entity).is_err());
    /// ```
    ///
    /// ## Array of [`Entity`]s
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    /// let e2 = world.spawn(Position { x: 1.0, y: 1.0 }).id();
    ///
    /// let [mut e1_ref, mut e2_ref] = world.entity_mut([e1, e2]);
    /// let mut e1_position = e1_ref.get_mut::<Position>().unwrap();
    /// e1_position.x = 1.0;
    /// assert_eq!(e1_position.x, 1.0);
    /// let mut e2_position = e2_ref.get_mut::<Position>().unwrap();
    /// e2_position.x = 2.0;
    /// assert_eq!(e2_position.x, 2.0);
    /// ```
    ///
    /// ## Slice of [`Entity`]s
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e2 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e3 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    ///
    /// let ids = vec![e1, e2, e3];
    /// for mut eref in world.entity_mut(&ids[..]) {
    ///     let mut pos = eref.get_mut::<Position>().unwrap();
    ///     pos.y = 2.0;
    ///     assert_eq!(pos.y, 2.0);
    /// }
    /// ```
    ///
    /// ## [`EntityHashSet`](crate::entity::hash_map::EntityHashMap)
    ///
    /// ```
    /// # use bevy_ecs::{prelude::*, entity::hash_set::EntityHashSet};
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e2 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    /// let e3 = world.spawn(Position { x: 0.0, y: 1.0 }).id();
    ///
    /// let ids = EntityHashSet::from_iter([e1, e2, e3]);
    /// for (_id, mut eref) in world.entity_mut(&ids) {
    ///     let mut pos = eref.get_mut::<Position>().unwrap();
    ///     pos.y = 2.0;
    ///     assert_eq!(pos.y, 2.0);
    /// }
    /// ```
    ///
    /// [`EntityHashSet`]: crate::entity::hash_set::EntityHashSet
    #[inline]
    #[track_caller]
    pub fn entity_mut<F: WorldEntityFetch>(&mut self, entities: F) -> F::Mut<'_> {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn panic_on_err(e: EntityFetchError) -> ! {
            panic!("{e}");
        }

        match self.get_entity_mut(entities) {
            Ok(fetched) => fetched,
            Err(e) => panic_on_err(e),
        }
    }

    /// Returns the components of an [`Entity`] through [`ComponentInfo`].
    #[inline]
    pub fn inspect_entity(&self, entity: Entity) -> impl Iterator<Item = &ComponentInfo> {
        let entity_location = self
            .entities()
            .get(entity)
            .unwrap_or_else(|| panic!("Entity {entity} does not exist"));

        let archetype = self
            .archetypes()
            .get(entity_location.archetype_id)
            .unwrap_or_else(|| {
                panic!(
                    "Archetype {:?} does not exist",
                    entity_location.archetype_id
                )
            });

        archetype
            .components()
            .filter_map(|id| self.components().get_info(id))
    }

    /// Returns [`EntityRef`]s that expose read-only operations for the given
    /// `entities`, returning [`Err`] if any of the given entities do not exist.
    /// Instead of immediately unwrapping the value returned from this function,
    /// prefer [`World::entity`].
    ///
    /// This function supports fetching a single entity or multiple entities:
    /// - Pass an [`Entity`] to receive a single [`EntityRef`].
    /// - Pass a slice of [`Entity`]s to receive a [`Vec<EntityRef>`].
    /// - Pass an array of [`Entity`]s to receive an equally-sized array of [`EntityRef`]s.
    /// - Pass a reference to a [`EntityHashSet`](crate::entity::hash_map::EntityHashMap) to receive an
    ///   [`EntityHashMap<EntityRef>`](crate::entity::hash_map::EntityHashMap).
    ///
    /// # Errors
    ///
    /// If any of the given `entities` do not exist in the world, the first
    /// [`Entity`] found to be missing will be returned in the [`Err`].
    ///
    /// # Examples
    ///
    /// For examples, see [`World::entity`].
    ///
    /// [`EntityHashSet`]: crate::entity::hash_set::EntityHashSet
    #[inline]
    pub fn get_entity<F: WorldEntityFetch>(&self, entities: F) -> Result<F::Ref<'_>, Entity> {
        let cell = self.as_unsafe_world_cell_readonly();
        // SAFETY: `&self` gives read access to the entire world, and prevents mutable access.
        unsafe { entities.fetch_ref(cell) }
    }

    /// Returns [`EntityMut`]s that expose read and write operations for the
    /// given `entities`, returning [`Err`] if any of the given entities do not
    /// exist. Instead of immediately unwrapping the value returned from this
    /// function, prefer [`World::entity_mut`].
    ///
    /// This function supports fetching a single entity or multiple entities:
    /// - Pass an [`Entity`] to receive a single [`EntityWorldMut`].
    ///    - This reference type allows for structural changes to the entity,
    ///      such as adding or removing components, or despawning the entity.
    /// - Pass a slice of [`Entity`]s to receive a [`Vec<EntityMut>`].
    /// - Pass an array of [`Entity`]s to receive an equally-sized array of [`EntityMut`]s.
    /// - Pass a reference to a [`EntityHashSet`](crate::entity::hash_map::EntityHashMap) to receive an
    ///   [`EntityHashMap<EntityMut>`](crate::entity::hash_map::EntityHashMap).
    ///
    /// In order to perform structural changes on the returned entity reference,
    /// such as adding or removing components, or despawning the entity, only a
    /// single [`Entity`] can be passed to this function. Allowing multiple
    /// entities at the same time with structural access would lead to undefined
    /// behavior, so [`EntityMut`] is returned when requesting multiple entities.
    ///
    /// # Errors
    ///
    /// - Returns [`EntityFetchError::NoSuchEntity`] if any of the given `entities` do not exist in the world.
    ///     - Only the first entity found to be missing will be returned.
    /// - Returns [`EntityFetchError::AliasedMutability`] if the same entity is requested multiple times.
    ///
    /// # Examples
    ///
    /// For examples, see [`World::entity_mut`].
    ///
    /// [`EntityHashSet`]: crate::entity::hash_set::EntityHashSet
    #[inline]
    pub fn get_entity_mut<F: WorldEntityFetch>(
        &mut self,
        entities: F,
    ) -> Result<F::Mut<'_>, EntityFetchError> {
        let cell = self.as_unsafe_world_cell();
        // SAFETY: `&mut self` gives mutable access to the entire world,
        // and prevents any other access to the world.
        unsafe { entities.fetch_mut(cell) }
    }

    /// Returns an [`Entity`] iterator of current entities.
    ///
    /// This is useful in contexts where you only have read-only access to the [`World`].
    #[inline]
    pub fn iter_entities(&self) -> impl Iterator<Item = EntityRef<'_>> + '_ {
        self.archetypes.iter().flat_map(|archetype| {
            archetype
                .entities()
                .iter()
                .enumerate()
                .map(|(archetype_row, archetype_entity)| {
                    let entity = archetype_entity.id();
                    let location = EntityLocation {
                        archetype_id: archetype.id(),
                        archetype_row: ArchetypeRow::new(archetype_row),
                        sub_storage: archetype.sub_storage(),
                        table_id: archetype.table_id(),
                        table_row: archetype_entity.table_row(),
                    };

                    // SAFETY: entity exists and location accurately specifies the archetype where the entity is stored.
                    let cell = UnsafeEntityCell::new(
                        self.as_unsafe_world_cell_readonly(),
                        entity,
                        location,
                    );
                    // SAFETY: `&self` gives read access to the entire world.
                    unsafe { EntityRef::new(cell) }
                })
        })
    }

    /// Returns a mutable iterator over all entities in the `World`.
    pub fn iter_entities_mut(&mut self) -> impl Iterator<Item = EntityMut<'_>> + '_ {
        let world_cell = self.as_unsafe_world_cell();
        world_cell.archetypes().iter().flat_map(move |archetype| {
            archetype
                .entities()
                .iter()
                .enumerate()
                .map(move |(archetype_row, archetype_entity)| {
                    let entity = archetype_entity.id();
                    let location = EntityLocation {
                        archetype_id: archetype.id(),
                        archetype_row: ArchetypeRow::new(archetype_row),
                        sub_storage: archetype.sub_storage(),
                        table_id: archetype.table_id(),
                        table_row: archetype_entity.table_row(),
                    };

                    // SAFETY: entity exists and location accurately specifies the archetype where the entity is stored.
                    let cell = UnsafeEntityCell::new(world_cell, entity, location);
                    // SAFETY: We have exclusive access to the entire world. We only create one borrow for each entity,
                    // so none will conflict with one another.
                    unsafe { EntityMut::new(cell) }
                })
        })
    }

    /// Spawns a new [`Entity`] and returns a corresponding [`EntityWorldMut`], which can be used
    /// to add components to the entity or retrieve its id.
    ///
    /// ```
    /// use bevy_ecs::{component::Component, world::World};
    ///
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    /// #[derive(Component)]
    /// struct Label(&'static str);
    /// #[derive(Component)]
    /// struct Num(u32);
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn_empty()
    ///     .insert(Position { x: 0.0, y: 0.0 }) // add a single component
    ///     .insert((Num(1), Label("hello"))) // add a bundle of components
    ///     .id();
    ///
    /// let position = world.entity(entity).get::<Position>().unwrap();
    /// assert_eq!(position.x, 0.0);
    /// ```
    #[track_caller]
    pub fn spawn_empty(&mut self) -> EntityWorldMut {
        self.flush();
        let entity = self.entities.alloc();
        // SAFETY: entity was just allocated
        unsafe {
            self.spawn_at_empty_internal(
                entity,
                #[cfg(feature = "track_location")]
                Location::caller(),
            )
        }
    }

    /// Spawns a new [`Entity`] with a given [`Bundle`] of [components](`Component`) and returns
    /// a corresponding [`EntityWorldMut`], which can be used to add components to the entity or
    /// retrieve its id. In case large batches of entities need to be spawned, consider using
    /// [`World::spawn_batch`] instead.
    ///
    /// ```
    /// use bevy_ecs::{bundle::Bundle, component::Component, world::World};
    ///
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// #[derive(Component)]
    /// struct Velocity {
    ///     x: f32,
    ///     y: f32,
    /// };
    ///
    /// #[derive(Component)]
    /// struct Name(&'static str);
    ///
    /// #[derive(Bundle)]
    /// struct PhysicsBundle {
    ///     position: Position,
    ///     velocity: Velocity,
    /// }
    ///
    /// let mut world = World::new();
    ///
    /// // `spawn` can accept a single component:
    /// world.spawn(Position { x: 0.0, y: 0.0 });
    ///
    /// // It can also accept a tuple of components:
    /// world.spawn((
    ///     Position { x: 0.0, y: 0.0 },
    ///     Velocity { x: 1.0, y: 1.0 },
    /// ));
    ///
    /// // Or it can accept a pre-defined Bundle of components:
    /// world.spawn(PhysicsBundle {
    ///     position: Position { x: 2.0, y: 2.0 },
    ///     velocity: Velocity { x: 0.0, y: 4.0 },
    /// });
    ///
    /// let entity = world
    ///     // Tuples can also mix Bundles and Components
    ///     .spawn((
    ///         PhysicsBundle {
    ///             position: Position { x: 2.0, y: 2.0 },
    ///             velocity: Velocity { x: 0.0, y: 4.0 },
    ///         },
    ///         Name("Elaina Proctor"),
    ///     ))
    ///     // Calling id() will return the unique identifier for the spawned entity
    ///     .id();
    /// let position = world.entity(entity).get::<Position>().unwrap();
    /// assert_eq!(position.x, 2.0);
    /// ```
    #[track_caller]
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> EntityWorldMut {
        self.spawn_with_caller(
            bundle,
            #[cfg(feature = "track_location")]
            Location::caller(),
        )
    }

    pub(crate) fn spawn_with_caller<B: Bundle>(
        &mut self,
        bundle: B,
        #[cfg(feature = "track_location")] caller: &'static Location<'static>,
    ) -> EntityWorldMut {
        self.flush();
        let change_tick = self.change_tick();
        let entity = self.entities.alloc();
        let mut bundle_spawner = BundleSpawner::new_in_sub_storage::<B>(self, change_tick, self.id);
        // SAFETY: bundle's type matches `bundle_info`, entity is allocated but non-existent
        let mut entity_location = unsafe {
            bundle_spawner.spawn_non_existent(
                entity,
                bundle,
                #[cfg(feature = "track_location")]
                caller,
            )
        };

        // SAFETY: command_queue is not referenced anywhere else
        if !unsafe { self.command_queue.is_empty() } {
            self.flush_commands();
            entity_location = self
                .entities()
                .get(entity)
                .unwrap_or(EntityLocation::INVALID);
        }

        #[cfg(feature = "track_location")]
        self.entities
            .set_spawned_or_despawned_by(entity.index(), caller);

        // SAFETY: entity and location are valid, as they were just created above
        unsafe { EntityWorldMut::new(self, entity, entity_location) }
    }

    /// # Safety
    /// must be called on an entity that was just allocated
    unsafe fn spawn_at_empty_internal(
        &mut self,
        entity: Entity,
        #[cfg(feature = "track_location")] caller: &'static Location,
    ) -> EntityWorldMut {
        let archetype = &mut self.archetypes[self.empty().clone()];
        // PERF: consider avoiding allocating entities in the empty archetype unless needed
        let table_row = self.tables[archetype.table_id()].allocate(entity);
        // SAFETY: no components are allocated by archetype.allocate() because the archetype is
        // empty
        let location = unsafe { archetype.allocate(entity, table_row) };
        self.entities.set(entity.index(), location);

        #[cfg(feature = "track_location")]
        self.entities
            .set_spawned_or_despawned_by(entity.index(), caller);

        EntityWorldMut::new(self, entity, location)
    }

    /// Spawns a batch of entities with the same component [`Bundle`] type. Takes a given
    /// [`Bundle`] iterator and returns a corresponding [`Entity`] iterator.
    /// This is more efficient than spawning entities and adding components to them individually
    /// using [`World::spawn`], but it is limited to spawning entities with the same [`Bundle`]
    /// type, whereas spawning individually is more flexible.
    ///
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World};
    ///
    /// #[derive(Component)]
    /// struct Str(&'static str);
    /// #[derive(Component)]
    /// struct Num(u32);
    ///
    /// let mut world = World::new();
    /// let entities = world.spawn_batch(vec![
    ///   (Str("a"), Num(0)), // the first entity
    ///   (Str("b"), Num(1)), // the second entity
    /// ]).collect::<Vec<Entity>>();
    ///
    /// assert_eq!(entities.len(), 2);
    /// ```
    #[track_caller]
    pub fn spawn_batch<I>(&mut self, iter: I) -> SpawnBatchIter<'_, I::IntoIter>
    where
        I: IntoIterator,
        I::Item: Bundle,
    {
        SpawnBatchIter::new(
            self,
            iter.into_iter(),
            #[cfg(feature = "track_location")]
            Location::caller(),
        )
    }

    /// Retrieves a reference to the given `entity`'s [`Component`] of the given type.
    /// Returns `None` if the `entity` does not have a [`Component`] of the given type.
    /// ```
    /// use bevy_ecs::{component::Component, world::World};
    ///
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    /// let position = world.get::<Position>(entity).unwrap();
    /// assert_eq!(position.x, 0.0);
    /// ```
    #[inline]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        self.get_entity(entity).ok()?.get()
    }

    /// Retrieves a mutable reference to the given `entity`'s [`Component`] of the given type.
    /// Returns `None` if the `entity` does not have a [`Component`] of the given type.
    /// ```
    /// use bevy_ecs::{component::Component, world::World};
    ///
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    /// let mut position = world.get_mut::<Position>(entity).unwrap();
    /// position.x = 1.0;
    /// ```
    #[inline]
    pub fn get_mut<T: Component<Mutability = Mutable>>(
        &mut self,
        entity: Entity,
    ) -> Option<Mut<T>> {
        self.get_entity_mut(entity).ok()?.into_mut()
    }

    /// Temporarily removes a [`Component`] `T` from the provided [`Entity`] and
    /// runs the provided closure on it, returning the result if `T` was available.
    /// This will trigger the `OnRemove` and `OnReplace` component hooks without
    /// causing an archetype move.
    ///
    /// This is most useful with immutable components, where removal and reinsertion
    /// is the only way to modify a value.
    ///
    /// If you do not need to ensure the above hooks are triggered, and your component
    /// is mutable, prefer using [`get_mut`](World::get_mut).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bevy_ecs::prelude::*;
    /// #
    /// #[derive(Component, PartialEq, Eq, Debug)]
    /// #[component(immutable)]
    /// struct Foo(bool);
    ///
    /// # let mut world = World::default();
    /// # world.register_component::<Foo>();
    /// #
    /// # let entity = world.spawn(Foo(false)).id();
    /// #
    /// world.modify_component(entity, |foo: &mut Foo| {
    ///     foo.0 = true;
    /// });
    /// #
    /// # assert_eq!(world.get::<Foo>(entity), Some(&Foo(true)));
    /// ```
    #[inline]
    pub fn modify_component<T: Component, R>(
        &mut self,
        entity: Entity,
        f: impl FnOnce(&mut T) -> R,
    ) -> Result<Option<R>, EntityFetchError> {
        let mut world = DeferredWorld::from(&mut *self);

        let result = match world.modify_component(entity, f) {
            Ok(result) => result,
            Err(EntityFetchError::AliasedMutability(..)) => {
                return Err(EntityFetchError::AliasedMutability(entity))
            }
            Err(EntityFetchError::NoSuchEntity(..)) => {
                return Err(EntityFetchError::NoSuchEntity(
                    entity,
                    self.entities().entity_does_not_exist_error_details(entity),
                ))
            }
        };

        self.flush();
        Ok(result)
    }

    /// Despawns the given [`Entity`], if it exists. This will also remove all of the entity's
    /// [`Components`](Component).
    ///
    /// Returns `true` if the entity is successfully despawned and `false` if
    /// the entity does not exist.
    ///
    /// # Note
    ///
    /// This will also despawn the entities in any [`RelationshipTarget`](crate::relationship::RelationshipTarget) that is configured
    /// to despawn descendants. For example, this will recursively despawn [`Children`](crate::hierarchy::Children).
    ///
    /// ```
    /// use bevy_ecs::{component::Component, world::World};
    ///
    /// #[derive(Component)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entity = world.spawn(Position { x: 0.0, y: 0.0 }).id();
    /// assert!(world.despawn(entity));
    /// assert!(world.get_entity(entity).is_err());
    /// assert!(world.get::<Position>(entity).is_none());
    /// ```
    #[track_caller]
    #[inline]
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if let Err(error) = self.despawn_with_caller(
            entity,
            #[cfg(feature = "track_location")]
            Location::caller(),
        ) {
            warn!("{error}");
            false
        } else {
            true
        }
    }

    /// Despawns the given `entity`, if it exists. This will also remove all of the entity's
    /// [`Components`](Component).
    ///
    /// Returns a [`TryDespawnError`] if the entity does not exist.
    ///
    /// # Note
    ///
    /// This will also despawn the entities in any [`RelationshipTarget`](crate::relationship::RelationshipTarget) that is configured
    /// to despawn descendants. For example, this will recursively despawn [`Children`](crate::hierarchy::Children).
    #[track_caller]
    #[inline]
    pub fn try_despawn(&mut self, entity: Entity) -> Result<(), TryDespawnError> {
        self.despawn_with_caller(
            entity,
            #[cfg(feature = "track_location")]
            Location::caller(),
        )
    }

    #[inline]
    pub(crate) fn despawn_with_caller(
        &mut self,
        entity: Entity,
        #[cfg(feature = "track_location")] caller: &'static Location,
    ) -> Result<(), TryDespawnError> {
        self.flush();
        if let Ok(entity) = self.get_entity_mut(entity) {
            entity.despawn_with_caller(
                #[cfg(feature = "track_location")]
                caller,
            );
            Ok(())
        } else {
            Err(TryDespawnError {
                entity,
                details: self.entities().entity_does_not_exist_error_details(entity),
            })
        }
    }

    /// Clears the internal component tracker state.
    ///
    /// The world maintains some internal state about changed and removed components. This state
    /// is used by [`RemovedComponents`] to provide access to the entities that had a specific type
    /// of component removed since last tick.
    ///
    /// The state is also used for change detection when accessing components and resources outside
    /// of a system, for example via [`World::get_mut()`] or [`World::get_resource_mut()`].
    ///
    /// By clearing this internal state, the world "forgets" about those changes, allowing a new round
    /// of detection to be recorded.
    ///
    /// When using `bevy_ecs` as part of the full Bevy engine, this method is called automatically
    /// by `bevy_app::App::update` and `bevy_app::SubApp::update`, so you don't need to call it manually.
    /// When using `bevy_ecs` as a separate standalone crate however, you do need to call this manually.
    ///
    /// ```
    /// # use bevy_ecs::prelude::*;
    /// # #[derive(Component, Default)]
    /// # struct Transform;
    /// // a whole new world
    /// let mut world = World::new();
    ///
    /// // you changed it
    /// let entity = world.spawn(Transform::default()).id();
    ///
    /// // change is detected
    /// let transform = world.get_mut::<Transform>(entity).unwrap();
    /// assert!(transform.is_changed());
    ///
    /// // update the last change tick
    /// world.clear_trackers();
    ///
    /// // change is no longer detected
    /// let transform = world.get_mut::<Transform>(entity).unwrap();
    /// assert!(!transform.is_changed());
    /// ```
    ///
    /// [`RemovedComponents`]: crate::removal_detection::RemovedComponents
    pub fn clear_trackers(&mut self) {
        self.removed_components.update();
        self.last_change_tick = self.increment_change_tick();
    }

    /// Returns [`QueryState`] for the given [`QueryData`], which is used to efficiently
    /// run queries on the [`World`] by storing and reusing the [`QueryState`].
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World};
    ///
    /// #[derive(Component, Debug, PartialEq)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// #[derive(Component)]
    /// struct Velocity {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// let entities = world.spawn_batch(vec![
    ///     (Position { x: 0.0, y: 0.0}, Velocity { x: 1.0, y: 0.0 }),
    ///     (Position { x: 0.0, y: 0.0}, Velocity { x: 0.0, y: 1.0 }),
    /// ]).collect::<Vec<Entity>>();
    ///
    /// let mut query = world.query::<(&mut Position, &Velocity)>();
    /// for (mut position, velocity) in query.iter_mut(&mut world) {
    ///    position.x += velocity.x;
    ///    position.y += velocity.y;
    /// }
    ///
    /// assert_eq!(world.get::<Position>(entities[0]).unwrap(), &Position { x: 1.0, y: 0.0 });
    /// assert_eq!(world.get::<Position>(entities[1]).unwrap(), &Position { x: 0.0, y: 1.0 });
    /// ```
    ///
    /// To iterate over entities in a deterministic order,
    /// sort the results of the query using the desired component as a key.
    /// Note that this requires fetching the whole result set from the query
    /// and allocation of a [`Vec`] to store it.
    ///
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World};
    ///
    /// #[derive(Component, PartialEq, Eq, PartialOrd, Ord, Debug)]
    /// struct Order(i32);
    /// #[derive(Component, PartialEq, Debug)]
    /// struct Label(&'static str);
    ///
    /// let mut world = World::new();
    /// let a = world.spawn((Order(2), Label("second"))).id();
    /// let b = world.spawn((Order(3), Label("third"))).id();
    /// let c = world.spawn((Order(1), Label("first"))).id();
    /// let mut entities = world.query::<(Entity, &Order, &Label)>()
    ///     .iter(&world)
    ///     .collect::<Vec<_>>();
    /// // Sort the query results by their `Order` component before comparing
    /// // to expected results. Query iteration order should not be relied on.
    /// entities.sort_by_key(|e| e.1);
    /// assert_eq!(entities, vec![
    ///     (c, &Order(1), &Label("first")),
    ///     (a, &Order(2), &Label("second")),
    ///     (b, &Order(3), &Label("third")),
    /// ]);
    /// ```
    #[inline]
    pub fn query<D: QueryData>(&mut self) -> QueryState<D, ()> {
        self.query_filtered::<D, ()>()
    }

    /// Returns [`QueryState`] for the given filtered [`QueryData`], which is used to efficiently
    /// run queries on the [`World`] by storing and reusing the [`QueryState`].
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World, query::With};
    ///
    /// #[derive(Component)]
    /// struct A;
    /// #[derive(Component)]
    /// struct B;
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(A).id();
    /// let e2 = world.spawn((A, B)).id();
    ///
    /// let mut query = world.query_filtered::<Entity, With<B>>();
    /// let matching_entities = query.iter(&world).collect::<Vec<Entity>>();
    ///
    /// assert_eq!(matching_entities, vec![e2]);
    /// ```
    #[inline]
    pub fn query_filtered<D: QueryData, F: QueryFilter>(&mut self) -> QueryState<D, F> {
        QueryState::new_in_sub_storage(self, sub_storage)
    }

    /// Returns [`QueryState`] for the given [`QueryData`], which is used to efficiently
    /// run queries on the [`World`] by storing and reusing the [`QueryState`].
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World};
    ///
    /// #[derive(Component, Debug, PartialEq)]
    /// struct Position {
    ///   x: f32,
    ///   y: f32,
    /// }
    ///
    /// let mut world = World::new();
    /// world.spawn_batch(vec![
    ///     Position { x: 0.0, y: 0.0 },
    ///     Position { x: 1.0, y: 1.0 },
    /// ]);
    ///
    /// fn get_positions(world: &World) -> Vec<(Entity, &Position)> {
    ///     let mut query = world.try_query::<(Entity, &Position)>().unwrap();
    ///     query.iter(world).collect()
    /// }
    ///
    /// let positions = get_positions(&world);
    ///
    /// assert_eq!(world.get::<Position>(positions[0].0).unwrap(), positions[0].1);
    /// assert_eq!(world.get::<Position>(positions[1].0).unwrap(), positions[1].1);
    /// ```
    ///
    /// Requires only an immutable world reference, but may fail if, for example,
    /// the components that make up this query have not been registered into the world.
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World};
    ///
    /// #[derive(Component)]
    /// struct A;
    ///
    /// let mut world = World::new();
    ///
    /// let none_query = world.try_query::<&A>();
    /// assert!(none_query.is_none());
    ///
    /// world.register_component::<A>();
    ///
    /// let some_query = world.try_query::<&A>();
    /// assert!(some_query.is_some());
    /// ```
    #[inline]
    pub fn try_query<D: QueryData>(&self) -> Option<QueryState<D, ()>> {
        self.try_query_filtered::<D, ()>()
    }

    /// Returns [`QueryState`] for the given filtered [`QueryData`], which is used to efficiently
    /// run queries on the [`World`] by storing and reusing the [`QueryState`].
    /// ```
    /// use bevy_ecs::{component::Component, entity::Entity, world::World, query::With};
    ///
    /// #[derive(Component)]
    /// struct A;
    /// #[derive(Component)]
    /// struct B;
    ///
    /// let mut world = World::new();
    /// let e1 = world.spawn(A).id();
    /// let e2 = world.spawn((A, B)).id();
    ///
    /// let mut query = world.try_query_filtered::<Entity, With<B>>().unwrap();
    /// let matching_entities = query.iter(&world).collect::<Vec<Entity>>();
    ///
    /// assert_eq!(matching_entities, vec![e2]);
    /// ```
    ///
    /// Requires only an immutable world reference, but may fail if, for example,
    /// the components that make up this query have not been registered into the world.
    #[inline]
    pub fn try_query_filtered<D: QueryData, F: QueryFilter>(&self) -> Option<QueryState<D, F>> {
        QueryState::try_new(self)
    }

    /// Returns an iterator of entities that had components of type `T` removed
    /// since the last call to [`World::clear_trackers`].
    pub fn removed<T: Component>(&self) -> impl Iterator<Item = Entity> + '_ {
        self.components
            .get_id(TypeId::of::<T>())
            .map(|component_id| self.removed_with_id(component_id))
            .into_iter()
            .flatten()
    }

    /// Returns an iterator of entities that had components with the given `component_id` removed
    /// since the last call to [`World::clear_trackers`].
    pub fn removed_with_id(&self, component_id: ComponentId) -> impl Iterator<Item = Entity> + '_ {
        self.removed_components
            .get(component_id)
            .map(|removed| removed.iter_current_update_events().cloned())
            .into_iter()
            .flatten()
            .map(Into::into)
    }
}

impl Index<SubWorldId> for SubWorlds {
    type Output = SubWorldStorage;

    #[inline]
    fn index(&self, index: SubWorldId) -> &Self::Output {
        &self.sub_storages[index.as_usize()]
    }
}

impl IndexMut<SubWorldId> for SubWorlds {
    #[inline]
    fn index_mut(&mut self, index: SubWorldId) -> &mut Self::Output {
        &mut self.sub_storages[index.as_usize()]
    }
}

impl SubWorldId {
    pub(crate) const INVALID: SubWorldId = SubWorldId(u32::MAX);

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
