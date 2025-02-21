use core::{
    alloc::Layout,
    any::{Any, TypeId},
    mem::needs_drop,
};
use std::{borrow::Cow, vec::Vec};

use bevy_ptr::OwningPtr;
use bevy_utils::TypeIdMap;

use crate::{prelude::Resource, storage::SparseSetIndex};

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct ResourceId(usize);

impl ResourceId {
    #[inline]
    pub const fn new(index: usize) -> ResourceId {
        ResourceId(index)
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0
    }
}

impl SparseSetIndex for ResourceId {
    #[inline]
    fn sparse_set_index(&self) -> usize {
        self.index()
    }

    #[inline]
    fn get_sparse_set_index(value: usize) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
pub struct ResourceInfo {
    id: ResourceId,
    descriptor: ResourceDescriptor,
}

impl ResourceInfo {
    /// Returns a value uniquely identifying the current component.
    #[inline]
    pub fn id(&self) -> ResourceId {
        self.id
    }

    /// Returns the name of the current component.
    #[inline]
    pub fn name(&self) -> &str {
        &self.descriptor.name
    }

    /// Returns the [`TypeId`] of the underlying component type.
    /// Returns `None` if the component does not correspond to a Rust type.
    #[inline]
    pub fn type_id(&self) -> Option<TypeId> {
        self.descriptor.type_id
    }

    /// Returns the layout used to store values of this component in memory.
    #[inline]
    pub fn layout(&self) -> Layout {
        self.descriptor.layout
    }

    #[inline]
    /// Get the function which should be called to clean up values of
    /// the underlying component type. This maps to the
    /// [`Drop`] implementation for 'normal' Rust components
    ///
    /// Returns `None` if values of the underlying component type don't
    /// need to be dropped, e.g. as reported by [`needs_drop`].
    pub fn drop(&self) -> Option<unsafe fn(OwningPtr<'_>)> {
        self.descriptor.drop
    }

    /// Create a new [`ComponentInfo`].
    pub(crate) fn new(id: ResourceId, descriptor: ResourceDescriptor) -> Self {
        Self { id, descriptor }
    }
}

#[derive(Default)]
pub struct ResourceComponents {
    resources: Vec<ResourceInfo>,
    indices: TypeIdMap<ResourceId>,
}

impl ResourceComponents {
    /// Returns the number of components registered with this instance.
    #[inline]
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Returns `true` if there are no components registered with this instance. Otherwise, this returns `false`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Type-erased equivalent of [`Components::resource_id()`].
    #[inline]
    pub fn get_resource_id(&self, type_id: TypeId) -> Option<ResourceId> {
        self.indices.get(&type_id).copied()
    }

    /// Returns the [`ComponentId`] of the given [`Resource`] type `T`.
    ///
    /// The returned `ComponentId` is specific to the `Components` instance
    /// it was retrieved from and should not be used with another `Components`
    /// instance.
    ///
    /// Returns [`None`] if the `Resource` type has not
    /// yet been initialized using [`Components::register_resource()`].
    ///
    /// ```
    /// use bevy_ecs::prelude::*;
    ///
    /// let mut world = World::new();
    ///
    /// #[derive(Resource, Default)]
    /// struct ResourceA;
    ///
    /// let resource_a_id = world.init_resource::<ResourceA>();
    ///
    /// assert_eq!(resource_a_id, world.components().resource_id::<ResourceA>().unwrap())
    /// ```
    ///
    /// # See also
    ///
    /// * [`Components::component_id()`]
    /// * [`Components::get_resource_id()`]
    #[inline]
    pub fn resource_id<T: Resource>(&self) -> Option<ResourceId> {
        self.get_resource_id(TypeId::of::<T>())
    }

    /// Registers a [`Resource`] of type `T` with this instance.
    /// If a resource of this type has already been registered, this will return
    /// the ID of the pre-existing resource.
    ///
    /// # See also
    ///
    /// * [`Components::resource_id()`]
    /// * [`Components::register_resource_with_descriptor()`]
    #[inline]
    pub fn register_resource<T: Resource>(&mut self) -> ResourceId {
        unsafe {
            self.get_or_register_resource_with(TypeId::of::<T>(), || {
                ResourceDescriptor::new_resource::<T>()
            })
        }
    }

    /// Registers a [`Resource`] described by `descriptor`.
    ///
    /// # Note
    ///
    /// If this method is called multiple times with identical descriptors, a distinct [`ComponentId`]
    /// will be created for each one.
    ///
    /// # See also
    ///
    /// * [`Components::resource_id()`]
    /// * [`Components::register_resource()`]
    pub fn register_resource_with_descriptor(
        &mut self,
        descriptor: ResourceDescriptor,
    ) -> ResourceId {
        Self::register_resource_inner(&mut self.resources, descriptor)
    }

    /// Registers a [non-send resource](crate::system::NonSend) of type `T` with this instance.
    /// If a resource of this type has already been registered, this will return
    /// the ID of the pre-existing resource.
    #[inline]
    pub fn register_non_send<T: Any>(&mut self) -> ResourceId {
        // SAFETY: The [`ComponentDescriptor`] matches the [`TypeId`]
        unsafe {
            self.get_or_register_resource_with(TypeId::of::<T>(), || {
                ResourceDescriptor::new_non_send::<T>()
            })
        }
    }

    /// # Safety
    ///
    /// The [`ComponentDescriptor`] must match the [`TypeId`]
    #[inline]
    unsafe fn get_or_register_resource_with(
        &mut self,
        type_id: TypeId,
        func: impl FnOnce() -> ResourceDescriptor,
    ) -> ResourceId {
        let resources = &mut self.resources;
        *self.indices.entry(type_id).or_insert_with(|| {
            let descriptor = func();
            Self::register_resource_inner(resources, descriptor)
        })
    }

    #[inline]
    fn register_resource_inner(
        resources: &mut Vec<ResourceInfo>,
        descriptor: ResourceDescriptor,
    ) -> ResourceId {
        let resource_id = ResourceId(resources.len());
        resources.push(ResourceInfo::new(resource_id, descriptor));
        resource_id
    }

    /// Gets an iterator over all components registered with this instance.
    pub fn iter(&self) -> impl Iterator<Item = &ResourceInfo> + '_ {
        self.resources.iter()
    }
}

/// A value describing a resource, which may or may not correspond to a Rust type.
#[derive(Clone)]
pub struct ResourceDescriptor {
    name: Cow<'static, str>,
    // SAFETY: This must remain private. It must only be set to "true" if this component is
    // actually Send + Sync
    is_send_and_sync: bool,
    type_id: Option<TypeId>,
    layout: Layout,
    // SAFETY: this function must be safe to call with pointers pointing to items of the type
    // this descriptor describes.
    // None if the underlying type doesn't need to be dropped
    drop: Option<for<'a> unsafe fn(OwningPtr<'a>)>,
}

impl ResourceDescriptor {
    /// # Safety
    ///
    /// `x` must point to a valid value of type `T`.
    unsafe fn drop_ptr<T>(x: OwningPtr<'_>) {
        // SAFETY: Contract is required to be upheld by the caller.
        unsafe {
            x.drop_as::<T>();
        }
    }

    /// Create a new `ComponentDescriptor` for a resource.
    ///
    /// The [`StorageType`] for resources is always [`StorageType::Table`].
    pub fn new_resource<T: Resource>() -> Self {
        Self {
            name: Cow::Borrowed(core::any::type_name::<T>()),
            // PERF: `SparseStorage` may actually be a more
            // reasonable choice as `storage_type` for resources.
            is_send_and_sync: true,
            type_id: Some(TypeId::of::<T>()),
            layout: Layout::new::<T>(),
            drop: needs_drop::<T>().then_some(Self::drop_ptr::<T> as _),
        }
    }

    fn new_non_send<T: Any>() -> Self {
        Self {
            name: Cow::Borrowed(core::any::type_name::<T>()),
            is_send_and_sync: false,
            type_id: Some(TypeId::of::<T>()),
            layout: Layout::new::<T>(),
            drop: needs_drop::<T>().then_some(Self::drop_ptr::<T> as _),
        }
    }
}
