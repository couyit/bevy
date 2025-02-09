//! Storage layouts for ECS data.
//!
//! This module implements the low-level collections that store data in a [`World`]. These all offer minimal and often
//! unsafe APIs, and have been made `pub` primarily for debugging and monitoring purposes.
//!
//! # Fetching Storages
//! Each of the below data stores can be fetched via [`Storages`], which can be fetched from a
//! [`World`] via [`World::storages`]. It exposes a top level container for each class of storage within
//! ECS:
//!
//!  - [`Tables`] - columnar contiguous blocks of memory, optimized for fast iteration.
//!  - [`SparseSets`] - sparse `HashMap`-like mappings from entities to components, optimized for random
//!    lookup and regular insertion/removal of components.
//!  - [`Resources`] - singleton storage for the resources in the world
//!
//! # Safety
//! To avoid trivially unsound use of the APIs in this module, it is explicitly impossible to get a mutable
//! reference to [`Storages`] from [`World`], and none of the types publicly expose a mutable interface.
//!
//! [`World`]: crate::world::World
//! [`World::storages`]: crate::world::World::storages

mod blob_array;
mod blob_vec;
mod resource;
mod sparse_set;
mod table;
mod thin_array_ptr;

use core::ops::{Index, IndexMut};
use std::vec::Vec;

use bevy_platform_support::collections::HashSet;
pub use resource::*;
pub use sparse_set::*;
pub use table::*;

use crate::{
    archetype::ArchetypeId,
    bundle::{BundleId, BundleInfo},
    component::{ComponentInfo, Components, StorageType},
};

/// The raw data stores of a [`World`](crate::world::World)
#[derive(Default)]
pub struct Storages {
    /// Backing storage for [`SparseSet`] components.
    /// Note that sparse sets are only present for components that have been spawned or have had a relevant bundle registered.
    // pub sparse_sets: SparseSets,
    /// Backing storage for [`Table`] components.
    // pub tables: Tables,
    pub sub_storages: SubStorages,
    /// Backing storage for resources.
    pub resources: Resources<true>,
    /// Backing storage for `!Send` resources.
    pub non_send_resources: Resources<false>,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubStorageId(pub u32);

#[derive(Default)]
pub struct SubStorages {
    pub sub_storages: Vec<SubStorage>,
}

pub struct SubStorage {
    pub empty: ArchetypeId,
    pub prepared: HashSet<BundleId>,
    pub sparse_sets: SparseSets,
    pub tables: Tables,
}

impl SubStorageId {
    pub(crate) const INVALID: SubStorageId = SubStorageId(u32::MAX);
    #[inline]
    pub const fn from_u32(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn from_usize(index: usize) -> Self {
        debug_assert!(index as u32 as usize == index);
        Self(index as u32)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn as_usize(self) -> usize {
        // usize is at least u32 in Bevy
        self.0 as usize
    }

    #[inline]
    pub const fn empty() -> Self {
        Self(0)
    }
}

impl SubStorages {
    #[inline]
    pub(crate) fn get_2_mut(
        &mut self,
        a: SubStorageId,
        b: SubStorageId,
    ) -> (&mut SubStorage, &mut SubStorage) {
        if a.as_usize() > b.as_usize() {
            let (b_slice, a_slice) = self.sub_storages.split_at_mut(a.as_usize());
            (&mut a_slice[0], &mut b_slice[b.as_usize()])
        } else {
            let (a_slice, b_slice) = self.sub_storages.split_at_mut(b.as_usize());
            (&mut a_slice[a.as_usize()], &mut b_slice[0])
        }
    }
}

impl Index<SubStorageId> for SubStorages {
    type Output = SubStorage;

    #[inline]
    fn index(&self, index: SubStorageId) -> &Self::Output {
        &self.sub_storages[index.as_usize()]
    }
}

impl IndexMut<SubStorageId> for SubStorages {
    #[inline]
    fn index_mut(&mut self, index: SubStorageId) -> &mut Self::Output {
        &mut self.sub_storages[index.as_usize()]
    }
}

impl SubStorage {
    pub fn empty(&self) -> &ArchetypeId {
        &self.empty
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
}
