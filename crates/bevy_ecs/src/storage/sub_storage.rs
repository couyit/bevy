use core::ops::{Index, IndexMut};
use std::vec::Vec;

use bevy_platform_support::collections::HashSet;
use bevy_utils::TypeIdMap;

use crate::{
    archetype::ArchetypeId,
    bundle::{BundleId, BundleInfo},
    component::{ComponentInfo, Components, StorageType},
};

use super::{SparseSets, Tables};

#[derive(Default)]
pub struct SubStorages {
    pub sub_storages: Vec<SubStorage>,
    pub indices: TypeIdMap<SubStorageId>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SubStorageId(pub u32);

pub struct SubStorage {
    pub(crate) id: SubStorageId,
    pub(crate) archetypes: Vec<ArchetypeId>,
    pub(crate) sparse_sets: SparseSets,
    pub(crate) tables: Tables,
    pub(crate) empty: ArchetypeId,
    pub(crate) prepared: HashSet<BundleId>,
}

pub trait Storage: Send + Sync + 'static {}

pub struct MainStorage;
pub struct InvalidStorage;

impl Storage for MainStorage {}
impl Storage for InvalidStorage {}

impl SubStorages {
    pub const MAIN_STORAGE: SubStorageId = SubStorageId(0);

    pub fn new() -> Self {
        Self {
            sub_storages: vec![SubStorageInfo {
                id: SubWorldId(0),
                archetypes: Vec::new(),
                storages: Storages::default(),
            }],
            indices: vec![(TypeId::of::<MainStorage>(), SubWorldId(0))]
                .into_iter()
                .collect(),
        }
    }
}

impl SubStorage {
    pub fn empty(&self) -> ArchetypeId {
        self.empty
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

impl SubStorageId {
    pub(crate) const INVALID: SubStorageId = SubStorageId(u32::MAX);

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}
