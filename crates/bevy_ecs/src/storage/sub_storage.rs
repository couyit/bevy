use core::{
    any::TypeId,
    ops::{Index, IndexMut},
};
use std::vec::Vec;

use bevy_platform_support::collections::HashSet;
use bevy_utils::TypeIdMap;

use crate::{
    archetype::ArchetypeId,
    bundle::{BundleId, BundleInfo},
    component::{ComponentInfo, Components, StorageType},
    observer::Observers,
    world::World,
};

use super::{SparseSets, TableId, Tables};

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

    pub(crate) fn new() -> Self {
        let mut sub_storages = Self {
            sub_storages: Vec::with_capacity(1),
            indices: TypeIdMap::default(),
        };

        sub_storages.sub_storages.push(SubStorage {
            id: SubStorageId(0),
            archetypes: Vec::new(),
            sparse_sets: Default::default(),
            tables: Default::default(),
            empty: ArchetypeId::MAIN_EMPTY,
            prepared: Default::default(),
        });

        sub_storages
            .indices
            .insert(TypeId::of::<MainStorage>(), SubStorageId(0));

        sub_storages
    }

    pub fn create_sub_storage<'w, T: Storage>(&mut self, world: &'w mut World) -> SubStorageId {
        let sub_storage = SubStorageId(self.sub_storages.len() as u32);

        let empty = unsafe {
            world.archetypes.get_id_or_insert(
                &Components::default(),
                &Observers::default(),
                TableId::empty(),
                sub_storage,
                Vec::new(),
                Vec::new(),
            )
        };

        self.sub_storages.push(SubStorage {
            id: sub_storage,
            archetypes: Vec::new(),
            sparse_sets: Default::default(),
            tables: Default::default(),
            empty,
            prepared: Default::default(),
        });

        self.indices.insert(TypeId::of::<T>(), sub_storage);

        sub_storage
    }

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
