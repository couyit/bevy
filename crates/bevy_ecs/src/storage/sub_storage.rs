use core::ops::{Index, IndexMut};
use std::vec::Vec;

use bevy_utils::TypeIdMap;

use crate::{archetype::ArchetypeId, storage::Storages};

#[derive(Default)]
pub struct SubStorages {
    pub sub_storages: Vec<SubStorageData>,
    pub indices: TypeIdMap<SubStorageId>,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct SubStorageId(pub u32);

pub struct SubStorageData {
    pub id: SubStorageId,
    pub archetypes: Vec<ArchetypeId>,
    pub storages: Storages,
}

pub trait SubStorage: Send + Sync + 'static {}

pub struct MainStorage;

impl SubStorage for MainStorage {}

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

impl Index<SubStorageId> for SubStorages {
    type Output = SubStorageData;

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
