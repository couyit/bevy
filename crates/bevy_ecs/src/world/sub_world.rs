use crate::storage::{SparseSets, SubStorageId, Tables};

use super::unsafe_world_cell::UnsafeWorldCell;

#[derive(Clone, Copy)]
pub struct SubWorld<'w> {
    pub(crate) world: UnsafeWorldCell<'w>,
    sub_storage: SubStorageId,
}

impl<'w> SubWorld<'w> {
    pub(crate) fn tables(&self) -> &'w Tables {
        &self.world.sub_storages()[self.sub_storage].tables
    }

    pub(crate) fn sparse_sets(&self) -> &'w SparseSets {
        &self.world.sub_storages()[self.sub_storage].sparse_sets
    }
}
