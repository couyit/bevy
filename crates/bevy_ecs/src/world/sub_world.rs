use crate::{
    entity::Entities,
    storage::{SparseSets, SubStorageId, Table},
};

use super::unsafe_world_cell::UnsafeWorldCell;

pub struct SubWorld<'w> {
    world: UnsafeWorldCell<'w>,
    sub_storage: SubStorageId,
}

impl<'w> SubWorld<'w> {
    pub(crate) fn shared_entities(&self) -> &'w Entities {
        self.world.entities()
    }

    pub(crate) fn table(&self) -> &'w Table {
        self.world.sub_storages()[self.sub_storage].storages.tables
    }
}
