use std::vec::Vec;

use crate::archetype::ArchetypeId;

#[derive(Default)]
pub struct SubWorlds {
    pub sub_worlds: Vec<SubWorldInfo>,
}

pub struct SubWorldId(pub u32);

pub struct SubWorldInfo {
    pub id: SubWorldId,
    pub archetypes: Vec<ArchetypeId>,
}

pub trait SubWorld: Send + Sync + 'static {}

pub struct MainSubWorld;

impl SubWorld for MainSubWorld {}

impl SubWorlds {
    pub fn new() -> Self {
        Self { sub_worlds: vec![] }
    }
}
