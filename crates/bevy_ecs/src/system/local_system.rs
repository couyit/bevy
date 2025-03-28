use crate::world::{unsafe_world_cell::UnsafeWorldCell, World, WorldLabel};

use super::{System, SystemIn};

// This is mainly used by local observers.
// When local observers are invoked, they have exclusive world access,
// so LocalSystem doesn't maintain any metadata about access.
pub trait LocalSystem<W: WorldLabel>: System {
    fn run_local(&mut self, input: SystemIn<'_, Self>, world: &mut World<W>) -> Self::Out;

    fn initialize_local(&mut self, world: &mut World<W>);

    unsafe fn validate_param_unsafe_local(&mut self, world: UnsafeWorldCell) -> bool;

    fn validate_param(&mut self, world: &World<W>) -> bool {
        let world_cell = world.as_unsafe_world_cell_readonly();
        self.update_archetype_component_access_local(world_cell);
        unsafe { self.validate_param_unsafe_local(world_cell) }
    }

    fn update_archetype_component_access_local(&mut self, world: UnsafeWorldCell);
}
