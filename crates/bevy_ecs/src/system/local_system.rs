use crate::world::{unsafe_world_cell::UnsafeWorldCell, World};

use super::{System, SystemIn};

// This is mainly used by local observers.
// When local observers are invoked, they have exclusive world access,
// so LocalSystem doesn't maintain any metadata about access.
pub trait LocalSystem: System {
    fn run_local(&mut self, input: SystemIn<'_, Self>, world: &mut World) -> Self::Out;

    fn initialize_local(&mut self, world: &mut World);

    unsafe fn validate_param_unsafe_local(&mut self, world: UnsafeWorldCell) -> bool;

    fn validate_param(&mut self, world: &World) -> bool {
        let world_cell = world.as_unsafe_world_cell_readonly();
        self.update_archetype_component_access_local(world_cell);
        unsafe { self.validate_param_unsafe_local(world_cell) }
    }

    fn update_archetype_component_access_local(&mut self, world: UnsafeWorldCell);
}
