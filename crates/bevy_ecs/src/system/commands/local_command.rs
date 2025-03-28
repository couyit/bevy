use crate::world::{World, WorldLabel};

pub trait LocalCommand<W: WorldLabel, Out = ()>: Send + 'static {
    fn apply(self, world: &mut World<W>) -> Out;
}
