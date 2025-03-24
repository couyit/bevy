use core::{any::type_name, fmt};

use crate::{
    entity::Entity,
    system::{entity_command::EntityCommandError, Command, EntityCommand},
    world::{error::EntityMutableFetchError, ComponentWorld, World},
};

use super::{default_error_handler, BevyError, ErrorContext};

/// Takes a [`Command`] that returns a Result and uses a given error handler function to convert it into
/// a [`Command`] that internally handles an error if it occurs and returns `()`.
pub trait HandleError<W: ComponentWorld, Out = ()> {
    /// Takes a [`Command`] that returns a Result and uses a given error handler function to convert it into
    /// a [`Command`] that internally handles an error if it occurs and returns `()`.
    fn handle_error_with(self, error_handler: fn(BevyError, ErrorContext)) -> impl Command<W>;
    /// Takes a [`Command`] that returns a Result and uses the default error handler function to convert it into
    /// a [`Command`] that internally handles an error if it occurs and returns `()`.
    fn handle_error(self) -> impl Command<W>
    where
        Self: Sized,
    {
        self.handle_error_with(default_error_handler())
    }
}

impl<C, T, E, W> HandleError<W, Result<T, E>> for C
where
    C: Command<W, Result<T, E>>,
    E: Into<BevyError>,
    W: ComponentWorld,
{
    fn handle_error_with(self, error_handler: fn(BevyError, ErrorContext)) -> impl Command<W> {
        move |world: &mut World<W>| match self.apply(world) {
            Ok(_) => {}
            Err(err) => (error_handler)(
                err.into(),
                ErrorContext::Command {
                    name: type_name::<C>().into(),
                },
            ),
        }
    }
}

impl<C, W> HandleError<W> for C
where
    C: Command<W>,
    W: ComponentWorld,
{
    #[inline]
    fn handle_error_with(self, _error_handler: fn(BevyError, ErrorContext)) -> impl Command<W> {
        self
    }
    #[inline]
    fn handle_error(self) -> impl Command<W>
    where
        Self: Sized,
    {
        self
    }
}

/// Passes in a specific entity to an [`EntityCommand`], resulting in a [`Command`] that
/// internally runs the [`EntityCommand`] on that entity.
///
// NOTE: This is a separate trait from `EntityCommand` because "result-returning entity commands" and
// "non-result returning entity commands" require different implementations, so they cannot be automatically
// implemented. And this isn't the type of implementation that we want to thrust on people implementing
// EntityCommand.
pub trait CommandWithEntity<W: ComponentWorld, Out> {
    /// Passes in a specific entity to an [`EntityCommand`], resulting in a [`Command`] that
    /// internally runs the [`EntityCommand`] on that entity.
    fn with_entity(self, entity: Entity) -> impl Command<W, Out> + HandleError<W, Out>;
}

impl<C, W> CommandWithEntity<W, Result<(), EntityMutableFetchError>> for C
where
    C: EntityCommand,
    W: ComponentWorld,
{
    fn with_entity(
        self,
        entity: Entity,
    ) -> impl Command<W, Result<(), EntityMutableFetchError>>
           + HandleError<W, Result<(), EntityMutableFetchError>> {
        move |world: &mut World<W>| -> Result<(), EntityMutableFetchError> {
            let entity = world.get_entity_mut(entity)?;
            self.apply(entity);
            Ok(())
        }
    }
}

impl<C, T, Err, W> CommandWithEntity<W, Result<T, EntityCommandError<Err>>> for C
where
    C: EntityCommand<Result<T, Err>>,
    Err: fmt::Debug + fmt::Display + Send + Sync + 'static,
    W: ComponentWorld,
{
    fn with_entity(
        self,
        entity: Entity,
    ) -> impl Command<W, Result<T, EntityCommandError<Err>>>
           + HandleError<W, Result<T, EntityCommandError<Err>>> {
        move |world: &mut World<W>| {
            let entity = world.get_entity_mut(entity)?;
            self.apply(entity)
                .map_err(EntityCommandError::CommandFailed)
        }
    }
}
