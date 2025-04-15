use crate::component::ComponentId;
use crate::storage::SparseSetIndex;
use crate::world::World;
use alloc::{format, string::String, vec, vec::Vec};
use core::{fmt, fmt::Debug, marker::PhantomData};
use derive_more::From;
use disqualified::ShortName;
use fixedbitset::FixedBitSet;
use thiserror::Error;

/// A wrapper struct to make Debug representations of [`FixedBitSet`] easier
/// to read, when used to store [`SparseSetIndex`].
///
/// Instead of the raw integer representation of the `FixedBitSet`, the list of
/// `T` valid for [`SparseSetIndex`] is shown.
///
/// Normal `FixedBitSet` `Debug` output:
/// ```text
/// read_and_writes: FixedBitSet { data: [ 160 ], length: 8 }
/// ```
///
/// Which, unless you are a computer, doesn't help much understand what's in
/// the set. With `FormattedBitSet`, we convert the present set entries into
/// what they stand for, it is much clearer what is going on:
/// ```text
/// read_and_writes: [ ComponentId(5), ComponentId(7) ]
/// ```
struct FormattedBitSet<'a, T: SparseSetIndex> {
    bit_set: &'a FixedBitSet,
    _marker: PhantomData<T>,
}

impl<'a, T: SparseSetIndex> FormattedBitSet<'a, T> {
    fn new(bit_set: &'a FixedBitSet) -> Self {
        Self {
            bit_set,
            _marker: PhantomData,
        }
    }
}

impl<'a, T: SparseSetIndex + Debug> Debug for FormattedBitSet<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.bit_set.ones().map(T::get_sparse_set_index))
            .finish()
    }
}

/// Tracks read and write access to specific elements in a collection.
///
/// Used internally to ensure soundness during system initialization and execution.
/// See the [`is_compatible`](Access::is_compatible) and [`get_conflicts`](Access::get_conflicts) functions.
#[derive(Eq, PartialEq)]
pub struct Access<T: SparseSetIndex> {
    /// All accessed components, or forbidden components if
    /// `Self::component_read_and_writes_inverted` is set.
    read_and_writes: FixedBitSet,
    /// All exclusively-accessed components, or components that may not be
    /// exclusively accessed if `Self::component_writes_inverted` is set.
    writes: FixedBitSet,
    /// Is `true` if this component can read all components *except* those
    /// present in `Self::component_read_and_writes`.
    read_and_writes_inverted: bool,
    /// Is `true` if this component can write to all components *except* those
    /// present in `Self::component_writes`.
    writes_inverted: bool,
    // Components that are not accessed, but whose presence in an archetype affect query results.
    archetypal: FixedBitSet,
    marker: PhantomData<T>,
}

// This is needed since `#[derive(Clone)]` does not generate optimized `clone_from`.
impl<T: SparseSetIndex> Clone for Access<T> {
    fn clone(&self) -> Self {
        Self {
            read_and_writes: self.read_and_writes.clone(),
            writes: self.writes.clone(),
            read_and_writes_inverted: self.read_and_writes_inverted,
            writes_inverted: self.writes_inverted,
            archetypal: self.archetypal.clone(),
            marker: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.read_and_writes.clone_from(&source.read_and_writes);
        self.writes.clone_from(&source.writes);
        self.read_and_writes_inverted = source.read_and_writes_inverted;
        self.writes_inverted = source.writes_inverted;
        self.archetypal.clone_from(&source.archetypal);
    }
}

impl<T: SparseSetIndex + Debug> Debug for Access<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Access")
            .field(
                "read_and_writes",
                &FormattedBitSet::<T>::new(&self.read_and_writes),
            )
            .field("writes", &FormattedBitSet::<T>::new(&self.writes))
            .field("read_and_writes_inverted", &self.read_and_writes_inverted)
            .field("writes_inverted", &self.writes_inverted)
            .field("archetypal", &FormattedBitSet::<T>::new(&self.archetypal))
            .finish()
    }
}

impl<T: SparseSetIndex> Default for Access<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SparseSetIndex> Access<T> {
    /// Creates an empty [`Access`] collection.
    pub const fn new() -> Self {
        Self {
            read_and_writes_inverted: false,
            writes_inverted: false,
            read_and_writes: FixedBitSet::new(),
            writes: FixedBitSet::new(),
            archetypal: FixedBitSet::new(),
            marker: PhantomData,
        }
    }

    fn add_sparse_set_index_read(&mut self, index: usize) {
        if !self.read_and_writes_inverted {
            self.read_and_writes.grow_and_insert(index);
        } else if index < self.read_and_writes.len() {
            self.read_and_writes.remove(index);
        }
    }

    fn add_sparse_set_index_write(&mut self, index: usize) {
        if !self.writes_inverted {
            self.writes.grow_and_insert(index);
        } else if index < self.writes.len() {
            self.writes.remove(index);
        }
    }

    /// Adds access to the component given by `index`.
    pub fn add_read(&mut self, index: T) {
        let sparse_set_index = index.sparse_set_index();
        self.add_sparse_set_index_read(sparse_set_index);
    }

    /// Adds exclusive access to the component given by `index`.
    pub fn add_write(&mut self, index: T) {
        let sparse_set_index = index.sparse_set_index();
        self.add_sparse_set_index_read(sparse_set_index);
        self.add_sparse_set_index_write(sparse_set_index);
    }

    fn remove_sparse_set_index_read(&mut self, index: usize) {
        if self.read_and_writes_inverted {
            self.read_and_writes.grow_and_insert(index);
        } else if index < self.read_and_writes.len() {
            self.read_and_writes.remove(index);
        }
    }

    fn remove_sparse_set_index_write(&mut self, index: usize) {
        if self.writes_inverted {
            self.writes.grow_and_insert(index);
        } else if index < self.writes.len() {
            self.writes.remove(index);
        }
    }

    /// Removes both read and write access to the component given by `index`.
    ///
    /// Because this method corresponds to the set difference operator ∖, it can
    /// create complicated logical formulas that you should verify correctness
    /// of. For example, A ∪ (B ∖ A) isn't equivalent to (A ∪ B) ∖ A, so you
    /// can't replace a call to `remove_component_read` followed by a call to
    /// `extend` with a call to `extend` followed by a call to
    /// `remove_component_read`.
    pub fn remove_read(&mut self, index: T) {
        let sparse_set_index = index.sparse_set_index();
        self.remove_sparse_set_index_write(sparse_set_index);
        self.remove_sparse_set_index_read(sparse_set_index);
    }

    /// Removes write access to the component given by `index`.
    ///
    /// Because this method corresponds to the set difference operator ∖, it can
    /// create complicated logical formulas that you should verify correctness
    /// of. For example, A ∪ (B ∖ A) isn't equivalent to (A ∪ B) ∖ A, so you
    /// can't replace a call to `remove_component_write` followed by a call to
    /// `extend` with a call to `extend` followed by a call to
    /// `remove_component_write`.
    pub fn remove_write(&mut self, index: T) {
        let sparse_set_index = index.sparse_set_index();
        self.remove_sparse_set_index_write(sparse_set_index);
    }

    /// Adds an archetypal (indirect) access to the component given by `index`.
    ///
    /// This is for components whose values are not accessed (and thus will never cause conflicts),
    /// but whose presence in an archetype may affect query results.
    ///
    /// Currently, this is only used for [`Has<T>`].
    ///
    /// [`Has<T>`]: crate::query::Has
    pub fn add_archetypal(&mut self, index: T) {
        self.archetypal.grow_and_insert(index.sparse_set_index());
    }

    /// Returns `true` if this can access the component given by `index`.
    pub fn has_read(&self, index: T) -> bool {
        self.read_and_writes_inverted ^ self.read_and_writes.contains(index.sparse_set_index())
    }

    /// Returns `true` if this can access any component.
    pub fn has_any_read(&self) -> bool {
        self.read_and_writes_inverted || !self.read_and_writes.is_clear()
    }

    /// Returns `true` if this can exclusively access the component given by `index`.
    pub fn has_write(&self, index: T) -> bool {
        self.writes_inverted ^ self.writes.contains(index.sparse_set_index())
    }

    /// Returns `true` if this accesses any component mutably.
    pub fn has_any_write(&self) -> bool {
        self.writes_inverted || !self.writes.is_clear()
    }

    /// Returns true if this has an archetypal (indirect) access to the component given by `index`.
    ///
    /// This is a component whose value is not accessed (and thus will never cause conflicts),
    /// but whose presence in an archetype may affect query results.
    ///
    /// Currently, this is only used for [`Has<T>`].
    ///
    /// [`Has<T>`]: crate::query::Has
    pub fn has_archetypal(&self, index: T) -> bool {
        self.archetypal.contains(index.sparse_set_index())
    }

    /// Sets this as having access to all components (i.e. `EntityRef`).
    #[inline]
    pub fn read_all(&mut self) {
        self.read_and_writes_inverted = true;
        self.read_and_writes.clear();
    }

    /// Sets this as having mutable access to all components (i.e. `EntityMut`).
    #[inline]
    pub fn write_all(&mut self) {
        self.read_all();
        self.writes_inverted = true;
        self.writes.clear();
    }

    /// Returns `true` if this has access to all components (i.e. `EntityRef`).
    #[inline]
    pub fn has_read_all(&self) -> bool {
        self.read_and_writes_inverted && self.read_and_writes.is_clear()
    }

    /// Returns `true` if this has write access to all components (i.e. `EntityMut`).
    #[inline]
    pub fn has_write_all(&self) -> bool {
        self.writes_inverted && self.writes.is_clear()
    }

    /// Removes all writes.
    pub fn clear_writes(&mut self) {
        self.writes_inverted = false;
        self.writes.clear();
    }

    /// Removes all accesses.
    pub fn clear(&mut self) {
        self.read_and_writes_inverted = false;
        self.writes_inverted = false;
        self.read_and_writes.clear();
        self.writes.clear();
    }

    /// Adds all access from `other`.
    pub fn extend(&mut self, other: &Access<T>) {
        let component_read_and_writes_inverted =
            self.read_and_writes_inverted || other.read_and_writes_inverted;
        let component_writes_inverted = self.writes_inverted || other.writes_inverted;

        match (
            self.read_and_writes_inverted,
            other.read_and_writes_inverted,
        ) {
            (true, true) => {
                self.read_and_writes.intersect_with(&other.read_and_writes);
            }
            (true, false) => {
                self.read_and_writes.difference_with(&other.read_and_writes);
            }
            (false, true) => {
                // We have to grow here because the new bits are going to get flipped to 1.
                self.read_and_writes
                    .grow(self.read_and_writes.len().max(other.read_and_writes.len()));
                self.read_and_writes.toggle_range(..);
                self.read_and_writes.intersect_with(&other.read_and_writes);
            }
            (false, false) => {
                self.read_and_writes.union_with(&other.read_and_writes);
            }
        }

        match (self.writes_inverted, other.writes_inverted) {
            (true, true) => {
                self.writes.intersect_with(&other.writes);
            }
            (true, false) => {
                self.writes.difference_with(&other.writes);
            }
            (false, true) => {
                // We have to grow here because the new bits are going to get flipped to 1.
                self.writes.grow(self.writes.len().max(other.writes.len()));
                self.writes.toggle_range(..);
                self.writes.intersect_with(&other.writes);
            }
            (false, false) => {
                self.writes.union_with(&other.writes);
            }
        }

        self.read_and_writes_inverted = component_read_and_writes_inverted;
        self.writes_inverted = component_writes_inverted;
    }

    /// Returns `true` if the access and `other` can be active at the same time,
    /// only looking at their component access.
    ///
    /// [`Access`] instances are incompatible if one can write
    /// an element that the other can read or write.
    pub fn is_compatible(&self, other: &Access<T>) -> bool {
        // We have a conflict if we write and they read or write, or if they
        // write and we read or write.
        for (
            lhs_writes,
            rhs_reads_and_writes,
            lhs_writes_inverted,
            rhs_reads_and_writes_inverted,
        ) in [
            (
                &self.writes,
                &other.read_and_writes,
                self.writes_inverted,
                other.read_and_writes_inverted,
            ),
            (
                &other.writes,
                &self.read_and_writes,
                other.writes_inverted,
                self.read_and_writes_inverted,
            ),
        ] {
            match (lhs_writes_inverted, rhs_reads_and_writes_inverted) {
                (true, true) => return false,
                (false, true) => {
                    if !lhs_writes.is_subset(rhs_reads_and_writes) {
                        return false;
                    }
                }
                (true, false) => {
                    if !rhs_reads_and_writes.is_subset(lhs_writes) {
                        return false;
                    }
                }
                (false, false) => {
                    if !lhs_writes.is_disjoint(rhs_reads_and_writes) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Returns `true` if the set's component access is a subset of another, i.e. `other`'s component access
    /// contains at least all the values in `self`.
    pub fn is_subset(&self, other: &Access<T>) -> bool {
        for (
            our_components,
            their_components,
            our_components_inverted,
            their_components_inverted,
        ) in [
            (
                &self.read_and_writes,
                &other.read_and_writes,
                self.read_and_writes_inverted,
                other.read_and_writes_inverted,
            ),
            (
                &self.writes,
                &other.writes,
                self.writes_inverted,
                other.writes_inverted,
            ),
        ] {
            match (our_components_inverted, their_components_inverted) {
                (true, true) => {
                    if !their_components.is_subset(our_components) {
                        return false;
                    }
                }
                (true, false) => {
                    return false;
                }
                (false, true) => {
                    if !our_components.is_disjoint(their_components) {
                        return false;
                    }
                }
                (false, false) => {
                    if !our_components.is_subset(their_components) {
                        return false;
                    }
                }
            }
        }

        true
    }

    fn get_conflicts(&self, other: &Access<T>) -> AccessConflicts {
        let mut conflicts = FixedBitSet::new();

        // We have a conflict if we write and they read or write, or if they
        // write and we read or write.
        for (
            lhs_writes,
            rhs_reads_and_writes,
            lhs_writes_inverted,
            rhs_reads_and_writes_inverted,
        ) in [
            (
                &self.writes,
                &other.read_and_writes,
                self.writes_inverted,
                other.read_and_writes_inverted,
            ),
            (
                &other.writes,
                &self.read_and_writes,
                other.writes_inverted,
                self.read_and_writes_inverted,
            ),
        ] {
            // There's no way that I can see to do this without a temporary.
            // Neither CNF nor DNF allows us to avoid one.
            let temp_conflicts: FixedBitSet =
                match (lhs_writes_inverted, rhs_reads_and_writes_inverted) {
                    (true, true) => return AccessConflicts::All,
                    (false, true) => lhs_writes.difference(rhs_reads_and_writes).collect(),
                    (true, false) => rhs_reads_and_writes.difference(lhs_writes).collect(),
                    (false, false) => lhs_writes.intersection(rhs_reads_and_writes).collect(),
                };
            conflicts.union_with(&temp_conflicts);
        }

        AccessConflicts::Individual(conflicts)
    }

    /// Returns the indices of the resources this has access to.
    pub fn resource_reads_and_writes(&self) -> impl Iterator<Item = T> + '_ {
        self.read_and_writes.ones().map(T::get_sparse_set_index)
    }

    /// Returns the indices of the resources this has non-exclusive access to.
    pub fn resource_reads(&self) -> impl Iterator<Item = T> + '_ {
        self.read_and_writes
            .difference(&self.writes)
            .map(T::get_sparse_set_index)
    }

    /// Returns the indices of the resources this has exclusive access to.
    pub fn resource_writes(&self) -> impl Iterator<Item = T> + '_ {
        self.writes.ones().map(T::get_sparse_set_index)
    }

    /// Returns the indices of the components that this has an archetypal access to.
    ///
    /// These are components whose values are not accessed (and thus will never cause conflicts),
    /// but whose presence in an archetype may affect query results.
    ///
    /// Currently, this is only used for [`Has<T>`].
    ///
    /// [`Has<T>`]: crate::query::Has
    pub fn archetypal(&self) -> impl Iterator<Item = T> + '_ {
        self.archetypal.ones().map(T::get_sparse_set_index)
    }

    /// Returns an iterator over the component IDs and their [`ComponentAccessKind`].
    ///
    /// Returns `Err(UnboundedAccess)` if the access is unbounded.
    /// This typically occurs when an [`Access`] is marked as accessing all
    /// components, and then adding exceptions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use bevy_ecs::query::{Access, ComponentAccessKind};
    /// let mut access = Access::<usize>::default();
    ///
    /// access.add_component_read(1);
    /// access.add_component_write(2);
    /// access.add_archetypal(3);
    ///
    /// let result = access
    ///     .try_iter_component_access()
    ///     .map(Iterator::collect::<Vec<_>>);
    ///
    /// assert_eq!(
    ///     result,
    ///     Ok(vec![
    ///         ComponentAccessKind::Shared(1),
    ///         ComponentAccessKind::Exclusive(2),
    ///         ComponentAccessKind::Archetypal(3),
    ///     ]),
    /// );
    /// ```
    pub fn try_iter_component_access(
        &self,
    ) -> Result<impl Iterator<Item = ComponentAccessKind<T>> + '_, UnboundedAccessError> {
        // component_writes_inverted is only ever true when component_read_and_writes_inverted is
        // also true. Therefore it is sufficient to check just component_read_and_writes_inverted.
        if self.read_and_writes_inverted {
            return Err(UnboundedAccessError {
                writes_inverted: self.writes_inverted,
                read_and_writes_inverted: self.read_and_writes_inverted,
            });
        }

        let reads_and_writes = self.read_and_writes.ones().map(|index| {
            let sparse_index = T::get_sparse_set_index(index);

            if self.writes.contains(index) {
                ComponentAccessKind::Exclusive(sparse_index)
            } else {
                ComponentAccessKind::Shared(sparse_index)
            }
        });

        let archetypal = self
            .archetypal
            .ones()
            .filter(|&index| !self.writes.contains(index) && !self.read_and_writes.contains(index))
            .map(|index| ComponentAccessKind::Archetypal(T::get_sparse_set_index(index)));

        Ok(reads_and_writes.chain(archetypal))
    }
}

/// Error returned when attempting to iterate over items included in an [`Access`]
/// if the access excludes items rather than including them.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Error)]
#[error("Access is unbounded")]
pub struct UnboundedAccessError {
    /// [`Access`] is defined in terms of _excluding_ [exclusive](ComponentAccessKind::Exclusive)
    /// access.
    pub writes_inverted: bool,
    /// [`Access`] is defined in terms of _excluding_ [shared](ComponentAccessKind::Shared) and
    /// [exclusive](ComponentAccessKind::Exclusive) access.
    pub read_and_writes_inverted: bool,
}

/// Describes the level of access for a particular component as defined in an [`Access`].
#[derive(PartialEq, Eq, Hash, Debug, Clone, Copy)]
pub enum ComponentAccessKind<T> {
    /// Archetypical access, such as `Has<Foo>`.
    Archetypal(T),
    /// Shared access, such as `&Foo`.
    Shared(T),
    /// Exclusive access, such as `&mut Foo`.
    Exclusive(T),
}

impl<T> ComponentAccessKind<T> {
    /// Gets the index of this `ComponentAccessKind`.
    pub fn index(&self) -> &T {
        let (Self::Archetypal(value) | Self::Shared(value) | Self::Exclusive(value)) = self;
        value
    }
}

/// An [`Access`] that has been filtered to include and exclude certain combinations of elements.
///
/// Used internally to statically check if queries are disjoint.
///
/// Subtle: a `read` or `write` in `access` should not be considered to imply a
/// `with` access.
///
/// For example consider `Query<Option<&T>>` this only has a `read` of `T` as doing
/// otherwise would allow for queries to be considered disjoint when they shouldn't:
/// - `Query<(&mut T, Option<&U>)>` read/write `T`, read `U`, with `U`
/// - `Query<&mut T, Without<U>>` read/write `T`, without `U`
///   from this we could reasonably conclude that the queries are disjoint but they aren't.
///
/// In order to solve this the actual access that `Query<(&mut T, Option<&U>)>` has
/// is read/write `T`, read `U`. It must still have a read `U` access otherwise the following
/// queries would be incorrectly considered disjoint:
/// - `Query<&mut T>`  read/write `T`
/// - `Query<Option<&T>>` accesses nothing
///
/// See comments the [`WorldQuery`](super::WorldQuery) impls of [`AnyOf`](super::AnyOf)/`Option`/[`Or`](super::Or) for more information.
#[derive(Debug, Eq, PartialEq)]
pub struct FilteredAccess<T: SparseSetIndex> {
    pub(crate) access: Access<T>,
    pub(crate) required: FixedBitSet,
    // An array of filter sets to express `With` or `Without` clauses in disjunctive normal form, for example: `Or<(With<A>, With<B>)>`.
    // Filters like `(With<A>, Or<(With<B>, Without<C>)>` are expanded into `Or<((With<A>, With<B>), (With<A>, Without<C>))>`.
    pub(crate) filter_sets: Vec<AccessFilters<T>>,
}

// This is needed since `#[derive(Clone)]` does not generate optimized `clone_from`.
impl<T: SparseSetIndex> Clone for FilteredAccess<T> {
    fn clone(&self) -> Self {
        Self {
            access: self.access.clone(),
            required: self.required.clone(),
            filter_sets: self.filter_sets.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.access.clone_from(&source.access);
        self.required.clone_from(&source.required);
        self.filter_sets.clone_from(&source.filter_sets);
    }
}

impl<T: SparseSetIndex> Default for FilteredAccess<T> {
    fn default() -> Self {
        Self::matches_everything()
    }
}

impl<T: SparseSetIndex> From<FilteredAccess<T>> for FilteredAccessSet<T> {
    fn from(filtered_access: FilteredAccess<T>) -> Self {
        let mut base = FilteredAccessSet::<T>::default();
        base.add(filtered_access);
        base
    }
}

/// Records how two accesses conflict with each other
#[derive(Debug, PartialEq, From)]
pub enum AccessConflicts {
    /// Conflict is for all indices
    All,
    /// There is a conflict for a subset of indices
    Individual(FixedBitSet),
}

impl AccessConflicts {
    fn add(&mut self, other: &Self) {
        match (self, other) {
            (s, AccessConflicts::All) => {
                *s = AccessConflicts::All;
            }
            (AccessConflicts::Individual(this), AccessConflicts::Individual(other)) => {
                this.extend(other.ones());
            }
            _ => {}
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::All => false,
            Self::Individual(set) => set.is_empty(),
        }
    }

    pub(crate) fn format_conflict_list(&self, world: &World) -> String {
        match self {
            AccessConflicts::All => String::new(),
            AccessConflicts::Individual(indices) => indices
                .ones()
                .map(|index| {
                    format!(
                        "{}",
                        ShortName(
                            &world
                                .components
                                .get_name(ComponentId::get_sparse_set_index(index))
                                .unwrap()
                        )
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    /// An [`AccessConflicts`] which represents the absence of any conflict
    pub(crate) fn empty() -> Self {
        Self::Individual(FixedBitSet::new())
    }
}

impl<T: SparseSetIndex> From<Vec<T>> for AccessConflicts {
    fn from(value: Vec<T>) -> Self {
        Self::Individual(value.iter().map(T::sparse_set_index).collect())
    }
}

impl<T: SparseSetIndex> FilteredAccess<T> {
    /// Returns a `FilteredAccess` which has no access and matches everything.
    /// This is the equivalent of a `TRUE` logic atom.
    pub fn matches_everything() -> Self {
        Self {
            access: Access::default(),
            required: FixedBitSet::default(),
            filter_sets: vec![AccessFilters::default()],
        }
    }

    /// Returns a `FilteredAccess` which has no access and matches nothing.
    /// This is the equivalent of a `FALSE` logic atom.
    pub fn matches_nothing() -> Self {
        Self {
            access: Access::default(),
            required: FixedBitSet::default(),
            filter_sets: Vec::new(),
        }
    }

    /// Returns a reference to the underlying unfiltered access.
    #[inline]
    pub fn access(&self) -> &Access<T> {
        &self.access
    }

    /// Returns a mutable reference to the underlying unfiltered access.
    #[inline]
    pub fn access_mut(&mut self) -> &mut Access<T> {
        &mut self.access
    }

    /// Adds access to the component given by `index`.
    pub fn add_component_read(&mut self, index: T) {
        self.access.add_read(index.clone());
        self.add_required(index.clone());
        self.and_with(index);
    }

    /// Adds exclusive access to the component given by `index`.
    pub fn add_component_write(&mut self, index: T) {
        self.access.add_write(index.clone());
        self.add_required(index.clone());
        self.and_with(index);
    }

    /// Adds access to the resource given by `index`.
    pub fn add_resource_read(&mut self, index: T) {
        self.access.add_resource_read(index.clone());
    }

    /// Adds exclusive access to the resource given by `index`.
    pub fn add_resource_write(&mut self, index: T) {
        self.access.add_resource_write(index.clone());
    }

    fn add_required(&mut self, index: T) {
        self.required.grow_and_insert(index.sparse_set_index());
    }

    /// Adds a `With` filter: corresponds to a conjunction (AND) operation.
    ///
    /// Suppose we begin with `Or<(With<A>, With<B>)>`, which is represented by an array of two `AccessFilter` instances.
    /// Adding `AND With<C>` via this method transforms it into the equivalent of  `Or<((With<A>, With<C>), (With<B>, With<C>))>`.
    pub fn and_with(&mut self, index: T) {
        for filter in &mut self.filter_sets {
            filter.with.grow_and_insert(index.sparse_set_index());
        }
    }

    /// Adds a `Without` filter: corresponds to a conjunction (AND) operation.
    ///
    /// Suppose we begin with `Or<(With<A>, With<B>)>`, which is represented by an array of two `AccessFilter` instances.
    /// Adding `AND Without<C>` via this method transforms it into the equivalent of  `Or<((With<A>, Without<C>), (With<B>, Without<C>))>`.
    pub fn and_without(&mut self, index: T) {
        for filter in &mut self.filter_sets {
            filter.without.grow_and_insert(index.sparse_set_index());
        }
    }

    /// Appends an array of filters: corresponds to a disjunction (OR) operation.
    ///
    /// As the underlying array of filters represents a disjunction,
    /// where each element (`AccessFilters`) represents a conjunction,
    /// we can simply append to the array.
    pub fn append_or(&mut self, other: &FilteredAccess<T>) {
        self.filter_sets.append(&mut other.filter_sets.clone());
    }

    /// Adds all of the accesses from `other` to `self`.
    pub fn extend_access(&mut self, other: &FilteredAccess<T>) {
        self.access.extend(&other.access);
    }

    /// Returns `true` if this and `other` can be active at the same time.
    pub fn is_compatible(&self, other: &FilteredAccess<T>) -> bool {
        if self.access.is_compatible(&other.access) {
            return true;
        }

        // If the access instances are incompatible, we want to check that whether filters can
        // guarantee that queries are disjoint.
        // Since the `filter_sets` array represents a Disjunctive Normal Form formula ("ORs of ANDs"),
        // we need to make sure that each filter set (ANDs) rule out every filter set from the `other` instance.
        //
        // For example, `Query<&mut C, Or<(With<A>, Without<B>)>>` is compatible `Query<&mut C, (With<B>, Without<A>)>`,
        // but `Query<&mut C, Or<(Without<A>, Without<B>)>>` isn't compatible with `Query<&mut C, Or<(With<A>, With<B>)>>`.
        self.filter_sets.iter().all(|filter| {
            other
                .filter_sets
                .iter()
                .all(|other_filter| filter.is_ruled_out_by(other_filter))
        })
    }

    /// Returns a vector of elements that this and `other` cannot access at the same time.
    pub fn get_conflicts(&self, other: &FilteredAccess<T>) -> AccessConflicts {
        if !self.is_compatible(other) {
            // filters are disjoint, so we can just look at the unfiltered intersection
            return self.access.get_conflicts(&other.access);
        }
        AccessConflicts::empty()
    }

    /// Adds all access and filters from `other`.
    ///
    /// Corresponds to a conjunction operation (AND) for filters.
    ///
    /// Extending `Or<(With<A>, Without<B>)>` with `Or<(With<C>, Without<D>)>` will result in
    /// `Or<((With<A>, With<C>), (With<A>, Without<D>), (Without<B>, With<C>), (Without<B>, Without<D>))>`.
    pub fn extend(&mut self, other: &FilteredAccess<T>) {
        self.access.extend(&other.access);
        self.required.union_with(&other.required);

        // We can avoid allocating a new array of bitsets if `other` contains just a single set of filters:
        // in this case we can short-circuit by performing an in-place union for each bitset.
        if other.filter_sets.len() == 1 {
            for filter in &mut self.filter_sets {
                filter.with.union_with(&other.filter_sets[0].with);
                filter.without.union_with(&other.filter_sets[0].without);
            }
            return;
        }

        let mut new_filters = Vec::with_capacity(self.filter_sets.len() * other.filter_sets.len());
        for filter in &self.filter_sets {
            for other_filter in &other.filter_sets {
                let mut new_filter = filter.clone();
                new_filter.with.union_with(&other_filter.with);
                new_filter.without.union_with(&other_filter.without);
                new_filters.push(new_filter);
            }
        }
        self.filter_sets = new_filters;
    }

    /// Sets the underlying unfiltered access as having access to all components.
    pub fn read_all(&mut self) {
        self.access.read_all();
    }

    /// Sets the underlying unfiltered access as having mutable access to all components.
    pub fn write_all(&mut self) {
        self.access.write_all();
    }

    /// Returns `true` if the set is a subset of another, i.e. `other` contains
    /// at least all the values in `self`.
    pub fn is_subset(&self, other: &FilteredAccess<T>) -> bool {
        self.required.is_subset(&other.required) && self.access().is_subset(other.access())
    }

    /// Returns the indices of the elements that this access filters for.
    pub fn with_filters(&self) -> impl Iterator<Item = T> + '_ {
        self.filter_sets
            .iter()
            .flat_map(|f| f.with.ones().map(T::get_sparse_set_index))
    }

    /// Returns the indices of the elements that this access filters out.
    pub fn without_filters(&self) -> impl Iterator<Item = T> + '_ {
        self.filter_sets
            .iter()
            .flat_map(|f| f.without.ones().map(T::get_sparse_set_index))
    }

    /// Returns true if the index is used by this `FilteredAccess` in any way
    pub fn contains(&self, index: T) -> bool {
        self.access().has_read(index.clone())
            || self.access().has_archetypal(index.clone())
            || self.filter_sets.iter().any(|f| {
                f.with.contains(index.sparse_set_index())
                    || f.without.contains(index.sparse_set_index())
            })
    }
}

#[derive(Eq, PartialEq)]
pub(crate) struct AccessFilters<T> {
    pub(crate) with: FixedBitSet,
    pub(crate) without: FixedBitSet,
    _index_type: PhantomData<T>,
}

// This is needed since `#[derive(Clone)]` does not generate optimized `clone_from`.
impl<T: SparseSetIndex> Clone for AccessFilters<T> {
    fn clone(&self) -> Self {
        Self {
            with: self.with.clone(),
            without: self.without.clone(),
            _index_type: PhantomData,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.with.clone_from(&source.with);
        self.without.clone_from(&source.without);
    }
}

impl<T: SparseSetIndex + Debug> Debug for AccessFilters<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AccessFilters")
            .field("with", &FormattedBitSet::<T>::new(&self.with))
            .field("without", &FormattedBitSet::<T>::new(&self.without))
            .finish()
    }
}

impl<T: SparseSetIndex> Default for AccessFilters<T> {
    fn default() -> Self {
        Self {
            with: FixedBitSet::default(),
            without: FixedBitSet::default(),
            _index_type: PhantomData,
        }
    }
}

impl<T: SparseSetIndex> AccessFilters<T> {
    fn is_ruled_out_by(&self, other: &Self) -> bool {
        // Although not technically complete, we don't consider the case when `AccessFilters`'s
        // `without` bitset contradicts its own `with` bitset (e.g. `(With<A>, Without<A>)`).
        // Such query would be considered compatible with any other query, but as it's almost
        // always an error, we ignore this case instead of treating such query as compatible
        // with others.
        !self.with.is_disjoint(&other.without) || !self.without.is_disjoint(&other.with)
    }
}

/// A collection of [`FilteredAccess`] instances.
///
/// Used internally to statically check if systems have conflicting access.
///
/// It stores multiple sets of accesses.
/// - A "combined" set, which is the access of all filters in this set combined.
/// - The set of access of each individual filters in this set.
#[derive(Debug, PartialEq, Eq)]
pub struct FilteredAccessSet<T: SparseSetIndex> {
    combined_access: Access<T>,
    filtered_accesses: Vec<FilteredAccess<T>>,
}

// This is needed since `#[derive(Clone)]` does not generate optimized `clone_from`.
impl<T: SparseSetIndex> Clone for FilteredAccessSet<T> {
    fn clone(&self) -> Self {
        Self {
            combined_access: self.combined_access.clone(),
            filtered_accesses: self.filtered_accesses.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.combined_access.clone_from(&source.combined_access);
        self.filtered_accesses.clone_from(&source.filtered_accesses);
    }
}

impl<T: SparseSetIndex> FilteredAccessSet<T> {
    /// Returns a reference to the unfiltered access of the entire set.
    #[inline]
    pub fn combined_access(&self) -> &Access<T> {
        &self.combined_access
    }

    /// Returns `true` if this and `other` can be active at the same time.
    ///
    /// Access conflict resolution happen in two steps:
    /// 1. A "coarse" check, if there is no mutual unfiltered conflict between
    ///    `self` and `other`, we already know that the two access sets are
    ///    compatible.
    /// 2. A "fine grained" check, it kicks in when the "coarse" check fails.
    ///    the two access sets might still be compatible if some of the accesses
    ///    are restricted with the [`With`](super::With) or [`Without`](super::Without) filters so that access is
    ///    mutually exclusive. The fine grained phase iterates over all filters in
    ///    the `self` set and compares it to all the filters in the `other` set,
    ///    making sure they are all mutually compatible.
    pub fn is_compatible(&self, other: &FilteredAccessSet<T>) -> bool {
        if self.combined_access.is_compatible(other.combined_access()) {
            return true;
        }
        for filtered in &self.filtered_accesses {
            for other_filtered in &other.filtered_accesses {
                if !filtered.is_compatible(other_filtered) {
                    return false;
                }
            }
        }
        true
    }

    /// Returns a vector of elements that this set and `other` cannot access at the same time.
    pub fn get_conflicts(&self, other: &FilteredAccessSet<T>) -> AccessConflicts {
        // if the unfiltered access is incompatible, must check each pair
        let mut conflicts = AccessConflicts::empty();
        if !self.combined_access.is_compatible(other.combined_access()) {
            for filtered in &self.filtered_accesses {
                for other_filtered in &other.filtered_accesses {
                    conflicts.add(&filtered.get_conflicts(other_filtered));
                }
            }
        }
        conflicts
    }

    /// Returns a vector of elements that this set and `other` cannot access at the same time.
    pub fn get_conflicts_single(&self, filtered_access: &FilteredAccess<T>) -> AccessConflicts {
        // if the unfiltered access is incompatible, must check each pair
        let mut conflicts = AccessConflicts::empty();
        if !self.combined_access.is_compatible(filtered_access.access()) {
            for filtered in &self.filtered_accesses {
                conflicts.add(&filtered.get_conflicts(filtered_access));
            }
        }
        conflicts
    }

    /// Adds the filtered access to the set.
    pub fn add(&mut self, filtered_access: FilteredAccess<T>) {
        self.combined_access.extend(&filtered_access.access);
        self.filtered_accesses.push(filtered_access);
    }

    /// Adds a read access to a resource to the set.
    pub fn add_unfiltered_read(&mut self, index: T) {
        let mut filter = FilteredAccess::default();
        filter.add_resource_read(index);
        self.add(filter);
    }

    /// Adds a write access to a resource to the set.
    pub fn add_unfiltered_write(&mut self, index: T) {
        let mut filter = FilteredAccess::default();
        filter.add_resource_write(index);
        self.add(filter);
    }

    /// Adds read access to all resources to the set.
    pub fn add_unfiltered_read_all(&mut self) {
        let mut filter = FilteredAccess::default();
        filter.access.read_all();
        self.add(filter);
    }

    /// Adds write access to all resources to the set.
    pub fn add_unfiltered_write_all(&mut self) {
        let mut filter = FilteredAccess::default();
        filter.access.write_all();
        self.add(filter);
    }

    /// Adds all of the accesses from the passed set to `self`.
    pub fn extend(&mut self, filtered_access_set: FilteredAccessSet<T>) {
        self.combined_access
            .extend(&filtered_access_set.combined_access);
        self.filtered_accesses
            .extend(filtered_access_set.filtered_accesses);
    }

    /// Marks the set as reading all possible indices of type T.
    pub fn read_all(&mut self) {
        let mut filter = FilteredAccess::matches_everything();
        filter.read_all();
        self.add(filter);
    }

    /// Marks the set as writing all T.
    pub fn write_all(&mut self) {
        let mut filter = FilteredAccess::matches_everything();
        filter.write_all();
        self.add(filter);
    }

    /// Removes all accesses stored in this set.
    pub fn clear(&mut self) {
        self.combined_access.clear();
        self.filtered_accesses.clear();
    }
}

impl<T: SparseSetIndex> Default for FilteredAccessSet<T> {
    fn default() -> Self {
        Self {
            combined_access: Default::default(),
            filtered_accesses: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::query::{
        access::AccessFilters, Access, AccessConflicts, ComponentAccessKind, FilteredAccess,
        FilteredAccessSet, UnboundedAccessError,
    };
    use alloc::{vec, vec::Vec};
    use core::marker::PhantomData;
    use fixedbitset::FixedBitSet;

    fn create_sample_access() -> Access<usize> {
        let mut access = Access::<usize>::default();

        access.add_read(1);
        access.add_read(2);
        access.add_write(3);
        access.add_archetypal(5);
        access.read_all();

        access
    }

    fn create_sample_filtered_access() -> FilteredAccess<usize> {
        let mut filtered_access = FilteredAccess::<usize>::default();

        filtered_access.add_component_write(1);
        filtered_access.add_component_read(2);
        filtered_access.add_required(3);
        filtered_access.and_with(4);

        filtered_access
    }

    fn create_sample_access_filters() -> AccessFilters<usize> {
        let mut access_filters = AccessFilters::<usize>::default();

        access_filters.with.grow_and_insert(3);
        access_filters.without.grow_and_insert(5);

        access_filters
    }

    fn create_sample_filtered_access_set() -> FilteredAccessSet<usize> {
        let mut filtered_access_set = FilteredAccessSet::<usize>::default();

        filtered_access_set.add_unfiltered_read(2);
        filtered_access_set.add_unfiltered_write(4);
        filtered_access_set.read_all();

        filtered_access_set
    }

    #[test]
    fn test_access_clone() {
        let original: Access<usize> = create_sample_access();
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_access_clone_from() {
        let original: Access<usize> = create_sample_access();
        let mut cloned = Access::<usize>::default();

        cloned.add_write(7);
        cloned.add_read(4);
        cloned.add_archetypal(8);
        cloned.write_all();

        cloned.clone_from(&original);

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_filtered_access_clone() {
        let original: FilteredAccess<usize> = create_sample_filtered_access();
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_filtered_access_clone_from() {
        let original: FilteredAccess<usize> = create_sample_filtered_access();
        let mut cloned = FilteredAccess::<usize>::default();

        cloned.add_component_write(7);
        cloned.add_component_read(4);
        cloned.append_or(&FilteredAccess::default());

        cloned.clone_from(&original);

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_access_filters_clone() {
        let original: AccessFilters<usize> = create_sample_access_filters();
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_access_filters_clone_from() {
        let original: AccessFilters<usize> = create_sample_access_filters();
        let mut cloned = AccessFilters::<usize>::default();

        cloned.with.grow_and_insert(1);
        cloned.without.grow_and_insert(2);

        cloned.clone_from(&original);

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_filtered_access_set_clone() {
        let original: FilteredAccessSet<usize> = create_sample_filtered_access_set();
        let cloned = original.clone();

        assert_eq!(original, cloned);
    }

    #[test]
    fn test_filtered_access_set_from() {
        let original: FilteredAccessSet<usize> = create_sample_filtered_access_set();
        let mut cloned = FilteredAccessSet::<usize>::default();

        cloned.add_unfiltered_read(7);
        cloned.add_unfiltered_write(9);
        cloned.write_all();

        cloned.clone_from(&original);

        assert_eq!(original, cloned);
    }

    #[test]
    fn read_all_access_conflicts() {
        // read_all / single write
        let mut access_a = Access::<usize>::default();
        access_a.add_write(0);

        let mut access_b = Access::<usize>::default();
        access_b.read_all();

        assert!(!access_b.is_compatible(&access_a));

        // read_all / read_all
        let mut access_a = Access::<usize>::default();
        access_a.read_all();

        let mut access_b = Access::<usize>::default();
        access_b.read_all();

        assert!(access_b.is_compatible(&access_a));
    }

    #[test]
    fn access_get_conflicts() {
        let mut access_a = Access::<usize>::default();
        access_a.add_read(0);
        access_a.add_read(1);

        let mut access_b = Access::<usize>::default();
        access_b.add_read(0);
        access_b.add_write(1);

        assert_eq!(access_a.get_conflicts(&access_b), vec![1_usize].into());

        let mut access_c = Access::<usize>::default();
        access_c.add_write(0);
        access_c.add_write(1);

        assert_eq!(
            access_a.get_conflicts(&access_c),
            vec![0_usize, 1_usize].into()
        );
        assert_eq!(
            access_b.get_conflicts(&access_c),
            vec![0_usize, 1_usize].into()
        );

        let mut access_d = Access::<usize>::default();
        access_d.add_read(0);

        assert_eq!(access_d.get_conflicts(&access_a), AccessConflicts::empty());
        assert_eq!(access_d.get_conflicts(&access_b), AccessConflicts::empty());
        assert_eq!(access_d.get_conflicts(&access_c), vec![0_usize].into());
    }

    #[test]
    fn filtered_combined_access() {
        let mut access_a = FilteredAccessSet::<usize>::default();
        access_a.add_unfiltered_read(1);

        let mut filter_b = FilteredAccess::<usize>::default();
        filter_b.add_resource_write(1);

        let conflicts = access_a.get_conflicts_single(&filter_b);
        assert_eq!(
            &conflicts,
            &AccessConflicts::from(vec![1_usize]),
            "access_a: {access_a:?}, filter_b: {filter_b:?}"
        );
    }

    #[test]
    fn filtered_access_extend() {
        let mut access_a = FilteredAccess::<usize>::default();
        access_a.add_component_read(0);
        access_a.add_component_read(1);
        access_a.and_with(2);

        let mut access_b = FilteredAccess::<usize>::default();
        access_b.add_component_read(0);
        access_b.add_component_write(3);
        access_b.and_without(4);

        access_a.extend(&access_b);

        let mut expected = FilteredAccess::<usize>::default();
        expected.add_component_read(0);
        expected.add_component_read(1);
        expected.and_with(2);
        expected.add_component_write(3);
        expected.and_without(4);

        assert!(access_a.eq(&expected));
    }

    #[test]
    fn filtered_access_extend_or() {
        let mut access_a = FilteredAccess::<usize>::default();
        // Exclusive access to `(&mut A, &mut B)`.
        access_a.add_component_write(0);
        access_a.add_component_write(1);

        // Filter by `With<C>`.
        let mut access_b = FilteredAccess::<usize>::default();
        access_b.and_with(2);

        // Filter by `(With<D>, Without<E>)`.
        let mut access_c = FilteredAccess::<usize>::default();
        access_c.and_with(3);
        access_c.and_without(4);

        // Turns `access_b` into `Or<(With<C>, (With<D>, Without<D>))>`.
        access_b.append_or(&access_c);
        // Applies the filters to the initial query, which corresponds to the FilteredAccess'
        // representation of `Query<(&mut A, &mut B), Or<(With<C>, (With<D>, Without<E>))>>`.
        access_a.extend(&access_b);

        // Construct the expected `FilteredAccess` struct.
        // The intention here is to test that exclusive access implied by `add_write`
        // forms correct normalized access structs when extended with `Or` filters.
        let mut expected = FilteredAccess::<usize>::default();
        expected.add_component_write(0);
        expected.add_component_write(1);
        // The resulted access is expected to represent `Or<((With<A>, With<B>, With<C>), (With<A>, With<B>, With<D>, Without<E>))>`.
        expected.filter_sets = vec![
            AccessFilters {
                with: FixedBitSet::with_capacity_and_blocks(3, [0b111]),
                without: FixedBitSet::default(),
                _index_type: PhantomData,
            },
            AccessFilters {
                with: FixedBitSet::with_capacity_and_blocks(4, [0b1011]),
                without: FixedBitSet::with_capacity_and_blocks(5, [0b10000]),
                _index_type: PhantomData,
            },
        ];

        assert_eq!(access_a, expected);
    }

    #[test]
    fn try_iter_component_access_simple() {
        let mut access = Access::<usize>::default();

        access.add_read(1);
        access.add_read(2);
        access.add_write(3);
        access.add_archetypal(5);

        let result = access
            .try_iter_component_access()
            .map(Iterator::collect::<Vec<_>>);

        assert_eq!(
            result,
            Ok(vec![
                ComponentAccessKind::Shared(1),
                ComponentAccessKind::Shared(2),
                ComponentAccessKind::Exclusive(3),
                ComponentAccessKind::Archetypal(5),
            ]),
        );
    }

    #[test]
    fn try_iter_component_access_unbounded_write_all() {
        let mut access = Access::<usize>::default();

        access.add_read(1);
        access.add_read(2);
        access.write_all();

        let result = access
            .try_iter_component_access()
            .map(Iterator::collect::<Vec<_>>);

        assert_eq!(
            result,
            Err(UnboundedAccessError {
                writes_inverted: true,
                read_and_writes_inverted: true
            }),
        );
    }

    #[test]
    fn try_iter_component_access_unbounded_read_all() {
        let mut access = Access::<usize>::default();

        access.add_read(1);
        access.add_read(2);
        access.read_all();

        let result = access
            .try_iter_component_access()
            .map(Iterator::collect::<Vec<_>>);

        assert_eq!(
            result,
            Err(UnboundedAccessError {
                writes_inverted: false,
                read_and_writes_inverted: true
            }),
        );
    }
}
