// SPDX-License-Identifier: MPL-2.0

use std::borrow::Borrow;
use std::fmt::{Debug, Display};
use std::hash::Hash;

use crate::{Ranges, SetRelation};

/// A set of versions.
///
/// See [`Ranges`] for an implementation.
///
/// The methods with default implementations can be overwritten for better performance, but their
/// output must be equal to the default implementation.
///
/// # Equality and hashing
///
/// It is important that the `Eq` and `Hash` traits are implemented so that if two sets contain the
/// same versions, they are equal under `Eq` and produce the same hash. In particular, you can only
/// derive these traits if equality is strictly equivalent to structural equality, i.e. if version
/// sets are always stored in canonical representations. Such problems may arise if your
/// implementations of `complement()` and `intersection()` do not return canonical representations.
///
/// For example, `>=1,<4 || >=2,<5` and `>=1,<4 || >=3,<5` are equal, because they can both be
/// normalized to `>=1,<5`.
///
/// Note that pubgrub does not know which versions actually exist for a package, the contract
/// is about upholding the mathematical properties of set operations, assuming all versions are
/// possible. This is required for the solver to determine the relationship of version sets to each
/// other.
pub trait VersionSet: Debug + Display + Clone + Eq + Hash {
    /// Version type associated with the sets manipulated.
    type V: Debug + Display + Clone + Ord;

    // Constructors

    /// An empty set containing no version.
    fn empty() -> Self;

    /// A set containing only the given version.
    fn singleton(v: Self::V) -> Self;

    // Operations

    /// The set of all version that are not in this set.
    fn complement(&self) -> Self;

    /// The set of all versions that are in both sets.
    fn intersection(&self, other: &Self) -> Self;

    /// Whether the version is part of this set.
    fn contains(&self, v: &Self::V) -> bool;

    /// Returns whether each version is part of this set, in iterator order.
    ///
    /// `versions` must be sorted in nondecreasing order. The default calls [`Self::contains`] for
    /// every version. Implementations can override this to traverse the versions and the set in a
    /// single pass.
    fn contains_many<'s, I, BV>(&'s self, versions: I) -> impl Iterator<Item = bool> + 's
    where
        I: Iterator<Item = BV> + 's,
        BV: Borrow<Self::V> + 's,
    {
        versions.map(move |version| self.contains(version.borrow()))
    }

    // Automatically implemented functions

    /// The set containing all versions.
    ///
    /// The default implementation is the complement of the empty set.
    fn full() -> Self {
        Self::empty().complement()
    }

    /// The set of all versions that are either (or both) of the sets.
    ///
    /// The default implementation is complement of the intersection of the complements of both sets
    /// (De Morgan's law).
    fn union(&self, other: &Self) -> Self {
        self.complement()
            .intersection(&other.complement())
            .complement()
    }

    /// The set of versions that are in `self` but not in `other`.
    ///
    /// Implementations can override this to avoid materializing the complement.
    fn difference(&self, other: &Self) -> Self {
        self.intersection(&other.complement())
    }

    /// Whether the ranges have no overlapping segments.
    fn is_disjoint(&self, other: &Self) -> bool {
        self.intersection(other) == Self::empty()
    }

    /// Whether all ranges of `self` are contained in `other`.
    fn subset_of(&self, other: &Self) -> bool {
        self == &self.intersection(other)
    }

    /// Classifies `self` as a subset of, disjoint from, or partially overlapping with `other`.
    ///
    /// Implementations can override this to avoid traversing both sets once for [`Self::subset_of`]
    /// and again for [`Self::is_disjoint`].
    /// An empty `self` must be classified as [`SetRelation::Subset`].
    fn relation(&self, other: &Self) -> SetRelation {
        if self.subset_of(other) {
            SetRelation::Subset
        } else if self.is_disjoint(other) {
            SetRelation::Disjoint
        } else {
            SetRelation::Overlapping
        }
    }
}

/// [`Ranges`] contains optimized implementations of all operations.
impl<T: Debug + Display + Clone + Eq + Ord + Hash> VersionSet for Ranges<T> {
    type V = T;

    fn empty() -> Self {
        Ranges::empty()
    }

    fn singleton(v: Self::V) -> Self {
        Ranges::singleton(v)
    }

    fn complement(&self) -> Self {
        Ranges::complement(self)
    }

    fn intersection(&self, other: &Self) -> Self {
        Ranges::intersection(self, other)
    }

    fn contains(&self, v: &Self::V) -> bool {
        Ranges::contains(self, v)
    }

    fn contains_many<'s, I, BV>(&'s self, versions: I) -> impl Iterator<Item = bool> + 's
    where
        I: Iterator<Item = BV> + 's,
        BV: Borrow<Self::V> + 's,
    {
        Ranges::contains_many(self, versions)
    }

    fn full() -> Self {
        Ranges::full()
    }

    fn union(&self, other: &Self) -> Self {
        Ranges::union(self, other)
    }

    fn difference(&self, other: &Self) -> Self {
        Ranges::difference(self, other)
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        Ranges::is_disjoint(self, other)
    }

    fn subset_of(&self, other: &Self) -> bool {
        Ranges::subset_of(self, other)
    }

    fn relation(&self, other: &Self) -> SetRelation {
        Ranges::relation(self, other)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::fmt::{Display, Formatter};
    use std::hash::{Hash, Hasher};
    use std::rc::Rc;

    use super::*;

    /// A version set that counts scalar membership checks.
    #[derive(Clone, Debug)]
    struct ScalarVersionSet {
        range: Ranges<u32>,
        contains_calls: Rc<Cell<usize>>,
    }

    impl PartialEq for ScalarVersionSet {
        fn eq(&self, other: &Self) -> bool {
            self.range == other.range
        }
    }

    impl Eq for ScalarVersionSet {}

    impl Hash for ScalarVersionSet {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.range.hash(state);
        }
    }

    impl Display for ScalarVersionSet {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.range, formatter)
        }
    }

    impl VersionSet for ScalarVersionSet {
        type V = u32;

        fn empty() -> Self {
            Self {
                range: Ranges::empty(),
                contains_calls: Rc::new(Cell::new(0)),
            }
        }

        fn singleton(version: Self::V) -> Self {
            Self {
                range: Ranges::singleton(version),
                contains_calls: Rc::new(Cell::new(0)),
            }
        }

        fn complement(&self) -> Self {
            Self {
                range: self.range.complement(),
                contains_calls: Rc::clone(&self.contains_calls),
            }
        }

        fn intersection(&self, other: &Self) -> Self {
            Self {
                range: self.range.intersection(&other.range),
                contains_calls: Rc::clone(&self.contains_calls),
            }
        }

        fn contains(&self, version: &Self::V) -> bool {
            self.contains_calls.set(self.contains_calls.get() + 1);
            self.range.contains(version)
        }
    }

    #[test]
    fn contains_many_default_delegates_to_contains() {
        let contains_calls = Rc::new(Cell::new(0));
        let range = ScalarVersionSet {
            range: Ranges::from_range_bounds(2_u32..=4_u32),
            contains_calls: Rc::clone(&contains_calls),
        };
        let versions = [1, 2, 2, 4, 5];

        assert_eq!(
            range.contains_many(versions.iter()).collect::<Vec<_>>(),
            [false, true, true, true, false]
        );
        assert_eq!(contains_calls.get(), versions.len());
    }
}
