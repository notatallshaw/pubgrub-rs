use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::convert::Infallible;

use crate::{
    Dependencies, DependencyConstraints, DependencyProvider, Map, Package,
    PackageResolutionStatistics, VersionSet,
};

/// A basic implementation of [DependencyProvider].
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "VS::V: serde::Serialize, VS: serde::Serialize, P: serde::Serialize",
        deserialize = "VS::V: serde::Deserialize<'de>, VS: serde::Deserialize<'de>, P: serde::Deserialize<'de>"
    ))
)]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct OfflineDependencyProvider<P: Package, VS: VersionSet> {
    dependencies: Map<P, BTreeMap<VS::V, DependencyConstraints<P, VS>>>,
}

impl<P: Package, VS: VersionSet> OfflineDependencyProvider<P, VS> {
    /// Creates an empty OfflineDependencyProvider with no dependencies.
    pub fn new() -> Self {
        Self {
            dependencies: Map::default(),
        }
    }

    /// Registers the dependencies of a package and version pair.
    /// Dependencies must be added with a single call to
    /// [add_dependencies](OfflineDependencyProvider::add_dependencies).
    /// All subsequent calls to
    /// [add_dependencies](OfflineDependencyProvider::add_dependencies) for a given
    /// package version pair will replace the dependencies by the new ones.
    ///
    /// The API does not allow to add dependencies one at a time to uphold an assumption that
    /// [OfflineDependencyProvider.get_dependencies(p, v)](OfflineDependencyProvider::get_dependencies)
    /// provides all dependencies of a given package (p) and version (v) pair.
    pub fn add_dependencies<I: IntoIterator<Item = (P, VS)>>(
        &mut self,
        package: P,
        version: impl Into<VS::V>,
        dependencies: I,
    ) {
        let package_deps = dependencies.into_iter().collect();
        let v = version.into();
        *self
            .dependencies
            .entry(package)
            .or_default()
            .entry(v)
            .or_default() = package_deps;
    }

    /// Lists packages that have been saved.
    pub fn packages(&self) -> impl Iterator<Item = &P> {
        self.dependencies.keys()
    }

    /// Lists versions of saved packages in sorted order.
    /// Returns [None] if no information is available regarding that package.
    pub fn versions(&self, package: &P) -> Option<impl Iterator<Item = &VS::V>> {
        self.dependencies.get(package).map(|k| k.keys())
    }

    /// Lists dependencies of a given package and version.
    /// Returns [None] if no information is available regarding that package and version pair.
    fn dependencies(&self, package: &P, version: &VS::V) -> Option<DependencyConstraints<P, VS>> {
        self.dependencies.get(package)?.get(version).cloned()
    }
}

/// An implementation of [DependencyProvider] that
/// contains all dependency information available in memory.
/// Currently packages are picked with the fewest versions contained in the constraints first.
/// But, that may change in new versions if better heuristics are found.
/// Versions are picked with the newest versions first.
impl<P: Package, VS: VersionSet> DependencyProvider for OfflineDependencyProvider<P, VS> {
    type P = P;
    type V = VS::V;
    type VS = VS;
    type M = String;

    type Err = Infallible;

    #[inline]
    fn choose_version(&self, package: &P, range: &VS) -> Result<Option<VS::V>, Infallible> {
        Ok(self
            .dependencies
            .get(package)
            .and_then(|versions| versions.keys().rev().find(|v| range.contains(v)).cloned()))
    }

    type Priority = (u32, Reverse<usize>);

    #[inline]
    fn prioritize(
        &self,
        package: &Self::P,
        range: &Self::VS,
        package_statistics: &PackageResolutionStatistics,
    ) -> Self::Priority {
        let version_count = self
            .dependencies
            .get(package)
            .map(|versions| {
                if versions.len() == 1 {
                    usize::from(
                        range.contains(versions.first_key_value().expect("length checked").0),
                    )
                } else {
                    range
                        .contains_many(versions.keys())
                        .filter(|contains| *contains)
                        .count()
                }
            })
            .unwrap_or(0);
        if version_count == 0 {
            return (u32::MAX, Reverse(0));
        }
        (package_statistics.conflict_count(), Reverse(version_count))
    }

    #[inline]
    fn get_dependencies(
        &self,
        package: &P,
        version: &VS::V,
    ) -> Result<Dependencies<P, VS, Self::M>, Infallible> {
        Ok(match self.dependencies(package, version) {
            None => {
                Dependencies::Unavailable("its dependencies could not be determined".to_string())
            }
            Some(dependencies) => Dependencies::Available(dependencies),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;
    use std::cell::Cell;
    use std::cmp::Ordering;
    use std::fmt::{Display, Formatter};
    use std::hash::{Hash, Hasher};
    use std::ops::Bound::Included;
    use std::rc::Rc;

    use super::*;
    use crate::Ranges;

    /// A version value that counts ordering comparisons.
    #[derive(Clone, Debug)]
    struct CountedVersion {
        value: u32,
        comparisons: Rc<Cell<usize>>,
    }

    impl CountedVersion {
        fn new(value: u32, comparisons: &Rc<Cell<usize>>) -> Self {
            Self {
                value,
                comparisons: Rc::clone(comparisons),
            }
        }
    }

    impl Display for CountedVersion {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.value, formatter)
        }
    }

    impl PartialEq for CountedVersion {
        fn eq(&self, other: &Self) -> bool {
            self.value == other.value
        }
    }

    impl Eq for CountedVersion {}

    impl PartialOrd for CountedVersion {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for CountedVersion {
        fn cmp(&self, other: &Self) -> Ordering {
            self.comparisons.set(self.comparisons.get() + 1);
            self.value.cmp(&other.value)
        }
    }

    impl Hash for CountedVersion {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.value.hash(state);
        }
    }

    /// A version set that records scalar and batched membership calls.
    #[derive(Clone, Debug)]
    struct TrackingVersionSet {
        range: Ranges<u32>,
        scalar_calls: Rc<Cell<usize>>,
        batched_calls: Rc<Cell<usize>>,
    }

    impl TrackingVersionSet {
        fn new(
            range: Ranges<u32>,
            scalar_calls: &Rc<Cell<usize>>,
            batched_calls: &Rc<Cell<usize>>,
        ) -> Self {
            Self {
                range,
                scalar_calls: Rc::clone(scalar_calls),
                batched_calls: Rc::clone(batched_calls),
            }
        }
    }

    impl PartialEq for TrackingVersionSet {
        fn eq(&self, other: &Self) -> bool {
            self.range == other.range
        }
    }

    impl Eq for TrackingVersionSet {}

    impl Hash for TrackingVersionSet {
        fn hash<H: Hasher>(&self, state: &mut H) {
            self.range.hash(state);
        }
    }

    impl Display for TrackingVersionSet {
        fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.range, formatter)
        }
    }

    impl VersionSet for TrackingVersionSet {
        type V = u32;

        fn empty() -> Self {
            Self {
                range: Ranges::empty(),
                scalar_calls: Rc::default(),
                batched_calls: Rc::default(),
            }
        }

        fn singleton(version: Self::V) -> Self {
            Self {
                range: Ranges::singleton(version),
                scalar_calls: Rc::default(),
                batched_calls: Rc::default(),
            }
        }

        fn complement(&self) -> Self {
            Self {
                range: self.range.complement(),
                scalar_calls: Rc::clone(&self.scalar_calls),
                batched_calls: Rc::clone(&self.batched_calls),
            }
        }

        fn intersection(&self, other: &Self) -> Self {
            Self {
                range: self.range.intersection(&other.range),
                scalar_calls: Rc::clone(&self.scalar_calls),
                batched_calls: Rc::clone(&self.batched_calls),
            }
        }

        fn contains(&self, version: &Self::V) -> bool {
            self.scalar_calls.set(self.scalar_calls.get() + 1);
            self.range.contains(version)
        }

        fn contains_many<'s, I, BV>(&'s self, versions: I) -> impl Iterator<Item = bool> + 's
        where
            I: Iterator<Item = BV> + 's,
            BV: Borrow<Self::V> + 's,
        {
            self.batched_calls.set(self.batched_calls.get() + 1);
            versions.map(move |version| self.range.contains(version.borrow()))
        }
    }

    #[test]
    fn prioritize_matches_scalar_membership_across_range_shapes() {
        let mut provider = OfflineDependencyProvider::<u8, Ranges<u32>>::new();
        for version in [1_u32, 3, 5, 7] {
            provider.add_dependencies(0, version, []);
        }
        let statistics = PackageResolutionStatistics::default();

        for range in [
            Ranges::empty(),
            Ranges::full(),
            Ranges::singleton(3_u32),
            Ranges::from_range_bounds(2_u32..=6_u32),
            Ranges::singleton(8_u32),
        ] {
            let version_count = provider
                .versions(&0)
                .unwrap()
                .filter(|version| range.contains(*version))
                .count();
            let expected = if version_count == 0 {
                (u32::MAX, Reverse(0))
            } else {
                (0, Reverse(version_count))
            };

            assert_eq!(provider.prioritize(&0, &range, &statistics), expected);
        }

        assert_eq!(
            provider.prioritize(&1, &Ranges::full(), &statistics),
            (u32::MAX, Reverse(0))
        );
    }

    #[test]
    fn prioritize_dispatches_by_version_count() {
        let scalar_calls = Rc::new(Cell::new(0));
        let batched_calls = Rc::new(Cell::new(0));
        let range = TrackingVersionSet::new(
            Ranges::from_range_bounds(1_u32..=2_u32),
            &scalar_calls,
            &batched_calls,
        );
        let statistics = PackageResolutionStatistics::default();
        let mut provider = OfflineDependencyProvider::new();

        assert_eq!(
            provider.prioritize(&0, &range, &statistics),
            (u32::MAX, Reverse(0))
        );
        assert_eq!(scalar_calls.get(), 0);
        assert_eq!(batched_calls.get(), 0);

        provider.add_dependencies(0, 2_u32, []);

        assert_eq!(
            provider.prioritize(&0, &range, &statistics),
            (0, Reverse(1))
        );
        assert_eq!(scalar_calls.get(), 1);
        assert_eq!(batched_calls.get(), 0);

        scalar_calls.set(0);
        provider.add_dependencies(0, 1_u32, []);

        assert_eq!(
            provider.prioritize(&0, &range, &statistics),
            (0, Reverse(2))
        );
        assert_eq!(scalar_calls.get(), 0);
        assert_eq!(batched_calls.get(), 1);
    }

    #[test]
    fn prioritize_dispatches_to_batched_range_membership() {
        let comparisons = Rc::new(Cell::new(0));
        let version = |value| CountedVersion::new(value, &comparisons);
        let mut provider = OfflineDependencyProvider::<u8, Ranges<CountedVersion>>::new();
        // Insert in reverse order to confirm the map supplies sorted keys.
        for value in (0..64).rev() {
            provider.add_dependencies(0, version(value), []);
        }
        let range = Ranges::from_iter(
            (0..64)
                .step_by(4)
                .map(|value| (Included(version(value)), Included(version(value)))),
        );

        comparisons.set(0);
        let priority = provider.prioritize(&0, &range, &PackageResolutionStatistics::default());
        let batched_comparisons = comparisons.get();

        comparisons.set(0);
        let scalar_count = provider
            .versions(&0)
            .unwrap()
            .filter(|candidate| range.contains(candidate))
            .count();
        let scalar_comparisons = comparisons.get();

        assert_eq!(priority, (0, Reverse(scalar_count)));
        assert_eq!(scalar_count, 16);
        assert!(
            batched_comparisons < scalar_comparisons,
            "batched traversal used {batched_comparisons} comparisons; scalar membership used {scalar_comparisons}"
        );
    }
}
