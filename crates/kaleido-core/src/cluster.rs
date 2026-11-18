//! Union-find grouping of perceptually similar hashes.
//!
//! Builds a threshold graph (an edge joins any pair whose hamming distance is
//! within `threshold`), merges connected components with union-find, and
//! reports groups of at least `min_size` members. Transitive chains merge into
//! a single group (a≈b, b≈c ⇒ {a,b,c}), which is the semantics the CLI's
//! `dupes` command is built on.

use crate::phash::Hash64;
use crate::unionfind::UnionFind;

/// Group indices of perceptually similar hashes.
///
/// Returns groups with at least `min_size` members. Members are ascending
/// indices; groups are ordered by their smallest member index, so output is
/// deterministic for a given input order.
pub fn cluster_similar(hashes: &[Hash64], threshold: u32, min_size: usize) -> Vec<Vec<usize>> {
    use rayon::prelude::*;

    let n = hashes.len();
    if n == 0 {
        return Vec::new();
    }

    // Threshold graph edges, discovered pairwise (parallel over the outer index).
    let edges: Vec<(usize, usize)> = hashes
        .par_iter()
        .enumerate()
        .flat_map_iter(|(i, h)| {
            hashes
                .iter()
                .enumerate()
                .skip(i + 1)
                .filter(|(_, other)| h.hamming(**other) <= threshold)
                .map(move |(j, _)| (i, j))
        })
        .collect();

    let mut uf = UnionFind::new(n);
    for (i, j) in edges {
        uf.union(i, j);
    }

    // Collect members per root, preserving ascending index order.
    let mut by_root: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for i in 0..n {
        by_root.entry(uf.find(i)).or_default().push(i);
    }
    let mut groups: Vec<Vec<usize>> = by_root
        .into_values()
        .filter(|group| group.len() >= min_size)
        .collect();
    groups.sort_by_key(|group| group[0]);
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_hashes_form_one_group() {
        let hashes = [Hash64(7), Hash64(7), Hash64(7)];
        assert_eq!(cluster_similar(&hashes, 0, 2), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn transitive_chain_merges_into_one_group() {
        // d(0,1)=1, d(1,2)=1, d(0,2)=2: threshold 1 merges all three
        // (connected components, not cliques).
        let hashes = [Hash64(0), Hash64(1), Hash64(0b11)];
        assert_eq!(cluster_similar(&hashes, 1, 2), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn distant_hashes_stay_singletons() {
        let hashes = [Hash64(0), Hash64(1), Hash64(0xF << 40), Hash64(u64::MAX)];
        assert_eq!(cluster_similar(&hashes, 1, 2), vec![vec![0, 1]]);
    }

    #[test]
    fn min_size_filters_singletons_and_empty_input() {
        let hashes = [Hash64(0), Hash64(1 << 20)];
        assert!(cluster_similar(&hashes, 0, 2).is_empty());
        assert!(cluster_similar(&[], 5, 2).is_empty());
    }
}
