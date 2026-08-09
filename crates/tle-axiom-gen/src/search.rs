//! Beam search over the knowledge graph for path discovery.
//!
//! Implements energy-guided beam search that explores the knowledge graph
//! starting from seed entities, expanding paths by following adjacent triples,
//! and pruning by energy score to maintain only the most promising candidates.

use std::collections::HashSet;
use tle_vsa::{Codebook, HyperVector};

use crate::energy::{compute_energy, EnergyConfig};
use crate::graph::{KnowledgeGraph, Triple};

/// Configuration for the beam search algorithm.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Number of candidate paths to maintain at each step.
    pub beam_width: usize,
    /// Maximum number of hops (triples) in a path.
    pub max_hops: usize,
    /// Minimum energy threshold for path acceptance.
    pub energy_threshold: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            beam_width: 32,
            max_hops: 10,
            energy_threshold: 0.8,
        }
    }
}

/// A scored path: a sequence of triple indices with an energy score.
#[derive(Debug, Clone)]
pub struct ScoredPath {
    /// Indices into `KnowledgeGraph::triples` forming the path.
    pub path: Vec<usize>,
    /// Energy score of the path (higher = better).
    pub energy: f32,
}

/// Perform beam search over the knowledge graph to find high-energy paths.
///
/// Algorithm:
/// 1. Initialize beam with single-triple paths from start entities
/// 2. For each step up to max_hops:
///    a. Expand each path by appending adjacent triples
///    b. Score all candidates with the energy function
///    c. Prune to keep only beam_width best candidates
/// 3. Return all discovered paths sorted by energy
pub fn beam_search(
    graph: &KnowledgeGraph,
    start_entities: &[usize],
    query_vector: &HyperVector,
    codebook: &mut Codebook,
    energy_config: &EnergyConfig,
    search_config: &SearchConfig,
) -> Vec<ScoredPath> {
    if start_entities.is_empty() || graph.triples.is_empty() {
        return Vec::new();
    }

    // Step 1: Initialize with single-triple paths from start entities
    let mut beam: Vec<ScoredPath> = Vec::new();

    for &entity_id in start_entities {
        for (idx, triple) in graph.triples.iter().enumerate() {
            if triple.subject_id == entity_id || triple.object_id == entity_id {
                let path_triples = vec![*triple];
                let energy = compute_energy(
                    &path_triples,
                    query_vector,
                    energy_config,
                    &graph.entities,
                    &graph.relations,
                    codebook,
                );
                beam.push(ScoredPath {
                    path: vec![idx],
                    energy,
                });
            }
        }
    }

    // Sort and prune initial beam
    beam.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));
    beam.truncate(search_config.beam_width);

    // Collect all discovered paths (both intermediate and final)
    let mut all_paths: Vec<ScoredPath> = beam.clone();

    // Step 2: Iteratively expand paths
    // Keep recursion bounded for safety while allowing deep compositional chains.
    for _hop in 1..search_config.max_hops.min(64) {
        let mut candidates: Vec<ScoredPath> = Vec::new();

        for scored_path in &beam {
            let last_triple_idx = *scored_path.path.last().unwrap();
            let last_triple = &graph.triples[last_triple_idx];

            // Expand from both the subject and object of the last triple
            let frontier_entities = [last_triple.subject_id, last_triple.object_id];

            for &frontier in &frontier_entities {
                for (idx, triple) in graph.triples.iter().enumerate() {
                    // Skip triples already in the path
                    if scored_path.path.contains(&idx) {
                        continue;
                    }

                    // Prevent graph cycles: a path may revisit its frontier, but
                    // the newly introduced endpoint must not already be in the path.
                    let mut visited_entities = HashSet::new();
                    for &path_idx in &scored_path.path {
                        let previous = graph.triples[path_idx];
                        visited_entities.insert(previous.subject_id);
                        visited_entities.insert(previous.object_id);
                    }
                    let introduces_new_entity = !visited_entities.contains(&triple.subject_id)
                        || !visited_entities.contains(&triple.object_id);
                    if !introduces_new_entity {
                        continue;
                    }

                    // Check adjacency
                    if triple.subject_id == frontier || triple.object_id == frontier {
                        let mut new_path = scored_path.path.clone();
                        new_path.push(idx);

                        let path_triples: Vec<Triple> = new_path
                            .iter()
                            .map(|&i| graph.triples[i])
                            .collect();

                        let energy = compute_energy(
                            &path_triples,
                            query_vector,
                            energy_config,
                            &graph.entities,
                            &graph.relations,
                            codebook,
                        );

                        candidates.push(ScoredPath {
                            path: new_path,
                            energy,
                        });
                    }
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Sort by energy and prune to beam width
        candidates.sort_by(|a, b| {
            b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(search_config.beam_width);

        // Save extended paths to all_paths
        all_paths.extend(candidates.iter().cloned());

        // The active beam becomes the extended candidates only
        beam = candidates;
    }

    // Final sort: all paths by energy
    all_paths.sort_by(|a, b| b.energy.partial_cmp(&a.energy).unwrap_or(std::cmp::Ordering::Equal));
    all_paths.dedup_by(|a, b| a.path == b.path);
    all_paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_basic() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "has", "short_wavelength");
        kg.add_triple("short_wavelength", "scatters", "in_atmosphere");

        let mut codebook = Codebook::new(2048, 42);
        let query_vec = codebook.get_or_insert("sky").clone();

        let energy_config = EnergyConfig::default();
        let search_config = SearchConfig {
            beam_width: 8,
            max_hops: 3,
            energy_threshold: 0.0,
        };

        let sky_id = kg.entity_id("sky").unwrap();
        let results = beam_search(
            &kg,
            &[sky_id],
            &query_vec,
            &mut codebook,
            &energy_config,
            &search_config,
        );

        assert!(!results.is_empty());
        // Best path should have at least 1 triple
        assert!(!results[0].path.is_empty());
        // Energy should be finite
        assert!(results[0].energy.is_finite());
    }

    #[test]
    fn test_beam_search_multi_hop() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("a", "to", "b");
        kg.add_triple("b", "to", "c");
        kg.add_triple("c", "to", "d");

        let mut codebook = Codebook::new(2048, 42);
        let query_vec = codebook.get_or_insert("a").clone();

        let energy_config = EnergyConfig::default();
        let search_config = SearchConfig {
            beam_width: 16,
            max_hops: 4,
            energy_threshold: 0.0,
        };

        let a_id = kg.entity_id("a").unwrap();
        let results = beam_search(
            &kg,
            &[a_id],
            &query_vec,
            &mut codebook,
            &energy_config,
            &search_config,
        );

        assert!(!results.is_empty());
        // Should find multi-hop paths
        let max_path_len = results.iter().map(|r| r.path.len()).max().unwrap();
        assert!(max_path_len >= 2, "Should find multi-hop paths, got max len {}", max_path_len);
    }

    #[test]
    fn test_beam_search_empty_graph() {
        let kg = KnowledgeGraph::new();
        let mut codebook = Codebook::new(2048, 42);
        let query_vec = HyperVector::zeros(2048);
        let energy_config = EnergyConfig::default();
        let search_config = SearchConfig::default();

        let results = beam_search(
            &kg,
            &[],
            &query_vec,
            &mut codebook,
            &energy_config,
            &search_config,
        );

        assert!(results.is_empty());
    }

    #[test]
    fn test_scored_paths_sorted_by_energy() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("sky", "contains", "clouds");
        kg.add_triple("blue", "wavelength", "short");

        let mut codebook = Codebook::new(2048, 42);
        let query_vec = codebook.get_or_insert("sky").clone();

        let energy_config = EnergyConfig::default();
        let search_config = SearchConfig {
            beam_width: 16,
            max_hops: 3,
            energy_threshold: 0.0,
        };

        let sky_id = kg.entity_id("sky").unwrap();
        let results = beam_search(
            &kg,
            &[sky_id],
            &query_vec,
            &mut codebook,
            &energy_config,
            &search_config,
        );

        // Verify sorted in descending energy order
        for i in 1..results.len() {
            assert!(
                results[i - 1].energy >= results[i].energy,
                "Results should be sorted by descending energy"
            );
        }
    }

    #[test]
    fn test_recursive_composition_reaches_ten_hops() {
        let mut kg = KnowledgeGraph::new();
        for i in 0..10 {
            kg.add_triple(&format!("n{}", i), "leads_to", &format!("n{}", i + 1));
        }
        let mut codebook = Codebook::new(2048, 42);
        let query_vec = codebook.get_or_insert("n0").clone();
        let results = beam_search(
            &kg,
            &[kg.entity_id("n0").unwrap()],
            &query_vec,
            &mut codebook,
            &EnergyConfig::default(),
            &SearchConfig { beam_width: 32, max_hops: 10, energy_threshold: 0.0 },
        );
        assert_eq!(results.iter().map(|path| path.path.len()).max(), Some(10));
    }

    #[test]
    fn test_recursive_composition_rejects_cycles() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("a", "to", "b");
        kg.add_triple("b", "to", "a");
        kg.add_triple("b", "to", "c");
        let mut codebook = Codebook::new(2048, 42);
        let query_vec = codebook.get_or_insert("a").clone();
        let results = beam_search(
            &kg,
            &[kg.entity_id("a").unwrap()],
            &query_vec,
            &mut codebook,
            &EnergyConfig::default(),
            &SearchConfig { beam_width: 16, max_hops: 10, energy_threshold: 0.0 },
        );
        assert!(results.iter().all(|path| path.path.len() <= 2));
    }
}
