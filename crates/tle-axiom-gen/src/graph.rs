//! Knowledge Graph: stores entities, relations, and triples for AXIOM-Gen traversal.
//!
//! The graph uses integer IDs for efficient traversal and lookup.
//! Entity/relation names are stored separately and mapped via indices.

use std::collections::{HashMap, VecDeque};

/// A triple (subject, relation, object) in the knowledge graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Triple {
    pub subject_id: usize,
    pub relation_id: usize,
    pub object_id: usize,
}

/// A pair of facts that disagree on the object for the same subject/relation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Contradiction {
    pub subject: String,
    pub relation: String,
    pub first_object: String,
    pub second_object: String,
}

/// A knowledge graph storing entities, relations, and their connections as triples.
#[derive(Debug, Clone)]
pub struct KnowledgeGraph {
    /// All entity names, indexed by ID.
    pub entities: Vec<String>,
    /// All relation names, indexed by ID.
    pub relations: Vec<String>,
    /// All triples in the graph.
    pub triples: Vec<Triple>,
    /// Reverse mapping from entity name to ID.
    pub entity_index: HashMap<String, usize>,
    /// Reverse mapping from relation name to ID.
    relation_index: HashMap<String, usize>,
    /// Adjacency: entity ID → indices of triples it participates in.
    /// Enables O(degree) neighbor expansion instead of O(all triples) scans.
    pub adjacency: Vec<Vec<usize>>,
}

impl KnowledgeGraph {
    /// Create a new empty knowledge graph.
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            triples: Vec::new(),
            entity_index: HashMap::new(),
            relation_index: HashMap::new(),
            adjacency: Vec::new(),
        }
    }

    /// Add an entity to the graph, returning its ID.
    /// If the entity already exists, returns the existing ID.
    pub fn add_entity(&mut self, name: &str) -> usize {
        if let Some(&id) = self.entity_index.get(name) {
            return id;
        }
        let id = self.entities.len();
        self.entities.push(name.to_string());
        self.entity_index.insert(name.to_string(), id);
        id
    }

    /// Add a relation to the graph, returning its ID.
    /// If the relation already exists, returns the existing ID.
    pub fn add_relation(&mut self, name: &str) -> usize {
        if let Some(&id) = self.relation_index.get(name) {
            return id;
        }
        let id = self.relations.len();
        self.relations.push(name.to_string());
        self.relation_index.insert(name.to_string(), id);
        id
    }

    /// Add a triple to the graph by entity/relation names.
    /// Automatically adds entities and relations if they don't exist.
    /// Returns the index of the added triple.
    pub fn add_triple(&mut self, subject: &str, relation: &str, object: &str) -> usize {
        let subj_id = self.add_entity(subject);
        let rel_id = self.add_relation(relation);
        let obj_id = self.add_entity(object);
        let triple = Triple {
            subject_id: subj_id,
            relation_id: rel_id,
            object_id: obj_id,
        };
        let idx = self.triples.len();
        self.triples.push(triple);
        // Maintain adjacency lists. Ensure vectors are sized to the new
        // entity count.
        while self.adjacency.len() <= subj_id.max(obj_id) {
            self.adjacency.push(Vec::new());
        }
        self.adjacency[subj_id].push(idx);
        self.adjacency[obj_id].push(idx);
        idx
    }

    /// Get the indices of all triples an entity participates in (as subject
    /// or object). O(degree), the fast adjacency path for beam expansion.
    pub fn adjacency_of(&self, entity_id: usize) -> &[usize] {
        self.adjacency
            .get(entity_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all triples where the given entity is the subject.
    pub fn get_triples_from(&self, entity_id: usize) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.subject_id == entity_id)
            .collect()
    }

    /// Export all triples as (subject, relation, object) name triples.
    ///
    /// This is the bridge to downstream consumers (e.g. the VSA-LM knowledge
    /// prior) that want the graph facts as plain strings.
    pub fn export_triples(&self) -> Vec<[String; 3]> {
        self.triples
            .iter()
            .map(|t| {
                [
                    self.entity_name(t.subject_id).to_string(),
                    self.relation_name(t.relation_id).to_string(),
                    self.entity_name(t.object_id).to_string(),
                ]
            })
            .collect()
    }

    /// Get all triples where the given entity is the object.
    pub fn get_triples_to(&self, entity_id: usize) -> Vec<&Triple> {
        self.triples
            .iter()
            .filter(|t| t.object_id == entity_id)
            .collect()
    }

    /// Get the ID of an entity by name, if it exists.
    pub fn entity_id(&self, name: &str) -> Option<usize> {
        self.entity_index.get(name).copied()
    }

    /// Get the name of an entity by ID.
    pub fn entity_name(&self, id: usize) -> &str {
        &self.entities[id]
    }

    /// Get the name of a relation by ID.
    pub fn relation_name(&self, id: usize) -> &str {
        &self.relations[id]
    }

    /// Heuristic confidence score for a triple (EGA-style gate proxy).
    ///
    /// Low-confidence triples have garbage subjects ("Together they") or
    /// verbose objects ("a Swiss professional tennis player who won the
    /// Australian Open in 1997").  Answer selection weights these scores so
    /// the beam's energy function implicitly favours reliable facts.
    pub fn triple_confidence(&self, triple_idx: usize) -> f32 {
        let t = &self.triples[triple_idx];
        let subj_words = self.entity_name(t.subject_id).split_whitespace().count();
        let obj_words = self.entity_name(t.object_id).split_whitespace().count();
        // Short entities are more reliable (long phrases are decomposition
        // noise).  Use a sigmoid-like ramp: 1.0 for ≤2 words, decaying to 0.
        let len_score = (0.8f32).powi(subj_words.saturating_sub(2) as i32)
            .min((0.8f32).powi(obj_words.saturating_sub(1) as i32));
        // Penalise relations that are bare copulas — they match too easily.
        let rel_name = self.relation_name(t.relation_id);
        let rel_penalty = if rel_name == "is" || rel_name == "are" || rel_name == "was" || rel_name == "were" {
            0.6
        } else {
            1.0
        };
        len_score * rel_penalty
    }

    /// BFS subgraph extraction: starting from a set of entities,
    /// explore up to `max_hops` hops and return all reachable triples.
    pub fn bfs_subgraph(&self, start_entities: &[usize], max_hops: usize) -> Vec<Triple> {
        let mut visited = vec![false; self.entities.len()];
        let mut result_triples = Vec::new();
        let mut seen_triples = vec![false; self.triples.len()];
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        for &entity_id in start_entities {
            if entity_id < self.entities.len() {
                visited[entity_id] = true;
                queue.push_back((entity_id, 0));
            }
        }

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            // Explore triples adjacent to the current entity via the adjacency
            // index (O(degree)) instead of scanning all triples.
            for &idx in self.adjacency_of(current) {
                if seen_triples[idx] {
                    continue;
                }
                let triple = &self.triples[idx];
                seen_triples[idx] = true;
                result_triples.push(*triple);
                if !visited[triple.object_id] {
                    visited[triple.object_id] = true;
                    queue.push_back((triple.object_id, depth + 1));
                }
                if !visited[triple.subject_id] {
                    visited[triple.subject_id] = true;
                    queue.push_back((triple.subject_id, depth + 1));
                }
            }
        }

        result_triples
    }

    /// Find deterministic subject/relation conflicts in insertion order.
    pub fn contradictions(&self) -> Vec<Contradiction> {
        let mut seen: HashMap<(usize, usize), usize> = HashMap::new();
        let mut conflicts = Vec::new();
        for triple in &self.triples {
            let key = (triple.subject_id, triple.relation_id);
            if let Some(&first_object) = seen.get(&key) {
                if first_object != triple.object_id {
                    let conflict = Contradiction {
                        subject: self.entity_name(triple.subject_id).to_string(),
                        relation: self.relation_name(triple.relation_id).to_string(),
                        first_object: self.entity_name(first_object).to_string(),
                        second_object: self.entity_name(triple.object_id).to_string(),
                    };
                    if !conflicts.contains(&conflict) {
                        conflicts.push(conflict);
                    }
                }
            } else {
                seen.insert(key, triple.object_id);
            }
        }
        conflicts
    }

    /// Remove earlier values for a subject/relation before inserting a replacement.
    pub fn remove_conflicting_triples(&mut self, subject: &str, relation: &str) -> usize {
        let Some(&subject_id) = self.entity_index.get(subject) else { return 0; };
        let Some(&relation_id) = self.relation_index.get(relation) else { return 0; };
        let before = self.triples.len();
        self.triples.retain(|triple| {
            triple.subject_id != subject_id || triple.relation_id != relation_id
        });
        before - self.triples.len()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entities_and_relations() {
        let mut kg = KnowledgeGraph::new();
        let sky = kg.add_entity("sky");
        let blue = kg.add_entity("blue");
        assert_eq!(sky, 0);
        assert_eq!(blue, 1);
        assert_eq!(kg.entity_id("sky"), Some(0));
        assert_eq!(kg.entity_name(0), "sky");
    }

    #[test]
    fn test_add_triple() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        assert_eq!(kg.triples.len(), 1);
        assert_eq!(kg.entities.len(), 2);
        assert_eq!(kg.relations.len(), 1);
    }

    #[test]
    fn test_get_triples_from() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("sky", "has", "clouds");
        kg.add_triple("ocean", "is", "blue");

        let sky_id = kg.entity_id("sky").unwrap();
        let from_sky = kg.get_triples_from(sky_id);
        assert_eq!(from_sky.len(), 2);
    }

    #[test]
    fn test_get_triples_to() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("ocean", "is", "blue");

        let blue_id = kg.entity_id("blue").unwrap();
        let to_blue = kg.get_triples_to(blue_id);
        assert_eq!(to_blue.len(), 2);
    }

    #[test]
    fn test_duplicate_entity() {
        let mut kg = KnowledgeGraph::new();
        let id1 = kg.add_entity("sky");
        let id2 = kg.add_entity("sky");
        assert_eq!(id1, id2);
        assert_eq!(kg.entities.len(), 1);
    }

    #[test]
    fn test_contradictions_ignore_duplicates() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "color", "blue");
        kg.add_triple("sky", "color", "blue");
        kg.add_triple("sky", "color", "green");
        let conflicts = kg.contradictions();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].first_object, "blue");
        assert_eq!(conflicts[0].second_object, "green");
    }

    #[test]
    fn test_export_triples() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "has", "short_wavelength");
        let triples = kg.export_triples();
        assert_eq!(triples.len(), 2);
        assert_eq!(triples[0], ["sky", "is", "blue"]);
        assert_eq!(triples[1], ["blue", "has", "short_wavelength"]);
    }

    #[test]
    fn test_bfs_subgraph() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("sky", "is", "blue");
        kg.add_triple("blue", "relates_to", "ocean");
        kg.add_triple("ocean", "contains", "fish");
        kg.add_triple("unrelated", "is", "separate");

        let sky_id = kg.entity_id("sky").unwrap();
        let subgraph = kg.bfs_subgraph(&[sky_id], 2);
        // Should reach sky->blue->ocean but not "unrelated"
        assert!(subgraph.len() >= 2);
        // The unrelated triple should not be in the subgraph
        let unrelated_id = kg.entity_id("unrelated").unwrap();
        assert!(!subgraph.iter().any(|t| t.subject_id == unrelated_id));
    }

    #[test]
    fn test_bfs_subgraph_max_hops() {
        let mut kg = KnowledgeGraph::new();
        kg.add_triple("a", "to", "b");
        kg.add_triple("b", "to", "c");
        kg.add_triple("c", "to", "d");

        let a_id = kg.entity_id("a").unwrap();

        // With 1 hop, should only get a->b
        let sub1 = kg.bfs_subgraph(&[a_id], 1);
        assert_eq!(sub1.len(), 1);

        // With 3 hops, should get all
        let sub3 = kg.bfs_subgraph(&[a_id], 3);
        assert_eq!(sub3.len(), 3);
    }
}
