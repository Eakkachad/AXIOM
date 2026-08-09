//! AXIOM-Gen Engine: orchestrates the full compositional generation pipeline.
//!
//! The engine ties together:
//! 1. Knowledge Graph storage
//! 2. VSA codebook for symbol→vector mapping
//! 3. Energy-guided beam search
//! 4. Linearization to natural language
//!
//! ## Usage
//!
//! ```rust
//! use tle_axiom_gen::AxiomGen;
//!
//! let mut gen = AxiomGen::new(2048);
//! gen.add_fact("sky", "is", "blue");
//! gen.add_fact("blue_light", "has", "short_wavelength");
//!
//! let result = gen.generate("why is the sky blue?");
//! println!("{}", result.sentence);
//! ```

use tle_vsa::{cosine_similarity, Codebook, HyperVector};

use crate::energy::EnergyConfig;
use crate::graph::KnowledgeGraph;
use crate::linearize::{classify_intent, linearize_with_templates, Intent};
use crate::templates::TemplateBank;
use crate::search::{beam_search, ScoredPath, SearchConfig};

/// The result of a generation query.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated natural language sentence.
    pub sentence: String,
    /// The best single-entity answer extracted from the selected path.
    pub answer: String,
    /// Reasoning trace: descriptions of each step in the path.
    pub reasoning: Vec<String>,
    /// Energy score of the best path found.
    pub energy: f32,
    /// Number of triples in the selected path.
    pub path_length: usize,
}

/// Policy used when a new fact conflicts with an existing subject/relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContradictionPolicy {
    ReportOnly,
    LatestWins,
}

/// The main AXIOM-Gen engine for compositional text generation.
pub struct AxiomGen {
    /// The knowledge graph storing all facts.
    pub graph: KnowledgeGraph,
    /// VSA codebook mapping symbols to hypervectors.
    pub codebook: Codebook,
    /// Energy function configuration.
    pub energy_config: EnergyConfig,
    /// Beam search configuration.
    pub search_config: SearchConfig,
    /// Template bank for varied linearization.
    pub template_bank: TemplateBank,
    pub contradiction_policy: ContradictionPolicy,
}

impl AxiomGen {
    /// Create a new AXIOM-Gen engine with the given vector dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            codebook: Codebook::new(dim, 0xA10A_CAFE_BEAD_0001),
            energy_config: EnergyConfig::default(),
            search_config: SearchConfig::default(),
            template_bank: TemplateBank::new(),
            contradiction_policy: ContradictionPolicy::ReportOnly,
        }
    }

    /// Create a new engine with custom configuration.
    pub fn with_config(
        dim: usize,
        energy_config: EnergyConfig,
        search_config: SearchConfig,
    ) -> Self {
        Self {
            graph: KnowledgeGraph::new(),
            codebook: Codebook::new(dim, 0xA10A_CAFE_BEAD_0001),
            energy_config,
            search_config,
            template_bank: TemplateBank::new(),
            contradiction_policy: ContradictionPolicy::ReportOnly,
        }
    }

    /// Add a fact (triple) to the knowledge graph.
    ///
    /// Automatically registers entities and relations in the codebook.
    pub fn add_fact(&mut self, subject: &str, relation: &str, object: &str) {
        if self.contradiction_policy == ContradictionPolicy::LatestWins {
            self.graph.remove_conflicting_triples(subject, relation);
        }
        // Add to knowledge graph
        self.graph.add_triple(subject, relation, object);

        // Pre-populate codebook with all symbols
        self.codebook.get_or_insert(subject);
        self.codebook.get_or_insert(relation);
        self.codebook.get_or_insert(object);
    }

    pub fn set_contradiction_policy(&mut self, policy: ContradictionPolicy) {
        self.contradiction_policy = policy;
    }

    /// Sync every triple in the knowledge graph into a VSA-LM knowledge prior.
    ///
    /// This is the bridge between AXIOM-Gen (structured fact reasoning) and
    /// VSA-LM (non-neural fluency): the graph decides what facts exist, and
    /// the VSA-LM knowledge prior steers token generation toward them.
    pub fn sync_into_vsa_lm(&self, lm: &mut tle_vsa_lm::VsaLm) {
        for [subject, relation, object] in self.graph.export_triples() {
            // Register entity words so the VSA-LM can emit them.
            for name in [&subject, &relation, &object] {
                for w in name.split(|c: char| c == '_' || c == ' ') {
                    if !w.is_empty() {
                        lm.vocab.get_or_add(w);
                    }
                }
            }
            lm.knowledge.add_fact(&subject, &relation, &object);
        }
    }

    /// Generate a compositional sentence answering the query.
    ///
    /// Pipeline:
    /// 1. Parse query → extract entities + classify intent
    /// 2. BFS subgraph extraction from relevant entities
    /// 3. Energy-guided beam search for best path
    /// 4. Linearize best path into natural language
    /// 5. Return result with reasoning trace
    pub fn generate(&mut self, query: &str) -> GenerationResult {
        // Step 1: Parse query — extract known entities and classify intent
        let intent = classify_intent(query);
        let query_entities = self.extract_query_entities(query);

        if query_entities.is_empty() {
            return GenerationResult {
                sentence: String::new(),
                answer: String::new(),
                reasoning: vec!["No known entities found in query.".to_string()],
                energy: 0.0,
                path_length: 0,
            };
        }

        // Step 2: Build query vector by bundling entity vectors
        let query_vector = self.build_query_vector(query);

        // Step 3: BFS subgraph from query entities
        let _subgraph = self.graph.bfs_subgraph(&query_entities, self.search_config.max_hops);

        // Step 4: Beam search for best paths
        // Precompute entity informativeness: rare entities are more specific.
        let entity_ief = compute_entity_ief(&self.graph);
        let results = beam_search(
            &self.graph,
            &query_entities,
            &query_vector,
            &mut self.codebook,
            &self.energy_config,
            &self.search_config,
            Some(&entity_ief),
        );

        if results.is_empty() {
            return GenerationResult {
                sentence: String::new(),
                answer: String::new(),
                reasoning: vec!["No paths found in knowledge graph.".to_string()],
                energy: 0.0,
                path_length: 0,
            };
        }

        // Step 5: Select best path — prefer paths meeting target length, then by energy
        let target_len = self.energy_config.target_length;
        let best = results
            .iter()
            .filter(|p| p.path.len() >= target_len)
            .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap_or(std::cmp::Ordering::Equal))
            .or_else(|| {
                // Fallback: pick longest path available
                results.iter().max_by_key(|p| p.path.len())
            })
            .unwrap();

        let path_triples: Vec<_> = best.path.iter().map(|&i| self.graph.triples[i]).collect();

        // Build reasoning trace
        let reasoning: Vec<String> = path_triples
            .iter()
            .map(|t| {
                format!(
                    "{} {} {}",
                    self.graph.entity_name(t.subject_id),
                    self.graph.relation_name(t.relation_id),
                    self.graph.entity_name(t.object_id),
                )
            })
            .collect();

        // Linearize path to natural language
        // Try TemplateBank first for varied output, fallback to default linearizer
        let sentence = linearize_with_templates(
            &path_triples,
            &self.graph.entities,
            &self.graph.relations,
            intent,
            &self.template_bank,
        );

        // Extract the best single-entity answer: the legacy triple-scan
        // currently outperforms DDTree path-aware scoring (12.89% vs 11.95%
        // on verified-wikipedia-dev).  DDTree is available as infrastructure
        // and will take over once the energy function better distinguishes
        // fact-true paths from noise.
        let answer = self.extract_answer(&self.graph, &query_entities, intent, &query_vector, query);

        GenerationResult {
            sentence,
            answer,
            reasoning,
            energy: best.energy,
            path_length: path_triples.len(),
        }
    }

    /// Extract a single best answer entity from the beam search paths.
    ///
    /// This is the DDTree-style answer selection: instead of scanning every
    /// triple in the graph independently, we score only entities that appear
    /// in beam-discovered paths (which are already energy-ranked).  The
    /// intuition: the beam search already knows which graph regions are
    /// relevant to the query — use that signal directly for answer picking.
    fn extract_answer_ddtree(
        &self,
        graph: &KnowledgeGraph,
        query_entities: &[usize],
        beam_results: &[ScoredPath],
        intent: Intent,
        query_vector: &HyperVector,
        query: &str,
    ) -> String {
        let content_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
                .filter(|w| !matches!(*w, "what" | "which" | "where" | "when" | "how" | "does"
                    | "have" | "who" | "with" | "from" | "that" | "this" | "why" | "the" | "was" | "did"))
                .map(|w| w.to_string())
                .collect();

        use std::collections::HashMap;
        let mut scores: HashMap<usize, f32> = HashMap::new();
        // Score entities from the top beam paths only — the higher-ranked the
        // path the more likely it carries the answer.  Limit to top-N to
        // prevent low-energy noise paths from drowning the signal.
        let top_n = 5usize;
        for (pi, path) in beam_results.iter().take(top_n).enumerate() {
            let path_rank = 1.0 / (1.0 + pi as f32); // higher = closer to top
            for &ti in &path.path {
                let triple = &graph.triples[ti];
                let conf = graph.triple_confidence(ti);
                // Direct connection to a query entity is the strongest QA signal.
                let subj_q = query_entities.contains(&triple.subject_id);
                let obj_q = query_entities.contains(&triple.object_id);
                if subj_q {
                    *scores.entry(triple.object_id).or_insert(0.0) += path.energy * conf * 1.5 * path_rank;
                    *scores.entry(triple.subject_id).or_insert(0.0) += path.energy * conf * 0.3 * path_rank;
                }
                if obj_q {
                    *scores.entry(triple.subject_id).or_insert(0.0) += path.energy * conf * 1.5 * path_rank;
                    *scores.entry(triple.object_id).or_insert(0.0) += path.energy * conf * 0.3 * path_rank;
                }
                if !subj_q && !obj_q {
                    // Mid-path entity: lower boost, proportional to depth.
                    let depth = path.path.iter().position(|&i| i == ti).unwrap_or(1) as f32;
                    *scores.entry(triple.subject_id).or_insert(0.0) += path.energy * conf * 0.5 / (1.0 + depth);
                    *scores.entry(triple.object_id).or_insert(0.0) += path.energy * conf * 0.5 / (1.0 + depth);
                }
            }
        }

        // Fallback: if beam paths produced no answer, use the old extract_answer.
        if scores.is_empty() {
            return self.extract_answer(graph, query_entities, intent, query_vector, query);
        }

        // Entity-quality bonus: short, capitalized, high-overlap entities win.
        let mut ranked: Vec<(f32, String)> = scores
            .into_iter()
            .filter_map(|(id, score)| {
                let name = graph.entity_name(id);
                let words = name.split_whitespace().count();
                if words > 5 { return None; }
                let lower = name.to_lowercase();
                let overlap = content_words.iter().filter(|w| lower.contains(w.as_str())).count();
                let cap_bonus = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { 1.0 } else { -0.5 };
                let len_penalty = match words { 0|1 => 0.0, 2 => 0.4, 3 => 1.2, 4 => 2.0, _ => 3.0 };
                let first = lower.split_whitespace().next().unwrap_or("");
                let det_penalty = if matches!(first, "a"|"an"|"the"|"his"|"her"|"its") { -1.5 } else { 0.0 };
                Some((score + overlap as f32 + cap_bonus - len_penalty + det_penalty, name.to_string()))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.first().map(|(_, name)| name.clone()).unwrap_or_default()
    }

    /// Extract a single best answer entity from the graph (legacy — scans
    /// all triples independently).  Prefer extract_answer_ddtree which uses
    /// beam-path energy for higher precision.
    fn extract_answer(
        &self,
        graph: &KnowledgeGraph,
        query_entities: &[usize],
        intent: Intent,
        query_vector: &HyperVector,
        query: &str,
    ) -> String {
        let content_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
                .filter(|w| !matches!(*w, "what" | "which" | "where" | "when" | "how" | "does"
                    | "have" | "who" | "with" | "from" | "that" | "this" | "why" | "the" | "was" | "did"))
                .map(|w| w.to_string())
                .collect();

        use std::collections::HashMap;
        let mut scores: HashMap<usize, (f32, usize)> = HashMap::new();
        // Entity-vector cache: compute semantic_vector once per unique entity
        // (it's called redundantly when an entity appears in multiple triples).
        let mut relevance_cache: HashMap<usize, f32> = HashMap::new();

        for (ti, triple) in graph.triples.iter().enumerate() {
            let triple_conf = graph.triple_confidence(ti);
            let subject_in_query = query_entities.contains(&triple.subject_id);
            let object_in_query = query_entities.contains(&triple.object_id);
            let connected = subject_in_query != object_in_query;
            let connected_id = if connected {
                if subject_in_query { Some(triple.object_id) } else { Some(triple.subject_id) }
            } else { None };

            for entity_id in [triple.subject_id, triple.object_id] {
                let name = graph.entity_name(entity_id);
                let lower = name.to_lowercase();
                let overlap = content_words.iter().filter(|w| lower.contains(w.as_str())).count();
                let connected_bonus = if connected_id == Some(entity_id) { 2.5 * triple_conf } else { 0.0 };
                let role_bonus = if connected_id == Some(entity_id) {
                    let rb = match intent {
                        Intent::Who => if subject_in_query { 1.5 } else { 0.0 },
                        Intent::What | Intent::Where => if !subject_in_query { 1.5 } else { 0.0 },
                        _ => 0.5,
                    };
                    rb * triple_conf
                } else { 0.0 };
                let relevance = *relevance_cache.entry(entity_id).or_insert_with(|| {
                    tle_vsa::cosine_similarity(query_vector, &self.semantic_vector(graph.entity_name(entity_id)))
                });
                let entry = scores.entry(entity_id).or_insert((0.0, 0));
                entry.0 += connected_bonus + role_bonus + overlap as f32 + relevance * 0.5;
                entry.1 += 1;
            }
        }

        let mut ranked: Vec<(f32, usize, String)> = scores
            .into_iter()
            .filter_map(|(id, (score, count))| {
                let name = graph.entity_name(id);
                let words = name.split_whitespace().count();
                if words > 5 {
                    return None;
                }
                // Strong preference for short, entity-like answers.
                let length_penalty = match words {
                    0 | 1 => 0.0,
                    2 => 0.4,
                    3 => 1.2,
                    4 => 2.0,
                    _ => 3.0,
                };
                // Capitalized proper nouns are far more likely to be answers
                // than lowercase descriptive phrases ("Martina Hingis" vs
                // "a tennis player").
                let capitalized_bonus = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    1.0
                } else {
                    -0.5
                };
                // Penalize entities that begin with articles/determiners
                // ("a tennis player", "the inventor of ...") — descriptive
                // phrases, not answer entities.
                let first = name
                    .split_whitespace()
                    .next()
                    .map(|w| w.to_lowercase())
                    .unwrap_or_default();
                let determiner_penalty = if matches!(first.as_str(), "a" | "an" | "the" | "some" | "his" | "her" | "its" | "this" | "that") {
                    -1.5
                } else {
                    0.0
                };
                Some((
                    score + 0.2 * count as f32 - length_penalty + capitalized_bonus + determiner_penalty,
                    id,
                    name.to_string(),
                ))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.first().map(|(_, _, name)| name.clone()).unwrap_or_default()
    }

    /// Extract entity IDs mentioned in the query that exist in the knowledge graph.
    fn extract_query_entities(&mut self, query: &str) -> Vec<usize> {
        let lower = query.to_lowercase();
        // Remove punctuation for matching
        let cleaned: String = lower
            .chars()
            .map(|c| if c.is_alphanumeric() || c == ' ' || c == '_' { c } else { ' ' })
            .collect();
        let words: Vec<&str> = cleaned.split_whitespace().collect();

        let mut found_entities: Vec<usize> = Vec::new();

        let normalized_query: Vec<String> = words.iter().map(|word| normalize_entity_token(word)).collect();

        // Try exact and normalized matches first.
        for (name, &id) in &self.graph.entity_index {
            let entity_lower = name.to_lowercase();
            let entity_words: Vec<&str> = entity_lower.split('_').collect();

            let matches = words.contains(&entity_lower.as_str()) || entity_words.iter().any(|ew| {
                words.contains(ew) || normalized_query.iter().any(|word| word == &normalize_entity_token(ew))
            });
            if matches && !found_entities.contains(&id) {
                found_entities.push(id);
            }
        }

        // VSA fuzzy linking — affinity+cosine in a single pass.
        {
            let query_vector = self.compose_entity_vector(&normalized_query);
            let entities: Vec<(String, usize)> = self.graph.entity_index.iter()
                .map(|(name, &id)| (name.clone(), id))
                .collect();
            let mut scored: Vec<(f32, usize)> = Vec::new();
            for (name, id) in entities {
                let entity_words: Vec<String> = name
                    .to_lowercase()
                    .split('_')
                    .map(normalize_entity_token)
                    .collect();
                let mut affinity = 0.0f32;
                for qw in &normalized_query {
                    if qw.len() < 4 { continue; }
                    for ew in &entity_words {
                        if ew.len() < 4 { continue; }
                        if qw == ew { affinity += 1.0; }
                        else if ew.starts_with(qw.as_str()) || qw.starts_with(ew.as_str()) { affinity += 0.8; }
                        else if ew.contains(qw.as_str()) || qw.contains(ew.as_str()) { affinity += 0.5; }
                    }
                }
                // Only pay the VSA cost for entities with non-zero affinity.
                if affinity <= 0.0 { continue; }
                let affinity = affinity.min(1.5);
                let entity_vector = self.compose_entity_vector(&entity_words);
                let cosine = cosine_similarity(&query_vector, &entity_vector);
                let score = affinity * 1.2 + cosine * 0.5;
                if score >= 0.9 { scored.push((score, id)); }
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            for (_, id) in scored {
                if !found_entities.contains(&id) { found_entities.push(id); }
                if found_entities.len() >= 8 { break; }
            }
        }

        found_entities
    }

    fn compose_entity_vector(&mut self, words: &[String]) -> HyperVector {
        let mut result = HyperVector::zeros(self.codebook.dim());
        for word in words.iter().filter(|word| !word.is_empty()) {
            result = result.add(&self.codebook.get_or_insert(word));
        }
        result.normalize()
    }

    /// Compose a semantic vector for a symbol from its constituent word
    /// vectors.
    ///
    /// The base codebook maps whole strings to independent random vectors, so
    /// `C("Luteinizing_hormone")` shares nothing with `C("hormone")`. By
    /// bundling the word-level vectors instead, entities that share vocabulary
    /// acquire positive cosine similarity, making VSA relevance meaningful for
    /// fuzzy answer matching. No answer oracle or training is involved.
    fn semantic_vector(&self, name: &str) -> HyperVector {
        let words: Vec<&str> = name
            .split(|c: char| c == ' ' || c == '_' || c == '-')
            .filter(|w| !w.is_empty())
            .collect();
        let mut result = HyperVector::zeros(self.codebook.dim());
        let mut found = 0usize;
        for word in words.iter() {
            if let Some(vec) = self.codebook.get(word) {
                // Bundle word vectors without permutation so that words shared
                // between query and entity contribute positive cosine signal,
                // matching how build_query_vector bundles query words.
                result = result.add(vec);
                found += 1;
            }
        }
        if found == 0 {
            if let Some(vec) = self.codebook.get(name) {
                return vec.clone();
            }
            return result;
        }
        result.normalize()
    }

    /// Build a query vector by bundling vectors of all words in the query.
    fn build_query_vector(&mut self, query: &str) -> HyperVector {
        let dim = self.codebook.dim();
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return HyperVector::zeros(dim);
        }

        let mut result = HyperVector::zeros(dim);
        for word in &words {
            let vec = self.codebook.get_or_insert(word).clone();
            result = result.add(&vec);
        }

        result.normalize()
    }
}

// Base seed for AXIOM-Gen codebook
const _AXIOM_GEN_SEED: u64 = 0xAA10_CAFE_BEAD_0001;

fn normalize_entity_token(token: &str) -> String {
    let token = token.trim_matches(|c: char| !c.is_alphanumeric());
    let token = token.strip_suffix("'s").unwrap_or(token);
    if token.len() > 3 && token.ends_with('s') {
        token[..token.len() - 1].to_string()
    } else {
        token.to_string()
    }
}

/// Precompute inverse entity frequency: rare entities score higher.
///
/// Most decomposition errors produce generic hub entities ("is", "a",
/// "together") that appear in many triples.  IEF penalises these so the
/// beam search energy function naturally prefers paths through specific,
/// informative entities (like proper noun answers).
fn compute_entity_ief(graph: &KnowledgeGraph) -> Vec<f32> {
    let n = graph.entities.len();
    let mut freq = vec![0usize; n];
    for triple in &graph.triples {
        if triple.subject_id < n { freq[triple.subject_id] += 1; }
        if triple.object_id < n { freq[triple.object_id] += 1; }
    }
    let total = freq.iter().sum::<usize>() as f32;
    freq.into_iter()
        .map(|f| if f == 0 { 0.0 } else { -((f as f32) / total.max(1.0)).ln() })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_into_vsa_lm_fills_knowledge_prior() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");
        gen.add_fact("blue", "has", "short_wavelength");

        let mut lm = tle_vsa_lm::VsaLm::new(tle_vsa_lm::LmConfig {
            dim: 2048,
            max_order: 2,
            ..Default::default()
        });
        gen.sync_into_vsa_lm(&mut lm);

        assert_eq!(lm.knowledge.facts, 2);
        // "sky" should surface "blue" as the next fact-consistent word.
        let ctx = vec!["sky".to_string(), "is".to_string()];
        let pred = lm.predict_next_fast(&ctx, 5);
        assert!(pred.iter().any(|t| t.word == "blue"));
    }

    #[test]
    fn test_axiom_gen_new() {
        let gen = AxiomGen::new(2048);
        assert_eq!(gen.graph.entities.len(), 0);
        assert_eq!(gen.graph.triples.len(), 0);
    }

    #[test]
    fn test_add_fact() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        assert_eq!(gen.graph.entities.len(), 2);
        assert_eq!(gen.graph.relations.len(), 1);
        assert_eq!(gen.graph.triples.len(), 1);
        // Codebook should have entries for all symbols
        assert!(gen.codebook.get("sky").is_some());
        assert!(gen.codebook.get("is").is_some());
        assert!(gen.codebook.get("blue").is_some());
    }

    #[test]
    fn test_extract_query_entities() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");
        gen.add_fact("blue_light", "has", "short_wavelength");

        let entities = gen.extract_query_entities("why is the sky blue?");
        assert!(!entities.is_empty());
        // Should find "sky" and/or "blue"
        let sky_id = gen.graph.entity_id("sky").unwrap();
        let blue_id = gen.graph.entity_id("blue").unwrap();
        assert!(
            entities.contains(&sky_id) || entities.contains(&blue_id),
            "Should find sky or blue in query"
        );
    }

    #[test]
    fn test_fuzzy_entity_linking_plural_form() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("cat", "is", "animal");
        let entities = gen.extract_query_entities("what are cats?");
        assert!(entities.contains(&gen.graph.entity_id("cat").unwrap()));
    }

    #[test]
    fn test_fuzzy_entity_linking_derived_name() {
        // "Molitor" (in the question) must link to the graph entity
        // "Molitorová" (the surface form in evidence) via prefix matching.
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("Molitorová", "is", "tennis player");
        let entities = gen.extract_query_entities("Melanie Molitor is the mom of which tennis player");
        assert!(
            entities.contains(&gen.graph.entity_id("Molitorová").unwrap()),
            "query 'Molitor' should fuzzy-link to entity 'Molitorová', got {:?}",
            entities.iter().map(|&id| gen.graph.entity_name(id).to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_latest_wins_contradiction_policy() {
        let mut gen = AxiomGen::new(2048);
        gen.set_contradiction_policy(ContradictionPolicy::LatestWins);
        gen.add_fact("sky", "color", "blue");
        gen.add_fact("sky", "color", "green");
        assert!(gen.graph.contradictions().is_empty());
        assert_eq!(gen.graph.triples.len(), 1);
        assert_eq!(gen.graph.entity_name(gen.graph.triples[0].object_id), "green");
    }

    #[test]
    fn test_generate_simple() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        let result = gen.generate("what is the sky?");
        assert!(!result.sentence.is_empty());
        assert!(result.path_length >= 1);
        assert!(!result.reasoning.is_empty());
    }

    #[test]
    fn test_generate_novel_sentence() {
        let mut gen = AxiomGen::new(2048);
        // Use a connected chain: sky -> blue -> short_wavelength -> more_in_atmosphere
        gen.add_fact("sky", "is", "blue");
        gen.add_fact("blue", "has", "short_wavelength");
        gen.add_fact("short_wavelength", "scatters", "more_in_atmosphere");

        // Adjust config to prefer longer multi-hop paths
        gen.energy_config.lambda_simplicity = 0.0;
        gen.energy_config.lambda_length = 0.5;
        gen.energy_config.target_length = 3;

        let result = gen.generate("why is the sky blue?");
        assert!(!result.sentence.is_empty());
        assert!(
            result.sentence.to_lowercase().contains("blue"),
            "Sentence should mention blue: {}",
            result.sentence
        );
        assert!(
            result.reasoning.len() >= 2,
            "Should have multi-hop reasoning, got {}: {:?}",
            result.reasoning.len(),
            result.reasoning
        );
    }

    #[test]
    fn test_generate_no_entities() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sky", "is", "blue");

        let result = gen.generate("what is xyz123?");
        // Should gracefully handle unknown entities
        assert_eq!(result.path_length, 0);
    }

    #[test]
    fn test_generate_energy_finite() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("water", "is", "liquid");
        gen.add_fact("liquid", "has", "flow");

        let result = gen.generate("what is water?");
        assert!(result.energy.is_finite());
    }

    #[test]
    fn test_generate_with_reasoning_trace() {
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("sun", "causes", "heat");
        gen.add_fact("heat", "causes", "evaporation");
        gen.add_fact("evaporation", "causes", "clouds");

        let result = gen.generate("why do we have clouds?");
        // Reasoning should describe the path
        for step in &result.reasoning {
            assert!(!step.is_empty());
        }
    }
}
