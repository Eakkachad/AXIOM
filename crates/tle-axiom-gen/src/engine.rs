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

use crate::answer_type::{matches_answer_type_oriented, predict_answer_type, AnswerType};
use crate::energy::EnergyConfig;
use crate::graph::KnowledgeGraph;
use crate::linearize::{classify_intent, linearize_with_templates, Intent};
use crate::semantic::SemanticLayer;
use crate::templates::TemplateBank;
use crate::search::{beam_search, ScoredPath, SearchConfig};

/// Read a T1.8 weight-search override from the environment, falling back to
/// the given default. Lets coordinate-ascent sweep extract_answer weights
/// without recompiling.
fn weight_env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(default)
}

/// T1.13 MDL-compression proxy: how many characters of `a` are "explained" by
/// `b`, measured by greedy shingle (length-`l` substring) coverage. This is a
/// deterministic, dependency-free stand-in for LZ77 match length (katgpt
/// `MatchLengthScorer`); longer shared runs mean `b` predicts `a` better, i.e.
/// a lower marginal compression cost of `a` given `b`.
fn shingle_cover(a: &str, b: &str, l: usize) -> usize {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    if ab.len() < l || bb.len() < l {
        return 0;
    }
    let mut set: std::collections::HashSet<&[u8]> = bb.windows(l).collect();
    let mut covered = 0usize;
    let mut i = 0;
    while i + l <= ab.len() {
        if set.contains(&ab[i..i + l]) {
            covered += l;
            i += l;
        } else {
            i += 1;
        }
    }
    covered
}

/// The result of a generation query.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub sentence: String,
    pub answer: String,
    pub reasoning: Vec<String>,
    pub energy: f32,
    pub path_length: usize,
    /// Per-entity score breakdown for top-N candidates.
    /// (final_score, name, connectivity, role, overlap, vsa, heuristics)
    pub diagnostics: Vec<(f32, String, f32, f32, f32, f32, f32)>,
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
    /// Co-occurrence semantic layer (T3.1) — enriches VSA relevance.
    pub semantic: SemanticLayer,
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
            semantic: SemanticLayer::new(),
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
            semantic: SemanticLayer::new(),
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
                diagnostics: Vec::new(),
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
                diagnostics: Vec::new(),
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

        let (answer, diag) = self.extract_answer(&self.graph, &query_entities, intent, &query_vector, query);

        GenerationResult {
            sentence,
            answer,
            reasoning,
            energy: best.energy,
            path_length: path_triples.len(),
            diagnostics: diag,
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
                    if matches!(intent, Intent::Who) {
                        *scores.entry(triple.object_id).or_insert(0.0) += 3.0 * conf * path_rank;
                    }
                }
                if obj_q {
                    *scores.entry(triple.subject_id).or_insert(0.0) += path.energy * conf * 1.5 * path_rank;
                    *scores.entry(triple.object_id).or_insert(0.0) += path.energy * conf * 0.3 * path_rank;
                    if matches!(intent, Intent::What | Intent::Where) {
                        *scores.entry(triple.subject_id).or_insert(0.0) += 3.0 * conf * path_rank;
                    }
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
            return self.extract_answer(graph, query_entities, intent, query_vector, query).0;
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
                let relevance = tle_vsa::cosine_similarity(query_vector, &self.semantic_vector(name));
                Some((score + overlap as f32 + cap_bonus - len_penalty + det_penalty + relevance * 2.0, name.to_string()))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.first().map(|(_, name)| name.clone()).unwrap_or_default()
    }

    /// Sense Reconstruction — iterative belief propagation for answer selection.
    ///
    /// Start from query entities, propagate belief scores along KG triples
    /// weighted by triple confidence, and converge to the entity that
    /// accumulates the most evidence.  Multiple propagation steps let belief
    /// diffuse through the graph, finding answer entities that are 2-3 hops
    /// from the query — without building explicit paths.
    fn sense_answer(
        &self,
        graph: &KnowledgeGraph,
        query_entities: &[usize],
        _intent: Intent,
        _query_vector: &HyperVector,
        query: &str,
    ) -> String {
        use std::collections::HashMap;
        let content_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
            .filter(|w| !matches!(*w, "what"|"which"|"where"|"when"|"how"|"does"
                |"have"|"who"|"with"|"from"|"that"|"this"|"why"|"the"|"was"|"did"))
            .map(|w| w.to_string())
            .collect();

        let mut beliefs: HashMap<usize, f32> = HashMap::new();
        for &e in query_entities { beliefs.insert(e, 1.0); }
        if beliefs.is_empty() { return String::new(); }

        let mut prev_top = 0usize;
        let mut stall = 0u32;
        for _ in 0..4 {
            let mut next: HashMap<usize, f32> = HashMap::new();
            for (&entity, &score) in &beliefs {
                for &ti in graph.adjacency_of(entity) {
                    let t = &graph.triples[ti];
                    let conf = graph.triple_confidence(ti);
                    let target = if t.subject_id == entity { t.object_id } else { t.subject_id };
                    *next.entry(target).or_insert(0.0) += score * conf * 0.8;
                }
            }
            for (&e, &s) in &beliefs { *next.entry(e).or_insert(0.0) += s * 0.5; }
            let total: f32 = next.values().sum::<f32>() + 1e-6;
            for v in next.values_mut() { *v /= total; }

            let top = next.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(&k, _)| k).unwrap_or(0);
            if top == prev_top { stall += 1; } else { stall = 0; prev_top = top; }
            if stall >= 2 { break; }
            beliefs = next;
        }

        let mut ranked: Vec<(f32, String)> = beliefs
            .into_iter()
            .filter_map(|(id, score)| {
                let name = graph.entity_name(id);
                let words = name.split_whitespace().count();
                if words > 5 { return None; }
                let lower = name.to_lowercase();
                let overlap = content_words.iter().filter(|w| lower.contains(w.as_str())).count();
                let cap = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { 1.0 } else { -0.5 };
                let len_pen = match words { 0|1 => 0.0, 2 => 0.4, 3 => 1.2, _ => 2.0 };
                let first = lower.split_whitespace().next().unwrap_or("");
                let det_pen = if matches!(first, "a"|"an"|"the"|"his"|"her"|"its") { -1.5 } else { 0.0 };
                Some((score + overlap as f32 + cap - len_pen + det_pen, name.to_string()))
            })
            .collect();
        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        ranked.first().map(|(_, name)| name.clone()).unwrap_or_default()
    }

    /// Extract a single best answer entity from the graph (legacy — scans
    /// all triples independently).  Prefer sense_answer which uses iterative
    /// belief propagation for higher precision.
    fn extract_answer(
        &self,
        graph: &KnowledgeGraph,
        query_entities: &[usize],
        intent: Intent,
        query_vector: &HyperVector,
        query: &str,
    ) -> (String, Vec<(f32, String, f32, f32, f32, f32, f32)>) {
        // returned: (answer, diagnostics: [(final_score, name, conn, role, overlap, vsa, heuristics), ...])
        let content_words: Vec<String> = query
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() >= 4)
                .filter(|w| !matches!(*w, "what" | "which" | "where" | "when" | "how" | "does"
                    | "have" | "who" | "with" | "from" | "that" | "this" | "why" | "the" | "was" | "did"))
                .map(|w| w.to_string())
                .collect();

        use std::collections::HashMap;
        let mut raw_conn: HashMap<usize, f32> = HashMap::new();
        let mut raw_conn_count: HashMap<usize, usize> = HashMap::new();
        let mut raw_role: HashMap<usize, f32> = HashMap::new();
        let mut raw_overlap: HashMap<usize, f32> = HashMap::new();
        let mut raw_count: HashMap<usize, usize> = HashMap::new();
        let mut relevance_cache: HashMap<usize, f32> = HashMap::new();
        // 2-hop connectivity: entities reachable from a query entity through
        // exactly one intermediate node.  Catches answers like "LH" that
        // connect via "Ovulation" but have no direct query link.
        let mut raw_2hop: HashMap<usize, f32> = HashMap::new();
        let mut raw_2hop_count: HashMap<usize, usize> = HashMap::new();
        // T1.18c D1: typed final-hop expansion (OPI-style). New signal for
        // answer-type-compatible candidates reachable at distance 3+ via a
        // bridge — rescues the Mode C golds (conn=0, ~20% of failures).
        let mut raw_typed: HashMap<usize, f32> = HashMap::new();
        let mut raw_typed_count: HashMap<usize, usize> = HashMap::new();

        // First pass: collect entities directly connected to query entities.
        let mut one_hop: Vec<usize> = Vec::new();
        for triple in &graph.triples {
            let subj_q = query_entities.contains(&triple.subject_id);
            let obj_q = query_entities.contains(&triple.object_id);
            if subj_q != obj_q {
                let other = if subj_q { triple.object_id } else { triple.subject_id };
                if !one_hop.contains(&other) { one_hop.push(other); }
            }
        }

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
                let ov = content_words.iter().filter(|w| lower.contains(w.as_str())).count() as f32;
                // Relation-typed connectivity: strong facts (location, family,
                // role) are more reliable answer links than weak mentions.
                let rel_name = graph.relations.get(triple.relation_id).map(|s| s.as_str()).unwrap_or("");
                let rel_weight = match rel_name {
                    "located_in" | "located_at" | "capital_of" | "born_in" | "born_on"
                    | "died_in" | "founded_in" | "took_place_in" | "occurred_in"
                    | "part_of" | "home_to" | "lived_in" | "played_for" | "won"
                    | "wrote" | "directed" | "painted" | "discovered" | "invented"
                    | "founded" | "designed" | "built" | "created" | "developed"
                    | "president_of" | "founder_of" | "leader_of" | "author_of"
                    | "child_of" | "has_parent" | "married_to" | "known_for"
                    | "released" | "published"
                    | "written_by" | "created_by" | "directed_by" | "played_by"
                    | "portrayed_by" | "founded_by" | "invented_by" | "discovered_by"
                    | "painted_by" | "built_by" => 2.0,
                    "mentions" | "is_related_to" | "named_after" => 0.8,
                    _ => 1.0,
                };
                let conn = if connected_id == Some(entity_id) { 2.5 * triple_conf * rel_weight } else { 0.0 };
                let role = if connected_id == Some(entity_id) {
                    let rb = match intent {
                        Intent::Who => if subject_in_query { 3.0 } else { 0.0 },
                        Intent::What | Intent::Where => if !subject_in_query { 3.0 } else { 0.0 },
                        _ => 0.5,
                    };
                    rb * triple_conf
                } else { 0.0 };
                let rel = *relevance_cache.entry(entity_id).or_insert_with(|| {
                    tle_vsa::cosine_similarity(query_vector, &self.semantic_vector(graph.entity_name(entity_id)))
                });
                *raw_conn.entry(entity_id).or_insert(0.0) += conn;
                if conn > 0.0 { *raw_conn_count.entry(entity_id).or_insert(0) += 1; }
                *raw_role.entry(entity_id).or_insert(0.0) += role;
                *raw_overlap.entry(entity_id).or_insert(0.0) += ov;
                *raw_count.entry(entity_id).or_insert(0) += 1;
            }
        }

        // Second pass: 2-hop connections.  For each 1-hop entity, its other
        // neighbors are 2 hops from the query.  Weight by relation type too.
        for &mid in &one_hop {
            for &ti in graph.adjacency_of(mid) {
                let t = &graph.triples[ti];
                let conf = graph.triple_confidence(ti);
                let other = if t.subject_id == mid { t.object_id } else { t.subject_id };
                if query_entities.contains(&other) || one_hop.contains(&other) { continue; }
                let rel_name = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                let rw = if matches!(rel_name, "located_in"|"capital_of"|"born_in"|"part_of"
                    |"founded_in"|"president_of"|"founder_of"|"leader_of"|"lived_in") { 1.0 }
                    else if matches!(rel_name, "mentions"|"is_related_to") { 0.4 } else { 0.6 };
                *raw_2hop.entry(other).or_insert(0.0) += 1.0 * rw * conf;
                *raw_2hop_count.entry(other).or_insert(0) += 1;
            }
        }

        // T1.18c D1 typed final-hop expansion: the principled widening. Instead
        // of blind 3-hop (QASA: T=4 degrades), expand ONLY along relations
        // whose tail type matches the predicted answer type (OPI 2606.28076:
        // "+4.6/+8.9 Hit@1"). Bridges = query entities + 1-hop + 2-hop nodes
        // (reaches distance 3, where Mode C golds live). Orientation-aware:
        // Person can be the SUBJECT of wrote/`*_by` relations. Only fires for
        // discriminative predictions (Place/Person/Number/Temporal) — the
        // Entity (default) case is non-discriminative and blanket boosts regress.
        let w_typed_env = weight_env("AXIOM_W_TYPED", 0.0);
        let predicted_type = if w_typed_env > 0.0 {
            predict_answer_type(intent, query)
        } else {
            AnswerType::Entity
        };
        if w_typed_env > 0.0 && predicted_type != AnswerType::Entity {
            let mut bridges: Vec<usize> = query_entities.to_vec();
            bridges.extend(one_hop.iter().copied());
            bridges.extend(raw_2hop.keys().copied());
            let mut visited: std::collections::HashSet<usize> = bridges.iter().copied().collect();
            for &b in &bridges {
                for &ti in graph.adjacency_of(b) {
                    let t = &graph.triples[ti];
                    let conf = graph.triple_confidence(ti);
                    let rel_name = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                    let other = if t.subject_id == b { t.object_id } else { t.subject_id };
                    if query_entities.contains(&other) || one_hop.contains(&other) { continue; }
                    if !visited.insert(other) { continue; }
                    let other_name = graph.entity_name(other);
                    let other_is_object = t.subject_id == b;
                    if matches_answer_type_oriented(predicted_type, rel_name, other_is_object, &other_name) {
                        let kind_w = if matches!(rel_name, "mentions"|"is_related_to") { 0.4 } else { 1.0 };
                        *raw_typed.entry(other).or_insert(0.0) += 1.0 * kind_w * conf;
                        *raw_typed_count.entry(other).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut ranked: Vec<(f32, String, f32, f32, f32, f32, f32)> = Vec::new();

        // T1.18e PathHD relation-schema retrieval (env AXIOM_W_PATHHD, default
        // 2.0 — best-known, measured +0.63 exact/f1 stable 3 runs). A
        // genuinely different signal: order-sensitive GHRR relation-sequence
        // matching. The question's relation INTENT is derived deterministically
        // (content words → graph relation names), each candidate's 1-hop/2-hop
        // relation paths from the query entities are bound with GHRR
        // block-unitary product (non-commutative), and scored by calibrated
        // blockwise cosine vs the intent. Fixes the deep-rank class that
        // query-connectivity/count signals cannot (all measured neutral).
        let w_pathhd = weight_env("AXIOM_W_PATHHD", 2.0);
        let pathhd_scores: HashMap<usize, f32> = if w_pathhd > 0.0 {
            self.ghrr_pathhd_signal(graph, query_entities, &one_hop, query, &content_words)
        } else {
            HashMap::new()
        };
        // Candidate set = entities with direct conn OR 2-hop conn.
        let mut candidate_ids: std::collections::HashSet<usize> = raw_conn.keys().copied().collect();
        candidate_ids.extend(raw_2hop.keys().copied());
        candidate_ids.extend(raw_typed.keys().copied());
        // T1.9a: RRF rank fusion (rank-position fusion instead of linear sum).
        // Enabled via AXIOM_RANK=rrf. Rank is invariant under per-signal scale,
        // so overlap (~50) can no longer dominate conn (~2) by magnitude.
        let rrf_mode = std::env::var("AXIOM_RANK").map(|v| v == "rrf").unwrap_or(false);
        let conformal_mode = std::env::var("AXIOM_RANK").map(|v| v == "conformal").unwrap_or(false);
        let k_rrf = weight_env("AXIOM_RRF_K", 60.0);
        // T1.9c: hub-corrected personalized PageRank as a 7th signal.
        // Weight search found 0.3 optimal (23.58→24.21%, stable 4+ runs):
        // fixes M4 (structural conn=0: answers reachable at 3+ hops) and
        // M3 (hub debias via π_q/π ratio) without hurting recall/substring.
        let w_ppr = weight_env("AXIOM_W_PPR", 0.3);
        let ppr_scores: Vec<f32> = if w_ppr > 0.0 {
            // T1.18g D3 query-aware PPR gate (env AXIOM_W_GATE, default off).
            // QASA-style gate σ[v] = lexical overlap of query content words in
            // the entity's fact text, normalized to [0,1]; gates the PPR walk
            // mass into v by σ[v]^γ so it drifts toward question-relevant
            // nodes instead of unrelated hubs. Never VSA cosine (noise).
            let w_gate = weight_env("AXIOM_W_GATE", 0.0);
            if w_gate > 0.0 {
                let gamma = weight_env("AXIOM_W_GATE_GAMMA", 0.5);
                let mut gate = vec![0.0f32; graph.entities.len()];
                let mut max_ov = 0.0f32;
                for (i, _) in graph.entities.iter().enumerate() {
                    let mut s = String::new();
                    for &ti in graph.adjacency_of(i) {
                        let t = &graph.triples[ti];
                        let rel = graph.relations.get(t.relation_id).map(|x| x.as_str()).unwrap_or("");
                        let other = if t.subject_id == i { t.object_id } else { t.subject_id };
                        s.push_str(rel);
                        s.push(' ');
                        s.push_str(&graph.entity_name(other).to_lowercase());
                        s.push(' ');
                    }
                    let ov = content_words.iter().filter(|w| s.contains(w.as_str())).count() as f32;
                    gate[i] = ov;
                    max_ov = max_ov.max(ov);
                }
                if max_ov > 0.0 {
                    for g in gate.iter_mut() {
                        *g /= max_ov;
                    }
                }
                graph.personalized_pagerank_query_aware(query_entities, 60, Some(&gate), gamma)
            } else {
                graph.personalized_pagerank(query_entities, 60)
            }
        } else {
            Vec::new()
        };
        // Raw per-candidate signals, captured for both scoring modes.
        // (id, name, conn_avg, role_avg, hop2_avg, ov, rel, heur, ppr, is_query_named)
        let mut cands: Vec<(usize, String, f32, f32, f32, f32, f32, f32, f32, bool)> = Vec::new();
        // T1.18h C1 exclusion-cue flag (computed once per question).
        let w_excl = weight_env("AXIOM_V2_EXCL", 0.0);
        let excl_cue = w_excl > 0.0 && crate::decompose::has_exclusion_cue(query);
        let letter_cue = crate::decompose::extract_letter_cue(query);
        let w_sheaf = weight_env("AXIOM_W_SHEAF", 0.0);
        // T1.13 MDL differenced tiebreak (env AXIOM_V1_MDL, default off).
        // For each candidate, Δ(e) = shingle_cover(q, name) − shingle_cover(q,
        // facts(e)): how much MORE the query is explained by the entity's
        // evidence than by its surface name. Lower Δ = the facts genuinely
        // discuss the question → the answer-like candidate. Tiebreak-only:
        // applied to candidates within AXIOM_MDL_BAND of the top score.
        let mdl_enabled = weight_env("AXIOM_V1_MDL", 0.0) > 0.0;
        let mdl_l = weight_env("AXIOM_MDL_L", 6.0).max(2.0) as usize;
        let query_lower = query.to_lowercase();
        let fact_texts: HashMap<usize, String> = if mdl_enabled {
            let mut m: HashMap<usize, String> = HashMap::new();
            for t in &graph.triples {
                let rel = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                let e = m.entry(t.subject_id).or_default();
                e.push_str(rel);
                e.push(' ');
                e.push_str(&graph.entity_name(t.object_id).to_lowercase());
                e.push(' ');
                let e = m.entry(t.object_id).or_default();
                e.push_str(rel);
                e.push(' ');
                e.push_str(&graph.entity_name(t.subject_id).to_lowercase());
                e.push(' ');
            }
            m
        } else {
            HashMap::new()
        };
        let mut mdls: HashMap<String, (f32, bool)> = HashMap::new();
        let prune_k = weight_env("AXIOM_PATHHD_PRUNE", 0.0) as usize;
        let mut pruned_ids: Vec<usize> = candidate_ids.into_iter().collect();
        if prune_k > 0 && w_pathhd > 0.0 && !pathhd_scores.is_empty() {
            let max_score = pruned_ids
                .iter()
                .map(|id| pathhd_scores.get(id).copied().unwrap_or(-9999.0))
                .fold(f32::NEG_INFINITY, f32::max);
            if max_score > -100.0 {
                pruned_ids.sort_by(|&a, &b| {
                    let sa = pathhd_scores.get(&a).copied().unwrap_or(-9999.0);
                    let sb = pathhd_scores.get(&b).copied().unwrap_or(-9999.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                pruned_ids.truncate(prune_k);
            }
        }
        for id in pruned_ids {
            let name = graph.entity_name(id);
            let words = name.split_whitespace().count();
            if words > 5 || words == 0 { continue; }
            let len_pen = match words { 0|1 => 0.0, 2 => 0.4, 3 => 1.2, 4 => 2.0, _ => 3.0 };
            let cap_bonus = if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) { 1.0 } else { -0.5 };
            let first = name.split_whitespace().next().map(|w| w.to_lowercase()).unwrap_or_default();
            let det_pen = if matches!(first.as_str(), "a"|"an"|"the"|"some"|"his"|"her"|"its"|"this"|"that") { -1.5 } else { 0.0 };
            // Frequency term is the hub amplifier (23/57 top-5 failures won by
            // heur). Split its weight from the length/cap/det penalties so we
            // can reduce ONLY the count contribution (T1.9b). Env AXIOM_W_COUNT.
            // T1.18d D2 (env AXIOM_W_RATIO, default off): query-conditional +
            // saturated count (BM25/RSJ-grounded). count_cond = triples where
            // the entity is actually connected to the query (conn+2hop), and
            // count_ratio = count_cond/count is the Milne-Witten relative
            // frequency (hub-invariant; the count analog of the relative PPR
            // that already worked +0.63). BM25_sat(c)=c(k1+1)/(c+k1) caps the
            // hub's multiplicative advantage while keeping absolute evidence
            // monotonic. Fixes Mode B (count dominance, ~15% of failures)
            // WITHOUT the failed global count cut (0.2→0.05 → 17.61%).
            let w_count = weight_env("AXIOM_W_COUNT", 0.2);
            let w_ratio = weight_env("AXIOM_W_RATIO", 0.0);
            let k1 = weight_env("AXIOM_W_RATIO_K1", 2.0);
            let raw_count_c = *raw_count.get(&id).unwrap_or(&0) as f32;
            let heur = if w_ratio > 0.0 {
                let count_cond = (*raw_conn_count.get(&id).unwrap_or(&0)
                    + *raw_2hop_count.get(&id).unwrap_or(&0)) as f32;
                let count_ratio = count_cond / raw_count_c.max(1.0);
                let sat = count_cond * (k1 + 1.0) / (count_cond + k1);
                w_count * sat + w_ratio * count_ratio - len_pen + cap_bonus + det_pen
            } else {
                w_count * raw_count_c - len_pen + cap_bonus + det_pen
            };
            // Average connectivity per link — neutralizes the hub problem:
            // an entity with 197 facts (Macron) must not beat an entity with
            // 1 strong link (Paris, capital_of France).
            let conn = *raw_conn.get(&id).unwrap_or(&0.0);
            let conn_links = *raw_conn_count.get(&id).unwrap_or(&1).max(&1);
            let conn_avg = conn / conn_links as f32;
            let role = *raw_role.get(&id).unwrap_or(&0.0);
            let role_links = *raw_conn_count.get(&id).unwrap_or(&1).max(&1);
            let role_avg = role / role_links as f32;
            // 2-hop bonus: entities reachable via one intermediate node get
            // a fraction of connectivity, capturing answers like "LH".
            let hop2 = *raw_2hop.get(&id).unwrap_or(&0.0);
            let hop2_links = *raw_2hop_count.get(&id).unwrap_or(&1).max(&1);
            let hop2_avg = hop2 / hop2_links as f32;
            let ov = *raw_overlap.get(&id).unwrap_or(&0.0);
            let rel = relevance_cache.get(&id).copied().unwrap_or(0.0);
            let is_query_named = query_entities.contains(&id);
            // Query-named penalty, intent-aware (T1.9b): for "What is X?" /
            // "Who is X?" the query-named entity X IS often the answer (Milky
            // Way), so the penalty must be milder. For Where/When/How/How-many
            // the query-named entity is the reference, never the answer — full
            // penalty. Env override for calibration.
            // T1.11b (deep-review, env AXIOM_V1_QNP): a query-named entity with
            // ZERO direct structural connectivity (conn==0 AND hop2==0) is the
            // reference/topic, not the answer — the answer must be connected to
            // the question's entities. It gets the FULL penalty regardless of
            // intent (deep-rank winners "Buddy Holly", "House of Habsburg" are
            // exactly this class: conn=0 but overlap+count dominate via the
            // What/Who mild 0.6 penalty). Direct-conn golds (Milky Way) are
            // unaffected.
            let qp_full = weight_env("AXIOM_QP_WHERE", 0.2);
            let qp_mild = weight_env("AXIOM_QP_WHAT", 0.6);
            // T1.18b (default ON): conditional full penalty for query-named
            // entities with zero direct structural connectivity (conn==0 AND
            // hop2==0). They are the reference/anchor, never the answer; the
            // answer must be connected to the question's entities. Measured on
            // the STRICT metric: exact 13.84→14.47% (+0.63, stable 3 runs).
            let qp_cond = weight_env("AXIOM_V1_QNP", 1.0);
            // T1.18h C1 exclusion-cue (env AXIOM_V2_EXCL, default off): when
            // the question has an exclusion/contrast cue ("the other one",
            // "besides X", "apart from X", "two of the three"), EVERY
            // query-named entity is a reference/anchor — the answer is the
            // OTHER — so they get the full penalty regardless of intent or
            // connectivity. Harder than QNP (which needs conn==0 && hop2==0);
            // Buddy Holly (conn=0, hop2>0) escapes QNP's mild penalty but must
            // not survive an exclusion question.
            let query_penalty = if is_query_named {
                if (qp_cond > 0.0 && conn_avg == 0.0 && hop2_avg == 0.0)
                    || (excl_cue && w_excl > 0.0)
                {
                    qp_full
                } else {
                    match intent {
                        Intent::What | Intent::Who => qp_mild,
                        Intent::Why => qp_mild,
                        _ => qp_full,
                    }
                }
            } else { 1.0 };
            let rel_weight = weight_env("AXIOM_W_VSA", 2.0);
            let w_conn = weight_env("AXIOM_W_CONN", 1.0);
            let w_role = weight_env("AXIOM_W_ROLE", 0.8);
            let w_hop2 = weight_env("AXIOM_W_HOP2", 0.5);
            let w_ov = weight_env("AXIOM_W_OV", 0.05);
            let w_heur = weight_env("AXIOM_W_HEUR", 1.0);
            // Connectivity-first: overlap is a weak tiebreaker, not primary.
            // T1.8a: weight search (full 318 bench, stable) found 0.05 beats
            // 0.15 — overlap dominance (question-named entities) was drowning
            // correct connected answers. Recall unchanged at 76.10%.
            let ppr = if w_ppr > 0.0 { ppr_scores.get(id).copied().unwrap_or(0.0) } else { 0.0 };
            // T1.18e PathHD relation-schema signal (per-candidate max path cos)
            let pathhd = if w_pathhd > 0.0 { pathhd_scores.get(&id).copied().unwrap_or(0.0) } else { 0.0 };
            // T1.18c D1 typed signal: average typed-final-hop connectivity. A
            // candidate that is the answer-type-compatible tail of a bridge
            // relation gets a dedicated boost — this is what reaches Mode C
            // golds that have conn=0 / hop2=0.
            let typed = *raw_typed.get(&id).unwrap_or(&0.0);
            let typed_links = *raw_typed_count.get(&id).unwrap_or(&1).max(&1);
            let typed_avg = typed / typed_links as f32;
            // T1.11 M1 conditional overlap-veto (env AXIOM_V1_M1, default off).
            // Overlap counts ONLY when the candidate is structurally connected to
            // the query entities: direct conn, 2-hop, or relative-PPR support
            // above tau. Kills overlap-dominance (M1, 21/165 failures): entities
            // that merely share surface words with the question but have no graph
            // connection to it can no longer win via overlap. The linear sum
            // provably cannot express "name-match only counts when connectivity
            // present" — a hard gate can, and it preserves magnitudes (immune to
            // the percentile/fusion failure class).
            let w_m1 = weight_env("AXIOM_V1_M1", 1.0);
            let m1_tau = weight_env("AXIOM_V1_M1_TAU", 0.0);
            let has_struct = conn_avg > 0.0
                || hop2_avg > 0.0
                || (w_ppr > 0.0 && ppr > m1_tau);
            let mut ov_eff = ov;
            if w_m1 > 0.0 && ov_eff > 0.0 && !has_struct {
                ov_eff = 0.0;
            }
            // T1.18d conditional VSA boost (env AXIOM_VSA_NOSTRUCT, default
            // off): a candidate with NO structural support (conn=0) that is
            // semantically close to the query (high VSA cosine) is the buried
            // Mode-C gold — measured: buried golds average vsa=0.12 vs winner
            // 0.04 (e.g. gold 'beetroot' vsa=0.97, 'potato' 0.99). Give the
            // semantic signal a conditional weight boost ONLY for structurally
            // unsupported candidates — connected candidates keep rel_weight,
            // so this cannot reproduce the global-VSA-weight regressions.
            let w_vsa_ns = weight_env("AXIOM_VSA_NOSTRUCT", 0.0);
            let vsa_weight = rel_weight
                + if w_vsa_ns > 0.0 && conn_avg == 0.0 { w_vsa_ns } else { 0.0 };
            let sheaf_score = if w_sheaf > 0.0 && !query_entities.is_empty() {
                let mut sheaf_triples = Vec::new();
                for &q in query_entities {
                    for t in graph.get_triples_from(q) {
                        if t.object_id == id {
                            sheaf_triples.push((q, graph.relation_name(t.relation_id).to_string(), id));
                        }
                    }
                    for t in graph.get_triples_to(q) {
                        if t.subject_id == id {
                            sheaf_triples.push((id, graph.relation_name(t.relation_id).to_string(), q));
                        }
                    }
                    // 2-hop expansion
                    for t1 in graph.get_triples_from(q) {
                        let m = t1.object_id;
                        for t2 in graph.get_triples_from(m) {
                            if t2.object_id == id {
                                sheaf_triples.push((q, graph.relation_name(t1.relation_id).to_string(), m));
                                sheaf_triples.push((m, graph.relation_name(t2.relation_id).to_string(), id));
                            }
                        }
                    }
                }
                if !sheaf_triples.is_empty() {
                    let energy = crate::sheaf::evaluate_subgraph_consistency(&sheaf_triples, query_entities, id);
                    (-energy as f32).exp()
                } else {
                    0.0f32
                }
            } else {
                0.0f32
            };
            let sheaf_gate = weight_env("AXIOM_SHEAF_GATE", 0.0);
            let conn_effective = if sheaf_gate > 0.0 && sheaf_score > 0.0 {
                conn_avg * ((1.0 - sheaf_gate) + sheaf_gate * sheaf_score)
            } else {
                conn_avg
            };
            let letter_penalty = if let Some(lc) = letter_cue {
                let first_char = name.trim().chars().next().map(|c| c.to_ascii_uppercase());
                if first_char == Some(lc) {
                    1.5
                } else {
                    0.05
                }
            } else {
                1.0
            };
            let total_penalty = query_penalty * letter_penalty;
            let score = (conn_effective * w_conn
                + role_avg * w_role
                + hop2_avg * w_hop2
                + ov_eff * w_ov
                + rel * vsa_weight
                + heur * w_heur
                + ppr * w_ppr
                + typed_avg * w_typed_env
                + pathhd * w_pathhd
                + sheaf_score * w_sheaf) * total_penalty;
            if rrf_mode || conformal_mode {
                cands.push((id, name.to_string(), conn_avg, role_avg, hop2_avg, ov, rel, heur, ppr, is_query_named));
            } else {
                ranked.push((score, name.to_string(), conn_avg, role_avg, ov, rel * rel_weight, heur));
                if mdl_enabled {
                    let facts = fact_texts.get(&id).map(|s| s.as_str()).unwrap_or("");
                    let use_rate = weight_env("AXIOM_USE_MDL_RATE", 0.0) > 0.0;
                    let metric_val = if use_rate {
                        let mut context = query_lower.clone();
                        if !facts.is_empty() {
                            context.push(' ');
                            context.push_str(facts);
                        }
                        crate::mdl::conditional_description_rate(context.as_bytes(), name.as_bytes(), mdl_l) as f32
                    } else {
                        let cov_name = shingle_cover(&query_lower, &name.to_lowercase(), mdl_l);
                        let cov_facts = shingle_cover(&query_lower, facts, mdl_l);
                        cov_name as f32 - cov_facts as f32
                    };
                    mdls.insert(name.to_string(), (metric_val, is_query_named));
                }
            }
        }

        if rrf_mode {
            // RRF: for each signal list, rank candidates descending; absent
            // from a list contributes 0. score(e) = Σ w_i/(k+rank_i(e)).
            let n = cands.len();
            // signal indices: 0=conn_avg, 1=role_avg, 2=hop2_avg, 3=ov, 4=rel, 5=heur
            let mut contrib = vec![0.0f64; n];
            // per-signal weights: default equal except VSA=0 (documented noise
            // with random codebook — demoted per research synthesis);
            // env AXIOM_RRF_W_{CONN,ROLE,HOP2,OV,VSA,HEUR}
            let wr = [
                weight_env("AXIOM_RRF_W_CONN", 1.0),
                weight_env("AXIOM_RRF_W_ROLE", 1.0),
                weight_env("AXIOM_RRF_W_HOP2", 1.0),
                weight_env("AXIOM_RRF_W_OV", 1.0),
                weight_env("AXIOM_RRF_W_VSA", 0.0),
                weight_env("AXIOM_RRF_W_HEUR", 1.0),
            ];
            let mut order: Vec<usize> = (0..n).collect();
            for (si, sig) in [0usize,1,2,3,4,5].iter().enumerate() {
                if wr[si] <= 0.0 { continue; }
                let get = |idx: usize| match *sig {
                    0 => cands[idx].2,
                    1 => cands[idx].3,
                    2 => cands[idx].4,
                    3 => cands[idx].5,
                    4 => cands[idx].6,
                    _ => cands[idx].7,
                };
                order.sort_by(|&a, &b| {
                    get(b).partial_cmp(&get(a)).unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut rank = 1usize;
                for &idx in &order {
                    // skip zero-valued signals (absent from this list)
                    if get(idx) > 0.0 {
                        contrib[idx] += (wr[si] as f64) / (k_rrf as f64 + rank as f64);
                        rank += 1;
                    }
                }
            }
            for (i, c) in cands.iter().enumerate() {
                // Re-apply the query-named penalty that linear mode uses
                // (entities named in the question are what we ask ABOUT).
                let qp = if c.9 { 0.2f32 } else { 1.0 };
                ranked.push((contrib[i] as f32 * qp, c.1.clone(), c.2, c.3, c.5, c.6, c.7));
            }
        }

        if conformal_mode {
            // T1.10a L4: conformal + calibrated log-odds fusion.
            // For each signal, compute the empirical p-value of every candidate
            // against the per-question candidate distribution:
            //   p_i(e) = ( #candidates with s_i >= s_i(e) ) / ( #candidates )
            // (higher raw signal = smaller p = more unusual = more answer-like).
            // Fusion in log space (product-of-experts) gives conditional
            // weighting: an entity with extreme overlap but ZERO connectivity
            // gets a huge negative log-odds from the conn signal that no linear
            // weight could cancel. sigmoid-never-softmax per candidate.
            let n = cands.len();
            // signal extractors: conn, role, hop2, ov, vsa, heur, ppr
            let sigs: Vec<fn(&(usize, String, f32, f32, f32, f32, f32, f32, f32, bool)) -> f32> = vec![
                |c| c.2, |c| c.3, |c| c.4, |c| c.5, |c| c.6, |c| c.7, |c| c.8,
            ];
            let n_sig = sigs.len();
            // per-signal weights (env-calibratable), default all 1.0 (calibrated
            // log-odds means the probability scale does the weighting).
            let wc: Vec<f32> = (0..n_sig).map(|si| {
                weight_env(&format!("AXIOM_CP_W{}", si), 1.0)
            }).collect();
            let temp = weight_env("AXIOM_CP_TEMP", 1.0);
            let mut contrib = vec![0.0f64; n];
            for (si, get) in sigs.iter().enumerate() {
                if wc[si] <= 0.0 { continue; }
                // empirical p-value: fraction of candidates with signal >= this one
                for i in 0..n {
                    let mut ge = 0usize;
                    for j in 0..n {
                        if get(&cands[j]) >= get(&cands[i]) { ge += 1; }
                    }
                    let p = ge as f64 / n as f64;
                    let p = p.clamp(1e-6, 1.0 - 1e-6);
                    // log-odds: ln(p/(1-p)); LOW p (extreme signal) → negative → 
                    // we negate so high signal contributes positively.
                    contrib[i] += (wc[si] as f64) * -(p.ln() - (1.0 - p).ln());
                }
            }
            // sigmoid with temperature, applied per-candidate (never softmax):
            // σ(x·T) ∈ (0,1), independent per candidate — no mass stealing.
            // AXIOM_CP_SIG=0 (default) skips the sigmoid: for argmax ranking the
            // raw log-odds ordering is what matters; sigmoid compresses all
            // scores near 0/1 and ties break on the query penalty.
            let use_sigmoid = weight_env("AXIOM_CP_SIG", 0.0) > 0.0;
            for (i, c) in cands.iter().enumerate() {
                let raw = contrib[i] as f32;
                let s = if use_sigmoid {
                    let x = raw * temp;
                    1.0 / (1.0 + (-x).exp())
                } else {
                    raw
                };
                let qp = if c.9 {
                    match intent {
                        Intent::What | Intent::Who => 0.6,
                        Intent::Why => 0.6,
                        _ => 0.2,
                    }
                } else { 1.0 };
                ranked.push((s * qp, c.1.clone(), c.2, c.3, c.5, c.6, c.7));
            }
        }

        ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // T1.13 MDL tiebreak: only reorder candidates within a tight score band
        // of the top (near-ties, M2) — by Δ ascending (facts explain the query
        // better than the name). Never a primary scorer; outside the band the
        // linear order is untouched. Query-named candidates (the reference, not
        // the answer) are excluded: their facts trivially match the query, so
        // Δ would wrongly promote them.
        if mdl_enabled && !mdls.is_empty() {
            let band = weight_env("AXIOM_MDL_BAND", 0.02);
            if let Some(&top) = ranked.first().map(|c| &c.0) {
                let band_idx: Vec<usize> = ranked
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| (top - c.0).abs() <= band)
                    .filter(|(i, _)| mdls.get(&ranked[*i].1).map(|(_, qn)| !qn).unwrap_or(false))
                    .map(|(i, _)| i)
                    .collect();
                if band_idx.len() >= 2 {
                    let mut sorted_band: Vec<usize> = band_idx.clone();
                    sorted_band.sort_by(|&ia, &ib| {
                        let da = mdls.get(&ranked[ia].1).map(|(d, _)| *d).unwrap_or(0.0);
                        let db = mdls.get(&ranked[ib].1).map(|(d, _)| *d).unwrap_or(0.0);
                        da.partial_cmp(&db)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                ranked[ib].0.partial_cmp(&ranked[ia].0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    });
                    let in_band: std::collections::HashSet<usize> = band_idx.iter().copied().collect();
                    let mut new_ranked = Vec::with_capacity(ranked.len());
                    for &i in &sorted_band {
                        new_ranked.push(ranked[i].clone());
                    }
                    for (i, c) in ranked.iter().enumerate() {
                        if !in_band.contains(&i) {
                            new_ranked.push(c.clone());
                        }
                    }
                    ranked = new_ranked;
                }
            }
        }

        // T1.18f PathHD adjudicator: intent-consistency re-rank of the top-K
        // (deterministic replacement for the paper's LLM step, spec §8, worth
        // +0.7-0.8pt in PathHD). For a discriminative answer type (Place/
        // Person/Number/Temporal), candidates whose connecting relation to a
        // query entity matches the predicted type rank ABOVE type-contradicting
        // ones — within the top-K only, resolving near-ties and direction
        // errors. Env AXIOM_V2_ADJ (default off); AXIOM_V2_ADJ_K = top-K size.
        let w_adj = weight_env("AXIOM_V2_ADJ", 0.0);
        if w_adj > 0.0 {
            let predicted = predict_answer_type(intent, query);
            if predicted != AnswerType::Entity {
                let k = weight_env("AXIOM_V2_ADJ_K", 5.0).max(2.0) as usize;
                let top: Vec<usize> = (0..ranked.len().min(k)).collect();
                let consistent: Vec<bool> = top
                    .iter()
                    .map(|&i| {
                        let name = &ranked[i].1;
                        for t in &graph.triples {
                            let subj_q = query_entities.contains(&t.subject_id);
                            let obj_q = query_entities.contains(&t.object_id);
                            if subj_q != obj_q {
                                let cand = if subj_q { t.object_id } else { t.subject_id };
                                if graph.entity_name(cand) == *name {
                                    let rel = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                                    let c_is_obj = subj_q;
                                    if matches_answer_type_oriented(predicted, rel, c_is_obj, name) {
                                        return true;
                                    }
                                }
                            }
                        }
                        false
                    })
                    .collect();
                if consistent.iter().any(|&c| c) && consistent.iter().any(|&c| !c) {
                    let mut order: Vec<usize> = top.clone();
                    order.sort_by(|&ia, &ib| {
                        consistent[ia]
                            .cmp(&consistent[ib])
                            .reverse()
                            .then_with(|| {
                                ranked[ib].0.partial_cmp(&ranked[ia].0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    });
                    let in_top: std::collections::HashSet<usize> = top.iter().copied().collect();
                    let mut new_ranked = Vec::with_capacity(ranked.len());
                    for &i in &order {
                        new_ranked.push(ranked[i].clone());
                    }
                    for (i, c) in ranked.iter().enumerate() {
                        if !in_top.contains(&i) {
                            new_ranked.push(c.clone());
                        }
                    }
                    ranked = new_ranked;
                }
            }
        }

        let answer = ranked.first().map(|(_, name, ..)| name.clone()).unwrap_or_default();
        (answer, ranked)
    }

    /// T1.18e PathHD relation-schema signal: per-candidate MAX calibrated
    /// blockwise-cosine of its 1-hop/2-hop relation paths from the query
    /// entities, scored against the question's relation INTENT (GHRR
    /// order-sensitive binding). Deterministic, zero-training.
    fn ghrr_pathhd_signal(
        &self,
        graph: &KnowledgeGraph,
        query_entities: &[usize],
        one_hop: &[usize],
        query: &str,
        content_words: &[String],
    ) -> std::collections::HashMap<usize, f32> {
        use tle_ghrr::retrieval::calibrated_score;
        use tle_ghrr::{GhrrCodebook, GhrrVector, RelationSchemaIndex};
        use std::collections::HashMap;

        let mut codebook = GhrrCodebook::new(0xA5E1_2016_1D12_5EED);
        let mut index = RelationSchemaIndex::new();
        for r in &graph.relations {
            index.count(r);
        }

        // Query relation intent: FIRST scan the question for the
        // RELATIONAL_PHRASES vocabulary (aligns with the graph's relation
        // names, e.g. "composed" / "was written by" → "written_by").
        let mut intent_rels: Vec<String> = crate::decompose::query_relations(query);
        if intent_rels.is_empty() {
            // fallback 1: content words → graph relation names (substring/
            // prefix match; "composed" ⊂ "composed_by").
            for w in content_words {
                if w.len() < 4 {
                    continue;
                }
                for r in &graph.relations {
                    let rn = r.replace('_', " ");
                    let rw = r.replace('_', "");
                    if rn.contains(w.as_str())
                        || rw.contains(w.as_str())
                        || w.as_str().starts_with(&rn)
                        || rn.starts_with(w.as_str())
                    {
                        if !intent_rels.contains(r) {
                            intent_rels.push(r.clone());
                        }
                    }
                }
            }
        }
        if intent_rels.is_empty() {
            // fallback 2: relation most frequent among query entities' triples
            let mut freq: HashMap<usize, usize> = HashMap::new();
            for &qe in query_entities {
                for &ti in graph.adjacency_of(qe) {
                    *freq.entry(graph.triples[ti].relation_id).or_insert(0) += 1;
                }
            }
            if let Some((rid, _)) = freq.iter().max_by_key(|(_, v)| **v) {
                if let Some(r) = graph.relations.get(*rid) {
                    intent_rels.push(r.clone());
                }
            }
        }
        if intent_rels.is_empty() {
            return HashMap::new();
        }
        // encode each intent relation separately (no bundling dilution)
        let intent_vecs: Vec<GhrrVector> =
            intent_rels.iter().map(|r| codebook.get_or_insert(r)).collect();

        let alpha = weight_env("AXIOM_PATHHD_ALPHA", 0.2);
        let beta = weight_env("AXIOM_PATHHD_BETA", 0.1);
        let lambda = weight_env("AXIOM_PATHHD_LAMBDA", 0.8);

        let mut best: HashMap<usize, f32> = HashMap::new();
        let qes: std::collections::HashSet<usize> = query_entities.iter().copied().collect();
        let mut record = |cand: usize, vec: &GhrrVector, idf: f32, len: usize| {
            let sim = intent_vecs
                .iter()
                .map(|iv| iv.blockwise_cosine(vec))
                .fold(f32::NEG_INFINITY, |a, b| a.max(b));
            let sc = calibrated_score(sim, idf, len, alpha, beta, lambda);
            best.entry(cand).and_modify(|e| *e = e.max(sc)).or_insert(sc);
        };

        // 1-hop paths: (query_entity, r, cand)
        for t in &graph.triples {
            let subj_q = qes.contains(&t.subject_id);
            let obj_q = qes.contains(&t.object_id);
            if subj_q != obj_q {
                let cand = if subj_q { t.object_id } else { t.subject_id };
                let r = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                let v = codebook.get_or_insert(r);
                record(cand, &v, index.idf(r), 1);
            }
        }

        // 2-hop paths: query_entity --r1--> mid --r2--> cand
        let one_hop_set: std::collections::HashSet<usize> = one_hop.iter().copied().collect();
        let mut mid_src: HashMap<usize, Vec<usize>> = HashMap::new();
        for t in &graph.triples {
            let subj_q = qes.contains(&t.subject_id);
            let obj_q = qes.contains(&t.object_id);
            if subj_q != obj_q {
                let mid = if subj_q { t.object_id } else { t.subject_id };
                if one_hop_set.contains(&mid) {
                    mid_src.entry(mid).or_default().push(t.relation_id);
                }
            }
        }
        for (&mid, srcs) in &mid_src {
            for &ti in graph.adjacency_of(mid) {
                let t = &graph.triples[ti];
                let cand = if t.subject_id == mid { t.object_id } else { t.subject_id };
                if qes.contains(&cand) || one_hop_set.contains(&cand) { continue; }
                let r2 = graph.relations.get(t.relation_id).map(|s| s.as_str()).unwrap_or("");
                for &r1id in srcs {
                    if t.relation_id == r1id { continue; }
                    let r1 = graph.relations.get(r1id).map(|s| s.as_str()).unwrap_or("");
                    let v1 = codebook.get_or_insert(r1);
                    let v2 = codebook.get_or_insert(r2);
                    let pv = GhrrVector::bind_path(&[&v1, &v2]);
                    record(cand, &pv, index.idf_seq(r1, r2), 2);
                }
            }
        }

        best
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

        // Exact and normalized matches using morphological decomposition.
        let morph = tle_afc::MorphTokenizer::new();
        let normalized_query: Vec<String> = words.iter().map(|w| morph_depluralize(&morph, w)).collect();

        // Exact and morphologically-normalized matches.
        for (name, &id) in &self.graph.entity_index {
            let entity_lower = name.to_lowercase();
            let entity_words: Vec<&str> = entity_lower.split('_').collect();
            let matches = words.contains(&entity_lower.as_str()) || entity_words.iter().any(|ew| {
                words.contains(ew) || normalized_query.iter().any(|word| {
                    word == &normalize_entity_token(ew) || morph_depluralize(&morph, ew) == *word
                })
            });
            if matches && !found_entities.contains(&id) {
                found_entities.push(id);
            }
        }

        // Punctuation-stripped matching (T1.9): "O'Hare" in a query is split by
        // the cleaner into ["o","hare"], so the entity "O'Hare" never matched —
        // question-named entities dodged the query penalty and won via overlap.
        // Strip ALL non-alphanumerics WITHIN each raw whitespace token, so
        // "O'Hare" → "ohare" can align with the entity "O'Hare" → "ohare".
        let raw_lower = query.to_lowercase();
        let query_clean: Vec<String> = raw_lower
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|w| w.len() >= 2)
            .collect();
        for (name, &id) in &self.graph.entity_index {
            let entity_clean: String = name.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric())
                .collect();
            if entity_clean.len() < 3 { continue; }
            let hit = query_clean.iter().any(|qw| {
                qw == &entity_clean
                    || (entity_clean.len() >= qw.len() && entity_clean.starts_with(qw.as_str()))
                    || (qw.len() >= entity_clean.len() && qw.starts_with(entity_clean.as_str()))
            });
            if hit && !found_entities.contains(&id) {
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

    /// Build a query vector by bundling vectors of content words only.
    /// Stopwords/functions words are excluded so they don't inject VSA noise
    /// into the relevance signal (root cause of T3.1b regression).
    fn build_query_vector(&mut self, query: &str) -> HyperVector {
        let dim = self.codebook.dim();
        let lower = query.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|w| !w.is_empty())
            .filter(|w| !crate::decompose::is_question_stop_word(w))
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

/// Depluralize using morphological tokenizer — returns root form.
fn morph_depluralize(morph: &tle_afc::MorphTokenizer, word: &str) -> String {
    let morphemes = morph.decompose(word);
    for m in &morphemes {
        if matches!(m.mtype, tle_afc::morph_tokenizer::MorphemeType::Root) {
            return m.text.clone();
        }
    }
    word.to_lowercase()
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
    fn test_shingle_cover_symmetric_explanation() {
        // A string fully contained in another is fully explained.
        let cov = shingle_cover("who won the 1998 world cup", "the 1998 world cup was won by france", 6);
        assert!(cov > 0, "query words must be found in the fact text");
        // A disjoint string explains nothing.
        assert_eq!(shingle_cover("quantum entanglement", "banana cake recipes", 6), 0);
        // Self-explanatory: greedy length-l coverage covers all but a <l tail
        // (10 chars at l=4 → 8 covered, the final 2-char tail is too short).
        let full = shingle_cover("abcdefghij", "abcdefghij", 4);
        assert_eq!(full, 8);
    }

    #[test]
    fn test_shingle_cover_deterministic() {
        let a = "which hormone helps control ovulation";
        let b = "luteinizing hormone helps control ovulation in mammals";
        assert_eq!(shingle_cover(a, b, 6), shingle_cover(a, b, 6));
    }

    #[test]
    fn test_mdl_differenced_direction() {
        // Gold entity: query is explained by its FACTS (evidence discusses the
        // question) more than by its bare name → Δ negative → ranks first.
        let q = "who won the 1998 world cup";
        let gold_name = "france";
        let gold_facts = "france won the 1998 world cup against brazil at the final";
        let junk_name = "the 1998 world cup";
        let junk_facts = "the tournament was held in paris";
        let gold_delta =
            shingle_cover(q, gold_name, 6) as f32 - shingle_cover(q, gold_facts, 6) as f32;
        let junk_delta =
            shingle_cover(q, junk_name, 6) as f32 - shingle_cover(q, junk_facts, 6) as f32;
        assert!(
            gold_delta < junk_delta,
            "gold Δ {gold_delta} must be lower than junk Δ {junk_delta}"
        );
    }

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
    fn test_extract_query_entities_punctuation_matching() {
        // T1.9: "O'Hare" in the query must link to the entity "O'Hare"
        // (and "O'Hare Airport"), so the query penalty fires. Previously the
        // cleaner split "O'Hare" -> ["o","hare"] and the entity never matched.
        let mut gen = AxiomGen::new(2048);
        gen.add_fact("O'Hare", "located_in", "Chicago");
        gen.add_fact("O'Hare_Airport", "serves", "Chicago");
        gen.add_fact("Chicago", "capital_of", "Illinois");

        let entities = gen.extract_query_entities("In which city would you find O'Hare International Airport?");
        let names: Vec<String> = entities.iter().map(|&id| gen.graph.entity_name(id).to_string()).collect();
        assert!(
            entities.contains(&gen.graph.entity_id("O'Hare").unwrap()),
            "query 'O'Hare' should match entity 'O'Hare', got {:?}",
            names
        );
        assert!(
            entities.contains(&gen.graph.entity_id("O'Hare_Airport").unwrap()),
            "query 'O'Hare' should match entity 'O'Hare_Airport', got {:?}",
            names
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
