//! # TLE-Chat v2: Full VSA-Powered Conversational Engine
//!
//! Architecture:
//! - VSA Intent Detection (encode input as HV, find nearest intent by cosine similarity)
//! - Subject Extraction (KB-aware word matching + stopword removal)
//! - Knowledge Retrieval (bind(subject, relation) key lookup)
//! - Transition Generation (corpus-based fallback)
//! - Template Assembly (fill slots from KB)
//! - Conversation Memory (last 5 turns, pronoun resolution)
//!
//! Zero parameters. Zero sampling. 100% deterministic.

mod corpus;

use std::collections::{HashMap, HashSet};
use std::io::{self, BufRead, Write};

use tle_vsa::{bind, bundle, cosine_similarity, Codebook, HyperVector, DEFAULT_DIM};
use tle_memory::MemoryBank;

// ═══════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════

const MAX_CONVERSATION_HISTORY: usize = 5;
const INTENT_CONFIDENCE_THRESHOLD: f32 = 0.15;

/// Stop words removed during subject extraction.
const STOP_WORDS: &[&str] = &[
    "what", "where", "how", "is", "the", "a", "an", "are", "do", "does",
    "can", "has", "have", "of", "in", "on", "it", "they", "tell", "me",
    "about", "which", "who", "please", "could", "would", "you", "know",
    "big", "color", "colour", "sound", "eat", "eats", "make", "makes",
    "capital", "many", "much", "why", "when", "did", "was", "were",
];

// ═══════════════════════════════════════════════════════════════
// Intent System
// ═══════════════════════════════════════════════════════════════

/// All supported intents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Intent {
    Greeting,
    Farewell,
    WhatIs,
    WhatColor,
    WhereIs,
    WhatEats,
    WhatCan,
    WhatHas,
    WhatSound,
    HowBig,
    CapitalOf,
    TellAbout,
    YesNoQuestion,
    Thanks,
    Help,
    Unknown,
}

impl Intent {
    fn all() -> &'static [Intent] {
        &[
            Intent::Greeting, Intent::Farewell, Intent::WhatIs,
            Intent::WhatColor, Intent::WhereIs, Intent::WhatEats,
            Intent::WhatCan, Intent::WhatHas, Intent::WhatSound,
            Intent::HowBig, Intent::CapitalOf, Intent::TellAbout,
            Intent::YesNoQuestion, Intent::Thanks, Intent::Help,
            Intent::Unknown,
        ]
    }

    /// Keyword patterns that define each intent for HV encoding.
    fn keywords(&self) -> &'static [&'static str] {
        match self {
            Intent::Greeting => &["hello", "hi", "hey", "greetings", "good", "morning", "afternoon", "evening", "howdy"],
            Intent::Farewell => &["bye", "goodbye", "farewell", "see", "you", "later", "quit", "exit"],
            Intent::WhatIs => &["what", "is", "define", "meaning", "explain", "describe"],
            Intent::WhatColor => &["what", "color", "colour"],
            Intent::WhereIs => &["where", "is", "located", "location", "find", "place"],
            Intent::WhatEats => &["what", "eat", "eats", "food", "diet", "feed"],
            Intent::WhatCan => &["what", "can", "able", "ability", "capable", "do"],
            Intent::WhatHas => &["what", "has", "have", "possess", "features", "parts"],
            Intent::WhatSound => &["what", "sound", "noise", "hear", "call"],
            Intent::HowBig => &["how", "big", "large", "size", "tall", "heavy", "weight"],
            Intent::CapitalOf => &["capital", "of", "city", "capital_of"],
            Intent::TellAbout => &["tell", "me", "about", "know", "information", "facts", "everything"],
            Intent::YesNoQuestion => &["is", "are", "can", "does", "do", "will", "has"],
            Intent::Thanks => &["thank", "thanks", "appreciate", "grateful"],
            Intent::Help => &["help", "commands", "what", "can", "you", "do", "how", "use"],
            Intent::Unknown => &["unknown"],
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Intent::Greeting => "greeting",
            Intent::Farewell => "farewell",
            Intent::WhatIs => "what_is",
            Intent::WhatColor => "what_color",
            Intent::WhereIs => "where_is",
            Intent::WhatEats => "what_eats",
            Intent::WhatCan => "what_can",
            Intent::WhatHas => "what_has",
            Intent::WhatSound => "what_sound",
            Intent::HowBig => "how_big",
            Intent::CapitalOf => "capital_of",
            Intent::TellAbout => "tell_about",
            Intent::YesNoQuestion => "yes_no_question",
            Intent::Thanks => "thanks",
            Intent::Help => "help",
            Intent::Unknown => "unknown",
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Conversation Turn
// ═══════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct Turn {
    user_input: String,
    bot_response: String,
    subject: Option<String>,
}

// ═══════════════════════════════════════════════════════════════
// Knowledge Memory (VSA-backed KB)
// ═══════════════════════════════════════════════════════════════

struct KnowledgeMemory {
    /// Maps (subject, relation) → list of objects (plain text for template filling)
    facts: HashMap<(String, String), Vec<String>>,
    /// All known subjects
    subjects: HashSet<String>,
    /// All facts indexed by subject for "tell about" queries
    by_subject: HashMap<String, Vec<(String, String)>>,
    /// VSA memory bank for binding-based lookup
    memory_bank: MemoryBank,
    /// Codebook for encoding symbols
    codebook: Codebook,
}

// ═══════════════════════════════════════════════════════════════
// Chat Engine
// ═══════════════════════════════════════════════════════════════

struct ChatEngine {
    /// Intent detection: encoded intent HVs
    intent_vectors: Vec<(Intent, HyperVector)>,
    /// Codebook for intent word encoding
    intent_codebook: Codebook,
    /// Knowledge memory
    knowledge: KnowledgeMemory,
    /// Response templates by intent name
    templates: HashMap<String, Vec<String>>,
    /// Conversation history (last N turns)
    history: Vec<Turn>,
    /// Last mentioned subject (for pronoun resolution)
    last_subject: Option<String>,
    /// Stats counters
    queries_handled: usize,
    intents_detected: HashMap<String, usize>,
    /// Corpus sentences for generation fallback
    corpus_sentences: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════
// KnowledgeMemory Implementation
// ═══════════════════════════════════════════════════════════════

impl KnowledgeMemory {
    fn new() -> Self {
        let mut codebook = Codebook::with_standard_roles(DEFAULT_DIM, 0xABCD_1234_5678_9000);
        let kb_triples = corpus::knowledge_base();

        let mut facts: HashMap<(String, String), Vec<String>> = HashMap::new();
        let mut subjects: HashSet<String> = HashSet::new();
        let mut by_subject: HashMap<String, Vec<(String, String)>> = HashMap::new();
        let mut memory_bank = MemoryBank::new(DEFAULT_DIM);

        for (subj, rel, obj) in &kb_triples {
            let subj_s = subj.to_string();
            let rel_s = rel.to_string();
            let obj_s = obj.to_string();

            // Store in plain-text index
            facts.entry((subj_s.clone(), rel_s.clone()))
                .or_default()
                .push(obj_s.clone());
            subjects.insert(subj_s.clone());
            by_subject.entry(subj_s.clone())
                .or_default()
                .push((rel_s.clone(), obj_s.clone()));

            // Store in VSA memory bank
            let subj_hv = codebook.get_or_insert(&subj_s).clone();
            let rel_hv = codebook.get_or_insert(&rel_s).clone();
            let obj_hv = codebook.get_or_insert(&obj_s).clone();
            let key_hv = bind(&subj_hv, &rel_hv);
            memory_bank.store(&key_hv, &obj_hv, 1.0);
        }

        Self {
            facts,
            subjects,
            by_subject,
            memory_bank,
            codebook,
        }
    }

    /// Query KB for (subject, relation) → list of objects
    fn query(&self, subject: &str, relation: &str) -> Vec<String> {
        self.facts
            .get(&(subject.to_string(), relation.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    /// Get all facts about a subject
    fn all_facts_about(&self, subject: &str) -> Vec<(String, String)> {
        self.by_subject
            .get(subject)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if a word is a known subject
    fn is_known_subject(&self, word: &str) -> bool {
        self.subjects.contains(word)
    }

    /// Get all subjects (for display)
    fn subject_count(&self) -> usize {
        self.subjects.len()
    }

    fn fact_count(&self) -> usize {
        self.facts.values().map(|v| v.len()).sum()
    }
}

// ═══════════════════════════════════════════════════════════════
// ChatEngine Implementation
// ═══════════════════════════════════════════════════════════════

impl ChatEngine {
    fn new() -> Self {
        // Build intent codebook and intent vectors
        let mut intent_codebook = Codebook::new(DEFAULT_DIM, 0xCAFE_BEEF_1234_5678);
        let mut intent_vectors: Vec<(Intent, HyperVector)> = Vec::new();

        for intent in Intent::all() {
            let keywords = intent.keywords();
            // Encode intent as bundle of its keyword HVs
            let word_hvs: Vec<HyperVector> = keywords
                .iter()
                .map(|w| intent_codebook.get_or_insert(w).clone())
                .collect();
            let refs: Vec<&HyperVector> = word_hvs.iter().collect();
            let intent_hv = bundle(&refs).normalize();
            intent_vectors.push((*intent, intent_hv));
        }

        // Build knowledge memory
        let knowledge = KnowledgeMemory::new();

        // Build templates map
        let raw_templates = corpus::response_templates();
        let mut templates: HashMap<String, Vec<String>> = HashMap::new();
        for (name, tmpl) in raw_templates {
            templates.entry(name.to_string())
                .or_default()
                .push(tmpl.to_string());
        }

        // Load corpus for fallback generation
        let corpus_sentences: Vec<String> = corpus::large_corpus()
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        Self {
            intent_vectors,
            intent_codebook,
            knowledge,
            templates,
            history: Vec::new(),
            last_subject: None,
            queries_handled: 0,
            intents_detected: HashMap::new(),
            corpus_sentences,
        }
    }

    // ───────────────────────────────────────────────────────
    // VSA Intent Detection
    // ───────────────────────────────────────────────────────

    /// Encode user input as a hypervector (bundle of word HVs)
    fn encode_input(&mut self, input: &str) -> HyperVector {
        let words: Vec<&str> = input
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| !w.is_empty())
            .collect();

        if words.is_empty() {
            return HyperVector::zeros(DEFAULT_DIM);
        }

        let word_hvs: Vec<HyperVector> = words
            .iter()
            .map(|w| self.intent_codebook.get_or_insert(&w.to_lowercase()).clone())
            .collect();
        let refs: Vec<&HyperVector> = word_hvs.iter().collect();
        bundle(&refs).normalize()
    }

    /// Detect intent by finding nearest intent vector via cosine similarity
    fn detect_intent(&mut self, input: &str) -> (Intent, f32) {
        let input_hv = self.encode_input(input);

        let mut best_intent = Intent::Unknown;
        let mut best_sim = f32::NEG_INFINITY;

        for (intent, intent_hv) in &self.intent_vectors {
            let sim = cosine_similarity(&input_hv, intent_hv);
            if sim > best_sim {
                best_sim = sim;
                best_intent = *intent;
            }
        }

        // Apply heuristic boosters for specific patterns
        let lower = input.to_lowercase();
        let boosted = self.heuristic_boost(&lower, best_intent, best_sim);

        boosted
    }

    /// Heuristic boosters to disambiguate close intent matches
    fn heuristic_boost(&self, lower: &str, vsa_intent: Intent, vsa_sim: f32) -> (Intent, f32) {
        // Strong pattern overrides when VSA is ambiguous
        if lower.starts_with("hello") || lower.starts_with("hi ") || lower == "hi" || lower.starts_with("hey") {
            return (Intent::Greeting, vsa_sim.max(0.8));
        }
        if lower.starts_with("bye") || lower.starts_with("goodbye") || lower == "quit" || lower == "exit" {
            return (Intent::Farewell, vsa_sim.max(0.8));
        }
        if lower.starts_with("thank") {
            return (Intent::Thanks, vsa_sim.max(0.8));
        }
        if lower == "help" || lower == "commands" {
            return (Intent::Help, 1.0);
        }
        if lower.contains("tell") && lower.contains("about") {
            return (Intent::TellAbout, vsa_sim.max(0.7));
        }
        if lower.contains("capital") {
            return (Intent::CapitalOf, vsa_sim.max(0.7));
        }
        if lower.contains("color") || lower.contains("colour") {
            return (Intent::WhatColor, vsa_sim.max(0.7));
        }
        if lower.contains("where") {
            return (Intent::WhereIs, vsa_sim.max(0.6));
        }
        if lower.contains("eat") {
            return (Intent::WhatEats, vsa_sim.max(0.6));
        }
        if (lower.starts_with("what can") || lower.contains("able to")) && !lower.contains("you") {
            return (Intent::WhatCan, vsa_sim.max(0.6));
        }
        if lower.contains("what") && lower.contains("has") {
            return (Intent::WhatHas, vsa_sim.max(0.6));
        }
        if lower.contains("sound") || lower.contains("noise") {
            return (Intent::WhatSound, vsa_sim.max(0.6));
        }
        if lower.contains("how big") || lower.contains("how large") || lower.contains("how tall") {
            return (Intent::HowBig, vsa_sim.max(0.6));
        }
        if lower.starts_with("is ") || lower.starts_with("are ") || lower.starts_with("can ") || lower.starts_with("does ") {
            return (Intent::YesNoQuestion, vsa_sim.max(0.5));
        }
        if lower.starts_with("what") {
            return (Intent::WhatIs, vsa_sim.max(0.4));
        }

        (vsa_intent, vsa_sim)
    }

    // ───────────────────────────────────────────────────────
    // Subject Extraction
    // ───────────────────────────────────────────────────────

    /// Extract the subject from user input.
    /// Strategy: find words that are known subjects in KB, or remove stop words.
    fn extract_subject(&self, input: &str) -> Option<String> {
        let lower = input.to_lowercase();
        let words: Vec<&str> = lower
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_'))
            .filter(|w| !w.is_empty())
            .collect();

        // Strategy 1: Check for multi-word subjects (e.g., "mount_everest", "pacific_ocean")
        // Try joining adjacent content words with underscore
        for window in 2..=3 {
            if words.len() >= window {
                for i in 0..=(words.len() - window) {
                    let candidate = words[i..i + window].join("_");
                    if self.knowledge.is_known_subject(&candidate) {
                        return Some(candidate);
                    }
                }
            }
        }

        // Strategy 2: Find a single word that is a known subject
        let stop_set: HashSet<&str> = STOP_WORDS.iter().copied().collect();
        for word in &words {
            if !stop_set.contains(word) && self.knowledge.is_known_subject(word) {
                return Some(word.to_string());
            }
        }

        // Strategy 3: Remove stop words, take the first remaining content word
        let content_words: Vec<&str> = words
            .iter()
            .copied()
            .filter(|w| !stop_set.contains(w))
            .collect();

        // Try underscore-joined versions of remaining content words
        if content_words.len() >= 2 {
            let joined = content_words.join("_");
            if self.knowledge.is_known_subject(&joined) {
                return Some(joined);
            }
        }

        content_words.first().map(|w| w.to_string())
    }

    /// Resolve pronouns ("it", "they") to last mentioned subject
    fn resolve_pronouns(&self, input: &str) -> Option<String> {
        let lower = input.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        if words.contains(&"it") || words.contains(&"they") || words.contains(&"its") || words.contains(&"their") {
            return self.last_subject.clone();
        }
        None
    }
}

impl ChatEngine {
    // ───────────────────────────────────────────────────────
    // Knowledge Retrieval + Response Generation
    // ───────────────────────────────────────────────────────

    /// Get relevant relation for an intent
    fn intent_to_relations(&self, intent: Intent) -> Vec<&'static str> {
        match intent {
            Intent::WhatIs => vec!["is_a", "is", "defined_as"],
            Intent::WhatColor => vec!["color", "is"],
            Intent::WhereIs => vec!["located_in", "lives_in", "capital_of"],
            Intent::WhatEats => vec!["eats"],
            Intent::WhatCan => vec!["can"],
            Intent::WhatHas => vec!["has"],
            Intent::WhatSound => vec!["sound", "can"],
            Intent::HowBig => vec!["is", "size", "weight"],
            Intent::CapitalOf => vec!["capital_of"],
            _ => vec!["is", "is_a", "has", "can"],
        }
    }

    /// Retrieve knowledge based on intent and subject
    fn retrieve_knowledge(&self, subject: &str, intent: Intent) -> Vec<String> {
        let relations = self.intent_to_relations(intent);
        let mut results = Vec::new();

        for rel in &relations {
            let objects = self.knowledge.query(subject, rel);
            for obj in objects {
                results.push(obj.replace('_', " "));
            }
        }

        results
    }

    /// Retrieve ALL facts about a subject (for "tell me about")
    fn retrieve_all_facts(&self, subject: &str) -> Vec<String> {
        let facts = self.knowledge.all_facts_about(subject);
        facts
            .iter()
            .map(|(rel, obj)| {
                let rel_display = rel.replace('_', " ");
                let obj_display = obj.replace('_', " ");
                format!("{} {}", rel_display, obj_display)
            })
            .collect()
    }

    /// Fill a template with extracted information
    fn fill_template(&self, template: &str, subject: &str, objects: &[String], facts_str: &str) -> String {
        let obj_str = if objects.is_empty() {
            "unknown".to_string()
        } else {
            objects.join(", ")
        };

        template
            .replace("{subject}", &subject.replace('_', " "))
            .replace("{object}", &obj_str)
            .replace("{facts}", facts_str)
            .replace("{relation}", "is")
    }

    /// Pick the best template for an intent
    fn pick_template(&self, intent: Intent, has_data: bool) -> String {
        let intent_name = intent.name();

        // Try to find a direct match
        if let Some(tmpls) = self.templates.get(intent_name) {
            return tmpls[0].clone();
        }

        // Fallback mappings
        let key = match intent {
            Intent::Greeting => "greeting",
            Intent::Farewell => "farewell",
            Intent::Thanks => "thanks",
            Intent::Help => "help",
            Intent::WhatIs => "what_is",
            Intent::WhatColor => "what_color",
            Intent::WhereIs => "where_is",
            Intent::WhatEats => "what_does_eat",
            Intent::WhatCan => "what_can",
            Intent::WhatHas => "what_has",
            Intent::TellAbout => if has_data { "tell_about" } else { "clarify" },
            Intent::YesNoQuestion => "yes_no_affirm",
            Intent::CapitalOf => "where_is_capital",
            _ => "what_is",
        };

        if let Some(tmpls) = self.templates.get(key) {
            tmpls[0].clone()
        } else {
            "{subject} is {object}.".to_string()
        }
    }

    /// Generation fallback: find a relevant corpus sentence
    fn generation_fallback(&self, subject: &str) -> Option<String> {
        let subj_lower = subject.to_lowercase();
        // Search corpus for sentences containing the subject
        for sentence in &self.corpus_sentences {
            if sentence.to_lowercase().contains(&subj_lower) {
                return Some(format!("Based on what I know... {}", sentence));
            }
        }
        None
    }

    // ───────────────────────────────────────────────────────
    // Main Processing Pipeline
    // ───────────────────────────────────────────────────────

    /// Process a single user input and return a response
    fn process(&mut self, raw_input: &str) -> String {
        let input = raw_input.trim();
        if input.is_empty() {
            return "Please type something! Type 'help' for available commands.".to_string();
        }

        // Handle special commands
        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                return "Goodbye! Thanks for chatting.".to_string();
            }
            "help" | "commands" => {
                return self.help_text();
            }
            "stats" => {
                return self.stats_text();
            }
            _ => {}
        }

        self.queries_handled += 1;

        // Step 1: VSA Intent Detection
        let (intent, confidence) = self.detect_intent(input);
        *self.intents_detected.entry(intent.name().to_string()).or_insert(0) += 1;

        // Step 2: Handle intents that don't need subject extraction
        match intent {
            Intent::Greeting => {
                let response = self.get_greeting_response();
                self.record_turn(input, &response, None);
                return response;
            }
            Intent::Farewell => {
                let response = "Goodbye! Have a great day!".to_string();
                self.record_turn(input, &response, None);
                return response;
            }
            Intent::Thanks => {
                let response = "You're welcome! Let me know if you need anything else.".to_string();
                self.record_turn(input, &response, None);
                return response;
            }
            Intent::Help => {
                return self.help_text();
            }
            _ => {}
        }

        // Step 3: Subject Extraction (with pronoun resolution)
        let subject = self.resolve_pronouns(input)
            .or_else(|| self.extract_subject(input));

        let subject = match subject {
            Some(s) => s,
            None => {
                let response = "I'm not sure what you're asking about. Could you be more specific?".to_string();
                self.record_turn(input, &response, None);
                return response;
            }
        };

        // Step 4: Knowledge Retrieval
        let response = match intent {
            Intent::TellAbout => {
                let facts = self.retrieve_all_facts(&subject);
                if facts.is_empty() {
                    self.try_fallback(&subject)
                } else {
                    let facts_str = facts.join("; ");
                    let template = self.pick_template(intent, true);
                    self.fill_template(&template, &subject, &[], &facts_str)
                }
            }
            Intent::CapitalOf => {
                // Special: user might ask "capital of France" → look up where subject=paris, rel=capital_of
                // Or look for (X, capital_of, subject)
                let result = self.find_capital(&subject);
                if let Some(capital) = result {
                    format!("The capital of {} is {}.", subject.replace('_', " "), capital.replace('_', " "))
                } else {
                    self.try_fallback(&subject)
                }
            }
            Intent::YesNoQuestion => {
                self.handle_yes_no(input, &subject)
            }
            _ => {
                let objects = self.retrieve_knowledge(&subject, intent);
                if objects.is_empty() {
                    self.try_fallback(&subject)
                } else {
                    let template = self.pick_template(intent, true);
                    self.fill_template(&template, &subject, &objects, "")
                }
            }
        };

        // Step 5: Record in conversation memory
        let response = self.smooth_response(&response);
        let response = self.context_response(&subject, &response);
        self.record_turn(input, &response, Some(subject.clone()));
        self.last_subject = Some(subject);

        // Append confidence if relevant
        if confidence < INTENT_CONFIDENCE_THRESHOLD && intent != Intent::Unknown {
            format!("{} (confidence: {:.0}%)", response, confidence * 100.0)
        } else {
            response
        }
    }

    /// Record a turn in conversation history
    fn record_turn(&mut self, input: &str, response: &str, subject: Option<String>) {
        self.history.push(Turn {
            user_input: input.to_string(),
            bot_response: response.to_string(),
            subject,
        });
        if self.history.len() > MAX_CONVERSATION_HISTORY {
            self.history.remove(0);
        }
    }

    /// Find capital of a country (reverse lookup)
    fn find_capital(&self, country: &str) -> Option<String> {
        // Direct lookup: check if any city has capital_of = country
        let kb = corpus::knowledge_base();
        for (subj, rel, obj) in kb {
            if rel == "capital_of" && obj.to_lowercase() == country.to_lowercase() {
                return Some(subj.to_string());
            }
            if rel == "capital_of" && subj.to_lowercase() == country.to_lowercase() {
                return Some(obj.to_string());
            }
        }
        None
    }

    /// Handle yes/no questions
    fn handle_yes_no(&self, input: &str, subject: &str) -> String {
        let lower = input.to_lowercase();

        // Try to determine what's being asked
        // "Is a cat an animal?" → check (cat, is_a, animal)
        // "Can a dog swim?" → check (dog, can, swim)
        let all_facts = self.knowledge.all_facts_about(subject);

        // Check for "cannot" facts FIRST (e.g., "can a penguin fly?" → No)
        let cannot_facts = self.knowledge.query(subject, "cannot");
        for obj in &cannot_facts {
            let obj_clean = obj.replace('_', " ").to_lowercase();
            if lower.contains(&obj_clean) || lower.contains(obj) {
                return format!("No, {} cannot {}.", subject.replace('_', " "), obj_clean);
            }
        }

        // Check if any fact matches words in the question
        for (rel, obj) in &all_facts {
            if rel == "cannot" {
                continue; // Already handled above
            }
            let obj_clean = obj.replace('_', " ").to_lowercase();
            if lower.contains(&obj_clean) || lower.contains(&obj.to_lowercase()) {
                let subj_display = subject.replace('_', " ");
                let obj_display = obj.replace('_', " ");
                return format!("Yes, {} {} {}.", subj_display, rel.replace('_', " "), obj_display);
            }
        }

        // Can't determine — try inference
        let lower_words: Vec<&str> = lower.split_whitespace().collect();
        let possible_property = lower_words.last().unwrap_or(&"");
        if let Some(inferred) = self.infer_yes_no(subject, possible_property) {
            return inferred;
        }

        if !all_facts.is_empty() {
            format!("I'm not certain about that specific question, but I know that {} {}.",
                subject.replace('_', " "),
                all_facts[0].1.replace('_', " "))
        } else {
            "I don't have enough information to answer that yes/no question.".to_string()
        }
    }

    /// Fallback when KB doesn't have the answer — now with INFERENCE CHAINS
    fn try_fallback(&self, subject: &str) -> String {
        // Step 1: Try inference chain (transitive reasoning)
        if let Some(inferred) = self.infer_chain(subject) {
            return inferred;
        }
        // Step 2: Try corpus-based generation
        if let Some(fallback) = self.generation_fallback(subject) {
            return fallback;
        }
        format!("I don't have specific information about {}. Try asking about animals, geography, science, or food!", subject.replace('_', " "))
    }

    // ═══════════════════════════════════════════════════════════
    // INFERENCE CHAINS (Transitive Reasoning)
    // ═══════════════════════════════════════════════════════════

    /// Perform transitive inference: if X is_a Y, and Y has Z → X has Z
    /// Supports multiple hops (up to 3) through the knowledge graph.
    ///
    /// Examples:
    /// - "does a cat have a heart?" → cat is_a animal, animal has heart → YES
    /// - "can a penguin swim?" → penguin is_a bird... but penguin can swim (direct)
    fn infer_chain(&self, subject: &str) -> Option<String> {
        // Get parent categories via is_a
        let parents = self.knowledge.query(subject, "is_a");
        if parents.is_empty() {
            return None;
        }

        // For each parent, check what properties it has
        for parent in &parents {
            let parent_facts = self.knowledge.all_facts_about(parent);
            if !parent_facts.is_empty() {
                // Collect inherited properties
                let mut inherited = Vec::new();
                for (rel, obj) in &parent_facts {
                    if rel == "is_a" { continue; } // Don't chain is_a → is_a endlessly
                    // Check if subject already has this directly
                    let direct = self.knowledge.query(subject, rel);
                    if direct.is_empty() {
                        inherited.push((rel.clone(), obj.clone(), parent.clone()));
                    }
                }

                if !inherited.is_empty() {
                    // Format: "Based on reasoning: X is a Y, and Ys have Z, so X likely has Z."
                    let first = &inherited[0];
                    let subj_d = subject.replace('_', " ");
                    let parent_d = parent.replace('_', " ");
                    let obj_d = first.1.replace('_', " ");
                    let rel_d = first.0.replace('_', " ");

                    let mut response = format!(
                        "Based on reasoning: {} is a {}, and {}s {} {}, so {} likely {} {} too.",
                        subj_d, parent_d, parent_d, rel_d, obj_d, subj_d, rel_d, obj_d
                    );

                    // Add more inherited facts if available
                    if inherited.len() > 1 {
                        let extras: Vec<String> = inherited[1..inherited.len().min(4)]
                            .iter()
                            .map(|(r, o, _)| format!("{} {}", r.replace('_', " "), o.replace('_', " ")))
                            .collect();
                        response.push_str(&format!(" Also: {}.", extras.join(", ")));
                    }

                    return Some(response);
                }
            }

            // Hop 2: grandparent reasoning
            let grandparents = self.knowledge.query(parent, "is_a");
            for gp in &grandparents {
                let gp_facts = self.knowledge.all_facts_about(gp);
                for (rel, obj) in &gp_facts {
                    if rel == "is_a" { continue; }
                    let direct = self.knowledge.query(subject, rel);
                    if direct.is_empty() {
                        let subj_d = subject.replace('_', " ");
                        let parent_d = parent.replace('_', " ");
                        let gp_d = gp.replace('_', " ");
                        let obj_d = obj.replace('_', " ");
                        let rel_d = rel.replace('_', " ");
                        return Some(format!(
                            "Based on reasoning: {} is a {}, {} is a {}, and {}s {} {}. So {} likely {} {} too.",
                            subj_d, parent_d, parent_d, gp_d, gp_d, rel_d, obj_d, subj_d, rel_d, obj_d
                        ));
                    }
                }
            }
        }

        None
    }

    /// Yes/No with inference support
    fn infer_yes_no(&self, subject: &str, property: &str) -> Option<String> {
        // Direct check
        let all_facts = self.knowledge.all_facts_about(subject);
        for (rel, obj) in &all_facts {
            let obj_clean = obj.replace('_', " ").to_lowercase();
            if obj_clean.contains(property) || property.contains(&obj_clean) {
                return Some(format!("Yes, {} {} {}.",
                    subject.replace('_', " "), rel.replace('_', " "), obj.replace('_', " ")));
            }
        }

        // Inference via is_a chain
        let parents = self.knowledge.query(subject, "is_a");
        for parent in &parents {
            let parent_facts = self.knowledge.all_facts_about(&parent);
            for (rel, obj) in &parent_facts {
                let obj_clean = obj.replace('_', " ").to_lowercase();
                if obj_clean.contains(property) || property.contains(&obj_clean) {
                    return Some(format!(
                        "Yes, because {} is a {}, and {}s {} {}.",
                        subject.replace('_', " "), parent.replace('_', " "),
                        parent.replace('_', " "), rel.replace('_', " "), obj.replace('_', " ")
                    ));
                }
            }
        }

        None
    }

    // ═══════════════════════════════════════════════════════════
    // FLUENCY SMOOTHING
    // ═══════════════════════════════════════════════════════════

    /// Smooth a response to read more naturally.
    /// Adds articles, capitalizes, fixes grammar patterns.
    fn smooth_response(&self, raw: &str) -> String {
        let mut result = raw.to_string();

        // Fix common grammar issues
        result = result.replace("animals has", "animals have");
        result = result.replace("animals needs", "animals need");
        result = result.replace("animals can", "animals can");
        result = result.replace("birds has", "birds have");
        result = result.replace("dogs has", "dogs have");
        result = result.replace("cats has", "cats have");
        result = result.replace("a animal", "an animal");
        result = result.replace("a elephant", "an elephant");
        result = result.replace("a eagle", "an eagle");
        result = result.replace("a ocean", "an ocean");
        result = result.replace("is a is", "is");
        result = result.replace("_", " ");

        // Capitalize first letter
        if let Some(first) = result.chars().next() {
            if first.is_lowercase() {
                result = first.to_uppercase().to_string() + &result[1..];
            }
        }

        // Add articles before singular nouns that need them
        let needs_article = ["cat", "dog", "bird", "fish", "horse", "elephant",
                            "snake", "whale", "penguin", "eagle", "shark",
                            "tree", "house", "car", "book", "star", "river",
                            "mountain", "flower", "person", "child", "baby"];

        for noun in &needs_article {
            // "cat is" → "A cat is"
            let pattern = format!("{} is", noun);
            let replacement = if starts_with_vowel(noun) {
                format!("an {} is", noun)
            } else {
                format!("a {} is", noun)
            };
            if result.starts_with(&pattern) {
                result = result.replacen(&pattern, &replacement, 1);
            }

            // "cat eats" → "A cat eats"
            let pattern2 = format!("{} eats", noun);
            let replacement2 = if starts_with_vowel(noun) {
                format!("an {} eats", noun)
            } else {
                format!("a {} eats", noun)
            };
            result = result.replacen(&pattern2, &replacement2, 1);

            // "cat can" → "A cat can"
            let pattern3 = format!("{} can", noun);
            let replacement3 = if starts_with_vowel(noun) {
                format!("an {} can", noun)
            } else {
                format!("a {} can", noun)
            };
            result = result.replacen(&pattern3, &replacement3, 1);

            // "cat has" → "A cat has"
            let pattern4 = format!("{} has", noun);
            let replacement4 = if starts_with_vowel(noun) {
                format!("an {} has", noun)
            } else {
                format!("a {} has", noun)
            };
            result = result.replacen(&pattern4, &replacement4, 1);
        }

        // Fix double articles
        result = result.replace("A a ", "A ").replace("a a ", "a ");
        result = result.replace("An an ", "An ").replace("an an ", "an ");

        // Ensure ends with period if not question/exclamation
        let trimmed = result.trim_end();
        if !trimmed.ends_with('.') && !trimmed.ends_with('!') && !trimmed.ends_with('?') {
            result = format!("{}.", trimmed);
        }

        // Re-capitalize after period
        let mut chars: Vec<char> = result.chars().collect();
        let mut capitalize_next = true;
        for c in chars.iter_mut() {
            if capitalize_next && c.is_alphabetic() {
                *c = c.to_uppercase().next().unwrap_or(*c);
                capitalize_next = false;
            }
            if *c == '.' || *c == '!' || *c == '?' {
                capitalize_next = true;
            }
        }
        result = chars.into_iter().collect();

        result
    }

    // ═══════════════════════════════════════════════════════════
    // CONTEXT-AWARE RESPONSES
    // ═══════════════════════════════════════════════════════════

    /// Generate a context-aware response that references previous conversation.
    fn context_response(&self, subject: &str, current_response: &str) -> String {
        if self.history.len() < 2 {
            return current_response.to_string();
        }

        // Check if previous turn was about a related subject
        let prev_subject = self.history.last()
            .and_then(|t| t.subject.as_ref());

        if let Some(prev_subj) = prev_subject {
            if prev_subj != subject {
                // Compare the two subjects
                let comparison = self.compare_subjects(prev_subj, subject);
                if let Some(comp) = comparison {
                    return format!("{} {}", current_response, comp);
                }
            }
        }

        current_response.to_string()
    }

    /// Compare two subjects and find differences/similarities.
    fn compare_subjects(&self, subj_a: &str, subj_b: &str) -> Option<String> {
        let facts_a = self.knowledge.all_facts_about(subj_a);
        let facts_b = self.knowledge.all_facts_about(subj_b);

        if facts_a.is_empty() || facts_b.is_empty() {
            return None;
        }

        // Find shared relations with different values
        let mut comparisons = Vec::new();
        for (rel_a, obj_a) in &facts_a {
            for (rel_b, obj_b) in &facts_b {
                if rel_a == rel_b && obj_a != obj_b && rel_a != "is_a" {
                    comparisons.push(format!(
                        "Unlike {}, which {} {}, {} {} {}",
                        subj_a.replace('_', " "), rel_a.replace('_', " "), obj_a.replace('_', " "),
                        subj_b.replace('_', " "), rel_b.replace('_', " "), obj_b.replace('_', " ")
                    ));
                }
            }
        }

        // Find shared properties
        let mut shared = Vec::new();
        for (rel_a, obj_a) in &facts_a {
            for (rel_b, obj_b) in &facts_b {
                if rel_a == rel_b && obj_a == obj_b {
                    shared.push(format!("both {} {}", rel_a.replace('_', " "), obj_a.replace('_', " ")));
                }
            }
        }

        if !comparisons.is_empty() {
            Some(format!("By the way, {}.", comparisons[0]))
        } else if !shared.is_empty() {
            Some(format!("Interestingly, {} and {} {}.",
                subj_a.replace('_', " "), subj_b.replace('_', " "), shared[0]))
        } else {
            None
        }
    }

    /// Get a greeting response
    fn get_greeting_response(&self) -> String {
        if let Some(tmpls) = self.templates.get("greeting") {
            tmpls[0].clone()
        } else {
            "Hello! How can I help you today?".to_string()
        }
    }

    /// Help text
    fn help_text(&self) -> String {
        "TLE-Chat v2 - VSA-Powered Knowledge Engine\n\
         ═══════════════════════════════════════════\n\
         I can answer questions about:\n\
         • Animals (cats, dogs, elephants, birds, fish, etc.)\n\
         • Geography (capitals, countries, rivers, mountains)\n\
         • Science (water, earth, sun, elements, physics)\n\
         • Food (fruits, grains, how things are made)\n\
         • Technology (computers, internet, robots)\n\
         • Properties (colors, sizes, textures)\n\n\
         Example questions:\n\
         • \"What does a cat eat?\"\n\
         • \"Tell me about elephants\"\n\
         • \"What is the capital of France?\"\n\
         • \"Where is Japan located?\"\n\
         • \"Can a penguin fly?\"\n\
         • \"What color is a banana?\"\n\n\
         Commands: help, stats, quit\n\
         Pronouns: use 'it'/'they' to refer to the last topic."
            .to_string()
    }

    /// Stats text
    fn stats_text(&self) -> String {
        let mem_stats = self.knowledge.memory_bank.stats();
        format!(
            "TLE-Chat v2 Statistics\n\
             ═══════════════════════\n\
             Knowledge base: {} subjects, {} facts\n\
             VSA dimensions: {}\n\
             Memory bank SNR: {:.1}\n\
             Queries handled: {}\n\
             Conversation turns: {}\n\
             Intent distribution: {:?}",
            self.knowledge.subject_count(),
            self.knowledge.fact_count(),
            DEFAULT_DIM,
            mem_stats.estimated_snr,
            self.queries_handled,
            self.history.len(),
            self.intents_detected,
        )
    }
}

// ═══════════════════════════════════════════════════════════════
// Main Entry Point
// ═══════════════════════════════════════════════════════════════

/// Check if a word starts with a vowel (for a/an article selection).
fn starts_with_vowel(word: &str) -> bool {
    matches!(word.chars().next(), Some('a' | 'e' | 'i' | 'o' | 'u'))
}

fn main() {
    eprintln!("TLE-Chat v2 — Topological Latent Engine");
    eprintln!("Initializing VSA knowledge base...");

    let mut engine = ChatEngine::new();

    eprintln!(
        "Ready! {} subjects, {} facts loaded into {} dimensions.",
        engine.knowledge.subject_count(),
        engine.knowledge.fact_count(),
        DEFAULT_DIM
    );
    eprintln!("Type 'help' for commands, 'quit' to exit.\n");

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout_lock = stdout.lock();
    let is_tty = atty_check();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };

        let input = line.trim().to_string();
        if input.is_empty() {
            if is_tty {
                let _ = write!(stdout_lock, "> ");
                let _ = stdout_lock.flush();
            }
            continue;
        }

        let response = engine.process(&input);
        let _ = writeln!(stdout_lock, "{}", response);
        let _ = stdout_lock.flush();

        // Exit on farewell
        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" || input.to_lowercase() == "q" {
            break;
        }

        if is_tty {
            let _ = write!(stdout_lock, "\n> ");
            let _ = stdout_lock.flush();
        }
    }
}

/// Simple check if stdin is likely a TTY (heuristic: not piped)
fn atty_check() -> bool {
    // If we're being piped input, don't print prompts
    // A simple heuristic: check if stdin is a terminal
    unsafe { libc_isatty(0) != 0 }
}

// Minimal FFI to check if stdin is a tty without adding a dependency
extern "C" {
    #[link_name = "isatty"]
    fn libc_isatty(fd: i32) -> i32;
}
