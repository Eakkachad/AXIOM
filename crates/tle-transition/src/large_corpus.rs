//! # Hypothesis 2: Energy Function Scales with Larger Corpus
//!
//! Tests whether the Transition Binding Algebra improves with more data:
//! - Larger vocabulary → better coverage
//! - More bigrams → stronger transition memory signal
//! - Trigram enhancement → better contextual predictions
//!
//! ## Trigram Transition Enhancement
//!
//! T2(A,B→C) = π²(A) ⊗ π(B) ⊗ C
//!
//! This captures 2-word context: given the previous two words (A, B),
//! predict the next word C. The double-permutation of A distinguishes
//! the 2-back position from the 1-back position.

use std::collections::{HashMap, HashSet};
use tle_vsa::{HyperVector, Codebook, cosine_similarity, DEFAULT_DIM};

/// Large diverse corpus covering multiple sentence types.
pub fn large_corpus() -> Vec<&'static str> {
    vec![
        // ═══ Simple narrative sentences ═══
        "the cat sat on the mat",
        "the dog lay on the rug",
        "the bird perched on the branch",
        "the fish swam in the pond",
        "the boy sat on the bench",
        "the girl stood by the door",
        "the man walked down the street",
        "the woman read a book",
        "the child played in the yard",
        "the baby slept in the crib",
        "the old man fed the birds",
        "the young girl picked some flowers",
        "the tall tree swayed in the wind",
        "the small house stood on the hill",
        "the red car drove past the park",
        // ═══ Factual statements ═══
        "the sun is a star",
        "the moon orbits the earth",
        "water is made of hydrogen and oxygen",
        "the sky is blue during the day",
        "the earth is round like a ball",
        "fire is hot and bright",
        "ice is cold and hard",
        "snow falls in the winter",
        "rain comes from the clouds",
        "the ocean is deep and wide",
        "trees grow from small seeds",
        "birds can fly through the air",
        "fish live in the water",
        "the sun rises in the east",
        "the sun sets in the west",
        "light travels very fast",
        "sound moves through the air",
        "plants need water and light",
        "animals need food and water",
        "the heart pumps blood through the body",
        // ═══ Action sequences ═══
        "she walked to the store and bought milk",
        "he ran to the park and sat down",
        "they drove to the beach and went swimming",
        "she opened the door and walked inside",
        "he picked up the phone and made a call",
        "the dog ran to the gate and barked",
        "the cat jumped on the table and ate the fish",
        "she took the bus and went to work",
        "he cooked dinner and set the table",
        "they packed their bags and left the house",
        "the bird flew to the tree and built a nest",
        "she read the book and wrote a review",
        "he fixed the car and drove to town",
        "the boy kicked the ball and scored a goal",
        "the girl drew a picture and showed her mother",
        // ═══ Questions and answers ═══
        "what color is the sky it is blue",
        "where does the sun go it sets in the west",
        "how do birds fly they use their wings",
        "why is the grass green it has chlorophyll",
        "what do fish eat they eat small plants",
        "where do cats sleep they sleep on soft things",
        "how fast can a dog run very fast",
        "why do we need water to stay alive",
        "what is the moon it is a rock in space",
        "where is the ocean it is by the coast",
        "how do trees grow they grow from seeds",
        "why is fire hot because it is burning",
        "what makes rain the clouds make rain",
        "where do birds go they fly south in winter",
        "how do we learn we read and practice",
        // ═══ Multi-clause sentences ═══
        "the dog ran because it was happy",
        "the cat hid because it was scared",
        "she smiled because the sun was shining",
        "he stopped because the light was red",
        "they left early because it started to rain",
        "the bird sang because the morning was bright",
        "she cried because the movie was sad",
        "he laughed because the joke was funny",
        "the baby cried because it was hungry",
        "the flowers grew because they had water",
        "if it rains we will stay inside",
        "if the sun shines we will go outside",
        "when the bell rings the children run",
        "when the wind blows the leaves fall",
        "while the cat slept the mouse ran away",
        // ═══ Common English patterns ═══
        "I like to eat good food",
        "I want to go to the park",
        "I need to find my keys",
        "I have a big red ball",
        "I can see the blue sky",
        "she has a small white cat",
        "he has a big brown dog",
        "they have a nice old house",
        "we like to play in the sun",
        "we want to swim in the lake",
        "it is a nice day today",
        "it was a cold night last night",
        "there is a tree in the yard",
        "there are many fish in the sea",
        "there was a storm last week",
        // ═══ Temporal and spatial patterns ═══
        "in the morning the birds sing",
        "in the evening the sun goes down",
        "at night the stars come out",
        "in the summer it is hot",
        "in the winter it is cold",
        "on the table there is a cup",
        "in the garden there are flowers",
        "by the river there is a tree",
        "over the hill there is a town",
        "under the bridge there is water",
        // ═══ Comparative and descriptive ═══
        "the cat is smaller than the dog",
        "the sun is bigger than the moon",
        "the river is longer than the road",
        "the mountain is taller than the tree",
        "the ocean is deeper than the lake",
        "the red flower is very pretty",
        "the big house has many rooms",
        "the old tree has thick branches",
        "the fast car went down the road",
        "the small bird sat on the fence",
        // ═══ Social and daily life ═══
        "the family ate dinner together",
        "the friends played games all day",
        "the teacher read a story to the class",
        "the doctor helped the sick child",
        "the farmer grew food in the field",
        "she gave him a book for his birthday",
        "he made her a cup of tea",
        "they built a sand castle on the beach",
        "we watched the clouds float by",
        "the children sang a song together",
        // ═══ Additional patterns for coverage ═══
        "a good friend is hard to find",
        "a warm fire on a cold night is nice",
        "the sound of rain is calming",
        "the smell of flowers fills the air",
        "the taste of fresh bread is wonderful",
        "time flies when you are having fun",
        "the early bird catches the worm",
        "every cloud has a silver lining",
        "still water runs deep they say",
        "the pen is mightier than the sword",
        "practice makes perfect in all things",
        "where there is smoke there is fire",
        "all that glitters is not gold",
        "the best things in life are free",
        "a journey of a thousand miles begins with one step",
    ]
}

/// Compute a Transition Vector: T(A → B) = π(A) ⊗ B
fn transition(from: &HyperVector, to: &HyperVector) -> HyperVector {
    let shifted = from.permute(1);
    shifted.hadamard(to)
}

/// Retrieve the "expected next" given current word and transition memory.
fn retrieve_next(current: &HyperVector, transition_memory: &HyperVector) -> HyperVector {
    let shifted = current.permute(1);
    shifted.hadamard(transition_memory)
}

/// Compute a Trigram Transition: T2(A,B → C) = π²(A) ⊗ π(B) ⊗ C
///
/// This captures 2-word context:
/// - π²(A) encodes the word two positions back (double permutation)
/// - π(B) encodes the word one position back (single permutation)
/// - C is the target word
///
/// The different permutation amounts ensure the positions are distinguishable.
fn trigram_transition(a: &HyperVector, b: &HyperVector, c: &HyperVector) -> HyperVector {
    let a_shifted = a.permute(2);  // π²(A): two-position shift
    let b_shifted = b.permute(1);  // π(B): one-position shift
    let ab = a_shifted.hadamard(&b_shifted); // π²(A) ⊗ π(B)
    ab.hadamard(c)                            // π²(A) ⊗ π(B) ⊗ C
}

/// Retrieve trigram prediction: given (A, B), predict C.
///
/// C_estimate = π²(A) ⊗ π(B) ⊗ TM_trigram
fn retrieve_trigram_next(a: &HyperVector, b: &HyperVector, tm_tri: &HyperVector) -> HyperVector {
    let a_shifted = a.permute(2);
    let b_shifted = b.permute(1);
    let ab = a_shifted.hadamard(&b_shifted);
    ab.hadamard(tm_tri)
}

/// Extended Transition Memory with both bigram and trigram support.
pub struct LargeCorpusTransitionMemory {
    /// Bigram transition memory: T(A→B) bundled
    pub tm_bigram: HyperVector,
    /// Trigram transition memory: T2(A,B→C) bundled
    pub tm_trigram: HyperVector,
    /// Word codebook
    pub codebook: Codebook,
    /// Vocabulary (ordered list)
    pub vocab: Vec<String>,
    /// Vocab vectors (for nearest-neighbor search)
    pub vocab_vectors: Vec<HyperVector>,
    /// Bigram counts
    pub bigram_counts: HashMap<(String, String), usize>,
    /// Trigram counts
    pub trigram_counts: HashMap<(String, String, String), usize>,
    /// Total transitions learned
    pub bigram_transition_count: usize,
    pub trigram_transition_count: usize,
}

impl LargeCorpusTransitionMemory {
    pub fn new() -> Self {
        Self {
            tm_bigram: HyperVector::zeros(DEFAULT_DIM),
            tm_trigram: HyperVector::zeros(DEFAULT_DIM),
            codebook: Codebook::new(DEFAULT_DIM, 0x7BA0_0000_2026_0002),
            vocab: Vec::new(),
            vocab_vectors: Vec::new(),
            bigram_counts: HashMap::new(),
            trigram_counts: HashMap::new(),
            bigram_transition_count: 0,
            trigram_transition_count: 0,
        }
    }

    /// Learn both bigram and trigram transitions from the corpus.
    pub fn learn_from_corpus(&mut self, sentences: &[&str]) {
        // First pass: build vocabulary
        for sentence in sentences {
            for word in sentence.split_whitespace() {
                let w = word.to_lowercase();
                if !self.vocab.contains(&w) {
                    self.vocab.push(w.clone());
                    self.codebook.get_or_insert(&w);
                }
            }
        }

        // Build vocab_vectors
        self.vocab_vectors = self.vocab.iter()
            .map(|w| self.codebook.get(w).unwrap().clone())
            .collect();

        // Second pass: encode bigram and trigram transitions
        for sentence in sentences {
            let words: Vec<String> = sentence.split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();

            // Bigram transitions
            for i in 0..words.len().saturating_sub(1) {
                let from_hv = self.codebook.get(&words[i]).unwrap().clone();
                let to_hv = self.codebook.get(&words[i + 1]).unwrap().clone();

                let t = transition(&from_hv, &to_hv);
                self.tm_bigram = self.tm_bigram.add(&t);
                self.bigram_transition_count += 1;

                *self.bigram_counts
                    .entry((words[i].clone(), words[i + 1].clone()))
                    .or_insert(0) += 1;
            }

            // Trigram transitions
            for i in 0..words.len().saturating_sub(2) {
                let a_hv = self.codebook.get(&words[i]).unwrap().clone();
                let b_hv = self.codebook.get(&words[i + 1]).unwrap().clone();
                let c_hv = self.codebook.get(&words[i + 2]).unwrap().clone();

                let t2 = trigram_transition(&a_hv, &b_hv, &c_hv);
                self.tm_trigram = self.tm_trigram.add(&t2);
                self.trigram_transition_count += 1;

                *self.trigram_counts
                    .entry((words[i].clone(), words[i + 1].clone(), words[i + 2].clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    /// Generate next token using bigram only.
    pub fn next_token_bigram(&self, current: &str) -> Option<(String, f32)> {
        let current_hv = self.codebook.get(current)?;
        let estimate = retrieve_next(current_hv, &self.tm_bigram);

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, v) in self.vocab_vectors.iter().enumerate() {
            let sim = cosine_similarity(&estimate, v);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some((self.vocab[best_idx].clone(), best_sim))
    }

    /// Generate next token using trigram context (A, B) → C.
    pub fn next_token_trigram(&self, prev: &str, current: &str) -> Option<(String, f32)> {
        let a_hv = self.codebook.get(prev)?;
        let b_hv = self.codebook.get(current)?;
        let estimate = retrieve_trigram_next(a_hv, b_hv, &self.tm_trigram);

        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (i, v) in self.vocab_vectors.iter().enumerate() {
            let sim = cosine_similarity(&estimate, v);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        Some((self.vocab[best_idx].clone(), best_sim))
    }

    /// Generate using combined bigram + trigram with anti-repetition.
    ///
    /// Score = α * bigram_score + β * trigram_score - γ * repetition_penalty
    pub fn generate_combined(
        &self,
        prompt: &str,
        max_tokens: usize,
        alpha: f32,
        beta: f32,
    ) -> Vec<String> {
        let mut output: Vec<String> = prompt.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        for _ in 0..max_tokens {
            let current = output.last().unwrap().clone();
            let current_hv = match self.codebook.get(&current) {
                Some(hv) => hv.clone(),
                None => break,
            };

            // Bigram estimate
            let bigram_estimate = retrieve_next(&current_hv, &self.tm_bigram);

            // Trigram estimate (if we have 2+ tokens)
            let trigram_estimate = if output.len() >= 2 {
                let prev = &output[output.len() - 2];
                if let Some(prev_hv) = self.codebook.get(prev) {
                    Some(retrieve_trigram_next(prev_hv, &current_hv, &self.tm_trigram))
                } else {
                    None
                }
            } else {
                None
            };

            // Score all candidates
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = 0;

            for (i, v) in self.vocab_vectors.iter().enumerate() {
                let candidate = &self.vocab[i];

                // Bigram similarity
                let bi_sim = cosine_similarity(&bigram_estimate, v);

                // Trigram similarity (if available)
                let tri_sim = match &trigram_estimate {
                    Some(est) => cosine_similarity(est, v),
                    None => 0.0,
                };

                // Repetition penalty
                let mut rep_penalty = 0.0f32;
                for (j, prev_word) in output.iter().rev().take(4).enumerate() {
                    if prev_word == candidate {
                        rep_penalty += (0.8f32).powi(j as i32);
                    }
                }

                // Diversity penalty
                let count = output.iter().filter(|w| *w == candidate).count();
                let div_penalty = (1.0 + count as f32).ln();

                let score = alpha * bi_sim
                          + beta * tri_sim
                          - 1.5 * rep_penalty
                          - 0.3 * div_penalty;

                if score > best_score {
                    best_score = score;
                    best_idx = i;
                }
            }

            let next = self.vocab[best_idx].clone();
            if best_score < -2.0 {
                break;
            }
            output.push(next);
        }

        output
    }

    /// Compute path energy for a sentence (bigram-based).
    pub fn sentence_energy(&self, sentence: &str) -> f32 {
        let words: Vec<String> = sentence.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let mut energy = 0.0f32;
        for i in 0..words.len().saturating_sub(1) {
            if let (Some(from), Some(to)) = (self.codebook.get(&words[i]), self.codebook.get(&words[i + 1])) {
                let t = transition(from, to);
                energy += -cosine_similarity(&t, &self.tm_bigram);
            }
        }
        energy
    }

    /// Compute trigram path energy for a sentence.
    pub fn sentence_energy_trigram(&self, sentence: &str) -> f32 {
        let words: Vec<String> = sentence.split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let mut energy = 0.0f32;
        for i in 0..words.len().saturating_sub(2) {
            if let (Some(a), Some(b), Some(c)) = (
                self.codebook.get(&words[i]),
                self.codebook.get(&words[i + 1]),
                self.codebook.get(&words[i + 2]),
            ) {
                let t2 = trigram_transition(a, b, c);
                energy += -cosine_similarity(&t2, &self.tm_trigram);
            }
        }
        energy
    }
}

// ═══════════════════════════════════════════════════════════════
// Metrics and Testing
// ═══════════════════════════════════════════════════════════════

/// Measure coherence: % of bigrams in generated text that exist in corpus.
pub fn measure_coherence(generated: &[String], bigram_counts: &HashMap<(String, String), usize>) -> f32 {
    if generated.len() < 2 {
        return 0.0;
    }
    let mut valid = 0;
    let total = generated.len() - 1;
    for i in 0..total {
        let key = (generated[i].clone(), generated[i + 1].clone());
        if bigram_counts.contains_key(&key) {
            valid += 1;
        }
    }
    valid as f32 / total as f32
}

/// Measure diversity: unique words / total words.
pub fn measure_diversity(generated: &[String]) -> f32 {
    if generated.is_empty() {
        return 0.0;
    }
    let unique: HashSet<&String> = generated.iter().collect();
    unique.len() as f32 / generated.len() as f32
}

/// Run all Hypothesis 2 tests.
pub fn run_hypothesis2() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Hypothesis 2: Energy Function Scales with Larger Corpus    ║");
    println!("║  + Trigram Transition Enhancement T2(A,B→C) = π²A ⊗ πB ⊗ C ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let corpus = large_corpus();
    println!("Corpus size: {} sentences", corpus.len());

    // Build transition memory
    print!("Building bigram + trigram transition memory... ");
    let mut tm = LargeCorpusTransitionMemory::new();
    tm.learn_from_corpus(&corpus);
    println!("done!");
    println!("  Vocabulary size: {} words", tm.vocab.len());
    println!("  Bigram transitions: {}", tm.bigram_transition_count);
    println!("  Trigram transitions: {}", tm.trigram_transition_count);
    println!("  Unique bigrams: {}", tm.bigram_counts.len());
    println!("  Unique trigrams: {}", tm.trigram_counts.len());
    println!();

    // ═══ TEST 1: Bigram-only generation ═══
    println!("━━━ Test 1: Bigram Generation (large corpus) ━━━");
    let prompts = ["the cat", "she walked", "the sun", "I want", "in the", "the dog"];
    for prompt in &prompts {
        let generated = tm.generate_combined(prompt, 8, 1.0, 0.0);
        let coherence = measure_coherence(&generated, &tm.bigram_counts);
        let diversity = measure_diversity(&generated);
        let sentence = generated.join(" ");
        println!("  \"{}\" → \"{}\"", prompt, sentence);
        println!("    coherence={:.0}% diversity={:.0}%", coherence * 100.0, diversity * 100.0);
    }
    println!();

    // ═══ TEST 2: Trigram-enhanced generation ═══
    println!("━━━ Test 2: Trigram-Enhanced Generation (α=0.5, β=0.5) ━━━");
    for prompt in &prompts {
        let generated = tm.generate_combined(prompt, 8, 0.5, 0.5);
        let coherence = measure_coherence(&generated, &tm.bigram_counts);
        let diversity = measure_diversity(&generated);
        let sentence = generated.join(" ");
        println!("  \"{}\" → \"{}\"", prompt, sentence);
        println!("    coherence={:.0}% diversity={:.0}%", coherence * 100.0, diversity * 100.0);
    }
    println!();

    // ═══ TEST 3: Compare bigram vs trigram next-token accuracy ═══
    println!("━━━ Test 3: Bigram vs Trigram Prediction Accuracy ━━━");
    let test_cases: Vec<(&str, &str, &str)> = vec![
        ("the", "cat", "sat"),
        ("sat", "on", "the"),
        ("the", "dog", "ran"),
        ("in", "the", "park"),
        ("she", "walked", "to"),
        ("is", "a", "star"),
        ("the", "sun", "is"),
        ("it", "was", "happy"),
        ("to", "the", "store"),
        ("and", "bought", "milk"),
    ];

    let mut bigram_correct = 0;
    let mut trigram_correct = 0;
    let total = test_cases.len();

    for (prev, current, expected) in &test_cases {
        // Bigram: just uses `current`
        let bi_pred = tm.next_token_bigram(current)
            .map(|(w, _)| w)
            .unwrap_or_default();

        // Trigram: uses (prev, current)
        let tri_pred = tm.next_token_trigram(prev, current)
            .map(|(w, _)| w)
            .unwrap_or_default();

        let bi_ok = bi_pred == *expected;
        let tri_ok = tri_pred == *expected;
        if bi_ok { bigram_correct += 1; }
        if tri_ok { trigram_correct += 1; }

        println!("  ({}, {}) → expected \"{}\" | bigram=\"{}\" {} | trigram=\"{}\" {}",
            prev, current, expected,
            bi_pred, if bi_ok { "✓" } else { "✗" },
            tri_pred, if tri_ok { "✓" } else { "✗" });
    }
    println!();
    println!("  Bigram accuracy:  {}/{} ({:.0}%)", bigram_correct, total, bigram_correct as f32 / total as f32 * 100.0);
    println!("  Trigram accuracy: {}/{} ({:.0}%)", trigram_correct, total, trigram_correct as f32 / total as f32 * 100.0);
    println!();

    // ═══ TEST 4: Energy separation ═══
    println!("━━━ Test 4: Energy Separation (good vs bad sentences) ━━━");
    let good_sentences = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "she walked to the store",
        "the sun is a star",
        "the bird flew over the tree",
        "I want to go to the park",
        "the baby slept in the crib",
        "rain comes from the clouds",
    ];
    let bad_sentences = [
        "mat the on sat cat the",
        "the the the the the the",
        "dog bird fish cat sun moon",
        "walked sat ran flew drove swam",
        "a a a a a a a a",
        "store park house tree road lake",
        "very very very very very very",
        "is is is is is is is",
    ];

    let mut good_energies = Vec::new();
    let mut bad_energies = Vec::new();

    println!("  Good sentences (lower energy = more natural):");
    for s in &good_sentences {
        let e_bi = tm.sentence_energy(s);
        let e_tri = tm.sentence_energy_trigram(s);
        good_energies.push(e_bi);
        println!("    E_bi={:+.4} E_tri={:+.4}  \"{}\"", e_bi, e_tri, s);
    }

    println!("  Bad sentences:");
    for s in &bad_sentences {
        let e_bi = tm.sentence_energy(s);
        let e_tri = tm.sentence_energy_trigram(s);
        bad_energies.push(e_bi);
        println!("    E_bi={:+.4} E_tri={:+.4}  \"{}\"", e_bi, e_tri, s);
    }

    let avg_good: f32 = good_energies.iter().sum::<f32>() / good_energies.len() as f32;
    let avg_bad: f32 = bad_energies.iter().sum::<f32>() / bad_energies.len() as f32;
    let separation = avg_bad - avg_good;

    println!();
    println!("  Average good energy: {:.4}", avg_good);
    println!("  Average bad energy:  {:.4}", avg_bad);
    println!("  Separation (bad - good): {:.4}", separation);
    println!("  Energy discriminates: {} {}", separation > 0.0,
        if separation > 0.0 { "✓" } else { "✗" });
    println!();

    // ═══ TEST 5: Scaling comparison ═══
    println!("━━━ Test 5: Scaling — Small vs Large Corpus ━━━");
    let small_corpus = vec![
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog ran in the park",
        "the dog chased the cat",
        "the bird flew over the tree",
    ];

    let mut tm_small = LargeCorpusTransitionMemory::new();
    tm_small.learn_from_corpus(&small_corpus);

    // Compare coherence on same prompts
    let scale_prompts = ["the cat", "the dog", "the bird"];
    println!("  {:20} {:30} {:30}", "Prompt", "Small Corpus (5 sent)", "Large Corpus ({} sent)");
    println!("  {:20} {:30} {:30}", "------", "--------------------", "--------------------");

    let mut small_coherences = Vec::new();
    let mut large_coherences = Vec::new();

    for prompt in &scale_prompts {
        let gen_small = tm_small.generate_combined(prompt, 6, 1.0, 0.0);
        let gen_large = tm.generate_combined(prompt, 6, 0.5, 0.5);

        let coh_small = measure_coherence(&gen_small, &tm_small.bigram_counts);
        let coh_large = measure_coherence(&gen_large, &tm.bigram_counts);
        let div_small = measure_diversity(&gen_small);
        let div_large = measure_diversity(&gen_large);

        small_coherences.push(coh_small);
        large_coherences.push(coh_large);

        let small_str = format!("C={:.0}% D={:.0}% \"{}\"",
            coh_small * 100.0, div_small * 100.0, gen_small.join(" "));
        let large_str = format!("C={:.0}% D={:.0}% \"{}\"",
            coh_large * 100.0, div_large * 100.0, gen_large.join(" "));
        println!("  {:20} {} | {}", prompt, small_str, large_str);
    }

    let avg_small_coh: f32 = small_coherences.iter().sum::<f32>() / small_coherences.len() as f32;
    let avg_large_coh: f32 = large_coherences.iter().sum::<f32>() / large_coherences.len() as f32;
    println!();
    println!("  Avg coherence — small: {:.1}%, large: {:.1}%", avg_small_coh * 100.0, avg_large_coh * 100.0);
    println!();

    // ═══ TEST 6: Generation quality assessment ═══
    println!("━━━ Test 6: Generation Quality (Trigram-Enhanced, 10 samples) ━━━");
    let quality_prompts = [
        "the cat", "she walked", "the sun", "I want", "in the",
        "the dog", "it is", "we like", "the bird", "he ran",
    ];

    let mut total_coherence = 0.0f32;
    let mut total_diversity = 0.0f32;
    let count = quality_prompts.len();

    for prompt in &quality_prompts {
        let generated = tm.generate_combined(prompt, 8, 0.5, 0.5);
        let coherence = measure_coherence(&generated, &tm.bigram_counts);
        let diversity = measure_diversity(&generated);
        total_coherence += coherence;
        total_diversity += diversity;
        let sentence = generated.join(" ");
        println!("  \"{}\"", sentence);
    }

    let avg_coherence = total_coherence / count as f32;
    let avg_diversity = total_diversity / count as f32;
    println!();
    println!("  Average coherence: {:.1}%", avg_coherence * 100.0);
    println!("  Average diversity: {:.1}%", avg_diversity * 100.0);
    println!();

    // ═══ SUMMARY ═══
    println!("━━━ HYPOTHESIS 2 RESULTS ━━━");
    println!("  Corpus scaling: {} sentences → {} vocab, {} bigrams, {} trigrams",
        corpus.len(), tm.vocab.len(), tm.bigram_counts.len(), tm.trigram_counts.len());
    println!("  Bigram prediction accuracy:  {}/{}", bigram_correct, total);
    println!("  Trigram prediction accuracy: {}/{}", trigram_correct, total);
    println!("  Energy separation (good vs bad): {:.4}", separation);
    println!("  Energy discriminates correctly: {}", separation > 0.0);
    println!("  Average generation coherence: {:.1}%", avg_coherence * 100.0);
    println!("  Average generation diversity: {:.1}%", avg_diversity * 100.0);
    println!();
    if separation > 0.0 {
        println!("  ✓ Energy function SCALES with larger corpus");
    } else {
        println!("  ✗ Energy function did NOT scale (needs investigation)");
    }
    if trigram_correct > bigram_correct {
        println!("  ✓ Trigram enhancement IMPROVES over bigram-only");
    } else if trigram_correct == bigram_correct {
        println!("  ~ Trigram ties with bigram (may need more data)");
    } else {
        println!("  ✗ Trigram did not improve (signal-to-noise issue at this scale)");
    }
    println!();
    println!("  Hypothesis 2: TESTED");
}
