//! δ-Mem — Online Associative Memory for Conversation Context
//!
//! Based on Research 24 (Delta-Mem Online Associative Memory):
//!   S_t = λ·S_{t-1} − β·(S_{t-1}·k_t)⊗k_t + β·v_t⊗k_t
//!
//! This gives AXIOM **conversation memory** — it tracks what's been discussed
//! and uses it to influence responses (pronoun resolution, topic continuity).
//!
//! ## Key Properties
//! - Updates every turn (no batch processing)
//! - Decays old context (λ < 1 → recent topics matter more)
//! - Delta rule: corrects errors (subtracts wrong prediction, adds correct value)
//! - Zero training — pure algebraic state update

use tle_vsa::HyperVector;

/// Delta Memory state — online associative memory for conversation context.
///
/// Maintains a low-rank state matrix as a single vector (simplified to D-dim)
/// that tracks the current conversation topic/context.
pub struct DeltaMem {
    /// Current memory state (D-dimensional context vector).
    pub state: HyperVector,
    /// Retention gate (how much to keep from previous state). 0 < λ < 1.
    pub lambda: f32,
    /// Write gate (how aggressively to update). 0 < β < 1.
    pub beta: f32,
    /// History of topics discussed (for pronoun resolution).
    topic_stack: Vec<String>,
    /// Last subject mentioned (for "it", "they" resolution).
    pub last_subject: Option<String>,
    /// Turn counter.
    pub turn: usize,
    /// Dimensionality.
    dim: usize,
}

impl DeltaMem {
    /// Create a new delta memory with given dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            state: HyperVector::zeros(dim),
            lambda: 0.85,
            beta: 0.3,
            topic_stack: Vec::new(),
            last_subject: None,
            turn: 0,
            dim,
        }
    }

    /// Create with custom gates.
    pub fn with_gates(dim: usize, lambda: f32, beta: f32) -> Self {
        Self {
            state: HyperVector::zeros(dim),
            lambda,
            beta,
            topic_stack: Vec::new(),
            last_subject: None,
            turn: 0,
            dim,
        }
    }

    /// Update memory with a new key-value observation.
    ///
    /// The delta rule:
    ///   prediction = state · key (dot product → scalar, used to scale)
    ///   error = value - prediction_scaled
    ///   state = λ·state + β·error
    ///
    /// Simplified for D-dimensional vectors:
    ///   state = λ·state - β·(sim(state,key))·key + β·value
    pub fn update(&mut self, key: &HyperVector, value: &HyperVector) {
        // Compute prediction: how much does current state predict this key?
        let state_norm = self.state.norm();
        let prediction_strength = if state_norm > 0.001 {
            self.state.dot(key) / (state_norm * key.norm().max(0.001))
        } else {
            0.0
        };

        // Delta update: decay old + remove wrong prediction + add new value
        let decayed = self.state.scale(self.lambda);
        let remove_old = key.scale(self.beta * prediction_strength);
        let add_new = value.scale(self.beta);

        self.state = decayed.sub(&remove_old).add(&add_new);
        self.turn += 1;
    }

    /// Update with a topic string (converts to vector via simple hash encoding).
    pub fn update_topic(&mut self, topic: &str) {
        // Track subject for pronoun resolution
        let clean = topic.to_lowercase()
            .replace('?', "")
            .replace('!', "")
            .replace('.', "");
        let words: Vec<&str> = clean.split_whitespace().collect();

        // Extract the main subject (first noun-like word > 2 chars)
        for word in &words {
            if word.len() > 2 && !is_stop_word(word) {
                self.last_subject = Some(word.to_string());
                break;
            }
        }

        // Push to topic stack (keep last 10)
        self.topic_stack.push(clean);
        if self.topic_stack.len() > 10 {
            self.topic_stack.remove(0);
        }
    }

    /// Resolve pronouns: "it", "they", "that" → last mentioned subject.
    pub fn resolve_pronoun(&self, input: &str) -> String {
        let lower = input.to_lowercase();

        if let Some(ref subject) = self.last_subject {
            let resolved = lower
                .replace(" it ", &format!(" {} ", subject))
                .replace(" they ", &format!(" {} ", subject))
                .replace(" that ", &format!(" {} ", subject))
                .replace("it ", &format!("{} ", subject))
                .replace("they ", &format!("{} ", subject));

            if resolved != lower {
                return resolved;
            }
        }

        input.to_string()
    }

    /// Check if a word was recently discussed.
    pub fn is_recent_topic(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.topic_stack.iter().rev().take(5).any(|t| t.contains(&lower))
    }

    /// Get the last N topics discussed.
    pub fn recent_topics(&self, n: usize) -> Vec<&str> {
        self.topic_stack.iter().rev().take(n).map(|s| s.as_str()).collect()
    }

    /// Get conversation context summary.
    pub fn context_summary(&self) -> String {
        if self.topic_stack.is_empty() {
            return "No conversation context yet.".to_string();
        }
        let topics: Vec<&str> = self.topic_stack.iter().rev().take(3).map(|s| s.as_str()).collect();
        format!("Recent topics: {}", topics.join(", "))
    }

    /// Reset memory state (new conversation).
    pub fn reset(&mut self) {
        self.state = HyperVector::zeros(self.dim);
        self.topic_stack.clear();
        self.last_subject = None;
        self.turn = 0;
    }
}

/// Check if a word is a stop word (not a useful subject).
fn is_stop_word(word: &str) -> bool {
    matches!(
        word,
        "the" | "a" | "an" | "is" | "are" | "was" | "were" | "be" | "been"
            | "have" | "has" | "had" | "do" | "does" | "did" | "will" | "would"
            | "can" | "could" | "may" | "might" | "shall" | "should" | "must"
            | "and" | "or" | "but" | "not" | "no" | "yes" | "in" | "on" | "at"
            | "to" | "for" | "of" | "with" | "from" | "by" | "about" | "into"
            | "what" | "who" | "where" | "when" | "why" | "how" | "which"
            | "this" | "that" | "these" | "those" | "it" | "they" | "them"
            | "i" | "you" | "he" | "she" | "we" | "my" | "your" | "his" | "her"
            | "its" | "our" | "their" | "me" | "him" | "us" | "tell"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pronoun_resolution() {
        let mut mem = DeltaMem::new(64);
        mem.update_topic("tell me about elephants");

        assert_eq!(mem.last_subject, Some("elephants".to_string()));

        let resolved = mem.resolve_pronoun("can they fly?");
        assert!(resolved.contains("elephants"));
    }

    #[test]
    fn test_topic_tracking() {
        let mut mem = DeltaMem::new(64);
        mem.update_topic("cats are cute");
        mem.update_topic("dogs are loyal");
        mem.update_topic("birds can fly");

        assert!(mem.is_recent_topic("cats"));
        assert!(mem.is_recent_topic("dogs"));
        assert!(!mem.is_recent_topic("fish"));
    }

    #[test]
    fn test_reset() {
        let mut mem = DeltaMem::new(64);
        mem.update_topic("hello world");
        assert_eq!(mem.turn, 0); // update_topic doesn't increment turn
        mem.reset();
        assert!(mem.topic_stack.is_empty());
        assert!(mem.last_subject.is_none());
    }
}
