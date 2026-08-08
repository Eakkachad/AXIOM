//! VSA Intent Detection — Algebraic semantic matching (replaces rule-based keywords).
//!
//! Instead of: `if input.starts_with("why") → Intent::Why`
//! We do: `cos(input_vec, why_prototype) > cos(input_vec, what_prototype) → Intent::Why`
//!
//! This handles:
//! - "tell me the reason" → Why (no keyword "why" present!)
//! - "what's the process" → How (semantic, not keyword)
//! - Typos and variations gracefully

use tle_vsa::{cosine_similarity, HyperVector, Codebook};

/// Intent types (same as before, but detected algebraically).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsaIntent {
    Why,
    What,
    How,
    Where,
    When,
    Who,
    YesNo,
    Greeting,
    Thanks,
    Command,
    Declarative,
}

/// Prototype words for each intent — bundled into a single vector per intent.
const WHY_WORDS: &[&str] = &["why", "reason", "cause", "because", "explain", "how come"];
const WHAT_WORDS: &[&str] = &["what", "define", "meaning", "describe", "tell", "about"];
const HOW_WORDS: &[&str] = &["how", "method", "process", "way", "steps", "procedure"];
const WHERE_WORDS: &[&str] = &["where", "location", "place", "located", "find", "position"];
const WHEN_WORDS: &[&str] = &["when", "time", "date", "year", "period", "during"];
const WHO_WORDS: &[&str] = &["who", "person", "people", "someone", "name"];
const YESNO_WORDS: &[&str] = &["is", "are", "does", "do", "can", "will", "could", "should"];
const GREETING_WORDS: &[&str] = &["hello", "hi", "hey", "greetings", "howdy", "morning"];
const THANKS_WORDS: &[&str] = &["thank", "thanks", "appreciate", "grateful", "cheers"];

/// VSA-based intent detector.
pub struct VsaIntentDetector {
    /// Prototype vectors for each intent.
    prototypes: Vec<(VsaIntent, HyperVector)>,
    /// Confidence threshold — below this, classify as Declarative.
    pub threshold: f32,
}

impl VsaIntentDetector {
    /// Build the intent detector from a codebook.
    ///
    /// Creates prototype vectors by bundling related words for each intent.
    pub fn build(codebook: &mut Codebook) -> Self {
        let mut prototypes = Vec::new();

        prototypes.push((VsaIntent::Why, Self::bundle_words(WHY_WORDS, codebook)));
        prototypes.push((VsaIntent::What, Self::bundle_words(WHAT_WORDS, codebook)));
        prototypes.push((VsaIntent::How, Self::bundle_words(HOW_WORDS, codebook)));
        prototypes.push((VsaIntent::Where, Self::bundle_words(WHERE_WORDS, codebook)));
        prototypes.push((VsaIntent::When, Self::bundle_words(WHEN_WORDS, codebook)));
        prototypes.push((VsaIntent::Who, Self::bundle_words(WHO_WORDS, codebook)));
        prototypes.push((VsaIntent::YesNo, Self::bundle_words(YESNO_WORDS, codebook)));
        prototypes.push((VsaIntent::Greeting, Self::bundle_words(GREETING_WORDS, codebook)));
        prototypes.push((VsaIntent::Thanks, Self::bundle_words(THANKS_WORDS, codebook)));

        Self {
            prototypes,
            threshold: 0.08,
        }
    }

    /// Detect intent from input text.
    ///
    /// Encodes input as VSA vector, finds highest cosine to intent prototypes.
    pub fn detect(&self, input: &str, codebook: &mut Codebook) -> (VsaIntent, f32) {
        let input_vec = Self::encode_input(input, codebook);

        let mut best_intent = VsaIntent::Declarative;
        let mut best_score = f32::NEG_INFINITY;

        for (intent, proto) in &self.prototypes {
            let sim = cosine_similarity(&input_vec, proto);
            if sim > best_score {
                best_score = sim;
                best_intent = *intent;
            }
        }

        if best_score < self.threshold {
            return (VsaIntent::Declarative, best_score);
        }

        (best_intent, best_score)
    }

    /// Encode input text as a bundled VSA vector.
    fn encode_input(input: &str, codebook: &mut Codebook) -> HyperVector {
        let lower = input.to_lowercase();
        let words: Vec<&str> = lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .collect();

        if words.is_empty() {
            let dim = codebook.get_or_insert("_empty_").dim();
            return HyperVector::zeros(dim);
        }

        // Bundle all words (additive superposition)
        let mut result = codebook.get_or_insert(words[0]).clone();
        for &word in &words[1..] {
            let wv = codebook.get_or_insert(word).clone();
            result = result.add(&wv);
        }

        result
    }

    /// Bundle a list of words into a single prototype vector.
    fn bundle_words(words: &[&str], codebook: &mut Codebook) -> HyperVector {
        let first = codebook.get_or_insert(words[0]).clone();
        let mut result = first;
        for &word in &words[1..] {
            let wv = codebook.get_or_insert(word).clone();
            result = result.add(&wv);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_why() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        let (intent, _) = detector.detect("why is the sky blue", &mut codebook);
        assert_eq!(intent, VsaIntent::Why);
    }

    #[test]
    fn test_detect_why_no_keyword() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        // "reason" is a WHY prototype word — should detect Why
        let (intent, _) = detector.detect("tell me the reason for rain", &mut codebook);
        assert_eq!(intent, VsaIntent::Why);
    }

    #[test]
    fn test_detect_what() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        let (intent, _) = detector.detect("what is a computer", &mut codebook);
        assert_eq!(intent, VsaIntent::What);
    }

    #[test]
    fn test_detect_greeting() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        let (intent, _) = detector.detect("hello there", &mut codebook);
        assert_eq!(intent, VsaIntent::Greeting);
    }

    #[test]
    fn test_detect_how() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        let (intent, _) = detector.detect("how does this work", &mut codebook);
        assert_eq!(intent, VsaIntent::How);
    }

    #[test]
    fn test_detect_yesno() {
        let mut codebook = Codebook::new(2048, 42);
        let detector = VsaIntentDetector::build(&mut codebook);

        let (intent, _) = detector.detect("can elephants swim", &mut codebook);
        assert_eq!(intent, VsaIntent::YesNo);
    }
}
