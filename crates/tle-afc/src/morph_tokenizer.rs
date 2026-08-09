//! VSA Morphological Tokenization — Novel subword composition via algebra.
//!
//! Instead of BPE/SentencePiece (requires training), we decompose words
//! into morphemes (prefix + root + suffix) and COMPOSE their vectors
//! algebraically using positional binding.
//!
//! ## Core Formula
//!
//! ```text
//! encode("unbelievable") = C("un") ⊙ ρ⁰(C("believe")) ⊙ ρ¹(C("able"))
//! encode("running")      = C("run") ⊙ ρ⁰(C("ing"))
//! encode("cats")         = C("cat") ⊙ ρ⁰(C("s"))
//! ```
//!
//! ## Properties
//! - "unbelievable" is SIMILAR to "unforgettable" (shared "un" prefix)
//! - "running" is SIMILAR to "swimming" (shared "ing" suffix)
//! - 5K roots + 200 affixes → covers 100K+ words
//! - No training — uses morphological rules + VSA composition
//! - Handles OOV gracefully (decompose novel words algebraically)

use tle_vsa::{HyperVector, Codebook};

/// Morpheme types.
#[derive(Debug, Clone, PartialEq)]
pub enum MorphemeType {
    Prefix,
    Root,
    Suffix,
}

/// A decomposed morpheme.
#[derive(Debug, Clone)]
pub struct Morpheme {
    pub text: String,
    pub mtype: MorphemeType,
}

/// Common English prefixes (sorted by length descending for longest-match).
const PREFIXES: &[&str] = &[
    "counter", "under", "over", "super", "inter", "trans", "multi",
    "semi", "anti", "auto", "mega", "micro", "mini", "mono",
    "non", "pre", "post", "pro", "sub", "mis", "dis", "out",
    "un", "re", "in", "im", "ir", "il", "en", "em",
];

/// Common English suffixes (sorted by length descending).
const SUFFIXES: &[&str] = &[
    "ization", "fulness", "ousness", "iveness", "ibility",
    "ation", "ition", "ment", "ness", "able", "ible", "ting",
    "ous", "ive", "ful", "less", "ism", "ist", "ize",
    "ing", "tion", "sion", "ence", "ance", "ity",
    "ly", "er", "or", "ed", "en", "al", "es", "s",
];

/// VSA Morphological Tokenizer.
pub struct MorphTokenizer {
    /// Minimum root length after stripping affixes.
    pub min_root_len: usize,
}

impl MorphTokenizer {
    /// Create a new morphological tokenizer.
    pub fn new() -> Self {
        Self { min_root_len: 3 }
    }

    /// Decompose a word into morphemes.
    ///
    /// Uses longest-match for prefix and suffix, with minimum root constraint.
    pub fn decompose(&self, word: &str) -> Vec<Morpheme> {
        let lower = word.to_lowercase();

        if lower.len() <= self.min_root_len {
            return vec![Morpheme { text: lower, mtype: MorphemeType::Root }];
        }

        let mut prefix: Option<&str> = None;
        let mut suffix: Option<&str> = None;

        // Find longest matching prefix
        for &p in PREFIXES {
            if lower.starts_with(p) && lower.len() - p.len() >= self.min_root_len {
                prefix = Some(p);
                break;
            }
        }

        // Find longest matching suffix
        let after_prefix = prefix.map(|p| &lower[p.len()..]).unwrap_or(&lower);
        for &s in SUFFIXES {
            if after_prefix.ends_with(s) && after_prefix.len() - s.len() >= self.min_root_len {
                suffix = Some(s);
                break;
            }
        }

        // Build morpheme list
        let mut morphemes = Vec::new();

        let start = prefix.map(|p| p.len()).unwrap_or(0);
        let end = lower.len() - suffix.map(|s| s.len()).unwrap_or(0);

        if let Some(p) = prefix {
            morphemes.push(Morpheme { text: p.to_string(), mtype: MorphemeType::Prefix });
        }

        morphemes.push(Morpheme {
            text: lower[start..end].to_string(),
            mtype: MorphemeType::Root,
        });

        if let Some(s) = suffix {
            morphemes.push(Morpheme { text: s.to_string(), mtype: MorphemeType::Suffix });
        }

        morphemes
    }

    /// Encode a word as a composed VSA vector.
    ///
    /// Uses ADDITIVE composition (bundling) so that shared morphemes
    /// contribute directly to cosine similarity.
    ///
    /// Formula: encode(word) = Σᵢ ρⁱ(C(morpheme_i))
    ///
    /// This means words sharing a morpheme will have cos > 0.
    pub fn encode(&self, word: &str, codebook: &mut Codebook) -> HyperVector {
        let morphemes = self.decompose(word);

        if morphemes.len() == 1 {
            return codebook.get_or_insert(&morphemes[0].text).clone();
        }

        // Additive composition: Σᵢ ρⁱ(C(morpheme_i))
        // This ensures shared morphemes contribute to similarity
        let first = codebook.get_or_insert(&morphemes[0].text).clone();
        let mut composed = first;
        for (i, morph) in morphemes.iter().enumerate().skip(1) {
            let morph_vec = codebook.get_or_insert(&morph.text).clone();
            let shifted = morph_vec.permute(i as i32);
            composed = composed.add(&shifted);
        }

        composed
    }

    /// Encode a sentence (split into words, encode each, bundle).
    pub fn encode_sentence(&self, sentence: &str, codebook: &mut Codebook) -> HyperVector {
        let words: Vec<&str> = sentence.split_whitespace().collect();
        if words.is_empty() {
            let dim = 2048; // default
            return HyperVector::zeros(dim);
        }

        let mut result = self.encode(words[0], codebook);
        for (i, word) in words.iter().enumerate().skip(1) {
            let word_vec = self.encode(word, codebook);
            let positional = word_vec.permute(i as i32);
            result = result.add(&positional);
        }

        result
    }
}

impl Default for MorphTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::cosine_similarity;

    #[test]
    fn test_decompose_prefix_suffix() {
        let tok = MorphTokenizer::new();

        let morphemes = tok.decompose("unbelievable");
        assert_eq!(morphemes.len(), 3);
        assert_eq!(morphemes[0].text, "un");
        assert_eq!(morphemes[0].mtype, MorphemeType::Prefix);
        assert_eq!(morphemes[1].text, "believ");
        assert_eq!(morphemes[1].mtype, MorphemeType::Root);
        assert_eq!(morphemes[2].text, "able");
        assert_eq!(morphemes[2].mtype, MorphemeType::Suffix);
    }

    #[test]
    fn test_decompose_suffix_only() {
        let tok = MorphTokenizer::new();

        let morphemes = tok.decompose("running");
        assert_eq!(morphemes.len(), 2);
        assert_eq!(morphemes[0].text, "runn");
        assert_eq!(morphemes[0].mtype, MorphemeType::Root);
        assert_eq!(morphemes[1].text, "ing");
        assert_eq!(morphemes[1].mtype, MorphemeType::Suffix);
    }

    #[test]
    fn test_decompose_short_word() {
        let tok = MorphTokenizer::new();
        let morphemes = tok.decompose("cat");
        assert_eq!(morphemes.len(), 1);
        assert_eq!(morphemes[0].text, "cat");
    }

    #[test]
    fn test_similar_prefixes() {
        let tok = MorphTokenizer::new();
        let mut codebook = Codebook::new(2048, 42);

        let v1 = tok.encode("unbelievable", &mut codebook);
        let v2 = tok.encode("unforgettable", &mut codebook);
        let v3 = tok.encode("basketball", &mut codebook);

        let sim_un = cosine_similarity(&v1, &v2);
        let sim_diff = cosine_similarity(&v1, &v3);

        // Words sharing "un" prefix should be more similar than unrelated words
        assert!(
            sim_un > sim_diff,
            "un-words should be more similar: un={:.3} vs diff={:.3}",
            sim_un, sim_diff
        );
    }

    #[test]
    fn test_similar_suffixes() {
        let tok = MorphTokenizer::new();
        let mut codebook = Codebook::new(2048, 42);

        let v1 = tok.encode("running", &mut codebook);
        let v2 = tok.encode("swimming", &mut codebook);
        let v3 = tok.encode("unbelievable", &mut codebook);

        let sim_ing = cosine_similarity(&v1, &v2);
        let sim_diff = cosine_similarity(&v1, &v3);

        // Words sharing "ing" suffix should be more similar
        assert!(
            sim_ing > sim_diff,
            "ing-words should be more similar: ing={:.3} vs diff={:.3}",
            sim_ing, sim_diff
        );
    }

    #[test]
    fn test_encode_deterministic() {
        let tok = MorphTokenizer::new();
        let mut codebook = Codebook::new(2048, 42);

        let v1 = tok.encode("programming", &mut codebook);
        let v2 = tok.encode("programming", &mut codebook);

        assert_eq!(v1, v2, "Same word must produce same vector");
    }

    #[test]
    fn test_thai_token_is_preserved_as_root() {
        let tok = MorphTokenizer::new();
        let morphemes = tok.decompose("แมว");
        assert_eq!(morphemes.len(), 1);
        assert_eq!(morphemes[0].text, "แมว");
        assert_eq!(morphemes[0].mtype, MorphemeType::Root);
    }

    #[test]
    fn test_mixed_thai_english_sentence_is_deterministic() {
        let tok = MorphTokenizer::new();
        let mut codebook = Codebook::new(2048, 42);
        let first = tok.encode_sentence("แมว is running", &mut codebook);
        let second = tok.encode_sentence("แมว is running", &mut codebook);
        assert_eq!(first, second);
        assert_eq!(first.dim(), 2048);
    }
}
