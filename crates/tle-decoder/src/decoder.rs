//! Latent Decoder: Converts hypervectors to English tokens.

use tle_vsa::{Codebook, HyperVector, similarity};
use tle_resonator::{ResonatorNetwork, ResonatorConfig, CleanupRule};
use tle_memory::MemoryBank;

/// Result of decoding a latent vector.
#[derive(Clone, Debug)]
pub struct DecodedToken {
    /// The decoded English token/word.
    pub token: String,
    /// Confidence of the decoding (cosine similarity to codebook entry).
    pub confidence: f32,
    /// Index in the codebook.
    pub codebook_index: usize,
}

/// The Latent-to-English Decoder.
///
/// Performs deterministic (zero-sampling) conversion from latent space
/// to discrete tokens via codebook nearest-neighbor lookup.
pub struct LatentDecoder {
    /// The vocabulary codebook mapping tokens ↔ hypervectors.
    codebook: Codebook,
    /// Resonator for pre-cleanup before decoding.
    resonator: ResonatorNetwork,
    /// Ordered list of token strings matching codebook order.
    token_list: Vec<String>,
    /// Ordered list of token vectors (for fast batch lookup).
    vector_list: Vec<HyperVector>,
}

impl LatentDecoder {
    /// Create a decoder with a pre-built codebook.
    pub fn new(mut codebook: Codebook, vocabulary: &[&str]) -> Self {
        let mut token_list = Vec::new();
        let mut vector_list = Vec::new();

        // Ensure all vocabulary items are in the codebook
        for &word in vocabulary {
            let hv = codebook.get_or_insert(word).clone();
            token_list.push(word.to_string());
            vector_list.push(hv);
        }

        let config = ResonatorConfig {
            max_iterations: 20,
            epsilon: 1e-6,
            cleanup_rule: CleanupRule::Sign,
            temperature: 1.0,
        };
        let mut resonator = ResonatorNetwork::with_config(config);
        resonator.set_codebook(vector_list.clone());

        Self {
            codebook,
            resonator,
            token_list,
            vector_list,
        }
    }

    /// Decode a single latent vector to its nearest token.
    ///
    /// This is the zero-sampling decoding: deterministic argmax over
    /// cosine similarity to all codebook entries.
    pub fn decode(&self, latent: &HyperVector) -> DecodedToken {
        // First apply sign cleanup to sharpen
        let cleaned = latent.sign();

        // Find nearest codebook entry
        let (idx, sim) = similarity::nearest_in_codebook(&cleaned, &self.vector_list);

        DecodedToken {
            token: self.token_list[idx].clone(),
            confidence: sim,
            codebook_index: idx,
        }
    }

    /// Decode a sequence of latent vectors.
    pub fn decode_sequence(&self, latents: &[HyperVector]) -> Vec<DecodedToken> {
        latents.iter().map(|l| self.decode(l)).collect()
    }

    /// Decode to a string (concatenating tokens with spaces).
    pub fn decode_to_string(&self, latents: &[HyperVector]) -> String {
        let tokens = self.decode_sequence(latents);
        tokens
            .iter()
            .map(|t| t.token.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Encode a string into latent hypervectors.
    /// Splits on whitespace and looks up each token.
    pub fn encode(&mut self, text: &str) -> Vec<HyperVector> {
        text.split_whitespace()
            .map(|word| self.codebook.get_or_insert(word).clone())
            .collect()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.token_list.len()
    }

    /// Check if a token is in the vocabulary.
    pub fn contains(&self, token: &str) -> bool {
        self.token_list.iter().any(|t| t == token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tle_vsa::DEFAULT_DIM;

    fn test_vocabulary() -> Vec<&'static str> {
        vec![
            "the", "cat", "sat", "on", "mat", "dog", "ran", "fast",
            "a", "is", "was", "big", "small", "red", "blue",
            "I", "you", "we", "they", "it", "he", "she",
            "love", "hate", "see", "hear", "think", "know",
            "good", "bad", "happy", "sad", "new", "old",
        ]
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let codebook = Codebook::default_params();
        let vocab = test_vocabulary();
        let decoder = LatentDecoder::new(codebook, &vocab);

        // Encode "the cat"
        let the_vec = decoder.vector_list[0].clone(); // "the"
        let cat_vec = decoder.vector_list[1].clone(); // "cat"

        // Decode back
        let decoded_the = decoder.decode(&the_vec);
        let decoded_cat = decoder.decode(&cat_vec);

        assert_eq!(decoded_the.token, "the");
        assert_eq!(decoded_cat.token, "cat");
        assert!((decoded_the.confidence - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_decode_sequence() {
        let codebook = Codebook::default_params();
        let vocab = test_vocabulary();
        let decoder = LatentDecoder::new(codebook, &vocab);

        let sequence = vec![
            decoder.vector_list[0].clone(), // "the"
            decoder.vector_list[1].clone(), // "cat"
            decoder.vector_list[2].clone(), // "sat"
        ];

        let result = decoder.decode_to_string(&sequence);
        assert_eq!(result, "the cat sat");
    }

    #[test]
    fn test_noisy_decode() {
        let codebook = Codebook::default_params();
        let vocab = test_vocabulary();
        let decoder = LatentDecoder::new(codebook, &vocab);

        // Add noise to "cat" vector
        let cat_vec = decoder.vector_list[1].clone();
        let noise = HyperVector::random_bipolar(DEFAULT_DIM, 999);
        let noisy_cat = cat_vec.add(&noise.scale(0.3)); // 30% noise

        let decoded = decoder.decode(&noisy_cat);
        // Sign cleanup should still recover "cat" due to majority voting
        assert_eq!(decoded.token, "cat", "Should decode noisy vector as 'cat'");
    }

    #[test]
    fn test_deterministic_decode() {
        let codebook = Codebook::default_params();
        let vocab = test_vocabulary();
        let decoder = LatentDecoder::new(codebook, &vocab);

        let test_vec = HyperVector::random_bipolar(DEFAULT_DIM, 12345);

        let d1 = decoder.decode(&test_vec);
        let d2 = decoder.decode(&test_vec);

        assert_eq!(d1.token, d2.token);
        assert_eq!(d1.confidence, d2.confidence);
        assert_eq!(d1.codebook_index, d2.codebook_index);
    }
}
