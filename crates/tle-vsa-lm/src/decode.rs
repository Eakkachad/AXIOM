//! VSA decoder: convert a prediction vector into a ranked list of tokens.
//!
//! The decoder is the replacement for the neural readout (`W_out · state`
//! followed by softmax). It measures cosine similarity between the prediction
//! vector and every codebook vector, then returns the top-k by similarity.
//! No softmax, no learned weights — pure algebraic lookup.

use tle_vsa::{cosine_similarity, HyperVector};

use crate::vocab::Vocab;

/// A ranked decoding result.
#[derive(Debug, Clone)]
pub struct DecodedToken {
    pub id: usize,
    pub word: String,
    pub similarity: f32,
}

/// Rank all vocabulary tokens by cosine similarity to `prediction`.
///
/// Optionally applies a `penalty` closure that subtracts a per-token penalty
/// (e.g. anti-repetition) from the similarity before ranking.
pub fn decode_topk(
    vocab: &Vocab,
    prediction: &HyperVector,
    k: usize,
    penalty: Option<&dyn Fn(usize) -> f32>,
) -> Vec<DecodedToken> {
    let mut ranked: Vec<DecodedToken> = vocab
        .iter()
        .map(|(id, word)| {
            let sim = cosine_similarity(prediction, vocab.vector_by_id(id).unwrap());
            let effective = sim - penalty.map(|p| p(id)).unwrap_or(0.0);
            DecodedToken { id, word: word.to_string(), similarity: effective }
        })
        .collect();
    ranked.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(k);
    ranked
}

/// Best single token for a prediction vector.
pub fn decode_best(vocab: &Vocab, prediction: &HyperVector) -> Option<DecodedToken> {
    decode_topk(vocab, prediction, 1, None).into_iter().next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_recovers_matching_vector() {
        let mut vocab = Vocab::new(2048, 42);
        vocab.get_or_add("sky");
        vocab.get_or_add("blue");
        let sun = vocab.get_or_add("sun");

        let sun_vec = vocab.vector_by_id(sun).unwrap();
        let best = decode_best(&vocab, sun_vec).unwrap();
        assert_eq!(best.id, sun);
    }

    #[test]
    fn test_decode_penalty_reranks() {
        let mut vocab = Vocab::new(2048, 42);
        let a = vocab.get_or_add("alpha");
        let b = vocab.get_or_add("beta");
        let vec = vocab.vector_by_id(a).unwrap();

        // Penalize alpha heavily; beta should win.
        let penalty = |id: usize| if id == a { 2.0 } else { 0.0 };
        let top = decode_topk(&vocab, vec, 3, Some(&penalty));
        assert_eq!(top[0].id, b);
    }
}
