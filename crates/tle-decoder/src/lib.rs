//! # Latent-to-English Decoder
//!
//! Symbolic lookup decoder that converts latent hypervectors back
//! to English tokens/words using codebook nearest-neighbor search.
//!
//! ## Decoding Process
//!
//! 1. Receive a latent hypervector from the pipeline
//! 2. Apply resonator cleanup to sharpen the signal
//! 3. Look up nearest codebook entry (deterministic argmax)
//! 4. Return the corresponding English token
//!
//! This is zero-sampling: no temperature, no top-p, no randomness.
//! The output is the deterministic nearest neighbor in the codebook.

pub mod decoder;

pub use decoder::LatentDecoder;
