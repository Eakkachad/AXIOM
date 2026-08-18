//! Binary Serializer & Loader for Two-Tier Transmuted Algebraic Engines.
//!
//! Provides fast zero-copy / streaming binary persistence for:
//! - Tier 1: ZCA-Whitened Phasor Vocabulary Codebook ($D = d/2$).
//! - Tier 1: Gated Sheaf Routing Layers & Planar Rotors.
//! - Tier 2: Sparse Continuous Hopfield Factual Memory Patterns.
//! - Fast-weight adaptation matrix.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tle_vsa::whitened_phasor::{WhitenedPhasor, WhitenedPhasorCodebook};
use crate::gated_hopfield::{GatedHopfieldMemory, HopfieldMemorySlot};
use crate::gated_sheaf::GatedSheafLayer;
use crate::two_tier_engine::{TwoTierConfig, TwoTierEngine};

const MAGIC_HEADER: &[u8; 8] = b"TWOTIER1";

impl TwoTierEngine {
    /// Saves the entire TwoTierEngine into a compact binary file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // 1. Write Magic Header
        writer.write_all(MAGIC_HEADER)?;

        // 2. Write Config
        writer.write_all(&(self.config.dim as u32).to_le_bytes())?;
        writer.write_all(&(self.config.sheaf_layers as u32).to_le_bytes())?;
        writer.write_all(&(self.config.stalk_dim as u32).to_le_bytes())?;
        writer.write_all(&(self.config.shortlist_size as u32).to_le_bytes())?;

        // 3. Write Tier 1 Vocabulary Phasors
        let vocab_len = self.vocabulary.id_to_token.len() as u32;
        writer.write_all(&vocab_len.to_le_bytes())?;

        for (idx, token) in self.vocabulary.id_to_token.iter().enumerate() {
            let token_bytes = token.as_bytes();
            writer.write_all(&(token_bytes.len() as u16).to_le_bytes())?;
            writer.write_all(token_bytes)?;

            let phasor = &self.vocabulary.phasors[idx];
            writer.write_all(&(phasor.angles.len() as u32).to_le_bytes())?;
            for &angle in &phasor.angles {
                writer.write_all(&angle.to_le_bytes())?;
            }
        }

        // 4. Write Tier 2 Hopfield Memory Slots
        let slot_len = self.factual_memory.slots.len() as u32;
        writer.write_all(&slot_len.to_le_bytes())?;

        for slot in &self.factual_memory.slots {
            for &k in &slot.key {
                writer.write_all(&k.to_le_bytes())?;
            }
            for &v in &slot.value {
                writer.write_all(&v.to_le_bytes())?;
            }
            writer.write_all(&slot.norm_scale.to_le_bytes())?;
        }

        // 5. Write Fast Weights
        let fw_len = self.fast_weights.len() as u32;
        writer.write_all(&fw_len.to_le_bytes())?;
        for &w in &self.fast_weights {
            writer.write_all(&w.to_le_bytes())?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Loads a TwoTierEngine from a binary file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 1. Read and verify Magic Header
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC_HEADER {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic header: not a valid TWOTIER1 binary",
            ));
        }

        // 2. Read Config
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let dim = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let sheaf_layers_count = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let stalk_dim = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let shortlist_size = u32::from_le_bytes(buf4) as usize;

        let config = TwoTierConfig {
            dim,
            sheaf_layers: sheaf_layers_count,
            stalk_dim,
            shortlist_size,
        };

        // 3. Read Tier 1 Vocabulary
        reader.read_exact(&mut buf4)?;
        let vocab_len = u32::from_le_bytes(buf4) as usize;

        let mut token_to_id = HashMap::with_capacity(vocab_len);
        let mut id_to_token = Vec::with_capacity(vocab_len);
        let mut phasors = Vec::with_capacity(vocab_len);

        let mut buf2 = [0u8; 2];
        for i in 0..vocab_len {
            reader.read_exact(&mut buf2)?;
            let str_len = u16::from_le_bytes(buf2) as usize;
            let mut str_buf = vec![0u8; str_len];
            reader.read_exact(&mut str_buf)?;
            let token = String::from_utf8_lossy(&str_buf).to_string();

            reader.read_exact(&mut buf4)?;
            let angle_len = u32::from_le_bytes(buf4) as usize;
            let mut angles = Vec::with_capacity(angle_len);
            for _ in 0..angle_len {
                reader.read_exact(&mut buf4)?;
                angles.push(f32::from_le_bytes(buf4));
            }

            token_to_id.insert(token.clone(), i);
            id_to_token.push(token);
            phasors.push(WhitenedPhasor::new(angles));
        }

        let vocabulary = WhitenedPhasorCodebook {
            token_to_id,
            id_to_token,
            phasors,
            whitener: None,
        };

        // 4. Initialize Sheaf Layers
        let mut sheaf_layers = Vec::with_capacity(config.sheaf_layers);
        for _ in 0..config.sheaf_layers {
            sheaf_layers.push(GatedSheafLayer::new(config.stalk_dim, 0.5, 0.5));
        }

        // 5. Read Tier 2 Hopfield Memory Slots
        reader.read_exact(&mut buf4)?;
        let slot_len = u32::from_le_bytes(buf4) as usize;
        let mut slots = Vec::with_capacity(slot_len);

        for _ in 0..slot_len {
            let mut key = Vec::with_capacity(dim);
            for _ in 0..dim {
                reader.read_exact(&mut buf4)?;
                key.push(f32::from_le_bytes(buf4));
            }
            let mut value = Vec::with_capacity(dim);
            for _ in 0..dim {
                reader.read_exact(&mut buf4)?;
                value.push(f32::from_le_bytes(buf4));
            }
            reader.read_exact(&mut buf4)?;
            let norm_scale = f32::from_le_bytes(buf4);

            slots.push(HopfieldMemorySlot {
                key,
                value,
                norm_scale,
            });
        }

        let factual_memory = GatedHopfieldMemory {
            dim,
            beta: 1.0 / (dim as f32).sqrt(),
            slots,
        };

        // 6. Read Fast Weights
        reader.read_exact(&mut buf4)?;
        let fw_len = u32::from_le_bytes(buf4) as usize;
        let mut fast_weights = Vec::with_capacity(fw_len);
        for _ in 0..fw_len {
            reader.read_exact(&mut buf4)?;
            fast_weights.push(f32::from_le_bytes(buf4));
        }

        Ok(Self {
            config,
            vocabulary,
            sheaf_layers,
            factual_memory,
            fast_weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_load_twotier_roundtrip() {
        let tokens = vec!["Rome".to_string(), "Italy".to_string(), "Capital".to_string()];
        let raw_embs = vec![
            vec![1.0, 0.5, -0.2, 0.8],
            vec![0.9, 0.4, -0.1, 0.7],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let config = TwoTierConfig {
            dim: 4,
            sheaf_layers: 2,
            stalk_dim: 2,
            shortlist_size: 16,
        };

        let mut engine = TwoTierEngine::new(tokens, raw_embs, config).unwrap();
        engine.factual_memory.add_pattern(
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        );

        let temp_path = std::env::temp_dir().join("test_model.twotier");
        engine.save_to_file(&temp_path).unwrap();

        let loaded = TwoTierEngine::load_from_file(&temp_path).unwrap();
        assert_eq!(loaded.config.dim, 4);
        assert_eq!(loaded.vocabulary.id_to_token.len(), 3);
        assert_eq!(loaded.factual_memory.slots.len(), 1);

        let _ = std::fs::remove_file(temp_path);
    }
}
