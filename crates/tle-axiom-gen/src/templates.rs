//! Template Bank — Extract and store sentence patterns from corpus.
//!
//! Extracts templates by replacing entities with [SUBJ], [OBJ], [REL] slots.
//! At generation time, select best template and fill slots from KG path.
//!
//! Example:
//!   Corpus: "Tokyo is the capital of Japan"
//!   Template: "[SUBJ] is the [REL] of [OBJ]"
//!   
//!   Generate: subject="Paris", rel="capital", obj="France"
//!   Output: "Paris is the capital of France"

use std::collections::HashMap;

/// A sentence template with slots.
#[derive(Clone, Debug)]
pub struct Template {
    /// The template pattern with placeholders.
    pub pattern: String,
    /// How many times this pattern was seen in corpus.
    pub frequency: u32,
    /// Number of slots (1-3 typically).
    pub num_slots: usize,
    /// The relation type this template best fits.
    pub relation_hint: Option<String>,
}

/// Template bank — stores extracted patterns indexed by relation type.
pub struct TemplateBank {
    /// Templates indexed by relation keyword.
    by_relation: HashMap<String, Vec<Template>>,
    /// All templates (for fallback).
    all_templates: Vec<Template>,
    /// Total templates stored.
    pub count: usize,
}

impl TemplateBank {
    /// Create empty template bank.
    pub fn new() -> Self {
        Self {
            by_relation: HashMap::new(),
            all_templates: Vec::new(),
            count: 0,
        }
    }

    /// Extract templates from a corpus (line by line).
    pub fn extract_from_corpus(text: &str) -> Self {
        let mut bank = Self::new();

        for line in text.lines() {
            let trimmed = line.trim().to_lowercase();
            if trimmed.len() < 10 || trimmed.len() > 150 {
                continue;
            }

            // Try to extract template from common patterns
            for (pattern, relation) in EXTRACTION_PATTERNS {
                if let Some(template) = try_extract(&trimmed, pattern, relation) {
                    bank.add_template(template);
                }
            }
        }

        bank.sort_by_frequency();
        bank
    }

    /// Add a template.
    pub fn add_template(&mut self, template: Template) {
        if let Some(ref rel) = template.relation_hint {
            self.by_relation
                .entry(rel.clone())
                .or_default()
                .push(template.clone());
        }
        self.all_templates.push(template);
        self.count += 1;
    }

    /// Get best templates for a relation type.
    pub fn get_templates(&self, relation: &str, limit: usize) -> Vec<&Template> {
        if let Some(templates) = self.by_relation.get(relation) {
            templates.iter().take(limit).collect()
        } else {
            self.all_templates.iter().take(limit).collect()
        }
    }

    /// Fill a template with actual values.
    pub fn fill_template(template: &Template, subject: &str, relation: &str, object: &str) -> String {
        template.pattern
            .replace("[SUBJ]", subject)
            .replace("[REL]", relation)
            .replace("[OBJ]", object)
    }

    /// Get the best template for a (subject, relation, object) triple and fill it.
    pub fn generate(&self, subject: &str, relation: &str, object: &str) -> Option<String> {
        let templates = self.get_templates(relation, 5);
        if templates.is_empty() {
            return None;
        }

        // Pick the most frequent template
        let best = templates[0];
        Some(Self::fill_template(best, subject, relation, object))
    }

    /// Sort all template lists by frequency (most frequent first).
    fn sort_by_frequency(&mut self) {
        self.all_templates.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        for templates in self.by_relation.values_mut() {
            templates.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> (usize, usize) {
        (self.count, self.by_relation.len())
    }
}

impl Default for TemplateBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Extraction patterns: (pattern_with_placeholders, relation_type)
const EXTRACTION_PATTERNS: &[(&str, &str)] = &[
    ("[SUBJ] is a [OBJ]", "is"),
    ("[SUBJ] is the [OBJ]", "is"),
    ("[SUBJ] is an [OBJ]", "is"),
    ("[SUBJ] are [OBJ]", "are"),
    ("[SUBJ] was [OBJ]", "was"),
    ("[SUBJ] has [OBJ]", "has"),
    ("[SUBJ] have [OBJ]", "have"),
    ("[SUBJ] can [OBJ]", "can"),
    ("[SUBJ] causes [OBJ]", "causes"),
    ("[SUBJ] produces [OBJ]", "produces"),
    ("[SUBJ] contains [OBJ]", "contains"),
    ("[SUBJ] requires [OBJ]", "requires"),
    ("[SUBJ] leads to [OBJ]", "leads to"),
    ("[SUBJ] is located in [OBJ]", "located_in"),
    ("[SUBJ] was born in [OBJ]", "born_in"),
    ("[SUBJ] is known for [OBJ]", "known_for"),
    ("[SUBJ] is part of [OBJ]", "part_of"),
    ("[SUBJ] was released in [OBJ]", "released_in"),
    ("[SUBJ] was founded in [OBJ]", "founded_in"),
];

/// Try to extract a template from a sentence given a pattern.
fn try_extract(sentence: &str, pattern: &str, relation: &str) -> Option<Template> {
    // Find the relation verb in the sentence
    let rel_phrase = pattern
        .replace("[SUBJ]", "")
        .replace("[OBJ]", "")
        .trim()
        .to_string();

    if sentence.contains(&rel_phrase) {
        // Found a match — create template
        let parts: Vec<&str> = sentence.splitn(2, &rel_phrase).collect();
        if parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty() {
            // Validate: subject and object should be reasonable lengths
            let subj_len = parts[0].trim().split_whitespace().count();
            let obj_len = parts[1].trim().split_whitespace().count();

            if subj_len >= 1 && subj_len <= 5 && obj_len >= 1 && obj_len <= 10 {
                return Some(Template {
                    pattern: pattern.to_string(),
                    frequency: 1,
                    num_slots: 2,
                    relation_hint: Some(relation.to_string()),
                });
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_from_corpus() {
        let corpus = "\
Tokyo is the capital of Japan
Paris is the capital of France
The cat is a small animal
The dog is a loyal pet
Water is a liquid
Plants require sunlight
The sun causes heat
";
        let bank = TemplateBank::extract_from_corpus(corpus);
        assert!(bank.count > 0, "Should extract at least some templates");
    }

    #[test]
    fn test_fill_template() {
        let template = Template {
            pattern: "[SUBJ] is the [REL] of [OBJ]".to_string(),
            frequency: 10,
            num_slots: 3,
            relation_hint: Some("is".to_string()),
        };

        let filled = TemplateBank::fill_template(&template, "Bangkok", "capital", "Thailand");
        assert_eq!(filled, "Bangkok is the capital of Thailand");
    }

    #[test]
    fn test_generate() {
        let mut bank = TemplateBank::new();
        bank.add_template(Template {
            pattern: "[SUBJ] is a [OBJ]".to_string(),
            frequency: 100,
            num_slots: 2,
            relation_hint: Some("is".to_string()),
        });

        let result = bank.generate("cat", "is", "animal");
        assert_eq!(result, Some("cat is a animal".to_string()));
    }

    #[test]
    fn test_get_templates_by_relation() {
        let mut bank = TemplateBank::new();
        bank.add_template(Template {
            pattern: "[SUBJ] is [OBJ]".to_string(),
            frequency: 50,
            num_slots: 2,
            relation_hint: Some("is".to_string()),
        });
        bank.add_template(Template {
            pattern: "[SUBJ] has [OBJ]".to_string(),
            frequency: 30,
            num_slots: 2,
            relation_hint: Some("has".to_string()),
        });

        let is_templates = bank.get_templates("is", 10);
        assert_eq!(is_templates.len(), 1);
        assert!(is_templates[0].pattern.contains("is"));
    }
}
