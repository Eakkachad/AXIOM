//! Multi-Sentence Generation — Paragraph planner for coherent multi-sentence responses.
//!
//! Given a topic, plans a sequence of subtopics to cover, then generates
//! one sentence per subtopic, joined with connectives.
//!
//! Example:
//!   Topic: "elephants"
//!   Plan: [definition, habitat, abilities, features]
//!   Output: "Elephants are large animals. They live in Africa and Asia.
//!            They can swim very well. They have long trunks and big ears."

use std::collections::HashMap;

/// A planned paragraph — sequence of subtopics to cover.
#[derive(Clone, Debug)]
pub struct ParagraphPlan {
    /// The main topic.
    pub topic: String,
    /// Ordered subtopics (relations to cover).
    pub subtopics: Vec<String>,
}

/// Paragraph generator — plans and generates multi-sentence responses.
pub struct ParagraphGenerator {
    /// Preferred relation ordering (more general → more specific).
    relation_priority: Vec<String>,
}

impl ParagraphGenerator {
    /// Create a new paragraph generator.
    pub fn new() -> Self {
        Self {
            relation_priority: vec![
                "is".to_string(),
                "are".to_string(),
                "located_in".to_string(),
                "lives_in".to_string(),
                "has".to_string(),
                "have".to_string(),
                "can".to_string(),
                "causes".to_string(),
                "produces".to_string(),
                "requires".to_string(),
                "contains".to_string(),
                "was".to_string(),
                "born_in".to_string(),
                "known_for".to_string(),
            ],
        }
    }

    /// Plan a paragraph for a topic given available facts.
    ///
    /// Selects and orders subtopics based on relation priority.
    pub fn plan(&self, topic: &str, facts: &[(String, String)]) -> ParagraphPlan {
        let mut ordered: Vec<String> = Vec::new();

        // Add facts in priority order
        for priority_rel in &self.relation_priority {
            for (rel, _) in facts {
                if rel == priority_rel && !ordered.contains(rel) {
                    ordered.push(rel.clone());
                }
            }
        }

        // Add any remaining relations not in priority list
        for (rel, _) in facts {
            if !ordered.contains(rel) {
                ordered.push(rel.clone());
            }
        }

        ParagraphPlan {
            topic: topic.to_string(),
            subtopics: ordered,
        }
    }

    /// Generate a multi-sentence paragraph from a plan and facts.
    ///
    /// Each fact becomes one sentence. Sentences are joined with
    /// appropriate transitions.
    pub fn generate(
        &self,
        topic: &str,
        facts: &[(String, String)],
    ) -> String {
        if facts.is_empty() {
            return format!("I don't know much about {} yet.", topic);
        }

        let plan = self.plan(topic, facts);
        let mut sentences: Vec<String> = Vec::new();
        let mut covered_relations: Vec<&str> = Vec::new();

        // Generate one sentence per subtopic (in planned order)
        for subtopic_rel in &plan.subtopics {
            // Find the fact for this relation
            for (rel, obj) in facts {
                if rel == subtopic_rel && !covered_relations.contains(&rel.as_str()) {
                    let sentence = self.fact_to_sentence(topic, rel, obj, sentences.is_empty());
                    sentences.push(sentence);
                    covered_relations.push(rel);
                    break;
                }
            }

            // Limit to 5 sentences per paragraph
            if sentences.len() >= 5 {
                break;
            }
        }

        // Join with transitions
        self.join_sentences(&sentences)
    }

    /// Convert a single fact to a sentence.
    fn fact_to_sentence(&self, topic: &str, relation: &str, object: &str, is_first: bool) -> String {
        let subject = if is_first {
            capitalize(topic)
        } else {
            // Use pronoun for subsequent sentences
            if topic.ends_with('s') {
                "They".to_string()
            } else {
                "It".to_string()
            }
        };

        match relation {
            "is" | "are" => format!("{} {} {}.", subject, relation, object),
            "has" | "have" => format!("{} {} {}.", subject, relation, object),
            "can" => format!("{} can {}.", subject, object),
            "causes" => format!("{} causes {}.", subject, object),
            "produces" => format!("{} produces {}.", subject, object),
            "located_in" | "lives_in" => format!("{} lives in {}.", subject, object),
            "born_in" => format!("{} was born in {}.", subject, object),
            "known_for" => format!("{} is known for {}.", subject, object),
            _ => format!("{} {} {}.", subject, relation, object),
        }
    }

    /// Join sentences with appropriate transitions.
    fn join_sentences(&self, sentences: &[String]) -> String {
        if sentences.is_empty() {
            return String::new();
        }
        if sentences.len() == 1 {
            return sentences[0].clone();
        }

        // Simple join — each sentence on its own
        sentences.join(" ")
    }
}

impl Default for ParagraphGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Capitalize first letter.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_ordering() {
        let gen = ParagraphGenerator::new();
        let facts = vec![
            ("can".to_string(), "swim".to_string()),
            ("is".to_string(), "an animal".to_string()),
            ("has".to_string(), "four legs".to_string()),
        ];

        let plan = gen.plan("cat", &facts);
        // "is" should come before "has" which should come before "can"
        assert_eq!(plan.subtopics[0], "is");
        assert_eq!(plan.subtopics[1], "has");
        assert_eq!(plan.subtopics[2], "can");
    }

    #[test]
    fn test_generate_paragraph() {
        let gen = ParagraphGenerator::new();
        let facts = vec![
            ("is".to_string(), "a large animal".to_string()),
            ("lives_in".to_string(), "Africa and Asia".to_string()),
            ("can".to_string(), "swim very well".to_string()),
            ("has".to_string(), "long trunks".to_string()),
        ];

        let paragraph = gen.generate("elephants", &facts);
        assert!(paragraph.contains("Elephants"));
        assert!(paragraph.contains("large animal"));
        assert!(paragraph.contains("swim"));
        // Should have multiple sentences
        let sentence_count = paragraph.matches('.').count();
        assert!(sentence_count >= 3, "Should have 3+ sentences, got {}", sentence_count);
    }

    #[test]
    fn test_pronoun_usage() {
        let gen = ParagraphGenerator::new();
        let facts = vec![
            ("is".to_string(), "a small animal".to_string()),
            ("has".to_string(), "soft fur".to_string()),
        ];

        let paragraph = gen.generate("cat", &facts);
        // First sentence uses topic name, second uses pronoun "It"
        assert!(paragraph.contains("Cat"));
        assert!(paragraph.contains("It "), "Should use pronoun: {}", paragraph);
    }

    #[test]
    fn test_empty_facts() {
        let gen = ParagraphGenerator::new();
        let paragraph = gen.generate("unknown", &[]);
        assert!(paragraph.contains("don't know"));
    }
}
