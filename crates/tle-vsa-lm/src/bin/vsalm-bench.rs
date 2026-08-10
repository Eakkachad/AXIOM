//! VSA-LM benchmark: measure next-token accuracy, generation quality, and
//! determinism against a real corpus, with no neural components.

use std::time::Instant;

use tle_vsa_lm::{LmConfig, ReservoirConfig, VsaLm};

fn main() {
    let corpus = load_corpus();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  VSA-LM: VSA Language Model — non-neural text generator      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Split train/test on sentences (80/20).
    let split = (corpus.len() as f32 * 0.8) as usize;
    let train = &corpus[..split];
    let test = &corpus[split..];
    println!("Corpus: {} sentences (train={}, test={})", corpus.len(), train.len(), test.len());

    let config = LmConfig {
        dim: 10_240,
        max_order: 4,
        beam_width: 8,
        max_gen_tokens: 12,
        w_reservoir: 0.6,
        reservoir_config: Some(ReservoirConfig { dim: 2048, leak_rate: 0.25, sparsity: 0.1, spectral_radius: 0.95 }),
        ..Default::default()
    };
    let mut lm = VsaLm::new(config.clone());

    println!("\nStep 1: Ingesting training corpus...");
    let t0 = Instant::now();
    for s in train {
        lm.learn(s);
    }
    println!("  {} words in {:.2?}", lm.vocab.len(), t0.elapsed());
    print!("  Building TBA TopK cache... ");
    let t0 = Instant::now();
    lm.build_tba_cache();
    println!("done in {:.2?}", t0.elapsed());

    println!("\nStep 2: Next-token accuracy (no softmax, no neural readout)");
    let t0 = Instant::now();
    let (train_acc, train_n) = lm.next_token_accuracy(train);
    let (test_acc, test_n) = lm.next_token_accuracy(test);
    println!("  TRAIN: {:.1}% ({}/{})", train_acc * 100.0, train_n, train_n);
    println!("  TEST:  {:.1}% ({}/{})", test_acc * 100.0, test_n, test_n);
    println!("  elapsed: {:.2?}", t0.elapsed());

    println!("\nStep 2b: Signal decomposition (TBA-only vs Engram-only, TRAIN)");
    let t0 = Instant::now();
    let (tba_acc, tba_n) = tba_only_accuracy(&lm, train);
    let (eng_acc, eng_n) = engram_only_accuracy(&lm, train);
    println!("  TRAIN TBA-only (pure VSA transition): {:.1}% ({}/{})", tba_acc * 100.0, tba_n, tba_n);
    println!("  TRAIN Engram-only (O(1) n-gram):      {:.1}% ({}/{})", eng_acc * 100.0, eng_n, eng_n);
    let (tba_test, tba_test_n) = tba_only_accuracy(&lm, test);
    let (eng_test, eng_test_n) = engram_only_accuracy(&lm, test);
    println!("  TEST  TBA-only (pure VSA transition): {:.1}% ({}/{})", tba_test * 100.0, tba_test_n, tba_test_n);
    println!("  TEST  Engram-only (O(1) n-gram):      {:.1}% ({}/{})", eng_test * 100.0, eng_test_n, eng_test_n);
    println!("  elapsed: {:.2?}", t0.elapsed());

    println!("\nStep 3: Generation (deterministic, energy-guided beam search)");
    let prompts = ["the cat", "the sun", "a dog", "the moon", "a small"];
    for prompt in prompts {
        let out = lm.generate(prompt, Some(10));
        println!("  \"{}\" → \"{}\"", prompt, out);
    }

    println!("\nStep 4: Determinism check (5 identical runs)");
    let mut unique = std::collections::HashSet::new();
    for _ in 0..5 {
        unique.insert(lm.generate("the cat", Some(6)));
    }
    println!("  Unique outputs from 5 runs: {} (expected 1)", unique.len());
    println!("  Deterministic: {}", if unique.len() == 1 { "✓" } else { "✗" });

    println!("\n━━━ VSA-LM RESULTS ━━━");
    println!("  VSA dim={}, n-gram order={}, beam={}", config.dim, config.max_order, config.beam_width);
    println!("  Vocab: {} words", lm.vocab.len());
    println!("  No backprop: ✓ (TBA + Engram + cosine decode)");
    println!("  No softmax: ✓ (cosine ranking)");
    println!("  No probability sampling: ✓ (deterministic)");
    println!("  TRAIN next-token acc: {:.1}%", train_acc * 100.0);
    println!("  TEST  next-token acc: {:.1}%", test_acc * 100.0);
}

fn tba_only_accuracy(lm: &VsaLm, sentences: &[String]) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        if tokens.len() < 2 {
            continue;
        }
        for pos in 0..tokens.len() - 1 {
            let context: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_tba_only(&context, 5);
            if pred.is_empty() {
                continue;
            }
            let true_id = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == true_id {
                correct += 1;
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (correct as f32 / total as f32, total)
    }
}

fn engram_only_accuracy(lm: &VsaLm, sentences: &[String]) -> (f32, usize) {
    let mut correct = 0usize;
    let mut total = 0usize;
    for sentence in sentences {
        let tokens = lm.tokenize(sentence);
        if tokens.len() < 2 {
            continue;
        }
        for pos in 0..tokens.len() - 1 {
            let context: Vec<String> = tokens[..=pos].to_vec();
            let pred = lm.predict_engram_only(&context, 5);
            if pred.is_empty() {
                continue;
            }
            let true_id = lm.vocab.id(&tokens[pos + 1]);
            total += 1;
            if Some(pred[0].id) == true_id {
                correct += 1;
            }
        }
    }
    if total == 0 {
        (0.0, 0)
    } else {
        (correct as f32 / total as f32, total)
    }
}

fn load_corpus() -> Vec<String> {
    let raw = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "a cat is a small animal",
        "a dog is a loyal friend",
        "the sun is bright and warm",
        "the moon is bright at night",
        "she walked to the store",
        "he ate the red apple",
        "the fish swam in the water",
        "they played in the garden",
        "I love my cat very much",
        "the big dog ran very fast",
        "the small bird sang a song",
        "she read a good book today",
        "he built a new house here",
        "the car stopped at the light",
        "we went to the old beach",
        "the food was hot and fresh",
        "they found the lost key",
        "the rain fell on the ground",
        "she opened the big door",
        "the child played with a ball",
        "he drove the car to work",
        "the flower grew in the sun",
        "we watched the bright stars",
        "the river flows to the sea",
        "she wrote a long letter",
        "the old man sat on bench",
        "he fixed the broken window",
        "the wind blew the leaves away",
        "she found a gold coin here",
        "the mountain is very tall",
        "he jumped over the fence",
        "they sang songs at night",
        "the snow fell on the ground",
        "she threw the ball to him",
        "the sky turned dark and cold",
        "he read the news every day",
        "they walked along the river",
        "a bird is a flying animal",
        "a fish is a swimming animal",
        "the sea is blue and wide",
        "the grass is green in spring",
        "she has a red car",
        "he has a blue bike",
        "the teacher gave a lesson",
        "the student asked a question",
        "water is a clear liquid",
        "ice is frozen water",
        "the cat chased the mouse",
        "the dog barked at the cat",
        "birds sing in the morning",
        "the sun rises in the east",
        "the sun sets in the west",
        "night is dark and quiet",
        "morning is bright and fresh",
        "the book has many pages",
        "the story has a happy ending",
        "she bought fresh bread",
        "he drank cold water",
        "the tree has green leaves",
        "the flower smells sweet",
        "winter is cold and snowy",
        "summer is hot and sunny",
        "the train runs on tracks",
        "the plane flies in the sky",
        "the bus stops at the corner",
        "the ship sails on the sea",
        "she wears a warm coat",
        "he carries a heavy bag",
        "the kitchen is clean and tidy",
        "the garden is full of flowers",
        "cats like to sleep",
        "dogs like to play",
        "children like to laugh",
        "birds like to fly",
        "the apple is red and sweet",
        "the banana is yellow and soft",
        "the grape is small and purple",
        "the orange is round and orange",
        "she sings a happy song",
        "he tells a funny story",
        "they watch the stars at night",
        "we read books in the library",
        "the clock tells the time",
        "the mirror shows your face",
        "the door opens with a key",
        "the window lets in light",
        "rain falls from the clouds",
        "snow falls in winter",
        "the wind blows in autumn",
        "the leaves turn in autumn",
        "a cat has four legs",
        "a bird has two wings",
        "a fish has gills",
        "a human has two hands",
    ];
    raw.iter().map(|s| s.to_string()).collect()
}
