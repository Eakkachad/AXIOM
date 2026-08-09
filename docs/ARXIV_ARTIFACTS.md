# AXIOM Paper Artifacts

This document records the reproducible claims currently supported by the
repository. It is not an arXiv submission.

## Reproduction Commands

```bash
cargo test --workspace --locked
cargo run --release --locked -p tle-axiom-gen --bin axiom-bench
cargo run --release --locked -p tle-axiom-gen --bin triviaqa-bench -- data/axiom_triviaqa.jsonl
```

## Verified Results

- Wikipedia web-learning gate: 345 extracted facts, 1,103 sentences, 1.53s.
- Recursive composition: deterministic 10-hop chain benchmark.
- Thai/English tokenizer: deterministic mixed-language vectors.
- TriviaQA smoke fixture: 5 records, 80% substring accuracy.

## Required Before Submission

- Run the full TriviaQA dataset with downloaded evidence.
- Report exact dataset version, hardware, and wall-clock measurements.
- Add baseline comparisons and confidence intervals.
- Separate smoke-fixture results from external benchmark results.
- Document known limitations: rule-assisted extraction, bounded search depth,
  incomplete Thai grammar, and unresolved fluency errors.
