# AXIOM Release Checklist

## Reproducible Build

```bash
cargo build --release --locked
cargo test --workspace --locked
```

The release profile uses thin LTO, one codegen unit, stripped symbols, and
abort-on-panic for a small deterministic CPU binary.

## Binaries

```bash
cargo build --release --locked -p tle-deepman
cargo build --release --locked -p tle-axiom-gen --bin axiom-bench
cargo build --release --locked -p tle-axiom-gen --bin triviaqa-bench
```

## Validation

```bash
cargo run --release --locked -p tle-axiom-gen --bin axiom-bench
cargo run --release --locked -p tle-axiom-gen --bin triviaqa-bench -- data/axiom_triviaqa.jsonl
# For TriviaQA JSONL plus a separate evidence-facts JSONL:
# cargo run --release --locked -p tle-axiom-gen --bin triviaqa-bench -- triviaqa.json evidence_facts.jsonl
```

Full TriviaQA scoring requires an external dataset and evidence ingestion;
the repository fixture is only a smoke benchmark.

## Local Linux Artifact

The release packaging step produces `dist/axiom-linux-x86_64.tar.gz` and
`dist/SHA256SUMS`. The `dist/` directory is intentionally ignored by git;
publish the archive through the release system rather than committing binaries.
