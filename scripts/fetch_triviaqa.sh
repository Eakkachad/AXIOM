#!/usr/bin/env bash
set -euo pipefail

# TriviaQA is external data and is intentionally not committed to this repo.
URL="https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz"
ROOT="${1:-data/triviaqa}"
ARCHIVE="${ROOT}/triviaqa-rc.tar.gz"

mkdir -p "${ROOT}"
if [[ ! -s "${ARCHIVE}" ]]; then
  curl --fail --location --retry 3 --connect-timeout 20 --output "${ARCHIVE}.tmp" "${URL}"
  mv "${ARCHIVE}.tmp" "${ARCHIVE}"
fi

tar -tzf "${ARCHIVE}" >/dev/null
tar -xzf "${ARCHIVE}" -C "${ROOT}"
printf 'TriviaQA extracted under %s\n' "${ROOT}"
printf 'Run the benchmark with the extracted JSON and an evidence-facts JSONL file.\n'
