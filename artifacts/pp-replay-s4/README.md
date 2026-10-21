# S4 acceptance artifacts

The implementation, supported scope, measurement definitions, and reproduction commands are in [the S4 report](../../docs/development/pp-replay-s4.md).

| Artifact | Purpose |
| --- | --- |
| [build.log](build.log) | Successful final build of test-llama-archs |
| [matrix.log](matrix.log) | Final 90-case matrix, 100 round trips, bounded journal, and failure/rejection tests |
| [real-generation.log](real-generation.log) | Frozen-binary 20-workload generation acceptance: 2560 exact tokens |
| [costs.log](costs.log) | Three interval types, valid state copy, interval replay, full shadow, and token-gap measurements |
| [source-and-binary.sha256](source-and-binary.sha256) | Tested source and executable fingerprints |
| [model.sha256](model.sha256) | Tested GGUF fingerprint |
| [hardware.txt](hardware.txt) | GPU identities, memory sizes, and driver |
| [base-commit.txt](base-commit.txt) | Base revision before the implementation commit |

The initial matrix and generation logs are also retained with `-initial` in their names. Use the final logs above as acceptance evidence. The raw logs contain actual program output, including detailed backend messages, and are not manually rewritten. The repository ignores `*.log`; the raw files remain available in this workspace and are also packaged in `logs.tar.gz` for retention.

The final generation process completed during an interrupted conversation. Its success summary and all 20 per-workload results are present, and the source/binary checksums match. The original process handle expired; no shell exit status has been reconstructed or invented.

The matrix log merges buffered stdout and backend stderr. Two result lines are split by intervening backend output: the `[0,1)` reverse=1, P=0, D=32 prefix resumes at line 29668, and the `[0,2)` reverse=0, P=6, D=0 result resumes at line 40123. All 90 records and the final success summary are present. The original log is preserved without reformatting.

This acceptance certifies the documented local single-sequence, single-token replay scope. It does not certify RPC traffic, concurrent PP-bubble scheduling, arbitrary cache formats, or universally stall-free migration.
