# Reproduce the offline checks

From the repository root with CPython 3.12 and NumPy 2.3.5 installed, run:

```text
python -m unittest discover -v -s fcg_paper_faithful/tests
python -m unittest discover -v -s fcg_ia5/tests
python -m unittest discover -v -s tests
```

Recorded environment: Windows x64, CPython 3.12.14, NumPy 2.3.5. No installation is needed when those versions already exist. Replay guarantees apply to the recorded environment and code. Cross-platform and cross-version floating-point replay have not been verified.

The paper fixture uses 4 development queries and 41 batches (164 fake predictions). The IA5 fixture uses 8 development queries and 81 batches (648 fake predictions over one complete run, including 8 zero-shot). These are synthetic computations, with zero real LLM calls, zero calibration runs and zero access to the official test.

Checkpoints contain only synthetic selections and binary predictions. Store them outside Git with a `.checkpoint.json` suffix. Manually resume with the same budget, input order for Development and configuration. SHA-256 checks accidental corruption, without authenticating the writer. A batch reserved before failure stays charged. A completed checkpoint replays its prefix without calling the predictor or writing the journal. Real resource/adapter and numerical paper parity checks remain pending.

WORK-04 recorded result: 6 paper, 7 IA5, and 4 integration tests passed. Failures: 0. Skipped: 0.
