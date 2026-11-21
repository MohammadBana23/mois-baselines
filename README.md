# MOIS baselines

Independent offline research implementations. Both runners use an in-process deterministic fake predictor and synthetic fixtures. Python 3.12 and NumPy 2.3.5 are required.

| Path | Purpose |
| --- | --- |
| [fcg_paper_faithful](fcg_paper_faithful) | Four-cell reconstruction of Hu et al. FCG, with explicit engineering assumptions |
| [fcg_ia5](fcg_ia5) | Eight-cell intersectional adaptation with proposed fitness and quota policies |
| [docs](docs) | Paper mapping, adaptation differences and reproducibility |
| [tests](tests) | Synthetic fixtures and cross-component integration |

[Reproduce the offline checks](docs/REPRODUCIBILITY.md). Future experiment manifests must pin the full repository commit, variant, config hash, environment, cohort roles and digests, and the applicable protocol approvals. This delivery supports synthetic checks only.
