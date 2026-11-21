# FCG-IA5

Proposed global intersectional adaptation. The entry point `fcg_ia5.src.runner.run` accepts synthetic Support and Development, a checkpoint file in an existing directory, and a synthetic attempt budget. `max_batches` pauses at a completed batch boundary. Resume by calling with identical data and budget. One process may own a checkpoint at a time. C06 and C09 remain **PROPOSED**. Use [the adaptation notes](../docs/ADAPTATION_DIFFERENCES.md) and [test commands](../docs/REPRODUCIBILITY.md).
