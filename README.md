# connectzero

AlphaZero for Connect 4, graded move by move against a perfect solver.

This commit holds the engine: C++20 bitboards, an exact alpha-beta solver, batched PUCT search,
and the Python training stack, all covered by tests. Training results follow.

    uv sync --frozen
    uv run --frozen pytest -m "not slow"
