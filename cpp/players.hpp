// Scripted opponents for the tournament. None of these use the network.
#pragma once

#include <array>
#include <cstdint>
#include <memory>

#include "board.hpp"
#include "mcts.hpp"
#include "rng.hpp"
#include "solver.hpp"

namespace cz {

// Uniform over legal columns.
int random_move(const Board& board, Rng& rng);

// Takes a win, blocks a loss, avoids moves that allow an immediate reply, else random.
int tactical_move(const Board& board, Rng& rng);

// Result of a uniformly random playout, for the side to move in `board`.
float random_playout(Board board, Rng& rng);

// Plain Monte Carlo tree search: uniform priors and random playouts as the evaluator.
int rollout_mcts_move(const Board& board, int simulations, float c_puct, Rng& rng);

// Fixed-depth alpha-beta on a hand-written line-counting evaluation.
int alphabeta_move(const Board& board, int depth, Rng& rng);

// Uniform among the columns the solver scores highest.
int perfect_move(const MoveScores& scores, Rng& rng);

}  // namespace cz
