// Batched game play. A pool runs many games at once, each with its own search tree,
// and stops at every point where trees need network evaluations. collect() advances
// all games as far as they can go and returns the leaves as one tensor. apply() takes
// the network's answers. Python owns the network, C++ owns everything else, and the
// GIL is released while the trees work.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "board.hpp"
#include "mcts.hpp"
#include "players.hpp"
#include "rng.hpp"
#include "solver.hpp"

namespace cz {

struct Seat {
    enum class Kind { Network, Random, Tactical, Rollout, AlphaBeta, Perfect };
    Kind kind = Kind::Network;
    int evaluator = 0;      // Network: which of the caller's nets answers this seat
    int strength = 0;       // Rollout: simulations. AlphaBeta: depth.
    int simulations = 200;  // Network: visits per move
};

struct PoolParams {
    int num_games = 1;
    int concurrency = 1;      // games in flight
    int leaves_per_tree = 1;  // selections per tree per batch
    int threads = 1;
    std::uint64_t seed = 0;
    SearchParams search;
    int temperature_plies = 0;  // sample proportional to visits for this many plies
    bool share_tree = false;    // both seats reuse one tree (self-play)
    bool record_samples = false;
    int solver_log2_entries = 22;  // shared by all workers, so repeat positions are instant
    std::array<Seat, 2> seats{};
    std::vector<std::string> openings;  // game i starts from openings[i % size]
};

struct GameResult {
    std::string moves;
    int winner_seat = -1;  // -1 draw
    int first_seat = 0;
};

struct Sample {
    Bits current = 0;
    Bits occupied = 0;
    std::array<float, kWidth> policy{};
    float outcome = 0.0f;  // from the view of the side to move
};

struct LeafBatch {
    std::vector<float> planes;  // n * kPlanes * kCells
    std::vector<std::int32_t> evaluators;
    int count() const { return static_cast<int>(evaluators.size()); }
};

class GamePool {
   public:
    explicit GamePool(PoolParams params);
    ~GamePool();

    // Advances games until each is waiting on evaluations or is finished. An empty
    // batch means all games are done.
    const LeafBatch& collect();
    void apply(const float* policy, const float* values);

    bool finished() const;
    int games_done() const { return games_done_; }
    std::int64_t evaluations() const { return evaluations_; }
    std::int64_t batches() const { return batches_; }

    const std::vector<GameResult>& results() const { return results_; }
    std::vector<Sample> drain_samples();

   private:
    struct Slot;

    void start_game(Slot& slot, int game_index);
    void step_slot(Slot& slot, int worker);
    void make_move(Slot& slot, int column);
    void finish_game(Slot& slot, int winner_seat);
    int builtin_move(Slot& slot, const Seat& seat, int worker);
    int choose_column(Slot& slot, Tree& tree);

    PoolParams params_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::vector<std::unique_ptr<Solver>> solvers_;
    std::vector<GameResult> results_;
    std::vector<std::vector<Sample>> samples_;
    std::atomic<int> next_game_{0};
    std::atomic<int> games_done_{0};
    LeafBatch batch_;
    std::int64_t evaluations_ = 0;
    std::int64_t batches_ = 0;
};

// One scripted move per position, for grading a baseline on the benchmark.
std::vector<int> seat_moves(const std::vector<Board>& boards, const Seat& seat, float c_puct,
                            int threads, std::uint64_t seed);

// Runs a fixed search from each given position, for scoring a network on the
// stratified benchmark and for the live demo. Uses no root noise.
class PositionSearch {
   public:
    PositionSearch(std::vector<Board> boards, SearchParams params, int simulations,
                   int leaves_per_tree, int threads);

    const LeafBatch& collect();
    void apply(const float* policy, const float* values);
    std::array<int, kWidth> visits(int index) const { return trees_[index].root_child_visits(); }
    float value(int index) const { return trees_[index].root_value(); }

   private:
    std::vector<Tree> trees_;
    std::vector<std::vector<Leaf>> pending_;
    int simulations_;
    int leaves_per_tree_;
    int threads_;
    LeafBatch batch_;
};

}  // namespace cz
