#include "pool.hpp"

#include <algorithm>
#include <atomic>
#include <random>

#include "parallel.hpp"

namespace cz {

namespace {

constexpr int planes_size = kPlanes * kCells;

// Gathers `pending`'s leaves into a batch, computing offsets first so workers can
// fill their own ranges in parallel.
template <typename LeafLists, typename EvaluatorOf>
void build_batch(LeafBatch& batch, const LeafLists& lists, int threads, EvaluatorOf evaluator_of) {
    const int count = static_cast<int>(lists.size());
    std::vector<int> offsets(count + 1, 0);
    for (int index = 0; index < count; ++index) {
        offsets[index + 1] = offsets[index] + static_cast<int>(lists[index]->size());
    }
    const int total = offsets[count];
    batch.planes.resize(static_cast<std::size_t>(total) * planes_size);
    batch.evaluators.resize(total);
    parallel_for(count, threads, [&](int index, int) {
        const auto& leaves = *lists[index];
        for (std::size_t item = 0; item < leaves.size(); ++item) {
            const int row = offsets[index] + static_cast<int>(item);
            encode_planes(leaves[item].board, batch.planes.data() + std::size_t(row) * planes_size);
            batch.evaluators[row] = evaluator_of(index);
        }
    });
}

}  // namespace

struct GamePool::Slot {
    explicit Slot(const SearchParams& search) : trees{Tree(search), Tree(search)} {}

    int game_index = -1;
    bool active = false;
    Board board;
    std::array<Tree, 2> trees;
    std::vector<Leaf> pending;
    int pending_tree = 0;
    int pending_evaluator = 0;
    Rng rng;
    int first_seat = 0;
    int opening_plies = 0;
    std::vector<int> played;
    std::vector<Sample> samples;

    int seat_to_move() const { return (board.moves() - opening_plies + first_seat) % 2; }
};

GamePool::GamePool(PoolParams params) : params_(std::move(params)) {
    params_.concurrency = std::min(params_.concurrency, params_.num_games);
    results_.resize(params_.num_games);
    samples_.resize(params_.num_games);
    for (int index = 0; index < params_.concurrency; ++index) {
        slots_.push_back(std::make_unique<Slot>(params_.search));
    }
    const bool needs_solver = params_.seats[0].kind == Seat::Kind::Perfect ||
                              params_.seats[1].kind == Seat::Kind::Perfect;
    if (needs_solver) {
        auto table = std::make_shared<TranspositionTable>(params_.solver_log2_entries);
        for (int worker = 0; worker < params_.threads; ++worker) {
            solvers_.push_back(std::make_unique<Solver>(table));
        }
    }
    for (auto& slot : slots_) start_game(*slot, next_game_.fetch_add(1));
}

GamePool::~GamePool() = default;

void GamePool::start_game(Slot& slot, int game_index) {
    slot.game_index = game_index;
    slot.active = true;
    slot.board = Board{};
    slot.first_seat = game_index % 2;
    slot.opening_plies = 0;
    slot.played.clear();
    slot.samples.clear();
    slot.pending.clear();
    // Seeding by game index, not slot, makes a game's course independent of which
    // slot or worker happens to run it.
    slot.rng = Rng(params_.seed ^ (0x51ED2705ULL * static_cast<std::uint64_t>(game_index + 1)));
    if (!params_.openings.empty()) {
        const auto& opening = params_.openings[game_index % params_.openings.size()];
        slot.board = Board::from_moves(opening);
        if (slot.board.is_over()) throw std::invalid_argument("opening ends the game");
        slot.opening_plies = slot.board.moves();
        for (char digit : opening) slot.played.push_back(digit - '1');
    }
    slot.trees[0].reset(slot.board);
    slot.trees[1].reset(slot.board);
}

int GamePool::choose_column(Slot& slot, Tree& tree) {
    const auto visits = tree.root_child_visits();
    if (slot.board.moves() < params_.temperature_plies) {
        int total = 0;
        for (int count : visits) total += count;
        int ticket = slot.rng.below(total);
        for (int col = 0; col < kWidth; ++col) {
            ticket -= visits[col];
            if (ticket < 0) return col;
        }
    }
    return static_cast<int>(std::max_element(visits.begin(), visits.end()) - visits.begin());
}

int GamePool::builtin_move(Slot& slot, const Seat& seat, int worker) {
    switch (seat.kind) {
        case Seat::Kind::Random:
            return random_move(slot.board, slot.rng);
        case Seat::Kind::Tactical:
            return tactical_move(slot.board, slot.rng);
        case Seat::Kind::Rollout:
            return rollout_mcts_move(slot.board, seat.strength, params_.search.c_puct, slot.rng);
        case Seat::Kind::AlphaBeta:
            return alphabeta_move(slot.board, seat.strength, slot.rng);
        case Seat::Kind::Perfect:
            return perfect_move(*solvers_[worker]->analyze(slot.board), slot.rng);
        case Seat::Kind::Network:
            break;
    }
    throw std::logic_error("network seat has no builtin move");
}

void GamePool::make_move(Slot& slot, int column) {
    const bool won = slot.board.is_winning_move(column);
    const int mover_seat = slot.seat_to_move();
    slot.board.play(column);
    slot.played.push_back(column);
    slot.trees[0].advance(column);
    if (!params_.share_tree) slot.trees[1].advance(column);
    if (won) {
        finish_game(slot, mover_seat);
    } else if (slot.board.is_full()) {
        finish_game(slot, -1);
    }
}

void GamePool::finish_game(Slot& slot, int winner_seat) {
    GameResult& result = results_[slot.game_index];
    result.moves = moves_to_string(slot.played);
    result.winner_seat = winner_seat;
    result.first_seat = slot.first_seat;
    if (params_.record_samples) {
        // Samples alternate sides, the last belonging to the winner when there is one.
        const float final_outcome = winner_seat < 0 ? 0.0f : 1.0f;
        float outcome = final_outcome;
        for (auto sample = slot.samples.rbegin(); sample != slot.samples.rend(); ++sample) {
            sample->outcome = outcome;
            outcome = -outcome;
        }
        samples_[slot.game_index] = std::move(slot.samples);
        slot.samples = {};
    }
    games_done_.fetch_add(1);
    const int next = next_game_.fetch_add(1);
    if (next < params_.num_games) {
        start_game(slot, next);
    } else {
        slot.active = false;
    }
}

void GamePool::step_slot(Slot& slot, int worker) {
    while (slot.active) {
        const int seat_index = slot.seat_to_move();
        const Seat& seat = params_.seats[seat_index];
        if (seat.kind != Seat::Kind::Network) {
            make_move(slot, builtin_move(slot, seat, worker));
            continue;
        }
        const int tree_index = params_.share_tree ? 0 : seat_index;
        Tree& tree = slot.trees[tree_index];
        if (tree.root_visits() >= seat.simulations) {
            const int column = choose_column(slot, tree);
            if (params_.record_samples) {
                Sample sample;
                sample.current = slot.board.current();
                sample.occupied = slot.board.occupied();
                const auto visits = tree.root_child_visits();
                float total = 0.0f;
                for (int count : visits) total += static_cast<float>(count);
                for (int col = 0; col < kWidth; ++col) {
                    sample.policy[col] = static_cast<float>(visits[col]) / total;
                }
                slot.samples.push_back(sample);
            }
            make_move(slot, column);
            if (slot.active && params_.search.dirichlet_fraction > 0.0f) {
                slot.trees[0].add_root_noise(slot.rng);
            }
            continue;
        }

        Leaf leaf;
        while (static_cast<int>(slot.pending.size()) < params_.leaves_per_tree &&
               tree.root_visits() + static_cast<int>(slot.pending.size()) < seat.simulations) {
            const Tree::Selection outcome = tree.select(leaf);
            if (outcome == Tree::Selection::NeedsEvaluation) {
                slot.pending.push_back(leaf);
            } else if (outcome == Tree::Selection::Collision) {
                break;
            }
        }
        if (!slot.pending.empty()) {
            slot.pending_tree = tree_index;
            slot.pending_evaluator = seat.evaluator;
            return;
        }
    }
}

const LeafBatch& GamePool::collect() {
    const int count = static_cast<int>(slots_.size());
    parallel_for(count, params_.threads,
                 [&](int index, int worker) { step_slot(*slots_[index], worker); });
    std::vector<const std::vector<Leaf>*> lists;
    lists.reserve(count);
    for (auto& slot : slots_) lists.push_back(&slot->pending);
    build_batch(batch_, lists, params_.threads,
                [&](int index) { return slots_[index]->pending_evaluator; });
    evaluations_ += batch_.count();
    ++batches_;
    return batch_;
}

void GamePool::apply(const float* policy, const float* values) {
    const int count = static_cast<int>(slots_.size());
    std::vector<int> offsets(count + 1, 0);
    for (int index = 0; index < count; ++index) {
        offsets[index + 1] = offsets[index] + static_cast<int>(slots_[index]->pending.size());
    }
    parallel_for(count, params_.threads, [&](int index, int) {
        Slot& slot = *slots_[index];
        Tree& tree = slot.trees[slot.pending_tree];
        for (std::size_t item = 0; item < slot.pending.size(); ++item) {
            const int row = offsets[index] + static_cast<int>(item);
            const bool was_root = slot.pending[item].depth == 0;
            tree.expand(slot.pending[item], policy + std::size_t(row) * kWidth, values[row]);
            if (was_root) tree.add_root_noise(slot.rng);
        }
        slot.pending.clear();
    });
}

bool GamePool::finished() const { return games_done_.load() == params_.num_games; }

std::vector<Sample> GamePool::drain_samples() {
    std::vector<Sample> out;
    for (auto& game : samples_) {
        out.insert(out.end(), game.begin(), game.end());
        game = {};
    }
    return out;
}

std::vector<int> seat_moves(const std::vector<Board>& boards, const Seat& seat, float c_puct,
                            int threads, std::uint64_t seed) {
    std::vector<int> moves(boards.size());
    parallel_for(static_cast<int>(boards.size()), threads, [&](int item, int) {
        Rng rng(seed ^ (0x51ED2705ULL * static_cast<std::uint64_t>(item + 1)));
        switch (seat.kind) {
            case Seat::Kind::Random:
                moves[item] = random_move(boards[item], rng);
                break;
            case Seat::Kind::Tactical:
                moves[item] = tactical_move(boards[item], rng);
                break;
            case Seat::Kind::Rollout:
                moves[item] = rollout_mcts_move(boards[item], seat.strength, c_puct, rng);
                break;
            case Seat::Kind::AlphaBeta:
                moves[item] = alphabeta_move(boards[item], seat.strength, rng);
                break;
            case Seat::Kind::Network:
            case Seat::Kind::Perfect:
                throw std::invalid_argument("seat has no scripted move");
        }
    });
    return moves;
}

PositionSearch::PositionSearch(std::vector<Board> boards, SearchParams params, int simulations,
                               int leaves_per_tree, int threads)
    : simulations_(simulations), leaves_per_tree_(leaves_per_tree), threads_(threads) {
    trees_.reserve(boards.size());
    for (const Board& board : boards) {
        trees_.emplace_back(params);
        trees_.back().reset(board);
    }
    pending_.resize(trees_.size());
}

const LeafBatch& PositionSearch::collect() {
    const int count = static_cast<int>(trees_.size());
    parallel_for(count, threads_, [&](int index, int) {
        Tree& tree = trees_[index];
        auto& pending = pending_[index];
        Leaf leaf;
        while (static_cast<int>(pending.size()) < leaves_per_tree_ &&
               tree.root_visits() + static_cast<int>(pending.size()) < simulations_) {
            const Tree::Selection outcome = tree.select(leaf);
            if (outcome == Tree::Selection::NeedsEvaluation) {
                pending.push_back(leaf);
            } else if (outcome == Tree::Selection::Collision) {
                break;
            }
        }
    });
    std::vector<const std::vector<Leaf>*> lists;
    lists.reserve(count);
    for (auto& pending : pending_) lists.push_back(&pending);
    build_batch(batch_, lists, threads_, [](int) { return 0; });
    return batch_;
}

void PositionSearch::apply(const float* policy, const float* values) {
    const int count = static_cast<int>(trees_.size());
    std::vector<int> offsets(count + 1, 0);
    for (int index = 0; index < count; ++index) {
        offsets[index + 1] = offsets[index] + static_cast<int>(pending_[index].size());
    }
    parallel_for(count, threads_, [&](int index, int) {
        for (std::size_t item = 0; item < pending_[index].size(); ++item) {
            const int row = offsets[index] + static_cast<int>(item);
            trees_[index].expand(pending_[index][item], policy + std::size_t(row) * kWidth,
                                 values[row]);
        }
        pending_[index].clear();
    });
}

}  // namespace cz
