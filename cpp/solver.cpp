#include "solver.hpp"

#include <algorithm>
#include <bit>
#include <memory>

#include "parallel.hpp"

namespace cz {

namespace {

// Centre columns take part in more lines, so trying them first gives earlier cutoffs.
constexpr std::array<int, kWidth> kColumnOrder = {3, 2, 4, 1, 5, 0, 6};

constexpr int lowest_score(int moves) { return -(kCells - moves) / 2; }
constexpr int highest_score(int moves) { return (kCells + 1 - moves) / 2; }

struct OrderedMove {
    Bits move;
    int rank;
};

// Insertion sort over at most seven moves, stable so that equal ranks keep the
// centre-first column order.
class MoveList {
   public:
    void push(Bits move, int rank) {
        int slot = size_++;
        while (slot > 0 && moves_[slot - 1].rank < rank) {
            moves_[slot] = moves_[slot - 1];
            --slot;
        }
        moves_[slot] = {move, rank};
    }
    const OrderedMove* begin() const { return moves_.data(); }
    const OrderedMove* end() const { return moves_.data() + size_; }

   private:
    std::array<OrderedMove, kWidth> moves_{};
    int size_ = 0;
};

}  // namespace

TranspositionTable::TranspositionTable(int log2_entries)
    : size_(std::size_t{1} << log2_entries),
      slots_(std::make_unique<std::atomic<std::uint64_t>[]>(size_)),
      shift_(64 - log2_entries) {}

void TranspositionTable::clear() {
    for (std::size_t slot = 0; slot < size_; ++slot) {
        slots_[slot].store(0, std::memory_order_relaxed);
    }
}

int Solver::negamax(const Board& board, int alpha, int beta) {
    ++nodes_;
    if (node_limit_ != 0 && nodes_ > node_limit_) {
        aborted_ = true;
        return alpha;
    }

    const Bits safe = board.non_losing_moves();
    if (safe == 0) return lowest_score(board.moves());
    const int moves = board.moves();
    if (moves >= kCells - 2) return 0;

    // The opponent cannot win on the next ply, and we cannot win on this one, so the
    // earliest possible finishes are excluded from the window.
    int floor = lowest_score(moves + 2);
    int ceiling = highest_score(moves + 2);
    alpha = std::max(alpha, floor);
    beta = std::min(beta, ceiling);
    if (alpha >= beta) return alpha;

    // Mirrored positions have the same value, which matters most near the opening,
    // where the mirror of a line is usually reachable.
    const Bits key = moves < 24 ? board.canonical_key() : board.key();
    if (const auto hit = table_->probe(key)) {
        if (hit->bound == TranspositionTable::Bound::Upper) {
            beta = std::min(beta, hit->score);
        } else {
            alpha = std::max(alpha, hit->score);
        }
        if (alpha >= beta) return alpha;
    }

    MoveList ordered;
    for (int col : kColumnOrder) {
        const Bits move = safe & column_cells(col);
        if (move == 0) continue;
        const int threats =
            std::popcount(completing_cells(board.current() | move, board.occupied() | move));
        ordered.push(move, threats);
    }

    for (const OrderedMove& entry : ordered) {
        Board next = board;
        next.play_bit(entry.move);
        const int score = -negamax(next, -beta, -alpha);
        if (aborted_) return alpha;
        if (score >= beta) {
            table_->store(key, score, TranspositionTable::Bound::Lower);
            return score;
        }
        alpha = std::max(alpha, score);
    }
    table_->store(key, alpha, TranspositionTable::Bound::Upper);
    return alpha;
}

int Solver::window_search(const Board& board, int low, int high) {
    while (low < high && !aborted_) {
        // Bisect, but lean towards zero: results near a draw are the common case
        // and small-magnitude windows are far cheaper to refute.
        int probe = low + (high - low) / 2;
        if (probe <= 0 && low / 2 < probe) {
            probe = low / 2;
        } else if (probe >= 0 && high / 2 > probe) {
            probe = high / 2;
        }
        const int score = negamax(board, probe, probe + 1);
        if (score <= probe) {
            high = score;
        } else {
            low = score;
        }
    }
    return low;
}

std::optional<int> Solver::solve(const Board& board, bool weak, std::int64_t budget) {
    if (board.can_win_next()) {
        const int score = immediate_win_score(board.moves());
        return weak ? 1 : score;
    }
    aborted_ = false;
    node_limit_ = budget == 0 ? 0 : nodes_ + budget;
    const int low = weak ? -1 : lowest_score(board.moves());
    const int high = weak ? 1 : highest_score(board.moves());
    const int score = window_search(board, low, high);
    node_limit_ = 0;
    if (aborted_) return std::nullopt;
    // A fail-soft bound can land outside the weak window, but only its sign is proved.
    if (weak) return (score > 0) - (score < 0);
    return score;
}

std::optional<MoveScores> Solver::analyze(const Board& board, bool weak, std::int64_t budget) {
    MoveScores scores;
    scores.fill(kIllegalScore);
    const std::int64_t stop = budget == 0 ? 0 : nodes_ + budget;
    for (int col : kColumnOrder) {
        if (!board.can_play(col)) continue;
        if (board.is_winning_move(col)) {
            scores[col] = weak ? 1 : immediate_win_score(board.moves());
            continue;
        }
        const Board child = board.after(col);
        if (child.is_full()) {
            scores[col] = 0;
            continue;
        }
        const std::int64_t left = stop == 0 ? 0 : std::max<std::int64_t>(stop - nodes_, 1);
        const auto reply = solve(child, weak, left);
        if (!reply) return std::nullopt;
        scores[col] = -*reply;
    }
    return scores;
}

namespace {

template <typename Result, typename Work>
std::vector<Result> over_boards(const std::vector<Board>& boards, int threads, int log2_entries,
                                Work work) {
    std::vector<Result> out(boards.size());
    const int workers = std::max(1, std::min<int>(threads, static_cast<int>(boards.size())));
    auto table = std::make_shared<TranspositionTable>(log2_entries);
    std::vector<std::unique_ptr<Solver>> solvers;
    for (int worker = 0; worker < workers; ++worker) {
        solvers.push_back(std::make_unique<Solver>(table));
    }
    parallel_for(static_cast<int>(boards.size()), workers,
                 [&](int item, int worker) { out[item] = work(*solvers[worker], boards[item]); });
    return out;
}

}  // namespace

std::vector<MoveScores> analyze_many(const std::vector<Board>& boards, int threads, bool weak,
                                     int log2_entries) {
    return over_boards<MoveScores>(
        boards, threads, log2_entries,
        [weak](Solver& solver, const Board& board) { return *solver.analyze(board, weak); });
}

std::vector<int> solve_many(const std::vector<Board>& boards, int threads, bool weak,
                            int log2_entries) {
    return over_boards<int>(
        boards, threads, log2_entries,
        [weak](Solver& solver, const Board& board) { return *solver.solve(board, weak); });
}

}  // namespace cz
