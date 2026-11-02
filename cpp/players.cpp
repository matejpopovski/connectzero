#include "players.hpp"

#include <algorithm>
#include <limits>

namespace cz {

namespace {

constexpr int kWinValue = 1000000;

int pick(const std::array<int, kWidth>& cols, int count, Rng& rng) {
    return cols[rng.below(count)];
}

// Counts windows of four cells with no opposing stone, weighted by how filled they
// are. Precomputed once because the same 69 windows are visited at every leaf.
struct Windows {
    std::array<Bits, 69> cells{};
    constexpr Windows() {
        int next = 0;
        for (int col = 0; col < kWidth; ++col) {
            for (int row = 0; row < kHeight; ++row) {
                constexpr std::array<std::array<int, 2>, 4> steps = {
                    std::array<int, 2>{0, 1}, {1, 0}, {1, 1}, {1, -1}};
                for (const auto& step : steps) {
                    const int end_col = col + 3 * step[0];
                    const int end_row = row + 3 * step[1];
                    if (end_col >= kWidth || end_row < 0 || end_row >= kHeight) continue;
                    Bits window = 0;
                    for (int cell = 0; cell < 4; ++cell) {
                        window |= Bits{1}
                                  << ((col + cell * step[0]) * kStride + row + cell * step[1]);
                    }
                    cells[next++] = window;
                }
            }
        }
    }
};
constexpr Windows kWindows;

int evaluate(const Board& board) {
    constexpr std::array<int, 5> weight = {0, 1, 8, 40, 0};
    const Bits own = board.current();
    const Bits other = board.previous();
    int total = 0;
    for (Bits window : kWindows.cells) {
        const int mine = std::popcount(window & own);
        const int theirs = std::popcount(window & other);
        if (mine && theirs) continue;
        total += weight[mine] - weight[theirs];
    }
    return total;
}

int search(const Board& board, int depth, int alpha, int beta) {
    if (board.can_win_next()) return kWinValue - board.moves();
    if (board.moves() >= kCells - 1) return 0;
    if (depth == 0) return evaluate(board);
    const Bits safe = board.non_losing_moves();
    if (safe == 0) return -kWinValue + board.moves() + 1;
    for (int col : {3, 2, 4, 1, 5, 0, 6}) {
        if (!(safe & column_cells(col))) continue;
        const int score = -search(board.after(col), depth - 1, -beta, -alpha);
        if (score >= beta) return score;
        alpha = std::max(alpha, score);
    }
    return alpha;
}

}  // namespace

int random_move(const Board& board, Rng& rng) {
    std::array<int, kWidth> cols{};
    int count = 0;
    for (int col = 0; col < kWidth; ++col) {
        if (board.can_play(col)) cols[count++] = col;
    }
    return pick(cols, count, rng);
}

int tactical_move(const Board& board, Rng& rng) {
    std::array<int, kWidth> cols{};
    int count = 0;
    for (int col = 0; col < kWidth; ++col) {
        if (board.is_winning_move(col)) return col;
    }
    const Bits safe = board.non_losing_moves();
    for (int col = 0; col < kWidth; ++col) {
        if (safe & column_cells(col)) cols[count++] = col;
    }
    return count > 0 ? pick(cols, count, rng) : random_move(board, rng);
}

float random_playout(Board board, Rng& rng) {
    float sign = 1.0f;
    while (!board.is_full()) {
        const int col = random_move(board, rng);
        if (board.is_winning_move(col)) return sign;
        board.play(col);
        sign = -sign;
    }
    return 0.0f;
}

int rollout_mcts_move(const Board& board, int simulations, float c_puct, Rng& rng) {
    SearchParams params;
    params.c_puct = c_puct;
    Tree tree(params);
    tree.reset(board);
    const std::array<float, kWidth> uniform = {1, 1, 1, 1, 1, 1, 1};
    Leaf leaf;
    while (tree.root_visits() < simulations) {
        if (tree.select(leaf) == Tree::Selection::NeedsEvaluation) {
            tree.expand(leaf, uniform.data(), random_playout(leaf.board, rng));
        }
    }
    const auto visits = tree.root_child_visits();
    return static_cast<int>(std::max_element(visits.begin(), visits.end()) - visits.begin());
}

int alphabeta_move(const Board& board, int depth, Rng& rng) {
    std::array<int, kWidth> best{};
    int count = 0;
    int best_score = std::numeric_limits<int>::min();
    for (int col = 0; col < kWidth; ++col) {
        if (!board.can_play(col)) continue;
        if (board.is_winning_move(col)) return col;
        const Board child = board.after(col);
        const int score =
            child.is_full() ? 0 : -search(child, depth - 1, -kWinValue * 2, kWinValue * 2);
        if (score > best_score) {
            best_score = score;
            count = 0;
        }
        if (score == best_score) best[count++] = col;
    }
    return pick(best, count, rng);
}

int perfect_move(const MoveScores& scores, Rng& rng) {
    std::array<int, kWidth> best{};
    int count = 0;
    int best_score = std::numeric_limits<int>::min();
    for (int col = 0; col < kWidth; ++col) {
        if (scores[col] == kIllegalScore) continue;
        if (scores[col] > best_score) {
            best_score = scores[col];
            count = 0;
        }
        if (scores[col] == best_score) best[count++] = col;
    }
    return pick(best, count, rng);
}

}  // namespace cz
