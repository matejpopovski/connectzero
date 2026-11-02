// Exact Connect 4 solver: negamax with alpha-beta, a transposition table and a
// null-window search on the score.
//
// The techniques follow Pascal Pons' published solver tutorial (blog.gamesolver.org):
// threat-aware move generation, ordering by the number of winning cells a move
// creates, and narrowing the score with zero-width windows biased towards zero. The
// code here is an independent implementation with a simpler table layout.
//
// Score convention, always for the side to move:
//    0        the game is a draw under perfect play
//   +n        the side to move wins, with n - 1 of its own stones to spare
//   -n        the side to move loses, the opponent having n - 1 stones to spare
// So the magnitude measures how early the game ends, and the sign is the result.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "board.hpp"

namespace cz {

inline constexpr int kIllegalScore = 127;
using MoveScores = std::array<int, kWidth>;

// Safe to share between threads without locks. A slot is one 64-bit word holding the
// full key, so a reader sees either a complete entry or a key mismatch, and a lost
// update only costs a little repeated search. Relaxed ordering is enough because no
// other memory depends on a slot's contents.
class TranspositionTable {
   public:
    explicit TranspositionTable(int log2_entries);

    enum class Bound : std::uint64_t { Upper = 0, Lower = 1 };
    struct Hit {
        int score;
        Bound bound;
    };

    void store(Bits key, int score, Bound bound) {
        const std::uint64_t slot = (key << 7) | (static_cast<std::uint64_t>(bound) << 6) |
                                   static_cast<std::uint64_t>(score - kScoreBias);
        slots_[index(key)].store(slot, std::memory_order_relaxed);
    }

    std::optional<Hit> probe(Bits key) const {
        const std::uint64_t slot = slots_[index(key)].load(std::memory_order_relaxed);
        if (slot == 0 || (slot >> 7) != key) return std::nullopt;
        return Hit{static_cast<int>(slot & 63) + kScoreBias, static_cast<Bound>((slot >> 6) & 1)};
    }

    void clear();

   private:
    // Keys fit in 50 bits and scores in 6, so a probe can never return a false match.
    // Scores span [-21, 21] and are biased to be non-negative. Fibonacci hashing
    // keeps the regular structure of the keys from clustering in the index.
    static constexpr int kScoreBias = -24;

    std::size_t index(Bits key) const { return (key * 0x9E3779B97F4A7C15ULL) >> shift_; }

    std::size_t size_;
    std::unique_ptr<std::atomic<std::uint64_t>[]> slots_;
    int shift_;
};

class Solver {
   public:
    explicit Solver(int log2_entries = 23)
        : table_(std::make_shared<TranspositionTable>(log2_entries)) {}
    explicit Solver(std::shared_ptr<TranspositionTable> table) : table_(std::move(table)) {}

    // Exact score of the position for the side to move. The position must not be
    // finished. Returns nullopt if the node budget ran out (0 means unlimited).
    std::optional<int> solve(const Board& board, bool weak = false, std::int64_t budget = 0);

    // Score of playing each column, from the perspective of the side to move now.
    // Illegal columns hold kIllegalScore. With weak set, each score is clamped to
    // -1, 0 or 1 (result only, which is far cheaper to prove).
    std::optional<MoveScores> analyze(const Board& board, bool weak = false,
                                      std::int64_t budget = 0);

    std::int64_t nodes() const { return nodes_; }
    void reset_nodes() { nodes_ = 0; }
    void clear_table() { table_->clear(); }

   private:
    int negamax(const Board& board, int alpha, int beta);
    int window_search(const Board& board, int low, int high);

    std::shared_ptr<TranspositionTable> table_;
    std::int64_t nodes_ = 0;
    std::int64_t node_limit_ = 0;
    bool aborted_ = false;
};

// Score of a position in which the side to move has an immediate win.
constexpr int immediate_win_score(int moves) { return (kCells + 1 - moves) / 2; }

// Solves many positions on worker threads. Workers share one table, so related
// positions (the normal case for a benchmark or a database) build on each other.
std::vector<MoveScores> analyze_many(const std::vector<Board>& boards, int threads, bool weak,
                                     std::shared_ptr<TranspositionTable> table);
std::vector<int> solve_many(const std::vector<Board>& boards, int threads, bool weak,
                            std::shared_ptr<TranspositionTable> table);

}  // namespace cz
