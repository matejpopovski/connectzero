// Connect 4 position as two 64-bit words.
//
// Each column takes seven bits: six playable cells and one permanently empty
// separator on top. The separator is what makes shift-based line detection safe,
// because a run of stones can never wrap from the top of one column into the
// bottom of the next.
//
//   col:  0  1  2  3  4  5  6
//         6 13 20 27 34 41 48   separator, always empty
//         5 12 19 26 33 40 47   row 5 (top)
//         ...
//         0  7 14 21 28 35 42   row 0 (bottom)
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace cz {

using Bits = std::uint64_t;

inline constexpr int kWidth = 7;
inline constexpr int kHeight = 6;
inline constexpr int kCells = kWidth * kHeight;
inline constexpr int kStride = kHeight + 1;
inline constexpr int kPlanes = 3;

constexpr Bits column_cells(int col) { return ((Bits{1} << kHeight) - 1) << (col * kStride); }
constexpr Bits bottom_cell(int col) { return Bits{1} << (col * kStride); }
constexpr Bits top_cell(int col) { return Bits{1} << (kHeight - 1 + col * kStride); }

constexpr Bits bottom_row() {
    Bits row = 0;
    for (int col = 0; col < kWidth; ++col) row |= bottom_cell(col);
    return row;
}

constexpr Bits all_cells() { return bottom_row() * ((Bits{1} << kHeight) - 1); }

inline constexpr std::array<int, 4> kDirections = {1, kStride - 1, kStride, kStride + 1};

constexpr bool has_four(Bits stones) {
    for (int step : kDirections) {
        const Bits pairs = stones & (stones >> step);
        if (pairs & (pairs >> (2 * step))) return true;
    }
    return false;
}

// Empty cells that would complete four in a row for `stones`. Includes cells that
// are not yet playable, which the solver needs for threat analysis.
constexpr Bits completing_cells(Bits stones, Bits occupied) {
    Bits hits = 0;
    for (int step : kDirections) {
        const Bits below1 = stones << step;
        const Bits below2 = stones << (2 * step);
        const Bits below3 = stones << (3 * step);
        const Bits above1 = stones >> step;
        const Bits above2 = stones >> (2 * step);
        const Bits above3 = stones >> (3 * step);
        hits |= below1 & below2 & below3;  // xxx_
        hits |= above1 & above2 & above3;  // _xxx
        hits |= below1 & above1 & above2;  // x_xx
        hits |= below1 & below2 & above1;  // xx_x
    }
    return hits & all_cells() & ~occupied;
}

class Board {
   public:
    constexpr Board() = default;
    constexpr Board(Bits current, Bits occupied)
        : current_(current), occupied_(occupied), moves_(std::popcount(occupied)) {}

    // Column digits '1' to '7', the notation used in the Connect 4 solving literature.
    static Board from_moves(std::string_view text) {
        Board board;
        for (char digit : text) {
            const int col = digit - '1';
            if (col < 0 || col >= kWidth) throw std::invalid_argument("column digit out of range");
            if (!board.can_play(col)) throw std::invalid_argument("column is full");
            if (board.is_over()) throw std::invalid_argument("game already over");
            board.play(col);
        }
        return board;
    }

    constexpr Bits current() const { return current_; }
    constexpr Bits occupied() const { return occupied_; }
    constexpr int moves() const { return moves_; }

    // The side to move is whoever has not just played, so the mover's stones are
    // recovered by xor and flip on every ply without any branching on colour.
    constexpr Bits previous() const { return current_ ^ occupied_; }

    constexpr bool can_play(int col) const { return (occupied_ & top_cell(col)) == 0; }

    constexpr void play(int col) {
        current_ ^= occupied_;
        occupied_ |= occupied_ + bottom_cell(col);
        ++moves_;
    }

    constexpr Board after(int col) const {
        Board next = *this;
        next.play(col);
        return next;
    }

    // Plays a single-bit move produced by playable(). Same effect as play(col).
    constexpr void play_bit(Bits move) {
        current_ ^= occupied_;
        occupied_ |= move;
        ++moves_;
    }

    constexpr Bits playable() const { return (occupied_ + bottom_row()) & all_cells(); }

    constexpr bool is_winning_move(int col) const {
        return (completing_cells(current_, occupied_) & playable() & column_cells(col)) != 0;
    }

    constexpr bool can_win_next() const {
        return (completing_cells(current_, occupied_) & playable()) != 0;
    }

    constexpr bool last_move_won() const { return has_four(previous()); }
    constexpr bool is_full() const { return moves_ == kCells; }
    constexpr bool is_over() const { return is_full() || last_move_won(); }

    // Moves that do not hand the opponent an immediate win. Assumes the side to
    // move cannot win at once. Returns 0 when every move loses by the next ply.
    constexpr Bits non_losing_moves() const {
        Bits safe = playable();
        const Bits threats = completing_cells(previous(), occupied_);
        const Bits forced = safe & threats;
        if (forced) {
            if (forced & (forced - 1)) return 0;  // two immediate threats cannot both be blocked
            safe = forced;
        }
        return safe & ~(threats >> 1);  // never play directly beneath an opposing win
    }

    // A unique perfect encoding: within a column, current + occupied places a
    // marker bit above the stack, so height and colours both survive and no
    // carry reaches the next column.
    constexpr Bits key() const { return current_ + occupied_; }

    constexpr Board mirrored() const { return {mirror_bits(current_), mirror_bits(occupied_)}; }

    constexpr Bits canonical_key() const {
        const Bits own = key();
        const Bits flipped = mirrored().key();
        return own < flipped ? own : flipped;
    }

    std::vector<int> legal_moves() const {
        std::vector<int> cols;
        if (is_over()) return cols;
        for (int col = 0; col < kWidth; ++col) {
            if (can_play(col)) cols.push_back(col);
        }
        return cols;
    }

    constexpr int height(int col) const { return std::popcount(occupied_ & column_cells(col)); }

    constexpr bool operator==(const Board& other) const {
        return current_ == other.current_ && occupied_ == other.occupied_;
    }

   private:
    static constexpr Bits mirror_bits(Bits bits) {
        Bits out = 0;
        const Bits column = (Bits{1} << kStride) - 1;
        for (int col = 0; col < kWidth; ++col) {
            out |= ((bits >> (col * kStride)) & column) << ((kWidth - 1 - col) * kStride);
        }
        return out;
    }

    Bits current_ = 0;
    Bits occupied_ = 0;
    int moves_ = 0;
};

// Network input, CHW with row 0 at the bottom. The third plane is 1 when the side
// to move played first: Connect 4 strategy depends on parity (zugzwang control), and
// although parity can be derived by counting stones a convolution cannot easily do so.
inline void encode_planes(const Board& board, float* out) {
    const Bits own = board.current();
    const Bits other = board.previous();
    const float first = (board.moves() % 2 == 0) ? 1.0f : 0.0f;
    for (int row = 0; row < kHeight; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            const Bits cell = Bits{1} << (col * kStride + row);
            const int offset = row * kWidth + col;
            out[offset] = (own & cell) ? 1.0f : 0.0f;
            out[kCells + offset] = (other & cell) ? 1.0f : 0.0f;
            out[2 * kCells + offset] = first;
        }
    }
}

inline std::string moves_to_string(const std::vector<int>& cols) {
    std::string text;
    text.reserve(cols.size());
    for (int col : cols) text.push_back(static_cast<char>('1' + col));
    return text;
}

}  // namespace cz
