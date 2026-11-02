// PUCT tree search with deferred leaf evaluation.
//
// A search never calls the network. select() walks to a leaf and hands it back, and
// expand() delivers the evaluation later, possibly after other selections have run.
// That split is what allows many trees to share one network batch.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "board.hpp"
#include "rng.hpp"

namespace cz {

struct SearchParams {
    float c_puct = 1.5f;
    float dirichlet_alpha = 1.0f;
    float dirichlet_fraction = 0.0f;
};

struct Node {
    enum class State : std::uint8_t { Fresh, Pending, Expanded, Won, Drawn };

    float prior = 0.0f;
    float value_sum = 0.0f;  // from the view of the player who moved into this node
    std::int32_t visits = 0;
    std::int32_t first_child = -1;
    std::int16_t inflight = 0;  // selections currently passing through, not yet backed up
    std::uint8_t num_children = 0;
    std::int8_t column = -1;
    State state = State::Fresh;
};

struct Leaf {
    Board board;
    std::array<std::int32_t, kCells + 1> path{};
    int depth = 0;  // index of the leaf in path
};

class Tree {
   public:
    enum class Selection {
        NeedsEvaluation,  // `leaf` must be evaluated and passed to expand()
        Finished,         // reached a terminal node, already backed up
        Collision,        // ran into a leaf another selection is waiting on
    };

    explicit Tree(const SearchParams& params = {}) : params_(params) { reset(Board{}); }

    void reset(const Board& root);
    Selection select(Leaf& leaf);

    // Attaches children using `policy`, a probability per column that need not be
    // normalised or zero on illegal columns, and backs up `value`, which is the
    // leaf's value for its own side to move.
    void expand(const Leaf& leaf, const float* policy, float value);

    // Moves the root to a child, keeping that subtree's statistics.
    void advance(int column);

    void add_root_noise(Rng& rng);

    const Board& root_board() const { return root_board_; }
    int root_visits() const { return nodes_[kRoot].visits; }
    int root_inflight() const { return nodes_[kRoot].inflight; }
    bool root_expanded() const { return nodes_[kRoot].state == Node::State::Expanded; }

    // Mean value of the root for its side to move, in [-1, 1].
    float root_value() const;
    std::array<int, kWidth> root_child_visits() const;
    std::array<float, kWidth> root_child_priors() const;

    // Introspection for invariant checks in the tests.
    const std::vector<Node>& nodes() const { return nodes_; }

   private:
    static constexpr std::int32_t kRoot = 0;

    int best_child(const Node& parent) const;
    void backup(const Leaf& leaf, float value_for_leaf_mover);
    void back_out(const Leaf& leaf);

    SearchParams params_;
    Board root_board_;
    std::vector<Node> nodes_;
    std::vector<Node> scratch_;
};

}  // namespace cz
