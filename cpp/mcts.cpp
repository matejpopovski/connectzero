#include "mcts.hpp"

#include <cmath>
#include <queue>
#include <random>
#include <utility>

namespace cz {

void Tree::reset(const Board& root) {
    root_board_ = root;
    nodes_.clear();
    nodes_.emplace_back();
}

int Tree::best_child(const Node& parent) const {
    const std::int32_t begin = parent.first_child;
    const float total = static_cast<float>(parent.visits + parent.inflight);
    const float scale = params_.c_puct * std::sqrt(std::max(total, 1.0f));
    int best = begin;
    float best_score = -1e9f;
    for (std::int32_t index = begin; index < begin + parent.num_children; ++index) {
        const Node& child = nodes_[index];
        const float seen = static_cast<float>(child.visits + child.inflight);
        // Each in-flight selection counts as a visit that lost. The penalty steers
        // later selections in the same batch towards different leaves.
        const float mean =
            seen > 0.0f ? (child.value_sum - static_cast<float>(child.inflight)) / seen : 0.0f;
        const float score = mean + scale * child.prior / (1.0f + seen);
        if (score > best_score) {
            best_score = score;
            best = index;
        }
    }
    return best;
}

Tree::Selection Tree::select(Leaf& leaf) {
    leaf.board = root_board_;
    leaf.depth = 0;
    leaf.path[0] = kRoot;
    std::int32_t index = kRoot;
    while (nodes_[index].state == Node::State::Expanded) {
        ++nodes_[index].inflight;
        index = best_child(nodes_[index]);
        leaf.board.play(nodes_[index].column);
        leaf.path[++leaf.depth] = index;
    }
    Node& node = nodes_[index];
    switch (node.state) {
        case Node::State::Pending:
            back_out(leaf);
            return Selection::Collision;
        case Node::State::Won:
            ++node.inflight;
            backup(leaf, 1.0f);
            return Selection::Finished;
        case Node::State::Drawn:
            ++node.inflight;
            backup(leaf, 0.0f);
            return Selection::Finished;
        case Node::State::Fresh:
        case Node::State::Expanded:
            break;
    }
    node.state = Node::State::Pending;
    ++node.inflight;
    return Selection::NeedsEvaluation;
}

void Tree::back_out(const Leaf& leaf) {
    for (int depth = 0; depth < leaf.depth; ++depth) --nodes_[leaf.path[depth]].inflight;
}

void Tree::backup(const Leaf& leaf, float value_for_leaf_mover) {
    float value = value_for_leaf_mover;
    for (int depth = leaf.depth; depth >= 0; --depth) {
        Node& node = nodes_[leaf.path[depth]];
        --node.inflight;
        ++node.visits;
        node.value_sum += value;
        value = -value;
    }
}

void Tree::expand(const Leaf& leaf, const float* policy, float value) {
    const Board& board = leaf.board;
    const auto first = static_cast<std::int32_t>(nodes_.size());
    float total = 0.0f;
    int count = 0;
    for (int col = 0; col < kWidth; ++col) {
        if (!board.can_play(col)) continue;
        Node child;
        child.column = static_cast<std::int8_t>(col);
        child.prior = policy[col];
        total += policy[col];
        if (board.is_winning_move(col)) {
            child.state = Node::State::Won;
        } else if (board.moves() + 1 == kCells) {
            child.state = Node::State::Drawn;
        }
        nodes_.push_back(child);
        ++count;
    }
    const float uniform = 1.0f / static_cast<float>(count);
    for (std::int32_t index = first; index < first + count; ++index) {
        nodes_[index].prior = total > 0.0f ? nodes_[index].prior / total : uniform;
    }
    Node& node = nodes_[leaf.path[leaf.depth]];
    node.first_child = first;
    node.num_children = static_cast<std::uint8_t>(count);
    node.state = Node::State::Expanded;
    backup(leaf, -value);
}

void Tree::advance(int column) {
    const Node& root = nodes_[kRoot];
    std::int32_t chosen = -1;
    for (std::int32_t index = root.first_child;
         root.first_child >= 0 && index < root.first_child + root.num_children; ++index) {
        if (nodes_[index].column == column) chosen = index;
    }
    root_board_.play(column);
    if (chosen < 0) {
        const Board board = root_board_;
        reset(board);
        return;
    }

    // Copy the kept subtree breadth first into fresh storage. A game abandons most of
    // the tree each move, and compacting keeps memory bounded by one search.
    scratch_.clear();
    scratch_.push_back(nodes_[chosen]);
    std::queue<std::int32_t> pending;
    pending.push(0);
    while (!pending.empty()) {
        const std::int32_t copied = pending.front();
        pending.pop();
        const std::int32_t old_first = scratch_[copied].first_child;
        const int count = scratch_[copied].num_children;
        if (old_first < 0) continue;
        const auto new_first = static_cast<std::int32_t>(scratch_.size());
        scratch_[copied].first_child = new_first;
        for (int offset = 0; offset < count; ++offset) {
            scratch_.push_back(nodes_[old_first + offset]);
            pending.push(new_first + offset);
        }
    }
    std::swap(nodes_, scratch_);
}

void Tree::add_root_noise(Rng& rng) {
    Node& root = nodes_[kRoot];
    if (params_.dirichlet_fraction <= 0.0f || root.first_child < 0) return;
    std::gamma_distribution<float> gamma(params_.dirichlet_alpha, 1.0f);
    std::array<float, kWidth> noise{};
    float total = 0.0f;
    for (int offset = 0; offset < root.num_children; ++offset) {
        noise[offset] = gamma(rng);
        total += noise[offset];
    }
    if (total <= 0.0f) return;
    for (int offset = 0; offset < root.num_children; ++offset) {
        Node& child = nodes_[root.first_child + offset];
        child.prior = (1.0f - params_.dirichlet_fraction) * child.prior +
                      params_.dirichlet_fraction * noise[offset] / total;
    }
}

float Tree::root_value() const {
    const Node& root = nodes_[kRoot];
    return root.visits > 0 ? -root.value_sum / static_cast<float>(root.visits) : 0.0f;
}

std::array<int, kWidth> Tree::root_child_visits() const {
    std::array<int, kWidth> visits{};
    const Node& root = nodes_[kRoot];
    for (int offset = 0; root.first_child >= 0 && offset < root.num_children; ++offset) {
        const Node& child = nodes_[root.first_child + offset];
        visits[child.column] = child.visits;
    }
    return visits;
}

std::array<float, kWidth> Tree::root_child_priors() const {
    std::array<float, kWidth> priors{};
    const Node& root = nodes_[kRoot];
    for (int offset = 0; root.first_child >= 0 && offset < root.num_children; ++offset) {
        const Node& child = nodes_[root.first_child + offset];
        priors[child.column] = child.prior;
    }
    return priors;
}

}  // namespace cz
