#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <memory>
#include <optional>

#include "board.hpp"
#include "mcts.hpp"
#include "players.hpp"
#include "pool.hpp"
#include "solver.hpp"

namespace py = pybind11;
using namespace cz;

namespace {

using FloatInput = py::array_t<float, py::array::c_style | py::array::forcecast>;
using BitsInput = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>;

std::vector<int> scores_to_list(const MoveScores& scores) { return {scores.begin(), scores.end()}; }

py::array_t<float> planes_array(const float* data, std::size_t rows) {
    py::array_t<float> out({rows, std::size_t{kPlanes}, std::size_t{kHeight}, std::size_t{kWidth}});
    std::copy_n(data, rows * kPlanes * kCells, out.mutable_data());
    return out;
}

py::tuple batch_arrays(const LeafBatch& batch) {
    const auto rows = static_cast<std::size_t>(batch.count());
    py::array_t<std::int32_t> evaluators(static_cast<py::ssize_t>(rows));
    std::copy_n(batch.evaluators.data(), rows, evaluators.mutable_data());
    return py::make_tuple(planes_array(batch.planes.data(), rows), evaluators);
}

void check_answers(const FloatInput& policy, const FloatInput& values, int rows) {
    if (policy.ndim() != 2 || policy.shape(0) != rows || policy.shape(1) != kWidth) {
        throw std::invalid_argument("policy must have shape (batch, 7)");
    }
    if (values.ndim() != 1 || values.shape(0) != rows) {
        throw std::invalid_argument("values must have shape (batch,)");
    }
}

}  // namespace

PYBIND11_MODULE(_engine, m) {
    m.doc() = "Connect 4 bitboards, exact solver, and batched PUCT search.";
    m.attr("WIDTH") = kWidth;
    m.attr("HEIGHT") = kHeight;
    m.attr("PLANES") = kPlanes;
    m.attr("ILLEGAL") = kIllegalScore;

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<Bits, Bits>(), py::arg("current"), py::arg("occupied"))
        .def_static("from_moves", &Board::from_moves, py::arg("moves"))
        .def_property_readonly("current", &Board::current)
        .def_property_readonly("occupied", &Board::occupied)
        .def_property_readonly("previous", &Board::previous)
        .def_property_readonly("moves", &Board::moves)
        .def("can_play", &Board::can_play, py::arg("column"))
        .def("play", &Board::after, py::arg("column"), "Return the position after a move.")
        .def("is_winning_move", &Board::is_winning_move, py::arg("column"))
        .def("can_win_next", &Board::can_win_next)
        .def("last_move_won", &Board::last_move_won)
        .def("is_full", &Board::is_full)
        .def("is_over", &Board::is_over)
        .def("legal_moves", &Board::legal_moves)
        .def("height", &Board::height, py::arg("column"))
        .def("non_losing_columns",
             [](const Board& board) {
                 std::vector<int> cols;
                 const Bits safe = board.non_losing_moves();
                 for (int col = 0; col < kWidth; ++col) {
                     if (safe & column_cells(col)) cols.push_back(col);
                 }
                 return cols;
             })
        .def("key", &Board::key)
        .def("canonical_key", &Board::canonical_key)
        .def("mirrored", &Board::mirrored)
        .def("planes",
             [](const Board& board) {
                 std::array<float, kPlanes * kCells> buffer;
                 encode_planes(board, buffer.data());
                 py::array_t<float> out({kPlanes, kHeight, kWidth});
                 std::copy(buffer.begin(), buffer.end(), out.mutable_data());
                 return out;
             })
        .def("__eq__", &Board::operator==)
        .def("__hash__", &Board::key)
        .def("__repr__", [](const Board& board) {
            return "Board(moves=" + std::to_string(board.moves()) +
                   ", key=" + std::to_string(board.key()) + ")";
        });

    m.def("has_four", [](Bits stones) { return has_four(stones); }, py::arg("stones"));
    m.def(
        "encode_batch",
        [](const BitsInput& current, const BitsInput& occupied) {
            const auto rows = static_cast<std::size_t>(current.size());
            if (occupied.size() != current.size()) {
                throw std::invalid_argument("current and occupied differ in length");
            }
            py::array_t<float> out(
                {rows, std::size_t{kPlanes}, std::size_t{kHeight}, std::size_t{kWidth}});
            float* data = out.mutable_data();
            const Bits* own = current.data();
            const Bits* all = occupied.data();
            {
                py::gil_scoped_release nogil;
                for (std::size_t row = 0; row < rows; ++row) {
                    encode_planes(Board(own[row], all[row]), data + row * kPlanes * kCells);
                }
            }
            return out;
        },
        py::arg("current"), py::arg("occupied"));

    py::class_<TranspositionTable, std::shared_ptr<TranspositionTable>>(m, "TranspositionTable")
        .def(py::init<int>(), py::arg("log2_entries"));

    py::class_<Solver>(m, "Solver")
        .def(py::init<int>(), py::arg("log2_entries") = 23)
        .def(py::init<std::shared_ptr<TranspositionTable>>(), py::arg("table"))
        .def(
            "solve",
            [](Solver& solver, const Board& board, bool weak, std::int64_t budget) {
                py::gil_scoped_release nogil;
                return solver.solve(board, weak, budget);
            },
            py::arg("board"), py::arg("weak") = false, py::arg("budget") = 0,
            "Score for the side to move, or None if the node budget ran out.")
        .def(
            "analyze",
            [](Solver& solver, const Board& board, bool weak,
               std::int64_t budget) -> std::optional<std::vector<int>> {
                std::optional<MoveScores> scores;
                {
                    py::gil_scoped_release nogil;
                    scores = solver.analyze(board, weak, budget);
                }
                if (!scores) return std::nullopt;
                return scores_to_list(*scores);
            },
            py::arg("board"), py::arg("weak") = false, py::arg("budget") = 0,
            "Score of each column for the side to move; ILLEGAL marks full columns.")
        .def_property_readonly("nodes", &Solver::nodes)
        .def("reset_nodes", &Solver::reset_nodes)
        .def("clear_table", &Solver::clear_table);

    m.def(
        "analyze_many",
        [](const std::vector<Board>& boards, int threads, bool weak, int log2_entries,
           std::shared_ptr<TranspositionTable> table) {
            std::vector<MoveScores> scores;
            {
                py::gil_scoped_release nogil;
                if (!table) table = std::make_shared<TranspositionTable>(log2_entries);
                scores = analyze_many(boards, threads, weak, table);
            }
            const auto rows = static_cast<py::ssize_t>(scores.size());
            py::array_t<std::int8_t> out({rows, py::ssize_t{kWidth}});
            auto view = out.mutable_unchecked<2>();
            for (py::ssize_t row = 0; row < rows; ++row) {
                for (int col = 0; col < kWidth; ++col) {
                    view(row, col) = static_cast<std::int8_t>(scores[row][col]);
                }
            }
            return out;
        },
        py::arg("boards"), py::arg("threads") = 1, py::arg("weak") = false,
        py::arg("log2_entries") = 23, py::arg("table") = nullptr);

    m.def(
        "solve_many",
        [](const std::vector<Board>& boards, int threads, bool weak, int log2_entries,
           std::shared_ptr<TranspositionTable> table) {
            py::gil_scoped_release nogil;
            if (!table) table = std::make_shared<TranspositionTable>(log2_entries);
            return solve_many(boards, threads, weak, table);
        },
        py::arg("boards"), py::arg("threads") = 1, py::arg("weak") = false,
        py::arg("log2_entries") = 23, py::arg("table") = nullptr);

    py::class_<Rng>(m, "Rng").def(py::init<std::uint64_t>(), py::arg("seed") = 0);
    m.def("random_move", &random_move, py::arg("board"), py::arg("rng"));
    m.def("tactical_move", &tactical_move, py::arg("board"), py::arg("rng"));
    m.def("alphabeta_move", &alphabeta_move, py::arg("board"), py::arg("depth"), py::arg("rng"));
    m.def("rollout_mcts_move", &rollout_mcts_move, py::arg("board"), py::arg("simulations"),
          py::arg("c_puct"), py::arg("rng"));
    m.def(
        "perfect_move",
        [](const std::vector<int>& scores, Rng& rng) {
            MoveScores fixed;
            std::copy_n(scores.begin(), kWidth, fixed.begin());
            return perfect_move(fixed, rng);
        },
        py::arg("scores"), py::arg("rng"));

    py::class_<SearchParams>(m, "SearchParams")
        .def(py::init([](float c_puct, float dirichlet_alpha, float dirichlet_fraction) {
                 SearchParams params;
                 params.c_puct = c_puct;
                 params.dirichlet_alpha = dirichlet_alpha;
                 params.dirichlet_fraction = dirichlet_fraction;
                 return params;
             }),
             py::arg("c_puct") = 1.5f, py::arg("dirichlet_alpha") = 1.0f,
             py::arg("dirichlet_fraction") = 0.0f);

    py::class_<Node>(m, "Node")
        .def_readonly("prior", &Node::prior)
        .def_readonly("value_sum", &Node::value_sum)
        .def_readonly("visits", &Node::visits)
        .def_readonly("first_child", &Node::first_child)
        .def_readonly("inflight", &Node::inflight)
        .def_readonly("num_children", &Node::num_children)
        .def_readonly("column", &Node::column);

    py::enum_<Tree::Selection>(m, "Selection")
        .value("NEEDS_EVALUATION", Tree::Selection::NeedsEvaluation)
        .value("FINISHED", Tree::Selection::Finished)
        .value("COLLISION", Tree::Selection::Collision);

    py::class_<Leaf>(m, "Leaf")
        .def(py::init<>())
        .def_readonly("board", &Leaf::board)
        .def_readonly("depth", &Leaf::depth);

    py::class_<Tree>(m, "Tree")
        .def(py::init<const SearchParams&>(), py::arg("params") = SearchParams{})
        .def("reset", &Tree::reset, py::arg("board"))
        .def(
            "select", [](Tree& tree, Leaf& leaf) { return tree.select(leaf); }, py::arg("leaf"))
        .def(
            "expand",
            [](Tree& tree, const Leaf& leaf, const std::vector<float>& policy, float value) {
                if (policy.size() != kWidth) throw std::invalid_argument("policy needs 7 entries");
                tree.expand(leaf, policy.data(), value);
            },
            py::arg("leaf"), py::arg("policy"), py::arg("value"))
        .def("advance", &Tree::advance, py::arg("column"))
        .def(
            "add_root_noise", [](Tree& tree, Rng& rng) { tree.add_root_noise(rng); },
            py::arg("rng"))
        .def_property_readonly("root_board", &Tree::root_board)
        .def_property_readonly("root_visits", &Tree::root_visits)
        .def_property_readonly("root_value", &Tree::root_value)
        .def_property_readonly("child_visits", &Tree::root_child_visits)
        .def_property_readonly("child_priors", &Tree::root_child_priors)
        .def_property_readonly("nodes", &Tree::nodes);

    py::class_<Seat>(m, "Seat")
        .def_static(
            "network",
            [](int evaluator, int simulations) {
                Seat seat;
                seat.evaluator = evaluator;
                seat.simulations = simulations;
                return seat;
            },
            py::arg("evaluator") = 0, py::arg("simulations") = 200)
        .def_static("random",
                    [] {
                        Seat seat;
                        seat.kind = Seat::Kind::Random;
                        return seat;
                    })
        .def_static("tactical",
                    [] {
                        Seat seat;
                        seat.kind = Seat::Kind::Tactical;
                        return seat;
                    })
        .def_static(
            "rollout",
            [](int simulations) {
                Seat seat;
                seat.kind = Seat::Kind::Rollout;
                seat.strength = simulations;
                return seat;
            },
            py::arg("simulations"))
        .def_static(
            "alphabeta",
            [](int depth) {
                Seat seat;
                seat.kind = Seat::Kind::AlphaBeta;
                seat.strength = depth;
                return seat;
            },
            py::arg("depth"))
        .def_static("perfect", [] {
            Seat seat;
            seat.kind = Seat::Kind::Perfect;
            return seat;
        });

    m.def(
        "seat_moves",
        [](const std::vector<Board>& boards, const Seat& seat, float c_puct, int threads,
           std::uint64_t seed) {
            py::gil_scoped_release nogil;
            return seat_moves(boards, seat, c_puct, threads, seed);
        },
        py::arg("boards"), py::arg("seat"), py::arg("c_puct") = 1.5f, py::arg("threads") = 1,
        py::arg("seed") = 0);

    py::class_<GameResult>(m, "GameResult")
        .def_readonly("moves", &GameResult::moves)
        .def_readonly("winner_seat", &GameResult::winner_seat)
        .def_readonly("first_seat", &GameResult::first_seat);

    py::class_<GamePool>(m, "GamePool")
        .def(py::init([](Seat seat_a, Seat seat_b, int num_games, int concurrency,
                         int leaves_per_tree, int threads, std::uint64_t seed, SearchParams search,
                         int temperature_plies, bool share_tree, bool record_samples,
                         std::vector<std::string> openings,
                         std::shared_ptr<TranspositionTable> solver_table) {
                 PoolParams params;
                 params.seats = {seat_a, seat_b};
                 params.num_games = num_games;
                 params.concurrency = concurrency;
                 params.leaves_per_tree = leaves_per_tree;
                 params.threads = threads;
                 params.seed = seed;
                 params.search = search;
                 params.temperature_plies = temperature_plies;
                 params.share_tree = share_tree;
                 params.record_samples = record_samples;
                 params.openings = std::move(openings);
                 params.solver_table = std::move(solver_table);
                 return std::make_unique<GamePool>(std::move(params));
             }),
             py::arg("seat_a"), py::arg("seat_b"), py::arg("num_games"), py::arg("concurrency"),
             py::arg("leaves_per_tree") = 1, py::arg("threads") = 1, py::arg("seed") = 0,
             py::arg("search") = SearchParams{}, py::arg("temperature_plies") = 0,
             py::arg("share_tree") = false, py::arg("record_samples") = false,
             py::arg("openings") = std::vector<std::string>{}, py::arg("solver_table") = nullptr)
        .def("collect",
             [](GamePool& pool) {
                 const LeafBatch* batch = nullptr;
                 {
                     py::gil_scoped_release nogil;
                     batch = &pool.collect();
                 }
                 return batch_arrays(*batch);
             })
        .def(
            "apply",
            [](GamePool& pool, const FloatInput& policy, const FloatInput& values, int rows) {
                check_answers(policy, values, rows);
                const float* policy_data = policy.data();
                const float* value_data = values.data();
                py::gil_scoped_release nogil;
                pool.apply(policy_data, value_data);
            },
            py::arg("policy"), py::arg("values"), py::arg("rows"))
        .def_property_readonly("finished", &GamePool::finished)
        .def_property_readonly("games_done", &GamePool::games_done)
        .def_property_readonly("evaluations", &GamePool::evaluations)
        .def_property_readonly("batches", &GamePool::batches)
        .def_property_readonly("results", &GamePool::results)
        .def("drain_samples", [](GamePool& pool) {
            const std::vector<Sample> samples = pool.drain_samples();
            const auto rows = static_cast<py::ssize_t>(samples.size());
            py::array_t<std::uint64_t> current(rows);
            py::array_t<std::uint64_t> occupied(rows);
            py::array_t<float> policy({rows, py::ssize_t{kWidth}});
            py::array_t<float> outcome(rows);
            auto current_view = current.mutable_unchecked<1>();
            auto occupied_view = occupied.mutable_unchecked<1>();
            auto policy_view = policy.mutable_unchecked<2>();
            auto outcome_view = outcome.mutable_unchecked<1>();
            for (py::ssize_t row = 0; row < rows; ++row) {
                const Sample& sample = samples[row];
                current_view(row) = sample.current;
                occupied_view(row) = sample.occupied;
                outcome_view(row) = sample.outcome;
                for (int col = 0; col < kWidth; ++col) policy_view(row, col) = sample.policy[col];
            }
            return py::make_tuple(current, occupied, policy, outcome);
        });

    py::class_<PositionSearch>(m, "PositionSearch")
        .def(py::init<std::vector<Board>, SearchParams, int, int, int>(), py::arg("boards"),
             py::arg("search"), py::arg("simulations"), py::arg("leaves_per_tree") = 8,
             py::arg("threads") = 1)
        .def("collect",
             [](PositionSearch& search) {
                 const LeafBatch* batch = nullptr;
                 {
                     py::gil_scoped_release nogil;
                     batch = &search.collect();
                 }
                 return batch_arrays(*batch);
             })
        .def(
            "apply",
            [](PositionSearch& search, const FloatInput& policy, const FloatInput& values,
               int rows) {
                check_answers(policy, values, rows);
                const float* policy_data = policy.data();
                const float* value_data = values.data();
                py::gil_scoped_release nogil;
                search.apply(policy_data, value_data);
            },
            py::arg("policy"), py::arg("values"), py::arg("rows"))
        .def("visits", &PositionSearch::visits, py::arg("index"))
        .def("value", &PositionSearch::value, py::arg("index"));
}
