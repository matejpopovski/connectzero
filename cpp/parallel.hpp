// A shared-counter parallel loop. Threads are spawned per call: each call sits between
// two network evaluations, which dwarf the spawn cost, and it means no idle thread pool
// has to be managed across the Python boundary.
#pragma once

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace cz {

// Calls body(item, worker) for every item in [0, count).
template <typename Body>
void parallel_for(int count, int threads, Body&& body) {
    const int workers = std::min(threads, count);
    if (workers <= 1) {
        for (int item = 0; item < count; ++item) body(item, 0);
        return;
    }
    std::atomic<int> next{0};
    std::exception_ptr failure;
    std::mutex failure_lock;
    auto run = [&](int worker) {
        try {
            for (int item = next.fetch_add(1); item < count; item = next.fetch_add(1)) {
                body(item, worker);
            }
        } catch (...) {
            std::lock_guard guard(failure_lock);
            if (!failure) failure = std::current_exception();
            next.store(count);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    for (int worker = 1; worker < workers; ++worker) pool.emplace_back(run, worker);
    run(0);
    for (auto& thread : pool) thread.join();
    if (failure) std::rethrow_exception(failure);
}

}  // namespace cz
