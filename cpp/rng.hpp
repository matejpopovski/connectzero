// xoshiro256++ seeded through splitmix64. Self-play draws millions of samples, and a
// fixed generator keeps a seed's games identical across standard libraries for
// everything except the gamma draws behind Dirichlet noise.
#pragma once

#include <bit>
#include <cstdint>
#include <limits>

namespace cz {

class Rng {
   public:
    using result_type = std::uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    explicit Rng(std::uint64_t seed = 0) {
        for (auto& word : state_) {
            seed += 0x9E3779B97F4A7C15ULL;
            std::uint64_t z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            word = z ^ (z >> 31);
        }
    }

    result_type operator()() {
        const std::uint64_t out = std::rotl(state_[0] + state_[3], 23) + state_[0];
        const std::uint64_t tmp = state_[1] << 17;
        state_[2] ^= state_[0];
        state_[3] ^= state_[1];
        state_[1] ^= state_[2];
        state_[0] ^= state_[3];
        state_[2] ^= tmp;
        state_[3] = std::rotl(state_[3], 45);
        return out;
    }

    // Lemire's multiply-shift. The bias is below 2^-58 for the small ranges used here.
    int below(int bound) {
        return static_cast<int>(((*this)() >> 32) * static_cast<std::uint64_t>(bound) >> 32);
    }

    double uniform() { return static_cast<double>((*this)() >> 11) * 0x1.0p-53; }

   private:
    std::uint64_t state_[4]{};
};

}  // namespace cz
