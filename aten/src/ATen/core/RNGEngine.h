#pragma once

/**
 * Base class for CPU pseudo random number generator
 *
 * Implement Threefry to support splittable PRNG.
 */

namespace at {

/**
 * note [Threefry implementation]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Reference for Threefish:
 * https://www.schneier.com/wp-content/uploads/2015/01/skein.pdf
 *
 * To trade cryptographic strength for speed, Threefry is
 * introduced by removing tweaks in:
 *     J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw,
 *     "Parallel random numbers: As easy as 1, 2, 3," SC '11:
 *     Proceedings of 2011 International Conference for High
 *     Performance Computing, Networking, Storage and Analysis,
 *     2011, pp. 1-12, doi: 10.1145/2063384.2063405.
 *
 * Permutation is also ignored since inputs are 2 uint32_t.
 */
namespace detail {

// rotation constants
static const uint8_t rot_c[8] = {13, 15, 26, 6, 17, 29, 16, 24};

// key schedule constant
static const uint32_t ks_c = 0x1BD11BDA;

static inline uint32_t rotate_left(uint32_t x, const uint8_t rot) {
  return (x << (rot & 31)) | (x >> (32 - (rot & 31)));
}

/*
 * Mix function
 *
 * x0     x1
 *  |      |
 *  v      |
 * add<----|
 *  |      |
 *  |      v
 *  |    rotate
 *  |      |
 *  |      v
 *  |---->xor
 *  |      |
 *  v      v
 * x0'    x1'
 */
static inline void mix(uint32_t & x0, uint32_t & x1, const uint8_t rot) {
  x0 += x1;
  x1 = rotate_left(x1, rot);
  x1 ^= x0;
}

static inline std::pair<uint32_t, uint32_t> threefry(uint32_t seed0, uint32_t seed1, uint32_t x0, uint32_t x1) {
  uint32_t ks[3] = {seed0, seed1, ks_c ^ seed0 ^ seed1};
  uint32_t sub_ks[2];

  // 5 key schedules: 20 rounds in total
  for (uint8_t sc = 0; sc < 5; ++sc) {
    // Add subkey
    x0 += ks[sc % 3];
    x1 += ks[(sc + 1) % 3] + sc;
    
    uint8_t rot_idx = (sc % 2) * 4;
    for (uint8_t round = 0; round < 4; ++round) {
      mix(x0, x1, rot_c[rot_idx + round]);
    }
  }

  // Add final key to result
  x0 += ks[2];
  x1 += ks[0] + 5;
  
  return std::pair<uint32_t, uint32_t>(x0, x1);
}

static inline uint64_t threefry(uint64_t seed, uint32_t x0, uint32_t x1) {
  uint32_t seed0 = static_cast<uint32_t>(seed);
  uint32_t seed1 = static_cast<uint32_t>(seed >> 32);
  auto ns = threefry(seed0, seed1, x0, x1);
  uint64_t new_seed = (static_cast<uint64_t>(ns.second) << 32) | ns.first;
  return new_seed;
}

} // namespace detail

class rng_engine {
  public:
    // virtual uint64_t seed() const = 0;
    // virtual bool is_valid() const = 0;
    virtual uint32_t operator()() = 0;

};

} // namespace at
