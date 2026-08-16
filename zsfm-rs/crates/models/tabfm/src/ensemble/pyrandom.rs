//! A bit-compatible port of CPython's `random.Random` (Mersenne Twister, MT19937), so ensemble
//! member generation can be validated numerically against the real `TabFMClassifier`/
//! `TabFMRegressor` sklearn wrapper (which seeds `random.Random(random_state)` and calls
//! `.sample()`/`.shuffle()` in a specific order — see `config_gen.rs`).
//!
//! Ported from CPython's `Modules/_randommodule.c` (MT19937 core, `init_by_array` seeding) and
//! `Lib/random.py` (`_randbelow`, `sample`, `shuffle`). Reference sequences used in the unit
//! tests below were generated with the project's own `.venv/bin/python3`.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

pub struct Mt19937 {
    mt: [u32; N],
    index: usize,
}

impl Mt19937 {
    fn init_genrand(seed: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = seed;
        for i in 1..N {
            mt[i] = (1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, index: N }
    }

    /// CPython seeds every integer via `init_by_array`, never bare `init_genrand` — the seed
    /// integer is first split into little-endian 32-bit words (`key`).
    pub fn from_seed_key(key: &[u32]) -> Self {
        let mut rng = Self::init_genrand(19_650_218);
        let key = if key.is_empty() { vec![0u32] } else { key.to_vec() };
        let key_length = key.len();
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..N.max(key_length) {
            let prev = rng.mt[i - 1];
            rng.mt[i] = (rng.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = rng.mt[i - 1];
            rng.mt[i] = (rng.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                rng.mt[0] = rng.mt[N - 1];
                i = 1;
            }
        }
        rng.mt[0] = 0x8000_0000;
        rng
    }

    /// Seed from a small non-negative integer (as CPython does for `random.Random(42)` etc).
    pub fn from_u64_seed(seed: u64) -> Self {
        if seed == 0 {
            return Self::from_seed_key(&[]);
        }
        let mut key = Vec::new();
        let mut s = seed;
        while s > 0 {
            key.push((s & 0xffff_ffff) as u32);
            s >>= 32;
        }
        Self::from_seed_key(&key)
    }

    fn regenerate(&mut self) {
        let mag01 = [0u32, MATRIX_A];
        for kk in 0..N - M {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        for kk in N - M..N - 1 {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        }
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.regenerate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// CPython's `getrandbits(k)`: words filled least-significant-first, 32 bits at a time.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k <= 32 {
            return (self.next_u32() >> (32 - k)) as u64;
        }
        let words = (k - 1) / 32 + 1;
        let mut result: u64 = 0;
        for w in 0..words {
            let mut r = self.next_u32();
            if w == words - 1 {
                r >>= 32 * words - k;
            }
            result |= (r as u64) << (32 * w);
        }
        result
    }
}

/// A CPython-compatible `random.Random` instance: MT19937 state plus the pure-Python wrapper
/// logic (`_randbelow`, `sample`, `shuffle`) from `Lib/random.py`.
pub struct PyRandom {
    mt: Mt19937,
}

impl PyRandom {
    pub fn new(seed: u64) -> Self {
        PyRandom { mt: Mt19937::from_u64_seed(seed) }
    }

    fn bit_length(n: u64) -> u32 {
        if n == 0 { 0 } else { 64 - n.leading_zeros() }
    }

    /// `Random._randbelow_with_getrandbits`: rejection-sampled `getrandbits`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = Self::bit_length(n);
        loop {
            let r = self.mt.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `Random.sample(range(n), k)` restricted to integer populations `0..n` (the only case
    /// needed here — feature/class/row indices), returning the sampled indices in draw order.
    pub fn sample_indices(&mut self, n: usize, k: usize) -> Vec<usize> {
        assert!(k <= n, "sample larger than population");
        let mut setsize: f64 = 21.0;
        if k > 5 {
            setsize += 4f64.powf((3.0 * k as f64).log(4.0).ceil());
        }
        let mut result = vec![0usize; k];
        if (n as f64) <= setsize {
            let mut pool: Vec<usize> = (0..n).collect();
            for i in 0..k {
                let j = self.randbelow((n - i) as u64) as usize;
                result[i] = pool[j];
                pool[j] = pool[n - i - 1];
            }
        } else {
            let mut selected = std::collections::HashSet::new();
            for i in 0..k {
                let mut j = self.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result[i] = j;
            }
        }
        result
    }

    /// `Random.sample(population, k)` for an arbitrary slice, via `sample_indices`.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Vec<T> {
        self.sample_indices(population.len(), k)
            .into_iter()
            .map(|i| population[i].clone())
            .collect()
    }

    /// `Random.shuffle(x)`: in-place Fisher-Yates via `_randbelow`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let len = x.len();
        if len < 2 {
            return;
        }
        for i in (1..len).rev() {
            let j = self.randbelow((i + 1) as u64) as usize;
            x.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_10_of_10() {
        // python3 -c "import random; print(random.Random(42).sample(range(10), 10))"
        let mut r = PyRandom::new(42);
        assert_eq!(r.sample_indices(10, 10), vec![1, 0, 4, 9, 6, 5, 8, 2, 3, 7]);
    }

    #[test]
    fn test_sample_3_of_5() {
        // python3 -c "import random; print(random.Random(42).sample(range(5), 3))"
        let mut r = PyRandom::new(42);
        assert_eq!(r.sample_indices(5, 3), vec![0, 4, 2]);
    }

    #[test]
    fn test_shuffle_8() {
        // python3 -c "import random; l=list(range(8)); random.Random(42).shuffle(l); print(l)"
        let mut r = PyRandom::new(42);
        let mut v: Vec<usize> = (0..8).collect();
        r.shuffle(&mut v);
        assert_eq!(v, vec![3, 4, 6, 7, 2, 5, 0, 1]);
    }

    #[test]
    fn test_sample_5_of_100_rejection_branch() {
        // python3 -c "import random; print(random.Random(7).sample(range(100), 5))"
        // k=5 (not >5) so setsize=21; n=100 > 21, exercises the rejection-set branch.
        let mut r = PyRandom::new(7);
        assert_eq!(r.sample_indices(100, 5), vec![41, 19, 50, 83, 6]);
    }
}
