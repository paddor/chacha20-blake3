#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::{BLOCK_SIZE, STATE_WORDS, extract_counter_from_state, inject_counter_into_state};

// https://doc.rust-lang.org/stable/core/arch/x86_64/

/// how many ChaCha blocks we compute in parallel (depends on the size of the SIMD vectors, here 256 / 32 = 8)
pub const SIMD_LANES: usize = 8;

/// A 8-lane array with guaranteed 32-byte alignment.
/// Used for _mm256_load_si256 / _mm256_store_si256 operations that should be faster than
/// unaligned operations (_mm256_loadu_si256 / _mm256_storeu_si256)
#[repr(align(32))]
struct AlignedU32x8([u32; SIMD_LANES]);

// AVX2 supports operations on 256-bit registers (vectors).
// Each vector can be seen as 8 lanes, where each lane is 32-bit wide (8 * 32 = 256), allowing us to compute
// 8 ChaCha blocks in parallel.
// Thus, in a single 256-bit vector we will get the follwing state:
// [ block1 (32-bits) || block2 (32-bits) || block3 (32-bits) || block4 (32-bits) || block5 (32-bits) ... ]
// then we perform the normal ChaCha operations on these vectors, meaning that we compute
// 8 ChaCha blocks in parallel for every operation on these vectors.
//
// # Safety
//
// Caller must ensure that AVX2 is available (e.g. via `is_x86_feature_detected!("avx2")`).
// The annotation enables the compiler to auto-vectorize the XOR and keystream-extraction
// loops with AVX2 instructions, not just the explicit intrinsic calls.
#[target_feature(enable = "avx2")]
pub unsafe fn chacha_avx2<const ROUNDS: usize>(
    state: &mut [u32; STATE_WORDS],
    input: &mut [u8],
    last_keystream_block: &mut [u8; BLOCK_SIZE],
) {
    // SAFETY: AVX2 availability is guaranteed by #[target_feature] + caller contract.
    unsafe { chacha_avx2_inner::<ROUNDS>(state, input, last_keystream_block) }
}

/// Inner function keeps the original code structure with a single `unsafe` scope.
#[target_feature(enable = "avx2")]
unsafe fn chacha_avx2_inner<const ROUNDS: usize>(
    state: &mut [u32; STATE_WORDS],
    input: &mut [u8],
    last_keystream_block: &mut [u8; BLOCK_SIZE],
) {
    let mut counter = extract_counter_from_state(state);
    let mut keystream = [0u8; SIMD_LANES * BLOCK_SIZE];

    // vpshufb masks for byte-aligned rotations (1 instruction instead of shift+shift+or)
    let rot16 = _mm256_setr_epi8(
        2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13,
    );
    let rot8 = _mm256_setr_epi8(
        3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 3, 0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14,
    );

    let mut initial_state: [__m256i; STATE_WORDS] = [
        // constant
        _mm256_set1_epi32(state[0] as i32),
        _mm256_set1_epi32(state[1] as i32),
        _mm256_set1_epi32(state[2] as i32),
        _mm256_set1_epi32(state[3] as i32),
        // key
        _mm256_set1_epi32(state[4] as i32),
        _mm256_set1_epi32(state[5] as i32),
        _mm256_set1_epi32(state[6] as i32),
        _mm256_set1_epi32(state[7] as i32),
        _mm256_set1_epi32(state[8] as i32),
        _mm256_set1_epi32(state[9] as i32),
        _mm256_set1_epi32(state[10] as i32),
        _mm256_set1_epi32(state[11] as i32),
        // counter, set it to 0 for now, it is injected later during each iteration of the loop
        _mm256_set1_epi32(0),
        _mm256_set1_epi32(0),
        // nonce
        _mm256_set1_epi32(state[14] as i32),
        _mm256_set1_epi32(state[15] as i32),
    ];

    // process input by chunks of 8 * 64 bytes
    for input_blocks in input.chunks_mut(BLOCK_SIZE * SIMD_LANES) {
        // inject counter (uint64 little-endian) as two 32-bit little-endian words for each lane
        // e.g for one 256-bit vector with 8 32-bit lanes: [counter, counter + 1, counter + 2, counter + 3...]
        let mut counter_lane_low = AlignedU32x8([0u32; SIMD_LANES]);
        let mut counter_lane_high = AlignedU32x8([0u32; SIMD_LANES]);
        for i in 0..SIMD_LANES {
            let counter_lane = counter.wrapping_add(i as u64);
            counter_lane_low.0[i] = counter_lane as u32;
            counter_lane_high.0[i] = (counter_lane >> 32) as u32;
        }

        unsafe {
            initial_state[12] = _mm256_load_si256(counter_lane_low.0.as_ptr() as *const __m256i);
            initial_state[13] = _mm256_load_si256(counter_lane_high.0.as_ptr() as *const __m256i);
        }

        // compute 8 64-byte ChaCha blocks in parallel
        unsafe {
            chacha_avx2_8blocks::<ROUNDS>(initial_state, &mut keystream, rot16, rot8);
        }

        // XOR plaintext with keystream
        input_blocks
            .iter_mut()
            .zip(keystream)
            .for_each(|(plaintext, keystream)| *plaintext ^= keystream);

        counter = counter.wrapping_add((input_blocks.len() as u64).div_ceil(BLOCK_SIZE as u64));
    }

    inject_counter_into_state(state, counter);

    if input.len() % BLOCK_SIZE != 0 {
        let last_keystream_block_index = ((input.len() - 1) / BLOCK_SIZE) % SIMD_LANES;
        let last_keystream_block_offset = last_keystream_block_index * BLOCK_SIZE;
        last_keystream_block
            .copy_from_slice(&keystream[last_keystream_block_offset..last_keystream_block_offset + BLOCK_SIZE]);
    }
}

/// Compute 8 64-byte ChaCha blocks in parallel using AVX2 vectors.
/// The keystream is the 8 64-byte blocks computed in parallel.
/// [ block1 (64 bytes) || block2 (64 bytes) || block3 (64 bytes) || block4 (64 bytes) ... ]
#[target_feature(enable = "avx2")]
unsafe fn chacha_avx2_8blocks<const ROUNDS: usize>(
    initial_state: [__m256i; STATE_WORDS],
    keystream: &mut [u8; SIMD_LANES * 64],
    rot16: __m256i,
    rot8: __m256i,
) {
    let keystream_ptr = keystream.as_mut_ptr();

    unsafe {
        let mut ws = initial_state;

        macro_rules! quarter_round {
            ($a:expr, $b:expr, $c:expr, $d:expr) => {
                // a += b; d ^= a; d <<<= 16 (vpshufb)
                $a = _mm256_add_epi32($a, $b);
                $d = _mm256_xor_si256($d, $a);
                $d = _mm256_shuffle_epi8($d, rot16);

                // c += d; b ^= c; b <<<= 12
                $c = _mm256_add_epi32($c, $d);
                $b = _mm256_xor_si256($b, $c);
                $b = _mm256_or_si256(_mm256_slli_epi32($b, 12), _mm256_srli_epi32($b, 20));

                // a += b; d ^= a; d <<<= 8 (vpshufb)
                $a = _mm256_add_epi32($a, $b);
                $d = _mm256_xor_si256($d, $a);
                $d = _mm256_shuffle_epi8($d, rot8);

                // c += d; b ^= c; b <<<= 7
                $c = _mm256_add_epi32($c, $d);
                $b = _mm256_xor_si256($b, $c);
                $b = _mm256_or_si256(_mm256_slli_epi32($b, 7), _mm256_srli_epi32($b, 25));
            };
        }

        for _ in 0..ROUNDS / 2 {
            // column rounds
            quarter_round!(ws[0], ws[4], ws[8], ws[12]);
            quarter_round!(ws[1], ws[5], ws[9], ws[13]);
            quarter_round!(ws[2], ws[6], ws[10], ws[14]);
            quarter_round!(ws[3], ws[7], ws[11], ws[15]);

            // diagonal rounds
            quarter_round!(ws[0], ws[5], ws[10], ws[15]);
            quarter_round!(ws[1], ws[6], ws[11], ws[12]);
            quarter_round!(ws[2], ws[7], ws[8], ws[13]);
            quarter_round!(ws[3], ws[4], ws[9], ws[14]);
        }

        // add initial state to working state
        for i in 0..STATE_WORDS {
            ws[i] = _mm256_add_epi32(ws[i], initial_state[i]);
        }

        // 8x8 transpose of ws[0..8] to extract words 0-7 of each block,
        // then store as the first 32 bytes of each 64-byte block in the keystream.
        //
        // Before transpose: ws[i] = [b0_wi, b1_wi, b2_wi, b3_wi, b4_wi, b5_wi, b6_wi, b7_wi]
        // After transpose:  row j  = [bj_w0, bj_w1, bj_w2, bj_w3, bj_w4, bj_w5, bj_w6, bj_w7]
        {
            let t0 = _mm256_unpacklo_epi32(ws[0], ws[1]);
            let t1 = _mm256_unpackhi_epi32(ws[0], ws[1]);
            let t2 = _mm256_unpacklo_epi32(ws[2], ws[3]);
            let t3 = _mm256_unpackhi_epi32(ws[2], ws[3]);
            let t4 = _mm256_unpacklo_epi32(ws[4], ws[5]);
            let t5 = _mm256_unpackhi_epi32(ws[4], ws[5]);
            let t6 = _mm256_unpacklo_epi32(ws[6], ws[7]);
            let t7 = _mm256_unpackhi_epi32(ws[6], ws[7]);

            let u0 = _mm256_unpacklo_epi64(t0, t2);
            let u1 = _mm256_unpackhi_epi64(t0, t2);
            let u2 = _mm256_unpacklo_epi64(t1, t3);
            let u3 = _mm256_unpackhi_epi64(t1, t3);
            let u4 = _mm256_unpacklo_epi64(t4, t6);
            let u5 = _mm256_unpackhi_epi64(t4, t6);
            let u6 = _mm256_unpacklo_epi64(t5, t7);
            let u7 = _mm256_unpackhi_epi64(t5, t7);

            _mm256_storeu_si256(
                keystream_ptr.add(0 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u0, u4, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(1 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u1, u5, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(2 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u2, u6, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(3 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u3, u7, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(4 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u0, u4, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(5 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u1, u5, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(6 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u2, u6, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(7 * 64) as *mut __m256i,
                _mm256_permute2x128_si256(u3, u7, 0x31),
            );
        }

        // 8x8 transpose of ws[8..16] to extract words 8-15 of each block,
        // then store as the second 32 bytes of each 64-byte block in the keystream.
        {
            let t0 = _mm256_unpacklo_epi32(ws[8], ws[9]);
            let t1 = _mm256_unpackhi_epi32(ws[8], ws[9]);
            let t2 = _mm256_unpacklo_epi32(ws[10], ws[11]);
            let t3 = _mm256_unpackhi_epi32(ws[10], ws[11]);
            let t4 = _mm256_unpacklo_epi32(ws[12], ws[13]);
            let t5 = _mm256_unpackhi_epi32(ws[12], ws[13]);
            let t6 = _mm256_unpacklo_epi32(ws[14], ws[15]);
            let t7 = _mm256_unpackhi_epi32(ws[14], ws[15]);

            let u0 = _mm256_unpacklo_epi64(t0, t2);
            let u1 = _mm256_unpackhi_epi64(t0, t2);
            let u2 = _mm256_unpacklo_epi64(t1, t3);
            let u3 = _mm256_unpackhi_epi64(t1, t3);
            let u4 = _mm256_unpacklo_epi64(t4, t6);
            let u5 = _mm256_unpackhi_epi64(t4, t6);
            let u6 = _mm256_unpacklo_epi64(t5, t7);
            let u7 = _mm256_unpackhi_epi64(t5, t7);

            _mm256_storeu_si256(
                keystream_ptr.add(0 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u0, u4, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(1 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u1, u5, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(2 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u2, u6, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(3 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u3, u7, 0x20),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(4 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u0, u4, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(5 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u1, u5, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(6 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u2, u6, 0x31),
            );
            _mm256_storeu_si256(
                keystream_ptr.add(7 * 64 + 32) as *mut __m256i,
                _mm256_permute2x128_si256(u3, u7, 0x31),
            );
        }
    }
}
