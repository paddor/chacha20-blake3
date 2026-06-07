#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{BLOCK_SIZE, STATE_WORDS, extract_counter_from_state, inject_counter_into_state};

const TWO_BLOCKS: usize = BLOCK_SIZE * 2;

#[target_feature(enable = "sse2")]
pub unsafe fn chacha_sse2<const ROUNDS: usize>(
    state: &mut [u32; STATE_WORDS],
    input: &mut [u8],
    last_keystream_block: &mut [u8; BLOCK_SIZE],
) {
    unsafe { chacha_sse2_inner::<ROUNDS>(state, input, last_keystream_block) }
}

#[target_feature(enable = "sse2")]
unsafe fn chacha_sse2_inner<const ROUNDS: usize>(
    state: &mut [u32; STATE_WORDS],
    input: &mut [u8],
    last_keystream_block: &mut [u8; BLOCK_SIZE],
) {
    let mut counter = extract_counter_from_state(state);

    // Process pairs of blocks for ILP
    let mut remaining = input.len();
    let mut offset = 0;

    unsafe {
        while remaining >= TWO_BLOCKS {
            let ptr = input.as_mut_ptr().add(offset);
            chacha_sse2_2blocks::<ROUNDS>(state, ptr, counter);
            counter = counter.wrapping_add(2);
            offset += TWO_BLOCKS;
            remaining -= TWO_BLOCKS;
        }
    }

    // Process remaining blocks one at a time
    if remaining > 0 {
        let tail = &mut input[offset..];
        for input_block in tail.chunks_mut(BLOCK_SIZE) {
            inject_counter_into_state(state, counter);

            unsafe {
                let mut a = _mm_loadu_si128(state[0..4].as_ptr() as *const __m128i);
                let mut b = _mm_loadu_si128(state[4..8].as_ptr() as *const __m128i);
                let mut c = _mm_loadu_si128(state[8..12].as_ptr() as *const __m128i);
                let mut d = _mm_loadu_si128(state[12..16].as_ptr() as *const __m128i);

                let initial_a = a;
                let initial_b = b;
                let initial_c = c;
                let initial_d = d;

                for _ in 0..ROUNDS / 2 {
                    quarter_round_sse2(&mut a, &mut b, &mut c, &mut d);
                    b = _mm_shuffle_epi32(b, 0b00_11_10_01);
                    c = _mm_shuffle_epi32(c, 0b01_00_11_10);
                    d = _mm_shuffle_epi32(d, 0b10_01_00_11);
                    quarter_round_sse2(&mut a, &mut b, &mut c, &mut d);
                    b = _mm_shuffle_epi32(b, 0b10_01_00_11);
                    c = _mm_shuffle_epi32(c, 0b01_00_11_10);
                    d = _mm_shuffle_epi32(d, 0b00_11_10_01);
                }

                a = _mm_add_epi32(a, initial_a);
                b = _mm_add_epi32(b, initial_b);
                c = _mm_add_epi32(c, initial_c);
                d = _mm_add_epi32(d, initial_d);

                if input_block.len() == BLOCK_SIZE {
                    let p = input_block.as_mut_ptr();
                    _mm_storeu_si128(
                        p.add(0) as *mut __m128i,
                        _mm_xor_si128(_mm_loadu_si128(p.add(0) as *const __m128i), a),
                    );
                    _mm_storeu_si128(
                        p.add(16) as *mut __m128i,
                        _mm_xor_si128(_mm_loadu_si128(p.add(16) as *const __m128i), b),
                    );
                    _mm_storeu_si128(
                        p.add(32) as *mut __m128i,
                        _mm_xor_si128(_mm_loadu_si128(p.add(32) as *const __m128i), c),
                    );
                    _mm_storeu_si128(
                        p.add(48) as *mut __m128i,
                        _mm_xor_si128(_mm_loadu_si128(p.add(48) as *const __m128i), d),
                    );
                } else {
                    let mut keystream = [0u8; BLOCK_SIZE];
                    _mm_storeu_si128(keystream.as_mut_ptr().add(0) as *mut __m128i, a);
                    _mm_storeu_si128(keystream.as_mut_ptr().add(16) as *mut __m128i, b);
                    _mm_storeu_si128(keystream.as_mut_ptr().add(32) as *mut __m128i, c);
                    _mm_storeu_si128(keystream.as_mut_ptr().add(48) as *mut __m128i, d);

                    input_block.iter_mut().zip(keystream).for_each(|(p, k)| *p ^= k);

                    last_keystream_block.copy_from_slice(&keystream);
                }
            }

            counter = counter.wrapping_add(1);
        }
    }

    inject_counter_into_state(state, counter);
}

/// Process two blocks simultaneously for instruction-level parallelism.
#[target_feature(enable = "sse2")]
unsafe fn chacha_sse2_2blocks<const ROUNDS: usize>(state: &mut [u32; STATE_WORDS], ptr: *mut u8, counter: u64) {
    unsafe {
        // Block 0
        inject_counter_into_state(state, counter);
        let mut a0 = _mm_loadu_si128(state[0..4].as_ptr() as *const __m128i);
        let mut b0 = _mm_loadu_si128(state[4..8].as_ptr() as *const __m128i);
        let mut c0 = _mm_loadu_si128(state[8..12].as_ptr() as *const __m128i);
        let mut d0 = _mm_loadu_si128(state[12..16].as_ptr() as *const __m128i);

        // Block 1
        inject_counter_into_state(state, counter.wrapping_add(1));
        let mut a1 = _mm_loadu_si128(state[0..4].as_ptr() as *const __m128i);
        let mut b1 = _mm_loadu_si128(state[4..8].as_ptr() as *const __m128i);
        let mut c1 = _mm_loadu_si128(state[8..12].as_ptr() as *const __m128i);
        let mut d1 = _mm_loadu_si128(state[12..16].as_ptr() as *const __m128i);

        // Save initial state (a/b/c are same for both blocks, only d differs due to counter)
        let ia = a0;
        let ib = b0;
        let ic = c0;
        let id0 = d0;
        let id1 = d1;

        for _ in 0..ROUNDS / 2 {
            // Column rounds - interleaved for ILP
            quarter_round_sse2(&mut a0, &mut b0, &mut c0, &mut d0);
            quarter_round_sse2(&mut a1, &mut b1, &mut c1, &mut d1);

            // Diagonal shuffle
            b0 = _mm_shuffle_epi32(b0, 0b00_11_10_01);
            c0 = _mm_shuffle_epi32(c0, 0b01_00_11_10);
            d0 = _mm_shuffle_epi32(d0, 0b10_01_00_11);
            b1 = _mm_shuffle_epi32(b1, 0b00_11_10_01);
            c1 = _mm_shuffle_epi32(c1, 0b01_00_11_10);
            d1 = _mm_shuffle_epi32(d1, 0b10_01_00_11);

            // Diagonal rounds
            quarter_round_sse2(&mut a0, &mut b0, &mut c0, &mut d0);
            quarter_round_sse2(&mut a1, &mut b1, &mut c1, &mut d1);

            // Unshuffle
            b0 = _mm_shuffle_epi32(b0, 0b10_01_00_11);
            c0 = _mm_shuffle_epi32(c0, 0b01_00_11_10);
            d0 = _mm_shuffle_epi32(d0, 0b00_11_10_01);
            b1 = _mm_shuffle_epi32(b1, 0b10_01_00_11);
            c1 = _mm_shuffle_epi32(c1, 0b01_00_11_10);
            d1 = _mm_shuffle_epi32(d1, 0b00_11_10_01);
        }

        // Add initial state
        a0 = _mm_add_epi32(a0, ia);
        b0 = _mm_add_epi32(b0, ib);
        c0 = _mm_add_epi32(c0, ic);
        d0 = _mm_add_epi32(d0, id0);

        a1 = _mm_add_epi32(a1, ia);
        b1 = _mm_add_epi32(b1, ib);
        c1 = _mm_add_epi32(c1, ic);
        d1 = _mm_add_epi32(d1, id1);

        // XOR with plaintext - block 0
        let p0 = ptr;
        _mm_storeu_si128(
            p0.add(0) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p0.add(0) as *const __m128i), a0),
        );
        _mm_storeu_si128(
            p0.add(16) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p0.add(16) as *const __m128i), b0),
        );
        _mm_storeu_si128(
            p0.add(32) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p0.add(32) as *const __m128i), c0),
        );
        _mm_storeu_si128(
            p0.add(48) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p0.add(48) as *const __m128i), d0),
        );

        // XOR with plaintext - block 1
        let p1 = ptr.add(64);
        _mm_storeu_si128(
            p1.add(0) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p1.add(0) as *const __m128i), a1),
        );
        _mm_storeu_si128(
            p1.add(16) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p1.add(16) as *const __m128i), b1),
        );
        _mm_storeu_si128(
            p1.add(32) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p1.add(32) as *const __m128i), c1),
        );
        _mm_storeu_si128(
            p1.add(48) as *mut __m128i,
            _mm_xor_si128(_mm_loadu_si128(p1.add(48) as *const __m128i), d1),
        );
    }
}

#[inline(always)]
unsafe fn quarter_round_sse2(a: &mut __m128i, b: &mut __m128i, c: &mut __m128i, d: &mut __m128i) {
    unsafe {
        *a = _mm_add_epi32(*a, *b);
        *d = _mm_xor_si128(*d, *a);
        *d = _mm_or_si128(_mm_slli_epi32(*d, 16), _mm_srli_epi32(*d, 16));

        *c = _mm_add_epi32(*c, *d);
        *b = _mm_xor_si128(*b, *c);
        *b = _mm_or_si128(_mm_slli_epi32(*b, 12), _mm_srli_epi32(*b, 20));

        *a = _mm_add_epi32(*a, *b);
        *d = _mm_xor_si128(*d, *a);
        *d = _mm_or_si128(_mm_slli_epi32(*d, 8), _mm_srli_epi32(*d, 24));

        *c = _mm_add_epi32(*c, *d);
        *b = _mm_xor_si128(*b, *c);
        *b = _mm_or_si128(_mm_slli_epi32(*b, 7), _mm_srli_epi32(*b, 25));
    }
}
