# Changelog

All notable changes to the `chacha20-blake3` crate are documented here.

## [Unreleased]

## [0.10.0] - 2025-06-09

### Added

- `Session<ROUNDS>` for stateful AEAD without per-message KDF. Performs KDF
  once at creation and tracks a continuous block counter across messages.
  Type aliases: `Session8`, `Session12`, `Session20`.
- `ChaCha8Blake3` and `ChaCha12Blake3` type aliases (were already available
  as `ChaChaBlake3<8>` and `ChaChaBlake3<12>`).

### Fixed

- Zeroize all derived key material (`kdf_out`, `encryption_key`,
  `authentication_key`, `encryption_nonce`) in both `encrypt_in_place_detached`
  and `decrypt_in_place_detached`. Previously only `kdf_out` was zeroized in the
  encrypt path, and nothing was zeroized in the decrypt path.
- `decrypt_in_place_detached` now zeroizes derived keys on MAC failure instead
  of returning early with secrets on the stack.

### Changed

- AVX2 8-block path: `vpshufb` rotations and 8x8 SIMD transpose for keystream
  extraction, replacing shift-or rotations and per-word extraction.
- New AVX2 4-block row-layout path for 256-511 byte messages.
- New SSE2 2-block path with SIMD XOR for sub-256-byte messages on x86_64.
- `no_std` AVX-512 dispatch threshold raised from 128 to 256 bytes to match the
  `std` runtime-detected threshold.
- `Session::advance_counter` uses `wrapping_add` instead of `+=` for
  consistency with all other counter arithmetic.

## [0.9.13] - 2025-05-23

### Changed

- Pin `blake3` to 1.8.0, upgrade `constant_time_eq` to 0.5.0.
- Pin `zeroize` to 1.8.0.

## [0.9.12] - 2025-05-18

### Fixed

- Fix compilation on aarch64 (NEON import path).

## [0.9.11] - 2025-05-18

### Changed

- Inline `chacha` crate into `chacha20-blake3` to publish as a single crate.
- Add `#[target_feature]` annotations to AVX2/AVX-512 functions so the
  compiler can auto-vectorize without requiring global `-C target-feature` flags.
