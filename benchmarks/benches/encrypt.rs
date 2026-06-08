use aes_gcm::Aes256Gcm;
use chacha::ChaCha12;
use chacha20::cipher::InOutBuf;
use chacha20_blake3::{ChaCha12Blake3, ChaCha20Blake3};
use chacha20_blake3_upstream::ChaCha20Blake3 as ChaCha20Blake3Upstream;
use chacha20poly1305::{KeyInit, XChaCha20Poly1305, aead::AeadInOut};
use criterion::*;

fn bench(c: &mut Criterion) {
    for n in [64, 256, 1024, 4096, 16384, 65536] {
        let mut group = c.benchmark_group(format!("{}", n));
        let mut plaintext = vec![0u8; n];

        let key = [0u8; 32];
        let nonce_8 = [0u8; 8];
        let nonce_12 = [0u8; 12];
        let nonce_24 = [0u8; 24];
        let aad = [0u8; 128];

        let chacha12_blake3_cipher = ChaCha12Blake3::new(key);
        let chacha20_blake3_cipher = ChaCha20Blake3::new(key);
        let chacha20_blake3_upstream_cipher = ChaCha20Blake3Upstream::new(key);
        let xchacha20poly1305_cipher = XChaCha20Poly1305::new(&key.try_into().unwrap());
        let aes_256_gcm_cipher = Aes256Gcm::new(&key.try_into().unwrap());

        let auth_key = [0u8; 32];

        group.throughput(Throughput::Bytes(plaintext.len() as u64));

        group.bench_function("AES-256-GCM", |b| {
            b.iter(|| {
                let _ = aes_256_gcm_cipher.encrypt_inout_detached(
                    (&nonce_12).into(),
                    &aad,
                    InOutBuf::from(plaintext.as_mut_slice()),
                );
            });
        });

        group.bench_function("ChaCha12-BLAKE3 (no KDF)", |b| {
            b.iter(|| {
                ChaCha12::new(&key, &nonce_8).xor_keystream(&mut plaintext);

                let mut mac_hasher = blake3::Hasher::new_keyed(&auth_key);
                mac_hasher.update(&aad);
                mac_hasher.update(&(aad.len() as u64).to_le_bytes());
                mac_hasher.update(&plaintext);
                mac_hasher.update(&(plaintext.len() as u64).to_le_bytes());
                let _tag: [u8; 32] = mac_hasher.finalize().into();
            });
        });

        group.bench_function("ChaCha20-BLAKE3 (no KDF)", |b| {
            b.iter(|| {
                chacha::ChaCha20::new(&key, &nonce_8).xor_keystream(&mut plaintext);

                let mut mac_hasher = blake3::Hasher::new_keyed(&auth_key);
                mac_hasher.update(&aad);
                mac_hasher.update(&(aad.len() as u64).to_le_bytes());
                mac_hasher.update(&plaintext);
                mac_hasher.update(&(plaintext.len() as u64).to_le_bytes());
                let _tag: [u8; 32] = mac_hasher.finalize().into();
            });
        });

        group.bench_function("ChaCha12-BLAKE3", |b| {
            b.iter(|| {
                let _ = chacha12_blake3_cipher.encrypt_in_place_detached(&nonce_24, &mut plaintext, &aad);
            });
        });

        group.bench_function("ChaCha20-BLAKE3", |b| {
            b.iter(|| {
                let _ = chacha20_blake3_cipher.encrypt_in_place_detached(&nonce_24, &mut plaintext, &aad);
            });
        });

        group.bench_function("ChaCha20-BLAKE3 (upstream)", |b| {
            b.iter(|| {
                let _ = chacha20_blake3_upstream_cipher.encrypt_in_place_detached(&nonce_24, &mut plaintext, &aad);
            });
        });

        group.bench_function("XChaCha20-Poly1305", |b| {
            b.iter(|| {
                let _ = xchacha20poly1305_cipher.encrypt_inout_detached(
                    (&nonce_24).into(),
                    &aad,
                    InOutBuf::from(plaintext.as_mut_slice()),
                );
            });
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
