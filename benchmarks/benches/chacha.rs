use chacha::{ChaCha12, ChaCha20};
use chacha20::{KeyIvInit, cipher::StreamCipher};
use criterion::*;

fn chacha(c: &mut Criterion) {
    for n in [64, 128, 256, 1024, 4096, 16384, 65536] {
        let mut group = c.benchmark_group(format!("{}", n));
        let mut plaintext = vec![0u8; n];

        let key = [0u8; 32];
        let djb_nonce = [0u8; 8];
        let ietf_96b_nonce = [0u8; 12];

        let mut chacha12_rust_crypto = chacha20::ChaCha12::new(&key.into(), &ietf_96b_nonce.into());
        let mut chacha12 = ChaCha12::new(&key, &djb_nonce);

        let mut chacha20_rust_crypto = chacha20::ChaCha20::new(&key.into(), &ietf_96b_nonce.into());
        let mut chacha20 = ChaCha20::new(&key, &djb_nonce);

        group.throughput(Throughput::Bytes(plaintext.len() as u64));
        group.bench_function("ChaCha12", |b| {
            b.iter(|| {
                chacha12.xor_keystream(&mut plaintext);
            });
        });
        group.bench_function("ChaCha12-RustCrypto", |b| {
            b.iter(|| {
                chacha12_rust_crypto.apply_keystream(&mut plaintext);
            });
        });

        group.bench_function("ChaCha20", |b| {
            b.iter(|| {
                chacha20.xor_keystream(&mut plaintext);
            });
        });
        group.bench_function("ChaCha20-RustCrypto", |b| {
            b.iter(|| {
                chacha20_rust_crypto.apply_keystream(&mut plaintext);
            });
        });
    }
}

criterion_group!(benches, chacha);
criterion_main!(benches);
