use chacha20_blake3::{ChaCha20Blake3, Session20, TAG_SIZE};

struct Test {
    plaintext: Vec<u8>,
    key: [u8; 32],
    nonce: [u8; 24],
    aad: Vec<u8>,
    expected_ciphertext: Vec<u8>,
}

#[test]
fn chacha20_blake3_test_vectors() {
    let tests = [
        Test {
            plaintext: [].to_vec(),
            key: hex::decode("0000000000000000000000000000000000000000000000000000000000000000")
                .unwrap()
                .try_into()
                .unwrap(),
            nonce: hex::decode("000000000000000000000000000000000000000000000000")
                .unwrap()
                .try_into()
                .unwrap(),
            aad: [].to_vec(),
            expected_ciphertext: hex::decode("4fbdd67d41f66924b4304f0fc1eaa87a8e90fc7c5304fe3078f0a1b6e6142c33")
                .unwrap(),
        },
        Test {
            plaintext: b"ChaCha20".to_vec(),
            key: hex::decode("0100000000000000000000000000000000000000000000000000000000000010")
                .unwrap()
                .try_into()
                .unwrap(),
            nonce: hex::decode("100000000000000000000000000000000000000000000001")
                .unwrap()
                .try_into()
                .unwrap(),
            aad: b"BLAKE3".to_vec(),
            expected_ciphertext: hex::decode(
                "48fecfaf8d9553bfe7121700da72362e77e09080ddd55101aaca18cdcf259953923150cb89e1fef2",
            )
            .unwrap(),
        },
        Test {
            plaintext: hex::decode("b8f60975cd7057a003ac84df00d514624fe40cb7855c50dd6594f59b3a2580e5").unwrap(),
            key: hex::decode("3eb02a239a2a66de159b9bb5486ccc10a6f63ddf5862ef076650513372353622")
                .unwrap()
                .try_into()
                .unwrap(),
            nonce: hex::decode("768e9bda14afb5686cc34de26210f9ff6fa1dfadc64ee3f0")
                .unwrap()
                .try_into()
                .unwrap(),
            aad: hex::decode("c8d69ca92da6c5fd22f1805179fcd36cb7a9d45848fa346ba7118c2f34d23a48").unwrap(),
            expected_ciphertext: hex::decode(
                "444d593bb2dea9ecde9cd3839d166141de70481340ce30739b3f0f28b059d63232324ace49e8a19729ac5110a093fba10acaeed93099dea1a9c20463a278c3a7",
            )
            .unwrap(),
        },
    ];

    for (i, test) in tests.iter().enumerate() {
        let cipher = ChaCha20Blake3::new(test.key);
        let ciphertext = cipher.encrypt(&test.nonce, &test.plaintext, &test.aad);

        assert_eq!(
            ciphertext,
            test.expected_ciphertext,
            "encryption [{i}] failed. Got: {}\nExpected: {}",
            hex::encode(&ciphertext),
            hex::encode(&test.expected_ciphertext)
        );

        let plaintext = cipher.decrypt(&test.nonce, &ciphertext, &test.aad).unwrap();
        assert_eq!(
            plaintext,
            test.plaintext,
            "decryption [{i}] failed. Got: {}\nExpected: {}",
            hex::encode(&plaintext),
            hex::encode(&test.plaintext)
        );
    }
}

#[test]
fn session_matches_aead_first_message() {
    let key = [0x01u8; 32];
    let nonce: [u8; 24] = [0x02u8; 24];
    let aad = b"test aad";
    let plaintext = b"hello world, this is a test of session vs aead";

    let mut kdf_out = [0u8; 72];
    blake3::Hasher::new_keyed(&key)
        .update(&nonce)
        .finalize_xof()
        .fill(&mut kdf_out);

    let enc_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
    let auth_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
    let enc_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

    let aead = ChaCha20Blake3::new(key);
    let aead_ct = aead.encrypt(&nonce, plaintext, aad);

    let mut session = Session20::new(enc_key, auth_key, enc_nonce);
    let session_ct = session.encrypt(plaintext, aad);

    assert_eq!(aead_ct, session_ct, "session first-message output must match AEAD output");
}

#[test]
fn session_multi_message_roundtrip() {
    let enc_key = [0x42u8; 32];
    let auth_key = [0x43u8; 32];
    let nonce = [0x44u8; 8];
    let aad = b"multi";

    let mut encoder = Session20::new(enc_key, auth_key, nonce);
    let mut decoder = Session20::new(enc_key, auth_key, nonce);

    let messages: &[&[u8]] = &[b"hello", b"world", b"!", &[0xffu8; 100], &[0u8; 200], b""];

    let encrypted: Vec<Vec<u8>> = messages.iter().map(|m| encoder.encrypt(m, aad)).collect();

    for (i, ct) in encrypted.iter().enumerate() {
        let pt = decoder.decrypt(ct, aad).unwrap();
        assert_eq!(pt, messages[i], "message {i} mismatch");
    }
}

#[test]
fn session_block_counter_advances() {
    let mut session = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    assert_eq!(session.block_counter(), 0);

    session.encrypt(b"", b"");
    assert_eq!(session.block_counter(), 0);

    session.encrypt(&[0u8; 1], b"");
    assert_eq!(session.block_counter(), 1);

    session.encrypt(&[0u8; 64], b"");
    assert_eq!(session.block_counter(), 2);

    session.encrypt(&[0u8; 65], b"");
    assert_eq!(session.block_counter(), 4);

    session.encrypt(&[0u8; 128], b"");
    assert_eq!(session.block_counter(), 6);
}

#[test]
fn session_wrong_tag_rejected() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct = encoder.encrypt(b"secret", b"aad");
    let mut corrupted = ct.clone();
    corrupted[0] ^= 0xff;

    assert!(decoder.decrypt(&corrupted, b"aad").is_err());
}

#[test]
fn session_wrong_aad_rejected() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct = encoder.encrypt(b"secret", b"correct aad");

    assert!(decoder.decrypt(&ct, b"wrong aad").is_err());
}

#[test]
fn session_counter_desync_produces_garbage() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct1 = encoder.encrypt(b"first", b"");
    let ct2 = encoder.encrypt(b"second", b"");

    // MAC doesn't bind to the counter, so out-of-order messages pass MAC verification
    // but decrypt to garbage because the keystream position is wrong.
    let garbage = decoder.decrypt(&ct2, b"").unwrap();
    assert_ne!(garbage.as_slice(), b"second");

    // ct1 also decrypts to garbage now because the decoder counter advanced past position 0.
    let garbage2 = decoder.decrypt(&ct1, b"").unwrap();
    assert_ne!(garbage2.as_slice(), b"first");
}

#[test]
fn session_size_sweep() {
    let sizes = [
        0, 1, 2, 3, 7, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
        4096, 16384, 65536,
    ];

    for &size in &sizes {
        let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
        let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

        let plaintext: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let ct = encoder.encrypt(&plaintext, b"aad");
        assert_eq!(ct.len(), plaintext.len() + TAG_SIZE);

        let pt = decoder.decrypt(&ct, b"aad").unwrap();
        assert_eq!(pt, plaintext, "roundtrip failed for size {size}");
    }
}

#[test]
fn session_size_sweep_multi_message() {
    let sizes = [
        0, 1, 2, 3, 7, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
        4096,
    ];

    let mut encoder = Session20::new([0xaau8; 32], [0xbbu8; 32], [0xccu8; 8]);
    let mut decoder = Session20::new([0xaau8; 32], [0xbbu8; 32], [0xccu8; 8]);

    let mut pairs: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for &size in &sizes {
        let plaintext: Vec<u8> = (0..size).map(|i| (i % 199) as u8).collect();
        let ct = encoder.encrypt(&plaintext, b"sweep");
        pairs.push((plaintext, ct));
    }

    for (i, (plaintext, ct)) in pairs.iter().enumerate() {
        let pt = decoder.decrypt(ct, b"sweep").unwrap();
        assert_eq!(
            &pt,
            plaintext,
            "multi-message size sweep failed at index {i} (size {})",
            plaintext.len()
        );
    }

    assert_eq!(encoder.block_counter(), decoder.block_counter());
}

#[test]
fn session_in_place_roundtrip() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let original = b"test message for in-place API roundtrip";
    let aad = b"some aad";

    let mut buf = original.to_vec();
    let tag = encoder.encrypt_in_place_detached(&mut buf, aad);
    assert_ne!(buf.as_slice(), original.as_slice());

    decoder.decrypt_in_place_detached(&mut buf, &tag, aad).unwrap();
    assert_eq!(buf.as_slice(), original.as_slice());
}

#[test]
fn session_in_place_multi_message() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let messages: &[&[u8]] = &[b"alpha", b"bravo", &[0u8; 300], b"", &[0xffu8; 65]];
    let mut encrypted: Vec<(Vec<u8>, [u8; 32])> = Vec::new();

    for msg in messages {
        let mut buf = msg.to_vec();
        let tag = encoder.encrypt_in_place_detached(&mut buf, b"ip");
        encrypted.push((buf, tag));
    }

    for (i, (ct, tag)) in encrypted.iter_mut().enumerate() {
        decoder.decrypt_in_place_detached(ct, tag, b"ip").unwrap();
        assert_eq!(ct.as_slice(), messages[i], "in-place message {i} mismatch");
    }
}

#[test]
fn session_1000_messages() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    for i in 0u32..1000 {
        let msg = i.to_le_bytes();
        let ct = encoder.encrypt(&msg, b"");
        let pt = decoder.decrypt(&ct, b"").unwrap();
        assert_eq!(pt, msg, "message {i} failed");
    }

    assert_eq!(encoder.block_counter(), decoder.block_counter());
    assert_eq!(encoder.block_counter(), 1000);
}

#[test]
fn session_alternating_sizes() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let sizes = [1, 1024, 3, 512, 7, 256, 63, 128, 65, 64, 0, 4096, 1, 0, 65535];

    let pairs: Vec<(Vec<u8>, Vec<u8>)> = sizes
        .iter()
        .map(|&size| {
            let plaintext: Vec<u8> = (0..size).map(|i| (i % 199) as u8).collect();
            let ct = encoder.encrypt(&plaintext, b"alt");
            (plaintext, ct)
        })
        .collect();

    for (i, (plaintext, ct)) in pairs.iter().enumerate() {
        let pt = decoder.decrypt(ct, b"alt").unwrap();
        assert_eq!(&pt, plaintext, "alternating size message {i} failed");
    }
}

#[test]
fn session_deterministic() {
    let plaintext = b"determinism check";
    let aad = b"aad";

    let ct1 = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]).encrypt(plaintext, aad);
    let ct2 = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]).encrypt(plaintext, aad);

    assert_eq!(ct1, ct2);
}

#[test]
fn session_different_enc_keys_different_ciphertext() {
    let mut s1 = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut s2 = Session20::new([9u8; 32], [2u8; 32], [3u8; 8]);

    let pt = b"same plaintext!!";
    let ct1 = s1.encrypt(pt, b"aad");
    let ct2 = s2.encrypt(pt, b"aad");

    assert_ne!(&ct1[..pt.len()], &ct2[..pt.len()]);
}

#[test]
fn session_different_auth_keys_same_ciphertext_different_tag() {
    let mut s1 = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut s2 = Session20::new([1u8; 32], [9u8; 32], [3u8; 8]);

    let pt = b"same plaintext!!";
    let ct1 = s1.encrypt(pt, b"aad");
    let ct2 = s2.encrypt(pt, b"aad");

    assert_eq!(&ct1[..pt.len()], &ct2[..pt.len()]);
    assert_ne!(&ct1[pt.len()..], &ct2[pt.len()..]);
}

#[test]
fn session_different_nonces_different_ciphertext() {
    let mut s1 = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut s2 = Session20::new([1u8; 32], [2u8; 32], [9u8; 8]);

    let pt = b"same plaintext!!";
    let ct1 = s1.encrypt(pt, b"aad");
    let ct2 = s2.encrypt(pt, b"aad");

    assert_ne!(&ct1[..pt.len()], &ct2[..pt.len()]);
}

#[test]
fn session_wrong_auth_key_rejected() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let ct = encoder.encrypt(b"secret", b"aad");

    let mut wrong_auth = Session20::new([1u8; 32], [9u8; 32], [3u8; 8]);
    assert!(wrong_auth.decrypt(&ct, b"aad").is_err());
}

#[test]
fn session_wrong_enc_key_produces_garbage() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let ct = encoder.encrypt(b"secret", b"aad");

    // Same auth_key means MAC passes, but decryption uses wrong keystream.
    let mut wrong_enc = Session20::new([9u8; 32], [2u8; 32], [3u8; 8]);
    let pt = wrong_enc.decrypt(&ct, b"aad").unwrap();
    assert_ne!(pt.as_slice(), b"secret");
}

#[test]
fn session_wrong_nonce_produces_garbage() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let ct = encoder.encrypt(b"secret", b"aad");

    // Same auth_key means MAC passes, but different nonce produces wrong keystream.
    let mut wrong_nonce = Session20::new([1u8; 32], [2u8; 32], [9u8; 8]);
    let pt = wrong_nonce.decrypt(&ct, b"aad").unwrap();
    assert_ne!(pt.as_slice(), b"secret");
}

#[test]
fn session_tag_single_bit_flips() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let ct = encoder.encrypt(b"flip test data!!", b"aad");
    let tag_start = ct.len() - TAG_SIZE;

    for byte_idx in 0..TAG_SIZE {
        for bit in 0..8u8 {
            let mut corrupted = ct.clone();
            corrupted[tag_start + byte_idx] ^= 1 << bit;

            let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
            assert!(
                decoder.decrypt(&corrupted, b"aad").is_err(),
                "tag bit flip at byte {byte_idx} bit {bit} not detected"
            );
        }
    }
}

#[test]
fn session_ciphertext_single_bit_flips() {
    let plaintext = b"ciphertext flip!";
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let ct = encoder.encrypt(plaintext, b"aad");

    for byte_idx in 0..plaintext.len() {
        for bit in 0..8u8 {
            let mut corrupted = ct.clone();
            corrupted[byte_idx] ^= 1 << bit;

            let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
            assert!(
                decoder.decrypt(&corrupted, b"aad").is_err(),
                "ciphertext bit flip at byte {byte_idx} bit {bit} not detected"
            );
        }
    }
}

#[test]
fn session_empty_message() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct = encoder.encrypt(b"", b"aad");
    assert_eq!(ct.len(), TAG_SIZE);
    assert_eq!(encoder.block_counter(), 0);

    let pt = decoder.decrypt(&ct, b"aad").unwrap();
    assert!(pt.is_empty());
    assert_eq!(decoder.block_counter(), 0);
}

#[test]
fn session_empty_aad() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct = encoder.encrypt(b"no aad here", b"");
    let pt = decoder.decrypt(&ct, b"").unwrap();
    assert_eq!(pt, b"no aad here");
}

#[test]
fn session_large_aad() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let aad = vec![0xab; 65536];
    let ct = encoder.encrypt(b"large aad", &aad);
    let pt = decoder.decrypt(&ct, &aad).unwrap();
    assert_eq!(pt, b"large aad");
}

#[test]
fn session_decrypt_too_short() {
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    assert!(decoder.decrypt(&[0u8; 0], b"").is_err());
    assert!(decoder.decrypt(&[0u8; 1], b"").is_err());
    assert!(decoder.decrypt(&[0u8; 31], b"").is_err());

    // Counter must not advance on failed decrypt.
    assert_eq!(decoder.block_counter(), 0);
}

#[test]
fn session_failed_decrypt_does_not_advance_counter() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let ct = encoder.encrypt(b"real message", b"aad");

    // Feed garbage: MAC check fails, counter must stay at 0.
    let garbage = vec![0xffu8; 64];
    assert!(decoder.decrypt(&garbage, b"aad").is_err());
    assert_eq!(decoder.block_counter(), 0);

    // The real ciphertext still decrypts because the counter didn't advance.
    let pt = decoder.decrypt(&ct, b"aad").unwrap();
    assert_eq!(pt, b"real message");
    assert_eq!(decoder.block_counter(), 1);
}

#[test]
fn session_matches_aead_detached() {
    let key = [0x55u8; 32];
    let nonce: [u8; 24] = [0x66u8; 24];
    let aad = b"detached check";
    let plaintext = b"verify detached api matches too";

    let mut kdf_out = [0u8; 72];
    blake3::Hasher::new_keyed(&key)
        .update(&nonce)
        .finalize_xof()
        .fill(&mut kdf_out);
    let enc_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
    let auth_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
    let enc_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

    let aead = ChaCha20Blake3::new(key);
    let mut aead_buf = plaintext.to_vec();
    let aead_tag = aead.encrypt_in_place_detached(&nonce, &mut aead_buf, aad);

    let mut session = Session20::new(enc_key, auth_key, enc_nonce);
    let mut session_buf = plaintext.to_vec();
    let session_tag = session.encrypt_in_place_detached(&mut session_buf, aad);

    assert_eq!(aead_buf, session_buf);
    assert_eq!(aead_tag, session_tag);
}

#[test]
fn session_counter_sync_encoder_decoder() {
    let mut encoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);
    let mut decoder = Session20::new([1u8; 32], [2u8; 32], [3u8; 8]);

    let sizes = [0, 1, 63, 64, 65, 128, 200, 512, 1000, 0, 7];
    for &size in &sizes {
        let msg: Vec<u8> = (0..size).map(|i| i as u8).collect();
        let ct = encoder.encrypt(&msg, b"");
        let pt = decoder.decrypt(&ct, b"").unwrap();
        assert_eq!(pt, msg);
        assert_eq!(
            encoder.block_counter(),
            decoder.block_counter(),
            "counter desync after encrypting {size} bytes"
        );
    }
}

#[test]
fn session_zeroize_on_drop() {
    use std::mem::ManuallyDrop;

    let mut session = ManuallyDrop::new(Session20::new([0x42u8; 32], [0x43u8; 32], [0x44u8; 8]));

    // Exercise the cipher so the internal ChaCha state is populated.
    let _ = session.encrypt(b"exercise cipher state with some data", b"aad");

    let ptr = &*session as *const Session20 as *const u8;
    let size = std::mem::size_of::<Session20>();

    // Verify the memory contains non-zero data before drop.
    let pre_drop: Vec<u8> = (0..size)
        .map(|i| unsafe { core::ptr::read_volatile(ptr.add(i)) })
        .collect();
    assert!(
        pre_drop.iter().any(|&b| b != 0),
        "session memory should be non-zero before drop"
    );

    unsafe {
        ManuallyDrop::drop(&mut session);
    }

    // Every byte of the Session must be zero after ZeroizeOnDrop runs.
    for i in 0..size {
        let byte = unsafe { core::ptr::read_volatile(ptr.add(i)) };
        assert_eq!(byte, 0, "session memory not zeroed at offset {i} (value: 0x{byte:02x})");
    }
}
