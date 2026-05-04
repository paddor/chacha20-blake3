#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc = include_str!("README.md")]

mod chacha;
use chacha::ChaCha20;
use constant_time_eq::constant_time_eq_32;

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

pub const KEY_SIZE: usize = 32;
pub const NONCE_SIZE: usize = 24;
pub const TAG_SIZE: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct Error {}

#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct ChaCha20Blake3 {
    key: [u8; 32],
}

impl ChaCha20Blake3 {
    pub fn new(key: [u8; 32]) -> Self {
        return ChaCha20Blake3 { key };
    }

    #[cfg(feature = "alloc")]
    pub fn encrypt(&self, nonce: &[u8; 24], plaintext: &[u8], aad: &[u8]) -> Vec<u8> {
        let mut ciphertext = alloc::vec![0u8; plaintext.len() + TAG_SIZE];
        ciphertext[..plaintext.len()].copy_from_slice(&plaintext);

        let tag = self.encrypt_in_place_detached(nonce, &mut ciphertext[..plaintext.len()], aad);
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        return ciphertext;
    }

    #[cfg(feature = "alloc")]
    pub fn decrypt(&self, nonce: &[u8; 24], ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, Error> {
        if ciphertext.len() < TAG_SIZE {
            return Err(Error {});
        }

        let mut plaintext = alloc::vec![0u8; ciphertext.len() - TAG_SIZE];
        plaintext.copy_from_slice(&ciphertext[..ciphertext.len() - TAG_SIZE]);

        self.decrypt_in_place_detached(
            nonce,
            &mut plaintext,
            &ciphertext[ciphertext.len() - TAG_SIZE..].try_into().unwrap(),
            aad,
        )?;

        return Ok(plaintext);
    }

    pub fn encrypt_in_place_detached(&self, nonce: &[u8; 24], in_out: &mut [u8], aad: &[u8]) -> [u8; 32] {
        // kdf_out = BLAKE3.keyed(key, nonce)
        let mut kdf_out = [0u8; 72];
        let mut blake3_kdf = blake3::Hasher::new_keyed(&self.key);
        blake3_kdf.update(nonce);
        blake3_kdf.finalize_xof().fill(&mut kdf_out);

        // encryption_key = kdf_out[0..32]
        // authentication_key = kdf_out[32..64]
        // encryption_nonce = kdf_out[64..72]
        let encryption_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
        let authentication_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
        let encryption_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

        ChaCha20::new(&encryption_key, &encryption_nonce).xor_keystream(in_out);

        // mac = BLAKE3.keyed(authentication_key, aad || aad.len_uint64_little_endian() || ciphertext || ciphertext.len_uint64_little_endian())
        let mut mac_hasher = blake3::Hasher::new_keyed(&authentication_key);
        mac_hasher.update(aad);
        mac_hasher.update(&(aad.len() as u64).to_le_bytes());
        mac_hasher.update(&in_out);
        mac_hasher.update(&(in_out.len() as u64).to_le_bytes());
        let tag = mac_hasher.finalize();

        #[cfg(feature = "zeroize")]
        kdf_out.zeroize();

        return tag.into();
    }

    pub fn decrypt_in_place_detached(
        &self,
        nonce: &[u8; 24],
        ciphertext: &mut [u8],
        tag: &[u8; 32],
        aad: &[u8],
    ) -> Result<(), Error> {
        // kdf_out = BLAKE3.keyed(key, nonce)
        let mut kdf_out = [0u8; 72];
        let mut blake3_kdf = blake3::Hasher::new_keyed(&self.key);
        blake3_kdf.update(nonce);
        blake3_kdf.finalize_xof().fill(&mut kdf_out);

        // encryption_key = kdf_out[0..32]
        // authentication_key = kdf_out[32..64]
        // encryption_nonce = kdf_out[64..72]
        let encryption_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
        let authentication_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
        let encryption_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

        let mut mac_hasher = blake3::Hasher::new_keyed(&authentication_key);
        mac_hasher.update(aad);
        mac_hasher.update(&(aad.len() as u64).to_le_bytes());
        mac_hasher.update(&ciphertext);
        mac_hasher.update(&(ciphertext.len() as u64).to_le_bytes());
        let mac = mac_hasher.finalize();

        if !constant_time_eq_32(mac.as_bytes(), tag) {
            return Err(Error {});
        }

        ChaCha20::new(&encryption_key, &encryption_nonce).xor_keystream(ciphertext);

        #[cfg(feature = "zeroize")]
        Zeroizing::new(encryption_key).zeroize();

        return Ok(());
    }
}
