#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![doc = include_str!("README.md")]

mod chacha;
use chacha::ChaCha;
use constant_time_eq::constant_time_eq_32;

#[cfg(feature = "zeroize")]
use zeroize::{Zeroize, ZeroizeOnDrop};

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Key size in bytes (256-bit).
pub const KEY_SIZE: usize = 32;
/// Nonce size in bytes (192-bit).
pub const NONCE_SIZE: usize = 24;
/// Authentication tag size in bytes (256-bit).
pub const TAG_SIZE: usize = 32;
/// ChaCha block size in bytes.
pub const BLOCK_SIZE: usize = 64;

/// Stateful AEAD session using ChaCha8.
pub type Session8 = Session<8>;
/// Stateful AEAD session using ChaCha12.
pub type Session12 = Session<12>;
/// Stateful AEAD session using ChaCha20.
pub type Session20 = Session<20>;

/// Authentication or decryption failure.
#[derive(Clone, Copy, Debug)]
pub struct Error {}

/// ChaCha8-BLAKE3 AEAD.
pub type ChaCha8Blake3 = ChaChaBlake3<8>;
/// ChaCha12-BLAKE3 AEAD.
pub type ChaCha12Blake3 = ChaChaBlake3<12>;
/// ChaCha20-BLAKE3 AEAD.
pub type ChaCha20Blake3 = ChaChaBlake3<20>;

/// AEAD using ChaCha with `ROUNDS` rounds for encryption and BLAKE3 for KDF
/// and MAC. 24-byte nonce, 32-byte tag.
///
/// Each call to [`encrypt`](Self::encrypt) /
/// [`decrypt`](Self::decrypt) derives per-message encryption and
/// authentication keys via BLAKE3 KDF. For high-throughput streaming where
/// the KDF cost matters, use [`Session`] instead.
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct ChaChaBlake3<const ROUNDS: usize> {
    key: [u8; 32],
}

impl<const ROUNDS: usize> ChaChaBlake3<ROUNDS> {
    /// Create a new cipher instance with the given 256-bit key.
    pub fn new(key: [u8; 32]) -> Self {
        ChaChaBlake3 { key }
    }

    /// Encrypt `plaintext` with associated data. Returns `ciphertext || tag`.
    #[cfg(feature = "alloc")]
    pub fn encrypt(&self, nonce: &[u8; 24], plaintext: &[u8], aad: &[u8]) -> Vec<u8> {
        let mut ciphertext = alloc::vec![0u8; plaintext.len() + TAG_SIZE];
        ciphertext[..plaintext.len()].copy_from_slice(plaintext);

        let tag = self.encrypt_in_place_detached(nonce, &mut ciphertext[..plaintext.len()], aad);
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        ciphertext
    }

    /// Decrypt `ciphertext` (which must end with the 32-byte tag) with
    /// associated data. Returns the plaintext or [`Error`] on authentication
    /// failure.
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

        Ok(plaintext)
    }

    /// Encrypt `in_out` in place. Returns the 32-byte authentication tag.
    #[cfg_attr(not(feature = "zeroize"), expect(unused_mut))]
    pub fn encrypt_in_place_detached(&self, nonce: &[u8; 24], in_out: &mut [u8], aad: &[u8]) -> [u8; 32] {
        let mut kdf_out = [0u8; 72];
        let mut blake3_kdf = blake3::Hasher::new_keyed(&self.key);
        blake3_kdf.update(nonce);
        blake3_kdf.finalize_xof().fill(&mut kdf_out);

        let mut encryption_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
        let mut authentication_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
        let mut encryption_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

        ChaCha::<ROUNDS>::new(&encryption_key, &encryption_nonce).xor_keystream(in_out);

        let mut mac_hasher = blake3::Hasher::new_keyed(&authentication_key);
        mac_hasher.update(aad);
        mac_hasher.update(&(aad.len() as u64).to_le_bytes());
        mac_hasher.update(in_out);
        mac_hasher.update(&(in_out.len() as u64).to_le_bytes());
        let tag = mac_hasher.finalize();

        #[cfg(feature = "zeroize")]
        {
            kdf_out.zeroize();
            encryption_key.zeroize();
            authentication_key.zeroize();
            encryption_nonce.zeroize();
        }

        tag.into()
    }

    /// Decrypt `ciphertext` in place. Returns [`Error`] if the tag does not
    /// match (constant-time comparison).
    #[cfg_attr(not(feature = "zeroize"), expect(unused_mut))]
    pub fn decrypt_in_place_detached(
        &self,
        nonce: &[u8; 24],
        ciphertext: &mut [u8],
        tag: &[u8; 32],
        aad: &[u8],
    ) -> Result<(), Error> {
        let mut kdf_out = [0u8; 72];
        let mut blake3_kdf = blake3::Hasher::new_keyed(&self.key);
        blake3_kdf.update(nonce);
        blake3_kdf.finalize_xof().fill(&mut kdf_out);

        let mut encryption_key: [u8; 32] = kdf_out[..32].try_into().unwrap();
        let mut authentication_key: [u8; 32] = kdf_out[32..64].try_into().unwrap();
        let mut encryption_nonce: [u8; 8] = kdf_out[64..].try_into().unwrap();

        let mut mac_hasher = blake3::Hasher::new_keyed(&authentication_key);
        mac_hasher.update(aad);
        mac_hasher.update(&(aad.len() as u64).to_le_bytes());
        mac_hasher.update(ciphertext);
        mac_hasher.update(&(ciphertext.len() as u64).to_le_bytes());
        let mac = mac_hasher.finalize();

        let result = if !constant_time_eq_32(mac.as_bytes(), tag) {
            Err(Error {})
        } else {
            ChaCha::<ROUNDS>::new(&encryption_key, &encryption_nonce).xor_keystream(ciphertext);
            Ok(())
        };

        #[cfg(feature = "zeroize")]
        {
            kdf_out.zeroize();
            encryption_key.zeroize();
            authentication_key.zeroize();
            encryption_nonce.zeroize();
        }

        result
    }
}

/// Stateful AEAD that skips the per-message KDF.
///
/// Accepts pre-derived encryption and authentication keys directly and
/// tracks a continuous ChaCha block counter across messages. Useful when
/// the caller manages key derivation externally or needs to encrypt many
/// small messages without paying the BLAKE3 KDF cost each time.
///
/// Messages must be processed in order. The MAC does not bind the counter,
/// so reordered messages pass authentication but decrypt to wrong
/// plaintext. Failed decryptions do not advance the counter.
#[cfg_attr(feature = "zeroize", derive(Zeroize, ZeroizeOnDrop))]
pub struct Session<const ROUNDS: usize> {
    cipher: ChaCha<ROUNDS>,
    auth_key: [u8; 32],
    block_counter: u64,
}

impl<const ROUNDS: usize> Session<ROUNDS> {
    /// Create a session from pre-derived keys and nonce. Counter starts at 0.
    pub fn new(encryption_key: [u8; 32], authentication_key: [u8; 32], encryption_nonce: [u8; 8]) -> Self {
        Session {
            cipher: ChaCha::<ROUNDS>::new(&encryption_key, &encryption_nonce),
            auth_key: authentication_key,
            block_counter: 0,
        }
    }

    /// Current ChaCha block counter (number of 64-byte blocks consumed).
    pub fn block_counter(&self) -> u64 {
        self.block_counter
    }

    /// Encrypt `plaintext` with associated data. Returns `ciphertext || tag`.
    #[cfg(feature = "alloc")]
    pub fn encrypt(&mut self, plaintext: &[u8], aad: &[u8]) -> Vec<u8> {
        let mut ciphertext = alloc::vec![0u8; plaintext.len() + TAG_SIZE];
        ciphertext[..plaintext.len()].copy_from_slice(plaintext);

        let tag = self.encrypt_in_place_detached(&mut ciphertext[..plaintext.len()], aad);
        ciphertext[plaintext.len()..].copy_from_slice(&tag);

        ciphertext
    }

    /// Decrypt `ciphertext` (ending with 32-byte tag). Returns [`Error`] on
    /// authentication failure without advancing the counter.
    #[cfg(feature = "alloc")]
    pub fn decrypt(&mut self, ciphertext: &[u8], aad: &[u8]) -> Result<Vec<u8>, Error> {
        if ciphertext.len() < TAG_SIZE {
            return Err(Error {});
        }

        let mut plaintext = alloc::vec![0u8; ciphertext.len() - TAG_SIZE];
        plaintext.copy_from_slice(&ciphertext[..ciphertext.len() - TAG_SIZE]);

        self.decrypt_in_place_detached(
            &mut plaintext,
            &ciphertext[ciphertext.len() - TAG_SIZE..].try_into().unwrap(),
            aad,
        )?;

        Ok(plaintext)
    }

    /// Encrypt `in_out` in place. Returns the 32-byte authentication tag.
    pub fn encrypt_in_place_detached(&mut self, in_out: &mut [u8], aad: &[u8]) -> [u8; 32] {
        self.cipher.set_counter(self.block_counter);
        self.cipher.xor_keystream(in_out);
        self.advance_counter(in_out.len());

        let mut mac = blake3::Hasher::new_keyed(&self.auth_key);
        mac.update(aad);
        mac.update(&(aad.len() as u64).to_le_bytes());
        mac.update(in_out);
        mac.update(&(in_out.len() as u64).to_le_bytes());
        mac.finalize().into()
    }

    /// Decrypt `ciphertext` in place. Returns [`Error`] on authentication
    /// failure without advancing the counter.
    pub fn decrypt_in_place_detached(
        &mut self,
        ciphertext: &mut [u8],
        tag: &[u8; 32],
        aad: &[u8],
    ) -> Result<(), Error> {
        let mut mac = blake3::Hasher::new_keyed(&self.auth_key);
        mac.update(aad);
        mac.update(&(aad.len() as u64).to_le_bytes());
        mac.update(ciphertext);
        mac.update(&(ciphertext.len() as u64).to_le_bytes());

        if !constant_time_eq_32(mac.finalize().as_bytes(), tag) {
            return Err(Error {});
        }

        self.cipher.set_counter(self.block_counter);
        self.cipher.xor_keystream(ciphertext);
        self.advance_counter(ciphertext.len());

        Ok(())
    }

    fn advance_counter(&mut self, bytes: usize) {
        self.block_counter = self.block_counter.wrapping_add((bytes as u64).div_ceil(64));
    }
}
