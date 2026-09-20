use crate::alphabet::Token;

pub const MAX_PACK_LEN: usize = 32;

#[inline]
pub fn pack_acgt(bytes: &[u8]) -> Option<u64> {
    if bytes.len() > MAX_PACK_LEN {
        return None;
    }
    let mut packed: u64 = 0;
    for &b in bytes {
        let bits = Token::from_acgt(b)? as u64;
        packed = (packed << 2) | bits;
    }
    Some(packed)
}

#[inline]
pub fn unpack_acgt(packed: u64, len: usize) -> Vec<u8> {
    debug_assert!(len <= MAX_PACK_LEN);
    let mut out = Vec::with_capacity(len);
    for i in (0..len).rev() {
        let bits = ((packed >> (2 * i)) & 0b11) as u8;
        out.push(match bits {
            0 => b'A',
            1 => b'C',
            2 => b'G',
            _ => b'T',
        });
    }
    out
}

#[inline]
pub fn hamming_packed(a: u64, b: u64) -> u32 {
    let bit_diffs = a ^ b;
    let two_bit_diffs = (bit_diffs | (bit_diffs >> 1)) & 0x5555_5555_5555_5555;
    two_bit_diffs.count_ones()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        for s in [
            &b"ACGT"[..],
            &b"AAAAAAAAAAAAAAAA"[..],
            &b"TTTTTTTTTTTT"[..],
            &b"ACGTACGTACGTACGTACGTACGTACGTACGT"[..],
        ] {
            let p = pack_acgt(s).expect("packable");
            assert_eq!(unpack_acgt(p, s.len()), s.to_vec(), "{s:?}");
        }
    }

    #[test]
    fn first_base_occupies_high_bits() {
        let c_first = pack_acgt(b"CAAA").unwrap();
        let c_last = pack_acgt(b"AAAC").unwrap();
        assert_eq!(c_first, 0b01_00_00_00);
        assert_eq!(c_last, 0b00_00_00_01);
    }

    #[test]
    fn n_is_rejected_not_substituted() {
        assert_eq!(pack_acgt(b"ACGN"), None);
        assert_eq!(pack_acgt(b"NNNN"), None);
        assert_ne!(pack_acgt(b"ACGN"), pack_acgt(b"ACGA"));
    }

    #[test]
    fn over_length_is_rejected() {
        assert_eq!(pack_acgt(&[b'A'; MAX_PACK_LEN + 1]), None);
        assert!(pack_acgt(&[b'A'; MAX_PACK_LEN]).is_some());
    }

    #[test]
    fn hamming_counts_bases_not_bits() {
        let a = pack_acgt(b"ACGTACGTAC").unwrap();
        assert_eq!(hamming_packed(a, a), 0);
        let one = pack_acgt(b"TCGTACGTAC").unwrap();
        assert_eq!(hamming_packed(a, one), 1);
        let two = pack_acgt(b"TCGTACGTAT").unwrap();
        assert_eq!(hamming_packed(a, two), 2);
    }
}
