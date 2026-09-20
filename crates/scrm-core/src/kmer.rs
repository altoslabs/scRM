use debruijn::kmer::{KmerSize, VarIntKmer};

pub const K_DEFAULT: usize = 19;

pub const K_VALUES: [usize; 12] = [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];

pub const K: usize = K_DEFAULT;

pub const MIN_TRIM_LEN: usize = 20;

const _: () = assert!(K_DEFAULT < MIN_TRIM_LEN);

pub const fn min_trim_len_for(k: usize) -> usize {
    if k + 1 > MIN_TRIM_LEN { k + 1 } else { MIN_TRIM_LEN }
}

pub fn validate_k(k: usize) -> Result<(), String> {
    if K_VALUES.contains(&k) {
        return Ok(());
    }
    let why = if k.is_multiple_of(2) {
        " (k must be odd: an even k-mer can be its own reverse complement)"
    } else {
        ""
    };
    Err(format!(
        "k = {k} is not supported{why}; this build offers {:?}",
        K_VALUES
    ))
}

const _: () = assert!(2 * K <= 64);

macro_rules! declare_ksizes {
    ($($marker:ident = $k:literal, $kmer:ident;)*) => {
        $(
            #[doc = concat!("Marker for k = ", stringify!($k), ".")]
            #[derive(Debug, Hash, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
            pub struct $marker;

            impl KmerSize for $marker {
                #[inline(always)]
                fn K() -> usize {
                    $k
                }
            }

            #[doc = concat!("k-mer type for k = ", stringify!($k), ", `u64`-backed.")]
            pub type $kmer = VarIntKmer<u64, $marker>;
        )*
    };
}

declare_ksizes! {
    K9  = 9,  Kmer9;
    K11 = 11, Kmer11;
    K13 = 13, Kmer13;
    K15 = 15, Kmer15;
    K17 = 17, Kmer17;
    K19 = 19, Kmer19;
    K21 = 21, Kmer21;
    K23 = 23, Kmer23;
    K25 = 25, Kmer25;
    K27 = 27, Kmer27;
    K29 = 29, Kmer29;
    K31 = 31, Kmer31;
}

#[macro_export]
macro_rules! dispatch_k {
    ($k:expr, |$KS:ident| $body:block) => {{
        match $k {
            9 => {
                type $KS = $crate::K9;
                $body
            }
            11 => {
                type $KS = $crate::K11;
                $body
            }
            13 => {
                type $KS = $crate::K13;
                $body
            }
            15 => {
                type $KS = $crate::K15;
                $body
            }
            17 => {
                type $KS = $crate::K17;
                $body
            }
            19 => {
                type $KS = $crate::K19;
                $body
            }
            21 => {
                type $KS = $crate::K21;
                $body
            }
            23 => {
                type $KS = $crate::K23;
                $body
            }
            25 => {
                type $KS = $crate::K25;
                $body
            }
            27 => {
                type $KS = $crate::K27;
                $body
            }
            29 => {
                type $KS = $crate::K29;
                $body
            }
            31 => {
                type $KS = $crate::K31;
                $body
            }
            other => panic!("unsupported k = {other}; call validate_k first"),
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use debruijn::Kmer;

    #[test]
    fn every_marker_reports_its_own_k() {
        assert_eq!(<K9 as KmerSize>::K(), 9);
        assert_eq!(Kmer9::k(), 9);
        assert_eq!(<K19 as KmerSize>::K(), 19);
        assert_eq!(Kmer19::k(), 19);
        assert_eq!(<K31 as KmerSize>::K(), 31);
        assert_eq!(Kmer31::k(), 31);
    }

    #[test]
    fn dispatch_reaches_the_requested_k() {
        for k in K_VALUES {
            let got = dispatch_k!(k, |KS| { <KS as KmerSize>::K() });
            assert_eq!(got, k, "dispatch_k sent {k} to a marker for {got}");
        }
    }

    #[test]
    fn only_odd_k_in_range_is_accepted() {
        for k in K_VALUES {
            assert!(validate_k(k).is_ok(), "{k}");
            assert!(!k.is_multiple_of(2), "{k} is even");
            assert!(2 * k <= 64, "k={k} does not fit a u64");
        }
        let e = validate_k(20).unwrap_err();
        assert!(e.contains("must be odd"), "{e}");
        assert!(validate_k(7).is_err());
        assert!(validate_k(33).is_err());
    }

    #[test]
    fn the_floor_always_leaves_two_kmers() {
        for k in K_VALUES {
            let floor = min_trim_len_for(k);
            assert!(floor > k, "k={k} floor={floor}");
            assert!(
                floor - k + 1 >= 2,
                "k={k} floor={floor} yields {} k-mers",
                floor - k + 1
            );
        }
        assert_eq!(min_trim_len_for(19), MIN_TRIM_LEN);
        assert_eq!(min_trim_len_for(9), MIN_TRIM_LEN);
        assert_eq!(min_trim_len_for(21), 22);
        assert_eq!(min_trim_len_for(31), 32);
    }

    #[test]
    fn shortest_retained_read_yields_two_kmers() {
        let n_kmers = MIN_TRIM_LEN - K + 1;
        assert_eq!(n_kmers, 2, "k={K}, min_trim_len={MIN_TRIM_LEN}");
    }
}
