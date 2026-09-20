//! Where the trimming time goes, per step, at the real read shape.
use scrm_trim::artifacts::ReadFlags;
use scrm_trim::{HOMOPOLYMER_MIN_RUN, ILLUMINA_TRUSEQ_3P, QUALITY_CUTOFF, adapter, homopolymer, quality, tso};
use std::time::Instant;

fn reads(n: usize, len: usize) -> Vec<Vec<u8>> {
    // xorshift, so the shape is realistic and the run is reproducible
    let mut x: u64 = 0x2545F4914F6CDD1D;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    (0..n)
        .map(|i| {
            let mut r: Vec<u8> = (0..len).map(|_| b"ACGT"[(next() % 4) as usize]).collect();
            if i % 1000 == 0 {
                // ~0.1% carry a read-through adapter, as measured
                let at = len - ILLUMINA_TRUSEQ_3P.len();
                r[at..].copy_from_slice(ILLUMINA_TRUSEQ_3P);
            }
            if i % 20 == 0 {
                // ~5% carry a TSO, as measured
                r[..tso::TSO.len().min(len)].copy_from_slice(&tso::TSO[..tso::TSO.len().min(len)]);
            }
            r
        })
        .collect()
}

fn main() {
    let n = 400_000;
    let rs = reads(n, 90);
    let quals: Vec<Vec<u8>> = rs.iter().map(|r| vec![b'I'; r.len()]).collect();
    let mut sink = 0usize;

    macro_rules! time {
        ($label:expr, $body:expr) => {{
            let t = Instant::now();
            $body;
            let e = t.elapsed().as_secs_f64();
            println!("  {:<34} {:>7.3}s   {:>10.0} reads/s", $label, e, n as f64 / e);
            e
        }};
    }

    let a = time!("homopolymer::measure (was there)", {
        for r in &rs {
            let mut f = ReadFlags::default();
            homopolymer::measure(r, &mut f);
            sink += (&f as *const _ as usize) & 1;
        }
    });
    let b = time!("tso::find, 30bp probe (was there)", {
        for r in &rs {
            sink += tso::find(r, tso::TSO, tso::MIN_SCORE).is_some() as usize;
        }
    });
    let c = time!("adapter::three_prime_cut (NEW)", {
        for r in &rs {
            sink += adapter::three_prime_cut(r, ILLUMINA_TRUSEQ_3P, 5, 0.1);
        }
    });
    let d = time!("quality::quality_trim_end (NEW)", {
        for q in &quals {
            sink += quality::quality_trim_end(q, QUALITY_CUTOFF, 33);
        }
    });
    let e = time!("homopolymer::trim_anchored (NEW)", {
        for r in &rs {
            sink += homopolymer::trim_anchored(r, HOMOPOLYMER_MIN_RUN).start;
        }
    });
    println!();
    println!(
        "  pre-existing: {:.3}s      newly added: {:.3}s   ({:.0}% added)",
        a + b,
        c + d + e,
        100.0 * (c + d + e) / (a + b)
    );
    println!("  (sink {sink})");
}
