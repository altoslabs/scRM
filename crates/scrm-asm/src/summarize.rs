use debruijn::Exts;
use debruijn::filter::KmerSummarizer;

pub const MIN_KMER_OBS: usize = 1;

#[derive(Debug, Clone, Copy, Default)]
pub struct WeightSummarizer;

impl KmerSummarizer<u32, u32> for WeightSummarizer {
    fn summarize<K, F: Iterator<Item = (K, Exts, u32)>>(&self, items: F) -> (bool, Exts, u32) {
        let mut all_exts = Exts::empty();
        let mut weight: u32 = 0;
        let mut n_obs: usize = 0;

        for (_, exts, w) in items {
            weight = weight.saturating_add(w);
            all_exts = all_exts.add(exts);
            n_obs += 1;
        }

        (n_obs >= MIN_KMER_OBS, all_exts, weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use debruijn::{Dir, Kmer};
    use scrm_core::Kmer19;

    fn obs(weights: &[u32]) -> Vec<(Kmer19, Exts, u32)> {
        weights
            .iter()
            .map(|&w| (Kmer19::empty(), Exts::mk_right(0), w))
            .collect()
    }

    #[test]
    fn sums_weights_rather_than_counting_observations() {
        let (valid, exts, w) = WeightSummarizer.summarize(obs(&[2, 1]).into_iter());
        assert!(valid);
        assert_eq!(w, 3);
        assert!(exts.has_ext(Dir::Right, 0));
    }

    #[test]
    fn a_single_observation_is_retained() {
        let (valid, _, w) = WeightSummarizer.summarize(obs(&[1]).into_iter());
        assert!(valid);
        assert_eq!(w, 1);
    }

    #[test]
    fn weight_saturates_instead_of_wrapping() {
        let (_, _, w) = WeightSummarizer.summarize(obs(&[u32::MAX, 5]).into_iter());
        assert_eq!(w, u32::MAX);
    }
}
