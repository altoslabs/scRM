pub const ALPHABET_SIZE: u8 = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Token {
    A = 0,
    C = 1,
    G = 2,
    T = 3,
    N = 4,
    Tab = 5,
    Newline = 6,
}

impl Token {
    #[inline]
    pub const fn as_ascii(self) -> u8 {
        match self {
            Token::A => b'A',
            Token::C => b'C',
            Token::G => b'G',
            Token::T => b'T',
            Token::N => b'N',
            Token::Tab => b'\t',
            Token::Newline => b'\n',
        }
    }

    #[inline]
    pub const fn from_acgt(b: u8) -> Option<Token> {
        match b {
            b'A' | b'a' => Some(Token::A),
            b'C' | b'c' => Some(Token::C),
            b'G' | b'g' => Some(Token::G),
            b'T' | b't' => Some(Token::T),
            _ => None,
        }
    }

    #[inline]
    pub const fn is_base(self) -> bool {
        matches!(self, Token::A | Token::C | Token::G | Token::T)
    }
}

const _: () = assert!(Token::Newline as u8 == ALPHABET_SIZE - 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_ids_match_debruijn_convention() {
        assert_eq!(Token::A as u8, debruijn::base_to_bits(b'A'));
        assert_eq!(Token::C as u8, debruijn::base_to_bits(b'C'));
        assert_eq!(Token::G as u8, debruijn::base_to_bits(b'G'));
        assert_eq!(Token::T as u8, debruijn::base_to_bits(b'T'));
    }

    #[test]
    fn n_is_not_parseable_from_input() {
        assert_eq!(Token::from_acgt(b'N'), None);
        assert_eq!(Token::from_acgt(b'n'), None);
    }

    #[test]
    fn separators_are_not_bases() {
        assert!(!Token::Tab.is_base());
        assert!(!Token::Newline.is_base());
        assert!(!Token::N.is_base());
    }
}
