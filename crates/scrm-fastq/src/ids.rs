#[inline]
pub fn core(id: &[u8]) -> &[u8] {
    strip_mate_suffix(strip_comment(id))
}

#[inline]
pub fn strip_comment(id: &[u8]) -> &[u8] {
    match id.iter().position(|&b| b == b' ' || b == b'\t') {
        Some(i) => &id[..i],
        None => id,
    }
}

#[inline]
pub fn strip_mate_suffix(id: &[u8]) -> &[u8] {
    let n = id.len();
    if n >= 2 && id[n - 2] == b'/' && (id[n - 1] == b'1' || id[n - 1] == b'2') {
        &id[..n - 2]
    } else {
        id
    }
}

#[inline]
pub fn matches(r1: &[u8], r2: &[u8]) -> bool {
    core(r1) == core(r2)
}

pub fn display(id: &[u8]) -> String {
    String::from_utf8_lossy(id).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_ids_match() {
        assert!(matches(b"A00519:1:HK:1:1101:1:1", b"A00519:1:HK:1:1101:1:1"));
    }

    #[test]
    fn mate_suffixes_are_stripped() {
        assert!(matches(b"READ001/1", b"READ001/2"));
        assert_eq!(core(b"READ001/1"), b"READ001");
    }

    #[test]
    fn casava_comment_fields_are_stripped() {
        assert!(matches(
            b"A00519:1:HK:1:1101:1:1 1:N:0:AGGTTCGA",
            b"A00519:1:HK:1:1101:1:1 2:N:0:AGGTTCGA"
        ));
    }

    #[test]
    fn both_forms_at_once_are_stripped() {
        assert!(matches(b"READ001/1 1:N:0:AGGT", b"READ001/2 2:N:0:AGGT"));
        assert_eq!(core(b"READ001/1 1:N:0:AGGT"), b"READ001");
    }

    #[test]
    fn tab_separated_comments_are_stripped() {
        assert!(matches(b"READ001\tBC:Z:AAA", b"READ001\tBC:Z:CCC"));
    }

    #[test]
    fn different_names_do_not_match() {
        assert!(!matches(b"READ001", b"READ002"));
        assert!(!matches(b"READ001/1", b"READ002/2"));
        assert!(!matches(
            b"A00519:1:HK:1:1101:1:1 1:N:0:AGGT",
            b"A00519:1:HK:1:1101:9:9 2:N:0:AGGT"
        ));
    }

    #[test]
    fn a_prefix_is_not_a_match() {
        assert!(!matches(b"READ1", b"READ10"));
    }

    #[test]
    fn only_one_and_two_count_as_mate_suffixes() {
        assert_eq!(core(b"READ/3"), b"READ/3");
        assert_eq!(core(b"READ/"), b"READ/");
        assert_eq!(core(b"/1"), b"");
    }

    #[test]
    fn empty_ids_are_handled() {
        assert!(matches(b"", b""));
        assert_eq!(core(b" 1:N:0:A"), b"");
    }
}
