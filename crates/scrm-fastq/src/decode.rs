use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

pub type Source = Box<dyn Read + Send>;

const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];

const BUF_BYTES: usize = 1 << 20;

pub fn is_gzip(path: &Path) -> io::Result<bool> {
    let mut magic = [0u8; 2];
    let mut file = File::open(path)?;
    match file.read(&mut magic)? {
        2 => Ok(magic == GZIP_MAGIC),
        _ => Ok(false),
    }
}

pub fn open(path: &Path, decoder_threads: usize) -> io::Result<Source> {
    if !is_gzip(path)? {
        return Ok(Box::new(BufReader::with_capacity(BUF_BYTES, File::open(path)?)));
    }
    open_gzip(path, decoder_threads)
}

fn open_gzip(path: &Path, decoder_threads: usize) -> io::Result<Source> {
    use rapidgzip_core::Decoder;

    let threads = decoder_threads.max(1);
    let decoder = Decoder::builder()
        .decoder_threads(threads)
        .build()
        .map_err(|e| io::Error::other(format!("rapidgzip config: {e}")))?;
    let reader = decoder
        .open(path)
        .map_err(|e| io::Error::other(format!("rapidgzip open {}: {e}", path.display())))?;
    Ok(Box::new(reader))
}

pub const fn backend() -> &'static str {
    "rapidgzip-core"
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scrm-decode-{}-{}", std::process::id(), name));
        p
    }

    #[test]
    fn plain_input_is_passed_through() {
        let path = tmp("plain.fastq");
        std::fs::write(&path, b"@r1\nACGT\n+\nIIII\n").unwrap();
        assert!(!is_gzip(&path).unwrap());
        let mut out = String::new();
        open(&path, 1).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "@r1\nACGT\n+\nIIII\n");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn gzip_input_is_decompressed() {
        let path = tmp("one.fastq.gz");
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(b"@r1\nACGT\n+\nIIII\n").unwrap();
        std::fs::write(&path, enc.finish().unwrap()).unwrap();
        assert!(is_gzip(&path).unwrap());
        let mut out = String::new();
        open(&path, 2).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out, "@r1\nACGT\n+\nIIII\n");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concatenated_gzip_members_all_decode() {
        let path = tmp("multi.fastq.gz");
        let mut bytes = Vec::new();
        for i in 0..3 {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            enc.write_all(format!("@r{i}\nACGT\n+\nIIII\n").as_bytes()).unwrap();
            bytes.extend_from_slice(&enc.finish().unwrap());
        }
        std::fs::write(&path, &bytes).unwrap();
        let mut out = String::new();
        open(&path, 2).unwrap().read_to_string(&mut out).unwrap();
        assert_eq!(out.matches('@').count(), 3, "{out}");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn an_empty_file_is_not_gzip() {
        let path = tmp("empty.fastq");
        std::fs::write(&path, b"").unwrap();
        assert!(!is_gzip(&path).unwrap());
        std::fs::remove_file(&path).ok();
    }
}
