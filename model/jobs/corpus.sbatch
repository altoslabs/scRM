#!/bin/bash
# Build one corpus.
#
#   sbatch --partition=... --account=... model/jobs/corpus.sbatch <key>
#
# <key> is a row in model/jobs/corpora.tsv. Fetches the reads, materialises the called-cell
# whitelist, builds, verifies the manifest against the declared geometry, deletes the reads.
# Reads are released only when manifest.json exists, including when the corpus was already built.
#SBATCH --job-name=scrm-corpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --gres=disk:tmp:20G
set -euo pipefail

KEY=${1:?usage: sbatch $0 <key>   (a key in model/jobs/corpora.tsv)}
REPO=${SCRM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)}
PROJ=${SCRM_PROJ:?set SCRM_PROJ to the data root (corpora, runs, eval_artifacts)}
OUT=${SCRM_CORPUS_ROOT:-${CORPUS_ROOT:-${CORPORA_ROOT:-$PROJ/corpus}}}
export CORPORA_ROOT="$OUT" SCRM_CORPUS_ROOT="$OUT"
TABLE=$REPO/model/jobs/corpora.tsv
BUILDER=$REPO/target/release/scrm-corpus

PY=${SCRM_PY:-$(command -v python3 || true)}
[ -x "$PY" ] || { >&2 echo "REFUSING: no interpreter. Set SCRM_PY to an absolute path."; exit 1; }
LIB=$(cd "$(dirname "$(dirname "$PY")")/lib" 2>/dev/null && pwd || true)
[ -n "$LIB" ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$LIB
export PYTHONPATH=$REPO/model:$REPO/eval:$REPO/target/py

[ -x "$BUILDER" ] || { >&2 echo "REFUSING: no $BUILDER. cargo build --release -p scrm-cli"; exit 1; }

# ---- the row ------------------------------------------------------------------------------------
row=$(awk -F'\t' -v k="$KEY" '!/^#/ && $1==k {print; exit}' "$TABLE")
[ -n "$row" ] || {
    >&2 echo "REFUSING: $KEY is not declared in $TABLE. Known keys:"
    awk -F'\t' '!/^#/ && NF>1 && $1!="key" {printf "  %s\n", $1}' "$TABLE" >&2
    exit 1
}
IFS=$'\t' read -r _k ROLE KIND SOURCE WHITELIST GEOM SPECIES TOOLS NOTE <<< "$row"
echo "== $KEY  role=$ROLE kind=$KIND species=$SPECIES"
echo "   geometry  $GEOM      (declared; chemistry is never auto-detected)"
echo "   whitelist $WHITELIST"
[ -n "$NOTE" ] && echo "   note      $NOTE"
[ "$WHITELIST" != UNRESOLVED ] || {
    >&2 echo "REFUSING $KEY: its called-cell whitelist is unresolved. Fetching reads and building"
    >&2 echo "  against a GUESSED cell list produces a corpus with a different cell population than"
    >&2 echo "  the pinned one, which is worse than not building it."
    exit 1
}
field() { printf '%s' "$SOURCE" | tr ';' '\n' | awk -F= -v k="$1" '$1==k{print $2; exit}'; }

RAW=$PROJ/raw/$KEY
WL=$RAW/barcodes.tsv.gz
mkdir -p "$RAW"

tenx_base() {
    case "$(field host)" in
      cf) printf '%s' "https://cf.10xgenomics.com/samples/cell-exp/$(field ver)" ;;
      s3) printf '%s' "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/$(field ver)" ;;
      *)  >&2 echo "REFUSING: unknown tenx-tar host '$(field host)'"; exit 1 ;;
    esac
}

# The extra archive, and whatever a `derive:` whitelist derives from it, are ensured whether or not
# the corpus already exists. A derivation that has already run is re-checked rather than rewritten.
ensure_raw_inputs() {
    local extra; extra=$(field extra)
    if [ -n "$extra" ]; then
        local dst=$RAW/${extra##*/}
        case "$KIND" in
          tenx-tar)
            [ -s "$dst" ] || {
                local url=$(tenx_base)/$extra
                echo "extra: fetching $url"
                curl -sfL --retry 5 --retry-delay 15 --retry-all-errors --max-time 1800 \
                     -o "$dst.part" "$url" \
                  || { >&2 echo "REFUSING: could not fetch $url"; exit 1; }
                mv "$dst.part" "$dst"
            }
            tar xzf "$dst" -C "$RAW"
            echo "extra: $dst extracted into $RAW"
            ;;
          czb-s3)
            [ -s "$dst" ] || {
                local key=s3://$(field bucket)/$extra
                echo "extra: fetching $key"
                aws s3 cp --no-sign-request --quiet "$key" "$dst.part" \
                  || { >&2 echo "REFUSING: could not fetch $key"; exit 1; }
                mv "$dst.part" "$dst"
            }
            echo "extra: $dst ($(stat -c %s "$dst") bytes)"
            ;;
          *) >&2 echo "REFUSING: $KEY declares extra= but kind $KIND resolves no locator"; exit 1 ;;
        esac
    fi
    case "$WHITELIST" in
      derive:*)
        local mode=""
        [ -s "$WL" ] && mode=--check
        "$PY" "$REPO/model/jobs/derive_labels.py" "${WHITELIST#derive:}" --raw "$RAW" $mode \
          || { >&2 echo "REFUSING: the derivation declared by $KEY did not succeed"; exit 1; }
        ;;
    esac
}
ensure_raw_inputs

# ---- release ------------------------------------------------------------------------------------
READS_DIR=""            # set by each backend to the directory whose reads may be released
release_reads() {
    [ -n "$READS_DIR" ] && [ -d "$READS_DIR" ] || return 0
    local n; n=$(find "$READS_DIR" -name '*.fastq*' 2>/dev/null | wc -l)
    rm -rf "${READS_DIR:?}"
    echo "reads released: $n file(s) under $READS_DIR"
}

# ---- already built ------------------------------------------------------------------------------
if [ -f "$OUT/$KEY/manifest.json" ]; then
    echo "skip $KEY: corpus exists at $OUT/$KEY"
    case "$KIND" in
      tenx-tar|sra-reads) READS_DIR=${SCRM_WORK_ROOT:-${TMPDIR:-/tmp}}/$KEY ;;
      sra-bam)   READS_DIR=$PROJ/raw/got_mpn/$(field tag)/fastqs ;;
      ebi-files) READS_DIR="" ;;   # per-lane files, released by name below
      czb-s3)    READS_DIR=$PROJ/raw/tabula_muris/$(field channel) ;;
    esac
    release_reads
    if [ "$KIND" = ebi-files ]; then
        d=$PROJ/xspecies/${SPECIES}
        for f in "$d"/*.fastq.gz "$d"/*.fastq.gz.part; do [ -e "$f" ] && rm -f "$f" && echo "  released $f"; done
    fi
    exit 0
fi
[ -d "$OUT/$KEY" ] && { echo "clearing partial $OUT/$KEY"; rm -rf "${OUT:?}/${KEY:?}"; }

# ---- disk, before anything is downloaded --------------------------------------------------------
floor() {  # path, floor GB, what it is for
    local free; free=$(df -BG --output=avail "$1" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${free:-0}" -ge "$2" ] || {
        >&2 echo "REFUSING $KEY: $1 has ${free:-0} GB free, floor $2 GB ($3)."
        >&2 echo "  Set SCRM_WORK_ROOT to a filesystem with room; the reads and the assembler's"
        >&2 echo "  spill both go there."
        exit 1
    }
    echo "disk: $1 has ${free} GB free (floor $2)"
}
WORK_ROOT=${SCRM_WORK_ROOT:-${TMPDIR:-/tmp}}
WORK=$WORK_ROOT/$KEY
# The builder picks its spill device from TMPDIR; point it at the filesystem checked above.
export TMPDIR="$WORK"
mkdir -p "$WORK" "$OUT"
floor "$OUT" "${SCRM_CORPUS_FLOOR_GB:-25}" "the corpus being written"
floor "$WORK_ROOT" "${SCRM_WORK_FLOOR_GB:-80}" "the reads and the assembler's spill"

# ---- external tools, pinned ---------------------------------------------------------------------
BTF=""
need_tool() {
    case "$1" in
      bamtofastq)
        local url=https://github.com/10XGenomics/bamtofastq/releases/download/v1.4.1/bamtofastq_linux
        local size=18540256
        local sha=fdf7db4fe6cf8e13a432e8a59815dad506415349627502fdf77f4be208a198ce
        BTF=${SCRM_BAMTOFASTQ:-$(command -v bamtofastq || true)}
        if [ -z "$BTF" ] || [ ! -x "$BTF" ]; then
            BTF=$PROJ/bin/bamtofastq
            if [ ! -x "$BTF" ]; then
                echo "fetching pinned bamtofastq v1.4.1"
                mkdir -p "$(dirname "$BTF")"
                curl -sfL --retry 5 --retry-delay 10 --retry-all-errors -o "$BTF.part" "$url" \
                  || { >&2 echo "REFUSING: could not fetch $url; place the binary at $BTF"; exit 1; }
                mv "$BTF.part" "$BTF"; chmod +x "$BTF"
            fi
        fi
        local gs gh; gs=$(stat -c %s "$BTF"); gh=$(sha256sum "$BTF" | cut -d' ' -f1)
        [ "$gs" = "$size" ] && [ "$gh" = "$sha" ] || {
            >&2 echo "REFUSING: $BTF is $gs bytes / $gh, the pin is $size / $sha ($url)."
            >&2 echo "  bamtofastq decides the lane split, so a different build changes the input."
            exit 1
        }
        echo "tool: bamtofastq $BTF (pinned v1.4.1, digest verified)" ;;
      sra-tools)
        for t in prefetch fasterq-dump; do
            command -v "$t" >/dev/null || [ -x "$(dirname "$PY")/$t" ] || {
                >&2 echo "REFUSING: $t not found. It is declared in envs/scrm.yml (sra-tools)."; exit 1; }
        done
        echo "tool: sra-tools $(dirname "$PY")" ;;
      aws-cli)
        command -v aws >/dev/null || { >&2 echo "REFUSING: aws not found; declared in envs/scrm.yml"; exit 1; }
        echo "tool: aws $(command -v aws)" ;;
      -|"") : ;;
      *) >&2 echo "REFUSING: $KEY declares unknown tool '$1'"; exit 1 ;;
    esac
}
IFS=';' read -r -a _tools <<< "$TOOLS"
for t in "${_tools[@]}"; do need_tool "$t"; done

trap 'rm -rf "$WORK"' EXIT

# ---- fetch: one function per declared kind ------------------------------------------------------
SHEET=$WORK/samplesheet.csv
: > "$WORK/.pairs"                      # r1<TAB>r2 per lane, appended by each backend

pair() { printf '%s\t%s\n' "$1" "$2" >> "$WORK/.pairs"; }

fetch_tenx_tar() {
    local sample base
    sample=$(field sample); base=$(tenx_base)
    local url=$base/$sample/${sample}_fastqs.tar
    echo "reads: streaming $url"
    curl -sfL --retry 5 --retry-delay 15 --retry-all-errors --max-time 21600 "$url" \
      | tar x -C "$WORK" --strip-components=1 \
      || { >&2 echo "REFUSING: could not stream $url"; exit 1; }
    # Gene expression only, by an allow-list that fails loud. 10x puts each library in its own
    # directory, <sample>_fastqs/<sample>_gex_fastqs/ beside a guide-capture or antibody sibling,
    # so the separation is by library and never by read length. An R1 that cannot be positively
    # classified stops the build.
    local -a all_r1=() gex=() unknown=() dropped=()
    mapfile -t all_r1 < <(find "$WORK" -name '*_R1_*.fastq.gz' | sort)
    [ "${#all_r1[@]}" -gt 0 ] || { >&2 echo "REFUSING: no R1 fastqs in the tar for $KEY"; exit 1; }
    local non_gex='antibody|_ab_|totalseq|csp|feature|cmo|multiplexing|multiplexcapture|_mc_|crispr|guide|hashing|vdj|_bcr_|_tcr_'
    local has_gex_dir=0 r1 low
    for r1 in "${all_r1[@]}"; do
        case "$r1" in *_gex_*) has_gex_dir=1 ;; esac
    done
    for r1 in "${all_r1[@]}"; do
        low=$(printf '%s' "$r1" | tr 'A-Z' 'a-z')
        if [ "$has_gex_dir" = 1 ]; then
            case "$low" in *_gex_*) gex+=("$r1") ;; *) dropped+=("$r1") ;; esac
        elif printf '%s' "$low" | grep -Eq "$non_gex"; then
            unknown+=("$r1")
        else
            gex+=("$r1")
        fi
    done
    if [ "${#unknown[@]}" -gt 0 ]; then
        >&2 echo "REFUSING: these R1 files carry a non-gene-expression marker but the tar has no"
        >&2 echo "  _gex_ library directory to select against, so gene expression cannot be isolated:"
        >&2 printf '   %s\n' "${unknown[@]}"
        exit 1
    fi
    [ "${#gex[@]}" -gt 0 ] || { >&2 echo "REFUSING: no gene-expression R1 matched for $KEY"; exit 1; }
    echo "libraries: ${#all_r1[@]} R1 file(s) in the tar, ${#gex[@]} selected as gene expression"
    [ "${#dropped[@]}" -eq 0 ] || printf '   dropped: %s\n' "${dropped[@]##*/}"
    for r1 in "${gex[@]}"; do
        [ -s "${r1/_R1_/_R2_}" ] || { >&2 echo "no mate for $r1"; exit 1; }
        pair "$r1" "${r1/_R1_/_R2_}"
    done
    TENX_BASE=$base; TENX_SAMPLE=$sample
    READS_DIR=$WORK
}

fetch_sra_reads() {
    local srr; srr=$(field srr)
    local bin; bin=$(dirname "$PY")
    echo "reads: prefetch + fasterq-dump $srr"
    "$bin/prefetch" --max-size 200G -O "$WORK/sra" "$srr" >/dev/null
    "$bin/fasterq-dump" --split-files --include-technical -e 8 -O "$WORK" \
        "$(find "$WORK/sra" -name "$srr.sra" | head -1)" >/dev/null
    gzip -1 "$WORK/${srr}"_*.fastq
    pair "$WORK/${srr}_1.fastq.gz" "$WORK/${srr}_2.fastq.gz"
    READS_DIR=$WORK
}

fetch_sra_bam() {
    local srr tag bam fqdir
    srr=$(field srr); tag=$(field tag)
    local d=$PROJ/raw/got_mpn/$tag
    mkdir -p "$d"
    bam=$d/$tag.bam
    # EBI's SRA mirror serves the submitted Cell Ranger BAM directly. It is kept after the build.
    if [ ! -s "$bam" ]; then
        local url=https://ftp.sra.ebi.ac.uk/vol1/run/${srr:0:6}/$srr/$tag.possorted_genome_bam.bam.1
        echo "reads: fetching the Cell Ranger BAM $url"
        curl -sfL -C - --retry 8 --retry-delay 15 --retry-all-errors --max-time 21600 -o "$bam.part" "$url" \
          || { >&2 echo "REFUSING: could not fetch $url (partial kept; resubmit resumes)"; exit 1; }
        mv "$bam.part" "$bam"
    fi
    fqdir=$d/fastqs
    if [ "$(find "$fqdir" -name '*.fastq.gz' 2>/dev/null | wc -l)" -eq 0 ]; then
        rm -rf "$fqdir"
        srun "$BTF" --nthreads=8 "$bam" "$fqdir"
    fi
    for r1 in $(find "$fqdir" -name '*_R1_*.fastq.gz' | sort); do
        [ -s "${r1/_R1_/_R2_}" ] || { >&2 echo "no mate for $r1"; exit 1; }
        pair "$r1" "${r1/_R1_/_R2_}"
    done
    READS_DIR=$fqdir
}

fetch_ebi_files() {
    local acc sample lanes suffix d base
    acc=$(field accession); sample=$(field sample); suffix=$(field lane_suffix)
    IFS=',' read -r -a lanes <<< "$(field lanes)"
    d=$PROJ/xspecies/$SPECIES; mkdir -p "$d"
    # FIRE buckets an accession by the last three digits of its number.
    local num=${acc##*-}
    [ ${#num} -ge 3 ] || { >&2 echo "REFUSING: accession '$acc' has no 3-digit number"; exit 1; }
    base=https://ftp.ebi.ac.uk/biostudies/fire/${acc%-*}-/${num: -3}/$acc/Files
    # The per-lane files are removed once they are concatenated, so test the concatenation, which
    # is what the builder consumes.
    if [ -s "$d/R1.fastq.gz" ] && [ -s "$d/R2.fastq.gz" ]; then
        echo "  reusing  $d/R1.fastq.gz and $d/R2.fastq.gz (already concatenated)"
        for r in R1 R2; do
            gzip -t "$d/$r.fastq.gz" \
              || { >&2 echo "REFUSING: $d/$r.fastq.gz is a corrupt gzip; delete it to re-fetch"; exit 1; }
        done
        pair "$d/R1.fastq.gz" "$d/R2.fastq.gz"
        EBI_DIR=$d
        return 0
    fi
    for r in R1 R2; do
        local whole=() g f lane
        for g in "${lanes[@]}"; do
            f=${sample}_${g}_${suffix}_${r}_001.fastq.gz
            lane=$d/$f
            if [ -s "$lane" ]; then
                echo "  reusing  $f ($(stat -c %s "$lane") bytes, already whole)"
            else
                echo "  fetching $f"
                curl -sfL -C - --retry 8 --retry-delay 15 --retry-all-errors --max-time 14400 \
                     "$base/$f" -o "$lane.part" \
                  || { >&2 echo "REFUSING: could not fetch $base/$f (partial kept; resubmit resumes)"; exit 1; }
                gzip -t "$lane.part" || { >&2 echo "REFUSING: $f is a corrupt gzip"; exit 1; }
                mv "$lane.part" "$lane"
            fi
            whole+=("$lane")
        done
        cat "${whole[@]}" > "$d/$r.fastq.gz.cat" && mv "$d/$r.fastq.gz.cat" "$d/$r.fastq.gz"
        rm -f "${whole[@]}"
    done
    pair "$d/R1.fastq.gz" "$d/R2.fastq.gz"
    EBI_DIR=$d
}

fetch_czb_s3() {
    local bucket prefix channel d
    bucket=$(field bucket); prefix=$(field prefix); channel=$(field channel)
    d=$PROJ/raw/tabula_muris/$channel; mkdir -p "$d"
    echo "reads: s3://$bucket/$prefix/$channel/ (no-sign-request)"
    aws s3 cp --no-sign-request --quiet --recursive --exclude '*' --include '*_R[12]_*.fastq.gz' \
        "s3://$bucket/$prefix/$channel/" "$d/" \
      || { >&2 echo "REFUSING: could not fetch s3://$bucket/$prefix/$channel/"; exit 1; }
    for r1 in $(find "$d" -name '*_R1_*.fastq.gz' | sort); do
        [ -s "${r1/_R1_/_R2_}" ] || { >&2 echo "no mate for $r1"; exit 1; }
        pair "$r1" "${r1/_R1_/_R2_}"
    done
    READS_DIR=$d
}

# permuted-from derives a corpus from a built one with scrm-permute: no reads, no whitelist, no
# samplesheet, so it completes here rather than entering the fetch path.
if [ "$KIND" = permuted-from ]; then
    SRC_KEY=$(field src)
    [ -n "$SRC_KEY" ] || { >&2 echo "REFUSING: $KEY declares no src="; exit 1; }
    SRC=$OUT/$SRC_KEY
    [ -f "$SRC/manifest.json" ] || {
        >&2 echo "REFUSING $KEY: $SRC has no manifest, so there is nothing to permute."
        >&2 echo "  Build $SRC_KEY first."
        exit 1
    }
    PERMUTE=$REPO/target/release/scrm-permute
    [ -x "$PERMUTE" ] || { >&2 echo "REFUSING: no $PERMUTE. cargo build --release"; exit 1; }
    SEED=$(field seed)
    [ -n "$SEED" ] || { >&2 echo "REFUSING: $KEY declares no seed="; exit 1; }
    srun "$PERMUTE" --src "$SRC" --out "$OUT/$KEY" --seed "$SEED"
    [ -f "$OUT/$KEY/manifest.json" ] || { >&2 echo "permute wrote no manifest for $KEY"; exit 1; }
    "$PY" - "$OUT/$KEY/manifest.json" "$SRC/manifest.json" <<'DERIVED_CHECK'
import json, sys
new, src = (json.load(open(p)) for p in sys.argv[1:3])
nc, sc = new["declared"]["config"], src["declared"]["config"]
for k in ("geometry", "k", "dedup_rule", "bubble_resolution", "min_umi_len", "max_umi_len"):
    if nc[k] != sc[k]:
        raise SystemExit(f"permuted corpus disagrees with its source on {k}: {nc[k]!r} vs {sc[k]!r}")
if new["n_rows"] != src["n_rows"]:
    raise SystemExit(f"permutation changed the molecule count: {new['n_rows']} vs {src['n_rows']}")
print(f"derived: {new['n_rows']:,} molecules permuted across cells, molecule definition unchanged")
DERIVED_CHECK
    echo "== $KEY complete (derived from $SRC_KEY; no reads fetched, so none released)"
    exit 0
fi

case "$KIND" in
  tenx-tar)  fetch_tenx_tar ;;
  sra-reads) fetch_sra_reads ;;
  sra-bam)   fetch_sra_bam ;;
  ebi-files) fetch_ebi_files ;;
  czb-s3)    fetch_czb_s3 ;;
  *) >&2 echo "REFUSING: $KEY declares unknown kind '$KIND'"; exit 1 ;;
esac
n_pairs=$(wc -l < "$WORK/.pairs")
[ "$n_pairs" -gt 0 ] || { >&2 echo "REFUSING: no read pairs for $KEY"; exit 1; }
echo "reads: $n_pairs lane pair(s)"

# ---- the called-cell whitelist ------------------------------------------------------------------
case "$WHITELIST" in
  h5|h5-multi)
    # 10x publishes two layouts for the filtered matrix and the row declares which one it is. For a
    # multi-library sample the h5 sits in a doubled directory under the name with the _Multiplex
    # suffix dropped.
    if [ "$WHITELIST" = h5-multi ]; then
        pre=${TENX_SAMPLE%_Multiplex}
        url=$TENX_BASE/${pre}_${pre}/${pre}_${pre}_count_sample_filtered_feature_bc_matrix.h5
    else
        url=$TENX_BASE/$TENX_SAMPLE/${TENX_SAMPLE}_filtered_feature_bc_matrix.h5
    fi
    [ -s "$RAW/filtered.h5" ] || curl -sfL --retry 5 --max-time 1800 -o "$RAW/filtered.h5" "$url" \
      || { >&2 echo "REFUSING: could not fetch $url"; exit 1; }
    "$PY" - "$RAW/filtered.h5" "$WL" <<'PYEOF'
import gzip, sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    g = f["matrix"] if "matrix" in f else f[list(f)[0]]
    bcs = [b.decode() if isinstance(b, bytes) else str(b) for b in g["barcodes"][:]]
with gzip.GzipFile(sys.argv[2], "wb", mtime=0) as fh:   # mtime=0: same barcodes -> same sha256
    fh.write("".join(b + "\n" for b in bcs).encode())
print(f"whitelist: {len(bcs)} called cells (from the sample's filtered_feature_bc_matrix.h5)")
PYEOF
    ;;
  tar:*)
    # Pre-h5 Cell Ranger: the called-cell list is a member of the filtered matrix tarball.
    member=${WHITELIST#tar:}
    url=$TENX_BASE/$TENX_SAMPLE/${TENX_SAMPLE}_filtered_gene_bc_matrices.tar.gz
    curl -sfL --retry 5 --max-time 1800 "$url" | tar xzO "$member" | gzip -n -1 > "$WL" \
      || { >&2 echo "REFUSING: could not extract $member from $url"; exit 1; }
    echo "whitelist: $(zcat "$WL" | wc -l) called cells (tarball member $member)"
    ;;
  repo:*)
    src=$REPO/${WHITELIST#repo:}
    [ -s "$src" ] || { >&2 echo "REFUSING: $src is not in this tree, and it is the called-cell source"; >&2 echo "  for $KEY: a list that cannot be fetched from a recorded URL ships in the tree."; exit 1; }
    cp "$src" "$WL"
    echo "whitelist: $(zcat "$WL" | wc -l) called cells (ships with the tree: ${WHITELIST#repo:})"
    ;;
  geo:*)
    # A GEO supplementary file. Called cells are those with >= 2 informative reads.
    spec=${WHITELIST#geo:}; gsm=${spec%%#*}; file=${spec#*#}
    tgt=$RAW/${gsm}_${file}
    [ -s "$tgt" ] || curl -sfL --retry 5 --max-time 1800 -o "$tgt" \
        "https://ftp.ncbi.nlm.nih.gov/geo/samples/${gsm:0:7}nnn/$gsm/suppl/${gsm}_${file}" \
      || { >&2 echo "REFUSING: could not fetch ${gsm}_${file}"; exit 1; }
    "$PY" - "$tgt" "$WL" "$RAW/labels.json" <<'PYEOF'
import gzip, json, sys
src, wl, lab = sys.argv[1:4]
labels = {}
with gzip.open(src, "rt") as f:
    f.readline()
    for line in f:
        bc, n_wt, n_mut = line.split()
        if int(n_wt) + int(n_mut) >= 2:
            labels[bc] = "MUT" if int(n_mut) > 0 else "WT"
json.dump(labels, open(lab, "w"), indent=0)
with gzip.GzipFile(wl, "wb", mtime=0) as f:
    f.write("".join(bc + "-1\n" for bc in sorted(labels)).encode())
print(f"whitelist: {len(labels)} genotyped cells, "
      f"{sum(v == 'MUT' for v in labels.values())} mutant")
PYEOF
    ;;
  json:*)
    # A per-sample barcode map on the data root; the expected count is declared in the table.
    spec=${WHITELIST#json:}; jf=${spec%%#*}; samp=${spec#*#}
    "$PY" - "$PROJ/$jf" "$samp" "$(field cells_called)" "$WL" <<'PYEOF'
import gzip, json, sys
src, samp, want, wl = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
import pathlib
p = pathlib.Path(src)
if not p.is_file():
    raise SystemExit(f"{p} is absent; it is the called-cell source for this sample")
per = json.loads(p.read_text())
if samp not in per:
    raise SystemExit(f"{p} names {sorted(per)}, not {samp!r}")
bcs = sorted(set(per[samp]))
if len(bcs) != want:
    raise SystemExit(
        f"whitelist has {len(bcs)} barcodes, the table declares {want}")
with gzip.GzipFile(wl, "wb", mtime=0) as f:
    f.write(("\n".join(bcs) + "\n").encode())
print(f"whitelist: {len(bcs)} called cells (declared count matches)")
PYEOF
    ;;
  matrix:*)
    # A published count matrix; its barcodes file is the called-cell list.
    mp=$PROJ/${WHITELIST#matrix:}
    [ -d "$mp" ] || { >&2 echo "REFUSING: $mp is absent. It is a published matrix;"; exit 1; }
    b=$(find "$mp" -maxdepth 1 -name 'barcodes.tsv*' | head -1)
    [ -s "$b" ] || { >&2 echo "REFUSING: no barcodes.tsv under $mp"; exit 1; }
    case "$b" in *.gz) cp "$b" "$WL" ;; *) gzip -n -1 -c "$b" > "$WL" ;; esac
    echo "whitelist: $(zcat "$WL" | wc -l) called cells (matrix ${WHITELIST#matrix:})"
    ;;
  derive:*)
    [ -s "$WL" ] || { >&2 echo "REFUSING: the derivation wrote no $WL"; exit 1; }
    echo "whitelist: $(zcat "$WL" | wc -l) cells (derived by model/jobs/derive_labels.py)"
    ;;
  *) >&2 echo "REFUSING: $KEY declares unknown whitelist scheme '$WHITELIST'"; exit 1 ;;
esac
[ -s "$WL" ] || { >&2 echo "REFUSING: the whitelist came out empty"; exit 1; }

# ---- build ---------------------------------------------------------------------------------------
{ echo "experiment,r1,r2,barcodes,geometry"
  while IFS=$'\t' read -r r1 r2; do echo "$KEY,$r1,$r2,$WL,$GEOM"; done < "$WORK/.pairs"
} > "$SHEET"
echo "samplesheet: $(( $(wc -l < "$SHEET") - 1 )) row(s)"
srun "$BUILDER" --samplesheet "$SHEET" --out "$OUT" --threads 32

# ---- verify, then release ------------------------------------------------------------------------
[ -f "$OUT/$KEY/manifest.json" ] || {
    >&2 echo "NOT released: $OUT/$KEY/manifest.json is absent, so the build did not finish."
    >&2 echo "  Reads kept."
    exit 1
}
"$PY" - "$OUT/$KEY/manifest.json" "$GEOM" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
cfg = m["declared"]["config"]
if cfg["geometry"] != sys.argv[2]:
    raise SystemExit(f"manifest geometry {cfg['geometry']!r} != declared {sys.argv[2]!r}")
if cfg["trim"].get("trimmed_here") is not True:
    raise SystemExit("manifest does not stamp trimmed_here=true; this binary trims in process")
print(f"built: {m['n_rows']:,} molecules, {m['n_tokens']:,} tokens, "
      f"{cfg['geometry']}, trimmed_here=true")
PYEOF
if [ "$KIND" = ebi-files ]; then
    for f in "$EBI_DIR"/*.fastq.gz "$EBI_DIR"/*.fastq.gz.part; do
        [ -e "$f" ] && rm -f "$f" && echo "reads released: $f"
    done
else
    release_reads
fi
echo "== $KEY complete"
