//! Multi-model OCR ensemble (pure-Rust, testable layer).
//!
//! Implements the ensemble design from `docs/research/ocr-vlm-models.md`
//! §4: per-engine `EngineBlock` inputs are normalized, pairwise-scored
//! (fuzzy IoU + partial normalized Levenshtein), matched per engine pair
//! with the Hungarian algorithm, clustered with union-find (at most one
//! block per engine per cluster), and merged into consensus `OcrBlock`s
//! with high/medium/low evidence levels.
//!
//! Actual model inference is out of scope here — engines are faked in
//! tests and wired to real runtimes (ort / llama-cpp-2 / tesseract-rs) in
//! a follow-up milestone.

use crate::unionfind::UnionFind;
use serde::Serialize;

/// Text scripts recognized by engines, per research §4 steps 1-2.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct ScriptSet {
    pub han: bool,
    pub kana: bool,
    pub latin: bool,
}

impl ScriptSet {
    pub const NONE: ScriptSet = ScriptSet {
        han: false,
        kana: false,
        latin: false,
    };
    pub const HAN: ScriptSet = ScriptSet {
        han: true,
        kana: false,
        latin: false,
    };
    pub const KANA: ScriptSet = ScriptSet {
        han: false,
        kana: true,
        latin: false,
    };
    pub const LATIN: ScriptSet = ScriptSet {
        han: false,
        kana: false,
        latin: true,
    };

    /// True when the two sets share at least one script.
    pub fn intersects(self, other: ScriptSet) -> bool {
        (self.han && other.han) || (self.kana && other.kana) || (self.latin && other.latin)
    }
}

/// Detect the scripts present in a text (Han / Kana / Latin code ranges).
pub fn detect_scripts(text: &str) -> ScriptSet {
    let mut set = ScriptSet::NONE;
    for c in text.chars() {
        let u = c as u32;
        if (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u) {
            set.han = true;
        } else if (0x3040..=0x30FF).contains(&u) || (0x31F0..=0x31FF).contains(&u) {
            set.kana = true;
        } else if c.is_ascii_alphabetic() || (0x00C0..=0x024F).contains(&u) {
            set.latin = true;
        }
    }
    set
}

/// Evidence level of a consensus block (research §4 step 6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Evidence {
    High,
    Medium,
    Low,
}

/// Normalized rectangle, coordinates in [0, 1] (research §4 input contract).
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Rect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

impl Rect {
    fn expand_vertically(self, amount: f32) -> Rect {
        Rect {
            x0: self.x0,
            y0: (self.y0 - amount).max(0.0),
            x1: self.x1,
            y1: (self.y1 + amount).min(1.0),
        }
    }

    fn area(&self) -> f32 {
        (self.x1 - self.x0).max(0.0) * (self.y1 - self.y0).max(0.0)
    }

    fn intersect(&self, other: &Rect) -> Option<Rect> {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = self.x1.min(other.x1);
        let y1 = self.y1.min(other.y1);
        (x1 > x0 && y1 > y0).then_some(Rect { x0, y0, x1, y1 })
    }

    /// Height/width ratio > 2 counts as vertical (tategaki) text.
    fn is_vertical(&self) -> bool {
        let w = (self.x1 - self.x0).max(f32::EPSILON);
        let h = (self.y1 - self.y0).max(f32::EPSILON);
        h / w > 2.0
    }
}

/// One text block emitted by one OCR engine (research §4 input).
#[derive(Debug, Clone, Serialize)]
pub struct EngineBlock {
    /// Engine identifier, e.g. "paddle-ocrv5-mobile".
    pub engine: String,
    /// Raw text as emitted (normalization is comparison-only).
    pub text: String,
    /// Normalized box in [0, 1]; `None` for engines that emit text only.
    pub bbox: Option<Rect>,
    /// Engine self-confidence in [0, 1], when provided.
    pub conf: Option<f32>,
    /// Scripts present in `text` (detected automatically via [`detect_scripts`]).
    pub scripts: ScriptSet,
}

impl EngineBlock {
    pub fn new(engine: &str, text: &str, bbox: Option<Rect>, conf: Option<f32>) -> Self {
        Self {
            engine: engine.to_string(),
            text: text.to_string(),
            bbox,
            conf,
            scripts: detect_scripts(text),
        }
    }
}

/// Static capability profile of an engine (research §4 steps 2/5-6).
#[derive(Debug, Clone)]
pub struct EngineProfile {
    /// Engine identifier; must match `EngineBlock::engine`.
    pub id: String,
    /// Vote weight (research §5 calibration: Paddle/Qwen 1.0, Tesseract 0.6,
    /// SmolVLM/Moondream 0.5 — starting values).
    pub weight: f32,
    /// Scripts the engine can read; blocks outside these are not voted on.
    pub scripts: ScriptSet,
    /// Detection-class engine (Paddle / Tesseract) vs VLM-class.
    pub detector: bool,
}

impl EngineProfile {
    pub fn new(id: &str, weight: f32, scripts: ScriptSet, detector: bool) -> Self {
        Self {
            id: id.to_string(),
            weight,
            scripts,
            detector,
        }
    }
}

/// Merged consensus text block (card acceptance shape).
#[derive(Debug, Clone, Serialize)]
pub struct OcrBlock {
    /// Consensus text (medoid's original, un-normalized).
    pub text: String,
    /// Medoid's box, when any member had one.
    pub bbox: Option<Rect>,
    /// Engine id of the medoid (consensus source).
    pub model: String,
    /// Evidence level.
    pub evidence: Evidence,
    /// VLM-only block with no detector frame nearby (research §4 step 6).
    pub possible_hallucination: bool,
}

/// Normalize text for comparison only: NFKC (also folds fullwidth to
/// halfwidth), strip zero-width characters, lowercase, collapse whitespace.
pub fn normalize(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    let folded: String = text
        .nfkc()
        .filter(|c| {
            let u = *c as u32;
            !(0x200B..=0x200D).contains(&u) && u != 0xFEFF
        })
        .collect();
    folded
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Standard Levenshtein edit distance over chars.
fn levenshtein(a: &[char], b: &[char]) -> usize {
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let substitution = prev[j] + usize::from(ca != cb);
            cur[j + 1] = substitution.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Levenshtein similarity in [0, 1] on normalized text (research §4 step 3).
pub fn normalized_levenshtein(a: &str, b: &str) -> f32 {
    let ca: Vec<char> = normalize(a).chars().collect();
    let cb: Vec<char> = normalize(b).chars().collect();
    let longest = ca.len().max(cb.len());
    if longest == 0 {
        return 1.0;
    }
    1.0 - (levenshtein(&ca, &cb) as f32 / longest as f32)
}

/// Partial similarity (research §4 step 3): the shorter string slides over
/// the longer one; the best window similarity wins. Handles VLM engines
/// merging multiple lines into one string.
pub fn partial_ratio(a: &str, b: &str) -> f32 {
    let sa: Vec<char> = normalize(a).chars().collect();
    let sb: Vec<char> = normalize(b).chars().collect();
    let (short, long) = if sa.len() <= sb.len() {
        (&sa, &sb)
    } else {
        (&sb, &sa)
    };
    if short.is_empty() {
        return 1.0;
    }
    if long.len() == short.len() {
        return normalized_levenshtein(a, b);
    }
    let mut best = 0.0f32;
    for start in 0..=(long.len() - short.len()) {
        let window = &long[start..start + short.len()];
        let distance = levenshtein(short, window);
        let score = 1.0 - (distance as f32 / short.len() as f32);
        if score > best {
            best = score;
        }
    }
    best
}

/// Fuzzy IoU (research §4 step 3): expand both boxes by 0.25× their line
/// height on each vertical side, then take `max(IoU, intersection / smaller_area)`.
pub fn fuzzy_iou(a: &Rect, b: &Rect) -> f32 {
    let ea = a.expand_vertically(0.25 * (a.y1 - a.y0));
    let eb = b.expand_vertically(0.25 * (b.y1 - b.y0));
    let inter = ea.intersect(&eb).map(|r| r.area()).unwrap_or(0.0);
    let union = ea.area() + eb.area() - inter;
    if union <= 0.0 {
        return 0.0;
    }
    let iou = inter / union;
    let smaller = ea.area().min(eb.area());
    let containment = if smaller > 0.0 { inter / smaller } else { 0.0 };
    iou.max(containment)
}

/// Pairing score and threshold (research §4 step 3):
/// both boxes → `S = 0.4·IoU' + 0.6·T`, τ = 0.55; any box missing →
/// `S = T`, τ = 0.75.
pub fn pair_score(a: &EngineBlock, b: &EngineBlock) -> (f32, f32) {
    match (a.bbox, b.bbox) {
        (Some(ra), Some(rb)) => {
            let s = 0.4 * fuzzy_iou(&ra, &rb) + 0.6 * normalized_levenshtein(&a.text, &b.text);
            (s, 0.55)
        }
        _ => (normalized_levenshtein(&a.text, &b.text), 0.75),
    }
}

/// Maximum-weight bipartite matching (Hungarian / Kuhn-Munkres).
///
/// `score[i][j]` = weight of pairing left `i` with right `j`; non-negative
/// weights, zero means "not matchable". Returns `(total, match)` where
/// `match[i] = Some(j)`. Rectangular inputs are padded to square with
/// zeros; lefts matched to padding come back as `None`.
fn hungarian(score: &[Vec<f32>]) -> (f32, Vec<Option<usize>>) {
    let lefts = score.len();
    if lefts == 0 {
        return (0.0, Vec::new());
    }
    let rights = score.first().map_or(0, |row| row.len());
    if rights == 0 {
        return (0.0, vec![None; lefts]);
    }
    let n = lefts.max(rights);
    // Minimize -weight: square cost matrix, zero padding for the larger side.
    let at = |i: usize, j: usize| -> f32 {
        if i < lefts && j < rights {
            -score[i][j]
        } else {
            0.0
        }
    };
    // Classic e-maxx assignment loop with potentials, 1-indexed helpers.
    const INF: f32 = 1.0e30;
    let mut u = vec![0.0f32; n + 1];
    let mut v = vec![0.0f32; n + 1];
    let mut p = vec![0usize; n + 1]; // p[j] = matched left (1-based), 0 = none
    let mut way = vec![0usize; n + 1];
    for i in 1..=n {
        p[0] = i;
        let mut minv = vec![INF; n + 1];
        let mut used = vec![false; n + 1];
        let mut j0 = 0usize;
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = INF;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = at(i0 - 1, j - 1) - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        while j0 != 0 {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        }
    }
    let mut total = 0.0f32;
    let mut matched = vec![None; lefts];
    for j in 1..=n {
        let i = p[j];
        if i > 0 && i <= lefts && j <= rights {
            let s = score[i - 1][j - 1];
            if s > 0.0 {
                matched[i - 1] = Some(j - 1);
                total += s;
            }
        }
    }
    (total, matched)
}

/// Match two engines' block lists (research §4 step 4): Hungarian pairing on
/// the score matrix, dropping pairs below their threshold. Inputs are
/// (block index, block) pairs so results map back to `blocks` by index.
fn match_engine_pair(
    a: &[(usize, &EngineBlock)],
    b: &[(usize, &EngineBlock)],
) -> Vec<(usize, usize, f32)> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let matrix: Vec<Vec<f32>> = a
        .iter()
        .map(|(_, ba)| {
            b.iter()
                .map(|(_, bb)| {
                    let (s, tau) = pair_score(ba, bb);
                    if s >= tau {
                        s
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect();
    let (_, matched) = hungarian(&matrix);
    a.iter()
        .zip(&matched)
        .filter_map(|((ia, ba), m)| {
            m.and_then(|j| {
                let (ib, bb) = b[j];
                let (s, tau) = pair_score(ba, bb);
                (s >= tau).then_some((*ia, ib, s))
            })
        })
        .collect()
}

/// Cluster blocks via matched edges (research §4 step 4): highest-scoring
/// edges first, and an edge is skipped when the merge would put two blocks
/// of the same engine into one cluster. Returns index groups in
/// deterministic order (by smallest member index).
fn cluster_indices(blocks: &[EngineBlock], edges: &[(usize, usize, f32)]) -> Vec<Vec<usize>> {
    let mut uf = UnionFind::new(blocks.len());
    let mut sorted: Vec<&(usize, usize, f32)> = edges.iter().collect();
    sorted.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap_or(std::cmp::Ordering::Equal));
    for &&(i, j, _) in &sorted {
        let (ri, rj) = (uf.find(i), uf.find(j));
        if ri == rj {
            continue;
        }
        let engines_left: Vec<&str> = (0..blocks.len())
            .filter(|&k| uf.find(k) == ri)
            .map(|k| blocks[k].engine.as_str())
            .collect();
        let engines_right: Vec<&str> = (0..blocks.len())
            .filter(|&k| uf.find(k) == rj)
            .map(|k| blocks[k].engine.as_str())
            .collect();
        let clashes = engines_left.iter().any(|e| engines_right.contains(e));
        if !clashes {
            uf.union(i, j);
        }
    }

    let mut by_root: std::collections::BTreeMap<usize, Vec<usize>> = Default::default();
    for k in 0..blocks.len() {
        by_root.entry(uf.find(k)).or_default().push(k);
    }
    let mut groups: Vec<Vec<usize>> = by_root.into_values().collect();
    groups.sort_by_key(|group| group[0]);
    groups
}

/// Run the ensemble (research §4): eligible filtering → pairwise Hungarian
/// matching → union-find clustering (≤1 block per engine) → medoid consensus
/// → evidence scoring → reading-order output.
pub fn ensemble(blocks: &[EngineBlock], profiles: &[EngineProfile]) -> Vec<OcrBlock> {
    if blocks.is_empty() || profiles.is_empty() {
        return Vec::new();
    }
    let profile_of = |engine: &str| profiles.iter().find(|p| p.id == engine);

    // Eligibility screening (research §4 step 2): drop blocks whose engine
    // does not support the scripts it emitted.
    let kept: Vec<(usize, &EngineBlock)> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| profile_of(&b.engine).is_some_and(|p| b.scripts.intersects(p.scripts)))
        .collect();

    // Pairwise Hungarian matching over every engine pair.
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for (ai, pa) in profiles.iter().enumerate() {
        for pb in &profiles[ai + 1..] {
            let a_list: Vec<(usize, &EngineBlock)> = kept
                .iter()
                .copied()
                .filter(|(_, b)| b.engine == pa.id)
                .collect();
            let b_list: Vec<(usize, &EngineBlock)> = kept
                .iter()
                .copied()
                .filter(|(_, b)| b.engine == pb.id)
                .collect();
            edges.extend(match_engine_pair(&a_list, &b_list));
        }
    }

    // VLM full-text pool for the medium-evidence single-detector rule
    // (research §4 step 6: any VLM-emitted text, box or not).
    let vlm_fulltexts: Vec<&str> = kept
        .iter()
        .filter(|(_, b)| profile_of(&b.engine).is_some_and(|p| !p.detector))
        .map(|(_, b)| b.text.as_str())
        .collect();

    let groups = cluster_indices(blocks, &edges);
    // Clusters built over all blocks; singleton groups of eligibility-dropped
    // blocks must not leak into the output.
    let kept_set: std::collections::HashSet<usize> = kept.iter().map(|(i, _)| *i).collect();
    let mut out: Vec<OcrBlock> = groups
        .iter()
        .filter(|group| group.iter().any(|i| kept_set.contains(i)))
        .map(|group| build_block(group, blocks, profiles, &vlm_fulltexts))
        .collect();
    sort_reading_order(&mut out);
    out
}

/// Similarity between two ALREADY-normalized texts.
fn norm_similarity(a: &str, b: &str) -> f32 {
    let ca: Vec<char> = a.chars().collect();
    let cb: Vec<char> = b.chars().collect();
    let longest = ca.len().max(cb.len());
    if longest == 0 {
        return 1.0;
    }
    1.0 - (levenshtein(&ca, &cb) as f32 / longest as f32)
}

/// Merge one cluster into a consensus `OcrBlock`.
fn build_block(
    group: &[usize],
    blocks: &[EngineBlock],
    profiles: &[EngineProfile],
    vlm_fulltexts: &[&str],
) -> OcrBlock {
    let profile_of = |engine: &str| profiles.iter().find(|p| p.id == engine);
    let members: Vec<&EngineBlock> = group.iter().map(|&i| &blocks[i]).collect();

    // Medoid consensus: member maximizing the similarity sum to others
    // (research §4 step 5, first version — unweighted).
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (mi, m) in members.iter().enumerate() {
        let mut total = 0.0f32;
        for (oi, o) in members.iter().enumerate() {
            if mi == oi {
                continue;
            }
            total += normalized_levenshtein(&m.text, &o.text);
        }
        if total > best_score {
            best_score = total;
            best_idx = mi;
        }
    }
    let medoid = members[best_idx];
    let consensus = normalize(&medoid.text);

    // Eligible engines for this cluster's scripts (research §4 step 6 denominator).
    let eligible_weight: f32 = profiles
        .iter()
        .filter(|p| medoid.scripts.intersects(p.scripts))
        .map(|p| p.weight)
        .sum();

    let mut weighted_sum = 0.0f32;
    let mut k = 0usize;
    let mut has_detector = false;
    let mut has_vlm = false;
    let mut detector_conf: Option<f32> = None;
    let mut detectors = 0usize;
    for m in &members {
        let profile = profile_of(&m.engine);
        let weight = profile.map_or(1.0, |p| p.weight);
        let t = norm_similarity(&normalize(&m.text), &consensus);
        weighted_sum += weight * t;
        if t >= 0.9 {
            k += 1;
        }
        if let Some(p) = profile {
            if p.detector {
                has_detector = true;
                detectors += 1;
                if m.conf.is_some() {
                    detector_conf = m.conf;
                }
            } else {
                has_vlm = true;
            }
        }
    }
    let evidence_score = if eligible_weight > 0.0 {
        weighted_sum / eligible_weight
    } else {
        0.0
    };

    let mut block = OcrBlock {
        text: medoid.text.clone(),
        bbox: medoid.bbox,
        model: medoid.engine.clone(),
        evidence: Evidence::Low,
        possible_hallucination: false,
    };

    // Research §4 step 6: high requires strong agreement AND class mixing
    // (or k ≥ 3 outright).
    let high = (k >= 2 && evidence_score >= 0.70 && has_detector && has_vlm) || k >= 3;
    if high {
        block.evidence = Evidence::High;
        return block;
    }

    let medium_pair = k >= 2 && evidence_score >= 0.40;
    let medium_single = detectors == 1
        && detector_conf.is_some_and(|c| c >= 0.90)
        && vlm_fulltexts
            .iter()
            .any(|full| partial_ratio(&medoid.text, full) >= 0.8);
    if medium_pair || medium_single {
        block.evidence = Evidence::Medium;
        return block;
    }

    if has_vlm && !has_detector {
        // VLM-only cluster with no detector frame nearby.
        block.possible_hallucination = true;
    }
    block
}

/// Reading order (research §4 step 7): top→bottom, left→right; boxes with
/// height/width ratio > 2 are vertical text and order right→left. Boxless
/// blocks (VLM page text) go last.
fn sort_reading_order(blocks: &mut [OcrBlock]) {
    blocks.sort_by(|a, b| {
        let key = |blk: &OcrBlock| match blk.bbox {
            Some(r) => (r.y0, if r.is_vertical() { -r.x0 } else { r.x0 }),
            None => (f32::INFINITY, 0.0),
        };
        let (ay, ax) = key(a);
        let (by, bx) = key(b);
        ay.partial_cmp(&by)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { x0, y0, x1, y1 }
    }

    fn profiles() -> Vec<EngineProfile> {
        vec![
            EngineProfile::new(
                "paddle",
                1.0,
                ScriptSet {
                    han: true,
                    kana: true,
                    latin: true,
                },
                true,
            ),
            EngineProfile::new(
                "tesseract",
                0.6,
                ScriptSet {
                    han: true,
                    kana: true,
                    latin: true,
                },
                true,
            ),
            EngineProfile::new("smolvlm", 0.5, ScriptSet::LATIN, false),
        ]
    }

    // ---- normalization ----

    #[test]
    fn normalize_folds_fullwidth_zero_width_and_case() {
        assert_eq!(normalize("ＨＥＬＬＯ　ＷＯＲＬＤ"), "hello world");
        assert_eq!(normalize("café\u{200b}\u{FEFF}  latte"), "café latte");
        assert_eq!(normalize("  multiple   spaces  "), "multiple spaces");
        // Japanese text passes through NFKC unchanged in practice.
        assert_eq!(normalize("画像内の文字"), "画像内の文字");
    }

    #[test]
    fn detect_scripts_marks_han_kana_latin() {
        let s = detect_scripts("画像のテキスト OCR");
        assert!(s.han && s.kana && s.latin);
        assert_eq!(detect_scripts("hello"), ScriptSet::LATIN);
        assert_eq!(detect_scripts("12345"), ScriptSet::NONE);
    }

    // ---- similarity ----

    #[test]
    fn levenshtein_similarity_basics() {
        assert!((normalized_levenshtein("hello", "hello") - 1.0).abs() < 1e-6);
        assert!((normalized_levenshtein("hello", "helo") - 0.8).abs() < 1e-6);
        assert!(normalized_levenshtein("hello", "world") < 0.5);
    }

    #[test]
    fn partial_ratio_finds_substring() {
        // VLM merged lines: full text contains the shorter line verbatim.
        assert!((partial_ratio("内の文字", "これは画像内の文字です") - 1.0).abs() < 1e-6);
        assert!(partial_ratio("完全不", "画像内の文字") < 0.5);
    }

    #[test]
    fn fuzzy_iou_prefers_containment_over_iou() {
        let big = rect(0.0, 0.0, 1.0, 1.0);
        let inside = rect(0.1, 0.1, 0.9, 0.9);
        // Raw IoU of a box fully inside another is low; the
        // intersection/smaller-area term is 1.0, so the fuzzy score is too.
        assert!(fuzzy_iou(&big, &inside) > 0.9);
        // Two disjoint boxes score zero.
        let other = rect(0.0, 0.0, 0.3, 0.1);
        let aside = rect(0.7, 0.7, 1.0, 0.9);
        assert!(fuzzy_iou(&other, &aside).abs() < 1e-6);
    }

    // ---- hungarian ----

    #[test]
    fn hungarian_beats_greedy_matching() {
        // Greedy takes a1-b1 (10); optimal is a1-b2 + a2-b1 = 18.
        let score = vec![vec![10.0, 9.0], vec![9.0, 0.0]];
        let (total, matched) = hungarian(&score);
        assert!((total - 18.0).abs() < 1e-4, "total {total}");
        assert_eq!(matched[0], Some(1));
        assert_eq!(matched[1], Some(0));
    }

    #[test]
    fn hungarian_handles_rectangular_and_empty_inputs() {
        let score = vec![vec![5.0, 7.0, 0.0], vec![6.0, 0.0, 0.0]];
        let (total, matched) = hungarian(&score);
        assert!((total - 13.0).abs() < 1e-4, "total {total}");
        assert_eq!(matched[1], Some(0));
        let (total, _) = hungarian(&[]);
        assert_eq!(total, 0.0);
    }

    // ---- ensemble: agree / partial / disagree ----

    #[test]
    fn agree_case_yields_high_evidence() {
        // Two detectors + VLM all read the same box → class-mixing high.
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "OCR result",
                Some(rect(0.1, 0.1, 0.9, 0.2)),
                Some(0.97),
            ),
            EngineBlock::new(
                "tesseract",
                "OCR result",
                Some(rect(0.11, 0.10, 0.9, 0.21)),
                Some(0.90),
            ),
            EngineBlock::new("smolvlm", "ocr result", None, None),
        ];
        let out = ensemble(&blocks, &profiles());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].evidence, Evidence::High);
        assert!(!out[0].possible_hallucination);
        assert_eq!(out[0].text, "OCR result");
    }

    #[test]
    fn partial_agreement_yields_medium_evidence() {
        // Two detectors agree; VLM text unrelated (its block stays a
        // low/flagged leftover). k=2 with E < 0.70 → medium.
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "請求書",
                Some(rect(0.1, 0.1, 0.5, 0.2)),
                Some(0.95),
            ),
            EngineBlock::new(
                "tesseract",
                "請求書",
                Some(rect(0.1, 0.1, 0.5, 0.2)),
                Some(0.95),
            ),
            EngineBlock::new("smolvlm", "invoice", None, None),
        ];
        let out = ensemble(&blocks, &profiles());
        assert_eq!(out.len(), 2, "merged block + leftover VLM text");
        let merged = out
            .iter()
            .find(|b| b.text == "請求書")
            .unwrap_or_else(|| panic!("merged block missing"));
        assert_eq!(merged.evidence, Evidence::Medium);
    }

    #[test]
    fn disagree_case_vlm_only_is_low_with_hallucination_flag() {
        // Only the VLM produced something, no detector box nearby.
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "合計金額",
                Some(rect(0.1, 0.5, 0.6, 0.6)),
                Some(0.98),
            ),
            EngineBlock::new("smolvlm", "total amount due", None, None),
            EngineBlock::new(
                "tesseract",
                "margin note",
                Some(rect(0.8, 0.1, 0.9, 0.2)),
                Some(0.4),
            ),
        ];
        let out = ensemble(&blocks, &profiles());
        let flagged: Vec<&OcrBlock> = out.iter().filter(|b| b.possible_hallucination).collect();
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].evidence, Evidence::Low);
        assert_eq!(flagged[0].text, "total amount due");
    }

    #[test]
    fn engines_do_not_vote_on_unsupported_scripts() {
        // SmolVLM (Latin-only) hallucinating CJK is dropped entirely.
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "日本語のテキスト",
                Some(rect(0.1, 0.1, 0.5, 0.2)),
                Some(0.99),
            ),
            EngineBlock::new("smolvlm", "日本語らしき文字", None, None),
        ];
        let out = ensemble(&blocks, &profiles());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "日本語のテキスト");
        assert!(!out[0].possible_hallucination);
        assert_eq!(
            out[0].evidence,
            Evidence::Low,
            "single detector, no corroboration"
        );
    }

    #[test]
    fn single_high_conf_detector_corroborated_by_vlm_fulltext_is_medium() {
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "total",
                Some(rect(0.2, 0.2, 0.4, 0.3)),
                Some(0.95),
            ),
            EngineBlock::new("smolvlm", "invoice total amount due", None, None),
        ];
        let out = ensemble(&blocks, &profiles());
        let blk = out.iter().find(|b| b.text == "total").unwrap_or_else(|| {
            panic!(
                "expected the detector block, got {:?}",
                out.iter().map(|b| &b.text).collect::<Vec<_>>()
            )
        });
        assert_eq!(blk.evidence, Evidence::Medium);
    }

    #[test]
    fn reading_order_is_top_bottom_then_left_right_with_vertical_right_to_left() {
        let blocks = vec![
            EngineBlock::new(
                "paddle",
                "middle",
                Some(rect(0.5, 0.5, 0.7, 0.6)),
                Some(0.9),
            ),
            EngineBlock::new(
                "tesseract",
                "top-left",
                Some(rect(0.1, 0.1, 0.3, 0.2)),
                Some(0.9),
            ),
            // Vertical column to the right of top-left in the same band.
            EngineBlock::new(
                "paddle",
                "tate-right",
                Some(rect(0.35, 0.1, 0.38, 0.45)),
                Some(0.9),
            ),
        ];
        let out = ensemble(&blocks, &profiles());
        let texts: Vec<&str> = out.iter().map(|b| b.text.as_str()).collect();
        assert_eq!(texts, vec!["tate-right", "top-left", "middle"]);
    }
}
