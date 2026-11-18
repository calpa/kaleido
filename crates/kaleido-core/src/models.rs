//! OCR model metadata and machine-readiness rules.
//!
//! Model requirement table and verdict thresholds come from
//! `docs/research/ocr-vlm-models.md` §5 (8 GB M2 default lineup) and §6
//! (bench rules). `readiness` is the pure, static part of the verdict that
//! M4's `kaleido bench` consumes; the live trial-run check (peak
//! phys_footprint, swap) is layered on top there.

use serde::Serialize;

/// How a model runs (drives Metal requirements and future loading strategy).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelKind {
    /// ONNX detection + CTC recognition (PaddleOCR via ort).
    Detector,
    /// Tesseract LSTM (tesseract-rs, CPU).
    Lstm,
    /// Small vision-language model (llama-cpp-2 with Metal).
    Vlm,
}

/// Machine profile as measured by `kaleido bench` (research §6 measurement
/// items; M4 fills these from real probes, tests construct them directly).
#[derive(Debug, Clone, Serialize)]
pub struct MachineProfile {
    pub arm64: bool,
    pub metal: bool,
    /// `MTLDevice.recommendedMaxWorkingSetSize`, in GB.
    pub gpu_limit_gb: f64,
    /// Physical memory, in GB.
    pub ram_gb: f64,
    pub unified_memory: bool,
    /// Free disk space, in GB.
    pub disk_free_gb: f64,
}

/// Readiness verdict (research §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Fits comfortably; full context, no degraded settings.
    Ready,
    /// Usable but the system must shrink ctx/max_pixels or run mmproj on CPU.
    ReadyWithLimits,
    /// Over the thresholds, or trial-run showed swap (M4 layers this in).
    NotRecommended,
    /// Platform cannot run this engine class (no Metal / not arm64 for VLMs).
    Unsupported,
}

/// Static requirement metadata for one OCR/vision model.
#[derive(Debug, Clone, Serialize)]
pub struct ModelSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub kind: ModelKind,
    /// Peak memory need in GB: weights + KV(context) + 25% compute buffer
    /// (research §6 "需求估算"). This is the `need` in the verdict rules.
    pub ram_need_gb: f64,
    /// Download size in GB; the disk check requires 2× this free.
    pub download_gb: f64,
    /// Requires Metal GPU (llama-cpp-2 mmproj offload).
    pub requires_metal: bool,
    /// Requires explicit user license consent in the GUI (Qwen2.5-VL).
    pub license_consent_required: bool,
    /// Part of the default 8 GB lineup (research §5).
    pub default_enabled: bool,
}

impl ModelSpec {
    /// Static readiness verdict for a machine (research §6 thresholds).
    ///
    /// - `Unsupported`: a Metal model on a machine without Metal/arm64
    ///   (CPU-class engines still run there — research §6 parenthetical).
    /// - `Ready`: need ≤ 0.6 × RAM, Metal models fit the GPU working-set
    ///   limit, and ≥ 2× download size is free on disk.
    /// - `ReadyWithLimits`: need ≤ 0.8 × RAM (system degrades ctx/max_pixels).
    /// - `NotRecommended`: everything else. The live trial-run checks
    ///   (peak footprint, swap) are added by M4 on top of this verdict.
    pub fn readiness(&self, machine: &MachineProfile) -> Verdict {
        if self.requires_metal && (!machine.arm64 || !machine.metal) {
            return Verdict::Unsupported;
        }
        let need = self.ram_need_gb;
        let fits_ram = need <= 0.6 * machine.ram_gb;
        let fits_gpu = !self.requires_metal || need <= machine.gpu_limit_gb;
        let fits_disk = machine.disk_free_gb >= 2.0 * self.download_gb;
        if fits_ram && fits_gpu && fits_disk {
            Verdict::Ready
        } else if need <= 0.8 * machine.ram_gb {
            Verdict::ReadyWithLimits
        } else {
            Verdict::NotRecommended
        }
    }
}

/// The catalog from research §5, with need values derived as
/// weights + KV + 25% buffer where the research gives estimates.
pub const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec {
        id: "paddle-ocrv5-mobile",
        label: "PP-OCRv5 mobile det+rec (ONNX)",
        kind: ModelKind::Detector,
        ram_need_gb: 0.5,
        download_gb: 0.02, // 20 MiB files
        requires_metal: false,
        license_consent_required: false,
        default_enabled: true,
    },
    ModelSpec {
        id: "tesseract-fast-jpn-eng",
        label: "Tesseract fast jpn+eng",
        kind: ModelKind::Lstm,
        ram_need_gb: 0.3,
        download_gb: 0.0066, // 2.5 + 4.1 MB traineddata
        requires_metal: false,
        license_consent_required: false,
        default_enabled: true,
    },
    ModelSpec {
        id: "smolvlm-500m-q8",
        label: "SmolVLM-500M Q8 (Metal)",
        kind: ModelKind::Vlm,
        ram_need_gb: 1.0,
        download_gb: 0.5,
        requires_metal: true,
        license_consent_required: false,
        default_enabled: true,
    },
    ModelSpec {
        id: "qwen2.5-vl-3b-q4km",
        label: "Qwen2.5-VL-3B Q4_K_M (Metal)",
        kind: ModelKind::Vlm,
        ram_need_gb: 3.5,
        download_gb: 2.0,
        requires_metal: true,
        license_consent_required: true,
        default_enabled: false,
    },
    ModelSpec {
        id: "moondream2-f16",
        label: "Moondream2 f16 (Metal)",
        kind: ModelKind::Vlm,
        // f16 weights + KV cache + buffer: listed "not recommended" in
        // research §5; the 7 GB need puts it past READY_WITH_LIMITS on 8 GB.
        ram_need_gb: 7.0,
        download_gb: 3.6,
        requires_metal: true,
        license_consent_required: false,
        default_enabled: false,
    },
];

/// Verdicts for the whole catalog on one machine, in `MODEL_SPECS` order.
pub fn readiness_all(machine: &MachineProfile) -> Vec<(&'static str, Verdict)> {
    MODEL_SPECS
        .iter()
        .map(|spec| (spec.id, spec.readiness(machine)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apple M2 8 GB unified memory: recommended working set ≈ 2/3 of RAM.
    fn m2_8gb() -> MachineProfile {
        MachineProfile {
            arm64: true,
            metal: true,
            gpu_limit_gb: 5.33,
            ram_gb: 8.0,
            unified_memory: true,
            disk_free_gb: 60.0,
        }
    }

    fn spec(id: &str) -> &ModelSpec {
        MODEL_SPECS
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("spec {id} missing from catalog"))
    }

    #[test]
    fn default_lineup_is_ready_on_8gb_m2() {
        // Research §6: "預設三引擎為 READY".
        assert_eq!(
            spec("paddle-ocrv5-mobile").readiness(&m2_8gb()),
            Verdict::Ready
        );
        assert_eq!(
            spec("tesseract-fast-jpn-eng").readiness(&m2_8gb()),
            Verdict::Ready
        );
        assert_eq!(spec("smolvlm-500m-q8").readiness(&m2_8gb()), Verdict::Ready);
    }

    #[test]
    fn qwen_lands_ready_or_better_on_8gb_m2() {
        // Research §6: Qwen is READY or READY_WITH_LIMITS here.
        let verdict = spec("qwen2.5-vl-3b-q4km").readiness(&m2_8gb());
        assert!(
            verdict == Verdict::Ready || verdict == Verdict::ReadyWithLimits,
            "got {verdict:?}"
        );
    }

    #[test]
    fn moondream_f16_is_not_recommended_on_8gb_m2() {
        assert_eq!(
            spec("moondream2-f16").readiness(&m2_8gb()),
            Verdict::NotRecommended
        );
    }

    #[test]
    fn metal_models_are_unsupported_without_metal_or_arm64() {
        let no_metal = MachineProfile {
            metal: false,
            gpu_limit_gb: 0.0,
            ..m2_8gb()
        };
        assert_eq!(
            spec("smolvlm-500m-q8").readiness(&no_metal),
            Verdict::Unsupported
        );
        // CPU-class engines still run: research §6 parenthetical.
        assert_eq!(
            spec("paddle-ocrv5-mobile").readiness(&no_metal),
            Verdict::Ready
        );

        let intel = MachineProfile {
            arm64: false,
            ..m2_8gb()
        };
        assert_eq!(
            spec("smolvlm-500m-q8").readiness(&intel),
            Verdict::Unsupported
        );
        assert_eq!(
            spec("paddle-ocrv5-mobile").readiness(&intel),
            Verdict::Ready
        );
    }

    #[test]
    fn disk_pressure_downgrades_but_ram_rules_dominate() {
        // 2× qwen's 2.0 GB download no longer free → falls below Ready;
        // need 3.5 ≤ 0.8×8 keeps it usable with limits.
        let tight_disk = MachineProfile {
            disk_free_gb: 1.0,
            ..m2_8gb()
        };
        assert_eq!(
            spec("qwen2.5-vl-3b-q4km").readiness(&tight_disk),
            Verdict::ReadyWithLimits
        );
        // Tiny disks don't bother the tiny models: 2× 0.02 GB still fits.
        assert_eq!(
            spec("paddle-ocrv5-mobile").readiness(&tight_disk),
            Verdict::Ready
        );
        // RAM band boundaries on a 1.4 GB machine: 0.6× = 0.84, 0.8× = 1.12.
        let tiny_ram = MachineProfile {
            ram_gb: 1.4,
            gpu_limit_gb: 1.0,
            ..m2_8gb()
        };
        assert_eq!(
            spec("smolvlm-500m-q8").readiness(&tiny_ram),
            Verdict::ReadyWithLimits,
            "1.0 > 0.84 but ≤ 1.12"
        );
        assert_eq!(
            spec("qwen2.5-vl-3b-q4km").readiness(&tiny_ram),
            Verdict::NotRecommended,
            "3.5 > 1.12"
        );
    }

    #[test]
    fn catalog_covers_expected_ids() {
        let ids: Vec<&str> = MODEL_SPECS.iter().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![
                "paddle-ocrv5-mobile",
                "tesseract-fast-jpn-eng",
                "smolvlm-500m-q8",
                "qwen2.5-vl-3b-q4km",
                "moondream2-f16",
            ]
        );
    }
}
