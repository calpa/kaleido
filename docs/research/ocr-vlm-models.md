<!-- 信息來自 Devin agent（Devin，雲端），20261118。人審：yuuka（結構與來源密度抽驗通過，未逐條覆核 URL）。 -->
收到，我會查證各模型的 HuggingFace 頁面、授權、crate 文件與基準數據，整理成含來源的完整 markdown 報告回覆給你。# kaleido OCR 集成：本地小型 VLM/OCR 模型與 Rust 推論堆疊研究

資料抓取日期：2026-11-18。`[S#]` 對應文末的來源表。標 **〔估算〕** 的數字是我依來源數據推算的，不是官方實測值，在 M2 上必須實測確認。

## 先看結論
- **Qwen2.5-VL-3B 的授權不是 Apache。** 上游授權是 *Qwen Research License*，只限非商業使用 [S17][S18]。ggml-org 的 GGUF 轉檔 repo 雖然標 apache-2.0 [S11]，仍應以上游授權為準（這是我的判斷）。它只能做成「使用者自行同意授權後下載」的選配，不能當預設。
- **日文能力：** SmolVLM 三個版本的模型卡都寫只支援 English [S13][S14][S15]。Moondream2 也沒有宣稱多語 [S21]。因此日文只能靠 PP-OCRv5、Qwen2.5-VL 和 Tesseract `jpn`。
- **預設組合：** PP-OCRv5 mobile（ort）＋ Tesseract `jpn+eng`（tesseract-rs）＋ SmolVLM-500M Q8（llama-cpp-2，只參與拉丁字母區塊的投票）。三個模型依序載入、各自執行，峰值記憶體約 1–2 GB 〔估算〕。

---

## 1. llama-cpp-2 + mtmd（mmproj）路線

**crate 現況**
- `llama-cpp-2` 最新版 0.1.161（2026-11-03），授權 MIT OR Apache-2.0 [S4]。
- 提供 `metal` 和 `mtmd` 兩個 feature [S4][S5]。在 macOS aarch64 上，`Cargo.toml` 會依 target 自動啟用 `metal` [S5]。
- `src/mtmd.rs` 提供 `MtmdContext::init_from_file`、`MtmdBitmap::from_image_data`、`tokenize`、`encode_chunk` [S6]。`examples/mtmd` 是 mtmd-cli 的「部分移植」[S7]。

**llama.cpp 本身**
- macOS 預設啟用 Metal [S3]，授權 MIT [S54]。
- 多模態需要「文字模型 GGUF＋mmproj GGUF」兩個檔案 [S2]。
- 官方警告 mtmd「under very heavy development, breaking changes are expected」[S2]，所以 crate 版本要鎖死。
- 官方預量化清單有 SmolVLM 500M、2.2B、SmolVLM2 2.2B、Qwen2.5-VL-3B 和 Moondream2 20250414 [S1]。

**KV cache 計算方式 〔估算〕**
- 公式：每 token 佔 2 × 層數 × KV heads × head_dim × 2 bytes（f16）。
- 各模型參數取自 config [S23][S24][S25][S26]：
  - Qwen2.5-VL-3B：36 層、2 個 KV head、128 dim，每 token 36 KB，4k context 約 151 MB。
  - SmolVLM 2.2B：24 層、32 個 KV head、64 dim，每 token 192 KB，4k context 約 805 MB。
  - SmolVLM-500M：32 層、5 個 KV head、64 dim，每 token 40 KB，4k context 約 168 MB。
  - Moondream2：24 層、32 個 KV head、64 dim，max_context 2048，約 403 MB。

**模型比較**

| 模型 | GGUF 檔案（HF） | 檔案大小 | 峰值 RAM 〔估算〕 | 授權 | OCR 相關分數 | 日文 |
|---|---|---|---|---|---|---|
| SmolVLM-500M | ggml-org/SmolVLM-500M-Instruct-GGUF：Q8_0 ＋ mmproj Q8_0 [S8] | 437＋109 MB [S8] | 約 0.8–1.2 GB；transformers 版官方值 1.23 GB GPU RAM [S13] | Apache-2.0 [S13] | 模型卡沒有公布 OCRBench [S13] | 不支援，僅 English [S13] |
| SmolVLM 2.2B | ggml-org/SmolVLM-Instruct-GGUF：Q4_K_M ＋ mmproj Q8_0 [S9] | 1112＋593 MB [S9] | 約 2.5–3 GB；transformers 版最低 5.02 GB [S16] | Apache-2.0 [S14] | OCRBench 65.5、DocVQA 81.6、TextVQA 72.7 [S15][S16] | 不支援 [S14] |
| SmolVLM2 2.2B | ggml-org/SmolVLM2-2.2B-Instruct-GGUF [S10] | 1113＋593 MB [S10] | 同上 | Apache-2.0 [S10] | OCRBench 72.9、DocVQA 79.98 [S15] | 不支援 [S15] |
| Qwen2.5-VL-3B | ggml-org/Qwen2.5-VL-3B-Instruct-GGUF：Q4_K_M ＋ mmproj Q8_0 [S11] | 1930＋845 MB [S11] | 約 3.2–4 GB（視 max_pixels 而定） | **Qwen Research，非商業** [S18] | OCRBench 797/1000、CC-OCR 74.5、DocVQA 93.9 [S19] | 官方宣稱 multi-language OCR，但沒有日文單項分數 [S20] |
| Moondream2 (20250414) | ggml-org/moondream2-20250414-GGUF，**只有 f16** [S12] | 2840＋910 MB [S12] | 約 4.2–4.6 GB | Apache-2.0 [S21] | DocVQA 79.3、TextVQA 76.3 [S22] | 沒有宣稱支援 [S21] |

**使用注意**
- **Qwen 的圖片 token 數：** 預設 4–16384，可用 `min_pixels`/`max_pixels` 限制，官方範例是 256–1280 token [S17]。在 8 GB 機器上一定要設上限。
- **框座標：** Qwen 可以輸出 bbox JSON [S17]，是唯一能給框的 VLM。
- **Moondream 的 detect/point：** 這兩個功能在官方 Python API 裡 [S21]，llama.cpp 文件沒有提到支援，所以我視為不可用。
- **Moondream 的 OCR prompt：** 要用 "Transcribe the text" [S22]。

## 2. ort + PaddleOCR v4/v5 路線

**crate 現況**
- `ort` 最新版 2.0.0-rc.15（2026-10-19），封裝 ONNX Runtime 1.31，授權 MIT/Apache [S32]。
- 有 `coreml` feature [S32]。執行時應先用 `ep::CoreML::is_available()` 檢查，因為註冊仍可能失敗 [S33]。
- 現成的上層 crate：
  - `oar-ocr` 0.10.3（Apache-2.0）：底層就是 ort，支援 PP-OCRv4/v5/v6，有 `coreml` feature，也能從記憶體載入 ONNX [S29]。
  - `paddle-ocr-rs` 0.6.1：實作對齊 RapidOCR [S34]。

**確切模型檔**

| 用途 | 檔名（ONNX） | 大小 | 來源 |
|---|---|---|---|
| 偵測（mobile） | `pp-ocrv5_mobile_det.onnx` | 4.6 MiB | [S28] |
| 辨識（mobile，簡中、繁中、英、日、拼音同一個模型） | `pp-ocrv5_mobile_rec.onnx` ＋ `ppocrv5_dict.txt` | 15.8 MiB | [S28][S27] |
| 偵測、辨識（server） | `pp-ocrv5_server_det.onnx`、`pp-ocrv5_server_rec.onnx` | 84.0、80.6 MiB | [S28] |
| 英文專用辨識（備用） | `en_pp-ocrv5_mobile_rec.onnx` | 7.5 MiB | [S28] |
| 官方 Paddle 格式 | PaddlePaddle/PP-OCRv5_mobile_det 等 | — | Apache-2.0 [S31] |

另一個 ONNX 鏡像是 monkt/paddleocr-onnx [S30]。

**辨識準確度（PP-OCRv5 官方表）[S27]**

| 模型 | 日文辨識 | 印刷英文辨識 | 直書 | 日文偵測 |
|---|---|---|---|---|
| PP-OCRv5_server | 0.7372 | 0.8679 | 0.9314 | 0.772 |
| PP-OCRv5_mobile | 0.7577 | 0.8753 | 0.8089 | 0.727 |
| PP-OCRv4_mobile | 0.4724 | 0.6598 | 0.5924 | 0.634 |

- v4 不支援日文，所以 v4 不考慮。
- 官方 RAM 數據是 Paddle Inference 在 Xeon CPU 上測的，不是 ort：v5_mobile 峰值 2220 MB、v5_server 峰值 4021 MB [S27]。ONNX 在 M2 上的數字要自行測量。
- PP-OCRv6（tiny/small/medium）已經有 ONNX [S28]，可以列為後續評估項目。
- PaddleOCR 授權 Apache-2.0 [S35]。

## 3. tesseract-rs 成熟度

| crate | 版本與更新日期 | 特點 |
|---|---|---|
| `tesseract-rs` | 0.4.0（2026-07-31），MIT [S36] | 從原始碼建置 Tesseract＋Leptonica（需要 CMake），預設下載 eng＋tur，可用 `embed-tessdata` 嵌入語言資料，也可用 `use-system-tesseract` 改連 brew 安裝的版本 [S37] |
| `tesseract` | 0.15.2（2025-04） | [S38] |
| `leptess` | 0.14.0（2023-02，已停滯） | [S39] |

- **語言資料：** `jpn`／`jpn_vert` 都有 [S40]。檔案大小 best 14.3 MB、fast 2.5 MB [S41]。官方建議一般使用者用 fast [S40]。
- **判斷：** 綁定本身可用，但官方沒有日文 CER 數據，只能當「便宜、失敗模式不同」的第二票。Tesseract 授權 Apache-2.0 [S42]。
- **不考慮的選項：** `ocrs` 只支援拉丁字母 [S43]。macOS Vision 的 `RecognizeTextRequest`（macOS 15+）[S52] 可以透過 `objc2-vision` 呼叫 [S52]，可列為第四個零下載的選配引擎。

---

## 4. 集成設計

**輸入：** 每個引擎輸出 `Block { engine, kind, text, bbox: Option<Rect>, conf: Option<f32>, scripts }`，`bbox` 正規化到 [0,1]。
- PP-OCRv5 和 Tesseract 一定給框。
- Qwen 可以用 grounding prompt 給框 [S17]。
- SmolVLM、Moondream 只給文字，由程式拆成行。

**演算法**
1. **正規化（只用於比對，顯示仍保留原文）**
   - 做 NFKC，把全形轉半形 [S48]。
   - 去除零寬字元、合併空白、拉丁字母轉小寫。
   - 標記主要文字系統（Han、Kana、Latin）。
2. **資格篩選**
   - 引擎不支援某個文字系統時，不參與該區塊的投票，也不計入分母。例如 SmolVLM、Moondream 碰到 CJK 區塊 [S13][S21]。
3. **兩兩相似度**
   - 有框時用模糊 IoU：先把兩個框各外擴 0.25 倍行高，取 `max(IoU, 交集/較小面積)`。
   - 文字相似度 `T = normalized_levenshtein`（逐字元比對，日文同樣適用）[S45]。VLM 常把多行合併，所以 T 取 partial 版本，也就是對較長字串做滑動子字串的最大值。
   - 兩邊都有框時 `S = 0.4·IoU' + 0.6·T`，門檻 τ = 0.55。缺框時 `S = T`，門檻提高到 τ = 0.75。
   - 先用 R-tree（`rstar` [S47]）找候選，避免 O(n²)。
4. **配對**
   - 每對引擎之間跑匈牙利演算法，求 S 最大化（`pathfinding::kuhn_munkres` [S46]），丟掉低於 τ 的配對。
   - 用 union-find 把配對合成跨引擎的群集，每個群集裡每個引擎最多一個區塊。
5. **共識文字**
   - 第一版取 medoid：群集中與其他成員相似度總和最高的那一個。
   - 進階版改成 ROVER 式的字元對齊加權投票 [S44]。
   - 引擎權重 w：PP-OCRv5 為 1.0、Qwen 為 1.0、Tesseract 為 0.6、SmolVLM／Moondream 為 0.5（初始值，需校準）。
6. **證據分數**
   - `E = Σ_{e∈群集} w_e·T(e, 共識) / Σ_{e∈有資格的引擎} w_e`
   - `k` 是與共識的 T ≥ 0.9 的引擎數。
   - **high**：k ≥ 2、E ≥ 0.70，且群集中同時有「偵測器類」（Paddle 或 Tesseract）和「VLM 類」，或 k ≥ 3。
   - **medium**：k ≥ 2 且 E ≥ 0.40；或者只有單一偵測器，但 conf ≥ 0.90，且共識文字能在某個 VLM 全文裡以 partial T ≥ 0.8 找到。
   - **low**：其他情況。
   - 只有 VLM、附近沒有任何偵測框的區塊，標成 `low` 並加上 `possible_hallucination` 旗標。
7. **輸出順序：** 由上到下、由左到右。框的高寬比 > 2 視為直書，改成由右到左的欄序。
8. **校準：** 準備約 200 張日英混合截圖做標註，計算每個證據等級的 CER 和精確率，再回頭調整 w、τ 和門檻。上面的數字都只是起始值，是我的提案，沒有外部來源。

## 5. 8 GB M2 預設組合

| 層級 | 引擎 | 種類 | 峰值 RAM 〔估算〕 |
|---|---|---|---|
| 預設 | PP-OCRv5 mobile det＋rec（ort，CPU；CoreML 可用時改用） | 偵測＋CTC | < 0.5 GB（檔案 20 MiB [S28]） |
| 預設 | Tesseract fast `jpn+eng`（tesseract-rs） | LSTM | < 0.3 GB（檔案 2.5＋4.1 MB [S41]） |
| 預設 | SmolVLM-500M Q8（llama-cpp-2，Metal） | VLM，只投拉丁字母區塊 | 約 1 GB |
| 選配：日文品質 | Qwen2.5-VL-3B Q4_K_M，`max_pixels = 1280·28·28` [S17] | VLM＋bbox | 約 3.5 GB，GUI 需先取得授權同意 [S18] |
| 不建議 | Moondream2（只有 f16 [S12]）、SmolVLM 2.2B（KV cache 偏大） | — | 4 GB 以上 |

- 依序載入、用完就卸載，峰值記憶體等於單一模型的最大值，不是加總。
- PP-OCRv5 可以常駐記憶體。

## 6. 機器就緒判定規則（`kaleido bench`）

**量測項目**
- 實體記憶體：`ProcessInfo.physicalMemory` [S50]
- GPU 可用量：`MTLDevice.recommendedMaxWorkingSetSize` [S49]
- 是否為統一記憶體：`hasUnifiedMemory` [S51]
- 目前可用記憶體與剩餘磁碟空間

**需求估算：** `need = 權重檔 + KV(ctx) + 25% 計算緩衝`。

**判定（門檻是我的提案）**
- **READY**：need ≤ 0.6 × 實體記憶體，且 ≤ GPU 可用量，且磁碟剩餘 ≥ 2 × 下載量，且試跑一張圖的峰值 phys_footprint 低於 need × 1.2。
- **READY_WITH_LIMITS**：need ≤ 0.8 × 實體記憶體。系統自動降低 ctx 和 max_pixels，或改用 CPU 跑 mmproj（`--no-mmproj-offload` 的等價設定 [S1]）。
- **NOT_RECOMMENDED**：超過以上門檻，或試跑出現 swap。
- **UNSUPPORTED**：不是 arm64、沒有 Metal（仍可只跑 ort 和 Tesseract 的 CPU 路線）。

**8 GB M2 的預期結果：** 預設三引擎為 READY；Qwen2.5-VL-3B 落在 READY 或 READY_WITH_LIMITS 之間；Moondream2 f16 為 NOT_RECOMMENDED 〔估算〕。

---

## 來源
| ID | URL |
|---|---|
| S1 | https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md |
| S2 | https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md |
| S3 | https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#metal-build |
| S4 | https://crates.io/crates/llama-cpp-2 |
| S5 | https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-2/Cargo.toml |
| S6 | https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-2/src/mtmd.rs |
| S7 | https://github.com/utilityai/llama-cpp-rs/blob/main/examples/mtmd/README.md |
| S8 | https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/tree/main |
| S9 | https://huggingface.co/ggml-org/SmolVLM-Instruct-GGUF/tree/main |
| S10 | https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/tree/main |
| S11 | https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/tree/main |
| S12 | https://huggingface.co/ggml-org/moondream2-20250414-GGUF/tree/main |
| S13 | https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct |
| S14 | https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct |
| S15 | https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct |
| S16 | https://huggingface.co/blog/smolvlm |
| S17 | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |
| S18 | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE |
| S19 | https://arxiv.org/abs/2502.13923 (Table 5) |
| S20 | https://qwenlm.github.io/blog/qwen2.5-vl/ |
| S21 | https://huggingface.co/vikhyatk/moondream2 |
| S22 | https://moondream.ai/blog/moondream-2025-04-14-release |
| S23 | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json |
| S24 | https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/blob/main/config.json |
| S25 | https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/blob/main/config.json |
| S26 | https://huggingface.co/vikhyatk/moondream2/blob/main/config.py |
| S27 | https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md |
| S28 | https://github.com/GreatV/oar-ocr/blob/main/docs/models.md |
| S29 | https://crates.io/crates/oar-ocr |
| S30 | https://huggingface.co/monkt/paddleocr-onnx |
| S31 | https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det |
| S32 | https://crates.io/crates/ort |
| S33 | https://ort.pyke.io/perf/execution-providers |
| S34 | https://github.com/mg-chao/paddle-ocr-rs |
| S35 | https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE |
| S36 | https://crates.io/crates/tesseract-rs |
| S37 | https://github.com/cafercangundogdu/tesseract-rs |
| S38 | https://crates.io/crates/tesseract |
| S39 | https://crates.io/crates/leptess |
| S40 | https://github.com/tesseract-ocr/tessdoc/blob/main/Data-Files.md |
| S41 | https://github.com/tesseract-ocr/tessdata_fast/blob/main/jpn.traineddata · https://github.com/tesseract-ocr/tessdata_best/blob/main/jpn.traineddata |
| S42 | https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE |
| S43 | https://github.com/robertknight/ocrs |
| S44 | https://doi.org/10.1109/ASRU.1997.659110 (ROVER, Fiscus 1997) |
| S45 | https://docs.rs/strsim/latest/strsim/fn.normalized_levenshtein.html |
| S46 | https://docs.rs/pathfinding/latest/pathfinding/kuhn_munkres/fn.kuhn_munkres.html |
| S47 | https://crates.io/crates/rstar |
| S48 | https://unicode.org/reports/tr15/ · https://crates.io/crates/unicode-normalization |
| S49 | https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize |
| S50 | https://developer.apple.com/documentation/foundation/processinfo/physicalmemory |
| S51 | https://developer.apple.com/documentation/metal/mtldevice/hasunifiedmemory |
| S52 | https://developer.apple.com/documentation/vision/recognizetextrequest · https://crates.io/crates/objc2-vision |
| S53 | https://github.com/calpa/kaleido |
| S54 | https://github.com/ggml-org/llama.cpp/blob/master/LICENSE |
