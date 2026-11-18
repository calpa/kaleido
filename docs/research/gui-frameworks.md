<!-- 信息來自 Devin agent（Devin，雲端），20261118。人審：yuuka（結構與來源密度抽驗通過，未逐條覆核 URL）。 -->
收到，我會查閱 egui/eframe、iced、Slint、Tauri v2 的官方文件、GitHub 與 crates.io 資料，整理成附來源的比較文件直接回覆。# kaleido 原生 GUI 框架選型：egui/eframe vs iced vs Slint vs Tauri v2

資料擷取日期：2026-11-18。GitHub 數據來自 REST API（`/repos/{owner}/{repo}`）。下載量來自 crates.io API，「近期」指 crates.io 的近 90 天統計。

## 0. kaleido 現況（影響選型的限制）
- 專案是 workspace，目前只有 `kaleido-core` 與 `kaleido-cli`，`rust-version = "1.85"`，授權 MIT：https://github.com/calpa/kaleido/blob/main/Cargo.toml
- README 規劃的功能有「Fast thumbnail grid with smooth zoom in/out」、「dark & light themes」與 Metal GPU bench：https://github.com/calpa/kaleido/blob/main/README.md
- 加入 GUI 後一定要提高 MSRV：egui 0.37 需要 1.95（https://github.com/emilk/egui/blob/0.37.0/Cargo.toml ），Slint 1.19 需要 1.95（https://github.com/slint-ui/slint/blob/v1.19.0/Cargo.toml ），iced 0.14 需要 1.88（https://github.com/iced-rs/iced/blob/0.14.0/Cargo.toml ），Tauri 2.12.7 需要 1.90（https://github.com/tauri-apps/tauri/blob/tauri-v2.12.7/Cargo.toml ）。

## 1. 比較表

| 準則 | egui / eframe 0.37 | iced 0.14 | Slint 1.19 | Tauri v2 (2.12) |
|---|---|---|---|---|
| (1) 10k+ 縮圖虛擬化 | `ScrollArea::show_rows` 只繪製可見列，官方範例就是 10,000 列[^e1]。網格可用「每列 N 張」實作 | 0.14.0 發行版沒有虛擬化 list（tag 中沒有 `widget/src/list.rs`）[^i1]。`Grid` 是 responsive grid[^i2]，`lazy` 只避免重建[^i3]。要自行用 `Scrollable::on_scroll` 的 viewport 做視窗化[^i4] | `ListView` 只實例化可見元素，「practically unlimited number of items」[^s1]。網格同樣用「每列 N 張」實作 | 前端用 JS 虛擬化函式庫（如 TanStack Virtual[^t1]）。圖片經 asset protocol 從本機讀取，預設關閉且需設定 scope[^t2] |
| (2) GPU 縮放/平移大圖 | eframe 預設 renderer 是 `wgpu`[^e2]（macOS 上是 Metal）。`Scene` 容器支援 pan/zoom[^e3]。egui-wgpu 預設要求 `max_texture_dimension_2d: 8192`[^e4]，大圖要切 tile | `iced_wgpu` 支援 Metal[^i5]。內建 `image::Viewer`，提供縮放、平移與 `filter_method`[^i6][^i7] | Skia renderer 支援 Metal，`femtovg-wgpu` 也支援 Metal[^s2]，但預設 renderer 是 femtovg（OpenGL）[^s3]。沒有專用 viewer，要自己組合 `Image` 元素[^s4] | 在 macOS 上用 WKWebView 繪製[^t3]。Rust 端不直接控制 Metal 管線（推論） |
| (3) 深色/淺色主題 | 可分別設定深/淺 `Style` 並跟隨系統[^e5]，有 `ThemePreference`[^e6]。0.36 起視窗標題列會同步主題[^e7] | 內建多套 `Theme`[^i8]，0.14 新增 system theme reactions[^i9] | `Palette.color-scheme` 可讀寫[^s5][^s6] | 有 `setTheme()` window API[^t4]，其餘用 CSS |
| (4) CJK 文字 | 0.37 起 native eframe 透過 `egui_system_font_provider`（fontique）載入系統字型[^e8]。仍有未解問題：CJK fallback 字形偏離基線（#8802 open）[^e9]、全形標點缺失（PR #8849 open）[^e10]、日文禁則斷行（#8826 open）[^e11] | 用 cosmic-text 做 shaping 與 font fallback[^i10]，已支援 IME[^i11]。open 的 CJK 問題集中在 WASM[^i12] | 自動使用系統字型[^s7]。1.19 修正 macOS 預設字型[^s8]，也修正 CJK 無斜體時的 `font-italic`[^s8] | 用 WebKit 的系統文字排版（推論，依據 WKWebView[^t3]） |
| (5) 維護與社群 | 32,363★，1,174 open issues+PRs[^g1]。2026 年有 0.34、0.35、0.36、0.37 共 4 個大版[^e12]。egui 總下載 32.1M，近期 10.2M[^c1] | 32,070★[^g2]，master 在 2026-11-16 仍有 push。但最新發行版是 2025-12-07 的 0.14.0，近一年沒有新版[^i13]。README 自稱「experimental software」[^i5]。總下載 3.4M，近期 0.84M[^c2] | 24,636★[^g3]。1.19.0 於 2026-11-16 發行[^s8]。總下載 2.6M，近期 1.13M[^c3] | 112,997★[^g4]。tauri-cli 2.12.6 於 2026-11-18 發行。總下載 45.8M，近期 19.1M[^c4] |
| (6) 授權 | MIT OR Apache-2.0[^e13] | MIT[^g2] | 三選一：GPLv3／Royalty-free（桌面免費，但須揭露使用 Slint）／商業授權[^s9][^s10] | MIT 與 Apache-2.0[^t5] |
| (7) .app + 公證 | 框架不負責打包。用 cargo-bundle[^p1] 或 cargo-packager（支援 .app/.dmg）[^p2]，再用 `notarytool` 送 Apple 公證[^p3] | 同 egui，需要外部工具[^p1][^p2][^p3] | 官方有 macOS 打包指南：產生 Xcode 專案、Developer ID 簽章、`notarytool` 公證[^s11] | 內建 bundler，可產生 .app[^t6] 與 .dmg[^t7]。官方文件說明簽章與公證（App Store Connect API 或 Apple ID）[^t8] |

## 2. 建議：egui/eframe

**適用情境**：8GB M2 Mac 上的純 Rust、本機優先圖庫。

**理由**
- **對準則 (1)(2) 的支援最直接**：`show_rows` 的虛擬化是一等公民[^e1]，預設 renderer 是 wgpu/Metal[^e2]。`TextureHandle` 可以直接上傳縮圖紋理[^e14]，`Scene` 提供縮放/平移[^e3]。
- **記憶體可控**（以下是推論與計算）：沒有 WebView 程序，也沒有 DOM。一張 256×256 RGBA 縮圖約 256 KiB，10k 張全放在 GPU 約 2.5 GiB，在 8GB 機器上不可行。所以要用「可見範圍 + 預取」的 LRU，例如常駐 500 張約 125 MiB。egui 的 immediate-mode 模型讓「每幀只畫 row_range」的寫法很自然。
- **大圖策略**：egui-wgpu 預設紋理上限是 8192 px[^e4]。超過的照片要做 tile 金字塔，或在 `kaleido-core` 預先產生縮放層級。這也讓 8GB 機器不必一次載入全解析度紋理（推論）。
- **授權與 kaleido 的 MIT 完全相容**[^e13]。
- **維護節奏最穩**：2026 年發了 4 個大版[^e12]，在 Rust 原生 GUI 中下載量最高[^c1]。

**風險與對策**
- **CJK 品質仍在修**：#8802、#8849、#8826 都還沒解決[^e9][^e10][^e11]。對策：內嵌固定 CJK 字型（如 Noto Sans CJK）並設為 fallback，並在 CI 加入中日文截圖測試。若 CJK 呈現是硬性要求，這是 egui 最大的扣分項。
- **MSRV 要從 1.85 升到 1.95**[^e15]。
- **打包要自己組裝**：cargo-packager 產生 .app/.dmg[^p2]，再用 `codesign` + `notarytool` 公證[^p3]。

## 3. 其他選項何時勝出
- **Slint**：想要宣告式 UI、官方完整的 macOS 公證流程[^s11]，CJK 也較成熟[^s7][^s8]。但要接受三選一授權：用 Royalty-free 必須在 App 內揭露使用 Slint[^s9]。而且預設 renderer 是 OpenGL，要手動切到 Skia/Metal[^s2][^s3]。
- **Tauri v2**：團隊偏好 Web 技術，或想要最成熟的打包/公證工具鏈[^t6][^t8]與最大社群[^g4]。代價是 WebView 的記憶體開銷，以及 Rust 端無法直接控制 Metal 繪製（推論）。
- **iced**：`image::Viewer` 內建縮放/平移[^i6]，Elm 架構清楚。但發行版近一年沒更新[^i13]，0.14 沒有虛擬化 list[^i1]，官方也自稱 experimental[^i5]，目前不建議作為主框架。

## 來源
[^g1]: https://github.com/emilk/egui
[^g2]: https://github.com/iced-rs/iced
[^g3]: https://github.com/slint-ui/slint
[^g4]: https://github.com/tauri-apps/tauri
[^c1]: https://crates.io/crates/egui
[^c2]: https://crates.io/crates/iced
[^c3]: https://crates.io/crates/slint
[^c4]: https://crates.io/crates/tauri
[^e1]: https://docs.rs/egui/0.37.0/egui/containers/scroll_area/struct.ScrollArea.html （`show_rows`）
[^e2]: https://github.com/emilk/egui/blob/0.37.0/crates/eframe/Cargo.toml （default 含 `wgpu`）
[^e3]: https://docs.rs/egui/0.37.0/egui/containers/scene/struct.Scene.html
[^e4]: https://github.com/emilk/egui/blob/0.37.0/crates/egui-wgpu/src/setup.rs#L261-L273
[^e5]: https://github.com/emilk/egui/pull/4744
[^e6]: https://docs.rs/egui/0.37.0/egui/enum.ThemePreference.html
[^e7]: https://github.com/emilk/egui/pull/8299
[^e8]: https://github.com/emilk/egui/pull/8512
[^e9]: https://github.com/emilk/egui/issues/8802
[^e10]: https://github.com/emilk/egui/pull/8849
[^e11]: https://github.com/emilk/egui/issues/8826
[^e12]: https://github.com/emilk/egui/blob/0.37.0/CHANGELOG.md
[^e13]: https://github.com/emilk/egui/blob/0.37.0/README.md
[^e14]: https://docs.rs/egui/0.37.0/egui/struct.Context.html （`load_texture`）
[^e15]: https://github.com/emilk/egui/blob/0.37.0/Cargo.toml
[^i1]: https://github.com/iced-rs/iced/tree/0.14.0/widget/src ；上游 `List` 見 https://github.com/iced-rs/iced/issues/3429
[^i2]: https://docs.rs/iced/0.14.0/iced/widget/grid/index.html
[^i3]: https://docs.rs/iced/0.14.0/iced/widget/fn.lazy.html
[^i4]: https://docs.rs/iced/0.14.0/iced/widget/scrollable/struct.Scrollable.html
[^i5]: https://github.com/iced-rs/iced/blob/master/README.md
[^i6]: https://docs.rs/iced/0.14.0/iced/widget/image/viewer/struct.Viewer.html
[^i7]: https://github.com/iced-rs/iced/pull/2324
[^i8]: https://docs.rs/iced/0.14.0/iced/enum.Theme.html
[^i9]: https://github.com/iced-rs/iced/pull/3051
[^i10]: https://github.com/iced-rs/iced/pull/1697
[^i11]: https://github.com/iced-rs/iced/pull/2777
[^i12]: https://github.com/iced-rs/iced/issues/2843
[^i13]: https://github.com/iced-rs/iced/releases/tag/0.14.0
[^s1]: https://docs.slint.dev/latest/docs/slint/reference/std-widgets/views/listview/
[^s2]: https://docs.slint.dev/latest/docs/slint/guide/backends-and-renderers/backend_winit/
[^s3]: https://github.com/slint-ui/slint/blob/master/api/rs/slint/Cargo.toml
[^s4]: https://docs.slint.dev/latest/docs/slint/reference/elements/image/
[^s5]: https://docs.slint.dev/latest/docs/slint/reference/std-widgets/globals/palette/
[^s6]: https://github.com/slint-ui/slint/blob/master/CHANGELOG.md （「Added `color-scheme`」）
[^s7]: https://docs.slint.dev/latest/docs/slint/guide/development/fonts/
[^s8]: https://github.com/slint-ui/slint/blob/master/CHANGELOG.md （1.19.0）
[^s9]: https://github.com/slint-ui/slint/blob/master/LICENSE.md
[^s10]: https://slint.dev/pricing
[^s11]: https://docs.slint.dev/latest/docs/slint/guide/platforms/desktop/packaging/macos/
[^t1]: https://tanstack.com/virtual/latest
[^t2]: https://v2.tauri.app/reference/config/#assetprotocolconfig
[^t3]: https://v2.tauri.app/reference/webview-versions/
[^t4]: https://v2.tauri.app/reference/javascript/api/namespacewindow/
[^t5]: https://github.com/tauri-apps/tauri/blob/dev/LICENSE.spdx
[^t6]: https://v2.tauri.app/distribute/macos-application-bundle/
[^t7]: https://v2.tauri.app/distribute/dmg/
[^t8]: https://v2.tauri.app/distribute/sign/macos/
[^p1]: https://github.com/burtonageo/cargo-bundle
[^p2]: https://github.com/crabnebula-dev/cargo-packager
[^p3]: https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution

標記「推論」的項目沒有官方文件直接佐證。表中所有效能描述都來自文件，我沒有在 M2 上實測。
