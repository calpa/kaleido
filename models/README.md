# models/

Local model store. Only `manifest.json` (and this file) are committed; model
binaries are git-ignored and fetched on demand.

```sh
scripts/fetch-models.sh                      # every model in the manifest
scripts/fetch-models.sh paddle-ocrv5-mobile  # one model
```

Each file is downloaded to `models/<model id>/<file name>` and its SHA-256 is
checked against the manifest; a mismatch aborts and deletes the partial file.
`KALEIDO_MODELS=/other/dir` relocates the store (copy `manifest.json` there).
At load time `ModelStore::verify` re-checks size and SHA-256, so a truncated
or tampered file is reported instead of being fed to ONNX Runtime.

Manual alternative: download each `url` from `manifest.json` into
`models/<id>/` and compare with `sha256sum` / `shasum -a 256`.

| id | files | source |
|---|---|---|
| `paddle-ocrv5-mobile` | `pp-ocrv5_mobile_det.onnx`, `pp-ocrv5_mobile_rec.onnx`, `ppocrv5_dict.txt` (≈21 MB) | PaddleOCR PP-OCRv5 mobile, ONNX export from [oar-ocr v0.3.0](https://github.com/GreatV/oar-ocr/releases/tag/v0.3.0) (Apache-2.0) |
