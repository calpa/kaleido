#!/usr/bin/env bash
# Download model files listed in models/manifest.json and verify SHA-256.
#
#   scripts/fetch-models.sh [MODEL_ID ...]   # default: every model
#   KALEIDO_MODELS=/path scripts/fetch-models.sh
#
# Files land in $KALEIDO_MODELS/<model id>/ (default ./models). Requires
# curl, python3 and sha256sum or shasum. Binaries are git-ignored.
set -euo pipefail

root="${KALEIDO_MODELS:-$(cd "$(dirname "$0")/.." && pwd)/models}"
manifest="$root/manifest.json"
[[ -f "$manifest" ]] || { echo "missing $manifest" >&2; exit 1; }

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

python3 - "$manifest" "$@" <<'PY' |
import json, sys
manifest, wanted = sys.argv[1], set(sys.argv[2:])
for model in json.load(open(manifest))["models"]:
    if wanted and model["id"] not in wanted:
        continue
    for f in model["files"]:
        print(model["id"], f["name"], f["sha256"], f["url"])
PY
while read -r id name sum url; do
  dir="$root/$id"; dest="$dir/$name"
  mkdir -p "$dir"
  if [[ -f "$dest" && "$(sha256 "$dest")" == "$sum" ]]; then
    echo "ok       $id/$name"; continue
  fi
  echo "fetch    $id/$name"
  curl -fL --retry 3 -o "$dest.part" "$url"
  got="$(sha256 "$dest.part")"
  if [[ "$got" != "$sum" ]]; then
    rm -f "$dest.part"
    echo "checksum mismatch for $id/$name: got $got, want $sum" >&2; exit 1
  fi
  mv "$dest.part" "$dest"
  echo "verified $id/$name"
done
