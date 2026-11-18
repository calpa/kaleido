# Contributing to kaleido

Thanks for helping! Quick rules:

1. **Rust-first**: core analysis lives in `crates/kaleido-core` as a pure
   library (no I/O beyond reading files). GUI concerns stay out of it.
2. `cargo fmt --all --check` and `cargo clippy --all-targets -- -D warnings`
   must pass; CI enforces both on macOS and Linux.
3. New behavior ⇒ new tests. Perceptual-hash and palette behavior changes need
   before/after evidence in the PR description.
4. Screenshots for UI changes: capture the window after your change and link it
   in the PR (milestone QA also captures screenshots — see roadmap M2+).
5. Keep `docs/research/` factual: every claim links a source URL.
