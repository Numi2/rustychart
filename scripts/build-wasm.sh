#!/usr/bin/env bash
set -euo pipefail

# Build the UI for the browser (CSR) and bundle with wasm-bindgen.
RUST_VERSION=${RUST_VERSION:-1.83.0}
WASM_BINDGEN_VERSION=${WASM_BINDGEN_VERSION:-0.2.106}

export PATH="$HOME/.cargo/bin:$PATH"

if ! command -v rustup >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain "${RUST_VERSION}"
  export PATH="$HOME/.cargo/bin:$PATH"
else
  rustup default "${RUST_VERSION}"
fi

rustup target add wasm32-unknown-unknown

cargo install --locked --version "${WASM_BINDGEN_VERSION}" wasm-bindgen-cli

cargo build -p ui --lib --release --target wasm32-unknown-unknown --no-default-features --features csr

rm -rf dist
mkdir -p dist/pkg

wasm-bindgen \
  --target web \
  --no-typescript \
  --out-dir dist/pkg \
  target/wasm32-unknown-unknown/release/ui.wasm

cp web/index.html dist/index.html
