# RustyChart Deployment

## URLs
- Backend (Fly): `https://rustychart.fly.dev/`
- Frontend (Vercel): `https://rustychart-7l4gz74k4-numi2s-projects.vercel.app`

## Backend (Fly)
Builds only the Axum server with rustls:
- Config: `fly.toml`
- Dockerfile: builds `backend` binary (no cargo-leptos).
- Deploy: `fly deploy`

## Frontend (Vercel)
Client-only WASM (Leptos hydrate) built via `scripts/build-wasm.sh`.

### Fresh build
```
./scripts/build-wasm.sh
```
Outputs `dist/` with `pkg/ui.js` and `index.html`.

### Prebuilt deploy (recommended)
```
./scripts/build-wasm.sh
rm -rf .vercel/output
mkdir -p .vercel/output/static
cp -R dist/. .vercel/output/static/
cp .vercel/output/static/index.html .vercel/output/index.html
cat > .vercel/output/config.json <<'EOF'
{
  "version": 3,
  "routes": [
    { "handle": "filesystem" },
    { "src": "/(.*)", "dest": "/index.html" }
  ]
}
EOF
vercel deploy --prebuilt --prod --yes --archive=tgz
```

### Build-and-deploy from source on Vercel
Vercel uses `vercel.json`:
- `buildCommand`: `./scripts/build-wasm.sh`
- `outputDirectory`: `dist`

If building on Vercel infra, ensure Rust and wasm-bindgen install succeed (the script bootstraps rustup and wasm32 target).

## Runtime configuration
Frontend reads API endpoints from browser globals:
- `window.RUSTYCHART_API_BASE` (default `https://rustychart.fly.dev/api`)
- `window.RUSTYCHART_WS_BASE` (default `wss://rustychart.fly.dev/api/ws`)
- `window.RUSTYCHART_DEFAULT_PROVIDER` (optional: `alpha`, `yahoo`, `coinbase`; defaults to `alpha` for non-crypto symbols, `coinbase` for crypto)

Override by setting those globals in a small inline script before loading `pkg/ui.js` or by editing `web/index.html`.

Backend:
- `ALPHAVANTAGE_API_KEY` enables Alpha Vantage history/search endpoints (free tier respected via local rate limit). If unset, Alpha routes return HTTP 503.

## Script engine quickstart
- Pine v6 templates live in `script-engine/src/templates.rs` (e.g., `starter`, `ema`, `macd`). ThinkScript subset templates mirror the same names.
- Validate sources via `script_engine::validate_script(source, lang)`; unsupported scripts return structured diagnostics with codes, spans, and hints for UI highlighting.
- Remediation helpers in `script-engine/src/remediation.rs` auto-suggest fixes such as adding `//@version=6` or a default plot.
- Compatibility reports include severity and hints; surface them in the UI to guide authors before execution.

## Script editor UX
- The in-app script drawer shows live diagnostics (severity, code, span, hint) per issue plus a readiness badge.
- Completions and templates are clickable pills; selection updates the script body immediately.
- Artifacts and manifests render in expandable panels so users can preview compilation output before running.
