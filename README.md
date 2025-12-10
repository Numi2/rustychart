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

Override by setting those globals in a small inline script before loading `pkg/ui.js` or by editing `web/index.html`.
