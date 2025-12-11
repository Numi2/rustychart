FROM rust:1.88-bullseye AS builder
WORKDIR /app

COPY . .

# Build only the backend server (frontend is deployed separately to Vercel)
RUN cargo build --release -p backend

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/target/release/backend /app/backend

ENV RUST_LOG=info
EXPOSE 8080
CMD ["./backend"]
