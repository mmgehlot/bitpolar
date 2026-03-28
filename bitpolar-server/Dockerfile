# Stage 1: Build
FROM rust:latest AS builder

WORKDIR /usr/src/bitpolar

# Install protobuf compiler (required by tonic-build)
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Copy entire workspace (code is at root in standalone repo)
COPY . .

# Build release binary
RUN cargo build --release -p bitpolar-server

# Stage 2: Runtime
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/bitpolar/target/release/bitpolar-server /usr/local/bin/bitpolar-server

EXPOSE 50051

ENV BITPOLAR_PORT=50051
ENV BITPOLAR_DIM=384
ENV BITPOLAR_BITS=4
ENV BITPOLAR_SEED=42

ENTRYPOINT ["bitpolar-server"]
