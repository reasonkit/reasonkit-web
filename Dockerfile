# syntax=docker/dockerfile:1.4
# ReasonKit Web - Multi-stage Production Dockerfile
# Web sensing and browser automation MCP sidecar
#
# Build: docker build -t reasonkit-web .
# Run:   docker run --rm -i reasonkit-web

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM rust:1.74-bookworm AS builder

# Install build dependencies for chromiumoxide
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for building
RUN useradd --create-home --user-group builder
USER builder
WORKDIR /home/builder/app

# Copy manifests first for better layer caching
COPY --chown=builder:builder Cargo.toml Cargo.lock* ./

# Create dummy source to cache dependencies
RUN mkdir -p src && \
    echo 'fn main() { println!("dummy"); }' > src/main.rs && \
    echo 'pub fn dummy() {}' > src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release && \
    rm -rf src target/release/deps/reasonkit_web* target/release/reasonkit-web*

# Copy actual source code
COPY --chown=builder:builder src ./src

# Build the actual application
RUN cargo build --release --locked && \
    strip target/release/reasonkit-web

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM debian:bookworm-slim AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="ReasonKit Web"
LABEL org.opencontainers.image.description="Web sensing and browser automation MCP sidecar"
LABEL org.opencontainers.image.vendor="ReasonKit"
LABEL org.opencontainers.image.url="https://reasonkit.sh"
LABEL org.opencontainers.image.source="https://github.com/reasonkit/reasonkit-web"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install runtime dependencies and Chromium for browser automation
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium browser
    chromium \
    # Chromium dependencies
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Fonts for proper rendering
    fonts-liberation \
    fonts-noto-color-emoji \
    # SSL/TLS support
    ca-certificates \
    # Health check utility
    curl \
    # Timezone data
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 reasonkit && \
    useradd --uid 1000 --gid reasonkit --shell /bin/bash --create-home reasonkit

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R reasonkit:reasonkit /app

# Copy binary from builder
COPY --from=builder --chown=reasonkit:reasonkit \
    /home/builder/app/target/release/reasonkit-web /app/reasonkit-web

# Set up Chromium sandbox (required for non-root operation)
RUN mkdir -p /home/reasonkit/.cache/chromium && \
    chown -R reasonkit:reasonkit /home/reasonkit

# Set environment variables
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROME_PATH=/usr/bin/chromium
# Disable Chromium sandbox in container (handled by container isolation)
ENV CHROMIUM_FLAGS="--no-sandbox --disable-gpu --disable-dev-shm-usage"
# Data directories
ENV REASONKIT_DATA_DIR=/app/data
ENV REASONKIT_CACHE_DIR=/app/cache
ENV REASONKIT_LOG_DIR=/app/logs

# Switch to non-root user
USER reasonkit
WORKDIR /app

# Expose health check port (if HTTP health endpoint is added)
EXPOSE 3847

# Health check - verify binary exists and Chromium is available
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD test -x /app/reasonkit-web && test -x /usr/bin/chromium || exit 1

# Default command - run MCP server (stdio mode)
ENTRYPOINT ["/app/reasonkit-web"]
CMD ["serve"]
