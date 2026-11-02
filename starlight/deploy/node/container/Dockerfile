# syntax=docker/dockerfile:1.7
# Starlight Node with the TensorFold CUDA engine; docs/nodes.md has the build and run commands.

ARG BASE=ubuntu:24.04

FROM golang:1.27.3-bookworm AS agent
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/starlight-node ./cmd/starlight-node

FROM ${BASE} AS engine
ARG TARGETARCH
ARG TENSORFOLD_VERSION=1.2.0
ARG TENSORFOLD_SHA256
RUN test -n "$TENSORFOLD_SHA256" || (echo "pass --build-arg TENSORFOLD_SHA256 with the archive's published SHA-256" && exit 1)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN set -eu; \
    case "$TARGETARCH" in amd64) arch=linux-x86_64 ;; arm64) arch=linux-aarch64 ;; *) echo "unsupported arch $TARGETARCH"; exit 1 ;; esac; \
    name="tensorfold-${TENSORFOLD_VERSION}-${arch}"; \
    curl -fsSL -o /tmp/engine.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v${TENSORFOLD_VERSION}/${name}.tar.gz"; \
    echo "${TENSORFOLD_SHA256}  /tmp/engine.tar.gz" | sha256sum -c -; \
    mkdir -p /opt/tensorfold; \
    tar -xzf /tmp/engine.tar.gz -C /opt/tensorfold --strip-components=1; \
    rm /tmp/engine.tar.gz; \
    /opt/tensorfold/bin/tensorfold-native --version

FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /var/lib/starlight-node --shell /usr/sbin/nologin starlight-node \
    && mkdir -p /var/lib/starlight-node/home && chown -R starlight-node:starlight-node /var/lib/starlight-node && chmod 0700 /var/lib/starlight-node
COPY --from=engine /opt/tensorfold /opt/tensorfold
COPY --from=agent /out/starlight-node /usr/local/bin/starlight-node
ENV HOME=/var/lib/starlight-node/home \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_VISIBLE_DEVICES=all
USER starlight-node
VOLUME ["/var/lib/starlight-node"]
EXPOSE 7443
ENTRYPOINT ["/usr/local/bin/starlight-node"]
CMD ["run", "--state-dir", "/var/lib/starlight-node", "--engine", "/opt/tensorfold/bin/tensorfold-native", "--listen", ":7443", "--hf-home", "/var/lib/starlight-node/hf"]
