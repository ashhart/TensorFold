# syntax=docker/dockerfile:1.7
FROM --platform=$BUILDPLATFORM golang:1.27 AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
ARG TARGETOS=linux
ARG TARGETARCH=amd64
ARG VERSION=0.0.0-dev
ARG COMMIT=
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 GOOS=$TARGETOS GOARCH=$TARGETARCH go build -trimpath \
    -ldflags "-s -w -X github.com/ashhart/TensorFold/starlight/internal/buildinfo.Version=${VERSION} -X github.com/ashhart/TensorFold/starlight/internal/buildinfo.Commit=${COMMIT}" \
    -o /out/starlight ./cmd/starlight

FROM gcr.io/distroless/static-debian12:nonroot
LABEL org.opencontainers.image.title="TensorFold Starlight" \
      org.opencontainers.image.description="Governed OpenAI- and Anthropic-compatible gateway and control plane for TensorFold engines" \
      org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.licenses="Apache-2.0"
COPY --from=build /out/starlight /usr/local/bin/starlight
# Metrics must listen on all interfaces inside a container; protect it with metrics.bearer_token_file and network policy.
ENV STARLIGHT_LISTEN_METRICS=:9090
USER nonroot:nonroot
EXPOSE 8443 9443 9090
ENTRYPOINT ["/usr/local/bin/starlight"]
CMD ["serve"]
