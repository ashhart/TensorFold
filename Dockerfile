# The signed Linux release archive in an image. Build, run and limits: docs/evaluating.md#container.
FROM ubuntu:24.04 AS fetch
ARG TENSORFOLD_VERSION=1.2.4
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl openssl \
    && rm -rf /var/lib/apt/lists/*
COPY packaging/runtime-trust.json /fetch/trust.json
WORKDIR /fetch
# the archive's checksum, then the release key's signature over it, then each unpacked file's checksum
RUN set -eu; \
    case "${TARGETARCH:-$(dpkg --print-architecture)}" in \
      amd64) p=linux-x86_64 ;; \
      arm64) p=linux-aarch64 ;; \
      *) echo "no TensorFold release archive for ${TARGETARCH:-this CPU}" >&2; exit 1 ;; \
    esac; \
    a="tensorfold-$TENSORFOLD_VERSION-$p.tar.gz"; \
    u="https://github.com/ashhart/TensorFold/releases/download/v$TENSORFOLD_VERSION"; \
    curl -fsSLO "$u/$a" -fsSLO "$u/$a.sha256" -fsSLO "$u/$a.sha256.sig"; \
    sha256sum -c "$a.sha256"; \
    sed -n 's/.*"ed25519PublicKey": *"\([^"]*\)".*/\1/p' trust.json | sed 's/\\n/\n/g' > release.pem; \
    { printf 'tensorfold-native-checksum-v1\n'; cat "$a.sha256"; } > signed; \
    openssl pkeyutl -verify -pubin -inkey release.pem -rawin -in signed -sigfile "$a.sha256.sig"; \
    mkdir /opt/tensorfold; \
    tar -xzf "$a" -C /opt/tensorfold --strip-components=1; \
    cd /opt/tensorfold && sha256sum --quiet -c SHA256SUMS

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* \
    && install -d -o ubuntu -g ubuntu /huggingface
COPY --from=fetch /opt/tensorfold /opt/tensorfold
ENV PATH=/opt/tensorfold/bin:$PATH \
    HF_HOME=/huggingface \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
USER ubuntu
EXPOSE 8080
ENTRYPOINT ["tensorfold-native"]
CMD ["--version"]
