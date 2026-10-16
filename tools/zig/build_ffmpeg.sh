#!/usr/bin/env bash
# FFmpeg 9.0.2 (PyAV 19.0.1's) for libtfvideo, static PIC, PyAV's decoders only: build_ffmpeg.sh PREFIX [SOURCE_DIR]
set -euo pipefail
version=9.0.2
sha256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
prefix="$(mkdir -p "$1" && cd "$1" && pwd)"
work="${2:-$prefix/src}"
mkdir -p "$work"
cd "$work"
if [ ! -d "ffmpeg-$version" ]; then
  [ -f "ffmpeg-$version.tar.xz" ] || curl -sSfLO "https://ffmpeg.org/releases/ffmpeg-$version.tar.xz"
  echo "$sha256  ffmpeg-$version.tar.xz" | shasum -a 256 -c -
  tar xf "ffmpeg-$version.tar.xz"
fi
cd "ffmpeg-$version"
./configure --prefix="$prefix" --enable-static --disable-shared --enable-pic \
  --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything \
  --disable-avdevice --disable-avfilter --disable-swresample \
  --enable-avcodec --enable-avformat --enable-swscale \
  --enable-decoder=h264,hevc,vp8,vp9,mpeg4 \
  --enable-parser=h264,hevc,vp8,vp9,mpeg4video \
  --enable-demuxer=mov,matroska \
  --enable-bsf=vp9_superframe_split
make -j"$(getconf _NPROCESSORS_ONLN)"
make install
echo "FFmpeg $version in $prefix"
