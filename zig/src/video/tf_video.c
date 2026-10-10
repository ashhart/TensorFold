// libtfvideo: a video's frames for the native server's video input, as transformers' PyAV loader reads them
// (video_utils.read_video_pyav): the first video stream; its frame count (the stream's nb_frames), rate (its average
// frame rate), width and height; then the frames at the indices the family samples, counted in decode order from the
// start, each converted to RGB as PyAV's VideoFrame.to_ndarray(format="rgb24") does (VideoReformatter: the size kept,
// SWS_BILINEAR, the frame's colour space and range kept, transfer and primaries unspecified, sws_scale_frame).
// Built against FFmpeg 9.0.2, the release PyAV 19.0.1 ships (tools/zig/build_ffmpeg.sh, `zig build video`); the server
// loads it with dlopen when a request carries a video.
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/ffversion.h>
#include <libswscale/swscale.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int64_t frames;     // the stream's nb_frames, or (counted = 1) its packets when the container doesn't say
    int rate_num, rate_den;   // the stream's average frame rate (0/1 when unknown)
    int width, height;
    int counted;
} tf_video_info;

// One decoded frame: `rgb` holds height rows of width * 3 bytes, `stride` bytes apart; nonzero stops the decode.
typedef int (*tf_frame_fn)(void* ctx, int k, int index, const uint8_t* rgb, int width, int height, int stride);

enum {
    TF_OK = 0,
    TF_INVALID = -1,      // not a container FFmpeg reads here
    TF_NO_STREAM = -2,    // no video stream
    TF_NO_DECODER = -3,   // a codec this build doesn't decode
    TF_MEMORY = -4,
    TF_DECODE = -5,       // the stream failed to decode
    TF_SHORT = -6,        // the stream ended before a wanted index
    TF_STOPPED = -7,      // the callback stopped it
};

typedef struct { const uint8_t* data; size_t len, pos; } mem_t;

typedef struct tf_video {
    mem_t mem;
    AVFormatContext* fmt;
    AVIOContext* io;
    int stream;
} tf_video;

static int mem_read(void* o, uint8_t* buf, int n) {
    mem_t* m = (mem_t*)o;
    if (m->pos >= m->len) return AVERROR_EOF;
    size_t k = m->len - m->pos < (size_t)n ? m->len - m->pos : (size_t)n;
    memcpy(buf, m->data + m->pos, k);
    m->pos += k;
    return (int)k;
}

static int64_t mem_seek(void* o, int64_t off, int whence) {
    mem_t* m = (mem_t*)o;
    if (whence == AVSEEK_SIZE) return (int64_t)m->len;
    whence &= ~AVSEEK_FORCE;
    int64_t at = whence == SEEK_SET ? off : whence == SEEK_CUR ? (int64_t)m->pos + off : (int64_t)m->len + off;
    if (at < 0 || at > (int64_t)m->len) return -1;
    m->pos = (size_t)at;
    return at;
}

// PyAV's _set_frame_colorspace(frame, colorspace, UNSPECIFIED): an SWS_CS_* value names the frame's colour space
// (the frame's own AVColorSpace number passes through it as PyAV passes it)
static void set_colorspace(AVFrame* f, int cs) {
    if (cs == SWS_CS_ITU709) f->colorspace = AVCOL_SPC_BT709;
    else if (cs == SWS_CS_FCC) f->colorspace = AVCOL_SPC_FCC;
    else if (cs == SWS_CS_ITU601) f->colorspace = AVCOL_SPC_SMPTE170M;
    else if (cs == SWS_CS_SMPTE240M) f->colorspace = AVCOL_SPC_SMPTE240M;
    else if (cs == SWS_CS_BT2020) f->colorspace = AVCOL_SPC_BT2020_NCL;
}

const char* tf_video_version(void) { return FFMPEG_VERSION; }

void tf_video_close(tf_video* v) {
    if (!v) return;
    if (v->fmt) avformat_close_input(&v->fmt);
    if (v->io) { av_freep(&v->io->buffer); avio_context_free(&v->io); }
    free(v);
}

// The container in `data` (which must outlive the handle) and its first video stream's facts.
int tf_video_open(const uint8_t* data, size_t len, tf_video** out, tf_video_info* info) {
    *out = NULL;
    memset(info, 0, sizeof(*info));
    av_log_set_level(AV_LOG_QUIET);
    tf_video* v = calloc(1, sizeof(tf_video));
    if (!v) return TF_MEMORY;
    v->mem = (mem_t){data, len, 0};
    unsigned char* buf = av_malloc(1 << 16);
    if (!buf) { free(v); return TF_MEMORY; }
    v->io = avio_alloc_context(buf, 1 << 16, 0, &v->mem, mem_read, NULL, mem_seek);
    if (!v->io) { av_free(buf); free(v); return TF_MEMORY; }
    v->fmt = avformat_alloc_context();
    if (!v->fmt) { tf_video_close(v); return TF_MEMORY; }
    v->fmt->pb = v->io;
    if (avformat_open_input(&v->fmt, NULL, NULL, NULL) < 0) { v->fmt = NULL; tf_video_close(v); return TF_INVALID; }
    if (avformat_find_stream_info(v->fmt, NULL) < 0) { tf_video_close(v); return TF_INVALID; }
    v->stream = -1;   // PyAV's streams.video[0]: the first video stream
    for (unsigned i = 0; i < v->fmt->nb_streams; ++i)
        if (v->fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { v->stream = (int)i; break; }
    if (v->stream < 0) { tf_video_close(v); return TF_NO_STREAM; }
    AVStream* st = v->fmt->streams[v->stream];
    info->frames = st->nb_frames;
    info->rate_num = st->avg_frame_rate.num;
    info->rate_den = st->avg_frame_rate.den;
    info->width = st->codecpar->width;
    info->height = st->codecpar->height;
    if (!avcodec_find_decoder(st->codecpar->codec_id)) { tf_video_close(v); return TF_NO_DECODER; }
    if (info->frames <= 0) {   // Matroska and WebM keep no count: the stream's packets, one a shown frame, then back
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) { tf_video_close(v); return TF_MEMORY; }
        int64_t n = 0;
        while (av_read_frame(v->fmt, pkt) >= 0) {
            if (pkt->stream_index == v->stream) ++n;
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
        info->frames = n;
        info->counted = 1;
    }
    *out = v;
    return TF_OK;
}

// The frames at `indices` (ascending, distinct), decoded from the start (read_video_pyav seeks to 0 and enumerates
// container.decode(video=0)), each handed to `fn` as RGB. Returns the frames delivered, or an error.
int tf_video_frames(tf_video* v, const int* indices, int n, tf_frame_fn fn, void* ctx) {
    if (n <= 0) return 0;
    int rc = TF_DECODE;
    AVStream* st = v->fmt->streams[v->stream];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb = av_frame_alloc();
    struct SwsContext* sws = sws_alloc_context();
    if (!dec || !pkt || !frame || !rgb || !sws) { rc = TF_MEMORY; goto done; }
    if (avcodec_parameters_to_context(dec, st->codecpar) < 0) goto done;
    dec->pkt_timebase = st->time_base;
    dec->thread_count = 0;   // PyAV's stream codec context: every CPU, slice threads
    dec->thread_type = FF_THREAD_SLICE;
    if (avcodec_open2(dec, codec, NULL) < 0) goto done;
    if (av_seek_frame(v->fmt, v->stream, 0, AVSEEK_FLAG_BACKWARD) < 0) avformat_seek_file(v->fmt, -1, INT64_MIN, 0, 0, 0);
    sws->threads = 0;
    sws->flags = SWS_BILINEAR;
    int at = 0, next = 0, flushing = 0;
    while (next < n) {
        if (!flushing) {
            int e = av_read_frame(v->fmt, pkt);
            if (e < 0) { flushing = 1; avcodec_send_packet(dec, NULL); }
            else {
                if (pkt->stream_index == v->stream) avcodec_send_packet(dec, pkt);
                av_packet_unref(pkt);
            }
        }
        for (;;) {
            int e = avcodec_receive_frame(dec, frame);
            if (e == AVERROR(EAGAIN)) break;
            if (e == AVERROR_EOF) { rc = TF_SHORT; goto done; }
            if (e < 0) goto done;
            if (at == indices[next]) {
                // PyAV's _reformat: the destination copies the frame's properties; colour space and range pass
                // through _set_frame_colorspace with the frame's own values; transfer and primaries unspecified
                av_frame_unref(rgb);
                av_frame_copy_props(rgb, frame);
                set_colorspace(frame, (int)frame->colorspace);
                rgb->colorspace = frame->colorspace;
                rgb->color_range = frame->color_range;
                set_colorspace(rgb, (int)frame->colorspace);
                const enum AVColorTransferCharacteristic trc = frame->color_trc;
                const enum AVColorPrimaries pri = frame->color_primaries;
                frame->color_trc = rgb->color_trc = AVCOL_TRC_UNSPECIFIED;
                frame->color_primaries = rgb->color_primaries = AVCOL_PRI_UNSPECIFIED;
                rgb->format = AV_PIX_FMT_RGB24;
                rgb->width = frame->width;
                rgb->height = frame->height;
                if (av_frame_get_buffer(rgb, 0) < 0) { rc = TF_MEMORY; goto done; }
                int e2 = sws_scale_frame(sws, rgb, frame);
                frame->color_trc = trc;
                frame->color_primaries = pri;
                if (e2 < 0) goto done;
                if (fn(ctx, next, at, rgb->data[0], rgb->width, rgb->height, rgb->linesize[0]) != 0) { rc = TF_STOPPED; goto done; }
                ++next;
                if (next == n) break;
            }
            ++at;
            av_frame_unref(frame);
        }
    }
    rc = next;
done:
    sws_freeContext(sws);
    av_frame_free(&rgb);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    return rc;
}
