// libtfvideo: a video's sampled frames for the native server's vision inputs, as the Python engine decodes them
// (tensorfold/vision/videos.py with PyAV): the first video stream; its rate (average, else guessed, else 24) and
// frame count (the stream's, else its duration times the rate); frames sampled at ``rate`` a second over the whole
// video (Qwen3-VL's rule: at least min_frames, at most max_frames, numpy's linspace rounded half to even), decoded
// in order and scaled straight to the size ``size`` picks as RGB the way PyAV's VideoFrame.to_ndarray(rgb24, width,
// height, interpolation="BICUBIC") does it (VideoReformatter: the frame's colour space and range kept, transfer and
// primaries unspecified, sws_scale_frame with SWS_BICUBIC alone). Built against FFmpeg 9 (the FFmpeg PyAV 19.0.1
// ships, so the frames equal Python's); the server loads it when a request carries a video.
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int (*tf_size_fn)(void* ctx, int frames, int height, int width, int* out_h, int* out_w);

typedef struct {
    uint8_t* rgb;   // frames x height x width x 3
    int frames, height, width;
    int* indices;   // each frame's index in the source
    double fps;     // the source's rate
} tf_video;

enum {
    TF_OK = 0,
    TF_INVALID = -1,      // not a decodable container or stream
    TF_NO_STREAM = -2,    // no video stream
    TF_DIMENSIONS = -3,   // missing or over max_dim
    TF_NO_COUNT = -4,     // the frame count or rate is missing
    TF_TOO_LONG = -5,     // over max_seconds
    TF_FEW_FRAMES = -6,   // fewer than two decodable frames
    TF_MEMORY = -7,
    TF_SIZE = -8,         // the size callback refused
};

typedef struct { const uint8_t* data; size_t len, pos; } mem_t;

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

static double round_half_even(double x) {
    double r = round(x);
    if (fabs(x - trunc(x)) == 0.5) r = 2.0 * round(x / 2.0);
    return r;
}

void tf_video_free(tf_video* v) {
    if (!v) return;
    free(v->rgb);
    free(v->indices);
    v->rgb = NULL;
    v->indices = NULL;
}

int tf_video_decode(const uint8_t* data, size_t len, double rate, int min_frames, int max_frames, double max_seconds,
                    int max_dim, tf_size_fn size, void* ctx, tf_video* out) {
    memset(out, 0, sizeof(*out));
    int rc = TF_INVALID;
    mem_t mem = {data, len, 0};
    AVFormatContext* fmt = NULL;
    AVIOContext* io = NULL;
    AVCodecContext* dec = NULL;
    AVPacket* pkt = NULL;
    AVFrame* frame = NULL;
    struct SwsContext* sws = NULL;
    int* wanted = NULL;
    unsigned char* iobuf = av_malloc(1 << 16);
    if (!iobuf) return TF_MEMORY;
    io = avio_alloc_context(iobuf, 1 << 16, 0, &mem, mem_read, NULL, mem_seek);
    if (!io) { av_free(iobuf); return TF_MEMORY; }
    fmt = avformat_alloc_context();
    if (!fmt) { rc = TF_MEMORY; goto done; }
    fmt->pb = io;
    if (avformat_open_input(&fmt, NULL, NULL, NULL) < 0) { fmt = NULL; goto done; }
    if (avformat_find_stream_info(fmt, NULL) < 0) goto done;
    int si = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (si < 0) { rc = TF_NO_STREAM; goto done; }
    // PyAV's streams.video[0] is the first video stream
    for (unsigned i = 0; i < fmt->nb_streams; ++i)
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) { si = (int)i; break; }
    AVStream* st = fmt->streams[si];
    int width = st->codecpar->width, height = st->codecpar->height;
    if (width <= 0 || height <= 0 || width > max_dim || height > max_dim) { rc = TF_DIMENSIONS; goto done; }
    AVRational r = st->avg_frame_rate;
    if (r.num <= 0 || r.den <= 0) r = av_guess_frame_rate(fmt, st, NULL);
    double fps = (r.num > 0 && r.den > 0) ? av_q2d(r) : 24.0;
    double seconds = 0.0;
    if (st->duration > 0 && st->duration != AV_NOPTS_VALUE) seconds = (double)st->duration * av_q2d(st->time_base);
    else if (fmt->duration > 0 && fmt->duration != AV_NOPTS_VALUE) seconds = (double)fmt->duration / 1e6;
    int64_t total = st->nb_frames > 0 ? st->nb_frames : (int64_t)round_half_even(seconds * fps);
    if (!(fps > 0 && fps <= 1000) || total <= 0) { rc = TF_NO_COUNT; goto done; }
    if ((double)total / fps > max_seconds) { rc = TF_TOO_LONG; goto done; }
    int64_t count = (int64_t)((double)total / fps * rate);
    if (count < min_frames) count = min_frames;
    if (count > max_frames) count = max_frames;
    if (count > total) count = total;
    wanted = malloc(sizeof(int) * (size_t)count);
    if (!wanted) { rc = TF_MEMORY; goto done; }
    // numpy.linspace(0, total - 1, count): step = (stop - start) / (count - 1), the last exactly stop
    double stop = (double)(total - 1);
    for (int64_t i = 0; i < count; ++i) {
        double v = count > 1 ? (double)i * (stop / (double)(count - 1)) : 0.0;
        if (count > 1 && i == count - 1) v = stop;
        wanted[i] = (int)round_half_even(v);
    }
    int oh = 0, ow = 0;
    if (size(ctx, (int)count, height, width, &oh, &ow) != 0 || oh <= 0 || ow <= 0) { rc = TF_SIZE; goto done; }
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) goto done;
    dec = avcodec_alloc_context3(codec);
    if (!dec || avcodec_parameters_to_context(dec, st->codecpar) < 0) goto done;
    dec->thread_count = 0;   // PyAV's thread_type "AUTO"
    dec->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    if (avcodec_open2(dec, codec, NULL) < 0) goto done;
    out->rgb = malloc((size_t)count * (size_t)oh * (size_t)ow * 3);
    out->indices = malloc(sizeof(int) * (size_t)count);
    pkt = av_packet_alloc();
    frame = av_frame_alloc();
    if (!out->rgb || !out->indices || !pkt || !frame) { rc = TF_MEMORY; goto done; }
    AVFrame* rgb = av_frame_alloc();
    if (!rgb) { rc = TF_MEMORY; goto done; }
    sws = sws_alloc_context();
    if (!sws) { av_frame_free(&rgb); rc = TF_MEMORY; goto done; }
    sws->threads = 0;
    sws->flags = SWS_BICUBIC;
    int n = 0, next = 0, got = 0, last = wanted[count - 1];
    int flushing = 0, finished = 0;
    while (!finished) {
        if (!flushing) {
            int e = av_read_frame(fmt, pkt);
            if (e < 0) { flushing = 1; avcodec_send_packet(dec, NULL); }
            else {
                if (pkt->stream_index == si) avcodec_send_packet(dec, pkt);
                av_packet_unref(pkt);
            }
        }
        for (;;) {
            int e = avcodec_receive_frame(dec, frame);
            if (e == AVERROR(EAGAIN)) {
                if (flushing) finished = 1;   // a flushed decoder only ends
                break;
            }
            if (e < 0) { finished = 1; break; }
            if (n > last) { finished = 1; av_frame_unref(frame); break; }
            if (next < count && n == wanted[next]) {
                // PyAV's _reformat: the destination copies the frame's properties; its colour space and range pass
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
                rgb->width = ow;
                rgb->height = oh;
                if (av_frame_get_buffer(rgb, 0) < 0) { rc = TF_MEMORY; av_frame_unref(frame); av_frame_free(&rgb); goto done; }
                int e2 = sws_scale_frame(sws, rgb, frame);
                frame->color_trc = trc;
                frame->color_primaries = pri;
                if (e2 < 0) { av_frame_unref(frame); av_frame_free(&rgb); goto done; }
                uint8_t* to = out->rgb + (size_t)got * oh * ow * 3;
                for (int y = 0; y < oh; ++y) memcpy(to + (size_t)y * ow * 3, rgb->data[0] + (size_t)y * rgb->linesize[0], (size_t)ow * 3);
                out->indices[got++] = n;
                while (next < count && wanted[next] == n) ++next;
            }
            ++n;
            av_frame_unref(frame);
        }
    }
    av_frame_free(&rgb);
    if (got < 2) { rc = TF_FEW_FRAMES; goto done; }
    out->frames = got;
    out->height = oh;
    out->width = ow;
    out->fps = fps;
    rc = TF_OK;
done:
    if (rc != TF_OK) tf_video_free(out);
    free(wanted);
    sws_freeContext(sws);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    if (fmt) avformat_close_input(&fmt);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    return rc;
}
