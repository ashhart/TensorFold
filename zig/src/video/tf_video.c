// libtfvideo: a video's stream facts and RGB frames as transformers' read_video_pyav reads them, over FFmpeg 9.0.2.
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/cpu.h>
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

// The most frame threads one decode starts: most of their speed at a bounded memory and cost to other streams
#define TF_FRAME_THREADS 4
// The packets one frame-thread decode tracks at once (the bits of a uint64_t); a frame past them starts over
#define TF_WINDOW 64

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

// PyAV's _reformat of `frame` into `rgb`, then to `fn` as indices[k] (frame `at`): 0, or the TF_* code that ends it.
static int put(struct SwsContext* sws, AVFrame* rgb, AVFrame* frame, tf_frame_fn fn, void* ctx, int k, int at) {
    // Props copied, colour space and range via _set_frame_colorspace, trc/primaries unset
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
    if (av_frame_get_buffer(rgb, 0) < 0) return TF_MEMORY;
    int e = sws_scale_frame(sws, rgb, frame);
    frame->color_trc = trc;
    frame->color_primaries = pri;
    if (e < 0) return TF_DECODE;
    return fn(ctx, k, at, rgb->data[0], rgb->width, rgb->height, rgb->linesize[0]) != 0 ? TF_STOPPED : 0;
}

// Whether frame threads decode a packet as main's path does: for H.264, at most one slice (NAL units 1 and 5), I or P
static int exact(const AVCodecParameters* par, const AVPacket* pkt) {
    if (par->codec_id != AV_CODEC_ID_H264) return 1;
    // New extradata, as side data or an in-band avcC, can change the NAL length size this walk reads: main's path
    if (av_packet_get_side_data(pkt, AV_PKT_DATA_NEW_EXTRADATA, NULL)) return 0;
    const int len = par->extradata_size > 4 && par->extradata[0] == 1 ? (par->extradata[4] & 3) + 1 : 0;
    if (len && pkt->size >= 9 && pkt->data[0] == 1 && pkt->data[2] == 0 && (pkt->data[4] & 0xFC) == 0xFC) return 0;
    int n = 0;
    for (int64_t i = 0, size = 1; i + len < pkt->size; i += size, size = 1) {
        if (len) {
            size = 0;
            for (int b = 0; b < len; ++b) size = size << 8 | pkt->data[i++];
            if (!size) continue;
        } else if (i + 3 >= pkt->size || pkt->data[i] || pkt->data[i + 1] || pkt->data[i + 2] != 1) continue;
        else i += 3;
        if ((pkt->data[i] & 31) != 1 && (pkt->data[i] & 31) != 5) continue;
        // first_mb_in_slice 0 is a 1 bit, then slice_type's ue(v) fits the byte: 0 and 5 are P, 2 and 7 are I
        const int h = i + 1 < pkt->size ? pkt->data[i + 1] : 0;
        int z = 0;
        while (z < 4 && !(h & (0x40 >> z))) ++z;
        const int type = z < 4 ? ((h >> (6 - 2 * z)) & ((2 << (2 * z)) - 1)) - 1 : -1;
        // B slices go to main's: FFmpeg's frame threads conceal their damage differently, with no error to restart on
        if (++n > 1 || !(h & 0x80) || (type != 0 && type != 2 && type != 5 && type != 7)) return 0;
    }
    return 1;
}

// One decode from the start, indices[*next..] to `fn`: threads 0 is PyAV's slice-thread loop, else frame threads.
static int decode(tf_video* v, int threads, const int* indices, int n, int* next, tf_frame_fn fn, void* ctx) {
    int rc = TF_DECODE;
    AVStream* st = v->fmt->streams[v->stream];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* rgb = av_frame_alloc();
    struct SwsContext* sws = sws_alloc_context();
    // Frame threads hold each wanted frame until every packet up to its own has given an unmarked frame
    struct { AVFrame* f; int64_t k; int at; } hold[TF_WINDOW] = {{0}};
    int h0 = 0, nh = 0;
    int64_t sent = 0, low = 0;   // packets numbered; packets 0..low-1 have each given an unmarked frame
    uint64_t seen = 0;           // bit i: packet low + i has given its frame
    if (!dec || !pkt || !frame || !rgb || !sws) { rc = TF_MEMORY; goto done; }
    if (avcodec_parameters_to_context(dec, st->codecpar) < 0) goto done;
    dec->pkt_timebase = st->time_base;
    dec->thread_count = threads;   // 0: every CPU, as PyAV's stream codec context
    dec->thread_type = threads ? FF_THREAD_FRAME : FF_THREAD_SLICE;
    if (threads) {
        dec->err_recognition |= AV_EF_EXPLODE;   // damage as an error at its packet, not a unit skipped
        dec->flags |= AV_CODEC_FLAG_COPY_OPAQUE;   // each frame carries its packet's number
    }
    if (avcodec_open2(dec, codec, NULL) < 0) goto done;
    if (av_seek_frame(v->fmt, v->stream, 0, AVSEEK_FLAG_BACKWARD) < 0)
        avformat_seek_file(v->fmt, -1, INT64_MIN, 0, 0, 0);
    sws->threads = 0;
    sws->flags = SWS_BILINEAR;
    int at = 0, flushing = 0;
    while (*next < n) {
        if (!flushing) {
            int e = av_read_frame(v->fmt, pkt);
            if (e < 0) {
                flushing = 1;
                if (!threads) avcodec_send_packet(dec, NULL);   // frame threads first wait for every frame in flight
            } else {
                e = 0;
                if (pkt->stream_index == v->stream) {
                    // Frame threads stop at H.264 in several slices (slice threads spread them over cores) or a B slice
                    if (threads && !exact(st->codecpar, pkt)) goto done;
                    // A packet the container discards gives no frame, so only the others are numbered
                    if (threads && !(pkt->flags & AV_PKT_FLAG_DISCARD)) pkt->opaque = (void*)(intptr_t)++sent;
                    e = avcodec_send_packet(dec, pkt);   // a refused packet stops frame threads; PyAV's loop goes on
                }
                av_packet_unref(pkt);
                if (e < 0 && threads) goto done;
            }
        }
        for (;;) {
            // Frame threads at the input's end: every thread finishes before the drain, whose output copies their marks
            int sync = threads && flushing == 1;
            int e = sync ? avcodec_receive_frame_flags(dec, frame, AV_CODEC_RECEIVE_FRAME_FLAG_SYNCHRONOUS)
                         : avcodec_receive_frame(dec, frame);
            if (e == AVERROR(EAGAIN) && sync) { flushing = 2; avcodec_send_packet(dec, NULL); continue; }
            if (e == AVERROR(EAGAIN)) break;
            if (e == AVERROR_EOF && threads && low == sent) low = INT64_MAX;   // every packet gave a frame: all go
            else if (e == AVERROR_EOF) { rc = threads ? TF_DECODE : TF_SHORT; goto done; }
            else if (e < 0) goto done;
            else {
                // Frame threads stop at a frame the decoder marks damaged or concealed, or one outside the window
                int64_t k = threads ? (int64_t)(intptr_t)frame->opaque - 1 : 0;
                int marked = frame->decode_error_flags || (frame->flags & AV_FRAME_FLAG_CORRUPT);
                if (threads && (marked || k < low || k >= low + TF_WINDOW)) goto done;
                if (threads)
                    for (seen |= (uint64_t)1 << (k - low); seen & 1; seen >>= 1) ++low;
                if (*next + nh < n && at == indices[*next + nh]) {
                    if (threads) {
                        int i = (h0 + nh) % TF_WINDOW;
                        if (!hold[i].f && !(hold[i].f = av_frame_alloc())) { rc = TF_MEMORY; goto done; }
                        av_frame_move_ref(hold[i].f, frame);
                        hold[i].k = k;
                        hold[i].at = at;
                        ++nh;
                    } else {
                        int r = put(sws, rgb, frame, fn, ctx, *next, at);
                        if (r) { rc = r; goto done; }
                        ++*next;
                    }
                }
                ++at;
                av_frame_unref(frame);
            }
            for (; nh && hold[h0].k < low; h0 = (h0 + 1) % TF_WINDOW, --nh) {
                int r = put(sws, rgb, hold[h0].f, fn, ctx, *next, hold[h0].at);
                av_frame_unref(hold[h0].f);
                if (r) { rc = r; goto done; }
                ++*next;
            }
            if (*next == n) break;
            if (low == INT64_MAX) { rc = TF_SHORT; goto done; }
        }
    }
    rc = *next;
done:
    for (int i = 0; i < TF_WINDOW; ++i) av_frame_free(&hold[i].f);
    sws_freeContext(sws);
    av_frame_free(&rgb);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    return rc;
}

// The frames at `indices` (ascending, distinct), decoded from the start, each to `fn` as RGB; returns frames delivered.
int tf_video_frames(tf_video* v, const int* indices, int n, tf_frame_fn fn, void* ctx) {
    if (n <= 0) return 0;
    const AVCodecParameters* par = v->fmt->streams[v->stream]->codecpar;
    int next = 0, cpus = av_cpu_count(), rc = TF_DECODE;
    // Frame threads where slice threads keep one core busy: MPEG-4, and H.264 declaring no B-frames (exact() says why)
    if (cpus > 1 && (par->codec_id == AV_CODEC_ID_MPEG4 || (par->codec_id == AV_CODEC_ID_H264 && !par->video_delay)))
        rc = decode(v, cpus < TF_FRAME_THREADS ? cpus : TF_FRAME_THREADS, indices, n, &next, fn, ctx);
    // Damage, slices or a B slice, any failure but a clean end, a stop or no memory: main's path from indices[next]
    if (rc < 0 && rc != TF_SHORT && rc != TF_STOPPED && rc != TF_MEMORY) rc = decode(v, 0, indices, n, &next, fn, ctx);
    return rc;
}
