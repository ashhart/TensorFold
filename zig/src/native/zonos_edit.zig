//! A speech edit's steps on the ZONOS3 host: a clip's codes, a fill's rows, the edited codes' audio.
const std = @import("std");
const api = @import("engine_api");
const z3 = @import("zonos3");
const chunkFrames = @import("zonos_layout.zig").chunkFrames;
const cuda = @import("cuda");

const cfg = z3.config;
const W = cfg.frame_width;

/// Whether an encode or decode step is one the host runs.
pub fn valid(sp: api.SpeechInfo, r: *const api.SpeechRequest, ed: *const api.SpeechEdit) bool {
    switch (ed.op) {
        .encode => {
            const frames = (r.reference_audio.len + z3.codec.frame_samples - 1) / z3.codec.frame_samples;
            if (r.reference_audio.len < z3.clip.min_samples or frames > z3.clip.max_frames or ed.codes.len < frames) return false;
            for (r.reference_audio) |x| if (!std.math.isFinite(x)) return false;
            return true;
        },
        .decode => {
            if (ed.codes.len == 0 or ed.codes.len > sp.max_frames or r.pcm.len < ed.codes.len * sp.frame_length) return false;
            for (ed.codes) |frame| for (frame) |code| if (code < 0 or code >= cfg.codebook_size) return false;
            return true;
        },
        .place => {
            if (ed.words.len == 0 or ed.spans_ms.len < ed.words.len or ed.presence.len < ed.words.len) return false;
            // 30 s of audio, at least one patch of it at 16 kHz, and the transcript's bytes
            if (r.reference_audio.len > 30 * 44100 or r.reference_audio.len < 1412) return false;
            var bytes: usize = ed.words.len - 1;
            for (ed.words) |w| bytes += w.len;
            for (r.reference_audio) |x| if (!std.math.isFinite(x)) return false;
            return bytes <= z3.align_words.max_bytes;
        },
        .fill => return false,
    }
}

/// The step's words placed in the clip by the aligner.
pub fn alignWords(gpa: std.mem.Allocator, aligner: *z3.aligner.Aligner, r: *const api.SpeechRequest, ed: *const api.SpeechEdit) !void {
    const wav = try z3.features.resample(gpa, r.reference_audio);
    defer gpa.free(wav);
    const found = try aligner.forward(gpa, wav, ed.words, null);
    defer gpa.free(found.start_ms);
    defer gpa.free(found.end_ms);
    defer gpa.free(found.presence);
    for (ed.spans_ms[0..ed.words.len], found.start_ms, found.end_ms) |*span, s, e| span.* = .{ s, e };
    @memcpy(ed.presence[0..ed.words.len], found.presence);
}

/// Device bytes the aligner is admitted for: its weights, and a forward's rows at its limits.
pub const aligner_bytes: u64 = 1_846_591_492 + (512 << 20);

/// What a fill adds to a spoken request's checks: the adapter, room for its rows, one stream, no clip.
pub fn validFill(e: *const z3.Engine, r: *const api.SpeechRequest, ed: *const api.SpeechEdit) bool {
    if (e.adapter == null or ed.rows.len < r.max_frames or r.stream) return false;
    if (r.reference_audio.len != 0 or r.cfg_gamma != 1 or r.laci != null) return false;
    if (r.steering) |st| if (st.guidance != 1) return false;
    return true;
}

/// The clip's codes into the step's room.
pub fn encode(gpa: std.mem.Allocator, clip: *z3.clip.Clip, r: *const api.SpeechRequest, ed: *const api.SpeechEdit) !void {
    const codes = try clip.encode(gpa, r.reference_audio);
    defer gpa.free(codes);
    if (codes.len > ed.codes.len) return error.EditCodesOverflow;
    @memcpy(ed.codes[0..codes.len], codes);
    ed.out.frames = codes.len;
}

/// The step's codes spoken into the request's buffer, in a reply's chunk schedule; the samples written.
pub fn decode(gpa: std.mem.Allocator, codec: *z3.Codec, headroom_db: f32, r: *const api.SpeechRequest, ed: *const api.SpeechEdit) !usize {
    try codec.reset();
    var done: usize = 0;
    while (done < ed.codes.len) {
        const n = @min(chunkFrames(done), ed.codes.len - done);
        try codec.decode(ed.codes[done..][0..n], null);
        done += n;
    }
    const pcm = try codec.emit(gpa, true, null);
    defer gpa.free(pcm);
    if (pcm.len > r.pcm.len) return error.AudioOverflow;
    for (pcm) |x| if (!std.math.isFinite(x)) return error.NonFiniteAudio;
    z3.speech.output(pcm, headroom_db);
    @memcpy(r.pcm[0..pcm.len], pcm);
    return pcm.len;
}

/// A finished fill's rows and span-end frame into the step's room.
pub fn fillOut(ed: *const api.SpeechEdit, rows: []const [W]i32, end_frame: ?usize) !void {
    if (rows.len > ed.rows.len) return error.EditRowsOverflow;
    @memcpy(ed.rows[0..rows.len], rows);
    ed.out.rows = rows.len;
    ed.out.end_frame = if (end_frame) |f| @intCast(f) else -1;
}

/// The clip encoder, loaded by the first request that brings a clip; `h` is the host.
pub fn clipEncoder(h: anytype) !*z3.clip.Clip {
    if (h.clip) |c| return c;
    const clip_bytes = @import("zonos_host.zig").clip_bytes;
    if (h.admit) |a| if (clip_bytes > try a.room(a.ptr, &h.e.ctx)) return error.ClipEncoderNeedsMoreDeviceMemory;
    const before = cuda.usage(false).device;
    h.clip = try z3.clip.Clip.create(h.gpa, h.io, &h.e.driver, h.e.stream, h.dir, h.rel.generator, h.codec.w);
    const took = cuda.usage(false).device - before;
    std.log.info("zonos3: clip encoder loaded, {d} bytes on the device ({d} admitted)", .{ took, clip_bytes });
    if (took > clip_bytes) return error.ClipEncoderOverItsAdmission;
    return h.clip.?;
}

/// The forced aligner, loaded by the first edit that reads a clip's words from its transcript.
pub fn wordAligner(h: anytype) !*z3.aligner.Aligner {
    if (h.aligner) |a| return a;
    if (h.rel.aligner.len == 0) return error.NoAlignerInRelease;
    if (h.admit) |a| if (aligner_bytes > try a.room(a.ptr, &h.e.ctx)) return error.AlignerNeedsMoreDeviceMemory;
    h.aligner = try z3.aligner.Aligner.create(h.gpa, h.io, &h.e.driver, h.e.stream, h.rel.aligner, &h.codec.blas);
    std.log.info("zonos3: aligner loaded, {d} bytes of weights on the device ({d} admitted)", .{ h.aligner.?.bytes(), aligner_bytes });
    return h.aligner.?;
}

/// Device bytes the speech-edit adapter at `path` adds to a load: its factors, and the scratch its passes take.
pub fn editBytes(gpa: std.mem.Allocator, io: std.Io, path: []const u8, context: u32) !u64 {
    const scratch = z3.Engine.stateBytes(.{ .max_ctx = context, .adapter = path }) - z3.Engine.stateBytes(.{ .max_ctx = context });
    return scratch + try z3.adapter.deviceBytes(gpa, io, path);
}
