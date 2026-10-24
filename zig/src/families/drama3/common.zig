//! What both Drama 3 backends share: request settings, the parity probe, streaming stats and the clock.
const std = @import("std");

/// The prompt path's default: one row a chunk, the arithmetic the single-row path used.
pub const single_row: usize = 1;

pub const Options = struct {
    device: u32 = 0,
    max_ctx: usize = 4096, // positions (prompt rows + frames)
    max_rows: usize = 1024, // prompt rows a chunk (Metal's capacity; the CUDA path asks for its own)
    attn_chunk: usize = 512, // query rows a prompt attention GEMM
    max_frames: usize = 2048,
    threads: usize = 16,
    pdl: bool = true, // CUDA: GEMVs launched with programmatic dependent launch
    warmup: bool = true, // a short prompt and frame at open: kernels, heuristics and the frame step ready
    split: bool = true, // CUDA: tower codes 2..9 on a second stream beside the next backbone row

    pub fn validatePrefill(o: Options) !void {
        if (o.max_rows == 0 or o.max_rows > 8192) return error.BadPrefillRows;
        if (o.attn_chunk == 0 or o.attn_chunk > 1024) return error.BadAttentionChunk;
    }

    pub fn promptRows(o: Options, remaining: usize) usize {
        return @min(o.max_rows, remaining);
    }
};

pub const Sampling = struct {
    temperature: f32 = 1.0,
    top_k: i32 = 30,
    top_p: f32 = 0.9,
    seed: u64 = 1,

    pub fn greedy(s: Sampling) bool {
        return s.temperature <= 0 or s.top_k == 1;
    }
};

/// Per-layer probe of a pass under the oracle's dump names, called with device rows [t, width] in bf16 or fp32.
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, rows: usize, width: usize, f32_: bool) anyerror!void,
};

pub const Sink = struct {
    ctx: *anyopaque,
    audio: *const fn (ctx: *anyopaque, samples: []const f32) anyerror!void,
};

pub const StreamOptions = struct {
    first_chunk: usize = 4, // frames before the first audio (4 = 186 ms of audio)
    chunk: usize = 16, // frames a later chunk, reached by 1.25x steps from first_chunk (generation runs ~1.5x
    // real time, so each chunk is made within the audio already handed over: no gap when played as it arrives)
};

pub const Stats = struct {
    frames: usize,
    eos: bool,
    prompt_rows: usize,
    prefill_ms: f64,
    first_audio_ms: f64, // request start -> the first samples handed to the sink
    total_ms: f64,
    audio_s: f64,

    pub fn fps(s: Stats) f64 {
        return @as(f64, @floatFromInt(s.frames)) / ((s.total_ms - s.prefill_ms) / 1e3);
    }
};

pub fn nowNs(io: std.Io) i128 {
    return @intCast(std.Io.Clock.awake.now(io).toNanoseconds());
}

pub fn ms(a: i128, b: i128) f64 {
    return @as(f64, @floatFromInt(b - a)) / 1e6;
}

pub fn referenceOffset(total: usize, rows: usize, width: usize, at: ?usize) !usize {
    if (width == 0 or total % width != 0 or rows > total / width) return error.BadProbeRows;
    const last = total / width - rows;
    const first = at orelse last;
    if (first > last) return error.BadProbeRows;
    return first * width;
}

test "prefill capacity rejects empty and oversized chunks" {
    const t = std.testing;
    try (Options{}).validatePrefill();
    for ([_]usize{ 1, 128, 1024, 8192 }) |rows| try (Options{ .max_rows = rows }).validatePrefill();
    for ([_]usize{ 0, 8193, std.math.maxInt(usize) }) |rows| {
        try t.expectError(error.BadPrefillRows, (Options{ .max_rows = rows }).validatePrefill());
    }
    for ([_]usize{ 0, 1025, std.math.maxInt(usize) }) |rows| {
        try t.expectError(error.BadAttentionChunk, (Options{ .attn_chunk = rows }).validatePrefill());
    }
}

test "prefill rounds cover the prompt and its partial tail" {
    const t = std.testing;
    const defaults: Options = .{};
    try t.expectEqual(@as(usize, 1024), defaults.max_rows); // Metal's capacity; the CUDA path asks for its own
    try t.expectEqual(@as(usize, 512), defaults.attn_chunk);
    // the default the CUDA path takes: one row a chunk, the arithmetic the single-row path used
    const single: Options = .{ .max_rows = single_row, .attn_chunk = single_row };
    try single.validatePrefill();
    try t.expectEqual(@as(usize, 1), single.promptRows(1));
    try t.expectEqual(@as(usize, 1), single.promptRows(4096));
    for ([_]usize{ 1, 2, 128, 1024 }) |capacity| {
        const o: Options = .{ .max_rows = capacity };
        for ([_]usize{ 2, 127, 128, 129, 1025, 8192 }) |n| {
            var at: usize = 0;
            var rounds: usize = 0;
            while (at < n) {
                const count = o.promptRows(n - at);
                try t.expect(count > 0 and count <= capacity and count <= n - at);
                at += count;
                rounds += 1;
            }
            try t.expectEqual(n, at);
            try t.expectEqual((n + capacity - 1) / capacity, rounds);
            try t.expectEqual(@as(usize, 0), o.promptRows(n - at));
        }
    }
}

test "prompt probes select matching rows instead of the reference tail" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 0), try referenceOffset(128 * 16, 1, 16, 0));
    try t.expectEqual(@as(usize, 64 * 16), try referenceOffset(128 * 16, 64, 16, 64));
    try t.expectEqual(@as(usize, 127 * 16), try referenceOffset(128 * 16, 1, 16, null));
    try t.expectError(error.BadProbeRows, referenceOffset(128 * 16, 2, 16, 127));
    try t.expectError(error.BadProbeRows, referenceOffset(128 * 16, 129, 16, 0));
    try t.expectError(error.BadProbeRows, referenceOffset(128 * 16, 1, 0, 0));
    try t.expectError(error.BadProbeRows, referenceOffset(128 * 16, 1, 16, std.math.maxInt(usize)));
}
