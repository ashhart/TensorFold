//! Transcribe's audio encoder on CUDA: log-mel, the ASR and streaming speaker models by window, fusion, projection.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const Config = cfg.Config;
const Stack = cfg.Stack;
const weights = @import("weights.zig");
const Weights = weights.Weights;
const Tower = weights.Tower;
const Kernels = @import("kernels.zig").Kernels;
const speaker = @import("speaker.zig");
const Allocator = std.mem.Allocator;
const vit = cuda.vit;

pub const Kind = enum { bf16, f32 };

/// Where the check CLI takes each stage: `rows` x `cols` values `ld` apart at `at` (from `row0`), stream not synced.
pub const Dump = struct {
    ctx: *anyopaque,
    put: *const fn (ctx: *anyopaque, s: cuda.Stream, name: []const u8, at: u64, row0: usize, rows: usize, cols: usize, ld: usize, kind: Kind) anyerror!void,
    layers: bool = false, // every transformer layer's input, q k v and output too

    fn emit(d: ?Dump, s: cuda.Stream, name: []const u8, at: u64, row0: usize, rows: usize, cols: usize, kind: Kind) !void {
        const x = d orelse return;
        try x.put(x.ctx, s, name, at, row0, rows, cols, cols, kind);
    }
};

/// Samples staged to the device at once: the frames of one mel launch and their context.
const chunk_frames = 4096;
const chunk_samples = chunk_frames * Config.hop + 2 * Config.fft;
/// The most rows one transformer call holds: an ASR window, or the speaker cache, queue and a window.
const asr_rows = (Config.core + 2 * Config.context) / Config.stack;
const spk_rows = Config.cache_rows + Config.fifo_rows + asr_rows;

pub const Encoder = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    c: Config,
    w: Weights,
    k: Kernels,
    vk: vit.Kernels,
    at: vit.Attention,
    g: vit.Gemms,
    scratch: cuda.DeviceBuffer,
    stage: cuda.copy_source.Stage,
    b: Buffers,
    bytes: u64, // on the device: weights, scratch and the Linears' workspace

    const Buffers = struct { samples: u64, mel: u64, norm: u64, h: u64, x: u64, a: u64, wide: u64, acc: u64, seq: u64, cache: u64, fifo: u64, preds: u64 };

    /// The encoder from checkpoint `dir`: its config.json's audio side and its `perception.*` tensors.
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Encoder {
        const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
        defer gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const c = try Config.parse(gpa, text);
        if (c.asr.width > vit.norm_most or c.spk.width > vit.norm_most or c.spk.dim() != 64 or c.asr.dim() != 128) return error.UnsupportedAudio;
        var w = try Weights.load(gpa, io, d, dir, &c);
        errdefer w.deinit(gpa);
        var k = try Kernels.load(d);
        errdefer k.unload();
        var vk = try vit.Kernels.load(d);
        errdefer vk.unload();
        var at = try vit.Attention.load(d);
        errdefer at.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        const frames = Config.limit() * Config.stack;
        const row = @max(asr_rows * c.asr.width, spk_rows * c.spk.width, Config.stack * spk_rows * c.head_width);
        const wide = @max(asr_rows * @max(3 * c.asr.width, c.asr.ffn), spk_rows * @max(3 * c.spk.width, c.spk.ffn, Config.stack * c.head_width), asr_rows * c.output);
        const sizes = [_]usize{ 4 * chunk_samples, 2 * frames * Config.features, 2 * frames * Config.features, 2 * row, 2 * row, 2 * row, 2 * wide, 4 * wide, 2 * spk_rows * c.spk.width, 2 * Config.cache_rows * c.spk.width, 2 * Config.fifo_rows * c.spk.width, 4 * spk_rows * c.speakers };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var ptrs: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &ptrs) |sz, *p| {
            p.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        var stage: cuda.copy_source.Stage = .{};
        errdefer stage.deinit();
        _ = try stage.take(d, 4 * chunk_samples);
        const b: Buffers = .{ .samples = ptrs[0], .mel = ptrs[1], .norm = ptrs[2], .h = ptrs[3], .x = ptrs[4], .a = ptrs[5], .wide = ptrs[6], .acc = ptrs[7], .seq = ptrs[8], .cache = ptrs[9], .fifo = ptrs[10], .preds = ptrs[11] };
        return .{ .gpa = gpa, .d = d, .c = c, .w = w, .k = k, .vk = vk, .at = at, .g = g, .scratch = scratch, .stage = stage, .b = b, .bytes = w.bytes + total + g.workspace.len };
    }

    pub fn deinit(e: *Encoder) void {
        e.stage.deinit();
        e.scratch.free();
        e.g.deinit();
        e.at.unload();
        e.vk.unload();
        e.k.unload();
        e.w.deinit(e.gpa);
    }

    /// The rows a clip of n samples takes.
    pub fn rows(n: usize) usize {
        return Config.rows(n);
    }

    /// A clip's rows, bf16 [rows(samples.len)][output] at `out`; the rows written.
    pub fn encode(e: *Encoder, s: cuda.Stream, samples: []const f32, out: u64, dump: ?Dump) !usize {
        const n = samples.len;
        const valid = n / Config.hop;
        if (valid < 2) return error.AudioTooShort;
        if (n > Config.max_seconds * Config.rate) return error.AudioTooLong;
        const total = Config.rows(n);
        const padded = total * Config.stack; // frames the stacked rows read
        try e.features(s, samples, padded, valid);
        try Dump.emit(dump, s, "mel.raw", e.b.mel, 0, padded, Config.features, .bf16);
        try e.k.norm(s, e.b.mel, e.b.norm, padded, valid, Config.norm_eps);
        try Dump.emit(dump, s, "mel.asr_normalized_full", e.b.norm, 0, padded, Config.features, .bf16);
        var st: speaker.State = .{};
        var row0: usize = 0;
        const f = Config.frames(n);
        const count = cfg.windowCount(valid);
        for (0..count) |i| {
            const win = cfg.window(f, valid, i);
            const base = st.cache + st.fifo;
            try e.speakers(s, &st, win, i, i + 1 < count, dump);
            try e.asr(s, win, i, dump);
            try e.fuse(s, win, e.b.preds + 4 * (base + win.lo) * e.c.speakers, row0, out, dump);
            row0 += win.hi - win.lo;
        }
        std.debug.assert(row0 == total);
        return total;
    }

    /// The log-mel features of frames [0, padded) into mel, a chunk of samples staged at a time.
    fn features(e: *Encoder, s: cuda.Stream, samples: []const f32, padded: usize, valid: usize) !void {
        const w = &e.w;
        var first: usize = 0;
        while (first < padded) : (first += chunk_frames) {
            const frames = @min(chunk_frames, padded - first);
            const lo = @min(samples.len, (first * Config.hop) -| (Config.fft / 2 - (Config.fft - Config.window) / 2 + 1));
            const hi = @min(samples.len, (first + frames) * Config.hop + Config.fft);
            const bytes = std.mem.sliceAsBytes(samples[lo..hi]);
            if (bytes.len > 0) {
                @memcpy(try e.stage.take(e.d, bytes.len), bytes);
                try e.stage.copyTo(e.scratch, e.b.samples - e.scratch.ptr, s);
            }
            try e.k.mel(s, e.b.samples, lo, samples.len, w.window, w.fb, w.dft_cos, w.dft_sin, e.b.mel, first, frames, valid, Config.preemph, Config.log_guard);
        }
    }

    /// One window's speaker call ([cache; queue; window rows] to preds), then, unless it is the last, the next state.
    fn speakers(e: *Encoder, s: cuda.Stream, st: *speaker.State, win: cfg.Window, i: usize, carry: bool, dump: ?Dump) !void {
        const c = e.c;
        const b = e.b;
        const w = &e.w;
        const sw = c.spk.width;
        const row = 2 * sw;
        const base = st.cache + st.fifo;
        const n = base + win.rows;
        try e.copy(s, b.seq, b.cache, st.cache * row);
        try e.copy(s, b.seq + st.cache * row, b.fifo, st.fifo * row);
        try e.g.run(s, b.mel + 2 * win.start * Config.features, w.spk.pre, b.seq + base * row, win.rows, sw, Config.stack * Config.features, .bf16);
        try named(dump, s, "spk.call{d}.pre_encode_out", .{i}, b.seq + base * row, 0, win.rows, sw, sw, .bf16);
        try named(dump, s, "spk.call{d}.enc_in", .{i}, b.seq, 0, n, sw, sw, .bf16);
        try e.vk.layerNorm(s, b.seq, w.spk.embed, w.spk.embed_bias, b.h, n, sw, Config.eps);
        try named(dump, s, "spk.call{d}.embed_norm_out", .{i}, b.h, 0, n, sw, sw, .bf16);
        try e.transformer(s, &w.spk, c.spk, n, "spk_enc", dump);
        try e.vk.layerNorm(s, b.h, w.spk.final, w.spk.final_bias, b.x, n, sw, Config.eps);
        try named(dump, s, "spk.call{d}.final_norm_out", .{i}, b.x, 0, n, sw, sw, .bf16);
        try e.head(s, n, i, dump);
        try named(dump, s, "spk.w{d}.preds", .{i}, b.preds + 4 * (base + win.lo) * c.speakers, 0, win.hi - win.lo, c.speakers, c.speakers, .f32);
        if (!carry) return;
        // the state update reads the call's predictions on the host: a readback a window
        try s.synchronize();
        const p = try e.gpa.alloc(f32, n * c.speakers);
        defer e.gpa.free(p);
        try e.scratch.download(b.preds - e.scratch.ptr, std.mem.sliceAsBytes(p));
        const next = try speaker.update(e.gpa, &c, st, p, win.lo, win.hi);
        var r: usize = 0;
        while (r < next.cache_len) {
            var run: usize = 1;
            switch (next.cache[r]) {
                .sil => try e.copy(s, b.cache + r * row, w.sil, row),
                .row => |x| {
                    while (r + run < next.cache_len and std.meta.eql(next.cache[r + run], speaker.Pick{ .row = x + @as(u32, @intCast(run)) })) run += 1;
                    try e.copy(s, b.cache + r * row, b.seq + x * row, run * row);
                },
            }
            r += run;
        }
        for (next.fifo[0..next.fifo_len], 0..) |x, j| try e.copy(s, b.fifo + j * row, b.seq + x * row, row);
        try named(dump, s, "spk.w{d}.state_after.spkcache", .{i}, b.cache, 0, next.cache_len, sw, sw, .bf16);
        try named(dump, s, "spk.w{d}.state_after.fifo", .{i}, b.fifo, 0, next.fifo_len, sw, sw, .bf16);
    }

    /// The speaker head on the speaker model's output (x, n rows): each row's eight sub-rows to predictions in preds.
    fn head(e: *Encoder, s: cuda.Stream, n: usize, i: usize, dump: ?Dump) !void {
        const c = e.c;
        const b = e.b;
        const w = &e.w;
        const hw = c.head_width;
        const sub = Config.stack * n;
        try vit.linear(&e.g, &e.vk, s, b.acc, b.x, w.enc_proj, w.enc_proj_bias, b.a, n, hw, c.spk.width);
        try named(dump, s, "spk_head.encoder_proj.call{d}.out", .{i}, b.a, 0, n, hw, hw, .bf16);
        try e.k.taps(s, b.a, b.h, n, hw);
        try vit.linear(&e.g, &e.vk, s, b.acc, b.h, w.up, w.up_bias, b.wide, n, Config.stack * hw, 3 * hw);
        try named(dump, s, "spk_head.subpixel_upsample.call{d}.out", .{i}, b.wide, 0, n, Config.stack * hw, Config.stack * hw, .bf16);
        try e.k.relu(s, b.wide, sub * hw);
        try vit.linear(&e.g, &e.vk, s, b.acc, b.wide, w.hidden, w.hidden_bias, b.x, sub, hw, hw);
        try named(dump, s, "spk_head.first_hidden_to_hidden.call{d}.out", .{i}, b.x, 0, sub, hw, hw, .bf16);
        try e.k.relu(s, b.x, sub * hw);
        try vit.linear(&e.g, &e.vk, s, b.acc, b.x, w.spks, w.spks_bias, b.a, sub, c.speakers, hw);
        try named(dump, s, "spk_head.single_hidden_to_spks.call{d}.out", .{i}, b.a, 0, sub, c.speakers, c.speakers, .bf16);
        try e.k.preds(s, b.a, b.preds, n, c.speakers, Config.stack);
    }

    /// One window through the ASR transformer, its output in x.
    fn asr(e: *Encoder, s: cuda.Stream, win: cfg.Window, i: usize, dump: ?Dump) !void {
        const b = e.b;
        const w = &e.w;
        const aw = e.c.asr.width;
        try named(dump, s, "asr.w{d}.stacked", .{i}, b.norm + 2 * win.start * Config.features, 0, win.rows, Config.stack * Config.features, Config.stack * Config.features, .bf16);
        try e.g.run(s, b.norm + 2 * win.start * Config.features, w.asr.pre, b.x, win.rows, aw, Config.stack * Config.features, .bf16);
        try named(dump, s, "asr.w{d}.pre_encode_out", .{i}, b.x, 0, win.rows, aw, aw, .bf16);
        try e.vk.layerNorm(s, b.x, w.asr.embed, w.asr.embed_bias, b.h, win.rows, aw, Config.eps);
        try named(dump, s, "asr.w{d}.embed_norm_out", .{i}, b.h, 0, win.rows, aw, aw, .bf16);
        try e.transformer(s, &w.asr, e.c.asr, win.rows, "asr", dump);
        try e.vk.layerNorm(s, b.h, w.asr.final, w.asr.final_bias, b.x, win.rows, aw, Config.eps);
        try named(dump, s, "asr.w{d}.final_norm_out", .{i}, b.x, 0, win.rows, aw, aw, .bf16);
    }

    /// The pre-norm layers on h (n rows): attention with q and k turned, then the GELU feed-forward, each added back.
    fn transformer(e: *Encoder, s: cuda.Stream, t: *const Tower, cs: Stack, n: usize, comptime tag: []const u8, dump: ?Dump) !void {
        const b = e.b;
        const w = cs.width;
        const dim = cs.dim();
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(dim))));
        const deep: ?Dump = if (dump) |d| (if (d.layers) d else null) else null;
        for (t.layers, 0..) |l, i| {
            try named(deep, s, "layers." ++ tag ++ ".L{d:0>2}.in", .{i}, b.h, 0, n, w, w, .bf16);
            try e.vk.layerNorm(s, b.h, l.norm1, l.norm1_bias, b.x, n, w, Config.eps);
            try e.g.run(s, b.x, l.qkv, b.wide, n, 3 * w, w, .bf16);
            try named(deep, s, "layers." ++ tag ++ ".L{d:0>2}.attn.w_qkv_out", .{i}, b.wide, 0, n, 3 * w, 3 * w, .bf16);
            try e.k.qk(s, b.wide, l.qk_norm, t.cos, t.sin, n, cs.heads, dim, Config.eps);
            try named(deep, s, "layers." ++ tag ++ ".L{d:0>2}.attn.rope_out_q", .{i}, b.wide, 0, n, w, 3 * w, .bf16);
            try named(deep, s, "layers." ++ tag ++ ".L{d:0>2}.attn.rope_out_k", .{i}, b.wide + 2 * w, 0, n, w, 3 * w, .bf16);
            if (dim == 64) try e.k.attention(s, b.wide, b.a, n, cs.heads, scale) else try e.at.full(s, b.wide, b.a, n, cs.heads, dim, scale);
            try vit.linearAct(&e.g, &e.vk, s, b.acc, b.a, l.out, l.out_bias, b.x, n, w, w, b.h, b.h, .none);
            try e.vk.layerNorm(s, b.h, l.norm2, l.norm2_bias, b.x, n, w, Config.eps);
            try vit.linearAct(&e.g, &e.vk, s, b.acc, b.x, l.ff1, l.ff1_bias, b.wide, n, cs.ffn, w, null, b.wide, .gelu);
            try vit.linearAct(&e.g, &e.vk, s, b.acc, b.wide, l.ff2, l.ff2_bias, b.x, n, w, cs.ffn, b.h, b.h, .none);
            try named(deep, s, "layers." ++ tag ++ ".L{d:0>2}.out", .{i}, b.h, 0, n, w, w, .bf16);
        }
    }

    /// A window's kept ASR rows (x from row lo) normed, their speakers' activity added, projected into out from row0.
    fn fuse(e: *Encoder, s: cuda.Stream, win: cfg.Window, preds: u64, row0: usize, out: u64, dump: ?Dump) !void {
        const c = e.c;
        const b = e.b;
        const w = &e.w;
        const aw = c.asr.width;
        const kept = win.hi - win.lo;
        const in = b.x + 2 * win.lo * aw;
        try Dump.emit(dump, s, "fusion.asr_norm_in", in, row0, kept, aw, .bf16);
        try e.vk.layerNorm(s, in, w.asr_norm, w.asr_norm_bias, b.a, kept, aw, Config.eps);
        try Dump.emit(dump, s, "fusion.asr_norm_out", b.a, row0, kept, aw, .bf16);
        try e.k.fuse(s, b.a, preds, w.diar_norm, w.diar_norm_bias, w.kernel, b.h, if (dump != null) b.wide else null, kept, aw, c.threshold, Config.eps);
        try Dump.emit(dump, s, "fusion.spk_norm_out", b.wide, row0, kept, c.speakers, .bf16);
        try Dump.emit(dump, s, "pe.out.fused", b.h, row0, kept, aw, .bf16);
        const to = out + 2 * row0 * c.output;
        try vit.linear(&e.g, &e.vk, s, b.acc, b.h, w.proj, w.proj_bias, to, kept, c.output, aw);
        try Dump.emit(dump, s, "rows.final", to, row0, kept, c.output, .bf16);
    }

    fn copy(e: *Encoder, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
    }
};

fn named(dump: ?Dump, s: cuda.Stream, comptime fmt: []const u8, args: anytype, at: u64, row0: usize, rows_: usize, cols: usize, ld: usize, kind: Kind) !void {
    const d = dump orelse return;
    var buf: [96]u8 = undefined;
    try d.put(d.ctx, s, try std.fmt.bufPrint(&buf, fmt, args), at, row0, rows_, cols, ld, kind);
}

test "nemo audio encoder: scratch rows cover the largest calls" {
    try std.testing.expectEqual(@as(usize, 600), asr_rows);
    try std.testing.expectEqual(@as(usize, 828), spk_rows);
    try std.testing.expect(spk_rows <= @import("kernels.zig").attention_rows and spk_rows <= weights.rope_rows);
}
