//! NeMo's TransformerEncoder over one window on CUDA: the ASR encoder and the Sortformer's encoder are two of these.
const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig");
const W = @import("weights.zig");
const mel = @import("mel.zig");
const Kernels = @import("kernels.zig").Kernels;
const Dump = @import("dump.zig").Dump;
const vit = cuda.vit;
const Allocator = std.mem.Allocator;

/// Rows a window holds: the ASR encoder's 600 frames, or the Sortformer's cache, FIFO and window (188 + 40 + 600).
pub const max_rows = Config.spkcache_len + Config.fifo_len + (Config.window_frames + 2 * Config.context_frames) / Config.sub;
/// Columns a buffer holds: the widest model width, and the widest Linear (the ASR FFN, 4 d).
const widest = 1280;

pub const Runner = struct {
    k: Kernels,
    vk: vit.Kernels,
    g: vit.Gemms,
    scratch: cuda.DeviceBuffer,
    b: Buffers,
    tables: [2]Rope, // the ASR encoder's, then the Sortformer's

    /// Scratch rows, bf16 [rows, widest] unless noted.
    pub const Buffers = struct { in: u64, x: u64, h: u64, a: u64, qkv: u64, wide: u64, acc: u64 };
    const Rope = struct { cos: u64, sin: u64, rot: usize };

    pub fn init(gpa: Allocator, d: *const cuda.Driver, c: Config) !Runner {
        if (c.asr.d > widest or c.sf.d > widest) return error.UnsupportedAudioConfig;
        var k = try Kernels.load(d);
        errdefer k.unload();
        var vk = try vit.Kernels.load(d);
        errdefer vk.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        const r = max_rows;
        const sizes = [_]usize{
            2 * r * Config.sub * Config.mels, 2 * r * widest,     2 * r * widest,     2 * r * widest,
            2 * r * 3 * widest,               2 * r * 4 * widest, 4 * r * 4 * widest,
        };
        const tables = [2]Config.Encoder{ c.asr, c.sf };
        var total: usize = 0;
        for (sizes) |sz| total += std.mem.alignForward(usize, sz, 256);
        for (tables) |e| total += 2 * std.mem.alignForward(usize, 2 * r * e.rot(), 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, total);
        errdefer scratch.free();
        var off: usize = 0;
        var ptrs: [sizes.len]u64 = undefined;
        for (sizes, &ptrs) |sz, *p| {
            p.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        var ropes: [2]Rope = undefined;
        for (tables, &ropes) |e, *rope| {
            const cs, const sn = try ropeTables(gpa, e, r);
            defer gpa.free(cs);
            defer gpa.free(sn);
            const bytes = std.mem.alignForward(usize, 2 * r * e.rot(), 256);
            rope.* = .{ .cos = try scratch.at(off), .sin = try scratch.at(off + bytes), .rot = e.rot() };
            try scratch.upload(off, std.mem.sliceAsBytes(cs));
            try scratch.upload(off + bytes, std.mem.sliceAsBytes(sn));
            off += 2 * bytes;
        }
        return .{ .k = k, .vk = vk, .g = g, .scratch = scratch, .b = .{ .in = ptrs[0], .x = ptrs[1], .h = ptrs[2], .a = ptrs[3], .qkv = ptrs[4], .wide = ptrs[5], .acc = ptrs[6] }, .tables = ropes };
    }

    pub fn deinit(r: *Runner) void {
        r.scratch.free();
        r.g.deinit();
        r.vk.unload();
        r.k.unload();
    }

    /// FeatureStacking's projection: `rows` stacked frames at b.in [rows, 1024] to `out` [rows, d].
    pub fn stackIn(r: *Runner, s: cuda.Stream, st: *const W.Stack, e: Config.Encoder, rows: usize, out: u64) !void {
        try r.g.run(s, r.b.in, st.pre, out, rows, e.d, Config.sub * Config.mels, .bf16);
    }

    /// The encoder over `rows` rows at b.x, the first `len` valid, into `out`; `which` 0 is the ASR encoder's RoPE.
    pub fn run(r: *Runner, s: cuda.Stream, st: *const W.Stack, e: Config.Encoder, which: usize, rows: usize, len: usize, out: u64, dump: ?*Dump, tag: []const u8) !void {
        if (rows > max_rows or len > rows) return error.AudioWindow;
        const b = r.b;
        const d = e.d;
        const g = &r.g;
        const vk = &r.vk;
        const rope = r.tables[which];
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(e.dim()))));
        try vk.layerNorm(s, b.x, st.embed_norm, st.embed_norm_bias, b.x, rows, d, e.eps);
        if (dump) |dm| try dm.device(s, tag, ".embed_norm", b.x, rows, d);
        for (st.layers, 0..) |l, i| {
            try vk.layerNorm(s, b.x, l.norm1, l.norm1_bias, b.h, rows, d, e.eps);
            try g.run(s, b.h, l.qkv, b.qkv, rows, 3 * d, d, .bf16);
            const norm: ?[4]u64 = if (e.qk_norm) .{ l.q_norm, l.q_norm_bias, l.k_norm, l.k_norm_bias } else null;
            try r.k.qkRope(s, b.qkv, norm, rope.cos, rope.sin, rows, e.heads, e.dim(), rope.rot, e.eps);
            try r.k.attend(s, b.qkv, b.a, rows, e.heads, e.dim(), len, scale);
            try vit.linearAct(g, vk, s, b.acc, b.a, l.out, l.out_bias, b.h, rows, d, d, b.x, b.x, .none);
            try vk.layerNorm(s, b.x, l.norm2, l.norm2_bias, b.h, rows, d, e.eps);
            try vit.linearAct(g, vk, s, b.acc, b.h, l.fc1, l.fc1_bias, b.wide, rows, 4 * d, d, null, b.wide, .gelu);
            try vit.linearAct(g, vk, s, b.acc, b.wide, l.fc2, l.fc2_bias, b.h, rows, d, 4 * d, b.x, b.x, .none);
            if (dump) |dm| if (i == 0) try dm.device(s, tag, ".layer0", b.x, rows, d);
        }
        try vk.layerNorm(s, b.x, st.final, st.final_bias, out, rows, d, e.eps);
    }
};

/// RotaryPositionalEncoding.create_pe's tables for positions [0, rows): fp32 frequencies cast to bf16, [rows, rot].
fn ropeTables(gpa: Allocator, e: Config.Encoder, rows: usize) !struct { []u16, []u16 } {
    const rot = e.rot();
    const cs = try gpa.alloc(u16, rows * rot);
    errdefer gpa.free(cs);
    const sn = try gpa.alloc(u16, rows * rot);
    errdefer gpa.free(sn);
    const half = rot / 2;
    const base: f32 = @floatCast(e.base);
    for (0..half) |j| {
        // inv_freq = 1 / base^(2j / rot), each op in fp32
        const ex = @as(f32, @floatFromInt(2 * j)) / @as(f32, @floatFromInt(rot));
        const inv: f32 = 1.0 / @as(f32, @floatCast(std.math.pow(f64, base, ex)));
        for (0..rows) |p| {
            const f: f32 = @as(f32, @floatFromInt(p)) * inv;
            const c = mel.bf16(@floatCast(@cos(@as(f64, f))));
            const si = mel.bf16(@floatCast(@sin(@as(f64, f))));
            for ([2]usize{ j, j + half }) |col| {
                cs[p * rot + col] = c;
                sn[p * rot + col] = si;
            }
        }
    }
    return .{ cs, sn };
}
