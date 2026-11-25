//! The audio encoder's `perception.*` tensors as stored (bf16) in one allocation, plus DFT, rotary and speaker tables.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const Config = cfg.Config;
const Stack = cfg.Stack;
const Allocator = std.mem.Allocator;
const st = core.safetensors;

const prefix = "perception.";
const asr_at = "perception.encoder.asr_encoder.";
const spk_at = "perception.encoder.diarization_model.encoder.";
const head_at = "perception.encoder.diarization_model.sortformer_modules.";
/// The rotary table's rows: the most rows one speaker call holds (its cache, queue and a whole window).
pub const rope_rows = 1024;

pub const Layer = struct {
    norm1: u64,
    norm1_bias: u64,
    qkv: u64, // [3 width, width], no bias: q rows, then k, then v, head by head
    qk_norm: ?[4]u64, // q weight, q bias, k weight, k bias [dim] (the ASR stack's)
    out: u64,
    out_bias: u64,
    norm2: u64,
    norm2_bias: u64,
    ff1: u64,
    ff1_bias: u64,
    ff2: u64,
    ff2_bias: u64,
};

pub const Tower = struct {
    pre: u64, // pre_encode.proj [width, 8 * 128], no bias
    embed: u64,
    embed_bias: u64,
    final: u64,
    final_bias: u64,
    cos: u64, // bf16 [rope_rows][32]
    sin: u64,
    layers: []Layer,
};

pub const Weights = struct {
    memory: cuda.DeviceBuffer,
    window: u64, // bf16 [400]
    fb: u64, // bf16 [128][257]
    dft_cos: u64, // f32 [512]
    dft_sin: u64,
    asr: Tower,
    spk: Tower,
    enc_proj: u64, // [head, spk width]
    enc_proj_bias: u64,
    up: u64, // [8 head, head 3]: the kernel-3 upsampling convolution, (channel, tap) columns
    up_bias: u64,
    hidden: u64,
    hidden_bias: u64,
    spks: u64, // [8, head]
    spks_bias: u64,
    sil: u64, // learnable_sil_emb [spk width]
    asr_norm: u64,
    asr_norm_bias: u64,
    diar_norm: u64,
    diar_norm_bias: u64,
    kernel: u64, // bf16 [8][asr width]: speaker s's sinusoid at its position s
    proj: u64, // [output, asr width]
    proj_bias: u64,
    bytes: u64,

    /// `dir`'s perception tensors on the device, checked against `c`.
    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: *const Config) !Weights {
        var files = try open(gpa, io, dir);
        defer {
            for (files.items) |*f| f.close(io);
            files.deinit(gpa);
        }
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        var p: Planner = .{ .a = arena.allocator(), .files = files.items };
        var w: Weights = undefined;
        w.asr.layers = try gpa.alloc(Layer, c.asr.layers);
        errdefer gpa.free(w.asr.layers);
        w.spk.layers = try gpa.alloc(Layer, c.spk.layers);
        errdefer gpa.free(w.spk.layers);
        try p.plan(&w, c);
        w.memory = try cuda.DeviceBuffer.alloc(d, p.bytes);
        errdefer {
            d.synchronize() catch {};
            w.memory.free();
        }
        for (p.pieces.items) |piece| try w.memory.upload(piece.dst, piece.bytes);
        w.bytes = p.bytes;
        const base = w.memory.ptr;
        shift(&w, base);
        shift(&w.asr, base);
        shift(&w.spk, base);
        for (w.asr.layers) |*l| shiftLayer(l, base);
        for (w.spk.layers) |*l| shiftLayer(l, base);
        return w;
    }

    pub fn deinit(w: *Weights, gpa: Allocator) void {
        w.memory.free();
        gpa.free(w.asr.layers);
        gpa.free(w.spk.layers);
        w.* = undefined;
    }
};

const Piece = struct { bytes: []const u8, dst: u64 };

const Planner = struct {
    a: Allocator,
    files: []const st.File,
    pieces: std.ArrayList(Piece) = .empty,
    bytes: u64 = 0,
    taken: std.StringHashMapUnmanaged(void) = .empty,

    fn place(p: *Planner, bytes: []const u8) !u64 {
        const at = std.mem.alignForward(u64, p.bytes, 256);
        try p.pieces.append(p.a, .{ .bytes = bytes, .dst = at });
        p.bytes = at + bytes.len;
        return at;
    }

    /// `name` as stored (bf16 of `shape`), an offset from the allocation's start.
    fn take(p: *Planner, name: []const u8, shape: []const usize) !u64 {
        for (p.files) |*f| if (f.get(name)) |t| {
            if (!t.is(.bf16, shape)) {
                std.log.err("{s}: {t} {any}, expected bf16 {any}", .{ name, t.dtype, t.shape[0..t.rank], shape });
                return error.UnexpectedTensor;
            }
            try p.taken.put(p.a, try p.a.dupe(u8, name), {});
            return p.place(t.bytes);
        };
        std.log.err("checkpoint has no tensor {s}", .{name});
        return error.MissingTensor;
    }

    fn takef(p: *Planner, comptime fmt: []const u8, args: anytype, shape: []const usize) !u64 {
        var buf: [160]u8 = undefined;
        return p.take(try std.fmt.bufPrint(&buf, fmt, args), shape);
    }

    fn plan(p: *Planner, w: *Weights, c: *const Config) !void {
        w.window = try p.take(prefix ++ "preprocessor.featurizer.window", &.{Config.window});
        w.fb = try p.take(prefix ++ "preprocessor.featurizer.fb", &.{ 1, Config.features, Config.bins });
        const twiddles = try dft(p.a);
        w.dft_cos = try p.place(std.mem.sliceAsBytes(twiddles[0]));
        w.dft_sin = try p.place(std.mem.sliceAsBytes(twiddles[1]));
        try p.tower(&w.asr, asr_at, c.asr);
        try p.tower(&w.spk, spk_at, c.spk);
        const sw = c.spk.width;
        const hw = c.head_width;
        w.enc_proj = try p.take(head_at ++ "encoder_proj.weight", &.{ hw, sw });
        w.enc_proj_bias = try p.take(head_at ++ "encoder_proj.bias", &.{hw});
        w.up = try p.take(head_at ++ "subpixel_upsample.weight", &.{ Config.stack * hw, hw, 3 });
        w.up_bias = try p.take(head_at ++ "subpixel_upsample.bias", &.{Config.stack * hw});
        w.hidden = try p.take(head_at ++ "first_hidden_to_hidden.weight", &.{ hw, hw });
        w.hidden_bias = try p.take(head_at ++ "first_hidden_to_hidden.bias", &.{hw});
        w.spks = try p.take(head_at ++ "single_hidden_to_spks.weight", &.{ c.speakers, hw });
        w.spks_bias = try p.take(head_at ++ "single_hidden_to_spks.bias", &.{c.speakers});
        w.sil = try p.take(head_at ++ "learnable_sil_emb", &.{sw});
        w.asr_norm = try p.take(prefix ++ "encoder.asr_norm.weight", &.{c.asr.width});
        w.asr_norm_bias = try p.take(prefix ++ "encoder.asr_norm.bias", &.{c.asr.width});
        w.diar_norm = try p.take(prefix ++ "encoder.diar_norm.weight", &.{c.speakers});
        w.diar_norm_bias = try p.take(prefix ++ "encoder.diar_norm.bias", &.{c.speakers});
        w.kernel = try p.place(std.mem.sliceAsBytes(try speakerKernel(p.a, c.speakers, c.asr.width, c.kernel_scale)));
        w.proj = try p.take(prefix ++ "proj.weight", &.{ c.output, c.asr.width });
        w.proj_bias = try p.take(prefix ++ "proj.bias", &.{c.output});
        // left unread: the speaker branch's own featurizer and hidden_to_spks, which this serving mode never calls
        const unread = [_][]const u8{ "encoder.diarization_model.preprocessor.", "encoder.diarization_model.sortformer_modules.hidden_to_spks." };
        for (p.files) |*f| for (f.names.keys()) |k| {
            if (!std.mem.startsWith(u8, k, prefix) or p.taken.contains(k)) continue;
            const skip = for (unread) |u| {
                if (std.mem.startsWith(u8, k[prefix.len..], u)) break true;
            } else false;
            if (skip) continue;
            std.log.err("the audio encoder has a tensor these kernels do not read: {s}", .{k});
            return error.UnreadTensor;
        };
    }

    fn tower(p: *Planner, t: *Tower, comptime at: []const u8, s: Stack) !void {
        const n = s.width;
        t.pre = try p.take(at ++ "pre_encode.proj.weight", &.{ n, Config.stack * Config.features });
        t.embed = try p.take(at ++ "embed_norm.weight", &.{n});
        t.embed_bias = try p.take(at ++ "embed_norm.bias", &.{n});
        t.final = try p.take(at ++ "final_norm.weight", &.{n});
        t.final_bias = try p.take(at ++ "final_norm.bias", &.{n});
        const angles = try rope(p.a, s);
        t.cos = try p.place(std.mem.sliceAsBytes(angles[0]));
        t.sin = try p.place(std.mem.sliceAsBytes(angles[1]));
        const dim = s.dim();
        for (t.layers, 0..) |*l, i| {
            l.norm1 = try p.takef(at ++ "layers.{d}.norm1.weight", .{i}, &.{n});
            l.norm1_bias = try p.takef(at ++ "layers.{d}.norm1.bias", .{i}, &.{n});
            l.qkv = try p.takef(at ++ "layers.{d}.attn.w_qkv.weight", .{i}, &.{ 3 * n, n });
            l.qk_norm = if (s.qk_norm) .{
                try p.takef(at ++ "layers.{d}.attn.q_norm.weight", .{i}, &.{dim}),
                try p.takef(at ++ "layers.{d}.attn.q_norm.bias", .{i}, &.{dim}),
                try p.takef(at ++ "layers.{d}.attn.k_norm.weight", .{i}, &.{dim}),
                try p.takef(at ++ "layers.{d}.attn.k_norm.bias", .{i}, &.{dim}),
            } else null;
            l.out = try p.takef(at ++ "layers.{d}.attn.out_proj.weight", .{i}, &.{ n, n });
            l.out_bias = try p.takef(at ++ "layers.{d}.attn.out_proj.bias", .{i}, &.{n});
            l.norm2 = try p.takef(at ++ "layers.{d}.norm2.weight", .{i}, &.{n});
            l.norm2_bias = try p.takef(at ++ "layers.{d}.norm2.bias", .{i}, &.{n});
            l.ff1 = try p.takef(at ++ "layers.{d}.ffn.net.0.weight", .{i}, &.{ s.ffn, n });
            l.ff1_bias = try p.takef(at ++ "layers.{d}.ffn.net.0.bias", .{i}, &.{s.ffn});
            l.ff2 = try p.takef(at ++ "layers.{d}.ffn.net.3.weight", .{i}, &.{ n, s.ffn });
            l.ff2_bias = try p.takef(at ++ "layers.{d}.ffn.net.3.bias", .{i}, &.{n});
        }
    }
};

/// Every u64 field but the byte count moved by `base`.
fn shift(s: anytype, base: u64) void {
    const info = @typeInfo(@TypeOf(s.*)).@"struct";
    inline for (info.field_names, info.field_types) |name, T| if (T == u64 and comptime !std.mem.eql(u8, name, "bytes")) {
        @field(s, name) += base;
    };
}

fn shiftLayer(l: *Layer, base: u64) void {
    shift(l, base);
    if (l.qk_norm) |*q| for (q) |*x| {
        x.* += base;
    };
}

fn bf16(x: f64) u16 {
    const f: f32 = @floatCast(x);
    const u: u32 = @bitCast(f);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16); // round to nearest even (finite inputs)
}

/// cos and sin of 2 pi j / 512, j < 512, as f32.
pub fn dft(a: Allocator) ![2][]f32 {
    const c = try a.alloc(f32, Config.fft);
    const s = try a.alloc(f32, Config.fft);
    for (c, s, 0..) |*x, *y, j| {
        const t = 2.0 * std.math.pi * @as(f64, @floatFromInt(j)) / Config.fft;
        x.* = @floatCast(@cos(t));
        y.* = @floatCast(@sin(t));
    }
    return .{ c, s };
}

/// bf16 cos and sin [rope_rows][rot / 2] of row p's angle p base^(-2i / rot), i < rot / 2.
pub fn rope(a: Allocator, s: Stack) ![2][]u16 {
    const half = s.rot / 2;
    const c = try a.alloc(u16, rope_rows * half);
    const n = try a.alloc(u16, rope_rows * half);
    for (0..rope_rows) |p| for (0..half) |i| {
        const inv = std.math.pow(f64, s.base, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(s.rot)));
        const t = @as(f64, @floatFromInt(p)) * inv;
        c[p * half + i] = bf16(@cos(t));
        n[p * half + i] = bf16(@sin(t));
    };
    return .{ c, n };
}

/// bf16 [speakers][width]: scale sin(s w_i) at column 2i and scale cos(s w_i) at 2i + 1, w_i = 10000^(-2i / width).
pub fn speakerKernel(a: Allocator, speakers: usize, width: usize, scale: f32) ![]u16 {
    const k = try a.alloc(u16, speakers * width);
    for (0..speakers) |s| for (0..width / 2) |i| {
        const t = @as(f64, @floatFromInt(s)) * std.math.pow(f64, 10000.0, -2.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(width)));
        k[s * width + 2 * i] = bf16(scale * @sin(t));
        k[s * width + 2 * i + 1] = bf16(scale * @cos(t));
    };
    return k;
}

/// The shards with a perception tensor, each read for its perception names alone.
fn open(gpa: Allocator, io: std.Io, dir: []const u8) !std.ArrayList(st.File) {
    var out: std.ArrayList(st.File) = .empty;
    errdefer {
        for (out.items) |*f| f.close(io);
        out.deinit(gpa);
    }
    const paths = try core.checkpoint.shardFiles(gpa, io, dir);
    defer core.checkpoint.freeShardFiles(gpa, paths);
    for (paths) |path| {
        var f = try st.File.openPrefix(gpa, io, path, prefix);
        if (f.names.count() == 0) {
            f.close(io);
            continue;
        }
        try out.append(gpa, f);
    }
    if (out.items.len == 0) return error.NoAudioEncoder;
    return out;
}

test "nemo audio tables: bf16 rounding, the speaker kernel's first rows" {
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.00390625)); // a tie rounds to even
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(1.01171875));
    const a = std.testing.allocator;
    const k = try speakerKernel(a, 8, 1280, 1.0);
    defer a.free(k);
    try std.testing.expectEqual(@as(u16, 0), k[0]); // sin 0
    try std.testing.expectEqual(@as(u16, 0x3f80), k[1]); // cos 0
    try std.testing.expectEqual(bf16(@sin(1.0)), k[1280]);
    const r = try rope(a, .{ .width = 512, .heads = 8, .layers = 1, .ffn = 1, .rot = 64, .qk_norm = false, .base = 10000 });
    defer a.free(r[0]);
    defer a.free(r[1]);
    try std.testing.expectEqual(bf16(@cos(3.0)), r[0][3 * 32]);
    try std.testing.expectEqual(bf16(@sin(3.0 * std.math.pow(f64, 10000.0, -2.0 / 64.0))), r[1][3 * 32 + 1]);
}
