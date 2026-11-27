//! TFQ3 (3/4/5-bit 2D-AffineCoset + 2D FWHT-128) dense Qwen3.5 CUDA loader (BF16 and MLX Q8/Q4 residuals).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const b16 = @import("cuda_bf16_rows.zig");

const Engine = engine.Engine;
const Proj = engine.Proj;
const aligned = engine.aligned;
const bf16Of = core.bf16_round.bits;

/// TFQ3 dense kernels, uploaded `.tffpw` shard buffers, and BF16 row kernels for residuals.
pub const Kit = struct {
    kernels: cuda.tfq3.Kernels,
    store: cuda.tfq3.TfpnStore,
    b16: b16.Kernels,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, g: geometry.Geometry) !*Kit {
        const k = try gpa.create(Kit);
        errdefer gpa.destroy(k);
        k.kernels = try cuda.tfq3.Kernels.load(d);
        errdefer k.kernels.unload();
        const max_k = @max(g.hidden, @max(g.intermediate, @max(g.convWidth(), @max(g.attnWidth(), g.valueWidth()))));
        try k.kernels.reserve(512 * max_k, 64 * 34816 * 8, 0);
        k.b16 = try b16.Kernels.load(d);
        errdefer k.b16.unload();
        k.store = try cuda.tfq3.TfpnStore.loadDir(gpa, io, d, dir, "");
        return k;
    }

    pub fn deinit(k: *Kit, gpa: std.mem.Allocator) void {
        k.store.deinit();
        k.b16.unload();
        k.kernels.unload();
        gpa.destroy(k);
    }
};

const Sink = struct {
    e: *Engine,
    measure: bool,
    at: usize = 0,

    fn alloc(s: *Sink, bytes: usize) !u64 {
        if (!s.measure) return s.e.alloc(bytes);
        s.at = aligned(s.at) + bytes;
        return 0;
    }

    fn put(s: *Sink, bytes: []const u8) !u64 {
        const p = try s.alloc(bytes.len);
        if (!s.measure) try s.e.arena.upload(p - s.e.arena.ptr, bytes);
        return p;
    }
};

fn bf16f(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn fmtName(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

fn detectPrefix(ck: *core.Checkpoint) ![]const u8 {
    const candidates = [_][]const u8{ "model.", "model.language_model.", "language_model.model." };
    var buf: [96]u8 = undefined;
    for (candidates) |p| {
        if (ck.has(try fmtName(&buf, "{s}embed_tokens.weight", .{p}))) return p;
    }
    return error.UnexpectedTensor;
}

/// Auto-detects 0-centered (HF raw `1 + w`) vs 1-centered (MLX pre-shifted `w`) RMSNorm weights.
fn needsNormShift(ck: *core.Checkpoint, prefix: []const u8, hidden: usize) !bool {
    var buf: [160]u8 = undefined;
    const t = try ck.expect(try fmtName(&buf, "{s}layers.0.input_layernorm.weight", .{prefix}), .bf16, &.{hidden});
    const n = @min(hidden, 256);
    var sum: f64 = 0;
    for (0..n) |i| sum += bf16f(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    return (sum / @as(f64, @floatFromInt(n))) < 0.5;
}

fn normVec(s: *Sink, ck: *core.Checkpoint, shift: bool, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.expect(try fmtName(&buf, fmt, args), .bf16, &.{n});
    if (!shift or s.measure) return s.put(t.bytes);
    const out = try s.e.gpa.alloc(u16, n);
    defer s.e.gpa.free(out);
    for (out, 0..) |*o, i| o.* = bf16Of(1.0 + bf16f(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
    return s.put(std.mem.sliceAsBytes(out));
}

fn f32Vec(s: *Sink, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try fmtName(&buf, fmt, args));
    if (t.is(.f32, &.{n})) return s.put(t.bytes);
    if (!t.is(.bf16, &.{n})) return error.UnexpectedTensor;
    if (s.measure) return s.alloc(n * 4);
    const words = try s.e.gpa.alloc(u32, n);
    defer s.e.gpa.free(words);
    for (words, 0..) |*w, i| w.* = @as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16;
    return s.put(std.mem.sliceAsBytes(words));
}

fn bf16Vec(s: *Sink, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try fmtName(&buf, fmt, args));
    if (t.is(.bf16, &.{n})) return s.put(t.bytes);
    if (!t.is(.f32, &.{n})) return error.UnexpectedTensor;
    if (s.measure) return s.alloc(n * 2);
    const out = try s.e.gpa.alloc(u16, n);
    defer s.e.gpa.free(out);
    for (out, 0..) |*o, i| o.* = bf16Of(@bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)));
    return s.put(std.mem.sliceAsBytes(out));
}

fn conv1d(s: *Sink, ck: *core.Checkpoint, p: []const u8, conv: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try fmtName(&buf, "{s}linear_attn.conv1d.weight", .{p}));
    if (t.is(.bf16, &.{ conv, 1, geometry.taps }) or t.is(.bf16, &.{ conv, geometry.taps, 1 })) return s.put(t.bytes);
    return error.UnexpectedTensor;
}

/// Loads a small residual projection (`in_proj_a`/`in_proj_b`) as BF16 `.bf`, accepting BF16, F16, or MLX Q4/Q8.
fn smallProj(s: *Sink, ck: *core.Checkpoint, p: []const u8, mod: []const u8, n: usize, k: usize) !Proj {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try fmtName(&buf, "{s}{s}.weight", .{ p, mod }));
    if (t.is(.bf16, &.{ n, k })) return .{ .bf = .{ .w = try s.put(t.bytes), .n = n, .k = k } };
    if (s.measure) return .{ .bf = .{ .w = try s.alloc(n * k * 2), .n = n, .k = k } };
    const out = try s.e.gpa.alloc(u16, n * k);
    defer s.e.gpa.free(out);
    if (t.is(.f16, &.{ n, k })) {
        for (out, 0..) |*o, i| {
            const h: f16 = @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
            o.* = bf16Of(@floatCast(h));
        }
        return .{ .bf = .{ .w = try s.put(std.mem.sliceAsBytes(out)), .n = n, .k = k } };
    }
    if (t.dtype != .u32 or t.rank != 2 or t.dim(0) != n) return error.UnexpectedTensor;
    const bits = t.dim(1) * 32 / k;
    if ((bits != 4 and bits != 8) or t.dim(1) != k * bits / 32) return error.UnexpectedTensor;
    const sc = try ck.expect(try fmtName(&buf, "{s}{s}.scales", .{ p, mod }), .bf16, &.{ n, k / 64 });
    const bi = try ck.expect(try fmtName(&buf, "{s}{s}.biases", .{ p, mod }), .bf16, &.{ n, k / 64 });
    const mask: u32 = if (bits == 8) 0xff else 0xf;
    const per_word: usize = 32 / bits;
    const groups = k / 64;
    for (0..n) |r| {
        for (0..k) |c| {
            const w = std.mem.readInt(u32, t.bytes[(r * (k / per_word) + c / per_word) * 4 ..][0..4], .little);
            const code: f32 = @floatFromInt((w >> @intCast(bits * (c % per_word))) & mask);
            const gi = (r * groups + c / 64) * 2;
            const sv = bf16f(std.mem.readInt(u16, sc.bytes[gi..][0..2], .little));
            const bv = bf16f(std.mem.readInt(u16, bi.bytes[gi..][0..2], .little));
            out[r * k + c] = bf16Of(code * sv + bv);
        }
    }
    return .{ .bf = .{ .w = try s.put(std.mem.sliceAsBytes(out)), .n = n, .k = k } };
}

fn tfq3Proj(s: *Sink, name: []const u8, n: usize, k: usize) !Proj {
    if (s.measure) return .{ .tfq3 = .{ .words = 0, .scales = 0, .tables = 0, .n = @intCast(n), .k = @intCast(k), .bits = 4 } };
    const w = try s.e.tfq3.?.store.findChecked(name, n, k);
    return .{ .tfq3 = w };
}

fn loadEmbed(s: *Sink, ck: *core.Checkpoint, m: []const u8) !void {
    const e = s.e;
    const g = e.g;
    var buf: [200]u8 = undefined;
    const t = try ck.get(try fmtName(&buf, "{s}embed_tokens.weight", .{m}));
    if (t.is(.bf16, &.{ g.vocab, g.hidden })) {
        const ptr: u64 = if (e.host_embed) |h|
            (if (s.measure) 0 else try @import("cuda_host_embed.zig").rows(h, t.bytes))
        else
            try s.put(t.bytes);
        if (!s.measure) {
            e.embed_w = ptr;
            e.embed_s = 0;
            e.embed_b = 0;
            e.embed_bits = 16;
        }
        return;
    }
    if (t.dtype != .u32 or t.rank != 2 or t.dim(0) != g.vocab) return error.UnexpectedTensor;
    const bits = t.dim(1) * 32 / g.hidden;
    if ((bits != 4 and bits != 8) or t.dim(1) != g.hidden * bits / 32) return error.UnexpectedTensor;
    const sc = try ck.expect(try fmtName(&buf, "{s}embed_tokens.scales", .{m}), .bf16, &.{ g.vocab, g.hidden / 64 });
    const bi = try ck.expect(try fmtName(&buf, "{s}embed_tokens.biases", .{m}), .bf16, &.{ g.vocab, g.hidden / 64 });
    const pw = try s.put(t.bytes);
    const ps = try s.put(sc.bytes);
    const pb = try s.put(bi.bytes);
    if (!s.measure) {
        e.embed_w = pw;
        e.embed_s = ps;
        e.embed_b = pb;
        e.embed_bits = @intCast(bits);
    }
}

fn all(s: *Sink, ck: *core.Checkpoint) !void {
    const e = s.e;
    const g = e.g;
    const m = try detectPrefix(ck);
    const shift = try needsNormShift(ck, m, g.hidden);
    try loadEmbed(s, ck, m);
    const head = try tfq3Proj(s, "lm_head", g.vocab, g.hidden);
    const final_norm = try normVec(s, ck, shift, "{s}norm.weight", .{m}, g.hidden);
    if (!s.measure) {
        e.head = head;
        e.final_norm = final_norm;
    }
    const conv = g.convWidth();
    var scratch_layer: engine.Layer = undefined;
    var pbuf: [96]u8 = undefined;
    var nbuf: [128]u8 = undefined;
    for (0..g.layers) |i| {
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        const l = if (s.measure) &scratch_layer else &e.layers[i];
        l.input_norm = try normVec(s, ck, shift, "{s}input_layernorm.weight", .{p}, g.hidden);
        l.post_norm = try normVec(s, ck, shift, "{s}post_attention_layernorm.weight", .{p}, g.hidden);
        if (g.linear[i]) {
            l.mixer = .{ .delta = .{
                .qkv = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.linear_attn.in_proj_qkv", .{i}), conv, g.hidden),
                .z = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.linear_attn.in_proj_z", .{i}), g.valueWidth(), g.hidden),
                .a = try smallProj(s, ck, p, "linear_attn.in_proj_a", g.v_heads, g.hidden),
                .b = try smallProj(s, ck, p, "linear_attn.in_proj_b", g.v_heads, g.hidden),
                .out = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.linear_attn.out_proj", .{i}), g.hidden, g.valueWidth()),
                .conv = try conv1d(s, ck, p, conv),
                .norm = try normVec(s, ck, false, "{s}linear_attn.norm.weight", .{p}, geometry.dk),
                .a_log = try f32Vec(s, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                .dt_bias = try bf16Vec(s, ck, "{s}linear_attn.dt_bias", .{p}, g.v_heads),
            } };
        } else {
            l.mixer = .{ .attention = .{
                .q = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.self_attn.q_proj", .{i}), 2 * g.attnWidth(), g.hidden),
                .k = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.self_attn.k_proj", .{i}), g.kv_heads * geometry.head_dim, g.hidden),
                .v = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.self_attn.v_proj", .{i}), g.kv_heads * geometry.head_dim, g.hidden),
                .o = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.self_attn.o_proj", .{i}), g.hidden, g.attnWidth()),
                .q_norm = try normVec(s, ck, shift, "{s}self_attn.q_norm.weight", .{p}, geometry.head_dim),
                .k_norm = try normVec(s, ck, shift, "{s}self_attn.k_norm.weight", .{p}, geometry.head_dim),
            } };
        }
        l.mlp = .{ .dense = .{
            .gate = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.mlp.gate_proj", .{i}), g.intermediate, g.hidden),
            .up = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.mlp.up_proj", .{i}), g.intermediate, g.hidden),
            .down = try tfq3Proj(s, try fmtName(&nbuf, "layers.{d}.mlp.down_proj", .{i}), g.hidden, g.intermediate),
        } };
    }
}

/// Returns total bytes of TFQ3 `.tffpw` shards + tables on disk (for discrete GPU host-embed fit budgeting).
pub fn shardBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) u64 {
    const paths = cuda.tfq3_format.listShards(gpa, io, dir, "") catch return 0;
    defer cuda.tfq3_format.freePaths(gpa, paths);
    var total: u64 = cuda.tfq3_format.tables_file_bytes;
    for (paths) |p| {
        const st = std.Io.Dir.cwd().statFile(io, p, .{}) catch continue;
        total += st.size;
    }
    return total;
}

/// Whether the checkpoint's embedding is BF16 (eligible for mapped host memory when discrete VRAM is tight).
pub fn hasBf16Embed(ck: *core.Checkpoint, g: geometry.Geometry) bool {
    const m = detectPrefix(ck) catch return false;
    var buf: [200]u8 = undefined;
    const t = ck.get(fmtName(&buf, "{s}embed_tokens.weight", .{m}) catch return false) catch return false;
    return t.is(.bf16, &.{ g.vocab, g.hidden });
}

pub fn measure(e: *Engine, ck: *core.Checkpoint) !usize {
    var s: Sink = .{ .e = e, .measure = true };
    try all(&s, ck);
    return aligned(s.at) + 256;
}

pub fn load(e: *Engine, ck: *core.Checkpoint) !void {
    var s: Sink = .{ .e = e, .measure = false };
    try all(&s, ck);
    try e.stream.synchronize();
}
