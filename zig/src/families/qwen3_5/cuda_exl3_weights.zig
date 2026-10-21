//! ExLlamaV3 (EXL3, mul1 codebook) Qwen3.5-MoE checkpoints on the CUDA engine, read under their HF tensor names.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const geometry = @import("cuda_geometry.zig");
const engine = @import("cuda_engine.zig");
const b16 = @import("cuda_bf16_rows.zig");

const Engine = engine.Engine;
const Proj = engine.Proj;
const aligned = engine.aligned;
const ex = cuda.exl3_experts;

/// The mul1 codebook marker's value.
const mul1_magic: u32 = 0x83DCD12D;

/// The EXL3 kernels beside the engine's own: dense rows, the grouped experts and the bf16 rows.
pub const Kit = struct {
    dense: cuda.exl3.Dense,
    experts: ex.Experts,
    grouped: cuda.exl3_grouped.Grouped,
    b16: b16.Kernels,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, g: geometry.Geometry) !*Kit {
        const k = try gpa.create(Kit);
        errdefer gpa.destroy(k);
        k.dense = try cuda.exl3.Dense.load(d);
        errdefer k.dense.unload();
        k.experts = try ex.Experts.load(d);
        errdefer k.experts.unload();
        k.b16 = try b16.Kernels.load(d);
        errdefer k.b16.unload();
        k.grouped = .{ .d = d, .kernels = &k.experts };
        // the grouped experts' scratch grows on first use: allocate it before any graph capture, for the widest call
        const pairs = engine.max_rows * g.slots();
        const items = cuda.grouped.maxItems(pairs, g.experts + 1, 16);
        const up = try cuda.exl3_grouped.Grouped.bytes(g.hidden, g.intermediate, items, try ex.plan(g.hidden, g.intermediate, true), 2);
        const down = try cuda.exl3_grouped.Grouped.bytes(g.intermediate, g.hidden, items, try ex.plan(g.intermediate, g.hidden, false), 1);
        k.grouped.storage = try cuda.DeviceBuffer.alloc(d, @max(up, down));
        return k;
    }

    pub fn deinit(k: *Kit, gpa: std.mem.Allocator) void {
        k.grouped.deinit();
        k.b16.unload();
        k.experts.unload();
        k.dense.unload();
        gpa.destroy(k);
    }
};

/// Where tensors go: the engine's arena, or (measure) just counted with the same alignment.
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

fn name(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args);
}

fn bf16Of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// f16 values (as stored) to bf16, round to nearest even.
fn f16Rows(gpa: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const out = try gpa.alloc(u16, bytes.len / 2);
    for (out, 0..) |*o, i| {
        const h: f16 = @bitCast(std.mem.readInt(u16, bytes[2 * i ..][0..2], .little));
        o.* = bf16Of(@floatCast(h));
    }
    return out;
}

/// Zero-centred RMSNorm weights (HF: y = x * (1 + w)) as the engine's norm kernels take them (y = x * w): bf16(1 + w).
fn shifted(gpa: std.mem.Allocator, bytes: []const u8) ![]u16 {
    const out = try gpa.alloc(u16, bytes.len / 2);
    for (out, 0..) |*o, i| {
        const w: f32 = @bitCast(@as(u32, std.mem.readInt(u16, bytes[2 * i ..][0..2], .little)) << 16);
        o.* = bf16Of(1.0 + w);
    }
    return out;
}

fn plain(s: *Sink, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, dtype: core.checkpoint.DType, shape: []const usize) !u64 {
    var buf: [200]u8 = undefined;
    return s.put((try ck.expect(try name(&buf, fmt, args), dtype, shape)).bytes);
}

/// A norm weight, shifted by one (see `shifted`).
fn norm1(s: *Sink, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.expect(try name(&buf, fmt, args), .bf16, &.{n});
    const w = try shifted(s.e.gpa, t.bytes);
    defer s.e.gpa.free(w);
    return s.put(std.mem.sliceAsBytes(w));
}

/// A vector the kernels read as f32, stored as bf16 (A_log).
fn f32vec(s: *Sink, ck: *core.Checkpoint, comptime fmt: []const u8, args: anytype, n: usize) !u64 {
    var buf: [200]u8 = undefined;
    const t = try ck.get(try name(&buf, fmt, args));
    if (!t.is(.bf16, &.{n})) return error.UnexpectedTensor;
    const words = try s.e.gpa.alloc(u32, n);
    defer s.e.gpa.free(words);
    for (words, 0..) |*w, i| w.* = @as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16;
    return s.put(std.mem.sliceAsBytes(words));
}

/// One EXL3 group's tensors, checked: trellis [k/16][n/16][16 * bits] i16, the mul1 marker, fp16 suh [k] and svh [n].
const Group = struct { trellis: core.checkpoint.Tensor, suh: core.checkpoint.Tensor, svh: core.checkpoint.Tensor, halves: u32 };

fn group(ck: *core.Checkpoint, base: []const u8, n: usize, k: usize) !Group {
    var buf: [240]u8 = undefined;
    const t = try ck.get(try name(&buf, "{s}.trellis", .{base}));
    if (t.dtype != .i16 or t.rank != 3 or t.dim(0) != k / 16 or t.dim(1) != n / 16 or t.dim(2) % 8 != 0) {
        std.log.err("{s}: unexpected trellis {any}", .{ base, t.shape[0..t.rank] });
        return error.UnexpectedTensor;
    }
    const marker = try ck.get(try name(&buf, "{s}.mul1", .{base}));
    if (marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != mul1_magic) return error.UnsupportedQwenQuantization;
    return .{
        .trellis = t,
        .suh = try ck.expect(try name(&buf, "{s}.suh", .{base}), .f16, &.{k}),
        .svh = try ck.expect(try name(&buf, "{s}.svh", .{base}), .f16, &.{n}),
        .halves = @intCast(t.dim(2) / 8),
    };
}

/// An n-by-k EXL3 projection, its trellis kept as stored (Dense reads the stored layout).
fn dense(s: *Sink, kit: ?*Kit, ck: *core.Checkpoint, prefix: []const u8, mod: []const u8, n: usize, k: usize) !Proj {
    var buf: [200]u8 = undefined;
    const gr = try group(ck, try name(&buf, "{s}{s}", .{ prefix, mod }), n, k);
    const st = cuda.exl3.strides(k, n, gr.halves, false);
    const w: cuda.exl3.Weight = .{
        .trellis = try s.put(gr.trellis.bytes),
        .suh = try s.put(gr.suh.bytes),
        .svh = try s.put(gr.svh.bytes),
        .k = @intCast(k),
        .n = @intCast(n),
        .halves = gr.halves,
        .codebook = .mul1,
        .stride_k = st.k,
        .stride_nb = st.nb,
    };
    if (kit) |x| try x.dense.reserve(w);
    return .{ .x3 = w };
}

/// A small f16 matrix [n][k] (in_proj_a / in_proj_b) as bf16 for denseRows.
fn smallBf16(s: *Sink, ck: *core.Checkpoint, prefix: []const u8, mod: []const u8, n: usize, k: usize) !Proj {
    var buf: [200]u8 = undefined;
    const t = try ck.expect(try name(&buf, "{s}{s}.weight", .{ prefix, mod }), .f16, &.{ n, k });
    const rows = try f16Rows(s.e.gpa, t.bytes);
    defer s.e.gpa.free(rows);
    return .{ .bf = .{ .w = try s.put(std.mem.sliceAsBytes(rows)), .n = n, .k = k } };
}

/// The router as bf16 [experts + 1][hidden]: the gate's rows, then shared_expert_gate's (both stored f16).
fn router(s: *Sink, ck: *core.Checkpoint, prefix: []const u8) !u64 {
    const g = s.e.g;
    const rows = try s.e.gpa.alloc(u16, (g.experts + 1) * g.hidden);
    defer s.e.gpa.free(rows);
    var buf: [200]u8 = undefined;
    const a = try ck.expect(try name(&buf, "{s}mlp.gate.weight", .{prefix}), .f16, &.{ g.experts, g.hidden });
    const b = try ck.expect(try name(&buf, "{s}mlp.shared_expert_gate.weight", .{prefix}), .f16, &.{ 1, g.hidden });
    for (rows[0 .. g.experts * g.hidden], 0..) |*o, i| o.* = bf16Of(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, a.bytes[2 * i ..][0..2], .little)))));
    for (rows[g.experts * g.hidden ..], 0..) |*o, i| o.* = bf16Of(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, b.bytes[2 * i ..][0..2], .little)))));
    return s.put(std.mem.sliceAsBytes(rows));
}

/// A layer's experts for the grouped EXL3 kernels: trellis pointers and widths per expert, the shared one last.
fn experts(s: *Sink, ck: *core.Checkpoint, p: []const u8) !cuda.experts.Layer {
    const g = s.e.g;
    const gpa = s.e.gpa;
    const count = g.experts + 1;
    const mods = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    var ptrs: [3]u64 = undefined;
    var k2: [3]u64 = undefined;
    var suh: [3]u64 = undefined;
    var svh: [3]u64 = undefined;
    var buf: [200]u8 = undefined;
    for (mods, 0..) |mod, m| {
        const k = if (m == 2) g.intermediate else g.hidden;
        const n = if (m == 2) g.hidden else g.intermediate;
        const tp = try gpa.alloc(u64, count);
        defer gpa.free(tp);
        const widths = try gpa.alloc(i32, count);
        defer gpa.free(widths);
        const su = try gpa.alloc(u8, count * k * 2);
        defer gpa.free(su);
        const sv = try gpa.alloc(u8, count * n * 2);
        defer gpa.free(sv);
        for (0..count) |e| {
            const base = if (e < g.experts) try name(&buf, "{s}mlp.experts.{d}.{s}", .{ p, e, mod }) else try name(&buf, "{s}mlp.shared_expert.{s}", .{ p, mod });
            const gr = try group(ck, base, n, k);
            tp[e] = try s.put(gr.trellis.bytes);
            widths[e] = @intCast(gr.halves);
            @memcpy(su[e * k * 2 ..][0 .. k * 2], gr.suh.bytes);
            @memcpy(sv[e * n * 2 ..][0 .. n * 2], gr.svh.bytes);
        }
        ptrs[m] = try s.put(std.mem.sliceAsBytes(tp));
        k2[m] = try s.put(std.mem.sliceAsBytes(widths));
        suh[m] = try s.put(su);
        svh[m] = try s.put(sv);
    }
    const lay: ex.Layer = .{
        .gate = ptrs[0],
        .up = ptrs[1],
        .down = ptrs[2],
        .gate_k2 = k2[0],
        .up_k2 = k2[1],
        .down_k2 = k2[2],
        .suh_g = suh[0],
        .suh_u = suh[1],
        .svh_g = svh[0],
        .svh_u = svh[1],
        .suh_d = suh[2],
        .svh_d = svh[2],
        .experts = count,
        .dims = g.hidden,
        .width = g.intermediate,
        .codebook = .mul1,
        .limit = 0, // Qwen's SwiGLU is unclamped
        .act_mode = 1, // fp32 silu(gate) * up
    };
    return .{ .format = .exl3, .exl3 = lay, .up = 0, .down = 0, .width = g.intermediate, .dims = g.hidden, .experts = count, .swiglu = true };
}

/// The whole checkpoint through `s`; the engine's fields are set only in the real pass.
fn all(s: *Sink, ck: *core.Checkpoint) !void {
    const e = s.e;
    const g = e.g;
    const kit = e.kit;
    const m = "model.language_model.";
    // embeddings stay bf16 rows (embedRows); the head is an EXL3 projection
    const embed = try plain(s, ck, "{s}embed_tokens.weight", .{m}, .bf16, &.{ g.vocab, g.hidden });
    const head = try dense(s, kit, ck, "", "lm_head", g.vocab, g.hidden);
    const final = try norm1(s, ck, "{s}norm.weight", .{m}, g.hidden);
    if (!s.measure) {
        e.embed_w = embed;
        e.embed_s = 0;
        e.embed_b = 0;
        e.embed_bits = 16;
        e.head = head;
        e.final_norm = final;
    }
    const conv = g.convWidth();
    var scratch_layer: engine.Layer = undefined;
    for (0..g.layers) |i| {
        var pbuf: [96]u8 = undefined;
        const p = try std.fmt.bufPrint(&pbuf, "{s}layers.{d}.", .{ m, i });
        const l = if (s.measure) &scratch_layer else &e.layers[i];
        l.input_norm = try norm1(s, ck, "{s}input_layernorm.weight", .{p}, g.hidden);
        l.post_norm = try norm1(s, ck, "{s}post_attention_layernorm.weight", .{p}, g.hidden);
        if (g.linear[i]) {
            l.mixer = .{
                .delta = .{
                    .qkv = try dense(s, kit, ck, p, "linear_attn.in_proj_qkv", conv, g.hidden),
                    .z = try dense(s, kit, ck, p, "linear_attn.in_proj_z", g.valueWidth(), g.hidden),
                    .a = try smallBf16(s, ck, p, "linear_attn.in_proj_a", g.v_heads, g.hidden),
                    .b = try smallBf16(s, ck, p, "linear_attn.in_proj_b", g.v_heads, g.hidden),
                    .out = try dense(s, kit, ck, p, "linear_attn.out_proj", g.hidden, g.valueWidth()),
                    // HF's conv1d is [channels][1][taps]: the same bytes as the engine's [channels][taps]
                    .conv = try plain(s, ck, "{s}linear_attn.conv1d.weight", .{p}, .bf16, &.{ conv, 1, geometry.taps }),
                    .norm = try plain(s, ck, "{s}linear_attn.norm.weight", .{p}, .bf16, &.{geometry.dk}), // gated norm: unshifted
                    .a_log = try f32vec(s, ck, "{s}linear_attn.A_log", .{p}, g.v_heads),
                    .dt_bias = try plain(s, ck, "{s}linear_attn.dt_bias", .{p}, .bf16, &.{g.v_heads}),
                },
            };
        } else {
            l.mixer = .{ .attention = .{
                .q = try dense(s, kit, ck, p, "self_attn.q_proj", 2 * g.attnWidth(), g.hidden),
                .k = try dense(s, kit, ck, p, "self_attn.k_proj", g.kv_heads * geometry.head_dim, g.hidden),
                .v = try dense(s, kit, ck, p, "self_attn.v_proj", g.kv_heads * geometry.head_dim, g.hidden),
                .o = try dense(s, kit, ck, p, "self_attn.o_proj", g.hidden, g.attnWidth()),
                .q_norm = try norm1(s, ck, "{s}self_attn.q_norm.weight", .{p}, geometry.head_dim),
                .k_norm = try norm1(s, ck, "{s}self_attn.k_norm.weight", .{p}, geometry.head_dim),
            } };
        }
        l.mlp = .{ .moe = .{ .router = try router(s, ck, p), .experts = try experts(s, ck, p) } };
    }
}

/// The arena bytes the weights take (the scratch is added by the caller).
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
