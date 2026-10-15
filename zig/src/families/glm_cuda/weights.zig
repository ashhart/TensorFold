//! GLM-5.3-Flash's prepared weights for one rank of two as weights.py prepares them, cut exactly, uploaded to its GPU.

const std = @import("std");
const cuda = @import("cuda");
pub const api = @import("api.zig");
pub const kern = @import("kern.zig");
/// The forward and its engine API (tf-cuda-test glm-ref reaches it through this module).
pub const forward = @import("forward.zig");
pub const decode = @import("decode.zig");
pub const vision = @import("vision.zig");

const ckpt = @import("ckpt.zig");
const ex = @import("experts_load.zig");
const qpack = @import("qpack.zig");
const BASE = ckpt.BASE;
const WORLD = ckpt.WORLD;
const Ckpt = ckpt.Ckpt;
const fullName = ckpt.fullName;
const numelOf = ckpt.numelOf;
const dtypeOf = ckpt.dtypeOf;
const bf16ToF32 = ckpt.bf16ToF32;
const experts = ex.experts;
const expertsExl3 = ex.expertsExl3;
pub const quantize = qpack.quantize;

// -- split.py's rules: a checkpoint tensor's rank part (rows, the last axis' columns, or the whole tensor) -------

const Split = enum { rep, row, col };

fn splitOf(name: []const u8) Split {
    const ends = std.mem.endsWith;
    const has = struct {
        fn f(n: []const u8, frag: []const u8) bool {
            return std.mem.indexOf(u8, n, frag) != null;
        }
    }.f;
    if (has(name, ".mlp.experts.") or has(name, ".mlp.shared_experts.") or has(name, ".mlp.gate_proj.") or has(name, ".mlp.up_proj.") or has(name, ".mlp.down_proj."))
        return if (has(name, "down_proj.")) .col else .row;
    if (has(name, ".self_attn.indexer.")) return .rep;
    for ([_][]const u8{ ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.", "_conv1d.", ".self_attn.f_b_proj.", ".self_attn.g_b_proj.", ".self_attn.b_proj.", ".self_attn.q_b_proj.", ".self_attn.kv_b_proj." }) |f| if (has(name, f)) return .row;
    if (ends(u8, name, ".self_attn.A_log") or ends(u8, name, ".self_attn.dt_bias")) return .row;
    if (has(name, ".self_attn.o_proj.")) return .col;
    return .rep;
}

pub const Builder = struct {
    gpa: std.mem.Allocator,
    ck: *const Ckpt,
    dry: bool,
    rank: usize,
    store: ?*api.Store = null,
    d: ?*const cuda.Driver = null,
    bytes: u64 = 0,
    tensors: usize = 0,
    /// Dense linears (all but the indexer's) and the head packed to these bits (0: bf16 as stored): see ``quantize``.
    dense_bits: u8 = 0,
    head_bits: u8 = 0,
    /// TF_GLM_KDA_PROJ_BITS: the KDA stacked in-projection (1.3 GB a rank at 6 bits) at its own width (0: dense_bits).
    kda_proj_bits: u8 = 0,
    /// A packed copy of the head the MTP drafts read (0: they read the head).
    draft_head_bits: u8 = 0,
    /// The routed experts' groups as scales alone (biases -8 x scales, checked); TF_GLM_EXPERTS_SYM=0 keeps affine.
    experts_sym: bool = true,
    /// Development (TF_GLM_EXPERT_SIM_BITS=2 or 3): the experts re-quantized to fewer levels in their 4-bit words.
    expert_sim_bits: u8 = 0,
    /// TF_GLM_EXPERT_BITS=3: the experts re-quantized to 3-bit codes and packed as glm/experts_sym3's planes.
    experts3: bool = false,
    /// TF_GLM_EXPERT3_PROJ / TF_GLM_EXPERT_SIM_PROJ=down|up: re-quantize that projection alone (a sensitivity probe).
    sim_proj: u8 = 0, // 0: all, 1: down only, 2: gate/up only
    /// The checkpoint's routed experts are EXL3 (4-bit "mcg" trellises, every other weight BF16): read as stored.
    exl3: bool = false,
    exl3_shx: bool = true, // every layer's experts share one input rotation for gate and up (cleared by a layer that does not)
    suh_first: [2 * 4096]u8 = undefined, // the first expert's gate suh of the layer being read

    pub fn put(b: *Builder, name: []const u8, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var t: api.Tensor = .{ .ptr = 0, .dtype = dtype, .nd = @intCast(shape.len) };
        for (shape, 0..) |x, i| t.shape[i] = x;
        const n = t.bytes();
        b.bytes += n;
        b.tensors += 1;
        if (b.dry) return;
        const store = b.store.?;
        var buf = try cuda.DeviceBuffer.alloc(b.d.?, @max(n, 1));
        errdefer buf.free();
        if (n > 0) try buf.upload(0, data.?[0..n]);
        t.ptr = buf.ptr;
        try store.owned.append(store.gpa, buf);
        try store.put(name, t);
    }

    pub fn putf(b: *Builder, comptime fmt: []const u8, args: anytype, dtype: api.DType, shape: []const i64, data: ?[]const u8) !void {
        var nb: [256]u8 = undefined;
        try b.put(try std.fmt.bufPrint(&nb, fmt, args), dtype, shape, data);
    }

    /// A checkpoint tensor's rank part as stored: its dtype, rank shape and bytes (null in a dry run).
    const Part = struct { dtype: api.DType, shape: [4]i64, nd: usize, data: ?[]align(16) u8 };

    fn part(b: *Builder, full: []const u8, how: Split) !Part {
        const e = try b.ck.get(full);
        var p: Part = .{ .dtype = try dtypeOf(e.dtype), .shape = .{ 1, 1, 1, 1 }, .nd = e.shape.len, .data = null };
        for (e.shape, 0..) |x, i| p.shape[i] = x;
        const elem = api.dtypeBytes(p.dtype);
        const rows: usize = @intCast(p.shape[0]);
        const inner: usize = numelOf(e.shape) / rows;
        switch (how) {
            .rep => {
                if (!b.dry) p.data = try b.ck.readAll(b.gpa, full);
            },
            .row => {
                const per = rows / WORLD;
                p.shape[0] = @intCast(per);
                if (!b.dry) {
                    p.data = try b.gpa.alignedAlloc(u8, .@"16", per * inner * elem);
                    try b.ck.readAt(e, b.rank * per * inner * elem, p.data.?);
                }
            },
            .col => {
                if (e.shape.len != 2) return error.ColumnSplitNeeds2D;
                const k: usize = @intCast(e.shape[1]);
                const ck = k / WORLD;
                p.shape[1] = @intCast(ck);
                if (!b.dry) {
                    const all = try b.ck.readAll(b.gpa, full);
                    defer b.gpa.free(all);
                    p.data = try b.gpa.alignedAlloc(u8, .@"16", rows * ck * elem);
                    for (0..rows) |r| @memcpy(p.data.?[r * ck * elem ..][0 .. ck * elem], all[(r * k + b.rank * ck) * elem ..][0 .. ck * elem]);
                }
            },
        }
        return p;
    }

    fn free(b: *Builder, p: Part) void {
        if (p.data) |x| b.gpa.free(x);
    }

    /// One checkpoint tensor (its rank part) under ``out``, cast to fp32 when ``f32`` (bf16 -> fp32 is exact).
    fn tensor(b: *Builder, out: []const u8, full: []const u8, f32_out: bool) !void {
        const p = try b.part(full, splitOf(full));
        defer b.free(p);
        if (f32_out and p.dtype == .bf16) {
            const n = numelOf(p.shape[0..p.nd]);
            if (b.dry) return b.put(out, .f32, p.shape[0..p.nd], null);
            const f = try b.gpa.alloc(f32, n);
            defer b.gpa.free(f);
            const h = std.mem.bytesAsSlice(u16, p.data.?);
            for (f, 0..) |*o, i| o.* = bf16ToF32(h[i]);
            return b.put(out, .f32, p.shape[0..p.nd], std.mem.sliceAsBytes(f));
        }
        try b.put(out, p.dtype, p.shape[0..p.nd], p.data);
    }

    /// BF16 linears of the same K, their rank rows stacked in order -> ``out``.weight [sum n, k] (stack_b16).
    fn stack(b: *Builder, out: []const u8, fulls: []const []const u8) !void {
        var parts: [8]Part = undefined;
        var n: i64 = 0;
        var kk: i64 = 0;
        for (fulls, 0..) |f, i| {
            parts[i] = try b.part(f, splitOf(f));
            if (parts[i].dtype != .bf16 or parts[i].nd != 2) return error.NotBf16Matrix;
            n += parts[i].shape[0];
            kk = parts[i].shape[1];
        }
        defer for (parts[0..fulls.len]) |p| b.free(p);
        var nb: [256]u8 = undefined;
        var bits: u8 = if (std.mem.indexOf(u8, out, "index.") != null or @mod(kk, 64) != 0) 0 else b.dense_bits;
        if (b.kda_proj_bits != 0 and fulls.len == 6) bits = b.kda_proj_bits; // the KDA in-projection: 6 stacked linears
        const name = try std.fmt.bufPrint(&nb, "{s}.weight", .{out});
        if (b.dry and bits == 0) return b.put(name, .bf16, &.{ n, kk }, null);
        if (b.dry) return b.putPacked(out, bits, n, kk, null);
        const all = try b.gpa.alignedAlloc(u8, .@"16", @intCast(n * kk * 2));
        defer b.gpa.free(all);
        var at: usize = 0;
        for (parts[0..fulls.len]) |p| {
            @memcpy(all[at..][0..p.data.?.len], p.data.?);
            at += p.data.?.len;
        }
        if (bits != 0) return b.putPacked(out, bits, n, kk, all);
        try b.put(name, .bf16, &.{ n, kk }, all);
    }

    /// bf16 rows [n, kk] -> ``out``.qw (u8 [n, kk * bits / 8]) and ``out``.qs (fp16 [n, kk / 64]).
    fn putPacked(b: *Builder, out: []const u8, bits: u8, n: i64, kk: i64, rows: ?[]align(16) const u8) !void {
        const nu: usize = @intCast(n);
        const ku: usize = @intCast(kk);
        var nb: [256]u8 = undefined;
        var nb2: [256]u8 = undefined;
        const qn = try std.fmt.bufPrint(&nb, "{s}.qw", .{out});
        const sn = try std.fmt.bufPrint(&nb2, "{s}.qs", .{out});
        const wshape = [_]i64{ n, @divExact(kk * bits, 8) };
        const sshape = [_]i64{ n, @divExact(kk, 64) };
        if (rows == null) {
            try b.put(qn, .u8, &wshape, null);
            return b.put(sn, .f16, &sshape, null);
        }
        const words = try b.gpa.alloc(u32, nu * ku * bits / 32);
        defer b.gpa.free(words);
        const scales = try b.gpa.alloc(u16, nu * ku / 64);
        defer b.gpa.free(scales);
        quantize(bits, std.mem.bytesAsSlice(u16, rows.?), nu, ku, words, scales);
        try b.put(qn, .u8, &wshape, std.mem.sliceAsBytes(words));
        try b.put(sn, .f16, &sshape, std.mem.sliceAsBytes(scales));
    }

    fn linear(b: *Builder, out: []const u8, full: []const u8) !void {
        try b.stack(out, &.{full});
    }
};

// -- the layers, as weights.py's load() builds them (names: its weight tree) ------------------------------------

fn hc(b: *Builder, out: []const u8, li: usize, site: []const u8) !void {
    var o: [128]u8 = undefined;
    var f: [256]u8 = undefined;
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.fn", .{out}), try fullName(&f, "layers.{d}.hc_{s}_fn", .{ li, site }), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.base", .{out}), try fullName(&f, "layers.{d}.hc_{s}_base", .{ li, site }), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.scale", .{out}), try fullName(&f, "layers.{d}.hc_{s}_scale", .{ li, site }), true);
}

fn kda(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var n: [6][256]u8 = undefined;
    var names: [6][]const u8 = undefined;
    for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj" }, 0..) |p, j|
        names[j] = try fullName(&n[j], "layers.{d}.self_attn.{s}.weight", .{ i, p });
    try b.stack(try std.fmt.bufPrint(&o, "{s}.proj", .{out}), &names);
    var f: [256]u8 = undefined;
    try b.linear(try std.fmt.bufPrint(&o, "{s}.fb", .{out}), try fullName(&f, "layers.{d}.self_attn.f_b_proj.weight", .{i}));
    try b.linear(try std.fmt.bufPrint(&o, "{s}.gb", .{out}), try fullName(&f, "layers.{d}.self_attn.g_b_proj.weight", .{i}));
    // conv: q | k | v taps, each [rank channels, 1, 4], as [3 HL 128, 4]
    {
        var parts: [3]Builder.Part = undefined;
        for ("qkv", 0..) |x, j| {
            var cn: [256]u8 = undefined;
            parts[j] = try b.part(try fullName(&cn, "layers.{d}.self_attn.{c}_conv1d.weight", .{ i, x }), .row);
        }
        defer for (parts) |p| b.free(p);
        const ch = parts[0].shape[0] * 3;
        const shape = [_]i64{ ch, 4 };
        if (b.dry) {
            try b.put(try std.fmt.bufPrint(&o, "{s}.conv", .{out}), .bf16, &shape, null);
        } else {
            const all = try b.gpa.alloc(u8, @intCast(ch * 4 * 2));
            defer b.gpa.free(all);
            var at: usize = 0;
            for (parts) |p| {
                @memcpy(all[at..][0..p.data.?.len], p.data.?);
                at += p.data.?.len;
            }
            try b.put(try std.fmt.bufPrint(&o, "{s}.conv", .{out}), .bf16, &shape, all);
        }
    }
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.a_log", .{out}), try fullName(&f, "layers.{d}.self_attn.A_log", .{i}), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.dt_bias", .{out}), try fullName(&f, "layers.{d}.self_attn.dt_bias", .{i}), true);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.norm", .{out}), try fullName(&f, "layers.{d}.self_attn.o_norm.weight", .{i}), false);
    try b.linear(try std.fmt.bufPrint(&o, "{s}.o", .{out}), try fullName(&f, "layers.{d}.self_attn.o_proj.weight", .{i}));
}

fn dsa(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    var f2: [256]u8 = undefined;
    try b.stack(try std.fmt.bufPrint(&o, "{s}.proj", .{out}), &.{ try fullName(&f, "layers.{d}.self_attn.q_a_proj.weight", .{i}), try fullName(&f2, "layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{i}) });
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.q_norm", .{out}), try fullName(&f, "layers.{d}.self_attn.q_a_layernorm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.kv_norm", .{out}), try fullName(&f, "layers.{d}.self_attn.kv_a_layernorm.weight", .{i}), false);
    try b.linear(try std.fmt.bufPrint(&o, "{s}.q_b", .{out}), try fullName(&f, "layers.{d}.self_attn.q_b_proj.weight", .{i}));
    // kv_b per head: 256 key rows then 256 value rows -> absorb.wk, absorb.wv [HL, 256, 512] (the latent path's)
    {
        const p = try b.part(try fullName(&f, "layers.{d}.self_attn.kv_b_proj.weight", .{i}), .row);
        defer b.free(p);
        const HL: usize = @intCast(kern.HL);
        const lw: usize = @intCast(p.shape[1]);
        const shape = [_]i64{ kern.HL, kern.QK, @intCast(lw) };
        if (b.dry) {
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wk", .{out}), .bf16, &shape, null);
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wv", .{out}), .bf16, &shape, null);
        } else {
            const half: usize = @intCast(kern.QK);
            const wk = try b.gpa.alloc(u8, HL * half * lw * 2);
            defer b.gpa.free(wk);
            const wv = try b.gpa.alloc(u8, HL * half * lw * 2);
            defer b.gpa.free(wv);
            const row = lw * 2;
            for (0..HL) |h| {
                @memcpy(wk[h * half * row ..][0 .. half * row], p.data.?[(h * 2 * half) * row ..][0 .. half * row]);
                @memcpy(wv[h * half * row ..][0 .. half * row], p.data.?[(h * 2 * half + half) * row ..][0 .. half * row]);
            }
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wk", .{out}), .bf16, &shape, wk);
            try b.put(try std.fmt.bufPrint(&o, "{s}.absorb.wv", .{out}), .bf16, &shape, wv);
        }
    }
    try b.linear(try std.fmt.bufPrint(&o, "{s}.o", .{out}), try fullName(&f, "layers.{d}.self_attn.o_proj.weight", .{i}));
    try b.stack(try std.fmt.bufPrint(&o, "{s}.index.kw", .{out}), &.{ try fullName(&f, "layers.{d}.self_attn.indexer.wk.weight", .{i}), try fullName(&f2, "layers.{d}.self_attn.indexer.weights_proj.weight", .{i}) });
    try b.linear(try std.fmt.bufPrint(&o, "{s}.index.qb", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.wq_b.weight", .{i}));
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ln_w", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.k_norm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ln_b", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.k_norm.bias", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.gate", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.index_kpool_compress_gate", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.index.ape", .{out}), try fullName(&f, "layers.{d}.self_attn.indexer.index_kpool_compress_ape", .{i}), false);
}

fn mlp(b: *Builder, out: []const u8, prefix: []const u8) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    var f2: [256]u8 = undefined;
    try b.stack(try std.fmt.bufPrint(&o, "{s}.gu", .{out}), &.{ try std.fmt.bufPrint(&f, "{s}gate_proj.weight", .{prefix}), try std.fmt.bufPrint(&f2, "{s}up_proj.weight", .{prefix}) });
    try b.linear(try std.fmt.bufPrint(&o, "{s}.down", .{out}), try std.fmt.bufPrint(&f, "{s}down_proj.weight", .{prefix}));
}

fn moe(b: *Builder, out: []const u8, i: usize) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.router", .{out}), try fullName(&f, "layers.{d}.mlp.gate.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.bias", .{out}), try fullName(&f, "layers.{d}.mlp.gate.e_score_correction_bias", .{i}), true);
    if (b.exl3) try expertsExl3(b, try std.fmt.bufPrint(&o, "{s}.experts", .{out}), i) else try experts(b, try std.fmt.bufPrint(&o, "{s}.experts", .{out}), i);
    try mlp(b, try std.fmt.bufPrint(&o, "{s}.shared", .{out}), try fullName(&f, "layers.{d}.mlp.shared_experts.", .{i}));
}

/// One decoder layer (``plain``: the MTP head's, without hyper-connections) under ``out``.
fn layer(b: *Builder, out: []const u8, i: usize, kda_layer: bool, dense: bool, plain: bool) !void {
    var o: [160]u8 = undefined;
    var f: [256]u8 = undefined;
    if (!plain) {
        try hc(b, try std.fmt.bufPrint(&o, "{s}.attn_hc", .{out}), i, "attn");
        try hc(b, try std.fmt.bufPrint(&o, "{s}.ffn_hc", .{out}), i, "ffn");
    }
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.in_norm", .{out}), try fullName(&f, "layers.{d}.input_layernorm.weight", .{i}), false);
    try b.tensor(try std.fmt.bufPrint(&o, "{s}.post_norm", .{out}), try fullName(&f, "layers.{d}.post_attention_layernorm.weight", .{i}), false);
    if (kda_layer) try kda(b, try std.fmt.bufPrint(&o, "{s}.kda", .{out}), i) else try dsa(b, try std.fmt.bufPrint(&o, "{s}.dsa", .{out}), i);
    if (dense) try mlp(b, try std.fmt.bufPrint(&o, "{s}.mlp", .{out}), try fullName(&f, "layers.{d}.mlp.", .{i})) else try moe(b, try std.fmt.bufPrint(&o, "{s}.moe", .{out}), i);
}

/// The text config fields the preparation reads, checked against the dimensions the kernels take.
pub const Cfg = struct {
    exl3: bool = false, // EXL3 routed experts (else the converted affine ones)
    layers: usize,
    kinds: [64]bool = @splat(false), // true: a KDA layer
    dense: usize, // first_k_dense_replace
    mtp: bool,
    vocab: usize,

    pub fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8) !Cfg {
        const path = try std.fs.path.join(a, &.{ dir, "config.json" });
        const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 22));
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const q = v.object.get("quantization_config") orelse return error.NotAffineExperts;
        var exl3 = false;
        if (q.object.get("quant_method")) |m| if (std.mem.eql(u8, m.string, "exl3")) {
            const cb = q.object.get("codebook") orelse return error.NotExl3Mcg4;
            const bits = q.object.get("bits") orelse return error.NotExl3Mcg4;
            if (!std.mem.eql(u8, cb.string, "mcg") or bits.integer != 4) return error.NotExl3Mcg4;
            exl3 = true;
        };
        if (!exl3) {
            const fmt = q.object.get("format") orelse return error.NotAffineExperts;
            if (!std.mem.eql(u8, fmt.string, "glm-affine-experts-v1")) return error.NotAffineExperts;
        }
        const t = v.object.get("text_config").?.object;
        const n: usize = @intCast(t.get("num_hidden_layers").?.integer);
        var c: Cfg = .{ .exl3 = exl3, .layers = n, .dense = @intCast(t.get("first_k_dense_replace").?.integer), .mtp = (if (t.get("num_nextn_predict_layers")) |x| x.integer else 0) > 0, .vocab = @intCast(t.get("vocab_size").?.integer) };
        if (n > 64) return error.TooManyLayers;
        for (t.get("layer_types").?.array.items, 0..) |x, i| c.kinds[i] = std.mem.eql(u8, x.string, "linear_attention");
        if (t.get("hidden_size").?.integer != kern.D or t.get("n_routed_experts").?.integer != kern.EXPERTS or t.get("num_experts_per_tok").?.integer != kern.TOPK or
            t.get("num_attention_heads").?.integer != kern.HL * WORLD or t.get("kv_lora_rank").?.integer != kern.KV_LORA) return error.UnexpectedDimensions;
        return c;
    }
};

fn build(b: *Builder, c: *const Cfg) !void {
    var o: [64]u8 = undefined;
    for (0..c.layers) |i| try layer(b, try std.fmt.bufPrint(&o, "layers.{d}", .{i}), i, c.kinds[i], i < c.dense, false);
    var f: [256]u8 = undefined;
    try b.tensor("norm", try fullName(&f, "norm.weight", .{}), false);
    try b.tensor("embed", BASE ++ "embed_tokens.weight", false);
    // the head: this rank's vocabulary rows (packed to head_bits when set)
    {
        const e = try b.ck.get("lm_head.weight");
        const rows: usize = @intCast(e.shape[0]);
        const kk: usize = @intCast(e.shape[1]);
        const vl = rows / WORLD;
        const shape = [_]i64{ @intCast(vl), @intCast(kk) };
        if (b.dry) {
            if (b.head_bits != 0) try b.putPacked("head", b.head_bits, shape[0], shape[1], null) else try b.put("head.weight", .bf16, &shape, null);
            if (b.draft_head_bits != 0) try b.putPacked("draft_head", b.draft_head_bits, shape[0], shape[1], null);
        } else {
            const buf = try b.gpa.alignedAlloc(u8, .@"16", vl * kk * 2);
            defer b.gpa.free(buf);
            try b.ck.readAt(e, b.rank * vl * kk * 2, buf);
            if (b.head_bits != 0) try b.putPacked("head", b.head_bits, shape[0], shape[1], buf) else try b.put("head.weight", .bf16, &shape, buf);
            if (b.draft_head_bits != 0) try b.putPacked("draft_head", b.draft_head_bits, shape[0], shape[1], buf);
        }
    }
    if (c.mtp) {
        const m = c.layers;
        try b.tensor("mtp.enorm", try fullName(&f, "layers.{d}.enorm.weight", .{m}), false);
        try b.tensor("mtp.hnorm", try fullName(&f, "layers.{d}.hnorm.weight", .{m}), false);
        try b.linear("mtp.eh", try fullName(&f, "layers.{d}.eh_proj.weight", .{m}));
        try b.tensor("mtp.norm", try fullName(&f, "layers.{d}.shared_head.norm.weight", .{m}), false);
        try layer(b, "mtp.layer", m, false, false, true);
    }
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, ctx: *const api.Ctx, dir: []const u8, rank: u8) !api.Store {
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var store = api.Store.init(gpa);
    errdefer store.deinit();
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = false, .rank = rank, .store = &store, .d = ctx.d, .dense_bits = try envBits("TF_GLM_DENSE_BITS"), .head_bits = try envBits("TF_GLM_HEAD_BITS"), .draft_head_bits = try envBits("TF_GLM_DRAFT_HEAD_BITS"), .kda_proj_bits = try envBits("TF_GLM_KDA_PROJ_BITS"), .experts_sym = envSym(), .expert_sim_bits = envSimBits(), .experts3 = envBits3(), .sim_proj = envSimProj() };
    b.exl3 = c.exl3;
    if (c.exl3) b.experts_sym = false;
    try build(&b, &c);
    const vl = c.vocab / WORLD;
    const facts = [_]struct { []const u8, i64 }{
        .{ "rank", rank },                                               .{ "world", WORLD },                             .{ "vocab", @intCast(c.vocab) },           .{ "vocab_shard", @intCast(vl) },
        .{ "vocab_offset", @intCast(rank * vl) },                        .{ "layers", @intCast(c.layers) },               .{ "dense", @intCast(c.dense) },           .{ "mtp", @intFromBool(c.mtp) },
        .{ "device_bytes", @intCast(b.bytes) },                          .{ "experts_sym", @intFromBool(b.experts_sym) }, .{ "experts3", @intFromBool(b.experts3) }, .{ "exl3", @intFromBool(b.exl3) },
        .{ "exl3_shx", @intFromBool(b.exl3 and b.exl3_shx and !b.dry) },
    };
    for (facts) |fct| try store.ints.put(try gpa.dupe(u8, fct[0]), fct[1]);
    var kinds: [64]u8 = undefined;
    for (0..c.layers) |i| kinds[i] = @intFromBool(c.kinds[i]);
    try store.host.put(try gpa.dupe(u8, "kinds"), try gpa.dupe(u8, kinds[0..c.layers]));
    return store;
}

/// Device bytes this rank's prepared weights take: the same preparation's shapes, nothing read but headers.
pub fn deviceBytes(io: std.Io, dir: []const u8) !u64 {
    const gpa = std.heap.page_allocator;
    var ck = try Ckpt.open(gpa, io, dir);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const c = try Cfg.read(arena.allocator(), io, dir);
    var b: Builder = .{ .gpa = gpa, .ck = &ck, .dry = true, .rank = 0, .dense_bits = try envBits("TF_GLM_DENSE_BITS"), .head_bits = try envBits("TF_GLM_HEAD_BITS"), .draft_head_bits = try envBits("TF_GLM_DRAFT_HEAD_BITS"), .kda_proj_bits = try envBits("TF_GLM_KDA_PROJ_BITS"), .experts_sym = envSym(), .expert_sim_bits = envSimBits(), .experts3 = envBits3(), .sim_proj = envSimProj() };
    b.exl3 = c.exl3;
    try build(&b, &c);
    return b.bytes;
}

fn envBits3() bool {
    const v = std.c.getenv("TF_GLM_EXPERT_BITS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "3");
}

fn envSimProj() u8 {
    const v = std.c.getenv("TF_GLM_EXPERT_SIM_PROJ") orelse return 0;
    const t = std.mem.span(v);
    return if (std.mem.eql(u8, t, "down")) 1 else if (std.mem.eql(u8, t, "up")) 2 else 0;
}

fn envSimBits() u8 {
    const v = std.c.getenv("TF_GLM_EXPERT_SIM_BITS") orelse return 0;
    return std.fmt.parseInt(u8, std.mem.span(v), 10) catch 0;
}

fn envSym() bool {
    const v = std.c.getenv("TF_GLM_EXPERTS_SYM") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_GLM_DENSE_BITS / TF_GLM_HEAD_BITS: 4, 6 or 8 (unset or 0: bf16 as stored).
fn envBits(name: [*:0]const u8) !u8 {
    const v = std.c.getenv(name) orelse return 0;
    const bits = std.fmt.parseInt(u8, std.mem.span(v), 10) catch return error.BadPackedBits;
    if (bits != 0 and bits != 4 and bits != 6 and bits != 8) return error.BadPackedBits;
    return bits;
}
