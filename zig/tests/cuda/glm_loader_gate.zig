//! `glm-loader-gate DIR`: each rank's uploads at TP 1, 2 and 3, byte for byte against the checkpoint cut on the host.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");

const DType = core.checkpoint.DType;
const Tensor = core.checkpoint.Tensor;
const Config = glm.config.Config;
const P = "model.language_model.";

/// A piece of an upload: tensor `name`'s outer rows [lo, hi) (a vector's elements), or a matrix's columns [lo, hi).
const Cut = struct { name: []const u8, cols: bool = false, lo: usize, hi: usize, to: DType = .bf16 };

/// What an upload is held against: its cut as stored, a host packer's bytes from the gate's own cut, or nothing.
const Want = union(enum) { cut, made: []u8, skip: []const u8 };

/// One upload's pieces as the Python family's weights.load cuts them, read from a checkpoint apart from the loader's.
const Plan = struct {
    a: std.mem.Allocator,
    gpa: std.mem.Allocator, // a packed upload's expected bytes, freed once compared
    ck: *core.Checkpoint,
    cuts: std.ArrayList(Cut) = .empty,

    fn rows(p: *Plan, comptime fmt: []const u8, args: anytype, lo: usize, hi: usize, to: DType) !void {
        try p.cuts.append(p.a, .{ .name = try std.fmt.allocPrint(p.a, fmt, args), .lo = lo, .hi = hi, .to = to });
    }

    fn whole(p: *Plan, comptime fmt: []const u8, args: anytype, to: DType) !void {
        const name = try std.fmt.allocPrint(p.a, fmt, args);
        try p.cuts.append(p.a, .{ .name = name, .lo = 0, .hi = (try p.ck.get(name)).dim(0), .to = to });
    }

    /// A hidden-size norm as the loader keeps it: float32 when stored so (FP4-Local's), else bf16.
    fn norm(p: *Plan, comptime fmt: []const u8, args: anytype) !void {
        const name = try std.fmt.allocPrint(p.a, fmt, args);
        const t = try p.ck.get(name);
        try p.cuts.append(p.a, .{ .name = name, .lo = 0, .hi = t.dim(0), .to = if (t.dtype == .f32) .f32 else .bf16 });
    }

    fn cols(p: *Plan, comptime fmt: []const u8, args: anytype, lo: usize, hi: usize) !void {
        try p.cuts.append(p.a, .{ .name = try std.fmt.allocPrint(p.a, fmt, args), .cols = true, .lo = lo, .hi = hi });
    }

    /// The outer rows of tensor `fmt`, for a cut by heads.
    fn outer(p: *Plan, comptime fmt: []const u8, args: anytype) !usize {
        return (try p.ck.get(try std.fmt.allocPrint(p.a, fmt, args))).dim(0);
    }
};

fn eql(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

/// The MTP head's uploads (`mtp.{what}`) and the tensors under layer `layers` they come from.
const mtp_sources = [_][2][]const u8{ .{ "enorm", "enorm" }, .{ "hnorm", "hnorm" }, .{ "eh", "eh_proj" }, .{ "norm", "shared_head.norm" } };

/// Upload `name`'s pieces at share `s` into `p`, or the bytes a host packer makes of them, or why it is left out.
fn plan(p: *Plan, c: *const Config, s: glm.shard.Share, name: []const u8) !Want {
    if (eql(name, "embed")) {
        try p.whole(P ++ "embed_tokens.weight", .{}, .bf16);
    } else if (eql(name, "norm")) {
        try p.norm(P ++ "norm.weight", .{});
    } else if (eql(name, "head")) {
        try p.rows("lm_head.weight", .{}, s.vocab.lo, s.vocab.hi, .bf16);
    } else if (eql(name, "head.codes") or eql(name, "head.scales")) {
        return laneMade(p, &.{.{ .module = "lm_head", .rows = .{ s.vocab.lo, s.vocab.hi } }}, name);
    } else if (std.mem.startsWith(u8, name, "draft_head.")) {
        return .{ .skip = "the GPU quantizer's 4-bit tiles of the head's rows" };
    } else if (std.mem.startsWith(u8, name, "mtp.")) {
        for (mtp_sources) |m| if (eql(name[4..], m[0])) {
            try p.whole(P ++ "layers.{d}.{s}.weight", .{ c.layers, m[1] }, .bf16);
            return .cut;
        };
        return error.UnknownUpload;
    } else if (std.mem.startsWith(u8, name, "layers.")) {
        const dot = std.mem.indexOfScalarPos(u8, name, "layers.".len, '.') orelse return error.UnknownUpload;
        const i = std.fmt.parseInt(u32, name["layers.".len..dot], 10) catch return error.UnknownUpload;
        return layer(p, c, s, i, name[dot + 1 ..]);
    } else return error.UnknownUpload;
    return .cut;
}

/// Layer `i`'s upload `what`, as `plan`.
fn layer(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const L = P ++ "layers.{d}.";
    if (eql(what, "in_norm")) {
        try p.norm(L ++ "input_layernorm.weight", .{i});
        return .cut;
    }
    if (eql(what, "post_norm")) {
        try p.norm(L ++ "post_attention_layernorm.weight", .{i});
        return .cut;
    }
    inline for (.{ "attn", "ffn" }) |site| {
        if (eql(what, site ++ "_hc.fn")) {
            try p.whole(L ++ "hc_" ++ site ++ "_fn", .{i}, .bf16);
            return .cut;
        }
        inline for (.{ "base", "scale" }) |x| if (eql(what, site ++ "_hc." ++ x)) {
            try p.whole(L ++ "hc_" ++ site ++ "_" ++ x, .{i}, .f32);
            return .cut;
        };
    }
    const lane = std.mem.endsWith(u8, what, ".codes") or std.mem.endsWith(u8, what, ".scales");
    // the lane's projections first: a dense MXFP8 MLP's mlp.gu. and mlp.down. before the NVFP4 one's .scales
    for ([_][]const u8{ "dsa.", "moe.shared.", "kda.", "mlp.gu.", "mlp.down." }) |x| {
        if (lane and std.mem.startsWith(u8, what, x)) return laneUpload(p, c, s, i, what);
    }
    if (std.mem.startsWith(u8, what, "kda.")) return kda(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "dsa.")) return dsa(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "moe.")) return moe(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "mlp.") and (std.mem.endsWith(u8, what, ".words") or std.mem.endsWith(u8, what, ".scales"))) {
        return denseMade(p, c, s, i, what);
    }
    return error.UnknownUpload;
}

/// KDA: [q | k | v | f_a | g_a | b], q k v b by the rank's heads; f_b, g_b, taps, dt_bias by heads; o by columns.
fn kda(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const at = P ++ "layers.{d}.self_attn.";
    const h = s.kda;
    const dk: usize = c.kda_dim;
    if (eql(what, "proj") or eql(what, "rest")) { // rest: a split stack's bf16 [f_a | g_a | b], its q k v on the lane
        if (eql(what, "proj")) inline for (.{ "q", "k", "v" }) |x| try p.rows(at ++ x ++ "_proj.weight", .{i}, h.lo * dk, h.hi * dk, .bf16);
        inline for (.{ "f_a", "g_a" }) |x| try p.whole(at ++ x ++ "_proj.weight", .{i}, .bf16);
        try p.rows(at ++ "b_proj.weight", .{i}, h.lo, h.hi, .bf16);
    } else if (eql(what, "fb") or eql(what, "gb")) {
        try p.rows(at ++ "{s}_b_proj.weight", .{ i, what[0..1] }, h.lo * dk, h.hi * dk, .bf16);
    } else if (eql(what, "conv")) {
        inline for (.{ "q", "k", "v" }) |x| try p.rows(at ++ x ++ "_conv1d.weight", .{i}, h.lo * dk, h.hi * dk, .f32);
    } else if (eql(what, "a_log")) {
        try p.rows(at ++ "A_log", .{i}, h.lo, h.hi, .f32);
    } else if (eql(what, "dt_bias")) {
        try p.rows(at ++ "dt_bias", .{i}, h.lo * dk, h.hi * dk, .f32);
    } else if (eql(what, "norm")) {
        try p.whole(at ++ "o_norm.weight", .{i}, .bf16);
    } else if (eql(what, "o")) {
        try p.cols(at ++ "o_proj.weight", .{i}, h.lo * dk, h.hi * dk);
    } else return error.UnknownUpload;
    return .cut;
}

/// Latent attention: [q_a | kv_a] whole; q_b, kv_b's keys and values by heads (rows from the tensors); o by columns.
fn dsa(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const at = P ++ "layers.{d}.self_attn.";
    const h = s.mla;
    const dv: usize = c.v_dim;
    if (eql(what, "proj")) {
        try p.whole(at ++ "q_a_proj.weight", .{i}, .bf16);
        try p.whole(at ++ "kv_a_proj_with_mqa.weight", .{i}, .bf16);
    } else if (eql(what, "q_norm")) {
        try p.whole(at ++ "q_a_layernorm.weight", .{i}, .bf16);
    } else if (eql(what, "kv_norm")) {
        try p.whole(at ++ "kv_a_layernorm.weight", .{i}, .bf16);
    } else if (eql(what, "q_b")) {
        const per = try p.outer(at ++ "q_b_proj.weight", .{i}) / c.mla_heads;
        try p.rows(at ++ "q_b_proj.weight", .{i}, h.lo * per, h.hi * per, .bf16);
    } else if (eql(what, "absorb.wk") or eql(what, "absorb.wv")) {
        const per = try p.outer(at ++ "kv_b_proj.weight", .{i}) / c.mla_heads; // a head's keys, then its values
        const keys = per - dv;
        const kv = try std.fmt.allocPrint(p.a, at ++ "kv_b_proj", .{i});
        const mx = (try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight", .{kv}))).dtype == .f8_e4m3; // MTP's MXFP8
        var wide: std.ArrayList(u8) = .empty;
        errdefer wide.deinit(p.gpa);
        for (h.lo..h.hi) |head| {
            const lo = if (eql(what, "absorb.wk")) head * per else head * per + keys;
            const hi = lo + (if (eql(what, "absorb.wk")) keys else dv);
            if (mx) try widen(p, kv, lo, hi, &wide) else try p.rows(at ++ "kv_b_proj.weight", .{i}, lo, hi, .bf16);
        }
        if (mx) return .{ .made = try wide.toOwnedSlice(p.gpa) };
    } else if (eql(what, "o")) {
        try p.cols(at ++ "o_proj.weight", .{i}, h.lo * dv, h.hi * dv);
    } else if (eql(what, "index.kw")) {
        try p.whole(at ++ "indexer.wk.weight", .{i}, .bf16);
        try p.whole(at ++ "indexer.weights_proj.weight", .{i}, .bf16);
    } else {
        const index = [_][2][]const u8{ .{ "index.qb", "wq_b.weight" }, .{ "index.ln_w", "k_norm.weight" }, .{ "index.ln_b", "k_norm.bias" }, .{ "index.gate", "index_kpool_compress_gate" }, .{ "index.ape", "index_kpool_compress_ape" } };
        for (index) |x| if (eql(what, x[0])) {
            try p.whole(at ++ "indexer.{s}", .{ i, x[1] }, .bf16);
            return .cut;
        };
        return error.UnknownUpload;
    }
    return .cut;
}

/// MXFP8 `module`'s rows [lo, hi) widened to bf16 bytes onto `out`, as the loader widens an MTP layer's kv_b.
fn widen(p: *Plan, module: []const u8, lo: usize, hi: usize, out: *std.ArrayList(u8)) !void {
    const MX = glm.weights.formats.mxfp8;
    const t = try MX.Mxfp8.read(p.ck, module);
    const rows = try p.gpa.alloc(u16, (hi - lo) * t.k);
    defer p.gpa.free(rows);
    try MX.bf16Rows(t, lo, hi, rows);
    try out.appendSlice(p.gpa, std.mem.sliceAsBytes(rows));
}

/// MoE: router and bias whole, the experts' tiles and global scales, shared gate and up by rows, down by columns.
fn moe(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const at = P ++ "layers.{d}.mlp.";
    if (eql(what, "router")) {
        try p.whole(at ++ "gate.weight", .{i}, .bf16);
    } else if (eql(what, "bias")) {
        try p.whole(at ++ "gate.e_score_correction_bias", .{i}, .f32);
    } else if (eql(what, "experts.up") or eql(what, "experts.down")) {
        return expertsMade(p, c, s, i, eql(what, "experts.up"));
    } else if (eql(what, "experts.up_scale") or eql(what, "experts.down_scale")) {
        // the MTP layer's bf16 experts take the GPU quantizer's scales; W4A16_NVFP4 ones (FP4-Local's) keep their own
        if (i == c.layers and !p.ck.has(try std.fmt.allocPrint(p.a, at ++ "experts.0.gate_proj.weight_scale_2", .{i}))) {
            return .{ .skip = "the GPU quantizer's, from the MTP layer's bf16 experts" };
        }
        for (0..c.experts) |e| {
            const projs: []const []const u8 = if (eql(what, "experts.up_scale")) &.{ "gate_proj", "up_proj" } else &.{"down_proj"};
            for (projs) |x| try p.whole(at ++ "experts.{d}.{s}.weight_scale_2", .{ i, e, x }, .f32);
        }
    } else if (eql(what, "shared.gu")) {
        inline for (.{ "gate", "up" }) |x| try p.rows(at ++ "shared_experts." ++ x ++ "_proj.weight", .{i}, s.shared.lo, s.shared.hi, .bf16);
    } else if (eql(what, "shared.down")) {
        try p.cols(at ++ "shared_experts.down_proj.weight", .{i}, s.shared.lo, s.shared.hi);
    } else return error.UnknownUpload;
    return .cut;
}

/// `module`'s e4m3 block scales [n, k/16]: as stored, or its csf_32x2_v1 parts decoded here.
fn scalesOf(p: *Plan, module: []const u8, n: usize, k: usize) ![]u8 {
    const out = try p.gpa.alloc(u8, n * k / 16);
    errdefer p.gpa.free(out);
    if (!p.ck.has(try std.fmt.allocPrint(p.a, "{s}.weight_scale.csf32_bases", .{module}))) {
        const t = try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight_scale", .{module}));
        if (t.dtype != .f8_e4m3 or t.bytes.len != out.len) return error.UnexpectedDType;
        @memcpy(out, t.bytes);
        return out;
    }
    var parts: core.nvfp4_csf32.Parts = undefined;
    inline for (.{ "bases", "codes", "page_bases", "offsets", "exceptions" }) |x| {
        @field(parts, x) = (try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight_scale.csf32_" ++ x, .{module}))).bytes;
    }
    try core.nvfp4_csf32.decode(parts, n, k / 16, out);
    return out;
}

/// NVFP4 matrix `module`'s rows [lo, hi) and inputs [klo, khi): its codes and its e4m3 scales, each its own copy.
fn cut4(p: *Plan, module: []const u8, lo: usize, hi: usize, klo: usize, khi: usize) ![2][]u8 {
    const w = try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight", .{module}));
    const kk = 2 * w.dim(1);
    if (w.dtype != .u8 or w.rank != 2 or lo > hi or hi > w.dim(0) or klo > khi or khi > kk or klo % 16 != 0 or (khi - klo) % 64 != 0) return error.CutOutside;
    const all = try scalesOf(p, module, w.dim(0), kk);
    defer p.gpa.free(all);
    const n = hi - lo;
    const k = khi - klo;
    const codes = try p.gpa.alloc(u8, n * k / 2);
    errdefer p.gpa.free(codes);
    const scales = try p.gpa.alloc(u8, n * k / 16);
    for (0..n) |r| {
        @memcpy(codes[r * k / 2 ..][0 .. k / 2], w.bytes[(lo + r) * kk / 2 + klo / 2 ..][0 .. k / 2]);
        @memcpy(scales[r * k / 16 ..][0 .. k / 16], all[(lo + r) * kk / 16 + klo / 16 ..][0 .. k / 16]);
    }
    return .{ codes, scales };
}

/// A dense NVFP4 linear's `.words` or `.scales`: cuda.nvfp4's host packers (the loader packs words on the GPU).
fn denseMade(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const proj = what["mlp.".len..std.mem.lastIndexOfScalar(u8, what, '.').?];
    const d: usize = c.hidden;
    const down = eql(proj, "down_proj"); // down: every row, the rank's inputs; gate and up: the rank's rows
    const at: [4]usize = if (down) .{ 0, d, s.dense.lo, s.dense.hi } else .{ s.dense.lo, s.dense.hi, 0, d };
    const q = try cut4(p, try std.fmt.allocPrint(p.a, P ++ "layers.{d}.mlp.{s}", .{ i, proj }), at[0], at[1], at[2], at[3]);
    defer for (q) |b| p.gpa.free(b);
    const n = at[1] - at[0];
    const k = at[3] - at[2];
    const npad = cuda.nvfp4.padded(n);
    if (std.mem.endsWith(u8, what, ".scales")) {
        const tiles = try p.gpa.alloc(u8, npad * k / 16);
        cuda.nvfp4.packScales(tiles, q[1], n, k);
        return .{ .made = tiles };
    }
    const words = try p.gpa.alloc(u32, npad / 64 * (k / cuda.nvfp4.group) * 512);
    defer p.gpa.free(words);
    cuda.nvfp4.packWords(words, q[0], n, k);
    return .{ .made = try p.gpa.dupe(u8, std.mem.sliceAsBytes(words)) };
}

/// The routed experts' tiles: gate and up by the rank's rows (packGateUp), down by its inputs (packOne).
fn expertsMade(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, up: bool) !Want {
    const at = P ++ "layers.{d}.mlp.experts.{d}.";
    if (i == c.layers and (try p.ck.get(try std.fmt.allocPrint(p.a, at ++ "gate_proj.weight", .{ i, 0 }))).dtype == .bf16) {
        return .{ .skip = "the GPU quantizer's, from the MTP layer's bf16 experts" };
    }
    const X = cuda.experts.Nvfp4Experts;
    const d: usize = c.hidden;
    const cut = s.moe;
    const ni = cut.len();
    const per = if (up) X.bytes(1, ni, d, 2) / 4 else X.bytes(1, d, ni, 1) / 4;
    const out = try p.gpa.alloc(u32, c.experts * per);
    defer p.gpa.free(out);
    for (0..c.experts) |e| {
        const block = out[e * per ..][0..per];
        if (!up) {
            const q = try cut4(p, try std.fmt.allocPrint(p.a, at ++ "down_proj", .{ i, e }), 0, d, cut.lo, cut.hi);
            defer for (q) |b| p.gpa.free(b);
            X.packOne(block, q[0], q[1], d, ni);
            continue;
        }
        const g = try cut4(p, try std.fmt.allocPrint(p.a, at ++ "gate_proj", .{ i, e }), cut.lo, cut.hi, 0, d);
        defer for (g) |b| p.gpa.free(b);
        const u = try cut4(p, try std.fmt.allocPrint(p.a, at ++ "up_proj", .{ i, e }), cut.lo, cut.hi, 0, d);
        defer for (u) |b| p.gpa.free(b);
        X.packGateUp(block, .{ g[0], g[1] }, .{ u[0], u[1] }, ni, d);
    }
    return .{ .made = try p.gpa.dupe(u8, std.mem.sliceAsBytes(out)) };
}

/// A stored part of a lane upload: its module and the rank's rows or inputs of it (all of either when null).
const Part = struct { module: []const u8, rows: ?[2]usize = null, cols: ?[2]usize = null };

/// Module `rel` of layer `i`, by its checkpoint name.
fn modOf(p: *Plan, i: u32, comptime rel: []const u8) ![]const u8 {
    return std.fmt.allocPrint(p.a, P ++ "layers.{d}." ++ rel, .{i});
}

/// Layer `i`'s lane upload `what` (.codes or .scales): its stored parts as the loader stacks and cuts them.
fn laneUpload(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !Want {
    const base = what[0..std.mem.lastIndexOfScalar(u8, what, '.').?];
    const at = P ++ "layers.{d}.";
    const h = s.mla;
    const sh = s.shared;
    const kh: [2]usize = .{ s.kda.lo * c.kda_dim, s.kda.hi * c.kda_dim };
    const dn: [2]usize = .{ s.dense.lo, s.dense.hi };
    if (eql(base, "dsa.proj")) return laneMade(p, &.{ .{ .module = try modOf(p, i, "self_attn.q_a_proj") }, .{ .module = try modOf(p, i, "self_attn.kv_a_proj_with_mqa") } }, what);
    if (eql(base, "dsa.q_b")) {
        const per = try p.outer(at ++ "self_attn.q_b_proj.weight", .{i}) / c.mla_heads;
        return laneMade(p, &.{.{ .module = try modOf(p, i, "self_attn.q_b_proj"), .rows = .{ h.lo * per, h.hi * per } }}, what);
    }
    if (eql(base, "dsa.o")) return laneMade(p, &.{.{ .module = try modOf(p, i, "self_attn.o_proj"), .cols = .{ h.lo * c.v_dim, h.hi * c.v_dim } }}, what);
    if (eql(base, "dsa.index.kw")) return laneMade(p, &.{ .{ .module = try modOf(p, i, "self_attn.indexer.wk") }, .{ .module = try modOf(p, i, "self_attn.indexer.weights_proj") } }, what);
    if (eql(base, "dsa.index.qb")) return laneMade(p, &.{.{ .module = try modOf(p, i, "self_attn.indexer.wq_b") }}, what);
    if (eql(base, "moe.shared.gu")) return laneMade(p, &.{ .{ .module = try modOf(p, i, "mlp.shared_experts.gate_proj"), .rows = .{ sh.lo, sh.hi } }, .{ .module = try modOf(p, i, "mlp.shared_experts.up_proj"), .rows = .{ sh.lo, sh.hi } } }, what);
    if (eql(base, "moe.shared.down")) return laneMade(p, &.{.{ .module = try modOf(p, i, "mlp.shared_experts.down_proj"), .cols = .{ sh.lo, sh.hi } }}, what);
    if (eql(base, "kda.qkv")) return laneMade(p, &.{ .{ .module = try modOf(p, i, "self_attn.q_proj"), .rows = kh }, .{ .module = try modOf(p, i, "self_attn.k_proj"), .rows = kh }, .{ .module = try modOf(p, i, "self_attn.v_proj"), .rows = kh } }, what);
    if (eql(base, "kda.o")) return laneMade(p, &.{.{ .module = try modOf(p, i, "self_attn.o_proj"), .cols = kh }}, what);
    if (eql(base, "mlp.gu")) return laneMade(p, &.{ .{ .module = try modOf(p, i, "mlp.gate_proj"), .rows = dn }, .{ .module = try modOf(p, i, "mlp.up_proj"), .rows = dn } }, what);
    if (eql(base, "mlp.down")) return laneMade(p, &.{.{ .module = try modOf(p, i, "mlp.down_proj"), .cols = dn }}, what);
    return error.UnknownUpload;
}

/// A lane upload packed from the rank's cut of each part: block FP8 or MXFP8, as its first scale is stored (f32 or u8).
fn laneMade(p: *Plan, parts: []const Part, what: []const u8) !Want {
    const want_codes = std.mem.endsWith(u8, what, ".codes");
    if ((try p.ck.get(try std.fmt.allocPrint(p.a, "{s}.weight_scale", .{parts[0].module}))).dtype == .u8) {
        const MX = glm.weights.formats.mxfp8;
        var stored: [3]MX.Mxfp8 = undefined;
        var shares: [3]MX.Share = undefined;
        for (parts, 0..) |q, j| {
            stored[j] = try MX.Mxfp8.read(p.ck, q.module);
            const r = q.rows orelse [2]usize{ 0, stored[j].n };
            const k = q.cols orelse [2]usize{ 0, stored[j].k };
            shares[j] = .{ .p = &stored[j], .lo = r[0], .hi = r[1], .klo = k[0], .khi = k[1] };
        }
        const z = try MX.sizes(shares[0..parts.len]);
        const codes = try p.gpa.alloc(u8, z.codes);
        errdefer p.gpa.free(codes);
        const tiles = try p.gpa.alloc(u8, z.scales);
        errdefer p.gpa.free(tiles);
        try MX.pack(p.gpa, shares[0..parts.len], codes, tiles);
        p.gpa.free(if (want_codes) tiles else codes);
        return .{ .made = if (want_codes) codes else tiles };
    }
    const FP = glm.weights.formats.fp8pb;
    var stored: [3]FP.Fp8pb = undefined;
    var made: usize = 0;
    defer for (stored[0..made]) |t| t.deinit(p.gpa);
    var shares: [3]FP.Share = undefined;
    for (parts, 0..) |q, j| {
        stored[j] = try FP.Fp8pb.read(p.gpa, p.ck, q.module);
        made += 1;
        const r = q.rows orelse [2]usize{ 0, stored[j].n };
        const k = q.cols orelse [2]usize{ 0, stored[j].k };
        shares[j] = .{ .p = &stored[j], .lo = r[0], .hi = r[1], .klo = k[0], .khi = k[1] };
    }
    const z = try FP.sizes(shares[0..parts.len]);
    const codes = try p.gpa.alloc(u8, z.codes);
    errdefer p.gpa.free(codes);
    const tiles = try p.gpa.alloc(f32, z.scales);
    defer p.gpa.free(tiles);
    try FP.pack(p.gpa, shares[0..parts.len], codes, tiles);
    if (want_codes) return .{ .made = codes };
    const scales = try p.gpa.dupe(u8, std.mem.sliceAsBytes(tiles)); // dupe first: the errdefer owns codes
    p.gpa.free(codes);
    return .{ .made = scales };
}

/// The bytes `cuts` make: each piece's elements in order, written as its dtype.
fn bytesOf(gpa: std.mem.Allocator, ck: *core.Checkpoint, cuts: []const Cut) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    for (cuts) |c| {
        const t = try ck.get(c.name);
        const outer = t.dim(0);
        const width = t.numel() / outer;
        if (c.cols) {
            if (t.rank != 2 or c.lo > c.hi or c.hi > width) return error.CutOutside;
            for (0..outer) |r| try put(&out, gpa, t, r * width + c.lo, r * width + c.hi, c.to);
        } else {
            if (c.lo > c.hi or c.hi > outer) return error.CutOutside;
            try put(&out, gpa, t, c.lo * width, c.hi * width, c.to);
        }
    }
    return out.toOwnedSlice(gpa);
}

/// Elements [lo, hi) of `t` appended as `to`.
fn put(out: *std.ArrayList(u8), gpa: std.mem.Allocator, t: Tensor, lo: usize, hi: usize, to: DType) !void {
    if (t.dtype == to) return out.appendSlice(gpa, t.bytes[lo * to.size() .. hi * to.size()]);
    for (lo..hi) |i| {
        const v: f32 = switch (t.dtype) {
            .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)) << 16),
            .f32 => @bitCast(std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little)),
            else => return error.UnexpectedDType,
        };
        switch (to) {
            .f32 => try out.appendSlice(gpa, &std.mem.toBytes(v)),
            .bf16 => try out.appendSlice(gpa, &std.mem.toBytes(bf16(v))),
            else => return error.UnexpectedDType,
        }
    }
}

/// fp32 to bf16 to nearest even (torch's), NaN kept quiet.
fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// What a run changes on purpose: one bit of upload `name`'s downloaded bytes; the copies changed and those caught.
const Tamper = struct { name: ?[]const u8, done: usize = 0, caught: usize = 0 };

/// Rank `rank` of `world` loaded, each upload compared: how many differ or are not known, one more if none was.
fn rankGate(gpu: check.Gpu, a: std.mem.Allocator, ck: *core.Checkpoint, c: *const Config, m: *const glm.forward.Modules, stream: cuda.Stream, dir: []const u8, layers: []const u32, rank: usize, world: usize, tamper: *Tamper) !usize {
    const s = try glm.shard.Share.of(c, rank, world);
    var w = try glm.weights.load(gpu.gpa, gpu.io, gpu.d, stream, m.tools(), dir, c.*, s, layers, true);
    defer w.deinit();
    var bad: usize = 0;
    var cut: usize = 0;
    var made: usize = 0;
    var bytes: usize = 0;
    var left: std.ArrayList([2][]const u8) = .empty;
    for (w.named.items) |n| {
        var p: Plan = .{ .a = a, .gpa = gpu.gpa, .ck = ck };
        const how = plan(&p, c, s, n.name) catch |e| {
            std.debug.print("FAIL TP={d} rank {d} {s}: {t}\n", .{ world, rank, n.name, e });
            bad += 1;
            continue;
        };
        const want = switch (how) {
            .skip => |why| {
                try left.append(a, .{ n.name, why });
                continue;
            },
            .cut => bytesOf(gpu.gpa, ck, p.cuts.items) catch |e| {
                std.debug.print("FAIL TP={d} rank {d} {s}: {t}\n", .{ world, rank, n.name, e });
                bad += 1;
                continue;
            },
            .made => |b| b,
        };
        defer gpu.gpa.free(want);
        const got = try gpu.gpa.alloc(u8, n.len);
        defer gpu.gpa.free(got);
        const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = n.ptr, .len = n.len, .borrowed = true };
        try b.download(0, got);
        if (tamper.name) |t| if (eql(t, n.name)) {
            got[got.len / 2] ^= 1;
            tamper.done += 1;
        };
        var label: [160]u8 = undefined;
        check.sameBytes(try std.fmt.bufPrint(&label, "TP={d} rank {d} {s}", .{ world, rank, n.name }), got, want) catch {
            bad += 1;
            if (tamper.name) |t| if (eql(t, n.name)) {
                tamper.caught += 1;
            };
            continue;
        };
        if (how == .made) made += 1 else cut += 1;
        bytes += n.len;
    }
    const line = "glm-loader-gate TP={d} rank {d}: {d} uploads as the checkpoint cut, {d} as the host packers make them " ++
        "({d} MiB), {d} left out, {d} differ or are not known";
    const args = .{ world, rank, cut, made, bytes >> 20, left.items.len, bad };
    const none = cut + made == 0; // a rank that compared nothing proves nothing
    if (bad == 0 and !none) check.pass(line, args) else std.debug.print("FAIL " ++ line ++ "\n", args);
    if (world == 1) for (left.items) |x| std.debug.print("  left out: {s} ({s})\n", .{ x[0], x[1] });
    return bad + @intFromBool(none);
}

/// Numbers from an environment value like "0-3,41,44", each under `max`; `all` when the variable is unset.
fn picks(a: std.mem.Allocator, name: [:0]const u8, all: []const u32, max: usize) ![]const u32 {
    const text = glm.ranks.environment(name) orelse return all;
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |part| {
        const dash = std.mem.indexOfScalar(u8, part, '-');
        const lo = try std.fmt.parseInt(u32, if (dash) |i| part[0..i] else part, 10);
        const hi = if (dash) |i| try std.fmt.parseInt(u32, part[i + 1 ..], 10) else lo;
        if (hi < lo or hi >= max) return error.BadGateList;
        for (lo..hi + 1) |v| try out.append(a, @intCast(v));
    }
    return out.toOwnedSlice(a);
}

/// TF_GATE_TAMPER=NAME flips a bit of upload NAME; TF_GATE_WORLDS, _RANKS and _LAYERS narrow a real checkpoint's run.
pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
    const c = try glm.config.parse(a, text);
    var m = try glm.forward.Modules.load(gpu.d, gpu.ctx);
    defer m.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var ck = try core.Checkpoint.openModel(gpu.gpa, gpu.io, dir);
    defer ck.close();
    const every = try a.alloc(u32, c.layers);
    for (every, 0..) |*l, i| l.* = @intCast(i);
    const layers = try picks(a, "TF_GATE_LAYERS", every, c.layers);
    const worlds = try picks(a, "TF_GATE_WORLDS", &.{ 1, 2, 3 }, 4);
    const ranks = try picks(a, "TF_GATE_RANKS", &.{ 0, 1, 2 }, 3);
    var tamper: Tamper = .{ .name = glm.ranks.environment("TF_GATE_TAMPER") };
    var bad: usize = 0;
    var gated: usize = 0;
    for (worlds) |world| {
        if (world == 0) return error.BadGateList;
        for (ranks) |rank| if (rank < world) {
            bad += try rankGate(gpu, a, &ck, &c, &m, stream, dir, layers, rank, world, &tamper);
            gated += 1;
        };
    }
    try check.expect(gated > 0, "glm-loader-gate: TF_GATE_WORLDS and TF_GATE_RANKS leave no rank to gate", .{});
    if (tamper.name) |t| {
        try check.expect(tamper.done > 0, "glm-loader-gate: TF_GATE_TAMPER={s} names no upload it compares", .{t});
        const ok = tamper.caught == tamper.done and bad == tamper.caught;
        const why = "glm-loader-gate: TF_GATE_TAMPER={s}: {d} of {d} flipped copies caught, {d} uploads differ";
        try check.expect(ok, why, .{ t, tamper.caught, tamper.done, bad });
        check.pass("glm-loader-gate: TF_GATE_TAMPER={s} red as planted, {d} copies caught", .{ t, tamper.done });
        return;
    }
    try check.expect(bad == 0, "glm-loader-gate: {d} uploads differ from the checkpoint or are not known", .{bad});
}
