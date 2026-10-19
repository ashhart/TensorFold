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

/// One upload's pieces as the Python family's weights.load cuts them, read from a checkpoint apart from the loader's.
const Plan = struct {
    a: std.mem.Allocator,
    ck: *core.Checkpoint,
    cuts: std.ArrayList(Cut) = .empty,

    fn rows(p: *Plan, comptime fmt: []const u8, args: anytype, lo: usize, hi: usize, to: DType) !void {
        try p.cuts.append(p.a, .{ .name = try std.fmt.allocPrint(p.a, fmt, args), .lo = lo, .hi = hi, .to = to });
    }

    fn whole(p: *Plan, comptime fmt: []const u8, args: anytype, to: DType) !void {
        const name = try std.fmt.allocPrint(p.a, fmt, args);
        try p.cuts.append(p.a, .{ .name = name, .lo = 0, .hi = (try p.ck.get(name)).dim(0), .to = to });
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

/// Upload `name`'s pieces at share `s` into `p`; false when it is packed (no host reference here yet).
fn plan(p: *Plan, c: *const Config, s: glm.shard.Share, name: []const u8) !bool {
    if (eql(name, "embed")) {
        try p.whole(P ++ "embed_tokens.weight", .{}, .bf16);
    } else if (eql(name, "norm")) {
        try p.whole(P ++ "norm.weight", .{}, .bf16);
    } else if (eql(name, "head")) {
        try p.rows("lm_head.weight", .{}, s.vocab.lo, s.vocab.hi, .bf16);
    } else if (std.mem.startsWith(u8, name, "draft_head.")) {
        return false; // the GPU's 4-bit tiles of the head's rows
    } else if (std.mem.startsWith(u8, name, "mtp.")) {
        for (mtp_sources) |m| if (eql(name[4..], m[0])) {
            try p.whole(P ++ "layers.{d}.{s}.weight", .{ c.layers, m[1] }, .bf16);
            return true;
        };
        return error.UnknownUpload;
    } else if (std.mem.startsWith(u8, name, "layers.")) {
        const dot = std.mem.indexOfScalarPos(u8, name, "layers.".len, '.') orelse return error.UnknownUpload;
        const i = std.fmt.parseInt(u32, name["layers.".len..dot], 10) catch return error.UnknownUpload;
        return layer(p, c, s, i, name[dot + 1 ..]);
    } else return error.UnknownUpload;
    return true;
}

/// Layer `i`'s upload `what`, as `plan`.
fn layer(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !bool {
    const L = P ++ "layers.{d}.";
    if (eql(what, "in_norm")) {
        try p.whole(L ++ "input_layernorm.weight", .{i}, .bf16);
        return true;
    }
    if (eql(what, "post_norm")) {
        try p.whole(L ++ "post_attention_layernorm.weight", .{i}, .bf16);
        return true;
    }
    inline for (.{ "attn", "ffn" }) |site| {
        if (eql(what, site ++ "_hc.fn")) {
            try p.whole(L ++ "hc_" ++ site ++ "_fn", .{i}, .bf16);
            return true;
        }
        inline for (.{ "base", "scale" }) |x| if (eql(what, site ++ "_hc." ++ x)) {
            try p.whole(L ++ "hc_" ++ site ++ "_" ++ x, .{i}, .f32);
            return true;
        };
    }
    if (std.mem.startsWith(u8, what, "kda.")) return kda(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "dsa.")) return dsa(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "moe.")) return moe(p, c, s, i, what[4..]);
    if (std.mem.startsWith(u8, what, "mlp.") and (std.mem.endsWith(u8, what, ".words") or std.mem.endsWith(u8, what, ".scales"))) {
        return false; // a dense NVFP4 linear: tf_pack_dense's words, packScales' tiles
    }
    return error.UnknownUpload;
}

/// KDA: [q | k | v | f_a | g_a | b], q k v b by the rank's heads; f_b, g_b, taps, dt_bias by heads; o by columns.
fn kda(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !bool {
    const at = P ++ "layers.{d}.self_attn.";
    const h = s.kda;
    const dk: usize = c.kda_dim;
    if (eql(what, "proj")) {
        inline for (.{ "q", "k", "v" }) |x| try p.rows(at ++ x ++ "_proj.weight", .{i}, h.lo * dk, h.hi * dk, .bf16);
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
    return true;
}

/// Latent attention: [q_a | kv_a] whole; q_b, kv_b's keys and values by heads (rows from the tensors); o by columns.
fn dsa(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !bool {
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
        for (h.lo..h.hi) |head| {
            const lo = if (eql(what, "absorb.wk")) head * per else head * per + keys;
            try p.rows(at ++ "kv_b_proj.weight", .{i}, lo, lo + (if (eql(what, "absorb.wk")) keys else dv), .bf16);
        }
    } else if (eql(what, "o")) {
        try p.cols(at ++ "o_proj.weight", .{i}, h.lo * dv, h.hi * dv);
    } else if (eql(what, "index.kw")) {
        try p.whole(at ++ "indexer.wk.weight", .{i}, .bf16);
        try p.whole(at ++ "indexer.weights_proj.weight", .{i}, .bf16);
    } else {
        const index = [_][2][]const u8{ .{ "index.qb", "wq_b.weight" }, .{ "index.ln_w", "k_norm.weight" }, .{ "index.ln_b", "k_norm.bias" }, .{ "index.gate", "index_kpool_compress_gate" }, .{ "index.ape", "index_kpool_compress_ape" } };
        for (index) |x| if (eql(what, x[0])) {
            try p.whole(at ++ "indexer.{s}", .{ i, x[1] }, .bf16);
            return true;
        };
        return error.UnknownUpload;
    }
    return true;
}

/// MoE: router and bias whole, the experts' global scales, the shared expert's gate and up by rows, down by columns.
fn moe(p: *Plan, c: *const Config, s: glm.shard.Share, i: u32, what: []const u8) !bool {
    const at = P ++ "layers.{d}.mlp.";
    if (eql(what, "router")) {
        try p.whole(at ++ "gate.weight", .{i}, .bf16);
    } else if (eql(what, "bias")) {
        try p.whole(at ++ "gate.e_score_correction_bias", .{i}, .f32);
    } else if (eql(what, "experts.up") or eql(what, "experts.down")) {
        return false; // packGateUp's and packOne's tiles
    } else if (eql(what, "experts.up_scale") or eql(what, "experts.down_scale")) {
        // the MTP layer's bf16 experts take the GPU quantizer's scales; W4A16_NVFP4 ones (FP4-Local's) keep their own
        if (i == c.layers and !p.ck.has(try std.fmt.allocPrint(p.a, at ++ "experts.0.gate_proj.weight_scale_2", .{i}))) return false;
        for (0..c.experts) |e| {
            const projs: []const []const u8 = if (eql(what, "experts.up_scale")) &.{ "gate_proj", "up_proj" } else &.{"down_proj"};
            for (projs) |x| try p.whole(at ++ "experts.{d}.{s}.weight_scale_2", .{ i, e, x }, .f32);
        }
    } else if (eql(what, "shared.gu")) {
        inline for (.{ "gate", "up" }) |x| try p.rows(at ++ "shared_experts." ++ x ++ "_proj.weight", .{i}, s.shared.lo, s.shared.hi, .bf16);
    } else if (eql(what, "shared.down")) {
        try p.cols(at ++ "shared_experts.down_proj.weight", .{i}, s.shared.lo, s.shared.hi);
    } else return error.UnknownUpload;
    return true;
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

/// What a run changes on purpose: one bit of upload `name`'s downloaded bytes, and how many copies it changed.
const Tamper = struct { name: ?[]const u8, done: usize = 0 };

/// Rank `rank` of `world` loaded, and each upload compared: how many differ or are not known.
fn rankGate(gpu: check.Gpu, a: std.mem.Allocator, ck: *core.Checkpoint, c: *const Config, m: *const glm.forward.Modules, stream: cuda.Stream, dir: []const u8, layers: []const u32, rank: usize, world: usize, tamper: *Tamper) !usize {
    const s = try glm.shard.Share.of(c, rank, world);
    var w = try glm.weights.load(gpu.gpa, gpu.io, gpu.d, stream, m.tools(), dir, c.*, s, layers, true);
    defer w.deinit();
    var bad: usize = 0;
    var same: usize = 0;
    var bytes: usize = 0;
    var skipped: std.ArrayList([]const u8) = .empty;
    for (w.named.items) |n| {
        var p: Plan = .{ .a = a, .ck = ck };
        const known = plan(&p, c, s, n.name) catch |e| {
            std.debug.print("FAIL TP={d} rank {d} {s}: {t}\n", .{ world, rank, n.name, e });
            bad += 1;
            continue;
        };
        if (!known) {
            try skipped.append(a, n.name);
            continue;
        }
        const want = try bytesOf(gpu.gpa, ck, p.cuts.items);
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
            continue;
        };
        same += 1;
        bytes += n.len;
    }
    check.pass("glm-loader-gate TP={d} rank {d}: {d} uploads ({d} MiB) as the checkpoint cut on the host, {d} packed not compared", .{ world, rank, same, bytes >> 20, skipped.items.len });
    if (world == 1) for (skipped.items) |name| std.debug.print("  packed, not compared: {s}\n", .{name});
    return bad;
}

/// TF_GATE_TAMPER=NAME flips a bit of upload NAME after each download, so a run shows the gate refusing it by name.
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
    const layers = try a.alloc(u32, c.layers);
    for (layers, 0..) |*l, i| l.* = @intCast(i);
    var tamper: Tamper = .{ .name = glm.ranks.environment("TF_GATE_TAMPER") };
    var bad: usize = 0;
    for (1..4) |world| {
        for (0..world) |rank| bad += try rankGate(gpu, a, &ck, &c, &m, stream, dir, layers, rank, world, &tamper);
    }
    if (tamper.name) |t| try check.expect(tamper.done > 0, "glm-loader-gate: TF_GATE_TAMPER={s} names no upload it compares", .{t});
    try check.expect(bad == 0, "glm-loader-gate: {d} uploads differ from the checkpoint or are not known", .{bad});
}
