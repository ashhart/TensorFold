//! A TP rank's weight bytes before a load: weights.load's walk over the headers, each store buffer as it allocates.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const config = @import("config.zig");
const names = @import("names.zig");
const cuts = @import("weights_cut.zig");

const Config = config.Config;
const Role = names.Role;
const Span = cuts.Span;
const Tensor = core.checkpoint.Tensor;

/// Store.alloc's own floor: a buffer is at least 16 bytes.
fn buffer(len: usize) u64 {
    return @max(len, 16);
}

/// An EXL3 group's buffers: trellis, suh and svh bytes.
pub const Parts = struct { trellis: usize, suh: usize, svh: usize };

/// An EXL3 group [np, k] (`tile` bytes a tile) as cuts.exl3 keeps it: `out` columns cut svh, `in` rows cut suh.
pub fn exl3Cut(k: usize, np: usize, tile: usize, out: ?usize, in: ?usize) Parts {
    if (in) |r| return .{ .trellis = (r / 16) * (np / 16) * tile, .suh = r * 2, .svh = np * 2 };
    const n = out orelse np;
    return .{ .trellis = (k / 16) * (n / 16) * tile, .suh = k * 2, .svh = n * 2 };
}

/// A bf16 lane weight of `rows` x `cols`: rows padded to whole 128-row blocks (modelopt.padded), as bf16Of stores it.
pub fn plain(rows: usize, cols: usize) usize {
    return cuda.modelopt.padded(rows) * cols * 2;
}

/// The walk: `sink.put(name, bytes)` for each buffer weights.load keeps for rank `rank` of `world`, under its name.
pub fn Walk(comptime Sink: type) type {
    return struct {
        ck: *core.Checkpoint,
        c: *const Config,
        rank: u32,
        world: u32,
        sink: Sink,
        buf: [256]u8 = undefined,
        nbuf: [256]u8 = undefined,

        const W = @This();

        fn put(w: *W, name: []const u8, len: usize) !void {
            try w.sink.put(name, buffer(len));
        }

        fn share(w: *const W, total: usize) ?Span {
            return if (w.world == 1) null else cuts.even(total, w.rank, w.world);
        }

        fn module(w: *W, i: u32, role: Role) ![]const u8 {
            return names.moduleName(&w.buf, i, role, null);
        }

        /// An EXL3 group's stored shape from its trellis header: tile bytes, padded width, the trellis's bytes.
        fn group(w: *W, m: []const u8) !?struct { tile: usize, np: usize, bytes: usize } {
            const name = try std.fmt.bufPrint(&w.nbuf, "{s}.trellis", .{m});
            if (w.c.quant != .exl3 or !w.ck.has(name)) return null;
            const t = try w.ck.get(name);
            if (t.rank != 3) return error.UnexpectedTensor;
            return .{ .tile = t.shape[2] * 2, .np = t.shape[1] * 16, .bytes = t.bytes.len };
        }

        fn exl3Parts(w: *W, m: []const u8, p: Parts) !void {
            try w.put(m, p.trellis);
            try w.put(try std.fmt.bufPrint(&w.nbuf, "{s}.suh", .{m}), p.suh);
            try w.put(try std.fmt.bufPrint(&w.nbuf, "{s}.svh", .{m}), p.svh);
        }

        /// Reader.linear: an EXL3 group as stored, else a bf16 lane weight padded to 128-row blocks.
        fn linear(w: *W, m: []const u8, n: usize, k: usize) !void {
            if (try w.group(m)) |g| return w.exl3Parts(m, .{ .trellis = g.bytes, .suh = k * 2, .svh = g.np * 2 });
            try w.put(m, plain(n, k));
        }

        /// cuts.linear: output rows `out` or input columns `in`; uncut as linear.
        fn cutLinear(w: *W, m: []const u8, n: usize, k: usize, out: []const Span, in: ?Span) !void {
            if (out.len == 0 and in == null) return w.linear(m, n, k);
            var rows: usize = 0;
            for (out) |s| rows += s.len();
            if (try w.group(m)) |g| return w.exl3Parts(m, if (in) |s| exl3Cut(k, g.np, g.tile, null, s.len()) else exl3Cut(k, g.np, g.tile, rows, null));
            if (in) |s| return w.put(m, plain(n, s.len()));
            try w.put(m, plain(rows, k));
        }

        /// Loader.cutLin.
        fn cutLin(w: *W, i: u32, role: Role, out: bool) !void {
            const nk = names.dims(w.c, role).linear;
            const m = try w.module(i, role);
            const s = w.share(if (out) nk[0] else nk[1]) orelse return w.linear(m, nk[0], nk[1]);
            if (out) try w.cutLinear(m, nk[0], nk[1], &.{s}, null) else try w.cutLinear(m, nk[0], nk[1], &.{}, s);
        }

        fn lin(w: *W, i: u32, role: Role) !void {
            const nk = names.dims(w.c, role).linear;
            try w.linear(try w.module(i, role), nk[0], nk[1]);
        }

        /// Loader.small: names.numel values as `size`-byte elements, under the tensor's name.
        fn small(w: *W, i: u32, role: Role, size: usize) !void {
            const m = try w.module(i, role);
            const bare = role == .kda_a_log or role == .kda_dt_bias or role == .router_bias;
            try w.put(if (bare) m else try std.fmt.bufPrint(&w.nbuf, "{s}.weight", .{m}), names.numel(w.c, role) * size);
        }

        /// Loader.mlp under `prefix`: gate and up by output rows, down by input columns at TP.
        fn mlp(w: *W, prefix: []const u8, gate: Role) !void {
            const nk = names.dims(w.c, gate).linear;
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
                var b: [192]u8 = undefined;
                const m = try std.fmt.bufPrint(&b, "{s}{s}", .{ prefix, p });
                if (w.share(nk[0])) |s| {
                    if (j < 2) try w.cutLinear(m, nk[0], nk[1], &.{s}, null) else try w.cutLinear(m, nk[1], nk[0], &.{}, s);
                } else if (j < 2) try w.linear(m, nk[0], nk[1]) else try w.linear(m, nk[1], nk[0]);
            }
        }

        fn kda(w: *W, i: u32) !void {
            const n = names.dims(w.c, .kda_q_conv).conv[0];
            var b: [96]u8 = undefined;
            if (w.c.quant != .exl3) {
                for ([_]Role{ .kda_q, .kda_k, .kda_v, .kda_b, .kda_f, .kda_g, .kda_o }) |r| try w.lin(i, r);
                try w.put(try std.fmt.bufPrint(&b, "model.layers.{d}.attention.conv", .{i}), 3 * n * w.c.conv * 4);
                try w.small(i, .kda_a_log, 4);
                try w.small(i, .kda_dt_bias, 4);
                return w.small(i, .kda_o_norm, 2);
            }
            const qkv = try std.fmt.bufPrint(&b, "model.layers.{d}.attention.qkv_proj", .{i});
            var cb: [96]u8 = undefined;
            const conv = try std.fmt.bufPrint(&cb, "model.layers.{d}.attention.conv1d.weight", .{i});
            if (w.world == 1) {
                try w.linear(qkv, 3 * n, w.c.hidden);
                for ([_]Role{ .kda_b, .kda_f, .kda_g, .kda_o }) |r| try w.lin(i, r);
                try w.put(conv, 3 * n * w.c.conv * 4);
                try w.small(i, .kda_a_log, 4);
                try w.small(i, .kda_dt_bias, 4);
                return w.small(i, .kda_o_norm, 2);
            }
            const h = n / w.world;
            const lo = w.rank * h;
            const sp = [3]Span{ .{ .lo = lo, .hi = lo + h }, .{ .lo = n + lo, .hi = n + lo + h }, .{ .lo = 2 * n + lo, .hi = 2 * n + lo + h } };
            try w.cutLinear(qkv, 3 * n, w.c.hidden, &sp, null);
            for ([_]Role{ .kda_b, .kda_f, .kda_g }) |r| try w.cutLin(i, r, true);
            try w.cutLin(i, .kda_o, false);
            try w.put(conv, 3 * h * w.c.conv * 4);
            try w.put(try w.module(i, .kda_a_log), w.share(w.c.kda_heads).?.len() * 4);
            try w.put(try w.module(i, .kda_dt_bias), w.share(n).?.len() * 4);
            try w.small(i, .kda_o_norm, 2);
        }

        fn mla(w: *W, i: u32) !void {
            try w.lin(i, .mla_kv_a);
            try w.small(i, .mla_kv_a_norm, 2);
            try w.cutLin(i, .mla_kv_b, true);
            try w.cutLin(i, .mla_o, false);
            if (w.c.q_lora > 0) {
                try w.lin(i, .mla_q_a);
                try w.small(i, .mla_q_a_norm, 2);
                try w.cutLin(i, .mla_q_b, true);
            } else try w.cutLin(i, .mla_q, true);
            if (w.c.head_gate) try w.cutLin(i, .mla_gate, true);
        }

        /// Reader.exl3Experts: per projection, the trellises in one buffer, then the pointer, k2, suh and svh tables.
        fn experts(w: *W, i: u32) !void {
            const c = w.c;
            const ni: usize = c.moe_inter;
            const d: usize = c.hidden;
            const cut = w.share(ni);
            var eb: [192]u8 = undefined;
            if (try w.group(try names.moduleName(&eb, i, .expert_gate, 0)) == null) {
                if (cut != null) return error.TpFormat;
                for (0..c.experts) |e| {
                    var pb: [96]u8 = undefined;
                    try w.mlpWhole(try std.fmt.bufPrint(&pb, "model.layers.{d}.mlp.experts.{d}.", .{ i, e }), ni, d);
                }
                return;
            }
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, m| {
                const k = if (m < 2) d else ni;
                var trellis: usize = 0;
                var suh: usize = 0;
                var svh: usize = 0;
                for (0..c.experts) |e| {
                    const g = (try w.group(try std.fmt.bufPrint(&eb, "model.layers.{d}.mlp.experts.{d}." ++ proj, .{ i, e }))) orelse return error.MixedExperts;
                    // gate and up cut by tile columns, down by tile rows (cuts.expertTrellis); uncut: as stored
                    const p: Parts = if (cut) |s| (if (m < 2) exl3Cut(k, g.np, g.tile, s.len(), null) else exl3Cut(k, g.np, g.tile, null, s.len())) else .{ .trellis = g.bytes, .suh = k * 2, .svh = g.np * 2 };
                    trellis += p.trellis;
                    suh += p.suh;
                    svh += p.svh;
                }
                const base = "model.layers.{d}.mlp.experts." ++ proj;
                try w.put(try std.fmt.bufPrint(&w.nbuf, base ++ ".trellis", .{i}), trellis);
                try w.put(try std.fmt.bufPrint(&w.nbuf, base ++ ".trellis_ptrs", .{i}), c.experts * 8);
                try w.put(try std.fmt.bufPrint(&w.nbuf, base ++ ".k2", .{i}), c.experts * 4);
                try w.put(try std.fmt.bufPrint(&w.nbuf, base ++ ".suh", .{i}), suh);
                try w.put(try std.fmt.bufPrint(&w.nbuf, base ++ ".svh", .{i}), svh);
            }
        }

        /// Reader.mlp: gate and up [width, dims], down [dims, width], uncut.
        fn mlpWhole(w: *W, prefix: []const u8, width: usize, dims: usize) !void {
            inline for (.{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, j| {
                var b: [192]u8 = undefined;
                const m = try std.fmt.bufPrint(&b, "{s}{s}", .{ prefix, p });
                if (j < 2) try w.linear(m, width, dims) else try w.linear(m, dims, width);
            }
        }

        fn layer(w: *W, i: u32) !void {
            const c = w.c;
            try w.small(i, .attn_norm, 2);
            try w.small(i, .mlp_norm, 2);
            switch (c.kind(i)) {
                .kda => try w.kda(i),
                .mla => try w.mla(i),
            }
            var b: [64]u8 = undefined;
            if (!c.isMoe(i)) return w.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.", .{i}), .dense_gate);
            try w.small(i, .router, 2);
            if (c.expert_bias) try w.small(i, .router_bias, 4);
            try w.experts(i);
            if (c.shared_inter > 0) try w.mlp(try std.fmt.bufPrint(&b, "model.layers.{d}.mlp.shared_experts.", .{i}), .shared_gate);
        }

        /// Every backbone layer (the MTP layer is never loaded), then the embeddings, final norm and head.
        pub fn all(w: *W) !void {
            for (0..w.c.layers) |i| try w.layer(@intCast(i));
            const vd = names.dims(w.c, .embed).linear;
            try w.put("model.word_embeddings.weight", vd[0] * vd[1] * 2);
            try w.put("model.norm.weight", names.numel(w.c, .final_norm) * 2);
            if (w.c.tie) return;
            const hd = names.dims(w.c, .lm_head).linear;
            if (w.share(hd[0])) |s| try w.cutLinear("lm_head", hd[0], hd[1], &.{s}, null) else try w.linear("lm_head", hd[0], hd[1]);
        }
    };
}

const Sum = struct {
    total: u64 = 0,
    fn put(s: *Sum, _: []const u8, bytes: u64) !void {
        s.total += bytes;
    }
};

/// Bytes rank `rank` of `world` keeps on its GPU for the checkpoint `ck`, as weights.load allocates them.
pub fn rankBytes(ck: *core.Checkpoint, c: *const Config, rank: u32, world: u32) !u64 {
    var w: Walk(*Sum) = .{ .ck = ck, .c = c, .rank = rank, .world = world, .sink = undefined };
    var s: Sum = .{};
    w.sink = &s;
    try w.all();
    return s.total;
}

/// rankBytes for the checkpoint at `dir`: its config and tensor headers, no weight bytes read.
pub fn rankBytesAt(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32) !u64 {
    var why: config.Why = .{};
    var opened = try @import("weights.zig").open(gpa, io, dir, &why);
    defer opened.ck.close();
    return rankBytes(&opened.ck, &opened.c, rank, world);
}

test "cut sizes: a 64-row plain projection pads to 128 rows a rank; svh follows output cuts, suh input cuts" {
    // b_proj [64, 4096] at TP=2: 32 rows a rank, stored as one 128-row block
    try std.testing.expectEqual(@as(usize, 1_048_576), plain(32, 4096));
    try std.testing.expectEqual(@as(usize, 1_048_576), plain(64, 4096));
    try std.testing.expectEqual(@as(usize, 2 * 1_048_576), plain(129, 4096));
    // an expert's down projection [4096, 1024] cut to 512 input rows: suh 512 values, svh whole, half the tiles
    const tile = 16 * 16 * 3 / 8; // 3 bits a weight
    const down = exl3Cut(1024, 4096, tile, null, 512);
    try std.testing.expectEqual(Parts{ .trellis = 32 * 256 * tile, .suh = 1024, .svh = 8192 }, down);
    // its gate projection [1024, 4096] cut to 512 output columns: svh 512 values, suh whole
    const gate = exl3Cut(4096, 1024, tile, 512, null);
    try std.testing.expectEqual(Parts{ .trellis = 256 * 32 * tile, .suh = 8192, .svh = 1024 }, gate);
    // uncut: the stored group
    try std.testing.expectEqual(Parts{ .trellis = 256 * 64 * tile, .suh = 8192, .svh = 2048 }, exl3Cut(4096, 1024, tile, null, null));
    try std.testing.expectEqual(@as(u64, 16), buffer(4));
}
