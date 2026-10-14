//! GLM-5.3-Flash's loader on synthetic checkpoints: one sparse shard whose header names every tensor at its width.
const std = @import("std");
const cfg = @import("config.zig");
const wts = @import("weights.zig");
const quant = @import("quant.zig");

const testing = std.testing;
const Width = quant.Width;

const Entry = struct { name: []const u8, dtype: []const u8, shape: []const usize };

/// A module stored at a width other than 4/64.
const Pack = struct { []const u8, Width };

/// The tensors the loader reads for `c` (its first `c.run` layers and the MTP layer), stored at `stored`'s widths.
const Fixture = struct {
    a: std.mem.Allocator,
    c: *const cfg.Config,
    stored: []const Pack = &.{}, // module -> its width on disk (else 4/64)
    list: std.ArrayList(Entry) = .empty,
    mlx: bool = false, // mlx-lm's layout: its names, stacked experts, the absorbed pair, bf16 everywhere unquantized
    fused: bool = false, // with mlx: one conv1d tensor for q, k and v, and fp32 kept (mixes, router, biases)

    fn plain(f: *Fixture, comptime fmt: []const u8, args: anytype, dtype: []const u8, shape: []const usize) !void {
        const t = if (f.mlx and !f.fused and std.mem.eql(u8, dtype, "F32")) "BF16" else dtype;
        try f.list.append(f.a, .{ .name = try std.fmt.allocPrint(f.a, fmt, args), .dtype = t, .shape = try f.a.dupe(usize, shape) });
    }

    /// A stack of `lead` matrices [n, k] (mlx-lm's experts and per-head maps), at its module's width.
    fn q3(f: *Fixture, comptime fmt: []const u8, args: anytype, lead: usize, n: usize, k: usize) !void {
        const module = try std.fmt.allocPrint(f.a, fmt, args);
        var w = quant.uniform;
        for (f.stored) |p| if (std.mem.eql(u8, p[0], module)) {
            w = p[1];
        };
        try f.plain("{s}.weight", .{module}, "U32", &.{ lead, n, k * w.bits / 32 });
        try f.plain("{s}.scales", .{module}, "BF16", &.{ lead, n, k / w.group });
        try f.plain("{s}.biases", .{module}, "BF16", &.{ lead, n, k / w.group });
    }

    fn q(f: *Fixture, comptime fmt: []const u8, args: anytype, n: usize, k: usize) !void {
        const module = try std.fmt.allocPrint(f.a, fmt, args);
        var w = quant.uniform;
        for (f.stored) |p| if (std.mem.eql(u8, p[0], module)) {
            w = p[1];
        };
        try f.plain("{s}.weight", .{module}, "U32", &.{ n, k * w.bits / 32 });
        try f.plain("{s}.scales", .{module}, "BF16", &.{ n, k / w.group });
        try f.plain("{s}.biases", .{module}, "BF16", &.{ n, k / w.group });
    }

    fn layer(f: *Fixture, i: usize) !void {
        const c = f.c;
        const D: usize = c.hidden;
        const p = "model.language_model.layers.{d}.";
        const a = p ++ "self_attn.";
        if (i != c.layers) for ([_][]const u8{ "attn", "ffn" }) |site| {
            try f.plain(p ++ "hc_{s}_fn", .{ i, site }, if (f.fused) "F32" else "BF16", &.{ 24, 4 * D });
            try f.plain(p ++ "hc_{s}_base", .{ i, site }, "F32", &.{24});
            try f.plain(p ++ "hc_{s}_scale", .{ i, site }, "F32", &.{3});
        };
        try f.plain(p ++ "input_layernorm.weight", .{i}, "BF16", &.{D});
        try f.plain(p ++ "post_attention_layernorm.weight", .{i}, "BF16", &.{D});
        switch (c.kind(@intCast(i))) {
            .kda => {
                const W: usize = c.kdaWidth();
                const dk: usize = c.kda_dim;
                if (f.fused) try f.plain(a ++ "conv1d.weight", .{i}, "BF16", &.{ 3 * W, c.conv, 1 }) else for ([_][]const u8{ "q", "k", "v" }) |x|
                    try f.plain(a ++ "{s}_conv1d.weight", .{ i, x }, "BF16", &.{ W, 1, c.conv });
                try f.plain(a ++ "o_norm.weight", .{i}, "BF16", &.{dk});
                for ([_][]const u8{ "q_proj", "k_proj", "v_proj" }) |x| try f.q(a ++ "{s}", .{ i, x }, W, D);
                try f.q(a ++ "f_a_proj", .{i}, dk, D);
                try f.q(a ++ "g_a_proj", .{i}, dk, D);
                try f.q(a ++ "b_proj", .{i}, c.kda_heads, D);
                try f.q(a ++ "f_b_proj", .{i}, W, dk);
                try f.q(a ++ "g_b_proj", .{i}, W, dk);
                try f.q(a ++ "o_proj", .{i}, D, W);
                try f.plain(a ++ "A_log", .{i}, "F32", &.{c.kda_heads});
                try f.plain(a ++ "dt_bias", .{i}, "F32", &.{W});
            },
            .mla => {
                const H: usize = c.mla_heads;
                try f.plain(a ++ "indexer.index_kpool_compress_gate", .{i}, "BF16", &.{ c.i_dim, D });
                try f.q(a ++ "q_a_proj", .{i}, c.q_lora, D);
                try f.q(a ++ "kv_a_proj_with_mqa", .{i}, c.kv_lora, D);
                try f.q(a ++ "indexer.wk", .{i}, c.i_dim, D);
                try f.q(a ++ "indexer.weights_proj", .{i}, c.i_heads, D);
                try f.q(a ++ "q_b_proj", .{i}, H * c.nope, c.q_lora);
                try f.q(a ++ "indexer.wq_b", .{i}, c.i_heads * c.i_dim, c.q_lora);
                if (f.mlx) {
                    try f.q3(a ++ "embed_q", .{i}, H, c.kv_lora, c.nope);
                    try f.q3(a ++ "unembed_out", .{i}, H, c.v_dim, c.kv_lora);
                } else try f.q(a ++ "kv_b_proj", .{i}, H * (c.nope + c.v_dim), c.kv_lora);
                try f.q(a ++ "o_proj", .{i}, D, H * c.v_dim);
                try f.plain(a ++ "q_a_layernorm.weight", .{i}, "BF16", &.{c.q_lora});
                try f.plain(a ++ "kv_a_layernorm.weight", .{i}, "BF16", &.{c.kv_lora});
                try f.plain(a ++ "indexer.k_norm.weight", .{i}, "BF16", &.{c.i_dim});
                try f.plain(a ++ "indexer.k_norm.bias", .{i}, "BF16", &.{c.i_dim});
                try f.plain(a ++ "indexer.index_kpool_compress_ape", .{i}, "BF16", &.{ c.kpool, c.i_dim });
            },
        }
        if (i == c.layers) {
            if (f.mlx) try f.plain(p ++ "eh_proj.weight", .{i}, "BF16", &.{ D, 2 * D }) else try f.q(p ++ "eh_proj", .{i}, D, 2 * D);
            for ([_][]const u8{ "enorm", "hnorm", "shared_head.norm" }) |x| try f.plain(p ++ "{s}.weight", .{ i, x }, "BF16", &.{D});
        }
        const m = p ++ "mlp.";
        if (c.isMoe(@intCast(i))) {
            try f.plain(m ++ "gate.weight", .{i}, if (f.fused) "F32" else "BF16", &.{ c.experts, D });
            try f.plain(m ++ "gate.e_score_correction_bias", .{i}, "F32", &.{c.experts});
            try f.q(m ++ "shared_experts.gate_proj", .{i}, c.moe_inter, D);
            try f.q(m ++ "shared_experts.up_proj", .{i}, c.moe_inter, D);
            try f.q(m ++ "shared_experts.down_proj", .{i}, D, c.moe_inter);
            if (f.mlx) for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |x| {
                const down = x[0] == 'd';
                try f.q3(m ++ "switch_mlp.{s}", .{ i, x }, c.experts, if (down) D else c.moe_inter, if (down) c.moe_inter else D);
            };
            if (!f.mlx) for (0..c.experts) |e| {
                try f.q(m ++ "experts.{d}.gate_proj", .{ i, e }, c.moe_inter, D);
                try f.q(m ++ "experts.{d}.up_proj", .{ i, e }, c.moe_inter, D);
                try f.q(m ++ "experts.{d}.down_proj", .{ i, e }, D, c.moe_inter);
            };
        } else {
            try f.q(m ++ "gate_proj", .{i}, c.dense_inter, D);
            try f.q(m ++ "up_proj", .{i}, c.dense_inter, D);
            try f.q(m ++ "down_proj", .{i}, D, c.dense_inter);
        }
    }

    /// config.json, the index and one shard (header only, its data a hole) in `dir`.
    fn write(f: *Fixture, tmp: std.testing.TmpDir, dir: []const u8, config: []const u8) !void {
        const c = f.c;
        try f.q("model.language_model.embed_tokens", .{}, c.vocab, c.hidden);
        try f.q("lm_head", .{}, c.vocab, c.hidden);
        try f.plain("model.language_model.norm.weight", .{}, "BF16", &.{c.hidden});
        for (0..c.run) |i| try f.layer(i);
        try f.layer(c.layers);
        var head: std.ArrayList(u8) = .empty;
        var index: std.ArrayList(u8) = .empty;
        try index.appendSlice(f.a, "{\"weight_map\": {");
        var at: usize = 0;
        for (f.list.items, 0..) |*e, j| {
            if (f.mlx) e.name = try mlxName(f.a, e.name);
            var n: usize = if (std.mem.eql(u8, e.dtype, "BF16")) 2 else 4;
            for (e.shape) |d| n *= d;
            const sep = if (j == 0) "{" else ",";
            try head.appendSlice(f.a, try std.fmt.allocPrint(f.a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ sep, e.name, e.dtype, try dims(f.a, e.shape), at, at + n }));
            try index.appendSlice(f.a, try std.fmt.allocPrint(f.a, "{s}\"{s}\": \"model-00001.safetensors\"", .{ if (j == 0) "" else ",", e.name }));
            at += n;
        }
        try head.appendSlice(f.a, "}");
        try index.appendSlice(f.a, "}}");
        const io = testing.io;
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = config });
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, head.items.len, .little);
        try tmp.dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = try std.mem.concat(f.a, u8, &.{ &len, head.items }) });
        const path = try std.fmt.allocPrintSentinel(f.a, "{s}/model-00001.safetensors", .{dir}, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return error.TestFile;
        defer _ = std.c.close(fd);
        if (std.c.ftruncate(fd, @intCast(8 + head.items.len + at)) != 0) return error.TestFile;
    }
};

/// An original-layout name as mlx-lm stores it.
fn mlxName(a: std.mem.Allocator, name: []const u8) ![]const u8 {
    if (std.mem.startsWith(u8, name, "lm_head.")) return std.mem.concat(a, u8, &.{ "language_model.", name });
    const short = name["model.language_model.".len..];
    const mtp = "layers.45.";
    if (std.mem.startsWith(u8, short, mtp)) {
        const rest = short[mtp.len..];
        if (std.mem.eql(u8, rest, "shared_head.norm.weight")) return "language_model.mtp.0.norm.weight";
        const top = std.mem.startsWith(u8, rest, "eh_proj.") or std.mem.startsWith(u8, rest, "enorm.") or std.mem.startsWith(u8, rest, "hnorm.");
        return std.mem.concat(a, u8, &.{ "language_model.mtp.0.", if (top) "" else "block.", rest });
    }
    inline for (.{ "attn", "ffn" }) |site| inline for (.{ "fn", "base", "scale" }) |part| {
        const old = ".hc_" ++ site ++ "_" ++ part;
        if (std.mem.endsWith(u8, short, old)) return std.mem.concat(a, u8, &.{ "language_model.model.", short[0 .. short.len - old.len], "." ++ site ++ "_hc." ++ part });
    };
    return std.mem.concat(a, u8, &.{ "language_model.model.", short });
}

fn dims(a: std.mem.Allocator, shape: []const usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (shape, 0..) |d, j| try out.appendSlice(a, try std.fmt.allocPrint(a, "{s}{d}", .{ if (j == 0) "" else ",", d }));
    return out.items;
}

/// A small GLM: four layers (three KDA and dense, one MLA and MoE) and the MTP layer, four routed experts.
fn small() cfg.Config {
    var c: cfg.Config = .{};
    c.run = 4;
    c.experts = 4;
    c.own = .{ 0, 4 };
    return c;
}

/// Every matrix's rows, columns and buffer offsets, and every other tensor's offset, hashed in load order.
fn digest(h: *std.hash.Wyhash, v: anytype) void {
    const T = @TypeOf(v);
    switch (@typeInfo(T)) {
        .@"struct" => |s| {
            if (@hasField(T, "buf") and @hasField(T, "off")) return h.update(std.mem.asBytes(&v.off));
            if (@hasField(T, "w") and @hasField(T, "n")) {
                h.update(std.mem.asBytes(&v.n));
                h.update(std.mem.asBytes(&v.k));
                inline for (.{ v.w, v.s, v.b }) |r| digest(h, r);
                return;
            }
            inline for (s.field_names) |name| digest(h, @field(v, name));
        },
        .@"union" => switch (v) {
            inline else => |x| digest(h, x),
        },
        .optional => if (v) |x| digest(h, x),
        .array => for (v) |x| digest(h, x),
        else => {},
    }
}

fn plan(w: *const wts.Weights, c: *const cfg.Config) u64 {
    var h = std.hash.Wyhash.init(0);
    inline for (.{ w.embed, w.head, w.norm, w.mtp }) |x| digest(&h, x);
    for (w.layers[0..c.run]) |l| digest(&h, l);
    digest(&h, w.layers[c.layers]);
    h.update(std.mem.asBytes(&w.bytes));
    return h.final();
}

const flat = "{\"quantization\": {\"group_size\": 64, \"bits\": 4, \"mode\": \"affine\"}}";

/// A dry load of a fixture stored at `stored` under `config`: the loader's plan, or its refusal.
fn dry(a: std.mem.Allocator, config: []const u8, stored: []const Pack) !*wts.Weights {
    return dryAs(a, config, stored, false);
}

fn dryAs(a: std.mem.Allocator, config: []const u8, stored: []const Pack, mlx: bool) !*wts.Weights {
    return dryWith(a, config, stored, mlx, false);
}

fn dryWith(a: std.mem.Allocator, config: []const u8, stored: []const Pack, mlx: bool, fused: bool) !*wts.Weights {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const c = small();
    var f: Fixture = .{ .a = a, .c = &c, .stored = stored, .mlx = mlx, .fused = fused };
    try f.write(tmp, dir, config);
    return wts.load(testing.allocator, undefined, dir, &c, 1, true);
}

fn free(w: *wts.Weights) void {
    w.deinit();
    testing.allocator.destroy(w);
}

test "a uniform 4/64 checkpoint plans the same matrices, offsets and bytes as the 4-bit-only loader" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = small();
    const w = try dry(arena.allocator(), flat, &.{});
    defer free(w);
    try testing.expectEqual(@as(usize, 1_512_774_688), w.bytes);
    try testing.expectEqual(@as(u64, 0xc1e86c075603176a), plan(w, &c));
    try testing.expectEqual(@as(u8, 0), w.layers[0].attn.kda.in_proj.more); // one width: one run, the matrix itself
}

/// An oQ-style table: o_proj of layer 0 at 8 bits under the short prefix, layer 3's shared up at 5 under the long.
const oq =
    \\{"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\ "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 8},
    \\ "language_model.model.layers.3.mlp.shared_experts.down_proj": {"group_size": 32, "bits": 5},
    \\ "language_model.mtp.0.block.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 2}}}
;
const oq_packed = [_]Pack{
    .{ "model.language_model.layers.0.self_attn.o_proj", .{ .bits = 8, .group = 64 } },
    .{ "model.language_model.layers.3.mlp.shared_experts.down_proj", .{ .bits = 5, .group = 32 } },
} ++ blk: {
    var out: [4]Pack = undefined;
    for (&out, 0..) |*o, e| o.* = .{ std.fmt.comptimePrint("model.language_model.layers.45.mlp.experts.{d}.gate_proj", .{e}), .{ .bits = 2, .group = 64 } };
    break :blk out;
};

test "the device admits each tensor at its override's width, whatever the key's prefix; a gate and up at two are refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectError(error.MixedWidths, dry(a, oq, &oq_packed)); // the MTP layer's gate at 2, its up at 4
    const both = oq[0 .. oq.len - 2] ++ ",\n \"language_model.mtp.0.block.mlp.switch_mlp.up_proj\": {\"group_size\": 64, \"bits\": 2}}}";
    const ups = comptime oq_packed ++ blk: {
        var out: [4]Pack = undefined;
        for (&out, 0..) |*o, e| o.* = .{ std.fmt.comptimePrint("model.language_model.layers.45.mlp.experts.{d}.up_proj", .{e}), .{ .bits = 2, .group = 64 } };
        break :blk out;
    };
    const c = small();
    const w = try dry(a, both, &ups);
    defer free(w);
    try testing.expectEqual(Width{ .bits = 8, .group = 64 }, w.layers[0].attn.kda.o_proj.width());
    try testing.expectEqual(Width{ .bits = 5, .group = 32 }, w.layers[3].mlp.moe.sh_down.width());
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, w.layers[c.layers].mlp.moe.gate.width());
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, w.layers[c.layers].mlp.moe.up.width());
    try testing.expectEqual(quant.uniform, w.layers[1].attn.kda.o_proj.width());
    try testing.expectEqual(quant.uniform, w.layers[3].mlp.moe.gate.width());
    try testing.expect(!wts.native(w, &c));
}

test "a tensor stored at another width than the config gives it is refused, never read at a guessed width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const eight = [_]Pack{.{ "model.language_model.layers.0.self_attn.o_proj", .{ .bits = 8, .group = 64 } }};
    try testing.expectError(error.UnexpectedTensor, dry(a, flat, &eight)); // global 4/64, stored at 8
    try testing.expectError(error.UnexpectedTensor, dry(a, oq, &.{})); // override 8, stored at 4
}

/// oQ4's split stacks on this fixture: the KDA input (q, v at 5; k, f_a, g_a, b at 8), the MLA x_proj (6 and 5).
const oq_split =
    \\{"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\ "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 5},
    \\ "model.layers.0.self_attn.k_proj": {"group_size": 64, "bits": 8},
    \\ "model.layers.0.self_attn.v_proj": {"group_size": 64, "bits": 5},
    \\ "model.layers.0.self_attn.f_a_proj": {"group_size": 64, "bits": 8},
    \\ "model.layers.0.self_attn.g_a_proj": {"group_size": 64, "bits": 8},
    \\ "model.layers.0.self_attn.b_proj": {"group_size": 64, "bits": 8},
    \\ "model.layers.3.self_attn.q_a_proj": {"group_size": 64, "bits": 6},
    \\ "model.layers.3.self_attn.kv_a_proj_with_mqa": {"group_size": 64, "bits": 6},
    \\ "model.layers.3.self_attn.indexer.wk": {"group_size": 64, "bits": 5},
    \\ "model.layers.3.self_attn.indexer.weights_proj": {"group_size": 64, "bits": 5}}}
;
const oq_split_stored = blk: {
    const bits = [_]struct { []const u8, u8 }{
        .{ "0.self_attn.q_proj", 5 },
        .{ "0.self_attn.k_proj", 8 },
        .{ "0.self_attn.v_proj", 5 },
        .{ "0.self_attn.f_a_proj", 8 },
        .{ "0.self_attn.g_a_proj", 8 },
        .{ "0.self_attn.b_proj", 8 },
        .{ "3.self_attn.q_a_proj", 6 },
        .{ "3.self_attn.kv_a_proj_with_mqa", 6 },
        .{ "3.self_attn.indexer.wk", 5 },
        .{ "3.self_attn.indexer.weights_proj", 5 },
    };
    var out: [bits.len]Pack = undefined;
    for (&out, bits) |*o, b| o.* = .{ "model.language_model.layers." ++ b[0], .{ .bits = b[1], .group = 64 } };
    break :blk out;
};

/// Run `i` of `q` is rows [row, row + n) at `bits`, its codes and tables its own, `pad`-row tiles apart.
fn expectRun(q: wts.Affine, i: usize, row: u32, n: u32, bits: u8, pad: usize) !void {
    const r = q.run(i);
    try testing.expectEqual(row, r.row);
    try testing.expectEqual(n, r.n);
    try testing.expectEqual(Width{ .bits = bits, .group = 64 }, Width{ .bits = r.bits, .group = r.group });
    const w: Width = .{ .bits = r.bits, .group = r.group };
    const room = std.mem.alignForward(usize, n, pad);
    try testing.expectEqual(std.mem.alignForward(usize, room * q.k * bits / 8, 256), r.s.off - r.w.off);
    try testing.expectEqual(std.mem.alignForward(usize, room * (q.k / w.group) * 2, 256), r.b.off - r.s.off);
}

test "a stack of mixed widths plans one run a member, as 0.6.6's QSplit calls each part, at its rows of the output" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = small();
    const w = try dry(arena.allocator(), oq_split, &oq_split_stored);
    defer free(w);
    const in = w.layers[0].attn.kda.in_proj;
    const W = c.kdaWidth();
    const dk = c.kda_dim;
    try testing.expectEqual(@as(u8, 5), in.more); // f_a, g_a and b share 8 bits yet stay apart
    try testing.expectEqual(c.kdaProj(), in.n);
    try expectRun(in, 0, 0, W, 5, 64);
    try expectRun(in, 1, W, W, 8, 64);
    try expectRun(in, 2, 2 * W, W, 5, 64);
    try expectRun(in, 3, 3 * W, dk, 8, 64);
    try expectRun(in, 4, 3 * W + dk, dk, 8, 64);
    try expectRun(in, 5, 3 * W + 2 * dk, c.kda_heads, 8, 64);
    const x = w.layers[3].attn.mla.x_proj;
    try testing.expectEqual(@as(u8, 3), x.more);
    try expectRun(x, 0, 0, c.q_lora, 6, 64);
    try expectRun(x, 1, c.q_lora, c.kv_lora, 6, 64);
    try expectRun(x, 2, c.q_lora + c.kv_lora, c.i_dim, 5, 64);
    try expectRun(x, 3, c.q_lora + c.kv_lora + c.i_dim, c.i_heads, 5, 64);
    try testing.expectEqual(@as(u8, 0), w.layers[1].attn.kda.in_proj.more);
    try testing.expectEqual(@as(u8, 0), w.layers[3].attn.mla.qr_proj.more);
    const t = wts.tally(w, &c);
    try testing.expectEqual(@as(u32, 4), t.count[5][1]); // q, v and the indexer's pair
    try testing.expectEqual(@as(u32, 2), t.count[6][1]);
    try testing.expectEqual(@as(u32, 4), t.count[8][1]);
    var buf: [256]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, t.summary(&buf), "5-bit/64 x4, 6-bit/64 x2, 8-bit/64 x4") != null);
}

/// oQ's spelling in mlx-lm's layout: the embedding at 8, a stack at 2, the absorbed pair at 5 and 6.
const oq_mlx =
    \\{"quantization": {"group_size": 64, "bits": 4, "mode": "affine",
    \\ "language_model.model.embed_tokens": {"group_size": 64, "bits": 8},
    \\ "language_model.model.layers.3.mlp.switch_mlp.down_proj": {"group_size": 64, "bits": 2},
    \\ "language_model.model.layers.3.self_attn.embed_q": {"group_size": 64, "bits": 5},
    \\ "language_model.mtp.0.block.self_attn.unembed_out": {"group_size": 64, "bits": 6},
    \\ "language_model.mtp.0.block.mlp.switch_mlp.gate_proj": {"group_size": 64, "bits": 2}}}
;
const oq_mlx_stored = [_]Pack{
    .{ "model.language_model.embed_tokens", .{ .bits = 8, .group = 64 } },
    .{ "model.language_model.layers.3.mlp.switch_mlp.down_proj", .{ .bits = 2, .group = 64 } },
    .{ "model.language_model.layers.3.self_attn.embed_q", .{ .bits = 5, .group = 64 } },
    .{ "model.language_model.layers.45.self_attn.unembed_out", .{ .bits = 6, .group = 64 } },
    .{ "model.language_model.layers.45.mlp.switch_mlp.gate_proj", .{ .bits = 2, .group = 64 } },
};

test "an mlx-lm checkpoint's every tensor is found, each at its own width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = small();
    const both = oq_mlx[0 .. oq_mlx.len - 2] ++ ",\n \"language_model.mtp.0.block.mlp.switch_mlp.up_proj\": {\"group_size\": 64, \"bits\": 2}}}";
    const up: Pack = .{ "model.language_model.layers.45.mlp.switch_mlp.up_proj", .{ .bits = 2, .group = 64 } };
    const w = try dryAs(arena.allocator(), both, &(oq_mlx_stored ++ [_]Pack{up}), true);
    defer free(w);
    try testing.expectEqual(Width{ .bits = 8, .group = 64 }, w.embed.width());
    try testing.expectEqual(quant.uniform, w.head.width());
    try testing.expectEqual(quant.uniform, w.layers[0].attn.kda.o_proj.width());
    const moe = w.layers[3].mlp.moe;
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, moe.down.width());
    try testing.expectEqual(quant.uniform, moe.gate.width());
    const mla = w.layers[3].attn.mla;
    try testing.expectEqual(@as(u32, 0), mla.kv_b.n);
    try testing.expectEqual(Width{ .bits = 5, .group = 64 }, mla.absorbed.?[0].width());
    try testing.expectEqual(@as(u32, c.mla_heads * c.kv_lora), mla.absorbed.?[0].n);
    try testing.expectEqual(@as(u32, c.nope), mla.absorbed.?[0].k);
    try testing.expectEqual(quant.uniform, mla.absorbed.?[1].width());
    try testing.expectEqual(Width{ .bits = 6, .group = 64 }, w.layers[c.layers].attn.mla.absorbed.?[1].width());
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, w.layers[c.layers].mlp.moe.gate.width());
    try testing.expectEqual(Width{ .bits = 2, .group = 64 }, w.layers[c.layers].mlp.moe.up.width());
    try testing.expect(w.mtp.?.eh_bf16 != null);
}

test "the device loads mlx-lm's absorbed pair and bf16 eh_proj; a routed gate and up at two widths are refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectError(error.MixedWidths, dryAs(a, oq_mlx, &oq_mlx_stored, true)); // the MTP gate at 2, up at 4
    const c = small();
    const w = try dryAs(a, flat, &.{}, true);
    defer free(w);
    try testing.expectEqual(quant.uniform, w.layers[3].attn.mla.absorbed.?[0].width());
    try testing.expect(w.mtp.?.eh_bf16 != null);
    try testing.expect(!wts.native(w, &c)); // the pair and the bf16 eh_proj take the row kernels
}

test "a uniform checkpoint in the original layout needs no other width's pipelines" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const c = small();
    const w = try dry(arena.allocator(), flat, &.{});
    defer free(w);
    try testing.expect(wts.native(w, &c));
}

test "mlx-lm's one conv1d tensor and fp32 mixes and router plan as the three conv tensors and bf16 ones do" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = small();
    const w = try dryWith(a, flat, &.{}, true, true);
    defer free(w);
    const v = try dryWith(a, flat, &.{}, true, false);
    defer free(v);
    try testing.expectEqual(plan(v, &c), plan(w, &c)); // the same buffers, offsets and bytes
}
