//! A rank's weights as named host tensors in the reference loader's forms, named as the reference capture names them.

const std = @import("std");
const plan = @import("plan.zig");
const names = @import("names.zig");
const stage = @import("stage.zig");
const Pack = @import("pack.zig").Pack;
const DType = @import("pack.zig").DType;
const Io = std.Io;

pub const Kind = enum { f32, bf16, f16, i16 };

/// One tensor the forward holds: its name (`L2.attn.wq_a.trellis`, `embed`), element kind, shape and bytes.
pub const Named = struct {
    name: []const u8,
    kind: Kind,
    shape: [4]usize,
    rank: u8,
    bytes: []align(16) u8,
};

/// What a native tensor becomes (Loader.native and its callers).
const Out = enum {
    /// fp32 values (`native(name)`: bf16 or fp32 storage widened)
    f32,
    /// bf16 values (`native(name, bf16, bf16_param=True)`: the router gate)
    bf16,
    /// fp32 holding bf16-rounded values (`native(name, bf16_param=True)`: the indexer's weights_proj)
    f32_via_bf16,
    /// Engram's q / k: combined into `engram.qk` = bf16(q) * bf16(k) in fp32
    qk,
};

const Rule = struct { name: []const u8, out: Out };

/// Native suffixes (names.zig blockNatives) -> the capture's field names and forms.
const rules = std.StaticStringMap(Rule).initComptime(.{
    .{ "attn_norm.weight", Rule{ .name = "attn_norm", .out = .f32 } },
    .{ "ffn_norm.weight", Rule{ .name = "ffn_norm", .out = .f32 } },
    .{ "hc_attn_fn", Rule{ .name = "hc_attn.fn", .out = .f32 } },
    .{ "hc_attn_base", Rule{ .name = "hc_attn.base", .out = .f32 } },
    .{ "hc_attn_scale", Rule{ .name = "hc_attn.scale", .out = .f32 } },
    .{ "hc_ffn_fn", Rule{ .name = "hc_ffn.fn", .out = .f32 } },
    .{ "hc_ffn_base", Rule{ .name = "hc_ffn.base", .out = .f32 } },
    .{ "hc_ffn_scale", Rule{ .name = "hc_ffn.scale", .out = .f32 } },
    .{ "attn.q_norm.weight", Rule{ .name = "attn.q_norm", .out = .f32 } },
    .{ "attn.kv_norm.weight", Rule{ .name = "attn.kv_norm", .out = .f32 } },
    .{ "attn.attn_sink", Rule{ .name = "attn.sink", .out = .f32 } },
    .{ "attn.compressor.norm.weight", Rule{ .name = "attn.comp_norm", .out = .f32 } },
    .{ "attn.indexer.k_norm.weight", Rule{ .name = "attn.ix_knorm", .out = .f32 } },
    .{ "attn.indexer.weights_proj.weight", Rule{ .name = "attn.ix_wp", .out = .f32_via_bf16 } },
    .{ "ffn.gate.weight", Rule{ .name = "moe.gate", .out = .bf16 } },
    .{ "ffn.gate.bias", Rule{ .name = "moe.bias", .out = .f32 } },
    .{ "engram.q_weight", Rule{ .name = "engram.qk", .out = .qk } },
    .{ "engram.k_weight", Rule{ .name = "engram.qk", .out = .qk } },
});

/// The capture's name of a block group (AttnW / MoEW / EngramW fields).
fn groupField(g: plan.GroupPlan, gl: u32, buf: []u8) ![]const u8 {
    return switch (g.proj) {
        .wq_a => "attn.wq_a",
        .wkv => "attn.wkv",
        .wq_b => "attn.wq_b",
        .wo_a => std.fmt.bufPrint(buf, "attn.wo_a.{d}", .{g.slice % gl}),
        .wo_b => "attn.wo_b",
        .comp_wkv => "attn.comp_wkv",
        .comp_wgate => "attn.comp_wgate",
        .ix_wq_b => "attn.ix_wq_b",
        .ix_wk => "attn.ix_wk",
        .shared_w1 => "moe.shared.0.w1",
        .shared_w2 => "moe.shared.0.w2",
        .shared_w3 => "moe.shared.0.w3",
        .engram_wkv => "engram.wkv",
    };
}

/// fp32 -> bf16, round to nearest even (torch's .to(bfloat16)); NaN stays a quiet NaN.
pub fn bf16Round(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

pub fn bf16Widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// One stored element as fp32 (bf16, fp16 and fp32 storage).
fn load(dtype: DType, src: []const u8, i: usize) !f32 {
    return switch (dtype) {
        .f32 => @bitCast(std.mem.readInt(u32, src[4 * i ..][0..4], .little)),
        .bf16 => bf16Widen(std.mem.readInt(u16, src[2 * i ..][0..2], .little)),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, src[2 * i ..][0..2], .little)))),
        else => error.UnexpectedTensor,
    };
}

pub const Builder = struct {
    gpa: std.mem.Allocator,
    io: Io,
    pack: *const Pack,
    out: std.ArrayList(Named) = .empty,

    pub fn deinit(b: *Builder) void {
        for (b.out.items) |n| {
            b.gpa.free(n.name);
            b.gpa.free(n.bytes);
        }
        b.out.deinit(b.gpa);
    }

    fn put(b: *Builder, name: []const u8, kind: Kind, shape: []const usize, bytes: []align(16) u8) !void {
        var n: Named = .{ .name = try b.gpa.dupe(u8, name), .kind = kind, .shape = @splat(1), .rank = @intCast(shape.len), .bytes = bytes };
        @memcpy(n.shape[0..shape.len], shape);
        try b.out.append(b.gpa, n);
    }

    fn copy(b: *Builder, src: []const u8) ![]align(16) u8 {
        const d = try b.gpa.alignedAlloc(u8, .@"16", src.len);
        @memcpy(d, src);
        return d;
    }

    /// A group's part as `<base>.trellis`, `.suh`, `.svh`.
    fn group(b: *Builder, base: []const u8, g: plan.GroupPlan) !void {
        var img = try stage.group(b.gpa, b.io, b.pack, g);
        defer img.deinit(b.gpa);
        const p = g.part.group;
        var nb: [160]u8 = undefined;
        try b.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), .i16, &.{ p.k_tiles, p.n_tiles, p.words }, try b.copy(img.trellis));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.suh", .{base}), .f16, &.{p.k()}, try b.copy(img.suh));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.svh", .{base}), .f16, &.{p.n()}, try b.copy(img.svh));
    }

    /// A routed projection: `.trellis` [E, kt, nt, words] (stack) or `.data` (ragged), `.suh` [E, K], `.svh` [E, N].
    fn experts(b: *Builder, base: []const u8, x: *const plan.ExpertsPlan) !void {
        var img = try stage.experts(b.gpa, b.io, b.pack, x);
        defer img.deinit(b.gpa);
        const l = &x.layout;
        const e = l.count();
        var nb: [160]u8 = undefined;
        if (l.kind == .stack) {
            try b.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), .i16, &.{ e, l.k_tiles, l.n_tiles, l.words[0] }, try b.copy(img.trellis));
        } else try b.put(try std.fmt.bufPrint(&nb, "{s}.data", .{base}), .i16, &.{@intCast(l.elements)}, try b.copy(img.trellis));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.suh", .{base}), .f16, &.{ e, 16 * @as(usize, l.k_tiles) }, try b.copy(img.suh));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.svh", .{base}), .f16, &.{ e, 16 * @as(usize, l.n_tiles) }, try b.copy(img.svh));
    }

    /// The rank's slice of a native tensor, as stored.
    fn raw(b: *Builder, v: plan.NativePlan, rank: u32, world: u32) ![]u8 {
        const whole = v.slice == .whole;
        const len = if (whole) v.info.nbytes else v.info.nbytes / world;
        const bytes = try b.gpa.alloc(u8, @intCast(len));
        errdefer b.gpa.free(bytes);
        try b.pack.read(b.io, v.info, .{ .offset = if (whole) 0 else rank * len, .len = len }, bytes);
        return bytes;
    }

    /// A native tensor in its forward form; Engram's q waits for its k (`pending`).
    fn native(b: *Builder, name: []const u8, rule: Rule, v: plan.NativePlan, rank: u32, world: u32, pending: *?[]f32) !void {
        const bytes = try b.raw(v, rank, world);
        defer b.gpa.free(bytes);
        const count = bytes.len / v.info.dtype.size();
        var shape: [4]usize = @splat(1);
        const dims = v.info.dims();
        @memcpy(shape[0..dims.len], dims);
        if (v.slice != .whole) shape[0] /= world;
        switch (rule.out) {
            .bf16 => {
                const out = try b.gpa.alignedAlloc(u8, .@"16", 2 * count);
                for (0..count) |i| std.mem.writeInt(u16, out[2 * i ..][0..2], bf16Round(try load(v.info.dtype, bytes, i)), .little);
                try b.put(name, .bf16, shape[0..dims.len], out);
            },
            .f32, .f32_via_bf16 => {
                const out = try b.gpa.alignedAlloc(u8, .@"16", 4 * count);
                for (0..count) |i| {
                    const x = try load(v.info.dtype, bytes, i);
                    const y = if (rule.out == .f32) x else bf16Widen(bf16Round(x));
                    std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(y), .little);
                }
                try b.put(name, .f32, shape[0..dims.len], out);
            },
            .qk => {
                const vals = try b.gpa.alloc(f32, count);
                for (vals, 0..) |*x, i| x.* = bf16Widen(bf16Round(try load(v.info.dtype, bytes, i)));
                if (pending.*) |q| {
                    defer b.gpa.free(q);
                    defer b.gpa.free(vals);
                    if (q.len != vals.len) return error.UnexpectedTensor;
                    const out = try b.gpa.alignedAlloc(u8, .@"16", 4 * count);
                    for (q, vals, 0..) |x, y, i| std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(x * y), .little);
                    pending.* = null;
                    try b.put(name, .f32, shape[0..dims.len], out);
                } else pending.* = vals;
            },
        }
    }

    pub fn layer(b: *Builder, l: *const plan.LayerPlan, o_groups: u32, rank: u32, world: u32) !void {
        var nb: [160]u8 = undefined;
        var fb: [32]u8 = undefined;
        const gl = o_groups / world;
        for (l.groups) |g| try b.group(try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ l.index, try groupField(g, gl, &fb) }), g);
        var pending: ?[]f32 = null;
        defer if (pending) |q| b.gpa.free(q);
        for (l.natives) |v| {
            const suffix = v.name[l.prefix.len + 1 ..];
            const rule = rules.get(suffix) orelse return error.UnnamedNative;
            try b.native(try std.fmt.bufPrint(&nb, "L{d}.{s}", .{ l.index, rule.name }), rule, v, rank, world, &pending);
        }
        if (pending != null) return error.EngramHalf;
        for (&l.experts, [_][]const u8{ "w1", "w2", "w3" }) |*x, w| try b.experts(try std.fmt.bufPrint(&nb, "L{d}.moe.{s}", .{ l.index, w }), x);
    }

    /// The vocabulary rows (bf16), the final norm (fp32) and the head's part.
    pub fn top(b: *Builder, p: *const plan.Plan) !void {
        var none: ?[]f32 = null;
        for (p.top) |v| {
            const is_embed = std.mem.eql(u8, v.name, "embed.weight");
            const rule: Rule = if (is_embed) .{ .name = "embed", .out = .bf16 } else .{ .name = "norm", .out = .f32 };
            try b.native(rule.name, rule, v, p.rank, p.world, &none);
        }
        if (p.head) |h| try b.group("head", h);
    }

    /// DSpark's heads: main_proj (whole), main_norm and norm fp32, Markov embed / head bf16, confidence head fp32.
    pub fn dspark(b: *Builder, p: *const plan.Plan) !void {
        const d = p.dspark orelse return;
        var none: ?[]f32 = null;
        try b.group("dspark.main_proj", d.main_proj);
        const out_names = [_][]const u8{ "dspark.main_norm", "dspark.norm", "dspark.markov.w1", "dspark.markov.w2" };
        const outs = [_]Out{ .f32, .f32, .bf16, .bf16 };
        for (d.natives, out_names, outs) |v, nm, o| try b.native(nm, .{ .name = nm, .out = o }, v, p.rank, p.world, &none);
        if (d.confidence) |c| try b.native("dspark.conf", .{ .name = "dspark.conf", .out = .f32 }, c, p.rank, p.world, &none);
    }

    /// Every planned block, then the model-level tensors (a whole-model plan only).
    pub fn all(b: *Builder, p: *const plan.Plan, o_groups: u32) !void {
        for (p.layers) |*l| try b.layer(l, o_groups, p.rank, p.world);
        try b.top(p);
        try b.dspark(p);
    }

    pub fn find(b: *const Builder, name: []const u8) ?*const Named {
        for (b.out.items) |*n| if (std.mem.eql(u8, n.name, name)) return n;
        return null;
    }
};

const testing = std.testing;

test "the loader's conversions: bf16 round to nearest even, widening, fp16 storage" {
    try testing.expectEqual(@as(u16, 0x3f80), bf16Round(1.0));
    // 1 + 2^-8 is a tie between 1.0 and 1 + 2^-7: even keeps 1.0; 1 + 3 * 2^-8 rounds up
    try testing.expectEqual(@as(u16, 0x3f80), bf16Round(1.00390625));
    try testing.expectEqual(@as(u16, 0x3f82), bf16Round(1.01171875));
    try testing.expectEqual(@as(f32, -2.5), bf16Widen(bf16Round(-2.5)));
    var h: [2]u8 = undefined;
    std.mem.writeInt(u16, &h, @bitCast(@as(f16, 0.333251953125)), .little);
    try testing.expectEqual(@as(f32, 0.333251953125), try load(.f16, &h, 0));
}
