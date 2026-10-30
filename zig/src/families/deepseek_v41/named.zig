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
    /// Engram q/k: decode the stored F32/BF16/F16 values, then multiply in FP32.
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
    metadata_only: bool = false,

    pub fn deinit(b: *Builder) void {
        for (b.out.items) |n| {
            b.gpa.free(n.name);
            b.gpa.free(n.bytes);
        }
        b.out.deinit(b.gpa);
    }

    fn put(b: *Builder, name: []const u8, kind: Kind, shape: []const usize, bytes: []align(16) u8) !void {
        errdefer b.gpa.free(bytes);
        try b.out.ensureUnusedCapacity(b.gpa, 1);
        const key = try b.gpa.dupe(u8, name);
        var n: Named = .{ .name = key, .kind = kind, .shape = @splat(1), .rank = @intCast(shape.len), .bytes = bytes };
        @memcpy(n.shape[0..shape.len], shape);
        b.out.appendAssumeCapacity(n);
    }

    fn copy(b: *Builder, src: []const u8) ![]align(16) u8 {
        const d = try b.gpa.alignedAlloc(u8, .@"16", src.len);
        @memcpy(d, src);
        return d;
    }

    /// A group's part as `<base>.trellis`, `.suh`, `.svh`.
    fn group(b: *Builder, base: []const u8, g: plan.GroupPlan) !void {
        var img: stage.GroupImage = if (b.metadata_only) .{ .trellis = &.{}, .suh = &.{}, .svh = &.{} } else try stage.group(b.gpa, b.io, b.pack, g);
        defer img.deinit(b.gpa);
        const p = g.part.group;
        var nb: [160]u8 = undefined;
        try b.put(try std.fmt.bufPrint(&nb, "{s}.trellis", .{base}), .i16, &.{ p.k_tiles, p.n_tiles, p.words }, try b.copy(img.trellis));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.suh", .{base}), .f16, &.{p.k()}, try b.copy(img.suh));
        try b.put(try std.fmt.bufPrint(&nb, "{s}.svh", .{base}), .f16, &.{p.n()}, try b.copy(img.svh));
    }

    /// A routed projection: `.trellis` [E, kt, nt, words] (stack) or `.data` (ragged), `.suh` [E, K], `.svh` [E, N].
    fn experts(b: *Builder, base: []const u8, x: *const plan.ExpertsPlan) !void {
        var img: stage.ExpertsImage = if (b.metadata_only) .{ .trellis = &.{}, .suh = &.{}, .svh = &.{}, .k2 = &.{}, .offsets = &.{} } else try stage.experts(b.gpa, b.io, b.pack, x);
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
        const s = if (v.slice == .vocab_rows) try @import("partition.zig").share(@intCast(v.info.shape[0]), rank, world) else null;
        const len = if (s) |part| v.info.nbytes / v.info.shape[0] * part.len else if (whole) v.info.nbytes else v.info.nbytes / world;
        const offset = if (s) |part| v.info.nbytes / v.info.shape[0] * part.lo else if (whole) 0 else rank * len;
        const bytes = try b.gpa.alloc(u8, @intCast(len));
        errdefer b.gpa.free(bytes);
        try b.pack.read(b.io, v.info, .{ .offset = offset, .len = len }, bytes);
        return bytes;
    }

    /// A native tensor in its forward form; Engram's q waits for its k (`pending`).
    fn native(b: *Builder, name: []const u8, rule: Rule, v: plan.NativePlan, rank: u32, world: u32, pending: *?[]f32) !void {
        var shape: [4]usize = @splat(1);
        const dims = v.info.dims();
        @memcpy(shape[0..dims.len], dims);
        if (v.slice == .vocab_rows) shape[0] = (try @import("partition.zig").share(@intCast(shape[0]), rank, world)).len else if (v.slice != .whole) shape[0] /= world;
        if (v.info.dtype != .f32 and v.info.dtype != .bf16 and v.info.dtype != .f16) return error.UnexpectedTensor;
        if (b.metadata_only) {
            if (rule.out == .qk and std.mem.endsWith(u8, v.name, "q_weight")) return;
            return b.put(name, if (rule.out == .bf16) .bf16 else .f32, shape[0..dims.len], try b.copy(&.{}));
        }
        const bytes = try b.raw(v, rank, world);
        defer b.gpa.free(bytes);
        const count = bytes.len / v.info.dtype.size();
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
                for (vals, 0..) |*x, i| x.* = try load(v.info.dtype, bytes, i);
                if (pending.*) |q| {
                    pending.* = null;
                    defer b.gpa.free(q);
                    defer b.gpa.free(vals);
                    if (q.len != vals.len) return error.UnexpectedTensor;
                    const out = try b.gpa.alignedAlloc(u8, .@"16", 4 * count);
                    for (q, vals, 0..) |x, y, i| std.mem.writeInt(u32, out[4 * i ..][0..4], @bitCast(x * y), .little);
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
test "named image publication consumes its owned input on every allocation failure" {
    const control = struct {
        fn run(a: std.mem.Allocator) !void {
            var b: Builder = .{ .gpa = a, .io = std.testing.io, .pack = undefined };
            defer b.deinit();
            const bytes = try a.alignedAlloc(u8, .@"16", 16);
            try b.put("owned", .bf16, &.{8}, bytes);
        }
    };
    try testing.checkAllAllocationFailures(testing.allocator, control.run, .{});
}

test "Engram native qk partial allocation has one pending owner" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = try @import("pack.zig").image(a, &.{.{ .name = "q", .dtype = "F32", .shape = &.{4} }});
    defer a.free(data);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = data });
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);
    var pack = try Pack.open(a, testing.io, dir);
    defer pack.deinit();
    const control = struct {
        fn run(gpa: std.mem.Allocator, p: *const Pack) !void {
            var b: Builder = .{ .gpa = gpa, .io = testing.io, .pack = p };
            defer b.deinit();
            var pending: ?[]f32 = null;
            defer if (pending) |owned| gpa.free(owned);
            const tensor: plan.NativePlan = .{ .name = "q", .info = p.get("q").?, .slice = .whole, .bytes = 16 };
            try b.native("qk", .{ .name = "qk", .out = .qk }, tensor, 0, 1, &pending);
            try b.native("qk", .{ .name = "qk", .out = .qk }, tensor, 0, 1, &pending);
        }
    };
    try testing.checkAllAllocationFailures(a, control.run, .{&pack});
}

test "Engram qk decodes stored F32 BF16 F16 before an independently widened product" {
    const a = testing.allocator;
    inline for (.{ "F32", "BF16", "F16" }) |dtype| {
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        const data = try @import("pack.zig").image(a, &.{.{ .name = "q", .dtype = dtype, .shape = &.{1} }});
        defer a.free(data);
        const value: f32 = 1.001;
        const size = (DType.parse(dtype).?).size();
        const expected: f64 = if (size == 4) value else if (comptime std.mem.eql(u8, dtype, "BF16")) bf16Widen(bf16Round(value)) else @as(f16, @floatCast(value));
        if (size == 4) std.mem.writeInt(u32, data[data.len - 4 ..][0..4], @bitCast(value), .little) else std.mem.writeInt(u16, data[data.len - 2 ..][0..2], if (comptime std.mem.eql(u8, dtype, "BF16")) bf16Round(value) else @bitCast(@as(f16, @floatCast(value))), .little);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = data });
        const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
        defer a.free(dir);
        var p = try Pack.open(a, testing.io, dir);
        defer p.deinit();
        var b: Builder = .{ .gpa = a, .io = testing.io, .pack = &p };
        defer b.deinit();
        var pending: ?[]f32 = null;
        defer if (pending) |q| a.free(q);
        const tensor: plan.NativePlan = .{ .name = "q", .info = p.get("q").?, .slice = .whole, .bytes = size };
        try b.native("qk", .{ .name = "qk", .out = .qk }, tensor, 0, 1, &pending);
        try b.native("qk", .{ .name = "qk", .out = .qk }, tensor, 0, 1, &pending);
        const got: f32 = @bitCast(std.mem.readInt(u32, b.out.items[0].bytes[0..4], .little));
        // Independent scalar FP64 gate reference: RMS-normalized h=2, k=3, D=1, value=5, residual=2.
        const norm: f64 = 1.0 / @sqrt(4.0 + 1e-6) / @sqrt(9.0 + 1e-6);
        const ideal_gate = 1 / (1 + @exp(-@sqrt(6 * expected * expected * norm)));
        const image_gate = 1 / (1 + @exp(-@sqrt(6 * @as(f64, got) * norm)));
        try testing.expectApproxEqAbs(2 + 5 * ideal_gate, 2 + 5 * image_gate, 1e-6);
        if (size == 4) {
            const old_weight = bf16Widen(bf16Round(value)) * bf16Widen(bf16Round(value));
            const old_gate = 1 / (1 + @exp(-@sqrt(6 * @as(f64, old_weight) * norm)));
            try testing.expect(@abs(ideal_gate - old_gate) > 1e-5);
        }
        try testing.expectEqual(@as(f32, @floatCast(expected * expected)), got);
        if (size == 4) try testing.expect(got != bf16Widen(bf16Round(value)) * bf16Widen(bf16Round(value)));
        pending = try a.alloc(f32, 2);
        try testing.expectError(error.UnexpectedTensor, b.native("qk", .{ .name = "qk", .out = .qk }, tensor, 0, 1, &pending));
        try testing.expect(pending == null);
    }
}

test "an F16 router bias and confidence head reach the forward as exactly widened fp32" {
    const a = testing.allocator;
    const values = [_]f16{ 0.333251953125, -2.5, 65504, 0x1p-24, -0.0 };
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const data = try @import("pack.zig").image(a, &.{.{ .name = "b", .dtype = "F16", .shape = &.{values.len} }});
    defer a.free(data);
    for (values, 0..) |v, i| std.mem.writeInt(u16, data[data.len - 2 * values.len ..][2 * i ..][0..2], @bitCast(v), .little);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = data });
    const dir = try tmp.dir.realPathFileAlloc(testing.io, ".", a);
    defer a.free(dir);
    var p = try Pack.open(a, testing.io, dir);
    defer p.deinit();
    var b: Builder = .{ .gpa = a, .io = testing.io, .pack = &p };
    defer b.deinit();
    var none: ?[]f32 = null;
    const tensor: plan.NativePlan = .{ .name = "b", .info = p.get("b").?, .slice = .whole, .bytes = 2 * values.len };
    try b.native("L0.moe.bias", rules.get("ffn.gate.bias").?, tensor, 0, 1, &none);
    try b.native("dspark.conf", .{ .name = "dspark.conf", .out = .f32 }, tensor, 0, 1, &none);
    for (b.out.items) |n| {
        try testing.expectEqual(Kind.f32, n.kind);
        for (values, 0..) |v, i| try testing.expectEqual(@as(u32, @bitCast(@as(f32, v))), std.mem.readInt(u32, n.bytes[4 * i ..][0..4], .little));
    }
}
