//! Kolibri-1's checkpoint tensors by name, each checked for dtype and shape before any upload.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;

const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;

pub const block = core.quant.fp8.block; // the checkpoint's scale block, both ways

/// A block-FP8 projection as stored: e4m3 codes [n, k] and fp32 scale_inv [ceil(n/128), ceil(k/128)].
pub const Fp8 = struct { w: Tensor, s: Tensor };
pub const Expert = struct { gate: Fp8, up: Fp8, down: Fp8 };
/// An attention projection: block FP8 (Kolibri-1), or bf16 rows with no scales (Kolibri 1.1).
pub const Proj = struct { w: Tensor, s: ?Tensor };

pub const Layer = struct {
    input_norm: Tensor,
    q: Proj,
    k: ?Proj, // null: the layer reads another layer's cache (kv_source_layers)
    v: ?Proj,
    q_norm: Tensor,
    k_norm: ?Tensor,
    o: Proj,
    post_attn_norm: Tensor,
    pre_moe_norm: Tensor, // post_attention_layernorm: the norm before the MoE, despite its name
    router: ?Tensor, // [E, D] bf16; null: a dense layer
    bias: ?Tensor, // [E] expert_bias
    experts: []Expert, // E routed, then the shared expert; a dense layer: its MLP alone
    post_moe_norm: Tensor,
};

pub const Map = struct {
    gpa: std.mem.Allocator,
    embed: Tensor,
    layers: []Layer,
    norm: Tensor,
    head: Tensor,

    pub fn deinit(m: *Map) void {
        for (m.layers) |l| m.gpa.free(l.experts);
        m.gpa.free(m.layers);
        m.* = undefined;
    }
};

const Names = struct {
    ck: *Checkpoint,
    buf: [160]u8 = undefined,

    fn name(self: *Names, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, fmt, args);
    }

    fn bf16(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.ck.expect(try self.name(fmt, args), .bf16, shape);
    }

    fn fp8(self: *Names, n: usize, k: usize, comptime fmt: []const u8, args: anytype) !Fp8 {
        const h = try core.quant.fp8.read(self.ck, try self.name(fmt, args), n, k, .f32, null);
        return .{ .w = h.weight, .s = h.scale_inv };
    }

    fn proj(self: *Names, c: Config, n: usize, k: usize, comptime fmt: []const u8, args: anytype) !Proj {
        if (c.v11) return .{ .w = try self.bf16(&.{ n, k }, fmt ++ ".weight", args), .s = null };
        const f = try self.fp8(n, k, fmt, args);
        return .{ .w = f.w, .s = f.s };
    }

    fn expert(self: *Names, c: Config, width: usize, comptime pre: []const u8, args: anytype) !Expert {
        return .{
            .gate = try self.fp8(width, c.hidden, pre ++ ".gate_proj", args),
            .up = try self.fp8(width, c.hidden, pre ++ ".up_proj", args),
            .down = try self.fp8(c.hidden, width, pre ++ ".down_proj", args),
        };
    }
};

/// Every tensor the loader reads, in its order; a missing, mistyped or leftover tensor refuses the checkpoint.
pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    if (c.shared_width != c.moe_width) {
        std.log.err("shared expert {d} wide, routed {d}: the experts stack only at one width", .{ c.shared_width, c.moe_width });
        return error.UnsupportedShapes;
    }
    var nm: Names = .{ .ck = ck };
    const d = c.hidden;
    var m: Map = .{ .gpa = gpa, .embed = undefined, .layers = try gpa.alloc(Layer, c.layers), .norm = undefined, .head = undefined };
    var done: usize = 0;
    errdefer {
        for (m.layers[0..done]) |l| gpa.free(l.experts);
        gpa.free(m.layers);
    }
    m.embed = try nm.bf16(&.{ c.vocab, d }, "model.embed_tokens.weight", .{});
    for (m.layers, 0..) |*l, i| {
        const dense = c.dense[i];
        const ex = try gpa.alloc(Expert, if (dense) 1 else c.experts + 1);
        errdefer gpa.free(ex);
        if (dense) ex[0] = try nm.expert(c, c.dense_width, "model.layers.{d}.mlp", .{i}) else {
            for (ex[0..c.experts], 0..) |*e, j| e.* = try nm.expert(c, c.moe_width, "model.layers.{d}.mlp.experts.{d}", .{ i, j });
            ex[c.experts] = try nm.expert(c, c.moe_width, "model.layers.{d}.mlp.shared_experts", .{i});
        }
        const qd = c.heads * c.head_dim;
        const kvd = c.kv_heads * c.head_dim;
        const own = c.kv_from[i] == null; // a layer reading another's cache has no k, v or k_norm
        l.* = .{
            .input_norm = try nm.bf16(&.{d}, "model.layers.{d}.input_layernorm.weight", .{i}),
            .q = try nm.proj(c, qd, d, "model.layers.{d}.self_attn.q_proj", .{i}),
            .k = if (own) try nm.proj(c, kvd, d, "model.layers.{d}.self_attn.k_proj", .{i}) else null,
            .v = if (own) try nm.proj(c, kvd, d, "model.layers.{d}.self_attn.v_proj", .{i}) else null,
            .q_norm = try nm.bf16(&.{c.head_dim}, "model.layers.{d}.self_attn.q_norm.weight", .{i}),
            .k_norm = if (own) try nm.bf16(&.{c.head_dim}, "model.layers.{d}.self_attn.k_norm.weight", .{i}) else null,
            .o = try nm.proj(c, d, qd, "model.layers.{d}.self_attn.o_proj", .{i}),
            .post_attn_norm = try nm.bf16(&.{d}, "model.layers.{d}.post_attn_norm.weight", .{i}),
            .pre_moe_norm = try nm.bf16(&.{d}, "model.layers.{d}.post_attention_layernorm.weight", .{i}),
            .router = if (dense) null else try nm.bf16(&.{ c.experts, d }, "model.layers.{d}.mlp.gate.weight", .{i}),
            .bias = if (dense) null else try nm.bf16(&.{c.experts}, "model.layers.{d}.moe.router.expert_bias", .{i}),
            .experts = ex,
            .post_moe_norm = try nm.bf16(&.{d}, "model.layers.{d}.post_ffn_norm.weight", .{i}),
        };
        done += 1;
    }
    m.norm = try nm.bf16(&.{d}, "model.norm.weight", .{});
    m.head = try nm.bf16(&.{ c.vocab, d }, "lm_head.weight", .{});
    if (ck.unused() != 0) return error.UnusedTensors;
    return m;
}

const testing = std.testing;
const io = testing.io;

const tiny: Config = .{ .layers = 2, .hidden = 256, .heads = 2, .kv_heads = 1, .head_dim = 128, .vocab = 8, .experts = 2, .top_k = 1, .moe_width = 128, .shared_width = 128, .window = 4, .rope_theta = 1e4, .eps = 1e-6 };

/// A zero-filled single-file checkpoint with each entry's dtype and shape.
fn writeTiny(a: std.mem.Allocator, tmp: testing.TmpDir, c: Config) !void {
    var entries: std.ArrayList(struct { name: []const u8, dtype: []const u8, shape: []const usize }) = .empty;
    const Add = struct {
        fn fp8(list: anytype, al: std.mem.Allocator, pre: []const u8, n: usize, k: usize) !void {
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight", .{pre}), .dtype = "F8_E4M3", .shape = try al.dupe(usize, &.{ n, k }) });
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale_inv", .{pre}), .dtype = "F32", .shape = try al.dupe(usize, &.{ core.quant.fp8.blocks(n), core.quant.fp8.blocks(k) }) });
        }
        fn bf(list: anytype, al: std.mem.Allocator, name: []const u8, shape: []const usize) !void {
            try list.append(al, .{ .name = name, .dtype = "BF16", .shape = try al.dupe(usize, shape) });
        }
    };
    const d = c.hidden;
    try Add.bf(&entries, a, "model.embed_tokens.weight", &.{ c.vocab, d });
    try Add.bf(&entries, a, "model.norm.weight", &.{d});
    try Add.bf(&entries, a, "lm_head.weight", &.{ c.vocab, d });
    for (0..c.layers) |i| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        for ([_][]const u8{ "input_layernorm", "post_attn_norm", "post_attention_layernorm", "post_ffn_norm" }) |n|
            try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}{s}.weight", .{ p, n }), &.{d});
        const own = c.kv_from[i] == null;
        try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.q_norm.weight", .{p}), &.{c.head_dim});
        if (own) try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.k_norm.weight", .{p}), &.{c.head_dim});
        const qd = c.heads * c.head_dim;
        const kvd = c.kv_heads * c.head_dim;
        for ([_]struct { []const u8, usize, usize, bool }{ .{ "q", qd, d, true }, .{ "k", kvd, d, own }, .{ "v", kvd, d, own }, .{ "o", d, qd, true } }) |x| if (x[3]) {
            const name = try std.fmt.allocPrint(a, "{s}self_attn.{s}_proj", .{ p, x[0] });
            if (c.v11) try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}.weight", .{name}), &.{ x[1], x[2] }) else try Add.fp8(&entries, a, name, x[1], x[2]);
        };
        if (c.dense[i]) {
            const e = try std.fmt.allocPrint(a, "{s}mlp", .{p});
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.gate_proj", .{e}), c.dense_width, d);
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.up_proj", .{e}), c.dense_width, d);
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.down_proj", .{e}), d, c.dense_width);
            continue;
        }
        try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}mlp.gate.weight", .{p}), &.{ c.experts, d });
        try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}moe.router.expert_bias", .{p}), &.{c.experts});
        for (0..c.experts + 1) |j| {
            const e = if (j < c.experts) try std.fmt.allocPrint(a, "{s}mlp.experts.{d}", .{ p, j }) else try std.fmt.allocPrint(a, "{s}mlp.shared_experts", .{p});
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.gate_proj", .{e}), c.moe_width, d);
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.up_proj", .{e}), c.moe_width, d);
            try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}.down_proj", .{e}), d, c.moe_width);
        }
    }
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: usize = 0;
    var first = true;
    for (entries.items) |e| {
        var n: usize = if (e.dtype[0] == 'F' and e.dtype[1] == '8') 1 else if (e.dtype[0] == 'B') 2 else 4;
        for (e.shape) |x| n *= x;
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (first) "" else ",", e.name, e.dtype });
        for (e.shape, 0..) |x, i| try header.writer.print("{s}{d}", .{ if (i == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        first = false;
        at += n;
    }
    try header.writer.writeAll("}");
    const h = header.written();
    const file = try a.alloc(u8, 8 + h.len + at);
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], h.len, .little);
    @memcpy(file[8..][0..h.len], h);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
}

fn tmpDir(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "every tensor of a checkpoint maps, the shared expert last" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTiny(a, tmp, tiny);
    var ck = try Checkpoint.openModel(testing.allocator, io, try tmpDir(a, tmp));
    defer ck.close();
    var m = try map(testing.allocator, &ck, tiny);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 3), m.layers[1].experts.len);
    try testing.expectEqual(@as(usize, 2), m.layers[1].experts[2].down.s.dim(0)); // the shared expert, last: 256 / 128
    try testing.expectEqual(core.safetensors.DType.f8_e4m3, m.layers[0].q.w.dtype);
}

test "the pinned checkpoint (TF_KOLIBRI_DIR): 116,303 tensors, none left over" {
    const dir = @import("sys").getenv("TF_KOLIBRI_DIR") orelse return error.SkipZigTest;
    const cfg = try Config.read(testing.allocator, io, dir);
    var ck = try Checkpoint.openModel(testing.allocator, io, dir);
    defer ck.close();
    var m = try map(testing.allocator, &ck, cfg);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 385), m.layers[49].experts.len);
}

/// Kolibri 1.1's kinds in three layers: 0 dense and sliding, 1 indexed with its own cache, 2 indexed on layer 1's.
fn tiny11() Config {
    var c = tiny;
    c.layers = 3;
    c.v11 = true;
    c.dense[0] = true;
    c.dense_width = 384;
    c.full[1] = true;
    c.full[2] = true;
    c.kv_from[2] = 1;
    return c;
}

test "kolibri1_1: bf16 attention, a dense layer's MLP alone, no k, v or k_norm on a layer reading another's cache" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const c = tiny11();
    try writeTiny(a, tmp, c);
    var ck = try Checkpoint.openModel(testing.allocator, io, try tmpDir(a, tmp));
    defer ck.close();
    var m = try map(testing.allocator, &ck, c);
    defer m.deinit();
    try testing.expectEqual(core.safetensors.DType.bf16, m.layers[1].q.w.dtype);
    try testing.expect(m.layers[1].q.s == null and m.layers[1].k != null and m.layers[2].k == null and m.layers[2].k_norm == null);
    try testing.expectEqual(@as(usize, 1), m.layers[0].experts.len);
    try testing.expectEqual(@as(usize, 384), m.layers[0].experts[0].gate.w.dim(0));
    try testing.expect(m.layers[0].router == null and m.layers[1].router != null);
    try testing.expectEqual(@as(usize, 3), m.layers[1].experts.len);
}

test "kolibri1_1: a layer reading another's cache that still ships k_proj leaves tensors over, and is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var written = tiny11();
    written.kv_from[2] = null; // the file has layer 2's k, v and k_norm
    try writeTiny(a, tmp, written);
    var ck = try Checkpoint.openModel(testing.allocator, io, try tmpDir(a, tmp));
    defer ck.close();
    try testing.expectError(error.UnusedTensors, map(testing.allocator, &ck, tiny11()));
}
