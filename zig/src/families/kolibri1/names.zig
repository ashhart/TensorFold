//! Kolibri-1's checkpoint tensors by name, each checked for dtype and shape before any upload.
const std = @import("std");
const core = @import("core");
const Config = @import("config.zig").Config;

const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;

pub const block = 128; // the checkpoint's scale block, both ways

/// A block-FP8 projection as stored: e4m3 codes [n, k] and fp32 scale_inv [ceil(n/128), ceil(k/128)].
pub const Fp8 = struct { w: Tensor, s: Tensor };
pub const Expert = struct { gate: Fp8, up: Fp8, down: Fp8 };

pub const Layer = struct {
    input_norm: Tensor,
    q: Fp8,
    k: Fp8,
    v: Fp8,
    q_norm: Tensor,
    k_norm: Tensor,
    o: Fp8,
    post_attn_norm: Tensor,
    pre_moe_norm: Tensor, // post_attention_layernorm: the norm before the MoE, despite its name
    router: Tensor, // [E, D] bf16
    bias: Tensor, // [E] expert_bias
    experts: []Expert, // E routed, then the shared expert
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

fn blocks(n: usize) usize {
    return (n + block - 1) / block;
}

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
        const w = try self.ck.expect(try self.name(fmt ++ ".weight", args), .f8_e4m3, &.{ n, k });
        const s = try self.ck.expect(try self.name(fmt ++ ".weight_scale_inv", args), .f32, &.{ blocks(n), blocks(k) });
        return .{ .w = w, .s = s };
    }

    fn expert(self: *Names, c: Config, comptime pre: []const u8, args: anytype) !Expert {
        return .{
            .gate = try self.fp8(c.moe_width, c.hidden, pre ++ ".gate_proj", args),
            .up = try self.fp8(c.moe_width, c.hidden, pre ++ ".up_proj", args),
            .down = try self.fp8(c.hidden, c.moe_width, pre ++ ".down_proj", args),
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
        const ex = try gpa.alloc(Expert, c.experts + 1);
        errdefer gpa.free(ex);
        for (ex[0..c.experts], 0..) |*e, j| e.* = try nm.expert(c, "model.layers.{d}.mlp.experts.{d}", .{ i, j });
        ex[c.experts] = try nm.expert(c, "model.layers.{d}.mlp.shared_experts", .{i});
        const qd = c.heads * c.head_dim;
        const kvd = c.kv_heads * c.head_dim;
        l.* = .{
            .input_norm = try nm.bf16(&.{d}, "model.layers.{d}.input_layernorm.weight", .{i}),
            .q = try nm.fp8(qd, d, "model.layers.{d}.self_attn.q_proj", .{i}),
            .k = try nm.fp8(kvd, d, "model.layers.{d}.self_attn.k_proj", .{i}),
            .v = try nm.fp8(kvd, d, "model.layers.{d}.self_attn.v_proj", .{i}),
            .q_norm = try nm.bf16(&.{c.head_dim}, "model.layers.{d}.self_attn.q_norm.weight", .{i}),
            .k_norm = try nm.bf16(&.{c.head_dim}, "model.layers.{d}.self_attn.k_norm.weight", .{i}),
            .o = try nm.fp8(d, qd, "model.layers.{d}.self_attn.o_proj", .{i}),
            .post_attn_norm = try nm.bf16(&.{d}, "model.layers.{d}.post_attn_norm.weight", .{i}),
            .pre_moe_norm = try nm.bf16(&.{d}, "model.layers.{d}.post_attention_layernorm.weight", .{i}),
            .router = try nm.bf16(&.{ c.experts, d }, "model.layers.{d}.mlp.gate.weight", .{i}),
            .bias = try nm.bf16(&.{c.experts}, "model.layers.{d}.moe.router.expert_bias", .{i}),
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
fn writeTiny(a: std.mem.Allocator, tmp: testing.TmpDir) !void {
    var entries: std.ArrayList(struct { name: []const u8, dtype: []const u8, shape: []const usize }) = .empty;
    const c = tiny;
    const Add = struct {
        fn fp8(list: anytype, al: std.mem.Allocator, pre: []const u8, n: usize, k: usize) !void {
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight", .{pre}), .dtype = "F8_E4M3", .shape = try al.dupe(usize, &.{ n, k }) });
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale_inv", .{pre}), .dtype = "F32", .shape = try al.dupe(usize, &.{ blocks(n), blocks(k) }) });
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
        for ([_][]const u8{ "q_norm", "k_norm" }) |n| try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.{s}.weight", .{ p, n }), &.{c.head_dim});
        try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}mlp.gate.weight", .{p}), &.{ c.experts, d });
        try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}moe.router.expert_bias", .{p}), &.{c.experts});
        try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.q_proj", .{p}), c.heads * c.head_dim, d);
        try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.k_proj", .{p}), c.kv_heads * c.head_dim, d);
        try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.v_proj", .{p}), c.kv_heads * c.head_dim, d);
        try Add.fp8(&entries, a, try std.fmt.allocPrint(a, "{s}self_attn.o_proj", .{p}), d, c.heads * c.head_dim);
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
    try writeTiny(a, tmp);
    var ck = try Checkpoint.openModel(testing.allocator, io, try tmpDir(a, tmp));
    defer ck.close();
    var m = try map(testing.allocator, &ck, tiny);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 3), m.layers[1].experts.len);
    try testing.expectEqual(@as(usize, 2), m.layers[1].experts[2].down.s.dim(0)); // the shared expert, last: 256 / 128
    try testing.expectEqual(core.safetensors.DType.f8_e4m3, m.layers[0].q.w.dtype);
}

test "the pinned checkpoint (TF_KOLIBRI_DIR): 116,303 tensors, none left over" {
    const dir = testing.environ.getPosix("TF_KOLIBRI_DIR") orelse return error.SkipZigTest;
    const cfg = try Config.read(testing.allocator, io, dir);
    var ck = try Checkpoint.openModel(testing.allocator, io, dir);
    defer ck.close();
    var m = try map(testing.allocator, &ck, cfg);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 385), m.layers[49].experts.len);
}
