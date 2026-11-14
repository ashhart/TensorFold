//! Dory's checkpoint tensors by name, each checked for dtype and shape before any upload; nothing may be left over.
const std = @import("std");
const core = @import("core");
const config = @import("config.zig");

const Config = config.Config;
const Tensor = core.checkpoint.Tensor;
const Checkpoint = core.checkpoint.Checkpoint;

/// A projection [n, k] as stored: NVFP4 codes a 2 and scales a 16, MXFP8 codes with u8 scales a 32, or bf16.
pub const Proj = struct {
    w: Tensor,
    blocks: ?Tensor = null, // NVFP4's or MXFP8's weight_scale
    scale: f32 = 1, // NVFP4's weight_scale_2; 1 for MXFP8 and bf16
    n: usize,
    k: usize,
};

pub const Attn = struct { alpha: Tensor, q: Proj, k: Proj, v: Proj, g: Proj, o: Proj, sqk: Tensor, s_gate: Tensor };
pub const Mlp = struct { alpha: Tensor, gate: Proj, up: Proj, down: Proj, suv_gate: Tensor, suv_up: Tensor };
pub const Sub = union(enum) { attn: Attn, mlp: Mlp };

pub const Map = struct {
    gpa: std.mem.Allocator,
    embed: Tensor, // bf16 [V, D]
    head: Tensor, // bf16 [V, D]
    logit_scale: Tensor, // bf16 [V]
    subs: []Sub,

    pub fn deinit(m: *Map) void {
        m.gpa.free(m.subs);
        m.* = undefined;
    }
};

const Names = struct {
    ck: *Checkpoint,
    quant: config.Quant,
    why: ?*core.modelopt.Why = null, // a part's refusal kept here instead of the log (tests)
    buf: [160]u8 = undefined,

    fn name(self: *Names, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&self.buf, fmt, args);
    }

    fn bf16(self: *Names, shape: []const usize, comptime fmt: []const u8, args: anytype) !Tensor {
        return self.ck.expect(try self.name(fmt, args), .bf16, shape);
    }

    fn proj(self: *Names, n: usize, k: usize, comptime fmt: []const u8, args: anytype) !Proj {
        if (self.quant == .mxfp8) {
            const w = try self.ck.expect(try self.name(fmt ++ ".weight", args), .f8_e4m3, &.{ n, k });
            const b = try self.ck.expect(try self.name(fmt ++ ".weight_scale", args), .u8, &.{ n, k / 32 });
            return .{ .w = w, .blocks = b, .n = n, .k = k };
        }
        // NVFP4 (W4A16: no input_scale, an unused one refuses the map) or bf16: core's ModelOpt part reader
        const kind: core.modelopt.part.Kind = if (self.quant == .bf16) .bf16 else .nvfp4;
        // Dory's own rules: weight_scale_2 positive and of shape [] only, input_scale left to the unused check
        const want: core.modelopt.part.Want = .{ .kind = kind, .n = n, .k = k, .act = .skip, .positive = true, .scalars = .rank0 };
        const module = try self.name(fmt, args);
        const p = if (self.why) |w| try core.modelopt.part.readWhy(self.ck, module, want, w) else try core.modelopt.part.read(self.ck, module, want);
        return .{ .w = p.weight, .blocks = p.blocks, .scale = p.scale, .n = n, .k = k };
    }
};

/// Every tensor the loader reads, sublayer by sublayer; a missing, mistyped or leftover tensor refuses the checkpoint.
pub fn map(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config) !Map {
    return mapWhy(gpa, ck, c, null);
}

/// `map`, a part's refusal or the first leftover tensor kept in `why` instead of the log (tests read it).
fn mapWhy(gpa: std.mem.Allocator, ck: *Checkpoint, c: Config, why: ?*core.modelopt.Why) !Map {
    const g = if (c.quant == .mxfp8) @as(usize, 32) else 16;
    if (c.hidden % g != 0 or c.intermediate % g != 0 or c.qDim() % g != 0) return error.UnsupportedShapes;
    var nm: Names = .{ .ck = ck, .quant = c.quant, .why = why };
    const d = c.hidden;
    const f = c.intermediate;
    const q = c.qDim();
    const kv = c.kvDim();
    var m: Map = .{ .gpa = gpa, .embed = undefined, .head = undefined, .logit_scale = undefined, .subs = try gpa.alloc(Sub, c.sublayers) };
    errdefer gpa.free(m.subs);
    m.embed = try nm.bf16(&.{ c.vocab, d }, "backbone.embeddings.weight", .{});
    for (m.subs, 0..) |*s, i| {
        const pre = "backbone.layers.{d}.";
        s.* = if (c.kinds[i] == .mlp) .{ .mlp = .{
            .alpha = try nm.bf16(&.{d}, pre ++ "mlp_alpha", .{i}),
            .gate = try nm.proj(f, d, pre ++ "mixer.gate_proj", .{i}),
            .up = try nm.proj(f, d, pre ++ "mixer.up_proj", .{i}),
            .down = try nm.proj(d, f, pre ++ "mixer.down_proj", .{i}),
            .suv_gate = try nm.bf16(&.{f}, pre ++ "mixer.suv_gate", .{i}),
            .suv_up = try nm.bf16(&.{f}, pre ++ "mixer.suv_up", .{i}),
        } } else .{ .attn = .{
            .alpha = try nm.bf16(&.{d}, pre ++ "attn_alpha", .{i}),
            .q = try nm.proj(q, d, pre ++ "mixer.q_proj", .{i}),
            .k = try nm.proj(kv, d, pre ++ "mixer.k_proj", .{i}),
            .v = try nm.proj(kv, d, pre ++ "mixer.v_proj", .{i}),
            .g = try nm.proj(q, d, pre ++ "mixer.g_proj", .{i}),
            .o = try nm.proj(d, q, pre ++ "mixer.o_proj", .{i}),
            .sqk = try nm.bf16(&.{q}, pre ++ "mixer.sqk", .{i}),
            .s_gate = try nm.bf16(&.{q}, pre ++ "mixer.s_gate", .{i}),
        } };
    }
    m.head = try nm.bf16(&.{ c.vocab, d }, "lm_head.weight", .{});
    m.logit_scale = try nm.bf16(&.{c.vocab}, "lm_head.logit_scale", .{});
    if (why) |w| for (ck.files.items) |*file| for (file.names.keys()) |key| if (!ck.used.contains(key)) {
        w.set("unused checkpoint tensor {s}", .{key});
        return error.UnusedTensors;
    };
    if (ck.unused() != 0) return error.UnusedTensors;
    return m;
}

const testing = std.testing;
const io = testing.io;

/// A tiny Dory: 2 input, 4 recurrent and 2 output sublayers, attention and MLP alternating, one of each full.
pub const tiny: Config = blk: {
    var c: Config = .{ .hidden = 64, .intermediate = 96, .vocab = 32, .heads = 4, .kv_heads = 2, .head_dim = 16, .sublayers = 8, .input = 2, .recurrent = 4, .output = 2, .loops = 2, .window = 3, .theta_sliding = 1e4, .theta_full = 1e6, .std = 0.0177, .alpha_init = 0.05, .gate_init = 8, .positions = 64, .quant = .nvfp4 };
    for (0..8) |i| c.kinds[i] = if (i % 2 == 1) .mlp else if (i == 4) .full else .sliding;
    break :blk c;
};

/// A safetensors file with each tensor `c` names, zero-filled, every NVFP4 scale 1.
pub fn writeFile(a: std.mem.Allocator, dir: std.Io.Dir, c: Config) !void {
    return writeFaulty(a, dir, c, .{});
}

/// What a test plants in an NVFP4 file: every weight_scale_2's value and shape, an input_scale beside each.
const Fault = struct { scale_2: f32 = 1, scale_2_shape: []const usize = &.{}, input_scale: bool = false };

fn writeFaulty(a: std.mem.Allocator, dir: std.Io.Dir, c: Config, fault: Fault) !void {
    const E = struct { name: []const u8, dtype: []const u8, shape: []const usize };
    var entries: std.ArrayList(E) = .empty;
    const Add = struct {
        fn bf(list: *std.ArrayList(E), al: std.mem.Allocator, nm: []const u8, shape: []const usize) !void {
            try list.append(al, .{ .name = nm, .dtype = "BF16", .shape = try al.dupe(usize, shape) });
        }
        fn proj(list: *std.ArrayList(E), al: std.mem.Allocator, cc: Config, plant: Fault, pre: []const u8, n: usize, k: usize) !void {
            if (cc.quant == .bf16) return bf(list, al, try std.fmt.allocPrint(al, "{s}.weight", .{pre}), &.{ n, k });
            if (cc.quant == .mxfp8) {
                try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight", .{pre}), .dtype = "F8_E4M3", .shape = try al.dupe(usize, &.{ n, k }) });
                return list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale", .{pre}), .dtype = "U8", .shape = try al.dupe(usize, &.{ n, k / 32 }) });
            }
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight", .{pre}), .dtype = "U8", .shape = try al.dupe(usize, &.{ n, k / 2 }) });
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale", .{pre}), .dtype = "F8_E4M3", .shape = try al.dupe(usize, &.{ n, k / 16 }) });
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.weight_scale_2", .{pre}), .dtype = "F32", .shape = plant.scale_2_shape });
            if (plant.input_scale) try list.append(al, .{ .name = try std.fmt.allocPrint(al, "{s}.input_scale", .{pre}), .dtype = "F32", .shape = &.{} });
        }
    };
    const d = c.hidden;
    try Add.bf(&entries, a, "backbone.embeddings.weight", &.{ c.vocab, d });
    try Add.bf(&entries, a, "lm_head.weight", &.{ c.vocab, d });
    try Add.bf(&entries, a, "lm_head.logit_scale", &.{c.vocab});
    for (0..c.sublayers) |i| {
        const p = try std.fmt.allocPrint(a, "backbone.layers.{d}.", .{i});
        if (c.kinds[i] == .mlp) {
            try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}mlp_alpha", .{p}), &.{d});
            for ([_][]const u8{ "gate_proj", "up_proj" }) |n| try Add.proj(&entries, a, c, fault, try std.fmt.allocPrint(a, "{s}mixer.{s}", .{ p, n }), c.intermediate, d);
            try Add.proj(&entries, a, c, fault, try std.fmt.allocPrint(a, "{s}mixer.down_proj", .{p}), d, c.intermediate);
            for ([_][]const u8{ "suv_gate", "suv_up" }) |n| try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}mixer.{s}", .{ p, n }), &.{c.intermediate});
        } else {
            try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}attn_alpha", .{p}), &.{d});
            for ([_][]const u8{ "q_proj", "g_proj" }) |n| try Add.proj(&entries, a, c, fault, try std.fmt.allocPrint(a, "{s}mixer.{s}", .{ p, n }), c.qDim(), d);
            for ([_][]const u8{ "k_proj", "v_proj" }) |n| try Add.proj(&entries, a, c, fault, try std.fmt.allocPrint(a, "{s}mixer.{s}", .{ p, n }), c.kvDim(), d);
            try Add.proj(&entries, a, c, fault, try std.fmt.allocPrint(a, "{s}mixer.o_proj", .{p}), d, c.qDim());
            for ([_][]const u8{ "sqk", "s_gate" }) |n| try Add.bf(&entries, a, try std.fmt.allocPrint(a, "{s}mixer.{s}", .{ p, n }), &.{c.qDim()});
        }
    }
    var header: std.Io.Writer.Allocating = .init(a);
    try header.writer.writeAll("{");
    var at: usize = 0;
    for (entries.items, 0..) |e, j| {
        var n: usize = if (e.dtype[0] == 'B') 2 else if (e.dtype[1] == '3') 4 else 1;
        for (e.shape) |x| n *= x;
        try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (j == 0) "" else ",", e.name, e.dtype });
        for (e.shape, 0..) |x, i| try header.writer.print("{s}{d}", .{ if (i == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + n });
        at += n;
    }
    try header.writer.writeAll("}");
    const h = header.written();
    const file = try a.alloc(u8, 8 + h.len + at);
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], h.len, .little);
    @memcpy(file[8..][0..h.len], h);
    // every weight_scale_2 `fault.scale_2` (one unless planted): the map refuses a zero scale
    var off: usize = 0;
    for (entries.items) |e| {
        var n: usize = if (e.dtype[0] == 'B') 2 else if (e.dtype[1] == '3') 4 else 1;
        for (e.shape) |x| n *= x;
        if (std.mem.endsWith(u8, e.name, "weight_scale_2")) std.mem.writeInt(u32, file[8 + h.len + off ..][0..4], @bitCast(fault.scale_2), .little);
        off += n;
    }
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
}

fn tmpPath(a: std.mem.Allocator, tmp: testing.TmpDir) ![]u8 {
    const cwd = try std.process.currentPathAlloc(io, a);
    return std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
}

test "every tensor of a tiny NVFP4, MXFP8 and bf16 checkpoint maps" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]config.Quant{ .nvfp4, .mxfp8, .bf16 }) |quant| {
        var c = tiny;
        c.quant = quant;
        var tmp = testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeFile(a, tmp.dir, c);
        var ck = try Checkpoint.openModel(testing.allocator, io, try tmpPath(a, tmp));
        defer ck.close();
        var m = try map(testing.allocator, &ck, c);
        defer m.deinit();
        try testing.expectEqual(@as(usize, 8), m.subs.len);
        try testing.expect(m.subs[1] == .mlp and m.subs[4] == .attn);
        const o = m.subs[4].attn.o;
        try testing.expectEqual(@as(usize, 64), o.n);
        try testing.expectEqual(@as(usize, 64), o.k);
        try testing.expectEqual(quant != .bf16, o.blocks != null);
    }
}

/// The checkpoint at the `env` directory: 72 sublayers, an mlp last, its end tokens.
fn pinnedMaps(env: [*:0]const u8) !void {
    const dir = @import("sys").getenv(env) orelse return error.SkipZigTest;
    const c = try Config.read(testing.allocator, io, dir);
    var ck = try Checkpoint.openModel(testing.allocator, io, dir);
    defer ck.close();
    var m = try map(testing.allocator, &ck, c);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 72), m.subs.len);
    try testing.expect(m.subs[71] == .mlp and m.subs[70] == .attn);
    try testing.expectEqualSlices(u32, &.{ 2, 11 }, c.eos[0..c.eos_count]);
}

test "the pinned NVFP4 checkpoint (TF_DORY_DIR): 1,083 tensors, none left over" {
    try pinnedMaps("TF_DORY_DIR");
}

test "the pinned MXFP8 checkpoint (TF_DORY_MXFP8_DIR): 795 tensors, none left over" {
    try pinnedMaps("TF_DORY_MXFP8_DIR");
}

test "the pinned BF16 checkpoint (TF_DORY_BF16_DIR): 507 tensors, none left over" {
    try pinnedMaps("TF_DORY_BF16_DIR");
}

/// The tiny NVFP4 checkpoint with `fault` planted refuses with `err`, the reason from `head` to `tail`.
fn refuses(fault: Fault, err: anyerror, head: []const u8, tail: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeFaulty(arena.allocator(), tmp.dir, tiny, fault);
    var ck = try Checkpoint.openModel(testing.allocator, io, try tmpPath(arena.allocator(), tmp));
    defer ck.close();
    var why: core.modelopt.Why = .{};
    try testing.expectError(err, mapWhy(testing.allocator, &ck, tiny, &why));
    try testing.expect(std.mem.startsWith(u8, why.text(), head));
    try testing.expect(std.mem.endsWith(u8, why.text(), tail));
}

test "Dory map: a zero, negative or NaN weight_scale_2 is refused by name" {
    for ([_]f32{ 0, -1, std.math.nan(f32) }) |v| try refuses(.{ .scale_2 = v }, error.UnexpectedTensor, "backbone.layers.0.mixer.q_proj.weight_scale_2: ", " is not a positive finite scale");
}

test "Dory map: a weight_scale_2 of shape [1] or [1, 1] is refused by name, as before the shared reader" {
    for ([_][]const usize{ &.{1}, &.{ 1, 1 } }) |shape| try refuses(.{ .scale_2_shape = shape }, error.UnexpectedTensor, "backbone.layers.0.mixer.q_proj.weight_scale_2: f32 ", "expected an F32 scalar of shape []");
}

test "Dory map: an NVFP4 checkpoint with an input_scale left over is refused (W4A16 reads none)" {
    try refuses(.{ .input_scale = true }, error.UnusedTensors, "unused checkpoint tensor backbone.layers.", ".input_scale");
}
