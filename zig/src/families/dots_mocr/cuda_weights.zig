//! dots.mocr's text decoder on CUDA: BF16 tensors on the device, q|k|v and gate|up fused into packed lane weights.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const inventory = @import("weights.zig");

const DeviceBuffer = cuda.DeviceBuffer;
const Weight = cuda.qmmf.Weight;

/// The text decoder's names: the tower's (vision_tower.) stay on the host until the tower is built on CUDA.
const text = [_][]const u8{ "model.", "lm_head." };

pub const Layer = struct {
    ln1: u64, // bf16 [hidden]: input_layernorm
    ln2: u64, // bf16 [hidden]: post_attention_layernorm
    qkv: Weight, // q|k|v [hidden + 2 kv width, hidden]
    bias: u64, // bf16 [hidden + 2 kv width]: q|k|v biases
    o: Weight,
    gate_up: Weight, // gate|up [2 intermediate, hidden]
    down: Weight,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: cfg.Config,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    layers: []Layer = &.{},
    embed: u64 = 0, // bf16 [vocab, hidden]
    norm: u64 = 0, // bf16 [hidden]: model.norm
    head: Weight = undefined, // the untied lm_head

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        w.gpa.free(w.layers);
    }

    fn alloc(w: *Weights, d: *const cuda.Driver, bytes: usize) !DeviceBuffer {
        try w.buffers.ensureUnusedCapacity(w.gpa, 1);
        const b = try DeviceBuffer.alloc(d, bytes);
        w.buffers.appendAssumeCapacity(b);
        return b;
    }
};

/// The text tensors' bytes as stored, which their device copies take too (every lane weight's rows a multiple of 128).
pub fn textBytes(gpa: std.mem.Allocator, c: cfg.Config) !u64 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var total: u64 = 0;
    for (try inventory.inventory(arena.allocator(), c)) |s| {
        for (text) |p| if (std.mem.startsWith(u8, s.name, p)) {
            total += s.bytes();
        };
    }
    return total;
}

/// Reads the checkpoint's text tensors into the device on `s`, fusing and packing the projections.
const Loader = struct {
    w: *Weights,
    ck: *core.Checkpoint,
    d: *const cuda.Driver,
    s: cuda.Stream,
    pack: *const cuda.modelopt.Packer,
    stage: DeviceBuffer,

    /// A bf16 tensor as stored.
    fn raw(l: *Loader, name: []const u8, shape: []const usize) !u64 {
        return l.joined(&.{name}, &.{shape[0]}, if (shape.len == 2) shape[1] else 1);
    }

    /// bf16 tensors [rows[i], k] one after another in a buffer of their own.
    fn joined(l: *Loader, names: []const []const u8, rows: []const usize, k: usize) !u64 {
        var n: usize = 0;
        for (rows) |r| n += r;
        const b = try l.w.alloc(l.d, n * k * 2);
        var at: usize = 0;
        for (names, rows) |name, r| {
            const t = try l.ck.expect(name, .bf16, if (k == 1) &.{r} else &.{ r, k });
            try b.upload(at, t.bytes);
            at += t.bytes.len;
        }
        return b.ptr;
    }

    /// bf16 projections [rows[i], k] stacked into one [sum rows, k] lane weight, packed on the device.
    fn lane(l: *Loader, names: []const []const u8, rows: []const usize, k: usize) !Weight {
        var n: usize = 0;
        var at: usize = 0;
        for (names, rows) |name, r| {
            const t = try l.ck.expect(name, .bf16, &.{ r, k });
            try l.stage.upload(at, t.bytes);
            at += t.bytes.len;
            n += r;
        }
        const b = try l.w.alloc(l.d, cuda.modelopt.padded(n) * k * 2);
        try l.pack.bf16Lane(l.s, l.stage.ptr, b.ptr, n, k);
        try l.s.synchronize(); // the next projection's upload reuses the stage
        return cuda.modelopt.bf16Weight(b.ptr, n, k);
    }

    /// A failed load's work proved complete (the packing stream, then the uploads' null stream) before buffers go.
    fn settle(l: *const Loader) void {
        cuda.completion.drain(l.d, l.s, "a dots.mocr weight load did not drain before its buffers were freed");
        cuda.completion.drain(l.d, .{ .d = l.d, .handle = null }, "dots.mocr weight uploads did not drain before their buffers were freed");
    }
};

/// The text decoder of the dots.mocr checkpoint at `dir`, every header checked first; a tensor left over refuses.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, s: cuda.Stream, dir: []const u8) !Weights {
    const c = try cfg.Config.read(gpa, io, dir);
    _ = try inventory.inspect(gpa, io, dir, c);
    var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &text);
    defer ck.close();
    var pack = try cuda.modelopt.Packer.load(d);
    defer pack.unload();
    var w: Weights = .{ .gpa = gpa, .config = c };
    errdefer w.deinit();
    w.layers = try gpa.alloc(Layer, c.layers);
    const h: usize = c.hidden;
    const kvw: usize = c.kvWidth();
    const m: usize = c.intermediate;
    var stage = try DeviceBuffer.alloc(d, @as(usize, c.vocab) * h * 2);
    defer stage.free();
    var l: Loader = .{ .w = &w, .ck = &ck, .d = d, .s = s, .pack = &pack, .stage = stage };
    errdefer l.settle(); // runs first on failure: the stage, the weights and the packer go after it
    w.embed = try l.raw("model.embed_tokens.weight", &.{ c.vocab, h });
    w.norm = try l.raw("model.norm.weight", &.{h});
    w.head = try l.lane(&.{"lm_head.weight"}, &.{c.vocab}, h);
    var names: [7][96]u8 = undefined;
    for (w.layers, 0..) |*y, i| {
        var n: [7][]const u8 = undefined;
        for (&n, &names, [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |*out, *buf, part|
            out.* = try std.fmt.bufPrint(buf, "model.layers.{d}.{s}.weight", .{ i, part });
        var bias: [3][96]u8 = undefined;
        var b: [3][]const u8 = undefined;
        for (&b, &bias, n[0..3]) |*out, *buf, weight| out.* = try std.fmt.bufPrint(buf, "{s}.bias", .{weight[0 .. weight.len - ".weight".len]});
        var norms: [2][96]u8 = undefined;
        y.* = .{
            .ln1 = try l.raw(try std.fmt.bufPrint(&norms[0], "model.layers.{d}.input_layernorm.weight", .{i}), &.{h}),
            .ln2 = try l.raw(try std.fmt.bufPrint(&norms[1], "model.layers.{d}.post_attention_layernorm.weight", .{i}), &.{h}),
            .qkv = try l.lane(n[0..3], &.{ h, kvw, kvw }, h),
            .bias = try l.joined(&b, &.{ h, kvw, kvw }, 1),
            .o = try l.lane(n[3..4], &.{h}, h),
            .gate_up = try l.lane(n[4..6], &.{ m, m }, h),
            .down = try l.lane(n[6..7], &.{h}, m),
        };
    }
    if (ck.unused() != 0) return error.DotsTensorCount;
    try s.synchronize();
    return w;
}

/// A device whose packing stream's wait fails: it counts buffers freed while a launch may still write them.
const Fake = struct {
    const queue: cuda.abi.Stream = @ptrFromInt(0x1800);
    var next: u64 = 0x10000;
    var live: usize = 0;
    var pending = false;
    var freed_pending: usize = 0;
    var proofs: usize = 0;
    fn alloc(out: *u64, len: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += len + 256;
        live += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        freed_pending += @intFromBool(pending);
        live -= 1;
        return 0;
    }
    fn copy(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        pending = true;
        return 0;
    }
    fn wait(s: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return if (s == queue) cuda.abi.error_not_ready else 0;
    }
    fn context() callconv(.c) cuda.abi.Result {
        proofs += 1;
        pending = false;
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = .{ .lib = undefined, .api = undefined };
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemcpyHtoD_v2 = copy;
        d.api.cuLaunchKernel = launch;
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

test "a failed pack wait proves completion by the context before the stage or any weight is freed" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const header = "{\"x.weight\":{\"dtype\":\"BF16\",\"shape\":[2,64],\"data_offsets\":[0,256]}}";
    var file: [8 + header.len + 256]u8 = @splat(0);
    std.mem.writeInt(u64, file[0..8], header.len, .little);
    @memcpy(file[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &file });
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer a.free(dir);
    var ck = try core.Checkpoint.openModelPrefixes(a, io, dir, &.{});
    defer ck.close();
    var d = Fake.driver();
    var pack: cuda.modelopt.Packer = undefined;
    pack.bf16_lane = .{ .d = &d, .handle = @ptrFromInt(1) };
    var w: Weights = .{ .gpa = a, .config = undefined };
    var stage = try DeviceBuffer.alloc(&d, 256);
    const l: Loader = .{ .w = &w, .ck = &ck, .d = &d, .s = .{ .d = &d, .handle = Fake.queue }, .pack = &pack, .stage = stage };
    var run = l;
    try std.testing.expectError(error.NotReady, run.lane(&.{"x.weight"}, &.{2}, 64));
    try std.testing.expect(Fake.pending and Fake.live == 2);
    l.settle();
    stage.free();
    w.deinit();
    try std.testing.expectEqual(@as(usize, 0), Fake.freed_pending);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
    try std.testing.expectEqual(@as(usize, 1), Fake.proofs);
}
