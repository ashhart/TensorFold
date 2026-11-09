//! dots.mocr's image tower on CUDA: BF16 [n, k] weights for cuBLASLt, fc1|fc3 stacked, the patch weight's K padded.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const cfg = @import("config.zig");
const inventory = @import("weights.zig");

const DeviceBuffer = cuda.DeviceBuffer;

/// The patch projection's K on the device: 3 x 14 x 14 = 588 zero-padded to a multiple of 8 for cuBLASLt's BF16 rows.
pub const patch_k = 592;

const prefix = "vision_tower.";

pub const Block = struct {
    norm1: u64, // bf16 [hidden]
    norm2: u64, // bf16 [hidden]
    qkv: u64, // bf16 [3 hidden, hidden], no bias
    proj: u64, // bf16 [hidden, hidden], no bias
    fc13: u64, // bf16 [2 intermediate, hidden]: fc1 (the SiLU gate) rows, then fc3's
    fc2: u64, // bf16 [hidden, intermediate]
};

pub const Tower = struct {
    gpa: std.mem.Allocator,
    config: cfg.Vision,
    buffers: std.ArrayList(DeviceBuffer) = .empty,
    blocks: []Block = &.{},
    patch: u64 = 0, // bf16 [hidden, patch_k], columns past 588 zero
    patch_bias: u64 = 0, // bf16 [hidden]
    patch_norm: u64 = 0, // bf16 [hidden]: the patchifier's RMSNorm
    post_norm: u64 = 0, // bf16 [hidden]: post_trunk_norm
    ln_w: u64 = 0, // bf16 [hidden]: merger.ln_q
    ln_b: u64 = 0, // bf16 [hidden]
    mlp0: u64 = 0, // bf16 [merged, merged]
    mlp0_b: u64 = 0, // bf16 [merged]
    mlp2: u64 = 0, // bf16 [text hidden, merged]
    mlp2_b: u64 = 0, // bf16 [text hidden]

    pub fn deinit(t: *Tower) void {
        for (t.buffers.items) |*b| b.free();
        t.buffers.deinit(t.gpa);
        t.gpa.free(t.blocks);
    }

    fn alloc(t: *Tower, d: *const cuda.Driver, bytes: usize) !DeviceBuffer {
        try t.buffers.ensureUnusedCapacity(t.gpa, 1);
        const b = try DeviceBuffer.alloc(d, bytes);
        t.buffers.appendAssumeCapacity(b);
        return b;
    }
};

/// The tower's tensors' bytes as stored in the checkpoint.
pub fn storedBytes(gpa: std.mem.Allocator, c: cfg.Config) !u64 {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var total: u64 = 0;
    for (try inventory.inventory(arena.allocator(), c)) |s| {
        if (std.mem.startsWith(u8, s.name, prefix)) total += s.bytes();
    }
    return total;
}

/// The tower's device bytes: as stored, but for the patch weight's padded columns.
pub fn deviceBytes(gpa: std.mem.Allocator, c: cfg.Config) !u64 {
    const v = c.vision;
    const k: u64 = 3 * @as(u64, v.patch) * v.patch;
    if (k > patch_k) return error.DotsShape;
    return try storedBytes(gpa, c) + @as(u64, v.hidden) * (patch_k - k) * 2;
}

/// Reads the checkpoint's tower tensors into the device; a tower tensor left over refuses.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Tower {
    const c = try cfg.Config.read(gpa, io, dir);
    _ = try inventory.inspect(gpa, io, dir, c);
    const v = c.vision;
    const k: usize = 3 * @as(usize, v.patch) * v.patch;
    if (k > patch_k) return error.DotsShape;
    var ck = try core.Checkpoint.openModelPrefixes(gpa, io, dir, &.{prefix});
    defer ck.close();
    var t: Tower = .{ .gpa = gpa, .config = v };
    errdefer t.deinit();
    t.blocks = try gpa.alloc(Block, v.layers);
    const h: usize = v.hidden;
    const m: usize = v.intermediate;
    const merged: usize = v.merged();
    var l: Loader = .{ .t = &t, .ck = &ck, .d = d };
    errdefer l.settle(); // runs first on failure: a copy whose wait failed lands before the buffers go
    t.patch = try l.padded(prefix ++ "patch_embed.patchifier.proj.weight", &.{ h, 3, v.patch, v.patch }, h, k);
    t.patch_bias = try l.rows(&.{prefix ++ "patch_embed.patchifier.proj.bias"}, &.{h}, 1);
    t.patch_norm = try l.rows(&.{prefix ++ "patch_embed.patchifier.norm.weight"}, &.{h}, 1);
    var names: [7][96]u8 = undefined;
    for (t.blocks, 0..) |*b, i| {
        var n: [7][]const u8 = undefined;
        for (&n, &names, [_][]const u8{ "norm1", "norm2", "attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc3", "mlp.fc2" }) |*out, *buf, part|
            out.* = try std.fmt.bufPrint(buf, prefix ++ "blocks.{d}.{s}.weight", .{ i, part });
        b.* = .{
            .norm1 = try l.rows(n[0..1], &.{h}, 1),
            .norm2 = try l.rows(n[1..2], &.{h}, 1),
            .qkv = try l.rows(n[2..3], &.{3 * h}, h),
            .proj = try l.rows(n[3..4], &.{h}, h),
            .fc13 = try l.rows(n[4..6], &.{ m, m }, h),
            .fc2 = try l.rows(n[6..7], &.{h}, m),
        };
    }
    t.post_norm = try l.rows(&.{prefix ++ "post_trunk_norm.weight"}, &.{h}, 1);
    t.ln_w = try l.rows(&.{prefix ++ "merger.ln_q.weight"}, &.{h}, 1);
    t.ln_b = try l.rows(&.{prefix ++ "merger.ln_q.bias"}, &.{h}, 1);
    t.mlp0 = try l.rows(&.{prefix ++ "merger.mlp.0.weight"}, &.{merged}, merged);
    t.mlp0_b = try l.rows(&.{prefix ++ "merger.mlp.0.bias"}, &.{merged}, 1);
    t.mlp2 = try l.rows(&.{prefix ++ "merger.mlp.2.weight"}, &.{c.hidden}, merged);
    t.mlp2_b = try l.rows(&.{prefix ++ "merger.mlp.2.bias"}, &.{c.hidden}, 1);
    if (ck.unused() != 0) return error.DotsTensorCount;
    return t;
}

/// Synchronous uploads from the mapped checkpoint into buffers the tower owns.
const Loader = struct {
    t: *Tower,
    ck: *core.Checkpoint,
    d: *const cuda.Driver,

    /// A failed load's uploads proved complete (the null stream, else the context) before buffers go.
    fn settle(l: *const Loader) void {
        cuda.completion.drain(l.d, .{ .d = l.d, .handle = null }, "dots.mocr tower uploads did not drain before their buffers were freed");
    }

    /// bf16 tensors [rows[i], k] one after another in a buffer of their own (k = 1: vectors).
    fn rows(l: *Loader, names: []const []const u8, counts: []const usize, k: usize) !u64 {
        var n: usize = 0;
        for (counts) |r| n += r;
        const b = try l.t.alloc(l.d, n * k * 2);
        var at: usize = 0;
        for (names, counts) |name, r| {
            const t = try l.ck.expect(name, .bf16, if (k == 1) &.{r} else &.{ r, k });
            try b.upload(at, t.bytes);
            at += t.bytes.len;
        }
        return b.ptr;
    }

    /// A bf16 tensor of `n` rows of `k` values, each row zero-padded to patch_k columns.
    fn padded(l: *Loader, name: []const u8, shape: []const usize, n: usize, k: usize) !u64 {
        const t = try l.ck.expect(name, .bf16, shape);
        const host = try l.t.gpa.alloc(u8, n * patch_k * 2);
        defer l.t.gpa.free(host);
        padRows(t.bytes, host, n, k);
        const b = try l.t.alloc(l.d, host.len);
        try b.upload(0, host);
        return b.ptr;
    }
};

/// Rows of `k` bf16 values from `src` into rows of patch_k in `dst`, the tail columns zero.
fn padRows(src: []const u8, dst: []u8, n: usize, k: usize) void {
    @memset(dst, 0);
    for (0..n) |r| @memcpy(dst[r * patch_k * 2 ..][0 .. k * 2], src[r * k * 2 ..][0 .. k * 2]);
}

test "dots tower bytes: 2,524,182,528 stored, 12,288 more on the device for the patch weight's padded columns" {
    const fixtures = @import("config_test.zig");
    const c = try cfg.Config.parse(std.testing.allocator, fixtures.sample, fixtures.generation);
    const v = c.vision;
    try std.testing.expect(v.hidden == 1536 and v.intermediate == 4224 and v.layers == 42 and v.patch == 14 and v.merged() == 6144);
    const h: u64 = 1536;
    const m: u64 = 4224;
    const block = 2 * h + 3 * h * h + h * h + 3 * m * h;
    const want = (h * 588 + 2 * h + 42 * block + h + 2 * h + 6144 * 6144 + 6144 + h * 6144 + h) * 2;
    try std.testing.expectEqual(@as(u64, 2524182528), want);
    try std.testing.expectEqual(want, try storedBytes(std.testing.allocator, c));
    try std.testing.expectEqual(@as(u64, 2524194816), try deviceBytes(std.testing.allocator, c));
}

test "the patch weight's rows keep their values and gain zero columns" {
    var src: [2 * 3 * 2]u8 = undefined;
    for (&src, 0..) |*b, i| b.* = @intCast(i + 1);
    var dst: [2 * patch_k * 2]u8 = @splat(0xAA);
    padRows(&src, &dst, 2, 3);
    try std.testing.expectEqualSlices(u8, src[0..6], dst[0..6]);
    try std.testing.expectEqualSlices(u8, src[6..12], dst[patch_k * 2 ..][0..6]);
    for (dst[6 .. patch_k * 2]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    for (dst[patch_k * 2 + 6 ..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
}

/// A device whose null-stream wait fails: it counts buffers freed while a copy may still write them.
const Fake = struct {
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
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return cuda.abi.error_not_ready;
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
        d.api.cuStreamSynchronize = wait;
        d.api.cuCtxSynchronize = context;
        return d;
    }
};

test "a failed tower load proves its uploads complete by the context before any buffer is freed" {
    const a = std.testing.allocator;
    var d = Fake.driver();
    var ck: core.Checkpoint = undefined;
    for ([_]bool{ false, true }) |settled| { // false: the planted control, freeing with a copy in flight
        Fake.freed_pending = 0;
        Fake.proofs = 0;
        var t: Tower = .{ .gpa = a, .config = undefined };
        _ = try t.alloc(&d, 256);
        _ = try t.alloc(&d, 512);
        Fake.pending = true; // an upload whose null-stream wait failed
        const l: Loader = .{ .t = &t, .ck = &ck, .d = &d };
        if (settled) l.settle();
        t.deinit();
        try std.testing.expectEqual(@as(usize, 0), Fake.live);
        try std.testing.expectEqual(@as(usize, if (settled) 0 else 2), Fake.freed_pending);
        try std.testing.expectEqual(@as(usize, @intFromBool(settled)), Fake.proofs);
    }
}
