//! The H3 CUDA family's block weights: the checkpoint's bf16 tensors copied to the device as they are stored.
const std = @import("std");
const cuda = @import("cuda");
const h3 = @import("cuda.zig");
const Checkpoint = @import("checkpoint.zig").Checkpoint;
const Model = h3.Model;

const staging_bytes = 64 << 20;

/// Reads the model's sizes from `path`, allocates its blocks and fills them from the checkpoint.
pub fn read(self: *Model, path: [*:0]const u8) !void {
    const d = &self.driver;
    var ck = try Checkpoint.open(path);
    defer ck.deinit();
    var name: [128]u8 = undefined;
    const qkv0 = try ck.bf16("blocks.0.attn.qkv_proj.weight");
    const fc2_0 = try ck.bf16("blocks.0.mlp.fc2.weight");
    if (qkv0.rows % 3 != 0) return error.BadGeometry;
    const plan = try layout(qkv0.cols, qkv0.rows / 3, fc2_0.cols);
    const layers = try blockCount(self.gpa, ck.header);
    // Validate every shape and byte extent before allocating any device or pinned memory.
    for (0..layers) |i| for (names, plan.shapes) |suffix, shape| {
        const entry = ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.{s}", .{ i, suffix })) catch |err| switch (err) {
            error.MissingTensor => return error.BadBlocks,
            else => return err,
        };
        if (entry.rows != shape[0] or entry.cols != shape[1]) return error.WrongShape;
    };
    self.hidden = qkv0.cols;
    self.inner = qkv0.rows / 3;
    self.mlp = fc2_0.cols;
    self.heads = self.inner / h3.head_dim;
    const staging = &self.load_stage;
    var submitted = false;
    defer if (!self.load_failed) staging.deinit();
    self.blocks = try self.gpa.alloc(h3.Block, layers);
    var made: usize = 0;
    errdefer {
        var complete = true;
        if (submitted) {
            self.drain() catch {
                complete = false;
            };
            if (complete) staging.close() catch {
                complete = false;
            };
        }
        if (complete) {
            for (self.blocks[0..made]) |*b| for (buffers(b)) |held| held.free();
            self.gpa.free(self.blocks);
            self.blocks = &.{};
        } else self.load_failed = true;
    }
    for (self.blocks) |*b| {
        b.* = try allocateBlock(d, plan.sizes);
        made += 1;
    }
    const into = [_]usize{ 0, 0, 1, 2, 3, 4, 4, 4, 4 };
    for (self.blocks, 0..) |*b, i| {
        const held = buffers(b);
        for (names, into, plan.offsets) |suffix, to, offset| {
            const entry = try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.{s}", .{ i, suffix }));
            submitted = true;
            try upload(staging, held[to].*, offset, entry.bytes);
        }
    }
}

// Enumerate advertised block keys, so a missing norm cannot truncate the model before shape validation.
fn blockCount(gpa: std.mem.Allocator, header: []const u8) !usize {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, header, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadBlocks;
    const obj = parsed.value.object;
    var last: ?usize = null;
    for (obj.keys()) |key| {
        if (!std.mem.startsWith(u8, key, "blocks.")) continue;
        const rest = key[7..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.BadBlocks;
        const id = std.fmt.parseInt(usize, rest[0..dot], 10) catch return error.BadBlocks;
        if (dot == 0 or (dot > 1 and rest[0] == '0') or id >= obj.count()) return error.BadBlocks;
        last = @max(last orelse 0, id);
    }
    return (last orelse return error.BadBlocks) + 1;
}

const names = [_][]const u8{ "attn.qkv_proj.weight", "attn.to_gate_compress.weight", "attn.out_proj.weight", "mlp.fc1.weight", "mlp.fc2.weight", "norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight" };
const Layout = struct { sizes: [5]usize, offsets: [9]usize, shapes: [9][2]usize };
fn product(factors: []const usize) !usize {
    var n: usize = 1;
    for (factors) |f| n = std.math.mul(usize, n, f) catch return error.WeightSizeOverflow;
    return n;
}
fn layout(hidden: usize, inner: usize, mlp: usize) !Layout {
    if (hidden == 0 or inner == 0 or mlp == 0 or inner % h3.head_dim != 0) return error.BadGeometry;
    const qkv = try product(&.{ 3, inner });
    const fc1 = try product(&.{ 2, mlp });
    const norms = std.math.add(usize, hidden, h3.head_dim) catch return error.WeightSizeOverflow;
    return .{
        .sizes = .{ try product(&.{ 4, inner, hidden, 2 }), try product(&.{ hidden, inner, 2 }), try product(&.{ 2, mlp, hidden, 2 }), try product(&.{ hidden, mlp, 2 }), try product(&.{ norms, 4 }) },
        .offsets = .{ 0, try product(&.{ qkv, hidden, 2 }), 0, 0, 0, 0, try product(&.{ hidden, 2 }), try product(&.{ hidden, 4 }), std.math.add(usize, try product(&.{ hidden, 4 }), h3.head_dim * 2) catch return error.WeightSizeOverflow },
        .shapes = .{ .{ qkv, hidden }, .{ inner, hidden }, .{ hidden, inner }, .{ fc1, hidden }, .{ hidden, mlp }, .{ 1, hidden }, .{ 1, hidden }, .{ 1, h3.head_dim }, .{ 1, h3.head_dim } },
    };
}
fn allocateBlock(d: *const cuda.Driver, sizes: [5]usize) !h3.Block {
    var b: h3.Block = undefined;
    b.qkvc = try cuda.DeviceBuffer.alloc(d, sizes[0]);
    errdefer b.qkvc.free();
    b.out = try cuda.DeviceBuffer.alloc(d, sizes[1]);
    errdefer b.out.free();
    b.fc1 = try cuda.DeviceBuffer.alloc(d, sizes[2]);
    errdefer b.fc1.free();
    b.fc2 = try cuda.DeviceBuffer.alloc(d, sizes[3]);
    errdefer b.fc2.free();
    b.norms = try cuda.DeviceBuffer.alloc(d, sizes[4]);
    return b;
}

/// The load is published only after all uploaded weights have synchronized.
pub fn readSynced(self: *Model, path: [*:0]const u8) !void {
    try read(self, path);
    errdefer free(self);
    try self.ctx.synchronize();
}

fn buffers(b: *h3.Block) [5]*cuda.DeviceBuffer {
    return .{ &b.qkvc, &b.out, &b.fc1, &b.fc2, &b.norms };
}

/// A block's projections: attn.qkv_proj, attn.to_gate_compress, attn.out_proj, mlp.fc1, mlp.fc2.
pub const Projection = enum(c_int) { qkv, gate_compress, out, fc1, fc2 };

/// One projection of `block` from `w`, bf16 (N, K) on the device (a LoRA merged in by the host), copied in.
pub fn set(self: *Model, block: usize, which: Projection, w: u64, rows: usize, cols: usize) !void {
    if (block >= self.blocks.len) return error.NoSuchBlock;
    try self.ctx.makeCurrent();
    const b = &self.blocks[block];
    const inner = self.inner;
    const hidden = self.hidden;
    const checked = try layout(hidden, inner, self.mlp);
    const want: [2]usize, const to: cuda.DeviceBuffer, const offset: usize = switch (which) {
        .qkv => .{ .{ checked.shapes[0][0], hidden }, b.qkvc, 0 },
        .gate_compress => .{ .{ inner, hidden }, b.qkvc, checked.offsets[1] },
        .out => .{ .{ hidden, inner }, b.out, 0 },
        .fc1 => .{ .{ checked.shapes[3][0], hidden }, b.fc1, 0 },
        .fc2 => .{ .{ hidden, self.mlp }, b.fc2, 0 },
    };
    if (rows != want[0] or cols != want[1]) return error.WrongShape;
    try self.stream.synchronize();
    try self.driver.check(self.driver.api.cuMemcpyDtoD_v2(std.math.add(u64, to.ptr, offset) catch return error.WeightSizeOverflow, w, try product(&.{ rows, cols, 2 })), "cuMemcpyDtoD");
    try self.ctx.synchronize();
}

/// `bytes` into `to` from `offset` through pinned memory; the driver copies pageable memory several times slower.
fn upload(staging: *cuda.copy_source.Stage, to: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(staging_bytes, bytes.len - done);
        @memcpy(try staging.take(to.d, n), bytes[done..][0..n]);
        try staging.copyTo(to, offset + done, .{ .d = to.d, .handle = null });
        try staging.wait();
        done += n;
    }
}

/// Frees what `read` allocated.
pub fn free(self: *Model) void {
    for (self.blocks) |*b| for (buffers(b)) |held| held.free();
    self.gpa.free(self.blocks);
    self.blocks = &.{};
}

test "h3 arithmetic: weight byte products are checked" {
    try std.testing.expectError(error.WeightSizeOverflow, layout(std.math.maxInt(usize), 128, 1));
    try std.testing.expectError(error.WeightSizeOverflow, layout(1, 128, std.math.maxInt(usize)));
    try std.testing.expectError(error.BadGeometry, layout(1, 127, 1));
}
