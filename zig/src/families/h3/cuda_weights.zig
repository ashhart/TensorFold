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
    var staging = try cuda.HostBuffer.alloc(d, staging_bytes);
    defer staging.free();
    var ck = try Checkpoint.open(path);
    defer ck.deinit();
    var name: [128]u8 = undefined;
    const qkv0 = try ck.bf16("blocks.0.attn.qkv_proj.weight");
    const fc2_0 = try ck.bf16("blocks.0.mlp.fc2.weight");
    self.hidden = qkv0.cols;
    self.inner = qkv0.rows / 3;
    self.mlp = fc2_0.cols;
    self.heads = self.inner / h3.head_dim;
    var layers: usize = 0;
    while (std.mem.indexOf(u8, ck.header, try std.fmt.bufPrint(&name, "\"blocks.{d}.norm1.weight\"", .{layers})) != null) layers += 1;
    const hidden = self.hidden;
    const inner = self.inner;
    self.blocks = try self.gpa.alloc(h3.Block, layers);
    var made: usize = 0;
    errdefer {
        for (self.blocks[0..made]) |*b| for (buffers(b)) |held| held.free();
        self.gpa.free(self.blocks);
    }
    for (self.blocks) |*b| {
        b.qkvc = try cuda.DeviceBuffer.alloc(d, 4 * inner * hidden * 2);
        b.out = try cuda.DeviceBuffer.alloc(d, hidden * inner * 2);
        b.fc1 = try cuda.DeviceBuffer.alloc(d, 2 * self.mlp * hidden * 2);
        b.fc2 = try cuda.DeviceBuffer.alloc(d, hidden * self.mlp * 2);
        b.norms = try cuda.DeviceBuffer.alloc(d, (2 * hidden + 2 * h3.head_dim) * 2);
        made += 1;
    }
    const Part = struct { []const u8, usize, usize };
    for (self.blocks, 0..) |*b, i| {
        // each tensor's name, the buffer it goes to by its place in `buffers`, and its byte offset there
        const parts = [_]Part{
            .{ "attn.qkv_proj.weight", 0, 0 },
            .{ "attn.to_gate_compress.weight", 0, 3 * inner * hidden * 2 },
            .{ "attn.out_proj.weight", 1, 0 },
            .{ "mlp.fc1.weight", 2, 0 },
            .{ "mlp.fc2.weight", 3, 0 },
            .{ "norm1.weight", 4, 0 },
            .{ "norm2.weight", 4, hidden * 2 },
            .{ "attn.q_norm.weight", 4, hidden * 4 },
            .{ "attn.k_norm.weight", 4, hidden * 4 + h3.head_dim * 2 },
        };
        const held = buffers(b);
        for (parts) |p| {
            const entry = try ck.bf16(try std.fmt.bufPrint(&name, "blocks.{d}.{s}", .{ i, p[0] }));
            if (p[2] + entry.bytes.len > held[p[1]].len) return error.WrongShape;
            try upload(&staging, held[p[1]].*, p[2], entry.bytes);
        }
    }
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
    const want: [2]usize, const to: cuda.DeviceBuffer, const offset: usize = switch (which) {
        .qkv => .{ .{ 3 * inner, hidden }, b.qkvc, 0 },
        .gate_compress => .{ .{ inner, hidden }, b.qkvc, 3 * inner * hidden * 2 },
        .out => .{ .{ hidden, inner }, b.out, 0 },
        .fc1 => .{ .{ 2 * self.mlp, hidden }, b.fc1, 0 },
        .fc2 => .{ .{ hidden, self.mlp }, b.fc2, 0 },
    };
    if (rows != want[0] or cols != want[1]) return error.WrongShape;
    try self.stream.synchronize();
    try self.driver.check(self.driver.api.cuMemcpyDtoD_v2(to.ptr + offset, w, rows * cols * 2), "cuMemcpyDtoD");
    try self.ctx.synchronize();
}

/// `bytes` into `to` from `offset` through pinned memory; the driver copies pageable memory several times slower.
fn upload(staging: *cuda.HostBuffer, to: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = @min(staging_bytes, bytes.len - done);
        @memcpy(staging.bytes[0..n], bytes[done..][0..n]);
        try to.upload(offset + done, staging.bytes[0..n]);
        done += n;
    }
}

/// Frees what `read` allocated.
pub fn free(self: *Model) void {
    for (self.blocks) |*b| for (buffers(b)) |held| held.free();
    self.gpa.free(self.blocks);
}
