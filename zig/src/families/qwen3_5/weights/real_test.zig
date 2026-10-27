//! The index check on a real checkpoint: TF_QWEN_DIR names a model directory; every tensor the loader reads resolves.

const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const host = @import("host.zig");

/// What a full pass over a checkpoint found.
pub const Summary = struct { layers: usize = 0, full: usize = 0, moe: usize = 0, mtp: bool = false, tied: bool = false, bytes: usize = 0 };

fn mlpBytes(m: host.Mlp) usize {
    return switch (m) {
        .dense => |d| d.gate.bytes() + d.up.bytes() + d.down.bytes(),
        .routed => |r| r.router.bytes.len + r.rows32.bytes.len + r.experts.fused.bytes() + r.experts.down.bytes(),
    };
}

/// Reads and checks every layer, the embedding, the head and the MTP layer of the checkpoint in `dir`.
pub fn run(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Summary {
    var ck = try checkpoint.Checkpoint.open(gpa, io, dir);
    defer ck.close();
    var sum: Summary = .{ .tied = ck.config.tied };
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const embed = try ck.embed(scratch.allocator());
    sum.bytes += embed.bytes();
    _ = try ck.finalNorm(scratch.allocator());
    if (try ck.head(scratch.allocator())) |h| sum.bytes += h.bytes();
    for (0..ck.spec().n_layers) |i| {
        var layer = try ck.layer(i);
        defer layer.deinit();
        sum.layers += 1;
        const mlp = switch (layer.body) {
            .full => |f| blk: {
                sum.full += 1;
                break :blk f.mlp;
            },
            .linear => |l| l.mlp,
        };
        if (mlp == .routed) sum.moe += 1;
        sum.bytes += mlpBytes(mlp);
    }
    if (try ck.mtp()) |m| {
        var head = m;
        defer head.deinit();
        sum.mtp = true;
        if (head.mlp) |x| sum.bytes += mlpBytes(x);
    }
    return sum;
}

test "the real checkpoint (TF_QWEN_DIR) resolves every tensor with consistent shapes" {
    const dir = @import("sys").getenv("TF_QWEN_DIR") orelse return error.SkipZigTest;
    const sum = try run(std.testing.allocator, std.testing.io, dir);
    std.debug.print("qwen35 index: {d} layers ({d} full attention, {d} MoE), tied head {}, MTP {}, {d} MiB of projections and MLPs\n", .{ sum.layers, sum.full, sum.moe, sum.tied, sum.mtp, sum.bytes >> 20 });
    try std.testing.expect(sum.layers > 0);
}

/// Every tensor the HIP loader can upload, hashed in its order: each part read, then its pages released.
pub fn releaseDigest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![32]u8 {
    var ck = try checkpoint.Checkpoint.open(gpa, io, dir);
    defer ck.close();
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for ([_][]const u8{ checkpoint.prefix ++ "embed_tokens", checkpoint.prefix ++ "norm.", "language_model.lm_head" }) |p| try hashPart(&h, &ck, p);
    ck.release(null);
    for (0..ck.spec().n_layers) |i| {
        var name: [128]u8 = undefined;
        try hashPart(&h, &ck, try std.fmt.bufPrint(&name, checkpoint.prefix ++ "layers.{d}.", .{i}));
        ck.release(i);
    }
    try hashPart(&h, &ck, "mtp.");
    return h.finalResult();
}

fn hashPart(h: *std.crypto.hash.sha2.Sha256, ck: *const checkpoint.Checkpoint, part: []const u8) !void {
    for (ck.main.files) |*f| for (f.names.keys()) |key| if (std.mem.startsWith(u8, key, part)) {
        h.update(key);
        h.update((try ck.main.get(key)).bytes);
    };
}

test "the real checkpoint (TF_QWEN_DIR) reads the same bytes with mmap off while the loader releases parts" {
    const dir = @import("sys").getenv("TF_QWEN_DIR") orelse return error.SkipZigTest;
    const mapped = try releaseDigest(std.testing.allocator, std.testing.io, dir);
    var threaded: std.Io.Threaded = .init(std.heap.page_allocator, .{ .disable_memory_mapping = true });
    defer threaded.deinit();
    const copied = try releaseDigest(std.testing.allocator, threaded.io(), dir);
    try std.testing.expectEqualSlices(u8, &mapped, &copied);
}
