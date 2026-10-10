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
    const dir = std.testing.environ.getPosix("TF_QWEN_DIR") orelse return error.SkipZigTest;
    const sum = try run(std.testing.allocator, std.testing.io, dir);
    std.debug.print("qwen35 index: {d} layers ({d} full attention, {d} MoE), tied head {}, MTP {}, {d} MiB of projections and MLPs\n", .{ sum.layers, sum.full, sum.moe, sum.tied, sum.mtp, sum.bytes >> 20 });
    try std.testing.expect(sum.layers > 0);
}
