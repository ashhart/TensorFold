//! engram_rows.zig against reference Engram host fixtures: bf16 table, packed shard via sync, threads and io_uring.

const std = @import("std");
const testing = std.testing;
const er = @import("engram_rows.zig");
const eh = @import("engram_host.zig");

const ref = @embedFile("fixtures/engram-rows-ref.txt");

/// engram_fixture.record: byte j of global row r.
fn recordByte(r: u64, j: u64) u8 {
    const h = (r * 2654435761 + j * 40503) % (1 << 32);
    return if (j < 256) @truncate(h >> 7) else @intCast(100 + (h >> 9) % 40);
}

const Fx = struct { lo: u64, rows: u64, total: u64, sets: [3]struct { idx: []i64, sha: [32]u8 } };

pub fn parseFx(a: std.mem.Allocator, lut_sha: *[32]u8) !Fx {
    var fx: Fx = undefined;
    var set: usize = 0;
    var it = std.mem.tokenizeScalar(u8, ref, '\n');
    while (it.next()) |line| {
        var w = std.mem.tokenizeScalar(u8, line, ' ');
        const kind = w.next().?;
        if (std.mem.eql(u8, kind, "lut")) {
            _ = try std.fmt.hexToBytes(lut_sha, w.next().?);
        } else if (std.mem.eql(u8, kind, "shard")) {
            fx.lo = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.rows = try std.fmt.parseInt(u64, w.next().?, 10);
            fx.total = try std.fmt.parseInt(u64, w.next().?, 10);
        } else {
            var ids: std.ArrayList(i64) = .empty;
            var c = std.mem.tokenizeScalar(u8, w.next().?, ',');
            while (c.next()) |x| try ids.append(a, try std.fmt.parseInt(i64, x, 10));
            fx.sets[set].idx = try ids.toOwnedSlice(a);
            _ = try std.fmt.hexToBytes(&fx.sets[set].sha, w.next().?);
            set += 1;
        }
    }
    return fx;
}

/// Writes the fixture's shard (the packer's layout) as engram-l1-r0of2.bin under `dir`.
pub fn writeShard(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fx: Fx) !void {
    return writeShardAs(a, io, dir, fx, 1);
}

fn writeShardAs(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, fx: Fx, layer: u64) !void {
    const n: usize = @intCast(er.header_bytes + fx.rows * er.row_bytes);
    const b = try a.alloc(u8, n);
    defer a.free(b);
    @memset(b, 0);
    for ([_]u64{ er.magic, layer, fx.lo, fx.lo + fx.rows, fx.total, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    for (0..fx.rows) |k| for (0..er.row_bytes) |j| {
        b[er.header_bytes + k * er.row_bytes + j] = recordByte(fx.lo + k, j);
    };
    var nb: [32]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&nb, "engram-l{d}-r0of2.bin", .{layer}), .data = b });
}

pub fn sha(out: []const u16) [32]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(out), &d, .{});
    return d;
}
