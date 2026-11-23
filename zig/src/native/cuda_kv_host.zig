//! One host-backed Qwen window, with its table and state run charged to device memory.
const std = @import("std");
const api = @import("engine_api");
const kv = @import("core").kv_pool;

pub fn plan(a: std.mem.Allocator, l: kv.Layout, room: u64, o: api.Open, host: u64, problem: *[]const u8) !?kv.Plan {
    if (o.lanes != 1) {
        problem.* = "TF_QWEN_KV_HOST supports one stream: set --parallel 1";
        return null;
    }
    const device = try deviceBytes(l);
    if (device > room) {
        problem.* = try std.fmt.allocPrint(a, "TF_QWEN_KV_HOST needs {d} device bytes for its table and state run; the budget leaves {d}", .{ device, room });
        return null;
    }
    return .{ .lanes = 1, .pages = @intCast(host / l.pageBytes()), .snaps = 0, .copied = 0 };
}

pub fn deviceBytes(l: kv.Layout) !u64 {
    return std.math.add(u64, try std.math.mul(u64, l.runPages(), l.pageBytes()), l.fixed_bytes);
}
