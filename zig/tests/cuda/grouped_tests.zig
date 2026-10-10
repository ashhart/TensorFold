//! The shared expert plan (cuda/grouped.zig) against experts.route's bytes (oracle/grouped_plan.py).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const grouped = cuda.grouped;

pub fn plan(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var it = std.mem.tokenizeScalar(u8, try fx.string("cases"), ',');
    var i: usize = 0;
    while (it.next()) |case| : (i += 1) {
        var f = std.mem.tokenizeScalar(u8, case, ':');
        const rows = try std.fmt.parseInt(usize, f.next().?, 10);
        const slots = try std.fmt.parseInt(usize, f.next().?, 10);
        const experts = try std.fmt.parseInt(usize, f.next().?, 10);
        const tile = try std.fmt.parseInt(usize, f.next().?, 10);
        const pairs = rows * slots;
        const wide = pairs > grouped.small;
        var name: [32]u8 = undefined;
        const picks = try fx.bytes(try std.fmt.bufPrint(&name, "picks{d}", .{i}));
        defer a.free(picks);
        var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, picks);
        defer dpicks.free();
        var members = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 4);
        defer members.free();
        var items = try cuda.DeviceBuffer.alloc(gpu.d, grouped.maxItems(pairs, experts, 16) * 12);
        defer items.free();
        var counts = try cuda.DeviceBuffer.alloc(gpu.d, 8);
        defer counts.free();
        var rank = try cuda.DeviceBuffer.alloc(gpu.d, (if (wide) pairs else 1) * 4);
        defer rank.free();
        var hist = try cuda.DeviceBuffer.alloc(gpu.d, (if (wide) (pairs + 1023) / 1024 * experts else 1) * 4);
        defer hist.free();
        const p: grouped.Plan = .{ .members = members.ptr, .items = items.ptr, .counts = counts.ptr, .rank = rank.ptr, .hist = hist.ptr };
        try router.route(stream, dpicks.ptr, pairs, experts, tile, p);
        try stream.synchronize();
        const got_counts = try check.download(gpu, counts);
        defer a.free(got_counts);
        const want_counts = try fx.bytes(try std.fmt.bufPrint(&name, "counts{d}", .{i}));
        defer a.free(want_counts);
        try check.sameBytes("plan counts", got_counts, want_counts);
        const n_items: usize = @intCast(std.mem.bytesToValue(i32, got_counts[0..4]));
        const got_items = try check.download(gpu, items);
        defer a.free(got_items);
        const want_items = try fx.bytes(try std.fmt.bufPrint(&name, "items{d}", .{i}));
        defer a.free(want_items);
        try check.sameBytes("plan items", got_items[0 .. n_items * 12], want_items[0 .. n_items * 12]);
        const got_members = try check.download(gpu, members);
        defer a.free(got_members);
        const want_members = try fx.bytes(try std.fmt.bufPrint(&name, "members{d}", .{i}));
        defer a.free(want_members);
        try check.sameBytes("plan members", got_members, want_members);
        check.pass("grouped plan: {d} rows x {d} slots, {d} experts, tile {d} ({s}, {d} items) equal experts.route's bytes", .{ rows, slots, experts, tile, if (wide) "wide" else "one block", n_items });
    }
}
