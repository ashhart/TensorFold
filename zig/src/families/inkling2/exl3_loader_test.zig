//! EXL3 storage keeps shared model geometry and refuses manifest drift before any device allocation.
const std = @import("std");
const cfg = @import("exl3_config.zig");
const wt = @import("weights.zig");
const release = @embedFile("fixtures/config.json");

fn manifest(a: std.mem.Allocator, text: std.json.Value) ![]u8 {
    const body = try std.json.Stringify.valueAlloc(a, text, .{});
    defer a.free(body);
    return std.fmt.allocPrint(a, "{{\"format\":\"kortexa-inkling2-exl3-v1\",\"status\":\"complete\",\"text_config\":{s}}}", .{body});
}

test "pack geometry uses the existing family config and EXL3 layer plan" {
    const a = std.testing.allocator;
    const raw = try std.json.parseFromSlice(std.json.Value, a, release, .{});
    defer raw.deinit();
    const bytes = try manifest(a, raw.value.object.get("text_config").?);
    defer a.free(bytes);
    const c = try cfg.admitSource(a, bytes, release);
    const p = try cfg.layerPlan(&c);
    try std.testing.expectEqual(@as(u32, 48), p.n);
    try std.testing.expectEqual(@as(u32, 8), p.count(.global));
    try std.testing.expectEqual(@as(u32, 40), p.count(.local));
    for (p.layers[0..p.n]) |l| try std.testing.expectEqual(.exl3, l.precision);
    try std.testing.expectEqual(@as(u32, 4608), c.text.hidden);
}

test "an arithmetic setting changed only in the manifest is refused" {
    const a = std.testing.allocator;
    const raw = try std.json.parseFromSlice(std.json.Value, a, release, .{});
    defer raw.deinit();
    const text = raw.value.object.getPtr("text_config").?;
    try text.object.put(raw.arena.allocator(), "rms_norm_eps", .{ .float = 0.1 });
    const bytes = try manifest(a, text.*);
    defer a.free(bytes);
    try std.testing.expectError(error.InklingSourceMismatch, cfg.admitSource(a, bytes, release));
}

test "EXL3 rank partition owns every whole expert once and rejects extra ranks" {
    const left = try cfg.Share.init(0, 2);
    const right = try cfg.Share.init(1, 2);
    try std.testing.expectEqual(left.expertStart() + left.expertCount(), right.expertStart());
    try std.testing.expectEqual(@as(usize, cfg.experts), right.expertStart() + right.expertCount());
    try std.testing.expectError(error.InklingRanks, cfg.Share.init(0, 3));
    _ = wt;
}

test "the shared tensor map keeps dense weights when experts use EXL3 storage" {
    const Sink = struct {
        dense: usize = 0,
        experts: usize = 0,
        pub fn put(self: *@This(), name: []const u8, _: @import("header.zig").Dtype, _: []const u64) !void {
            self.dense += 1;
            if (std.mem.indexOf(u8, name, ".mlp.experts.") != null) self.experts += 1;
        }
    };
    const c = try cfg.source(std.testing.allocator, release);
    const formats: [48]@import("plan.zig").Precision = @splat(.exl3);
    var sink: Sink = .{};
    try @import("names.zig").all(&c, &formats, .model, &sink);
    try std.testing.expect(sink.dense > 100);
    try std.testing.expectEqual(@as(usize, 0), sink.experts);
}
