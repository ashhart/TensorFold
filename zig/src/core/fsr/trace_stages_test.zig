const std = @import("std");
const traces = @import("trace.zig");
const plan = @import("plan.zig");
const a = std.testing.allocator;
const Sink = struct {
    bytes: *std.ArrayList(u8),
    pub fn write(s: @This(), text: []const u8) !void {
        try s.bytes.appendSlice(a, text);
    }
};
const Stages = traces.TraceOf(Sink, 3, 128);

test "FSR Flash Next plan and all trace modes equal supplied main golden bytes" {
    const p = try plan.pool(8 << 30, .{ 819200, 102400, 102400, 819200, 102400, 102400, 819200, 102400, 102400 }, 49 * 512, 512);
    const text = try std.json.Stringify.valueAlloc(a, p, .{});
    defer a.free(text);
    try std.testing.expectEqualStrings(@embedFile("fixtures/flashnext-plan.json"), text);
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    var trace = try traces.Trace(Sink).init(.{ .bytes = &bytes }, 48, 512, 3072000, 0xabc);
    inline for (.{ .decode, .prompt, .batch }) |mode| {
        for (0..48) |layer| try trace.routeMeta(mode, layer, &.{ 0, 9, 511 }, 2, 2);
    }
    try trace.route(.mtp, 48, &.{ 5, 511 });
    try trace.close();
    try std.testing.expectEqualStrings(@embedFile("fixtures/flashnext-trace.jsonl"), bytes.items);
}

test "FSR trace includes all three MTP stages with their own expert range" {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    var trace = try Stages.init(.{ .bytes = &bytes }, 40, 384, 18800640, 0xdef);
    for (0..40) |layer| try trace.route(.decode, layer, &.{ 0, 383 });
    try trace.route(.mtp, 40, &.{127});
    try std.testing.expectError(error.IncompleteTrace, trace.close());
    try trace.route(.mtp, 41, &.{ 0, 127 });
    try std.testing.expectError(error.IncompleteTrace, trace.close());
    try trace.route(.mtp, 42, &.{1});
    try trace.close();
    try std.testing.expectEqualStrings(@embedFile("fixtures/mtp-stages.jsonl"), bytes.items);
    try std.testing.expectEqual(@as(usize, 2), trace.windows);
}

test "FSR multi-stage trace refuses a missing reordered or out-of-range stage and incomplete close" {
    for (0..5) |case| {
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(a);
        var trace = try Stages.init(.{ .bytes = &bytes }, 40, 384, 18800640, 0xdef);
        switch (case) {
            0 => try std.testing.expectError(error.InvalidTraceOrder, trace.route(.mtp, 41, &.{0})),
            1 => {
                try trace.route(.mtp, 40, &.{0});
                try std.testing.expectError(error.InvalidTraceOrder, trace.route(.mtp, 42, &.{0}));
            },
            2 => {
                try trace.route(.mtp, 40, &.{0});
                try std.testing.expectError(error.InvalidTraceOrder, trace.route(.decode, 0, &.{0}));
            },
            3 => try std.testing.expectError(error.InvalidExpert, trace.route(.mtp, 40, &.{128})),
            4 => try std.testing.expectError(error.InvalidExpert, trace.route(.decode, 0, &.{384})),
            else => unreachable,
        }
        try std.testing.expectError(error.IncompleteTrace, trace.close());
    }
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    try std.testing.expectError(error.InvalidTraceHeader, Stages.init(.{ .bytes = &bytes }, 0, 384, 18800640, 0));
    try std.testing.expectError(error.InvalidTraceHeader, Stages.init(.{ .bytes = &bytes }, std.math.maxInt(usize), 384, 18800640, 0));
    try std.testing.expectEqual(@as(usize, 0), bytes.items.len);
}
