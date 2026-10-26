//! Complete actual-route observations. A missing close or a partial window is rejected by the locality tool.
const std = @import("std");
pub fn Trace(comptime Sink: type) type {
    return struct {
        const Self = @This();
        sink: Sink,
        layers: usize,
        experts: usize,
        windows: usize = 0,
        next: ?usize = null,
        mode: enum { decode, prompt, batch, mtp } = .decode,
        failed: bool = false,

        pub fn init(sink: Sink, layers: usize, experts: usize, stride: usize, checkpoint: u64) !Self {
            if (layers == 0 or experts == 0 or stride == 0) return error.InvalidTraceHeader;
            var s: Self = .{ .sink = sink, .layers = layers, .experts = experts };
            try s.print("{{\"format\":\"tensorfold-fsr-v1\",\"layers\":{d},\"experts\":{d},\"slot_bytes\":{d},\"checkpoint\":\"{x}\",\"workload\":\"native-serving\",\"modes\":{{\"decode\":[", .{ layers + 1, experts, stride, checkpoint });
            try s.indices(layers);
            try s.sink.write("],\"prompt\":[");
            try s.indices(layers);
            try s.sink.write("],\"batch\":[");
            try s.indices(layers);
            try s.print("],\"mtp\":[{d}]}}}}\n", .{layers});
            return s;
        }

        fn print(s: *Self, comptime fmt: []const u8, args: anytype) !void {
            var buffer: [512]u8 = undefined;
            try s.sink.write(try std.fmt.bufPrint(&buffer, fmt, args));
        }
        fn indices(s: *Self, n: usize) !void {
            for (0..n) |i| try s.print("{s}{d}", .{ if (i == 0) @as([]const u8, "") else ",", i });
        }

        /// IDs are the real router's complete rows, including every hit. Forecasts never call this method.
        pub fn route(s: *Self, mode: @FieldType(Self, "mode"), layer: usize, ids: []const u32) !void {
            return s.routeMeta(mode, layer, ids, 1, 1);
        }

        pub fn routeMeta(s: *Self, mode: @FieldType(Self, "mode"), layer: usize, ids: []const u32, rows: usize, streams: usize) !void {
            errdefer s.failed = true;
            if (s.failed or ids.len == 0 or rows == 0 or streams == 0) return error.InvalidTraceRoute;
            if (s.next == null) {
                if (layer != (if (mode == .mtp) s.layers else @as(usize, 0))) return error.InvalidTraceOrder;
                s.mode = mode;
                s.next = layer;
                try s.print("{{\"event\":\"begin\",\"window\":{d},\"mode\":\"{s}\",\"rows\":{d},\"streams\":{d}}}\n", .{ s.windows, @tagName(mode), rows, streams });
            }
            if (s.mode != mode or s.next.? != layer) return error.InvalidTraceOrder;
            for (ids) |id| if (id >= s.experts) return error.InvalidExpert;
            try s.print("{{\"event\":\"route\",\"window\":{d},\"layer\":{d},\"ids\":[", .{ s.windows, layer });
            for (ids, 0..) |id, i| try s.print("{s}{d}", .{ if (i == 0) @as([]const u8, "") else ",", id });
            try s.sink.write("]}\n");
            if (mode == .mtp or layer + 1 == s.layers) {
                try s.print("{{\"event\":\"end\",\"window\":{d}}}\n", .{s.windows});
                s.windows += 1;
                s.next = null;
            } else s.next = layer + 1;
        }

        pub fn close(s: *Self) !void {
            if (s.failed or s.next != null) return error.IncompleteTrace;
            try s.print("{{\"event\":\"closed\",\"windows\":{d}}}\n", .{s.windows});
        }
    };
}

test "FSR native trace includes hits and refuses an incomplete or reordered window" {
    const Sink = struct {
        a: std.mem.Allocator,
        bytes: *std.ArrayList(u8),
        pub fn write(s: @This(), text: []const u8) !void {
            try s.bytes.appendSlice(s.a, text);
        }
    };
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(std.testing.allocator);
    var trace = try Trace(Sink).init(.{ .a = std.testing.allocator, .bytes = &bytes }, 2, 4, 96, 42);
    try trace.route(.decode, 0, &.{ 1, 1, 3 });
    try std.testing.expectError(error.IncompleteTrace, trace.close());
    try trace.route(.decode, 1, &.{ 1, 2 });
    try trace.route(.mtp, 2, &.{3});
    try trace.close();
    var lines = std.mem.tokenizeScalar(u8, bytes.items, '\n');
    while (lines.next()) |line| {
        const value = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, line, .{});
        defer value.deinit();
        try std.testing.expect(value.value == .object);
    }
    try std.testing.expect(std.mem.indexOf(u8, bytes.items, "\"ids\":[1,1,3]") != null);
    try std.testing.expect(std.mem.endsWith(u8, bytes.items, "{\"event\":\"closed\",\"windows\":2}\n"));
    try trace.route(.prompt, 0, &.{0});
    try std.testing.expectError(error.InvalidTraceOrder, trace.route(.prompt, 0, &.{1}));
    try std.testing.expectError(error.IncompleteTrace, trace.close());
}
