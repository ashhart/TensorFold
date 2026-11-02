//! Startup names the disk cache directory and the device budget reported by the engine.
const std = @import("std");
const api = @import("engine_api");

pub fn shown(dir: []const u8, info: api.Info) Shown {
    return .{ .dir = dir, .bytes = if (info.prompt_cache_plan) |p| p.budget_bytes else if (info.prompt_cache_capacity) |c| c.budget_bytes else null };
}

const Shown = struct {
    dir: []const u8,
    bytes: ?u64,

    pub fn format(s: Shown, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("prompt cache: ");
        if (s.bytes) |bytes|
            try w.print("{d:.1} GiB on device", .{@as(f64, @floatFromInt(bytes)) / (1 << 30)})
        else
            try w.writeAll("device budget not reported");
        try w.writeAll(", disk tier at ");
        for (s.dir) |c| try w.writeByte(if (std.ascii.isControl(c)) ' ' else c);
    }
};

test "prompt cache disk: startup uses the engine plan, then capacity, and never invents a device budget" {
    const cases = [_]struct { info: api.Info, dir: []const u8 = "cache", text: []const u8 }{
        .{ .info = .{ .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = 3 << 29 }, .prompt_cache_capacity = .{ .budget_bytes = 8 << 30, .minimum_state_bytes = 1 } }, .text = "prompt cache: 1.5 GiB on device, disk tier at cache" },
        .{ .info = .{ .prompt_cache_capacity = .{ .budget_bytes = 2 << 30, .minimum_state_bytes = 1 } }, .text = "prompt cache: 2.0 GiB on device, disk tier at cache" },
        .{ .info = .{ .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = 0 } }, .text = "prompt cache: 0.0 GiB on device, disk tier at cache" },
        .{ .info = .{}, .text = "prompt cache: device budget not reported, disk tier at cache" },
        .{ .info = .{}, .dir = "cache\nnew\r\x1b", .text = "prompt cache: device budget not reported, disk tier at cache new  " },
    };
    for (cases) |c| {
        var buf: [256]u8 = undefined;
        try std.testing.expectEqualStrings(c.text, try std.fmt.bufPrint(&buf, "{f}", .{shown(c.dir, c.info)}));
    }
}
