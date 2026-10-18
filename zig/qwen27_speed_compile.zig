//! Compile the real Qwen27 startup, vision offer, lane callbacks and GPU encoding without running this object.
const std = @import("std");
const host = @import("qwen27_served");
export fn qwen27SpeedCompile(a: *std.mem.Allocator, io: *std.Io, options: *host.Options) usize {
    var why: []const u8 = "";
    const h = host.open(a.*, io.*, ".", options.*, a.*, &why) catch return 0;
    return @intFromPtr(h.engine().vtable);
}

export fn qwen27EmptyKeepCompile(init: *std.process.Init) u8 {
    @import("tests/qwen27/empty_keep.zig").main(init.*) catch return 1;
    return 0;
}

export fn qwen27SpeedServedCompile(init: *std.process.Init) u8 {
    @import("tests/qwen27/speed_served.zig").main(init.*) catch return 1;
    return 0;
}
