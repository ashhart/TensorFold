const std = @import("std");
const reply_text = @import("reply_text.zig");
const Server = @import("server.zig").Server;

/// The request's stop strings as the engine checks them after each token: the newest tokens' text holds one.
pub const StopHook = struct {
    srv: *Server,
    stops: reply_text.Stops,

    pub fn check(ctx: *anyopaque, emitted: []const u32) bool {
        const h: *StopHook = @ptrCast(@alignCast(ctx));
        var arena: std.heap.ArenaAllocator = .init(h.srv.gpa);
        defer arena.deinit();
        // Inkling 2's byte tokens can spell one character in several: its window counts the stop's bytes
        const inkling = @import("chat_inkling2.zig");
        const window = if (inkling.enabled(h.srv)) inkling.stopWindow(h.stops.strings) else h.stops.tail();
        const tail = emitted[emitted.len -| window..];
        if (inkling.enabled(h.srv)) return inkling.stopIn(arena.allocator(), h.srv.text, tail, h.stops.strings) catch false;
        const text = h.srv.text.decode(arena.allocator(), tail) catch return false;
        for (h.stops.strings) |stop| if (std.mem.indexOf(u8, text, stop) != null) return true;
        return false;
    }
};
