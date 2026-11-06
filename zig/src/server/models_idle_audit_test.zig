//! A controlled reload interleaving at the idle manager's actual mutex, without a timer or device.
const std = @import("std");
const json = @import("json");
const Server = @import("server.zig").Server;
const Manager = @import("idle_unload.zig").Manager;
const Conn = @import("http_conn.zig").Conn;

const Reload = struct {
    srv: *Server,
    manager: *Manager,
    waits: usize = 0,

    fn wait(ctx: ?*anyopaque, ptr: *const u32, _: u32) void {
        const r: *Reload = @ptrCast(@alignCast(ctx.?));
        std.debug.assert(ptr == @as(*const u32, @ptrCast(&r.manager.mutex.state.raw)));
        r.waits += 1;
        r.srv.info = .{ .context_window = 8192 };
        r.manager.mutex.unlock(std.testing.io);
    }

    fn wake(_: ?*anyopaque, ptr: *const u32, count: u32) void {
        std.testing.io.vtable.futexWake(std.testing.io.userdata, ptr, count);
    }
};

const Wire = struct {
    out: std.Io.Writer.Allocating,
    fn write(ctx: *anyopaque, bytes: []const u8) error{Closed}!void {
        const self: *Wire = @ptrCast(@alignCast(ctx));
        self.out.writer.writeAll(bytes) catch return error.Closed;
    }
};

test "capability audit: model discovery waits for an idle reload and reads the published info" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Only discovery's fields are initialized; neither an engine nor a model is attached.
    var srv: Server = undefined;
    srv.io = std.testing.io;
    srv.info = .{ .context_window = 4096 };
    srv.config = .{ .served_name = "model", .model_ids = &.{ "model", "alias" } };
    srv.thinking_switch = false;
    srv.effort_levels = &.{};
    srv.think_close = &.{};
    // No close marker, so discovery reports the off switch without rendering a request.
    srv.close_id = null;
    // Discovery reads the template for effort use; this one names none.
    var text_vtable: @TypeOf(srv.text.vtable.*) = undefined;
    text_vtable.template_source = struct {
        fn source(_: *anyopaque) []const u8 {
            return "";
        }
    }.source;
    srv.text = .{ .ctx = undefined, .vtable = &text_vtable };
    var manager: Manager = .{ .gpa = a, .io = std.testing.io, .args = .{}, .re = undefined, .dir = "", .model_type = "", .idle_s = 1 };
    srv.idle_manager = &manager;
    var reload: Reload = .{ .srv = &srv, .manager = &manager };
    var vtable = std.testing.io.vtable.*;
    vtable.futexWaitUncancelable = Reload.wait;
    vtable.futexWake = Reload.wake;
    manager.io = .{ .userdata = &reload, .vtable = &vtable };
    manager.mutex.lockUncancelable(std.testing.io);
    defer if (manager.mutex.state.load(.monotonic) != .unlocked) manager.mutex.unlock(std.testing.io);
    var wire: Wire = .{ .out = .init(a) };
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a, .version = "HTTP/1.1", .method = "GET", .write_hooks = .{ .ctx = &wire, .write = Wire.write } };
    try @import("status_routes.zig").models(&srv, &conn, a);
    try std.testing.expectEqual(@as(usize, 1), reload.waits);
    const bytes = wire.out.written();
    const split = std.mem.indexOf(u8, bytes, "\r\n\r\n").? + 4;
    const value = (try json.parse(a, bytes[split..])).ok;
    for (value.get("data").?.array) |model| {
        try std.testing.expectEqual(@as(i64, 8192), model.get("context_length").?.int64().?);
        try std.testing.expectEqual(@as(i64, 8192), model.get("max_model_len").?.int64().?);
    }
}
