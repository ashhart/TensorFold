//! Server-owned durable image jobs using the production image engine and cancellation sink.
const std = @import("std");
const api = @import("engine_api");
const media = @import("media_job.zig");
const images = @import("images_generations.zig");
const json = @import("json");
const errors = @import("errors.zig");
const Allocator = std.mem.Allocator;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;

const Active = struct {
    serial: u64,
    engine_id: ?api.Id = null,
    cancelled: bool = false,
};

pub const Service = struct {
    srv: *Server = undefined,
    registry: media.Registry = undefined,
    mutex: std.Io.Mutex = .init,
    active: std.ArrayList(Active) = .empty,
    enabled: bool = false,

    pub fn init(s: *Service, srv: *Server, directory: []const u8, limits: media.Limits) !void {
        s.* = .{ .srv = srv };
        errdefer s.active.deinit(srv.gpa);
        try s.registry.init(srv.gpa, srv.io, directory, limits);
        errdefer {
            s.registry.shutdown() catch {};
            s.registry.deinit();
        }
        try s.registry.attach(.{ .ctx = s, .start = start, .cancel = cancel });
        s.enabled = true;
    }

    pub fn deinit(s: *Service) void {
        if (!s.enabled) return;
        s.registry.shutdown() catch |e| std.log.err("durable media shutdown: {t}", .{e});
        s.active.deinit(s.srv.gpa);
        s.registry.deinit();
        s.* = .{};
    }

    fn find(s: *Service, serial: u64) ?usize {
        for (s.active.items, 0..) |job, index| if (job.serial == serial) return index;
        return null;
    }

    fn start(ctx: *anyopaque, invocation: media.Invocation) media.StartError!void {
        const s: *Service = @ptrCast(@alignCast(ctx));
        s.mutex.lockUncancelable(s.srv.io);
        defer s.mutex.unlock(s.srv.io);
        s.active.append(s.srv.gpa, .{ .serial = invocation.serial }) catch return error.Busy;
        errdefer _ = s.active.pop();
        const thread = std.Thread.spawn(.{}, work, .{ s, invocation }) catch return error.Busy;
        thread.detach();
    }

    fn cancelled(s: *Service, serial: u64) bool {
        s.mutex.lockUncancelable(s.srv.io);
        defer s.mutex.unlock(s.srv.io);
        const index = s.find(serial) orelse return true;
        return s.active.items[index].cancelled;
    }

    fn setEngine(s: *Service, serial: u64, id: ?api.Id) void {
        s.mutex.lockUncancelable(s.srv.io);
        defer s.mutex.unlock(s.srv.io);
        const index = s.find(serial) orelse return;
        s.active.items[index].engine_id = id;
    }

    fn cancel(ctx: *anyopaque, serial: u64) void {
        const s: *Service = @ptrCast(@alignCast(ctx));
        s.mutex.lockUncancelable(s.srv.io);
        const index = s.find(serial) orelse {
            s.mutex.unlock(s.srv.io);
            return;
        };
        s.active.items[index].cancelled = true;
        const engine_id = s.active.items[index].engine_id;
        s.mutex.unlock(s.srv.io);
        if (engine_id) |id| s.srv.engine.cancel(id);
    }

    fn leave(s: *Service, serial: u64) void {
        s.mutex.lockUncancelable(s.srv.io);
        defer s.mutex.unlock(s.srv.io);
        const index = s.find(serial) orelse return;
        _ = s.active.swapRemove(index);
    }

    fn alive(_: *anyopaque) bool {
        return false;
    }

    fn work(s: *Service, invocation: media.Invocation) void {
        var arena: std.heap.ArenaAllocator = .init(s.srv.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        var pngs: std.ArrayList(std.ArrayList(u8)) = .empty;
        defer for (pngs.items) |*png| png.deinit(s.srv.gpa);
        const outcome = execute(s, invocation, a, &pngs) catch |e| media.Outcome{ .failed = @errorName(e) };
        s.leave(invocation.serial);
        invocation.finish(outcome) catch |e| std.log.err("durable media completion {s}: {t}", .{ invocation.id, e });
    }

    fn execute(s: *Service, invocation: media.Invocation, a: Allocator, pngs: *std.ArrayList(std.ArrayList(u8))) !media.Outcome {
        if (invocation.kind != .image) return .{ .failed = "this durable executor serves image jobs only" };
        if (s.cancelled(invocation.serial)) return .cancelled;
        const cap = s.srv.info.image orelse return .{ .failed = "the loaded model does not serve images" };
        const raw = switch (try json.parse(a, invocation.request)) {
            .ok => |value| value,
            .err => |message| return .{ .failed = message },
        };
        var cx: errors.Cx = .{ .a = a };
        const plan = images.prepare(s.srv, &cx, cap, raw) catch return .{ .failed = cx.message };
        if (plan.stream) return .{ .failed = "durable image jobs do not accept stream or preview fields" };
        var outputs: std.ArrayList(media.Output) = .empty;
        var empty: [0]u8 = .{};
        var conn: Conn = .{
            .fd = -1,
            .peer = "",
            .buf = &empty,
            .gpa = a,
            .gone_hook = .{ .ctx = s, .call = alive },
        };
        for (0..plan.count) |index| {
            if (s.cancelled(invocation.serial)) return .cancelled;
            try pngs.append(a, .empty);
            const png = &pngs.items[pngs.items.len - 1];
            var result: api.ImageResult = .{};
            const engine_id = s.srv.next_id.fetchAdd(1, .monotonic);
            s.setEngine(invocation.serial, engine_id);
            images.generateSubmitted(s.srv, &conn, &cx, plan, plan.seeds[index], engine_id, &result, png) catch {
                s.setEngine(invocation.serial, null);
                if (s.cancelled(invocation.serial)) return .cancelled;
                return .{ .failed = if (cx.message.len > 0) cx.message else "the image generation failed" };
            };
            s.setEngine(invocation.serial, null);
            try outputs.append(a, .{ .media_type = "image/png", .bytes = png.items[0..@min(result.bytes, png.items.len)] });
        }
        if (s.cancelled(invocation.serial)) return .cancelled;
        return .{ .succeeded = outputs.items };
    }
};
