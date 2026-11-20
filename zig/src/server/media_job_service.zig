//! Server-owned durable image jobs using the production image engine and cancellation sink.
const std = @import("std");
const api = @import("engine_api");
const media = @import("media_job.zig");
const images = @import("images_generations.zig");
const json = @import("json");
const errors = @import("errors.zig");
const request_gate = @import("request_gate.zig");
const Allocator = std.mem.Allocator;
const Conn = @import("http_conn.zig").Conn;
const Server = @import("server.zig").Server;

const Active = struct {
    serial: u64,
    engine_id: ?api.Id = null,
    cancelled: bool = false,
    ended: bool = false,
    thread: ?std.Thread = null,
};

/// Deterministic host coverage for cancellation before Active registration.
pub var before_active_hook: ?*const fn (media.Invocation) void = null;

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
            s.shutdown();
            s.registry.deinit();
        }
        try s.registry.attach(.{ .ctx = s, .start = start, .cancel = cancel, .join = join });
        s.enabled = true;
    }

    pub fn deinit(s: *Service) void {
        if (!s.enabled) return;
        s.shutdown();
        s.active.deinit(s.srv.gpa);
        s.registry.deinit();
        s.* = .{};
    }

    fn find(s: *Service, serial: u64) ?usize {
        for (s.active.items, 0..) |job, index| if (job.serial == serial) return index;
        return null;
    }

    fn reapEnded(s: *Service) void {
        var index: usize = 0;
        while (index < s.active.items.len) {
            if (!s.active.items[index].ended) {
                index += 1;
                continue;
            }
            const job = s.active.items[index];
            job.thread.?.join(); // ended is published after the worker's final use of this mutex
            _ = s.active.swapRemove(index);
        }
    }

    fn shutdown(s: *Service) void {
        s.registry.shutdown() catch |e| std.log.err("durable media shutdown: {t}", .{e});
    }

    fn start(ctx: *anyopaque, invocation: media.Invocation) media.StartError!void {
        const s: *Service = @ptrCast(@alignCast(ctx));
        if (@import("builtin").is_test) if (before_active_hook) |hook| hook(invocation);
        s.mutex.lockUncancelable(s.srv.io);
        s.reapEnded();
        defer s.mutex.unlock(s.srv.io);
        s.active.append(s.srv.gpa, .{ .serial = invocation.serial }) catch return error.Busy;
        // A cancel that found no Active entry above is durable in the registry and must win here.
        s.active.items[s.active.items.len - 1].cancelled = invocation.cancelRequested();
        errdefer _ = s.active.pop();
        const thread = std.Thread.spawn(.{}, work, .{ s, invocation }) catch return error.Busy;
        s.active.items[s.active.items.len - 1].thread = thread;
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
        // The next admission or shutdown joins the handle after the thread really exits.
        s.active.items[index].ended = true;
    }

    fn join(ctx: *anyopaque) void {
        const s: *Service = @ptrCast(@alignCast(ctx));
        while (true) {
            s.mutex.lockUncancelable(s.srv.io);
            if (s.active.items.len == 0) {
                s.mutex.unlock(s.srv.io);
                return;
            }
            const job = s.active.items[0];
            s.mutex.unlock(s.srv.io);
            job.thread.?.join();
            s.mutex.lockUncancelable(s.srv.io);
            _ = s.active.swapRemove(0);
            s.mutex.unlock(s.srv.io);
        }
    }

    fn alive(_: *anyopaque) bool {
        return false;
    }

    fn work(s: *Service, invocation: media.Invocation) void {
        {
            var arena: std.heap.ArenaAllocator = .init(s.srv.gpa);
            defer arena.deinit();
            const a = arena.allocator();
            var entered = request_gate.enter(s.srv, a, invocation.owner, .other, .{ .text = false, .embedding = false, .rerank = false, .systemone = false });
            defer entered.held.release();
            var pngs: std.ArrayList(std.ArrayList(u8)) = .empty;
            defer for (pngs.items) |*png| png.deinit(s.srv.gpa);
            const outcome = if (entered.refusal) |refusal| media.Outcome{ .failed = switch (refusal) {
                .draining => "the durable media service is shutting down",
                .overloaded => |refused| @import("request_admission.zig").message(refused.reason),
                .model_kind => "the loaded model does not serve images",
                .waiting => |waiting| waiting.problem,
            } } else execute(s, invocation, a, &pngs) catch |e| media.Outcome{ .failed = @errorName(e) };
            invocation.finish(outcome) catch |e| std.log.err("durable media completion {s}: {t}", .{ invocation.id, e });
        }
        s.leave(invocation.serial);
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
