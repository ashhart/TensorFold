//! Serialized lesson events and an independent heartbeat while writing or a learner unit is blocked.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const sse = @import("sse.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

pub const Out = struct {
    conn: *Conn,
    a: std.mem.Allocator,
    open: bool = true,
    io: ?std.Io = null,
    mutex: std.Io.Mutex = .init,
    ended: std.Io.Event = .unset,
    thread: ?std.Thread = null,
    interval_ms: i64 = 5000,
    id: u32 = 0,
    phase: []const u8 = "writing",
    at: u32 = 0,
    of: u32 = 0,

    /// Start before any model generation; failure refuses the lesson instead of silently losing its heartbeat.
    pub fn start(o: *Out, io: std.Io, interval_ms: i64) !void {
        o.io = io;
        o.interval_ms = interval_ms;
        o.thread = try std.Thread.spawn(.{}, beat, .{o});
    }

    pub fn stop(o: *Out) void {
        if (o.thread) |th| {
            o.ended.set(o.io.?);
            th.join();
            o.thread = null;
        }
    }

    fn lock(o: *Out) void {
        if (o.io) |io| o.mutex.lockUncancelable(io);
    }

    fn unlock(o: *Out) void {
        if (o.io) |io| o.mutex.unlock(io);
    }

    fn beat(o: *Out) void {
        const io = o.io.?;
        while (true) {
            o.ended.waitTimeout(io, .{ .duration = .{ .raw = .fromMilliseconds(o.interval_ms), .clock = .awake } }) catch {};
            o.lock();
            defer o.unlock();
            if (o.ended.isSet() or !o.open) return;
            var buf: [256]u8 = undefined;
            const frame = std.fmt.bufPrint(&buf, "event: progress\ndata: {{\"id\":{d},\"phase\":\"{s}\",\"at\":{d},\"of\":{d},\"heartbeat\":true}}\n\n", .{ o.id, o.phase, o.at, o.of }) catch unreachable;
            o.conn.writeAll(frame) catch {
                o.open = false;
                return;
            };
        }
    }

    pub fn check(o: *Out) bool {
        o.lock();
        defer o.unlock();
        return !o.open or o.conn.peerGone();
    }

    pub fn stage(o: *Out, id: u32, phase: []const u8) void {
        o.lock();
        o.id = id;
        o.phase = phase;
        o.at = 0;
        o.of = 0;
        o.unlock();
        o.event("progress", .{ .id = id, .phase = phase, .at = @as(u32, 0), .of = @as(u32, 0) });
    }

    pub fn progress(o: *Out, id: u32, p: api.LearnProgress) void {
        o.lock();
        o.id = id;
        o.phase = @tagName(p.phase);
        o.at = p.at;
        o.of = p.of;
        o.unlock();
        o.event("progress", .{ .id = id, .phase = @tagName(p.phase), .at = p.at, .of = p.of, .loss = @as(f64, p.loss) });
    }

    pub fn event(o: *Out, kind: []const u8, fields: anytype) void {
        o.lock();
        defer o.unlock();
        if (!o.open) return;
        const obj = json.newObject(o.a) catch return;
        inline for (@typeInfo(@TypeOf(fields)).@"struct".field_names) |name| {
            const v = @field(fields, name);
            const value: json.Value = switch (@TypeOf(v)) {
                bool => .{ .bool = v },
                u32 => json.intValue(o.a, v) catch return,
                f64 => .{ .float = @round(v * 10) / 10 },
                else => .{ .string = v },
            };
            obj.put(o.a, name, value) catch return;
        }
        sse.event(o.conn, o.a, kind, .{ .object = obj }) catch {
            o.open = false;
        };
    }

    pub fn done(o: *Out) void {
        o.stop();
        if (o.open) sse.done(o.conn) catch {};
        o.open = false;
    }

    pub fn fail(o: *Out, cx: *errors.Cx, e: anyerror) void {
        o.event("failed", .{ .message = @import("slide.zig").words(cx, e) });
        o.done();
    }
};
