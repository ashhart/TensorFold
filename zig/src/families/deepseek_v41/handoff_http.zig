//! Decoder side of the HTTP hand-off: `GET <producer>/v1/handoff/<id>` streams into `handoff_file.pullFrom`.
const std = @import("std");
const fabric = @import("fabric");
const file = @import("handoff_file.zig");
const decoder = fabric.decoder;

pub const Source = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    /// `http://host[:port]`, no path
    base: []const u8,
    /// the producer's API key, sent as a Bearer token (null: none)
    key: ?[]const u8 = null,
    /// the server stops: a fetch under way is cut off
    closing: ?*const std.atomic.Value(bool) = null,
    /// the whole fetch's time
    budget_ns: u64 = 60 * std.time.ns_per_s,

    /// Pulls hand-off `id` into `sink` as the file replay does; adds NotHttp, Unauthorized, NotHeld, Refused, TimedOut.
    pub fn pull(s: Source, arena: std.mem.Allocator, id: [16]u8, expect: decoder.Expect, sink: decoder.Sink, reason: *decoder.Reason) anyerror!decoder.Pulled {
        var x: Exchange = .{ .s = s, .id = id, .op = .get, .arena = arena, .expect = expect, .sink = sink, .reason = reason };
        try x.bounded(s.budget_ns);
        return x.pulled.?;
    }

    /// The producer's copy released (DELETE): best effort, a few seconds at most.
    pub fn release(s: Source, id: [16]u8) void {
        var x: Exchange = .{ .s = s, .id = id, .op = .delete, .arena = s.gpa, .expect = undefined, .sink = undefined, .reason = undefined };
        x.bounded(5 * std.time.ns_per_s) catch {};
    }
};

const Exchange = struct {
    s: Source,
    id: [16]u8,
    op: enum { get, delete },
    arena: std.mem.Allocator,
    expect: decoder.Expect,
    sink: decoder.Sink,
    reason: *decoder.Reason,
    done: std.atomic.Value(bool) = .init(false),
    failure: ?anyerror = null,
    pulled: ?decoder.Pulled = null,

    /// The exchange on its own task, so a stalled producer or a stopping server can cut it off.
    fn bounded(x: *Exchange, budget_ns: u64) anyerror!void {
        const io = x.s.io;
        const deadline = fabric.words.nowNs() + budget_ns;
        var future = io.concurrent(run, .{x}) catch return error.Unavailable;
        while (!x.done.load(.acquire)) {
            const closing = if (x.s.closing) |c| c.load(.acquire) else false;
            if (closing or fabric.words.nowNs() >= deadline) {
                future.cancel(io);
                return if (closing) error.Closed else error.TimedOut;
            }
            io.sleep(.fromMilliseconds(1), .awake) catch {
                future.cancel(io);
                return error.TimedOut;
            };
        }
        future.await(io);
        if (x.failure) |err| return err;
    }

    fn run(x: *Exchange) void {
        defer x.done.store(true, .release);
        x.exchange() catch |err| {
            x.failure = err;
        };
    }

    fn exchange(x: *Exchange) !void {
        const s = x.s;
        var client: std.http.Client = .{ .allocator = s.gpa, .io = s.io };
        defer client.deinit();
        const url = try std.fmt.allocPrint(s.gpa, "{s}/v1/handoff/{x}", .{ std.mem.trimEnd(u8, s.base, "/"), &x.id });
        defer s.gpa.free(url);
        const uri = try std.Uri.parse(url);
        if (!std.ascii.eqlIgnoreCase(uri.scheme, "http")) return error.NotHttp;
        var bearer: [4096 + 7]u8 = undefined;
        const auth: std.http.Client.Request.Headers.Value = if (s.key) |k| .{ .override = try std.fmt.bufPrint(&bearer, "Bearer {s}", .{k}) } else .omit;
        var req = try client.request(if (x.op == .get) .GET else .DELETE, uri, .{
            .keep_alive = false,
            .redirect_behavior = .unhandled,
            .headers = .{ .accept_encoding = .{ .override = "identity" }, .authorization = auth },
        });
        defer req.deinit();
        try req.sendBodiless();
        var resp = try req.receiveHead(&.{});
        if (x.op == .delete) return; // released (or already gone): nothing to read
        switch (resp.head.status) {
            .ok => {},
            .unauthorized => return error.Unauthorized,
            .not_found => return error.NotHeld,
            .conflict => {
                try x.refusal(&resp);
                return error.Refused;
            },
            else => return error.BadStatus,
        }
        if (resp.head.content_length == null or resp.head.transfer_encoding != .none or resp.head.content_encoding != .identity) return error.NoLength;
        const buffer = try s.gpa.alloc(u8, file.reply_area);
        defer s.gpa.free(buffer);
        const body = resp.reader(buffer);
        x.pulled = try file.pullFrom(x.arena, body, x.expect, x.sink, x.reason);
    }

    /// The producer's reason for a failed export (a 409's JSON message) into `reason`.
    fn refusal(x: *Exchange, resp: *std.http.Client.Response) !void {
        var buffer: [1024]u8 = undefined;
        const r = resp.reader(&buffer);
        const text = r.allocRemaining(x.s.gpa, .limited(4096)) catch return;
        defer x.s.gpa.free(text);
        const parsed = std.json.parseFromSlice(std.json.Value, x.s.gpa, text, .{}) catch return setReason(x.reason, text);
        defer parsed.deinit();
        setReason(x.reason, errorMessage(parsed.value) orelse text);
    }

    /// `{"error": {"message": "..."}}`'s message.
    fn errorMessage(v: std.json.Value) ?[]const u8 {
        if (v != .object) return null;
        const e = v.object.get("error") orelse return null;
        if (e != .object) return null;
        const m = e.object.get("message") orelse return null;
        return if (m == .string) m.string else null;
    }

    fn setReason(reason: *decoder.Reason, text: []const u8) void {
        reason.len = @min(text.len, reason.buf.len);
        @memcpy(reason.buf[0..reason.len], text[0..reason.len]);
    }
};
