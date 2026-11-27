//! Diagnostic cost replay uses the same untraced observations in both arms, never tracing-inclusive wall time.
const std = @import("std");
const be = @import("backend.zig");
const Stream = @import("stream.zig").Stream;
pub const Event = struct {
    kind: enum { round, start, elapsed, verify, frame, commit, done },
    key: u64 = 0,
    mark: be.Mark = .cost,
    ms: f64 = 0,
};
threadlocal var current: ?*Tape = null;
pub fn frame(key: u64) void {
    if (current) |p| _ = p.apply(.{ .kind = .frame, .key = key });
}
pub const Tape = struct {
    a: std.mem.Allocator,
    base: be.Clock,
    replay: ?[]const Event = null,
    recorded: std.ArrayList(Event) = .empty,
    at: usize = 0,
    flushed: usize = 0,
    table: be.Clock.VTable = undefined,
    pub fn deinit(p: *Tape) void {
        if (current == p) current = null;
        p.recorded.deinit(p.a);
    }
    pub fn take(p: *Tape, event: Event) !f64 {
        if (p.replay) |events| {
            if (p.at >= events.len) return error.PolicyClockTapeExhausted;
            const saved = events[p.at];
            if (saved.kind != event.kind or saved.key != event.key or saved.mark != event.mark)
                return error.PolicyClockWorkloadChanged;
            if (!std.math.isFinite(saved.ms) or saved.ms < 0) return error.InvalidPolicyClockCost;
            p.at += 1;
            return saved.ms;
        }
        try p.recorded.append(p.a, event);
        return event.ms;
    }
    fn apply(p: *Tape, event: Event) f64 {
        return p.take(event) catch |err| std.debug.panic("TF_POLICY_ERROR {s}", .{@errorName(err)});
    }
    pub fn clock(p: *Tape) be.Clock {
        p.table = .{ .start = start, .elapsed_ms = elapsed, .host_ns = if (p.base.vtable.host_ns != null) host else null, .parallel_host_ns = p.base.vtable.parallel_host_ns, .policy = policy };
        return .{ .ptr = p, .vtable = &p.table };
    }
    fn self(ptr: *anyopaque) *Tape {
        return @ptrCast(@alignCast(ptr));
    }
    fn start(ptr: *anyopaque) void {
        const p = self(ptr);
        current = p;
        p.base.start();
        _ = p.apply(.{ .kind = .start });
    }
    fn elapsed(ptr: *anyopaque, mark: be.Mark) f64 {
        const p = self(ptr);
        return p.apply(.{ .kind = .elapsed, .mark = mark, .ms = if (p.replay == null) p.base.elapsedMs(mark) else 0 });
    }
    fn host(ptr: *anyopaque) i128 {
        const p = self(ptr);
        return p.base.vtable.host_ns.?(p.base.ptr);
    }
    fn policy(ptr: *anyopaque, kind: be.PolicyMark, key: u64) void {
        const p = self(ptr);
        _ = p.apply(.{ .kind = switch (kind) {
            .round => .round,
            .verify => .verify,
            .commit => .commit,
            .done => .done,
        }, .key = key });
        if (kind == .done) {
            current = null;
            const events = if (p.replay) |v| v[0..p.at] else p.recorded.items;
            for (events[p.flushed..]) |event| {
                var buffer: [256]u8 = undefined;
                var out = std.Io.Writer.fixed(&buffer);
                std.json.Stringify.value(event, .{}, &out) catch @panic("policy event encoding");
                std.debug.print("TF_POLICY {s}\n", .{out.buffered()});
            }
            p.flushed = events.len;
        }
    }
    pub fn complete(p: *Tape) bool {
        return if (p.replay) |events| p.at == events.len else p.flushed == p.recorded.items.len;
    }
};
pub fn roundKey(streams: []const *Stream) u64 {
    var h = std.hash.Wyhash.init(0);
    for (streams) |s| {
        h.update(std.mem.sliceAsBytes(s.prompt()));
        h.update(std.mem.sliceAsBytes(s.emitted()));
        h.update(std.mem.asBytes(&s.cache_len));
        if (s.pending) |pending| h.update(std.mem.asBytes(&pending));
    }
    return h.final();
}
pub fn verifyKey(windows: []const be.Window) u64 {
    var h = std.hash.Wyhash.init(0);
    for (windows) |w| {
        h.update(std.mem.asBytes(&w.pending));
        h.update(std.mem.asBytes(&w.held));
        h.update(std.mem.sliceAsBytes(w.tokens));
        h.update(std.mem.sliceAsBytes(w.positions));
        if (w.parents) |parents| h.update(std.mem.sliceAsBytes(parents));
    }
    return h.final();
}
