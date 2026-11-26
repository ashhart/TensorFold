//! Bounded timestamp receipts from existing command-buffer waits; no handlers, extra waits or retained buffers.
const std = @import("std");
pub const Kind = enum { other, target, argmax };
const max_commands = 128;
pub const Meta = struct {
    streams: usize = 0,
    rows: usize = 0,
    eligible: bool = false,
    keys: [128]usize = @splat(0),
    turns: [128]u64 = @splat(0),
    grants: [128]u32 = @splat(0),
    growth: usize = 0,
};
const Command = struct {
    key: usize,
    use: u32,
    kind: Kind,
    submit: f64,
    observed: f64 = 0,
    gpu: ?[2]f64 = null,
};
pub const Round = struct {
    id: u64 = 0,
    meta: Meta = .{},
    host: f64 = 0,
    target_start: f64 = 0,
    target_end: f64 = 0,
    target_submit: f64 = 0,
    target_observed: f64 = 0,
    arg_start: f64 = 0,
    arg_end: f64 = 0,
    submit: f64 = 0,
    observed: f64 = 0,
    covered_ms: f64 = 0,
    boundary: bool = false,
    valid: bool = false,
};
threadlocal var current: ?*Trace = null;
pub const Trace = struct {
    queue: usize,
    now: *const fn () f64,
    kind: Kind = .other,
    commands: [max_commands]Command = undefined,
    n: usize = 0,
    overflow: bool = false,
    round: Round = .{},
    previous: ?Round = null,
    serial: u64 = 0,

    pub fn begin(t: *Trace, meta: Meta) void {
        std.debug.assert(current == null);
        t.n = 0;
        t.overflow = false;
        t.kind = .other;
        t.serial += 1;
        t.round = .{ .id = t.serial, .meta = meta, .host = t.now() };
        current = t;
    }
    pub fn end(t: *Trace, complete: bool) void {
        current = null;
        const r = t.finish(complete);
        if (t.previous) |p| emit(p, r);
        t.previous = r;
    }
    pub fn flush(t: *Trace) void {
        if (t.previous) |p| emit(p, .{});
        t.previous = null;
    }
    pub fn finish(t: *Trace, complete: bool) Round {
        var r = t.round;
        r.valid = complete and !t.overflow and t.n > 0;
        var arg_count: usize = 0;
        for (t.commands[0..t.n]) |c| {
            if (c.kind == .target) {
                r.target_submit = if (r.target_submit == 0) c.submit else @min(r.target_submit, c.submit);
                r.target_observed = @max(r.target_observed, c.observed);
            } else if (c.kind == .argmax) {
                arg_count += 1;
                r.submit = c.submit;
                r.observed = c.observed;
            }
            const gpu = c.gpu orelse {
                if (c.kind != .other) r.valid = false;
                continue;
            };
            if (!(gpu[0] > 0 and gpu[1] >= gpu[0] and std.math.isFinite(gpu[1]))) {
                r.valid = false;
                continue;
            }
            if (c.kind == .target) {
                r.target_start = if (r.target_start == 0) gpu[0] else @min(r.target_start, gpu[0]);
                r.target_end = @max(r.target_end, gpu[1]);
            } else if (c.kind == .argmax) {
                r.arg_start = gpu[0];
                r.arg_end = gpu[1];
            }
        }
        r.boundary = arg_count == 1;
        r.valid = r.valid and r.target_start > 0 and (arg_count == 0 or (arg_count == 1 and r.arg_start >= r.target_end));
        if (r.boundary) {
            var spans: [max_commands][2]f64 = undefined;
            var count: usize = 0;
            for (t.commands[0..t.n]) |c| if (c.kind == .other) {
                const gpu = c.gpu orelse {
                    if (c.submit < r.submit) r.valid = false;
                    continue;
                };
                const lo = @max(r.target_end, gpu[0]);
                const hi = @min(r.arg_start, gpu[1]);
                if (hi > lo) {
                    spans[count] = .{ lo, hi };
                    count += 1;
                }
            };
            std.mem.sort([2]f64, spans[0..count], {}, struct {
                fn less(_: void, x: [2]f64, y: [2]f64) bool {
                    return x[0] < y[0];
                }
            }.less);
            var covered_end = r.target_end;
            for (spans[0..count]) |s| {
                r.covered_ms += @max(0, s[1] - @max(covered_end, s[0])) * 1000;
                covered_end = @max(covered_end, s[1]);
            }
        }
        return r;
    }
};

pub fn submitted(cb: anytype) void {
    const t = current orelse return;
    if (cb.traceQueue() != t.queue) return;
    if (t.n == t.commands.len) {
        t.overflow = true;
        return;
    }
    t.commands[t.n] = .{ .key = cb.traceKey(), .use = cb.use, .kind = t.kind, .submit = t.now() };
    t.n += 1;
}
pub fn waited(cb: anytype) void {
    const t = current orelse return;
    var i = t.n;
    while (i > 0) {
        i -= 1;
        const c = &t.commands[i];
        if (c.key == cb.traceKey() and c.use == cb.use) {
            c.gpu = cb.gpuTimes();
            c.observed = t.now();
            return;
        }
    }
}
pub fn same(a: Meta, b: Meta) bool {
    if (a.streams == 0 or a.streams > a.keys.len or a.streams != b.streams or a.growth != b.growth) return false;
    for (0..a.streams) |i| if (a.keys[i] != b.keys[i] or b.turns[i] != a.turns[i] + 1 or a.turns[i] < 2) return false;
    return true;
}
fn emit(p: Round, next: Round) void {
    var labels: [1024]u8 = undefined;
    var used: usize = 0;
    for (p.meta.grants[0..@min(p.meta.streams, p.meta.grants.len)], 0..) |grant, i| {
        const part = std.fmt.bufPrint(labels[used..], "{s}{d}", .{ if (i == 0) "" else "/", grant }) catch break;
        used += part.len;
    }
    std.log.info("q27_phase v=1 id={d} streams={d} rows={d} grants={s} next_rows={d} same={d} eligible={d} gpu={d} boundary={d} target_start={d:.9} target_end={d:.9} arg_start={d:.9} arg_end={d:.9} next_start={d:.9} host={d:.9} next_host={d:.9} target_submit={d:.9} target_observed={d:.9} submit={d:.9} observed={d:.9} covered_ms={d:.9}", .{ p.id, p.meta.streams, p.meta.rows, labels[0..used], next.meta.rows, @intFromBool(same(p.meta, next.meta)), @intFromBool(p.meta.eligible and next.meta.eligible), @intFromBool(p.valid and next.valid), @intFromBool(p.boundary), p.target_start, p.target_end, p.arg_start, p.arg_end, next.target_start, p.host, next.host, p.target_submit, p.target_observed, p.submit, p.observed, p.covered_ms });
}

var reads: usize = 0;
fn testClock() f64 {
    reads += 1;
    return 10;
}
const Fake = struct {
    use: u32 = 1,
    id: usize = 1,
    gpu: ?[2]f64 = .{ 1, 2 },
    fn traceQueue(_: @This()) usize {
        reads += 1;
        return 7;
    }
    fn traceKey(f: @This()) usize {
        reads += 1;
        return f.id;
    }
    fn gpuTimes(f: @This()) ?[2]f64 {
        reads += 1;
        return f.gpu;
    }
};
test "phase trace: off never reads clocks/buffers or installs a handler" {
    current = null;
    reads = 0;
    submitted(Fake{});
    waited(Fake{});
    try std.testing.expectEqual(@as(usize, 0), reads);
}
test "phase trace: completed times exclude argmax runtime and union covered work" {
    var t: Trace = .{ .queue = 7, .now = testClock };
    t.begin(.{});
    defer current = null;
    t.kind = .target;
    submitted(Fake{ .id = 1 });
    waited(Fake{ .id = 1, .gpu = .{ 1, 2 } });
    t.kind = .other;
    submitted(Fake{ .id = 2 });
    waited(Fake{ .id = 2, .gpu = .{ 2.1, 2.3 } });
    submitted(Fake{ .id = 3 });
    waited(Fake{ .id = 3, .gpu = .{ 2.2, 2.4 } });
    t.kind = .argmax;
    submitted(Fake{ .id = 4 });
    waited(Fake{ .id = 4, .gpu = .{ 2.5, 4 } });
    const r = t.finish(true);
    try std.testing.expect(r.valid and r.boundary);
    try std.testing.expectApproxEqAbs(@as(f64, 300), r.covered_ms, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 200), (r.arg_start - r.target_end) * 1000 - r.covered_ms, 1e-9);
}
test "phase trace: missing feedback, overflow and request transitions are invalid" {
    var t: Trace = .{ .queue = 7, .now = testClock };
    t.begin(.{});
    defer current = null;
    t.kind = .target;
    submitted(Fake{});
    waited(Fake{ .gpu = null });
    try std.testing.expect(!t.finish(true).valid);
    try std.testing.expectEqual(@as(f64, 10), t.finish(true).target_observed);
    waited(Fake{});
    try std.testing.expect(t.finish(true).valid);
    for (0..max_commands) |_| submitted(Fake{});
    try std.testing.expect(t.overflow);
    var a: Meta = .{ .streams = 1 };
    a.keys[0] = 1;
    a.turns[0] = 2;
    var b = a;
    b.turns[0] += 1;
    try std.testing.expect(same(a, b));
    b.keys[0] = 2;
    try std.testing.expect(!same(a, b));
}
