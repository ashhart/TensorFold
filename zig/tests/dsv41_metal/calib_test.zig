//! The draft cost table timed on the tiny model: a sane table, slots left as new, replies unchanged by it.
const std = @import("std");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fs = @import("fixture_serve.zig");
const ff = @import("fixture_fwd.zig");

const api = host.api;
const Served = host.serve_lanes.Served;
const testing = std.testing;
const gpa = testing.allocator;
const io = testing.io;
const seed = 83;

/// Every byte a slot's state could differ in, and its bookkeeping, over all slots.
fn digest(m: *pkg.serve.Metal) u64 {
    var h = std.hash.Wyhash.init(0);
    for (m.t.slots) |s| {
        for (s.kv.regions) |r| {
            h.update(r.window.contents()[0..r.window.length()]);
            inline for (.{ "pool", "keys", "carry", "inputs" }) |f| if (@field(r, f)) |b| h.update(b.contents()[0..b.length()]);
        }
        if (s.kv.taps) |b| h.update(b.contents()[0..b.length()]);
        h.update(s.ptaps.contents()[0..s.ptaps.length()]);
        h.update(std.mem.sliceAsBytes(s.ring));
        h.update(std.mem.asBytes(&[_]u64{ s.hist.items.len, s.high, s.base, s.rows, s.ptaps_rows, s.tail.end, s.tail.done, @intFromBool(s.live), @intFromBool(s.open), @intFromBool(s.after_prompt) }));
    }
    return h.final();
}

fn say(ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s} dsv41 metal calib: ", .{if (ok) "PASS" else "FAIL"});
    std.debug.print(fmt ++ "\n", args);
}

const want: host.calib_env.Shape = .{ .slots = 4, .world = 1, .dspark = true, .context = fs.limit };

test "dsv41 metal calib: a measured table, every slot as new after it, the stats untouched, the cache never fed" {
    const m = try fs.open(gpa, io, seed, .{ .cache_bytes = 1 << 30 });
    const before = digest(m);
    const stats = m.t.stats;
    const md = m.model();
    var got = try md.calibrate(md.ptr, gpa, io, ".", .{ .mode = "measure" }, want);
    defer got.costs.deinit(gpa);
    const v = got.costs.verify;
    var sane = v.len >= 32 and got.costs.draft > 0.0 and got.costs.slot >= 0.0;
    for (v, 0..) |x, i| sane = sane and std.math.isFinite(x) and x > 0.0 and (i == 0 or x >= v[i - 1]);
    say(sane, "table of {d} rows: 1 / 8 / 16 / 17 / 32 rows {d:.3} / {d:.3} / {d:.3} / {d:.3} / {d:.3} ms, pass {d:.3} ms", .{ v.len, v[0], v[7], v[15], v[16], v[31], got.costs.draft });
    try testing.expect(sane);
    const after = digest(m);
    say(after == before, "slot state digest {x} before, {x} after", .{ before, after });
    try testing.expectEqual(before, after);
    try testing.expectEqual(stats, m.t.stats);
    try testing.expectEqual(@as(u64, 1 << 30), m.t.cache.cap);
    try testing.expectEqual(@as(usize, 0), m.t.cache.entries.items.len);
    md.close(md.ptr);
}

/// Every byte a slot's buffers hold.
fn slotBytes(s: anytype) u64 {
    var n: u64 = s.ptaps.length() + s.ring.len * 2;
    for (s.kv.regions) |r| {
        n += r.window.length();
        inline for (.{ "pool", "keys", "carry", "inputs" }) |f| if (@field(r, f)) |b| {
            n += b.length();
        };
    }
    if (s.kv.taps) |b| n += b.length();
    return n;
}

test "dsv41 metal calib: three fresh starts each price a pass that is plausible against their own rows" {
    for (0..3) |start| {
        const m = try fs.open(gpa, io, seed, .{});
        const md = m.model();
        defer md.close(md.ptr);
        var got = try md.calibrate(md.ptr, gpa, io, ".", .{ .mode = "measure" }, want);
        defer got.costs.deinit(gpa);
        say(got.costs.plausible(), "start {d}: pass {d:.3} ms against a 1-row window of {d:.3} ms", .{ start, got.costs.draft, got.costs.windowMs(1) });
        try testing.expect(got.costs.plausible());
    }
}

test "dsv41 metal calib: the work a measurement does is bounded: a few prompt rows, the planned windows, only the slots it used" {
    const m = try fs.open(gpa, io, seed, .{ .cache_bytes = 1 << 30 });
    const md = m.model();
    defer md.close(md.ptr);
    var full: u64 = 0;
    for (m.t.slots) |s| full += slotBytes(s);
    var rep: pkg.calib.Report = .{};
    var got = try m.calibrateReport(gpa, io, ".", .{ .mode = "measure" }, want, &rep);
    got.costs.deinit(gpa);
    var plan = try host.calib_measure.Plan.initWide(gpa, 4, 64, 16, host.calib_measure.launch_bound, true, &host.calib_measure.launch_wide);
    defer plan.deinit(gpa);
    // a launch counts a window for each slot in it
    var planned: u64 = 0;
    for (plan.runs.items) |r| planned += if (r.what == .draft) 0 else r.nsegs;
    say(true, "{d} prompts of {d} rows, {d} slot windows of {d} planned, {d} slots wiped, {d} of {d} bytes cleared", .{ rep.prompts, rep.prompt_rows, rep.windows, planned, rep.wiped.slots, rep.wiped.bytes, full });
    try testing.expectEqual(@as(u64, plan.n), rep.prompts);
    try testing.expect(rep.prompt_rows <= 8);
    try testing.expect(rep.windows <= planned);
    try testing.expectEqual(plan.n, rep.wiped.slots);
    try testing.expect(rep.wiped.bytes <= @as(u64, plan.n) * (full / m.t.slots.len));
    try testing.expect(rep.wiped.bytes < full);
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "dsv41 metal calib: a malformed timing setting stops the start instead of falling back to the defaults" {
    const m = try fs.open(gpa, io, seed, .{});
    const md = m.model();
    defer md.close(md.ptr);
    const before = digest(m);
    for ([_][2][:0]const u8{ .{ "TF_DSV41_CALIB_BUDGET_MS", "abc" }, .{ "TF_DSV41_CALIB_REPS", "0" }, .{ "TF_DSV41_CALIB_STAT", "mean" } }) |kv| {
        _ = setenv(kv[0], kv[1], 1);
        defer _ = unsetenv(kv[0]);
        const got = md.calibrate(md.ptr, gpa, io, ".", .{ .mode = "measure" }, want);
        say(std.meta.isError(got), "{s}={s} refused by name before any window ran", .{ kv[0], kv[1] });
        try testing.expectError(error.InvalidEnvironment, got);
    }
    try testing.expectEqual(before, digest(m));
    var ok = try md.calibrate(md.ptr, gpa, io, ".", .{ .mode = "measure" }, want);
    ok.costs.deinit(gpa);
}

test "dsv41 metal calib: a reply after the measurement equals the serial decode, drafted and not" {
    const prompt = try ff.ids(gpa, &fs.config(), 200, 9);
    defer gpa.free(prompt);
    const ref = try fs.open(gpa, io, seed, .{ .slots = 1, .drafts = false });
    const want_tokens = try ref.t.serial(0, prompt, 40, null);
    defer gpa.free(want_tokens);
    fs.close(ref);
    const m = try fs.open(gpa, io, seed, .{ .measure_costs = true });
    const md = m.model();
    // Served.open times the table (mode `measure` from the model's own knobs) before its first request
    const s = try Served.open(gpa, io, md, ".", null);
    defer Served.close(s);
    const e = s.engine();
    for ([_]bool{ true, false }) |drafts| {
        var box: Box = .{};
        defer box.deinit();
        try e.submit(1 + @as(api.Id, @intFromBool(drafts)), &.{ .prompt = prompt, .max_tokens = 40, .drafts = drafts }, .{ .ctx = &box, .event = Box.event });
        try box.wait();
        const same = std.mem.eql(u32, want_tokens, box.out.items);
        say(same, "reply after the measurement (drafts {any}) equals the serial decode: {d} tokens", .{ drafts, box.out.items.len });
        try testing.expect(same);
    }
    try testing.expect(m.t.stats.prompts >= 2);
}

/// A reply's tokens, written on LaneHost's thread and read on the test's.
const Box = struct {
    mutex: std.Io.Mutex = .init,
    out: std.ArrayList(u32) = .empty,
    done: bool = false,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |ts| b.out.appendSlice(gpa, ts) catch {},
            .finished => b.done = true,
            else => {},
        }
    }

    fn wait(b: *Box) !void {
        for (0..600_000) |_| {
            b.mutex.lockUncancelable(io);
            const done = b.done;
            b.mutex.unlock(io);
            if (done) return;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }

    fn deinit(b: *Box) void {
        b.out.deinit(gpa);
    }
};
