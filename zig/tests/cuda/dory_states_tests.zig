//! Dory's kept prompt states: a state at 8k or 32k resumed into wiped caches gives the fresh pass's replies.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const dory = @import("dory");
const check = @import("check.zig");
const serve = @import("dory_serve_tests.zig");
const Gpu = check.Gpu;

const be = dory.cuda_backend;
const snap = dory.cuda_snap;

/// Positions kept: off the chunk grid at 8k and 32k, and on it; each prompt runs 2,600 rows past (a chunk, a window).
const marks = [_]u32{ 8000, 8192, 32001 };
const past = 2600;

/// Every cache of every slot overwritten with 0xFF bytes (bf16 NaN): only what a restore writes back is read.
fn wipe(m: *const dory.cuda_model.Model, s: cuda.Stream) !void {
    if (m.paging) |p| try m.d.check(m.d.api.cuMemsetD8Async(p.arena, 0xFF, p.pool * p.pageBytes(), s.handle), "cuMemsetD8Async");
    for (m.k, m.v, 0..) |kc, vc, i| {
        if (m.paging != null and m.sched.cache_kind[i] == .full) continue; // the arena above
        const bytes = m.slots * m.cacheLen(i) * m.c.kvDim() * 2;
        try m.d.check(m.d.api.cuMemsetD8Async(kc, 0xFF, bytes, s.handle), "cuMemsetD8Async");
        try m.d.check(m.d.api.cuMemsetD8Async(vc, 0xFF, bytes, s.handle), "cuMemsetD8Async");
    }
    try s.synchronize();
}

const Keep = struct {
    db: *be.Dory,
    saved: ?*snap.Saved = null,
    fn hook(ptr: *anyopaque, s: *lanes.Stream, mark: u32) void {
        const k: *@This() = @ptrCast(@alignCast(ptr));
        k.saved = k.db.saveState(s, mark) catch null;
    }
};

/// One stream's reply from `saved` at `at`, through the lane core (copy drafts off), into `out`.
fn resumed(gpu: Gpu, db: *be.Dory, p: []const u32, sampling: ?lanes.Sampling, saved: *snap.Saved, at: u32, count: u32, out: *std.ArrayList(u32)) !u64 {
    const gpa = gpu.gpa;
    var t = try lanes.Stream.init(gpa, .{ .id = "resume", .prompt = p, .max_new = count, .sampling = sampling, .drafts = false, .reuse = .{ .saved = saved, .at = at } });
    defer t.deinit(gpa);
    var cfg = try lanes.Config.init(gpa, db.facts(), be.max_window, be.max_window - 1);
    defer cfg.deinit(gpa);
    var clock: lanes.fake.FixedClock = .{};
    var engine = lanes.Engine.init(gpa, &cfg, db.backend(), clock.clock());
    defer engine.deinit();
    try engine.addStream(&t);
    while (engine.activeCount() > 0) try engine.step();
    try out.appendSlice(gpa, t.emitted());
    return if (t.reuse_failed) 0 else t.cached;
}

/// tf-cuda-test dory-states MODEL LOOPS TEXT N: greedy and sampled replies resumed at each mark against fresh.
pub fn states(gpu: Gpu, rest: []const []const u8) !void {
    const gpa = gpu.gpa;
    var x: serve.Setup = undefined;
    try serve.setup(&x, gpu, rest, marks[marks.len - 1] + past, 2);
    defer serve.teardown(&x, gpa);
    const sp: lanes.Sampling = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.95 };
    var bad: usize = 0;
    for (marks) |at| {
        const p = x.ids[0 .. at + past];
        const fresh = try serve.lanesRun(gpu, &x.db, &.{ .{ .prompt = p }, .{ .prompt = p, .sampling = sp } }, x.count);
        defer serve.freeRun(gpa, fresh);
        var keep: Keep = .{ .db = &x.db };
        var ms = [_]u32{at};
        var s = try lanes.Stream.init(gpa, .{ .id = "keep", .prompt = p[0 .. at + 100], .max_new = 1, .drafts = false, .reuse = .{ .marks = &ms, .hook = .{ .ptr = &keep, .at = Keep.hook } } });
        defer s.deinit(gpa);
        const b = x.db.backend();
        try b.prefill(&s);
        b.release(&s);
        const saved = keep.saved orelse return error.NothingKept;
        defer x.db.dropState(saved);
        for ([_]?lanes.Sampling{ null, sp }, 0..) |sampling, i| {
            try wipe(&x.rig.m, x.rig.s);
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(gpa);
            const cached = try resumed(gpu, &x.db, p, sampling, saved, at, x.count, &got);
            const eq = cached == at and std.mem.eql(u32, got.items, fresh.out[i]);
            if (!eq) bad += 1;
            std.debug.print("RESULT dory L={d} kept state at {d} ({d} prompt tokens, {s}): {d} cached, {x} resumed vs {x} fresh {s}\n", .{ x.loops, at, p.len, if (sampling == null) "greedy" else "sampled", cached, serve.hashOf(got.items), serve.hashOf(fresh.out[i]), if (eq) "same" else "DIFFER" });
        }
        std.debug.print("RESULT dory L={d} state at {d}: {d:.1} MiB kept\n", .{ x.loops, at, @as(f64, @floatFromInt(snap.bytes(&x.rig.m, at))) / (1 << 20) });
    }
    try check.expect(bad == 0, "dory L={d} states: {d} resumed replies differ from fresh", .{ x.loops, bad });
    check.pass("dory L={d} states: resumed == fresh at 8000, 8192 and 32001 into wiped caches (greedy and sampled, {d} tokens)", .{ x.loops, x.count });
}
