//! The served engine's round planner (TF_DSV41_PREFILL_PIECES=1, rank 0): queue, prefill pieces and page release.

const std = @import("std");
const api = @import("engine_api");
const kv = @import("kv");
const target = @import("target.zig");
const sg = @import("sessions_gpu.zig");

const sched = kv.sched;
const Hit = kv.sessions.store.Hit;

pub const SchedHost = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    p: sched.Planner,
    gt: *target.GpuTarget,
    round: sched.Round = .{},
    /// the request `admit` asks about (the store's callbacks read it)
    cur: api.Rounds.Ask = undefined,
    /// admitted this round: their admissions until their streams begin
    admits: std.AutoHashMapUnmanaged(usize, sched.Admission) = .empty,
    /// the round's spills ride with the first admission that begins
    spills_due: bool = false,
    pieces: std.ArrayList(sched.Piece) = .empty,
    finals: std.ArrayList(usize) = .empty,
    /// the log lines' clocks: queued requests' waits, long prompts' prefills
    waits: kv.progress.Waits = .{},
    prefills: kv.progress.Prefills = .{},
    /// the pages the last pool decision wanted (an admission wait's line)
    want: u32 = 0,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, gt: *target.GpuTarget, s: sched.Settings) SchedHost {
        return .{ .gpa = gpa, .io = io, .p = sched.Planner.init(gpa, s), .gt = gt };
    }

    pub fn deinit(h: *SchedHost) void {
        h.p.deinit();
        h.round.deinit(h.gpa);
        h.admits.deinit(h.gpa);
        h.pieces.deinit(h.gpa);
        h.finals.deinit(h.gpa);
        h.waits.deinit(h.gpa);
        h.prefills.deinit(h.gpa);
    }

    /// The batcher's settings; the rows a round are the forward's prefill segment.
    pub fn settings(gt: *const target.GpuTarget) !sched.Settings {
        var s = try sched.Settings.fromEnv();
        s.rows = @intCast(@import("forward_prefill.zig").segmentRows(gt.f));
        // TF_DSV41_PF_TBO: a round's piece holds a pair of segments (forward_prefill.prompt runs them on two streams)
        if (gt.f.opts.pf_tbo) s.rows *= 2;
        s.sessions = gt.sessions != null;
        if (flag("TF_DSV41_PIECES_FLOOR")) try s.withFloorFromEnv();
        return s;
    }

    pub fn rounds(h: *SchedHost) api.Rounds {
        return .{ .ctx = h, .vtable = &.{ .begin_admit = beginAdmit, .admit = admit, .admitted = admitted, .arm = arm, .began = began, .plan = plan, .left = left, .after = after } };
    }

    pub fn describe(h: *const SchedHost, buf: []u8) []const u8 {
        const s = h.p.s;
        return std.fmt.bufPrint(buf, "prompts in pieces of {d} rows between decode rounds (short {d}, share {d}, long > {d} tokens {d} at a time{s})", .{ s.rows, s.short, s.share, s.long_prompt, s.concurrent, if (s.floor != null) ", memory floor and adaptive rows" else "" }) catch "prompts in pieces";
    }

    fn self(ctx: *anyopaque) *SchedHost {
        return @ptrCast(@alignCast(ctx));
    }

    fn now(h: *const SchedHost) f64 {
        return @as(f64, @floatFromInt(std.Io.Clock.awake.now(h.io).toNanoseconds())) / 1e9;
    }

    fn memory(h: *const SchedHost) ?sched.Meminfo {
        if (h.p.s.floor == null) return null;
        return readMeminfo(h.io) catch null;
    }

    fn beginAdmit(ctx: *anyopaque) void {
        const h = self(ctx);
        h.round.deinit(h.gpa);
        h.round = h.p.beginAdmit();
    }

    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, words: *[]const u8) api.Rounds.Verdict {
        const h = self(ctx);
        h.cur = ask;
        const store: ?sched.Store = if (h.gt.sessions != null) .{ .ctx = h, .vtable = &.{ .find = findFn, .pool = poolFn } } else null;
        const before = h.p.counts;
        const v = h.p.admit(&h.round, .{ .key = ask.key, .n = ask.prompt.len, .max_new = ask.max_tokens, .submitted = ask.submitted_s }, store, h.now(), h.memory()) catch |e| {
            words.* = @errorName(e);
            return .refuse;
        };
        h.logWait(ask, v, before);
        switch (v) {
            .admit => |a| {
                h.admits.put(h.gpa, ask.key, a) catch {
                    h.p.ended(ask.key);
                    words.* = "out of memory";
                    return .refuse;
                };
                return .admit;
            },
            .wait => return .wait,
            .skip => return .skip,
            .refuse => |r| {
                words.* = switch (r) {
                    .too_large => "the request needs more KV pool pages than the pool has",
                    .memory => "the prompt's prefill cannot fit in this server's memory now (TF_DSV41_FLOOR_HARD_GIB)",
                };
                return .refuse;
            },
        }
    }

    /// An admission wait's line (first, then every 30 s) and the line when a request that waited is admitted.
    fn logWait(h: *SchedHost, ask: api.Rounds.Ask, v: sched.Verdict, before: sched.Counts) void {
        const log = std.log.scoped(.dsv41);
        const t = h.now();
        const c = h.p.counts;
        switch (v) {
            .wait, .skip => {
                const waited = (h.waits.waiting(h.gpa, ask.key, t) catch return) orelse return;
                if (c.pool_waits > before.pool_waits) {
                    const free = if (h.gt.sessions) |ss| ss.store.pool.available() else 0;
                    log.info("admission: request {x} ({d} prompt tokens) has waited {d:.0} s for the KV pool: {d} pages wanted, {d} free, running requests hold the rest", .{ ask.key, ask.prompt.len, waited, h.want, free });
                } else if (c.mem_waits > before.mem_waits) {
                    log.info("admission: request {x} ({d} prompt tokens) has waited {d:.0} s for the memory floor (TF_DSV41_FLOOR_HARD_GIB)", .{ ask.key, ask.prompt.len, waited });
                } else log.info("admission: request {x} ({d} prompt tokens) has waited {d:.0} s behind a long prompt prefilling (TF_DSV41_PREFILL_CONCURRENT)", .{ ask.key, ask.prompt.len, waited });
            },
            .admit => if (h.waits.done(ask.key, t)) |waited| log.info("admission: request {x} admitted after {d:.1} s", .{ ask.key, waited }),
            .refuse => _ = h.waits.done(ask.key, t),
        }
    }

    /// Long prompts' prefill lines: every 30 s while their pieces run, and once when they are in.
    fn logPrefill(h: *SchedHost) void {
        const log = std.log.scoped(.dsv41);
        const t = h.now();
        for (h.pieces.items) |x| {
            const q = h.p.seq(x.key) orelse continue;
            const l = (h.prefills.at(h.gpa, x.key, q.n, x.end, t) catch continue) orelse continue;
            log.info("prefill: request {x} {d} of {d} tokens ({d:.0} %), {d:.0} tok/s over the last 30 s, {d:.0} s in", .{ x.key, l.done, l.n, 100 * @as(f64, @floatFromInt(l.done)) / @as(f64, @floatFromInt(l.n)), l.rate, l.seconds });
        }
        for (h.finals.items) |key| {
            const q = h.p.seq(key) orelse continue;
            if (h.prefills.end(key, q.n - 1, t)) |l| log.info("prefill: request {x} in: {d} tokens in {d:.1} s ({d:.0} tok/s)", .{ key, l.n, l.seconds, l.rate });
        }
    }

    fn findFn(ctx: *anyopaque, key: usize) anyerror!?sched.Store.Hit {
        const h = self(ctx);
        std.debug.assert(key == h.cur.key);
        const x = (try h.gt.sessions.?.find(h.cur.prompt)) orelse return null;
        return .{ .id = x.id, .pos = x.pos, .ram = x.ram };
    }

    fn poolFn(ctx: *anyopaque, key: usize, hit: ?sched.Store.Hit, held: []const u32, extra: u32, spills: *std.ArrayList(u32)) anyerror!kv.sess.Plan {
        const h = self(ctx);
        std.debug.assert(key == h.cur.key);
        const x: ?Hit = if (hit) |y| .{ .id = y.id, .pos = y.pos, .ram = y.ram } else null;
        const pp = try h.gt.sessions.?.poolPlan(h.cur.prompt.len, h.cur.max_tokens, x, held, extra, spills);
        h.want = pp.want;
        return pp;
    }

    /// The pass is decided: last round's released slots go back, then spills ride with the first admission.
    fn admitted(ctx: *anyopaque) anyerror!void {
        const h = self(ctx);
        h.gt.settleReleases();
        h.spills_due = true;
    }

    fn arm(ctx: *anyopaque, key: usize) void {
        const h = self(ctx);
        const a = h.admits.get(key) orelse return;
        const hit: ?Hit = if (a.hit) |y| .{ .id = y.id, .pos = y.pos, .ram = y.ram } else null;
        h.gt.admission = .{ .need = a.need, .hit = hit, .spills = if (h.spills_due) h.round.spills.items else &.{} };
        h.spills_due = false;
    }

    fn began(ctx: *anyopaque, key: usize, damaged: bool) void {
        const h = self(ctx);
        _ = h.admits.remove(key);
        if (damaged) h.p.damaged(key);
    }

    fn plan(ctx: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        const h = self(ctx);
        h.admits.clearRetainingCapacity(); // admitted streams that never began left through `left`
        try h.p.plan(h.memory(), &h.pieces, &h.finals);
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
        for (h.pieces.items) |x| try pieces.append(h.gpa, .{ .key = x.key, .start = x.start, .end = x.end, .save = x.save, .run = x.run });
        try finals.appendSlice(h.gpa, h.finals.items);
        h.logPrefill();
    }

    fn left(ctx: *anyopaque, key: usize) void {
        const h = self(ctx);
        _ = h.admits.remove(key);
        _ = h.waits.done(key, 0);
        _ = h.prefills.end(key, 0, 0);
        h.p.ended(key);
    }

    fn after(ctx: *anyopaque, piece_s: f64, decode_s: f64) void {
        self(ctx).p.after(piece_s, decode_s);
    }
};

fn flag(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return false;
    return std.mem.eql(u8, std.mem.trim(u8, std.mem.span(v), " "), "1");
}

/// /proc/meminfo's lines the floor reads, in bytes.
fn readMeminfo(io: std.Io) !sched.Meminfo {
    var buf: [8192]u8 = undefined;
    const text = try std.Io.Dir.cwd().readFile(io, "/proc/meminfo", &buf);
    var mi: sched.Meminfo = .{ .free = 0, .available = 0 };
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const key = line[0..colon];
        var f = std.mem.tokenizeScalar(u8, line[colon + 1 ..], ' ');
        const kib = std.fmt.parseInt(i64, f.next() orelse continue, 10) catch continue;
        const b = kib * 1024;
        if (std.mem.eql(u8, key, "MemFree")) mi.free = b else if (std.mem.eql(u8, key, "MemAvailable")) mi.available = b else if (std.mem.eql(u8, key, "Dirty")) mi.dirty = b else if (std.mem.eql(u8, key, "Writeback")) mi.writeback = b else if (std.mem.eql(u8, key, "Mapped")) mi.mapped = b;
    }
    return mi;
}
