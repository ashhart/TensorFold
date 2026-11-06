//! DeepSeek-V4.1's served lanes over a backend `Model`: costs, draft trees, the lane engine, LaneHost and the recorder.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const ln = @import("draft/lanes.zig");
const drive = @import("draft/drive.zig");
const costs_mod = @import("draft/costs.zig");
const tree = @import("draft/tree.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_lanes = @import("calib_lanes.zig");
const dspark = @import("draft/dspark.zig");
const spec = @import("draft/spec.zig");
const lookup = @import("draft/lookup.zig");
const ph = @import("phases.zig");
const timed = @import("phases_timed.zig");
const Allocator = std.mem.Allocator;

/// A served backend as the lanes see it: target, pass, calibration and the host's hooks, each null when absent.
pub const Model = struct {
    ptr: *anyopaque,
    /// LaneHost's name ("dsv41-cuda", "dsv41-metal")
    name: []const u8,
    slots: u32,
    /// the context limit the model was opened with (a request's `context` is clamped to it)
    limit: u64,
    /// tensor-parallel ranks (1: none)
    world: u32,
    prefill_step: u32,
    /// the widest forward the depth policy plans (row windows' cap)
    forward_rows: u32,
    /// DSpark's shape at this world, as the backend's model() declares it
    shape: dspark.Shape,
    /// the verify windows (iface.Target); backend-owned
    target: *const fn (ptr: *anyopaque) @import("draft/iface.zig").Target,
    /// DSpark's pass (null: serial windows through the same lanes)
    pass: *const fn (ptr: *anyopaque) ?@import("draft/iface.zig").Pass,
    /// TF_DSV41_CALIB=measure / zig: the draft-cost table, measured at boot, and its stored file's path (or null)
    calibrate: *const fn (ptr: *anyopaque, gpa: Allocator, io: std.Io, dir: []const u8, knobs: calib_env.Knobs, want: calib_env.Shape) anyerror!calib_lanes.Boot,
    /// TF_DSV41_PHASES (phases.zig): the round's phases around the target and the pass; null: none
    sync: ?ph.Sync = null,
    /// the phases report's family counters (phases.zig Extra); null: none
    extra: ?ph.Extra = null,
    /// the lanes run on LaneHost's thread: a backend whose state is per thread binds it there
    on_thread: ?api.LaneHost.ThreadInit = null,
    /// /metrics' device memory; null: Engine.memory reports none
    memory: ?api.MemorySource = null,
    health: ?api.HealthSource = null,
    /// TF_DSV41_PREFILL_PIECES=1: the batcher plans the host's rounds
    rounds: ?*const fn (ptr: *anyopaque, gpa: Allocator, io: std.Io, host: *api.LaneHost) anyerror!void = null,
    /// `close`, before the backend's own teardown (the sched_host's settle); null: nothing
    settle: ?*const fn (ptr: *anyopaque) void = null,
    /// the followers' loops end (TP); null: nothing
    stop: ?*const fn (ptr: *anyopaque) void = null,
    /// TF_DSV41_PHASES: the forward's timers point at the served Phases (null clears them)
    set_phases: ?*const fn (ptr: *anyopaque, p: ?*ph.Phases) void = null,
    close: *const fn (ptr: *anyopaque) void,
    /// image input (the lanes' Info.vision); null: text only
    vision: ?api.Vision = null,
    /// the calibration knobs (null: TF_DSV41_CALIB / _DIR / _FILE and HOME, as read at open)
    calib: ?calib_env.Knobs = null,
};

pub const Served = struct {
    gpa: Allocator,
    io: std.Io,
    backend: Model,
    x: *ln.Lanes,
    nd: drive.NoDraft = .{},
    costs: costs_mod.Costs,
    arena: std.heap.ArenaAllocator,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    rec: Recorder,
    /// TF_DSV41_PHASES (phases.zig): the round's phases around the target and the pass
    phases: ?*ph.Phases = null,
    tt: timed.TimedTarget = undefined,
    tpass: timed.TimedPass = undefined,
    /// TF_DSV41_LOOKUP_PLAN=1 (draft/lookup.zig): priced lookup drafts for each request's copies
    lookup: ?*lookup.Planner = null,

    /// The engine over a backend's model (its DSpark pass, else serial windows through the same lanes).
    pub fn open(gpa: Allocator, io: std.Io, backend: Model, dir: []const u8, context: ?i64) !*Served {
        const s = try gpa.create(Served);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        s.io = io;
        s.backend = backend;
        s.nd = .{};
        const dspark_on = backend.pass(backend.ptr) != null;
        const limit: i64 = @intCast(backend.limit);
        const window: u64 = @intCast(@min(context orelse limit, limit));
        // draft pricing: the cached calibration and depth knobs (TF_DSV41_CALIB=measure: timed now, stored)
        const knobs = backend.calib orelse calib_env.Knobs.fromEnv();
        const want: calib_env.Shape = .{ .slots = backend.slots, .world = backend.world, .dspark = dspark_on, .context = window };
        var cal = try calib_env.load(gpa, io, knobs, want, 64);
        defer if (cal.path) |p| gpa.free(p);
        if (cal.source == .measure) {
            errdefer cal.costs.deinit(gpa);
            const got = try backend.calibrate(backend.ptr, gpa, io, dir, knobs, want);
            cal.costs.deinit(gpa);
            cal.costs = got.costs;
            cal.source = .measured;
            if (cal.path) |p| gpa.free(p);
            cal.path = got.path;
        }
        s.costs = cal.costs;
        std.log.info("dsv41 draft costs: {t}{s}{s}, verify 1 / 16 rows {d:.3} / {d:.3} ms, pass {d:.3} ms", .{ cal.source, if (cal.path != null) " " else "", cal.path orelse "", s.costs.windowMs(1), s.costs.windowMs(16), s.costs.draft });
        var pass = backend.pass(backend.ptr) orelse s.nd.pass();
        var tgt = backend.target(backend.ptr);
        s.phases = try ph.Phases.fromEnv(gpa);
        if (s.phases) |x| {
            if (backend.sync) |sync| x.sync = sync;
            if (backend.extra) |extra| x.extra = extra;
            if (backend.set_phases) |set| set(backend.ptr, x);
            s.tt = .{ .inner = tgt, .p = x, .gpa = gpa, .io = io };
            s.tpass = .{ .inner = pass, .p = x };
            tgt = s.tt.target();
            pass = s.tpass.pass();
        }
        // draft trees (TF_DSV41_TREE / _DUP_MS / _ROWS; off by default): first-position siblings
        const trees = if (dspark_on) try tree.Settings.fromEnv() else tree.Settings{};
        s.x = try ln.Lanes.init(gpa, tgt, pass, s.costs, .{ .shape = backend.shape, .slots = backend.slots, .siblings = trees.siblings, .sib_rows = trees.sib_rows, .dup = trees.dup, .depth = try calib_env.depthSettings(&calib_env.env), .spec = if (dspark_on) try spec.Settings.fromEnv() else .{} });
        s.arena = std.heap.ArenaAllocator.init(gpa);
        s.cfg = try lanes.Config.init(gpa, try s.x.model(s.arena.allocator(), 64), 16, 15);
        s.x.policy.depth.forward_rows = backend.forward_rows;
        // drafts sized from cost tables, not the wall clock, so HTTP == CLI replay (TF_DSV41_DRAFT_TIMED=1: measured)
        s.cfg.timed = if (std.c.getenv("TF_DSV41_DRAFT_TIMED")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        // the depth policy as booted (the reference's "DSpark on (depth ...)"): what a check reads from the log
        const dset = s.x.policy.depth.set;
        std.log.info("dsv41 depth: {t}, cap {d}, skip {s}, joint {d}, drafts priced {s}", .{ dset.mode, dset.cap, if (dset.skip) "on" else "off", dset.joint, if (s.cfg.timed) "by the wall clock" else "from the cost tables" });
        s.clock = .{ .io = io };
        s.core = lanes.Engine.init(gpa, &s.cfg, s.x.backend(), s.clock.clock());
        s.x.attach(&s.core);
        s.host = api.LaneHost.init(gpa, io, &s.core, .{ .name = backend.name, .lanes = backend.slots, .context_window = @intCast(window), .prefill_step = backend.prefill_step, .vision = backend.vision });
        s.host.info_.draft_sources = draftSources(dspark_on, s.cfg.max_copy > 0);
        // the lanes run on LaneHost's thread: a backend whose state is per thread binds it there
        s.host.on_thread = backend.on_thread;
        // /metrics' device memory: bytes this rank holds (weights, KV pool, staging), peak
        s.host.memory = backend.memory;
        s.host.health = backend.health;
        // TF_DSV41_PREFILL_PIECES=1: the batcher plans the rounds; served shape only, same setting on every rank
        if (backend.rounds) |rounds| try rounds(backend.ptr, gpa, io, &s.host);
        s.lookup = null;
        const ls = try lookup.Settings.fromEnv();
        if (ls.plan) {
            const pl = try gpa.create(lookup.Planner);
            pl.* = lookup.Planner.init(ls, &s.x.policy.depth.costs, dspark_on);
            s.lookup = pl;
            s.host.proposers = .{ .ctx = pl, .make = makeLookup, .free = freeLookup };
            std.log.info("dsv41 lookup: Python's priced plan (min match {d}, up to {d} rows{s})", .{ ls.min_match, ls.max_rows, if (ls.lookup) "" else ", lookup off" });
        }
        try s.host.start();
        s.rec = Recorder.init(gpa, io, s.host.engine());
        return s;
    }

    pub fn engine(s: *Served) api.Engine {
        return s.rec.engine();
    }

    fn makeLookup(ctx: *anyopaque, gpa: Allocator, prompt: []const u32, eos: []const u32) anyerror!lanes.proposer.Proposer {
        const pl: *const lookup.Planner = @ptrCast(@alignCast(ctx));
        const x = try gpa.create(lookup.Proposer);
        errdefer gpa.destroy(x);
        x.* = try lookup.Proposer.init(pl, gpa, prompt, eos);
        return x.proposer();
    }

    fn freeLookup(_: *anyopaque, p: lanes.proposer.Proposer) void {
        const x: *lookup.Proposer = @ptrCast(@alignCast(p.ptr));
        const gpa = x.req.index.gpa;
        x.deinit();
        gpa.destroy(x);
    }

    /// A drain's deadline: rounds end at a boundary first, so `close` can stop the followers cleanly (exit 0, not 70).
    pub fn halt(ctx: *anyopaque, reason: []const u8) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        s.host.halt(reason);
    }

    pub fn close(ctx: *anyopaque) void {
        const s: *Served = @ptrCast(@alignCast(ctx));
        const gpa = s.gpa;
        s.host.stop();
        if (s.lookup) |pl| gpa.destroy(pl);
        if (s.backend.settle) |settle| settle(s.backend.ptr);
        if (s.backend.stop) |stop| stop(s.backend.ptr);
        if (s.phases) |x| {
            x.flush(gpa, s.io);
            if (s.backend.set_phases) |set| set(s.backend.ptr, null);
            gpa.destroy(x);
        }
        s.rec.deinit();
        s.core.deinit();
        s.cfg.deinit(gpa);
        s.arena.deinit();
        s.x.reportSpec();
        s.x.deinit();
        s.costs.deinit(gpa);
        s.backend.close(s.backend.ptr);
        gpa.destroy(s);
    }
};

/// The engine as served, with each finished reply's prompt and reply ids appended to TF_DSV41_TRACE_TOKENS (when set).
pub const Recorder = struct {
    gpa: Allocator,
    io: std.Io,
    inner: api.Engine,
    path: ?[]const u8,
    mutex: std.Io.Mutex = .init,
    live: std.AutoHashMapUnmanaged(api.Id, *Rec) = .empty,
    /// every recorded line so far (the file is rewritten whole: one reply at a time)
    log: std.ArrayList(u8) = .empty,

    /// `sampling`: the request's own (seed included), so `tf-dsv41 generate` replays a sampled reply as it was asked
    const Rec = struct { r: *Recorder, sink: api.Sink, prompt: []u32, sampling: ?api.Sampling = null, out: std.ArrayList(u32) = .empty, drafted: u64 = 0, structure: ?Structure = null };
    /// a structured request's grammar (TF_DSV41_GRAMMAR), so `generate` replays it under the same grammar
    const Structure = struct { kind: []const u8, text: []const u8 };

    pub fn init(gpa: Allocator, io: std.Io, inner: api.Engine) Recorder {
        return .{ .gpa = gpa, .io = io, .inner = inner, .path = if (std.c.getenv("TF_DSV41_TRACE_TOKENS")) |p| std.mem.span(p) else null };
    }

    pub fn deinit(r: *Recorder) void {
        var it = r.live.valueIterator();
        while (it.next()) |x| r.drop(x.*);
        r.live.deinit(r.gpa);
        r.log.deinit(r.gpa);
    }

    fn drop(r: *Recorder, x: *Rec) void {
        r.gpa.free(x.prompt);
        if (x.structure) |st| r.gpa.free(st.text);
        x.out.deinit(r.gpa);
        r.gpa.destroy(x);
    }

    pub fn engine(r: *Recorder) api.Engine {
        if (r.path == null) return r.inner;
        return .{ .ctx = r, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn self(ctx: *anyopaque) *Recorder {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(ctx: *anyopaque) api.Info {
        return self(ctx).inner.info();
    }
    fn cancel(ctx: *anyopaque, id: api.Id) void {
        self(ctx).inner.cancel(id);
    }
    fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
        self(ctx).inner.status(out, stream_tokens);
    }
    fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
        return self(ctx).inner.memory(reset_peak);
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const r = self(ctx);
        const x = r.gpa.create(Rec) catch return error.Busy;
        x.* = .{ .r = r, .sink = sink, .sampling = request.sampling, .prompt = r.gpa.dupe(u32, request.prompt) catch {
            r.gpa.destroy(x);
            return error.Busy;
        } };
        if (request.structure) |st| x.structure = .{ .kind = @tagName(st.kind), .text = r.gpa.dupe(u8, st.text) catch {
            r.drop(x);
            return error.Busy;
        } };
        {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            r.live.put(r.gpa, id, x) catch {
                r.drop(x);
                return error.Busy;
            };
        }
        return r.inner.submit(id, request, .{ .ctx = x, .event = event });
    }

    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const x: *Rec = @ptrCast(@alignCast(ctx));
        const r = x.r;
        switch (ev.*) {
            .tokens => |t| x.out.appendSlice(r.gpa, t) catch {},
            .finished => |fin| {
                x.drafted = fin.stats.drafted;
                r.write(x);
            },
            else => {},
        }
        x.sink.event(x.sink.ctx, id, ev);
        if (ev.* == .finished) {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            _ = r.live.remove(id);
            r.drop(x);
        }
    }

    fn write(r: *Recorder, x: *const Rec) void {
        var line: std.Io.Writer.Allocating = .init(r.gpa);
        defer line.deinit();
        std.json.Stringify.value(.{ .prompt = x.prompt, .tokens = x.out.items, .drafted = x.drafted, .sampling = x.sampling, .structure = x.structure }, .{}, &line.writer) catch return;
        line.writer.writeByte('\n') catch return;
        r.log.appendSlice(r.gpa, line.written()) catch return;
        std.Io.Dir.cwd().writeFile(r.io, .{ .sub_path = r.path.?, .data = r.log.items }) catch {};
    }
};

fn draftSources(drafter: bool, copies: bool) api.DraftSources {
    return .{ .copies = copies, .drafter = if (drafter) .{ .kind = "DSpark" } else null };
}

test "draft sources: DeepSeek names the loaded DSpark pass and never calls NoDraft an MTP head" {
    const t = std.testing;
    const active = draftSources(true, true);
    try t.expect(active.copies and !active.mtp);
    try t.expectEqualStrings("DSpark", active.drafter.?.kind);
    const plain = draftSources(false, true);
    try t.expect(plain.copies and !plain.mtp and plain.drafter == null);
}
