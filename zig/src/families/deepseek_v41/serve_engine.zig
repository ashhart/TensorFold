//! DeepSeek-V4.1's engine for the native server: lanes over the GPU target with DSpark on rank 0; other ranks follow.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const api = @import("engine_api");
const lanes = @import("lanes");
const model = @import("cuda_model.zig");
const ln = @import("draft/lanes.zig");
const drive = @import("draft/drive.zig");
const costs_mod = @import("draft/costs.zig");
const tree = @import("draft/tree.zig");
const calib_env = @import("draft/calib_env.zig");
const calib_gpu = @import("calib_gpu.zig");
const spec = @import("draft/spec.zig");
const lookup = @import("draft/lookup.zig");
const gate_lanes = @import("gate_lanes.zig");
const ph = @import("phases.zig");
const timed = @import("phases_timed.zig");
const Allocator = std.mem.Allocator;
const sched_host = @import("sched_host.zig");

pub const backends: []const []const u8 = &.{"cuda"};
pub const families: []const api.Family = &.{.{ .model_type = "deepseek_v41", .formats = &.{"exl3"}, .raw_text = false }};

pub fn chip(_: Allocator) ?[]const u8 {
    return null;
}

/// The engine for `o.dir`, or null with `problem` set (the native_engines contract).
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (@import("sys").experimental.refusal(o.model_type, @import("sys").getenv("TF_EXPERIMENTAL"))) |text| {
        problem.* = text;
        return null;
    }
    if (!std.mem.eql(u8, o.model_type, "deepseek_v41")) {
        problem.* = try std.fmt.allocPrint(a, "this build serves deepseek_v41 checkpoints only, not {s}", .{o.model_type});
        return null;
    }
    const assets = std.mem.span(std.c.getenv("TF_DSV41_ASSETS") orelse {
        problem.* = "TF_DSV41_ASSETS: the dir with rope.bin, engram-host.bin and aot/ (model.zig)";
        return null;
    });
    const s = Served.open(gpa, io, o.dir, assets, o.drafts, o.context) catch |e| {
        problem.* = try std.fmt.allocPrint(a, "the DeepSeek-V4.1 CUDA engine cannot load {s} ({s})", .{ o.dir, @errorName(e) });
        return null;
    };
    return .{ .engine = s.engine(), .close = Served.close, .ctx = s, .halt = Served.halt };
}

pub const Served = struct {
    gpa: Allocator,
    io: std.Io,
    m: *model.Model,
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
    /// TF_DSV41_PREFILL_PIECES=1 (sched_host.zig): the batcher plans the host's rounds
    sched: ?*sched_host.SchedHost = null,
    /// TF_DSV41_LOOKUP_PLAN=1 (draft/lookup.zig): priced lookup drafts for each request's copies
    lookup: ?*lookup.Planner = null,

    /// Rank 0's engine over a loaded model (`drafts`: DSpark's pass, else serial windows through the same lanes).
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, assets: []const u8, drafts: bool, context: ?i64) !*Served {
        const m = try model.Model.open(gpa, io, dir, assets, .{ .drafts = drafts, .own_prefill = model.ownPrefillEnv(), .layers = try model.layersEnv(gpa) });
        errdefer m.close();
        if (m.rank() != 0) return error.NotRankZero; // the other ranks run `tf-dsv41 follow`
        const s = try gpa.create(Served);
        errdefer gpa.destroy(s);
        s.gpa = gpa;
        s.io = io;
        s.m = m;
        s.nd = .{};
        // draft pricing: the cached calibration and depth knobs (TF_DSV41_CALIB=measure: timed now, stored)
        const knobs = calib_env.Knobs.fromEnv();
        const want: calib_env.Shape = .{ .slots = m.slots, .world = m.f.comm.world(), .dspark = m.pass != null, .context = @intCast(@min(context orelse m.f.opts.limit, m.f.opts.limit)) };
        var cal = try calib_env.load(gpa, io, knobs, want, 64);
        defer if (cal.path) |p| gpa.free(p);
        if (cal.source == .measure) {
            errdefer cal.costs.deinit(gpa);
            const got = try calib_gpu.boot(gpa, io, m, dir, knobs, want);
            cal.costs.deinit(gpa);
            cal.costs = got.costs;
            cal.source = .measured;
            if (cal.path) |p| gpa.free(p);
            cal.path = got.path;
        }
        s.costs = cal.costs;
        std.log.info("dsv41 draft costs: {t}{s}{s}, verify 1 / 16 rows {d:.3} / {d:.3} ms, pass {d:.3} ms", .{ cal.source, if (cal.path != null) " " else "", cal.path orelse "", s.costs.windowMs(1), s.costs.windowMs(16), s.costs.draft });
        var pass = m.draftPass() orelse s.nd.pass();
        var tgt = m.gt.target();
        s.phases = try ph.Phases.fromEnv(gpa);
        if (s.phases) |x| {
            x.sync = .{ .ctx = m, .run = syncStream };
            m.f.phases = x;
            x.extra = .{ .ctx = m, .write = writeEngram };
            s.tt = .{ .inner = tgt, .p = x, .gpa = gpa, .io = io };
            s.tpass = .{ .inner = pass, .p = x };
            tgt = s.tt.target();
            pass = s.tpass.pass();
        }
        // draft trees (TF_DSV41_TREE / _DUP_MS / _ROWS; off by default): first-position siblings
        const trees = if (m.pass != null) try tree.Settings.fromEnv() else tree.Settings{};
        s.x = try ln.Lanes.init(gpa, tgt, pass, s.costs, .{ .shape = gate_lanes.shapeOf(&m.f), .slots = m.slots, .siblings = trees.siblings, .sib_rows = trees.sib_rows, .dup = trees.dup, .depth = try calib_env.depthSettings(&calib_env.env), .spec = if (m.pass != null) try spec.Settings.fromEnv() else .{} });
        s.arena = std.heap.ArenaAllocator.init(gpa);
        s.cfg = try lanes.Config.init(gpa, try s.x.model(s.arena.allocator(), 64), 16, 15);
        s.x.policy.depth.forward_rows = if (m.rows) |b| b.cap else 16;
        // drafts sized from cost tables, not the wall clock, so HTTP == CLI replay (TF_DSV41_DRAFT_TIMED=1: measured)
        s.cfg.timed = if (std.c.getenv("TF_DSV41_DRAFT_TIMED")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        // the depth policy as booted (the reference's "DSpark on (depth ...)"): what a check reads from the log
        const dset = s.x.policy.depth.set;
        std.log.info("dsv41 depth: {t}, cap {d}, skip {s}, joint {d}, drafts priced {s}", .{ dset.mode, dset.cap, if (dset.skip) "on" else "off", dset.joint, if (s.cfg.timed) "by the wall clock" else "from the cost tables" });
        s.clock = .{ .io = io };
        s.core = lanes.Engine.init(gpa, &s.cfg, s.x.backend(), s.clock.clock());
        s.x.attach(&s.core);
        const window: u32 = @intCast(@min(context orelse m.f.opts.limit, m.f.opts.limit));
        s.host = api.LaneHost.init(gpa, io, &s.core, .{ .name = "dsv41-cuda", .lanes = m.slots, .context_window = window, .prefill_step = m.gt.prefill_rows });
        // the lanes run on LaneHost's thread: the model's CUDA context must be current there
        s.host.on_thread = .{ .ctx = m, .run = bindContext };
        // /metrics' device memory: bytes this rank holds in cuda.DeviceBuffers (weights, KV pool, staging), peak
        s.host.memory = .{ .read = readMemory };
        s.host.health = .{ .ctx = &m.ctx, .read = contextHealthy };
        // TF_DSV41_PREFILL_PIECES=1: the batcher plans the rounds; served shape only, same setting on every rank
        s.sched = null;
        if (model.Model.piecesEnv()) {
            if (!(m.gt.own_prefill and m.gt.prompt_tail == .verify)) {
                if (!quiet) std.log.warn("TF_DSV41_PREFILL_PIECES=1 needs TF_DSV41_OWN_PREFILL=1 and TF_DSV41_PROMPT_TAIL=verify (the served shape); prompts run whole", .{});
            } else {
                const sh = try gpa.create(sched_host.SchedHost);
                sh.* = sched_host.SchedHost.init(gpa, io, &m.gt, try sched_host.SchedHost.settings(&m.gt));
                s.sched = sh;
                s.host.rounds = sh.rounds();
                m.gt.defer_release = true;
                var buf: [256]u8 = undefined;
                std.log.info("dsv41 rounds: {s}", .{sh.describe(&buf)});
            }
        } else std.log.info("dsv41 rounds: whole prompts at admission (TF_DSV41_PREFILL_PIECES=1: the reference server's pieces between decode rounds)", .{});
        s.lookup = null;
        const ls = try lookup.Settings.fromEnv();
        if (ls.plan) {
            const pl = try gpa.create(lookup.Planner);
            pl.* = lookup.Planner.init(ls, &s.x.policy.depth.costs, m.pass != null);
            s.lookup = pl;
            s.host.proposers = .{ .ctx = pl, .make = makeLookup, .free = freeLookup };
            std.log.info("dsv41 lookup: Python's priced plan (min match {d}, up to {d} rows{s})", .{ ls.min_match, ls.max_rows, if (ls.lookup) "" else ", lookup off" });
        }
        try s.host.start();
        s.rec = Recorder.init(gpa, io, s.host.engine());
        return s;
    }

    fn syncStream(ctx: *anyopaque) anyerror!void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        try m.stream.synchronize();
    }

    fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
        const u = @import("cuda").usage(reset_peak);
        return .{ .active = u.device, .cache = 0, .peak = u.peak };
    }

    fn contextHealthy(ctx: *anyopaque) bool {
        return @as(*@import("cuda").Context, @ptrCast(@alignCast(ctx))).healthy();
    }

    fn bindContext(ctx: *anyopaque) anyerror!void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        try m.ctx.makeCurrent();
    }

    pub fn engine(s: *Served) api.Engine {
        return s.rec.engine();
    }

    /// The phases report's "engram" and "rowgraphs" counters (cumulative): row reader records and the gate's rounds.
    fn writeEngram(ctx: *anyopaque, w: *std.Io.Writer) anyerror!void {
        const m: *model.Model = @ptrCast(@alignCast(ctx));
        try w.writeAll(", \"engram\": {");
        if (m.f.engram_rows) |r| {
            const st = r.stats;
            try w.print("\"hits\": {d}, \"waited\": {d}, \"misses\": {d}, \"issued\": {d}, \"reads\": {d}, \"bytes\": {d}, \"wait_ms\": {d:.2}", .{ st.hits, if (@hasField(@TypeOf(st), "waited")) st.waited else 0, st.misses, st.issued, st.reads, st.bytes, @as(f64, @floatFromInt(if (@hasField(@TypeOf(st), "wait_ns")) st.wait_ns else 0)) / 1e6 });
        } else try w.writeAll("\"rows\": null");
        if (@import("engram_gate_fwd.zig").of(&m.f)) |g| try w.print(", \"gate_rounds\": {d}, \"gate_syncs\": {d}, \"gate_warm\": {d}, \"gate_failed\": {d}", .{ g.stats.rounds, g.stats.syncs, g.stats.warm, g.stats.failed });
        try w.writeAll("}");
        // the row windows' first uses (batch.zig): programs emitted and graphs captured, their host ms
        if (m.gt.rows) |b| {
            const st = b.stats;
            try w.print(", \"rowgraphs\": {{\"windows\": {d}, \"rows\": {d}, \"padded\": {d}, \"builds\": {d}, \"build_ms\": {d:.2}, \"captures\": {d}, \"capture_ms\": {d:.2}, \"eager\": {d}, \"prep_ms\": {d:.2}, \"stage_ms\": {d:.2}, \"launch_ms\": {d:.2}, \"keep_ms\": {d:.2}}}", .{ st.windows, st.rows, st.padded, st.builds, @as(f64, @floatFromInt(st.build_ns)) / 1e6, st.captures, @as(f64, @floatFromInt(st.capture_ns)) / 1e6, st.eager, @as(f64, @floatFromInt(st.prep_ns)) / 1e6, @as(f64, @floatFromInt(st.stage_ns)) / 1e6, @as(f64, @floatFromInt(st.launch_ns)) / 1e6, @as(f64, @floatFromInt(st.keep_ns)) / 1e6 });
        }
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
        if (s.sched) |sh| {
            s.m.gt.settleReleases();
            sh.deinit();
            gpa.destroy(sh);
        }
        s.m.f.stop() catch {}; // the followers' loops end
        if (s.phases) |x| {
            x.flush(gpa, s.io);
            s.m.f.phases = null;
            gpa.destroy(x);
        }
        s.rec.deinit();
        s.core.deinit();
        s.cfg.deinit(gpa);
        s.arena.deinit();
        s.x.reportSpec();
        s.x.deinit();
        s.costs.deinit(gpa);
        s.m.close();
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
