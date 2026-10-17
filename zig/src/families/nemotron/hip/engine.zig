//! Nemotron-H on one HIP device: the kernels, a stream, the rank's weights, the KV pool and the scratch.
const std = @import("std");
const hip = @import("hip");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const model = @import("model.zig");
const loader = @import("load.zig");
const pages = @import("pages.zig");
const plan = @import("plan.zig");
const admit = @import("admit.zig");
const forward = @import("forward.zig");
const prefix = @import("prefix.zig");
const rounds = @import("round.zig");
const graphs = @import("graphs.zig");

/// Prompt rows a forward takes at most: the routed experts see enough rows a tile, within the scratch admit counts.
pub const prompt_rows: usize = 2048;

pub const Options = struct {
    device: c_int = 0,
    rank: usize = 0,
    world: usize = 1,
    group: ?hip.collective.Group = null,
    /// The startup steps the ranks take together over the group's link (null: the engine's own).
    startup: ?*hip.Startup = null,
    policy: hip.Policy = .{},
    /// Positions a stream holds at most.
    capacity: usize = 4096,
    /// Prompt rows a forward takes at most.
    chunk: usize = prompt_rows,
    /// Streams the KV pool holds at the full capacity, and pages beside them (a prompt cache's).
    streams: usize = 1,
    extra_pages: usize = 0,
    /// Rank 0 loads the MTP head when its file is there.
    mtp: bool = true,
    /// The KV pool is made at `open` from `streams` and `extra_pages`; false leaves it to `sizePool` (after agreeing).
    pool: bool = true,
};

pub const Engine = struct {
    pub const Cancel = struct { ctx: *anyopaque, check: *const fn (*anyopaque) bool };

    pub const Poll = struct {
        stream: hip.Stream,
        cancel: Cancel,

        fn layer(ctx: *anyopaque, _: usize, _: hip.ops.Tensor, _: usize) anyerror!void {
            const p: *Poll = @ptrCast(@alignCast(ctx));
            try p.stream.synchronize();
            if (p.cancel.check(p.cancel.ctx)) return error.Cancelled;
        }
    };

    /// The prompt's layer trace only where cancellation cannot strand another rank in a collective.
    pub fn poll(e: anytype, cancel: ?Cancel, rows: usize, p: *Poll) ?forward.Trace {
        const c = cancel orelse return null;
        if (!prefix.pollsLayers(e.o.world, rows)) return null;
        p.* = .{ .stream = e.stream, .cancel = c };
        return .{ .ctx = p, .layer = Poll.layer };
    }

    gpa: std.mem.Allocator,
    io: std.Io,
    driver: hip.Runtime,
    ctx: hip.Context,
    lib: hip.Launcher,
    stream: hip.Stream,
    o: Options,
    cfg: config.Config,
    weights: loader.Weights,
    act: model.Kind,
    comm: ?hip.collective.Comm = null,
    arena: hip.Arena,
    pool: pages.Pool,
    /// A lane round's plan and its windows, sized with the pool for `o.streams` streams of `rounds.max_window` rows.
    plan: rounds.Plan,
    wins: []rounds.Window,
    sized: bool,
    /// Each block's place among its kind; the MTP head's attention takes the pool slot after the model's.
    place: [config.max_layers]usize,
    mtp_slot: usize,
    /// Rounds replayed from captured graphs, unless the policy says `graphs=off`.
    graphs: graphs.Graphs,
    replay: bool,
    /// Every rank's vote on a captured round, in buffers made with the engine (a group only).
    votes: ?hip.Votes,
    /// A rank could not take a round: every rank stops before another collective.
    stopped: bool,

    /// `load`, this rank's failure heard by the other ranks at the step they wait at.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        var own: hip.Startup = .{ .link = if (o.world > 1) (o.group orelse return error.NoGroup).link() else null };
        var with = o;
        with.startup = o.startup orelse &own;
        return load(gpa, io, dir, with) catch |err| {
            with.startup.?.failed(err);
            return err;
        };
    }

    /// The checkpoint `dir` (admitted before) on `o.device`, this rank's share loaded; the caller refuses a failure.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options) !*Engine {
        var own: hip.Startup = .{ .link = if (o.world > 1) (o.group orelse return error.NoGroup).link() else null };
        const steps = o.startup orelse &own;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.o = o;
        e.stopped = false;
        e.driver = try hip.Runtime.open();
        errdefer e.driver.close();
        e.ctx = try hip.Context.init(&e.driver, o.device);
        errdefer e.ctx.deinit();
        const caps = try hip.Device.kernelCaps(&e.driver, o.device);
        e.lib = try hip.Launcher.load(&e.driver, o.policy.choices(), hip.kernels.images);
        errdefer e.lib.unload();
        e.act = if (caps.act == .f16) .f16 else .bf16;
        e.stream = try hip.Stream.init(&e.driver);
        errdefer e.stream.deinit();
        e.comm = null;
        // RCCL's init waits for every rank: none enters it unless every rank got this far
        _ = try steps.done(.{ 0, 0 }, .first);
        if (o.world > 1) e.comm = try (o.group orelse return error.NoGroup).open();
        errdefer if (e.comm) |c| c.close();
        _ = try steps.done(.{ 0, 0 }, .first);
        e.cfg = try config.Config.read(gpa, io, dir);
        const mtp = o.mtp and o.rank == 0 and hasFile(gpa, dir, tensors.mtp_file);
        e.weights = try loader.load(
            gpa,
            io,
            &e.driver,
            dir,
            &e.cfg,
            .{ .world = o.world, .rank = o.rank, .mtp = mtp, .act = e.act },
        );
        errdefer e.weights.deinit();
        e.weights.model.tp = e.comm;
        e.arena = try hip.Arena.init(&e.driver, admit.scratch_bytes);
        errdefer e.arena.deinit();
        e.votes = if (o.world > 1) try hip.Votes.init(&e.driver, o.world) else null;
        errdefer if (e.votes) |*v| v.deinit();
        e.sized = false;
        e.graphs = .init(gpa);
        e.replay = o.policy.graphsOn();
        _ = forward.slots(&e.cfg, &e.place);
        e.mtp_slot = e.cfg.count(.attention);
        if (o.pool) try e.sizePool(o.capacity, o.streams, o.extra_pages);
        _ = try steps.done(.{ 0, 0 }, .first);
        return e;
    }

    /// The KV pool: `streams` streams of `capacity` positions and `extra_pages` beside them (a prompt cache's).
    pub fn sizePool(e: *Engine, capacity: usize, streams: usize, extra_pages: usize) !void {
        if (e.sized) return error.AlreadySized;
        const d = e.weights.model.dims;
        const mtp = e.weights.model.mtp != null;
        e.o.capacity = capacity;
        e.o.streams = streams;
        e.o.extra_pages = extra_pages;
        // the MTP head's slot holds every kv head: the head is whole on its rank, however the model's are cut
        const head_kv: ?usize = if (mtp) e.cfg.kv_heads else null;
        e.pool = try pages.Pool.init(
            e.gpa,
            &e.driver,
            plan.attentionLayers(&e.cfg, mtp),
            streams * pages.pagesFor(capacity) + extra_pages,
            d.kv_heads,
            d.head_dim,
            e.act,
            head_kv,
        );
        errdefer e.pool.deinit();
        e.plan = try rounds.Plan.init(&e.driver, &e.pool, streams * rounds.max_window, streams);
        errdefer e.plan.deinit();
        e.wins = try e.gpa.alloc(rounds.Window, streams);
        e.sized = true;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        e.graphs.deinit();
        if (e.sized) {
            e.gpa.free(e.wins);
            e.plan.deinit();
            e.pool.deinit();
        }
        if (e.votes) |*v| v.deinit();
        e.arena.deinit();
        e.weights.deinit();
        if (e.comm) |c| c.close();
        e.stream.deinit();
        e.lib.unload();
        e.ctx.deinit();
        e.driver.close();
        e.gpa.destroy(e);
    }

    pub fn model_(e: *const Engine) *const model.Model {
        return &e.weights.model;
    }

    /// The arena emptied for the next op.
    pub fn fresh(e: *Engine) !void {
        e.arena.reset();
    }

    pub fn ops(e: *Engine, arena: *hip.Arena) hip.ops.Ops {
        return .{ .l = &e.lib, .bf16 = e.act == .bf16, .stream = e.stream.handle, .arena = arena };
    }

    /// A new stream at the engine's capacity: its page table and zeroed Mamba states.
    pub fn openStream(e: *Engine) !pages.Stream {
        return pages.Stream.init(
            e.gpa,
            &e.driver,
            &e.pool,
            e.weights.model.dims,
            e.act,
            e.cfg.count(.mamba),
            e.o.capacity,
        );
    }

    pub fn closeStream(e: *Engine, s: *pages.Stream) void {
        e.stream.synchronize() catch {};
        s.deinit(&e.pool);
    }

    /// `tokens` as int32 ids in fresh arena memory, the copy queued on the stream.
    pub fn ids(e: *Engine, o: hip.ops.Ops, tokens: []const u32) !u64 {
        const dst = try o.arena.of(i32, tokens.len);
        try hip.runtime.check(e.driver.api.hipMemcpyHtoDAsync(
            @ptrFromInt(dst),
            @ptrCast(@constCast(tokens.ptr)),
            tokens.len * 4,
            o.stream,
        ));
        try e.stream.synchronize();
        return dst;
    }

    /// A prompt's `tokens` from `s.len` into `s`, kept: the final-normed rows of its last chunk (valid until the next).
    pub fn prompt(e: *Engine, s: *pages.Stream, tokens: []const u32) !hip.ops.Tensor {
        if (e.stopped) return error.GroupStopped;
        var out: hip.ops.Tensor = undefined;
        var at: usize = 0;
        while (at < tokens.len) {
            const n = @min(e.o.chunk, tokens.len - at);
            try s.cover(&e.pool, s.len + n);
            try e.stream.synchronize();
            try e.fresh();
            const o = e.ops(&e.arena);
            const dev = try e.ids(o, tokens[at..][0..n]);
            out = try forward.run(
                o,
                &e.weights.model,
                e.place[0..e.cfg.layers],
                .{ .pool = &e.pool, .stream = s },
                dev,
                n,
                s.len,
            );
            s.len += n;
            at += n;
        }
        return out;
    }

    /// The output head over `rows` final rows: whole logits rows (act dtype), joined across ranks in rank order.
    pub fn logits(e: *Engine, o: hip.ops.Ops, normed: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        const m = &e.weights.model;
        const y = try o.project(normed, m.head, rows, false);
        const c = m.tp orelse return y;
        const width = m.head.n;
        const size = y.kind.size();
        const parts = try o.arena.take(c.world * rows * width * size);
        try hip.reduce.gather(o, c, y, parts, rows * width);
        const out = try o.arena.take(c.world * rows * width * size);
        for (0..rows) |row| for (0..c.world) |r| {
            const to = out + (row * c.world + r) * width * size;
            const from = parts + (r * rows + row) * width * size;
            try e.copy(to, from, width * size, o.stream);
        };
        return .{ .ptr = out, .kind = y.kind };
    }

    /// The output head over a round's `rows` rows on the decode tile, as each row alone takes it, at any row count.
    pub fn roundLogits(e: *Engine, o: hip.ops.Ops, normed: hip.ops.Tensor, rows: usize) !hip.ops.Tensor {
        var w = o;
        w.window = true;
        return e.logits(w, normed, rows);
    }

    /// `n` bytes from device address `from` to device address `to`, queued on `stream`.
    pub fn copy(e: *Engine, to: u64, from: u64, n: usize, stream: hip.abi.Stream) !void {
        try hip.runtime.check(e.driver.api.hipMemcpyDtoDAsync(@ptrFromInt(to), @ptrFromInt(from), n, stream));
    }

    /// Every stream's window in one round from its kept length (the arena fresh): the final-normed rows in order.
    pub fn round(e: *Engine, rows: []const rounds.Rows, trace: ?forward.Trace) !hip.ops.Tensor {
        if (e.stopped) return error.GroupStopped;
        if (rows.len > e.wins.len) return error.WindowTooWide;
        for (rows, e.wins[0..rows.len]) |r, *w| w.* = .{ .s = r.s, .pos = r.s.len, .tokens = r.tokens };
        try e.fresh();
        const o = e.ops(&e.arena);
        const l = try e.plan.send(e.wins[0..rows.len], e.o.capacity, e.stream);
        const key = (if (trace == null and e.replay) graphs.keyOf(rows, l.span) else null) orelse
            return rounds.forwardRound(o, &e.weights.model, e.place[0..e.cfg.layers], rows, l, trace);
        e.graphs.cache.rounds += 1;
        if (e.graphs.lookup(key)) |entry| switch (entry.state) {
            .ready => {
                e.arena.used = entry.out.used;
                try entry.exec.?.launchOn(e.stream);
                e.graphs.cache.replayed += 1;
                return entry.out.hidden;
            },
            .failed => return rounds.forwardRound(o, &e.weights.model, e.place[0..e.cfg.layers], rows, l, null),
            .seen => {},
        };
        if (!e.graphs.again(key)) return rounds.forwardRound(
            o,
            &e.weights.model,
            e.place[0..e.cfg.layers],
            rows,
            l,
            null,
        );
        return e.capture(try e.graphs.cache.find(key), rows, l);
    }

    /// Records the round, then every rank votes: all kept replay it, one decline runs it eagerly, a failure stops all.
    fn capture(e: *Engine, entry: *graphs.Cache.Entry, rows: []const rounds.Rows, l: rounds.Launch) !hip.ops.Tensor {
        const m = &e.weights.model;
        const at = e.arena.mark();
        const began = nowNs();
        const body: Body = .{ .e = e, .rows = rows, .l = l };
        const got = hip.vote.record(
            graphs.Out,
            e.gpa,
            e.stream,
            &m.tp,
            e.o.policy.failsCapture(e.o.rank),
            body,
            Body.run,
        );
        e.graphs.cache.capture_ns += nowNs() - began;
        if (got.split) |split| e.graphs.cache.hold(entry, split, got.out);
        const all = if (e.votes) |*v| try v.agree(e.comm.?, got.vote, e.stream) else got.vote;
        if (all == .kept) {
            try entry.exec.?.launchOn(e.stream);
            return entry.out.hidden;
        }
        e.graphs.cache.revoke(entry);
        e.arena.release(at);
        // nothing ran while it recorded: the round runs now, eagerly, as it will from now on
        if (all == .declined) return rounds.forwardRound(e.ops(&e.arena), m, e.place[0..e.cfg.layers], rows, l, null);
        if (e.o.world == 1) return error.OutOfDeviceMemory;
        // no rank runs another collective once one could not take a round
        e.stopped = true;
        return error.GroupStopped;
    }
};

/// A round's forward as the recorder runs it: the final-normed rows and the arena's use after it.
const Body = struct {
    e: *Engine,
    rows: []const rounds.Rows,
    l: rounds.Launch,

    fn run(b: Body) anyerror!graphs.Out {
        const e = b.e;
        const hidden = try rounds.forwardRound(
            e.ops(&e.arena),
            &e.weights.model,
            e.place[0..e.cfg.layers],
            b.rows,
            b.l,
            null,
        );
        return .{ .hidden = hidden, .used = e.arena.used };
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Whether `dir`/`name` exists.
pub fn hasFile(gpa: std.mem.Allocator, dir: []const u8, name: []const u8) bool {
    const path = std.fs.path.joinZ(gpa, &.{ dir, name }) catch return false;
    defer gpa.free(path);
    return std.c.access(path, std.c.F_OK) == 0;
}
