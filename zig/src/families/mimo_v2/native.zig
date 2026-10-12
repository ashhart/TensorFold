//! MiMo-V2.6-Flash behind the native server: the lane backend over engine.zig, on one GPU or two TP ranks.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const tp = @import("tp");
const fwd = @import("forward.zig");
const engine = @import("engine.zig");
const prompt_mod = @import("prompt.zig");
const be = lanes.backend;
const Engine = engine.Engine;

pub const model_type = "mimo_v2";
pub const formats: []const []const u8 = &.{"exl3"};
pub const default_context: i64 = 131072;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fwd.DECODE_ROWS;

/// The server's settings: `lanes` (--parallel) is the streams' cache slots, `policy` (--policy) key=value pairs.
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, lanes: u32 = 4, policy: []const u8 = "" };
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

pub const Loaded = struct {
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *anyopaque, // no prompt cache (no snap_* decls): never read
    learns: ?*anyopaque = null, // no --slide
    learn_bytes: usize = 0,
    policy: []const u8 = "", // the active policy, for /health
};

/// --policy: `kv=fp16` (the default) or `kv=fp8` (the full layers' 8-bit key/value cache, kv8.zig).
pub const Policy = struct {
    kv8: bool = false,

    pub fn parse(text: []const u8) !Policy {
        var out: Policy = .{};
        var it = std.mem.tokenizeScalar(u8, text, ',');
        while (it.next()) |pair| {
            const eq = std.mem.indexOfScalar(u8, pair, '=') orelse return error.Policy;
            const key = std.mem.trim(u8, pair[0..eq], " ");
            const value = std.mem.trim(u8, pair[eq + 1 ..], " ");
            if (!std.mem.eql(u8, key, "kv")) return error.Policy;
            out.kv8 = if (std.mem.eql(u8, value, "fp8")) true else if (std.mem.eql(u8, value, "fp16")) false else return error.Policy;
        }
        return out;
    }

    pub fn line(p: Policy) []const u8 {
        return if (p.kv8) "kv=fp8" else "kv=fp16";
    }
};

/// The acceptance of DFlash drafts at depth 1, 2, ... given the ones before, until a stream has its own.
const draft_prior = [_]f64{ 0.789, 0.78, 0.801, 0.833, 0.833, 0.846, 0.825 };

/// A lone stream's forward of r rows against one row (r = 1 .. 8) and a draft pass, as the Python decoder prices them.
const depth_costs = [_]f64{ 1.00, 1.26, 1.50, 1.75, 2.01, 2.25, 2.46, 2.67 };
const draft_cost = 0.25;
/// A round's extra cost per 1,000 positions past 9,000: a step and a row, in units of a 22.7 ms one-row forward.
const depth_ctx = [2]f64{ 0.0723 / 22.7, 0.0060 / 22.7 };
const first_copy_rows = 6; // a lone stream's first copy window
const prompt_min = 256; // prompts under this many tokens run as decode rows, longer ones on the prompt path

const Lane = struct { slot: u32, held: [16]u32 = undefined, nheld: u32 = 0 };
const ring = 1024;

pub const Lanes = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    free: u32, // slot bits
    drawn: [ring]u32 = undefined,
    next: u64 = 0,
    costs: [8]lanes.config.Cost = undefined,
    shared: [32]lanes.config.Cost = undefined,
    nshared: usize = 0,
    draft_ms: f64 = 0,

    fn of(p: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(p));
    }

    pub fn backend(l: *Lanes) be.Backend {
        return .{ .ptr = l, .vtable = &.{ .prefill = prefillFn, .first = firstFn, .queue = queueFn, .read = readFn, .verify = verifyFn, .keep = keepFn, .draft = draftFn, .release = releaseFn } };
    }

    pub fn facts(l: *const Lanes) lanes.Model {
        const drafting = l.e.dr != null;
        const B: u32 = if (l.e.dr) |dr| @intCast(dr.block) else 1;
        return .{
            .exact_width = B,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .draft_prior = &draft_prior,
            .drafts = B - 1,
            .window_costs = l.costs[0..B],
            .mtp_step_ms = l.draft_ms,
            .hidden_rows = true,
            .batch_rows = fwd.DECODE_ROWS,
            .max_streams = @intCast(l.e.model.slots),
            .shared_costs = l.shared[0..l.nshared],
            .draft_streams = true,
            .geometric_depth = true,
            .geometric_costs = &depth_costs,
            .geometric_draft_cost = draft_cost,
            .geometric_ctx = depth_ctx,
            .first_copy_rows = first_copy_rows,
        };
    }

    /// Graphed forward costs at load on distinct tokens (identical rows share experts and price wide windows too low).
    pub fn measure(l: *Lanes, io: std.Io) !void {
        const e = l.e;
        const B: usize = if (e.dr) |dr| dr.block else 1;
        var rows: [32]fwd.Row = undefined;
        const reps = 16; // windows; shared rows below take 8
        var rng = std.Random.DefaultPrng.init(0x31A0);
        const V = e.w.full.vocab;
        for (1..B + 1) |w| {
            for (0..w) |j| rows[j] = .{ .slot = 0, .pos = @intCast(64 + j), .token = 0 };
            l.costs[w - 1] = .{ .width = @intCast(w), .ms = try l.time(io, rows[0..w], reps, &rng, V) };
        }
        l.nshared = 0;
        const S = e.model.slots;
        for (2..33) |n| {
            for (0..n) |j| rows[j] = .{ .slot = @intCast(j % S), .pos = @intCast(64 + j / S), .token = 0 };
            l.shared[l.nshared] = .{ .width = @intCast(n), .ms = try l.time(io, rows[0..n], 8, &rng, V) };
            l.nshared += 1;
        }
        if (e.dr != null) {
            var out: [16]i32 = undefined;
            for (0..3) |_| try e.draft(&.{1000}, &.{64}, &.{0}, &out);
            var dms: [16]f64 = undefined;
            for (&dms, 0..) |*m, i| {
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                try e.draft(&.{@intCast(1000 + 7919 * i)}, &.{64}, &.{0}, &out);
                m.* = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
            }
            std.mem.sort(f64, &dms, {}, std.sort.asc(f64));
            l.draft_ms = dms[8];
        }
        std.log.info("mimo costs (distinct tokens): window ms {d:.2} .. {d:.2} (1 .. {d} rows), shared 8 / 16 / 32 rows {d:.2} / {d:.2} / {d:.2}, draft {d:.2}", .{ l.costs[0].ms, l.costs[B - 1].ms, B, l.shared[6].ms, l.shared[14].ms, l.shared[l.nshared - 1].ms, l.draft_ms });
    }

    /// Median ms of `reps` graphed forwards of `rows`, each on fresh pseudo-random tokens, after warm replays.
    fn time(l: *Lanes, io: std.Io, rows: []fwd.Row, reps: usize, rng: *std.Random.DefaultPrng, V: usize) !f64 {
        const fill = struct {
            fn f(rs: []fwd.Row, r: *std.Random.DefaultPrng, v: usize) void {
                for (rs) |*x| x.token = @intCast(1000 + r.random().uintLessThan(usize, v - 2000));
            }
        }.f;
        for (0..3) |_| { // capture, then warm
            fill(rows, rng, V);
            try l.e.forward(rows, .ids);
        }
        try l.e.s.synchronize();
        var ms: [32]f64 = undefined;
        const n = @min(reps, ms.len);
        for (0..n) |i| {
            fill(rows, rng, V);
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            try l.e.forward(rows, .ids);
            try l.e.s.synchronize();
            ms[i] = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
        }
        std.mem.sort(f64, ms[0..n], {}, std.sort.asc(f64));
        return ms[n / 2];
    }

    fn take(l: *Lanes, token: u32) u64 {
        const h = l.next;
        l.drawn[h % ring] = token;
        l.next += 1;
        return h;
    }

    fn value(l: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| l.drawn[h % ring],
            .value => |v| v,
        };
    }

    fn rowOf(l: *Lanes, r: usize) []const f32 {
        const V = l.e.w.full.vocab;
        return l.e.logits[r * V ..][0..V];
    }

    fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const l = of(p);
        const e = l.e;
        const ids = s.prompt();
        const B: usize = if (e.dr) |dr| dr.block else 1;
        if (ids.len == 0 or ids.len + s.max_new + B > e.model.context) return error.PromptTooLong;
        const gop = try l.lanes.getOrPut(l.gpa, s);
        if (!gop.found_existing) {
            if (l.free == 0) {
                l.lanes.removeByPtr(gop.key_ptr);
                return error.NoFreeSlot;
            }
            const slot: u32 = @ctz(l.free);
            l.free &= ~(@as(u32, 1) << @intCast(slot));
            gop.value_ptr.* = .{ .slot = slot };
        }
        gop.value_ptr.nheld = 0;
        const slot = gop.value_ptr.slot;
        s.cached = 0;
        const greedy = s.sampling == null or s.sampling.?.temperature <= 0;
        if (ids.len >= prompt_min) {
            const pr = e.pr;
            // the prompt path: Python's lone-stream chunks, split to this engine's chunk rows (prompt.plan)
            const cuts = try prompt_mod.plan(l.gpa, 0, ids.len, pr.cap);
            defer l.gpa.free(cuts);
            var at: usize = 0;
            for (cuts) |n| {
                if (s.isCancelled()) return error.Cancelled;
                const last = at + n == ids.len;
                try e.promptChunk(slot, at, ids[at .. at + n], if (!last) .none else if (greedy) .ids else .logits);
                if (last) _ = l.take(if (greedy) @intCast(e.ids[0]) else try draw(l.gpa, l.rowOf(0), s.sampling, ids.len));
                at += n;
            }
            return;
        }
        var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
        var at: usize = 0;
        while (at < ids.len) {
            if (s.isCancelled()) return error.Cancelled;
            const n = @min(fwd.DECODE_ROWS, ids.len - at);
            for (0..n) |j| rows[j] = .{ .slot = slot, .pos = @intCast(at + j), .token = ids[at + j] };
            const last = at + n == ids.len;
            try e.forward(rows[0..n], if (!last) .none else if (greedy) .ids else .logits);
            if (last) _ = l.take(if (greedy) @intCast(e.ids[n - 1]) else try draw(l.gpa, l.rowOf(n - 1), s.sampling, ids.len));
            at += n;
        }
    }

    fn firstFn(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const l = of(p);
        if (position != s.prompt_len) return error.PositionMismatch;
        return l.next - 1;
    }

    fn queueFn(_: *anyopaque, _: *lanes.Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.NotPipelined;
    }

    fn readFn(p: *anyopaque, handle: u64) anyerror!u32 {
        const l = of(p);
        if (handle >= l.next or l.next - handle > ring) return error.NoSuchToken;
        return l.drawn[handle % ring];
    }

    /// Every window's rows in one forward; each row drawn with its stream's sampling at its keyed position.
    fn verifyFn(p: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const l = of(p);
        var rows: [fwd.DECODE_ROWS]fwd.Row = undefined;
        var n: usize = 0;
        for (windows) |w| {
            if (w.parents != null) return error.TreesNotBuilt;
            const lane = l.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            if (n + w.rows() > rows.len) return error.WindowTooWide;
            if (w.held > lane.nheld) return error.NoHeldDrafts;
            for (0..w.rows()) |r| {
                const tok: u32 = if (r == 0) w.pending else if (w.held > 0) lane.held[r - 1] else w.tokens[r - 1];
                rows[n] = .{ .slot = lane.slot, .pos = @intCast(w.positions[r] - 1), .token = tok };
                n += 1;
            }
        }
        var all_greedy = true;
        for (windows) |w| if (w.stream.sampling) |sm| if (sm.temperature > 0) {
            all_greedy = false;
        };
        try l.e.forward(rows[0..n], if (all_greedy) .ids else .logits);
        var r0: usize = 0;
        for (windows, out) |w, *o| {
            const lane = l.lanes.getPtr(w.stream).?;
            for (0..w.rows()) |r| o.sampled[r] = if (all_greedy) @intCast(l.e.ids[r0 + r]) else try draw(l.gpa, l.rowOf(r0 + r), w.stream.sampling, w.positions[r]);
            if (w.held > 0) @memcpy(o.drafts, lane.held[0..w.held]) else @memcpy(o.drafts, w.tokens);
            r0 += w.rows();
        }
    }

    fn keepFn(_: *anyopaque, _: []const be.Window, paths: []const []const u32) anyerror!void {
        for (paths) |path| {
            if (path.len == 0) return error.EmptyPath;
            for (path, 0..) |r, i| if (r != i) return error.TreesNotBuilt;
        }
    }

    /// One draft pass for every request that asks for drafts (their contexts are fed already).
    fn draftFn(p: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const l = of(p);
        const dr = l.e.dr orelse return error.NoDraftHead;
        var toks: [16]i64 = undefined;
        var pos0: [16]u32 = undefined;
        var sl: [16]u32 = undefined;
        var who: [16]*Lane = undefined;
        var depth: [16]u32 = undefined;
        var n: usize = 0;
        for (requests) |r| {
            const lane = l.lanes.getPtr(r.stream) orelse return error.UnknownStream;
            lane.nheld = 0;
            if (r.depth == 0) continue;
            if (n == toks.len) return error.TooManyStreams;
            const pending: u32 = if (r.follow.len > 0) r.follow[r.follow.len - 1] else l.value(r.first orelse return error.NoFirstToken);
            toks[n] = pending;
            pos0[n] = @intCast(r.position - 1);
            sl[n] = lane.slot;
            who[n] = lane;
            depth[n] = @min(r.depth, @as(u32, @intCast(dr.block - 1)));
            n += 1;
        }
        if (n == 0) return;
        var out: [16 * 16]i32 = undefined;
        try l.e.draft(toks[0..n], pos0[0..n], sl[0..n], &out);
        for (0..n) |i| {
            for (0..depth[i]) |j| who[i].held[j] = @intCast(out[i * (dr.block - 1) + j]);
            who[i].nheld = depth[i];
        }
    }

    fn releaseFn(p: *anyopaque, s: *lanes.Stream) void {
        const l = of(p);
        const kv = l.lanes.fetchRemove(s) orelse return;
        l.free |= @as(u32, 1) << @intCast(kv.value.slot);
    }
};

/// The token for a row: the first maximum when greedy (or temperature 0), else the keyed sampler over the top-k.
pub fn draw(gpa: std.mem.Allocator, row: []const f32, settings: ?lanes.Sampling, position: u64) !u32 {
    const best = fwd.argmax(row);
    if (!std.math.isFinite(row[best])) return error.NonfiniteLogits;
    const sampling = settings orelse return @intCast(best);
    if (sampling.temperature <= 0) return @intCast(best);
    const k = if (sampling.top_k == 0) row.len else @min(row.len, sampling.top_k);
    const values = try gpa.alloc(f64, k);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, k);
    defer gpa.free(ids);
    var filled: usize = 0;
    for (row, 0..) |x32, id| {
        const x: f64 = x32;
        if (k == row.len) {
            values[id] = x;
            ids[id] = id;
            continue;
        }
        if (filled == k and x <= values[k - 1]) continue;
        var at = @min(filled, k - 1);
        while (at > 0 and x > values[at - 1]) : (at -= 1) {
            values[at] = values[at - 1];
            ids[at] = ids[at - 1];
        }
        values[at] = x;
        ids[at] = id;
        filled = @min(filled + 1, k);
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, sampling));
}

const Owned = struct { gpa: std.mem.Allocator, e: *Engine, l: Lanes, session: ?*tp.Session };

/// The native family entry: TF_TP_* for two ranks (rank 0 serves; `tf-mimo follow` on rank 1) and a captured set.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    const aot_dir = kernels orelse return error.NoCapturedTritonSet; // TENSORFOLD_CUDA_KERNELS
    if (o.lanes == 0 or o.lanes > 16) return error.Slots;
    const policy = try Policy.parse(o.policy);
    const su: engine.Setup = .{ .context = o.context, .slots = o.lanes, .drafts = o.drafts, .kv8 = policy.kv8 };
    var session: ?*tp.Session = null;
    const cfg = tp.Config.fromEnv() catch null;
    if (cfg) |c| if (c.world > 1) {
        if (c.rank != 0) return error.FollowerServes; // rank 1 runs `tf-mimo follow`
        const ss = try gpa.create(tp.Session);
        errdefer gpa.destroy(ss);
        try ss.open(ctx.d, c, @import("tp_kernel").mailbox); // RoCE needs the mailbox kernel (TF_COMM_BACKEND=roce)
        session = ss;
        try ss.plan.send(&engine.hello(su));
    };
    errdefer if (session) |ss| {
        ss.announceStop();
        ss.close();
        gpa.destroy(ss);
    };
    const e = try Engine.init(gpa, io, ctx.d, dir, aot_dir, su, session);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .l = .{ .gpa = gpa, .e = e, .free = if (su.slots == 32) ~@as(u32, 0) else (@as(u32, 1) << @intCast(su.slots)) - 1 }, .session = session };
    try own.l.measure(io);
    return .{
        .backend = own.l.backend(),
        .facts = own.l.facts(),
        .rows = if (e.dr) |dr| @intCast(dr.block) else 1,
        .stream_bytes = 0,
        .free_streams = @intCast(su.slots),
        .ctx = own,
        .deinit = release,
        .target = own,
        .policy = policy.line(),
    };
}

/// The weights this rank loads: the checkpoint over TF_TP_WORLD ranks (the drafter and embeddings, whole on each).
pub fn rankWeights(total: u64) u64 {
    const text = std.mem.span(std.c.getenv("TF_TP_WORLD") orelse return total);
    const world = std.fmt.parseInt(u64, text, 10) catch 1;
    return total / @max(world, 1);
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.NoFreeSlot => "every stream slot is busy: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.session) |ss| {
        ss.plan.send(&.{ @backingInt(engine.Op.stop), 0 }) catch {};
    }
    own.l.lanes.deinit(own.gpa);
    own.e.deinit();
    if (own.session) |ss| {
        ss.announceStop();
        ss.close();
        own.gpa.destroy(ss);
    }
    own.gpa.destroy(own);
}

test "policy: kv=fp16 by default, kv=fp8 opts in, anything else is refused" {
    try std.testing.expect(!(try Policy.parse("")).kv8);
    try std.testing.expect((try Policy.parse("kv=fp8")).kv8);
    try std.testing.expect(!(try Policy.parse("kv=fp8,kv=fp16")).kv8);
    try std.testing.expectEqualStrings("kv=fp8", (try Policy.parse(" kv = fp8 ")).line());
    try std.testing.expectError(error.Policy, Policy.parse("kv=int4"));
    try std.testing.expectError(error.Policy, Policy.parse("matrix=off"));
    try std.testing.expectError(error.Policy, Policy.parse("kv"));
}
