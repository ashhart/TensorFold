//! The decode forward: a slot's windows over the whole model, block.zig's program run by run.zig, TP exchanges.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const Config = @import("config.zig").Config;
const eh = @import("engram_host.zig");
const graphs = @import("graphs.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const kvs = @import("cuda_kv.zig");
const er = @import("engram_rows.zig");
pub const kops = @import("dsv41_kernels").ops;
const ph = @import("phases.zig");
const hs = @import("host_stage.zig");
const fwd_sample = @import("forward_sample.zig");
const fwd_follow = @import("forward_follow.zig");
const fwd_glue = @import("forward_glue.zig");
const fwd_engram = @import("forward_engram.zig");
const fwd_graph = @import("forward_graph.zig");

pub const Error = error{ GlueNotBuilt, BadWindow };

/// One slot's position, Engram tail and pending window (what keep() commits).
pub const Slot = struct {
    pos: u64 = 0,
    /// the last max_ngram - 1 committed ids (Engram's lookback; fewer at the sequence start)
    tail: [eh.max_ngram]u32 = undefined,
    tail_len: usize = 0,
    /// the window in flight: its first position, its rows
    pending: ?struct { start: u64, n: u32 } = null,
    /// CED replay (TF_DSV41_PREFILL=replay): the positions and ids of the stash ring "s.ced.*" (ced.zig)
    ced: ft.Stash = .{},
};

pub const Forward = struct {
    gpa: std.mem.Allocator,
    cfg: *const Config,
    widths: *const block.Widths,
    opts: block.Options,
    runner: *run.Runner,
    comm: tp.collective.Collective,
    /// the window's calls by row count, emitted once (arena-owned)
    arena: std.heap.ArenaAllocator,
    programs: std.AutoHashMapUnmanaged(u32, []const calls.Call) = .empty,
    slot: Slot = .{},
    transaction: @import("cuda_transaction.zig").Transaction = .{},
    /// pinned staging of the window's token ids, and their device copy (int64, the embedding step reads them)
    ids_stage: hs.Stage = .{},
    uploads: hs.Stage = .{},
    init_owners: std.ArrayList(@import("construction.zig").Owner) = .empty,
    dev_ids: ?cuda.DeviceBuffer = null,
    /// Engram's hasher (recipe/dsv41_engram_host.py's tables) and the rows' source; none: the synthetic rows
    engram: ?*const eh.Host = null,
    gathered: ?cuda.DeviceBuffer = null,
    rows_stage: [eh.max_layers]hs.Stage = @splat(.{}), // Engram's own pinned staging, one a layer (copies may overlap)
    /// a bf16 [rows, hidden] send buffer for the exchanges whose partial is not bf16 already (embedding, MoE)
    send: ?cuda.DeviceBuffer = null,
    /// the window's token ids (the embedding step reads them; window() copies them into ids_own)
    ids: []const u32 = &.{},
    /// TF_DSV41_IMAGES=native (vision_rows.zig): every rank keeps image positions out of Engram; rank 0's image rows
    images: bool = false,
    /// TF_DSV41_BIAS_VL loaded (every rank): image rows route with gate.bias_vl ("L<i>.moe.bias_vl")
    image_bias: bool = false,
    vision: ?*anyopaque = null, // vision_rows.Rows
    ids_own: std.ArrayList(u32) = .empty,
    /// the backbone layers the forward runs (all of them unless a gate runs a prefix)
    layers: ?[]const u32 = null,
    /// TP: rank 0 leads (sends each forward operation to the others over the plan link); the others `follow`
    link: ?*tp.planlink.PlanLink = null,
    /// a follower's drafter (the leader's lanes ask its pass; every rank runs it)
    drafter: ?Drafter = null,
    /// each block's traced streams (block.Options.trace) as SHA-256 hex by layer, for the last window
    trace: ?*std.AutoArrayHashMapUnmanaged(u32, [64]u8) = null,
    /// the largest row bucket plan() sized the buffers for (a graphed window never has more rows)
    planned_rows: u32 = 0,
    graphed: ?*Graphed = null, // decode windows through CUDA graphs (TF_DSV41_GRAPHS=1); null until the knob is read
    graphs_read: bool = false,
    /// the graphs' default when TF_DSV41_GRAPHS is unset (model.zig: on for the served model; the gates' drivers: off)
    graphs_default: bool = false,
    /// Engram's local shards (TF_DSV41_ENGRAM_DIR, engram_rows.zig), opened on first use; unset: synthetic index rows
    engram_rows: ?*er.Rows = null,
    engram_opened: bool = false,
    /// Engram read-ahead at the DSpark pass (TF_DSV41_ENGRAM_PREFETCH=0: rows read when the window needs them)
    engram_prefetch: bool = true,
    gate: ?*anyopaque = null, // TF_DSV41_ENGRAM_GATE: opened with the table (engram_gate.Gate)
    gated: bool = false, // the window being issued armed the gate: its engram_rows steps launch the gate kernel
    /// TF_DSV41_ROUND_GRAPH's `gate`: recorded after each gated window, the gate's arm waits for it (not the stream)
    gated_end: ?cuda.Event = null,
    /// TF_DSV41_BRANCHES / TF_DSV41_MHC_DEFER (branches.zig): the runner's side streams, owned here
    branches: ?*anyopaque = null, // branches.Branches
    /// TF_DSV41_PHASES (phases.zig): the window path's timers
    phases: ?*ph.Phases = null,
    /// greedy's pinned logits / pairs staging and device pairs, kept across windows
    pick_host: ?cuda.HostBuffer = null,
    pick_dev: ?cuda.DeviceBuffer = null,
    greedy_gpu: ?bool = null, // TF_DSV41_GREEDY_GPU=1: each rank's best column on GPU, pairs copied back; null: unread
    greedy_dev: ?cuda.DeviceBuffer = null,
    /// the last GPU pick on the device (`pickDevice`), null when it was merged on the host
    pick_last: ?PickDevice = null,
    after_pick: ?AfterPick = null, // TF_DSV41_DRAFT_GRAPHS + TF_DSV41_SPEC_DRAFT: enqueued after the device pick
    after_sample: ?AfterSample = null,
    pick_done: ?cuda.Event = null, // the pick's copy-back event the host waits on
    /// the prefill's state (forward_prefill.State, forward_prefill.st), made at engine open
    prefill_state: ?*anyopaque = null,
    /// rows of "w.logits" the last prefill left (its last segment's, or its tail's): greedy takes out[0..last_rows]
    last_rows: u32 = 0,
    /// rank 0 inside a prefill: its windows and keeps are the followers' own (they got the whole prompt), not sent
    quiet: bool = false,
    /// the KV families in the paged pool (null: the slot's contiguous buffers); opts.pool is its emitter view
    kv: ?*kvs.Kv = null,
    /// the session store (sessions_gpu.zig): the ids each keep commits, and the operations a follower runs
    sess: ?Sess = null,
    /// a workstream's plan-link operations (slots.zig 16-19), run by a follower in the leader's order
    ext: ?Ext = null,
    /// rank 0's clean stop of every rank (model.zig: the TP session's announceStop), run by `stop`
    on_stop: ?struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void } = null,
    pf_switch: ?PfSwitch = null, // several live slots: a multi-segment prefill run's slot switch; null: one slot
    pf_stash: ?PfStash = null, // the CED stash ring made the current slot's before a stash step (swapped lazily)
    /// the multi-segment prefill run in flight (forward_prefill.promptMulti; its glue reads it), null outside one
    multi: ?*ft.Multi = null,
    /// the keyed sampler (sampling_gpu.zig): a follower runs rank 0's sampled choices too (their gathers pair up)
    sampler: ?Sampler = null,
    grammar: ?Grammar = null, // TF_DSV41_GRAMMAR=1: grammar masks applied to "w.logits" before a pick
    calib_marks: std.ArrayList(u64) = .empty, // TF_DSV41_CALIB=measure: a follower's clock at each op_calib_mark

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    pub fn upload(f: *Forward, destination: u64, bytes: []const u8) !void {
        const owned = try f.uploads.take(f.runner.d, bytes.len);
        @memcpy(owned, bytes);
        try f.uploads.copyTo(.{ .d = f.runner.d, .ptr = destination, .len = bytes.len }, 0, f.runner.stream);
        try f.uploads.wait();
    }
    pub fn drain(f: *Forward) !void {
        if (f.init_owners.items.len > 0) try f.runner.d.synchronize();
        try f.runner.stream.synchronize();
        try f.ids_stage.wait();
        try f.uploads.wait();
        for (&f.rows_stage) |*x| try x.wait();
        if (f.graphed) |g| try g.drain();
    }
    pub fn close(f: *Forward) !void {
        try f.drain();
        f.deinitAfterDrain();
    }
    pub fn deinitAfterDrain(f: *Forward) void {
        @import("construction.zig").deinit(f);
        f.ids_stage.close() catch unreachable;
        f.uploads.close() catch unreachable;
        for (&f.rows_stage) |*x| x.close() catch unreachable;
        f.transaction.deinit(f.gpa);
        if (f.graphed) |g| {
            g.deinitAfterDrain();
            f.gpa.destroy(g);
        }
        if (f.dev_ids) |*b| b.free();
        if (f.send) |*b| b.free();
        if (f.gathered) |*b| b.free();
        if (f.prefill_state != null) hooks.set.prefill.?.deinit(f);
        f.calib_marks.deinit(f.gpa);
        if (f.gate) |x| hooks.set.gate.?.close(x);
        if (f.engram_rows) |x| x.close();
        if (f.branches != null) hooks.set.branches.?.close(f);
        if (f.pick_host) |*b| b.free();
        if (f.pick_dev) |*b| b.free();
        if (f.greedy_dev) |*b| b.free();
        if (f.pick_done) |*e| e.deinit();
        if (f.gated_end) |*e| e.deinit();
        f.programs.deinit(f.gpa);
        f.ids_own.deinit(f.gpa);
        f.arena.deinit();
    }

    /// The backbone's layers, then the head.
    pub fn backbone(f: *Forward, a: std.mem.Allocator) ![]const u32 {
        if (f.layers) |ls| return ls;
        const ls = try a.alloc(u32, f.cfg.layers);
        for (ls, 0..) |*l, i| l.* = @intCast(i);
        return ls;
    }

    /// The buffer plan of every row bucket the forward runs (bind the runner with it once, before the first window).
    pub fn plan(f: *Forward, buckets: []const u32) !buffers.Plan {
        var p: buffers.Plan = .{ .a = f.arena.allocator() };
        for (buckets) |n| try p.add(try f.program(n));
        for (buckets) |n| f.planned_rows = @max(f.planned_rows, n);
        return p;
    }

    /// The calls of an `n`-row window at the slot's last position (the largest every role gets; plans size from it).
    pub fn program(f: *Forward, n: u32) ![]const calls.Call {
        if (f.programs.get(n)) |cs| return cs;
        const a = f.arena.allocator();
        // the emitter's positions only size the indexer's scores at the window's end; the largest the slot reaches
        const cs = try block.emit(a, f.cfg, f.widths, f.opts, try f.backbone(a), n, f.opts.limit - n, true);
        try f.programs.put(f.gpa, n, cs);
        return cs;
    }

    /// Adds the prefill program's roles (segments of up to `rows` rows) to `p` before bind; shared roles: the larger.
    pub fn planPrefill(f: *Forward, p: *buffers.Plan, rows: u32) !void {
        return (try hooks.prefill()).plan(f, p, rows);
    }

    /// The next prompt piece's ids for the Engram read-ahead (TF_DSV41_PREFETCH_AHEAD), every rank; I/O only.
    pub fn prefillAhead(f: *Forward, next: []const u32) !void {
        const ahead = (try hooks.prefill()).ahead(f) orelse return;
        if (next.len == 0) return;
        if (f.leads()) {
            const msg = try f.gpa.alloc(i64, 1 + next.len);
            defer f.gpa.free(msg);
            msg[0] = op_ahead;
            for (next, msg[1..]) |t, *y| y.* = t;
            try f.link.?.send(msg);
        }
        ahead.clearRetainingCapacity();
        try ahead.appendSlice(f.gpa, next);
    }

    /// The prompt `ids` from the slot's position: segments kept whole, a tail of at most 16 rows as decode windows.
    pub fn prefill(f: *Forward, ids: []const u32) !void {
        if (f.slot.pending != null) return error.BadWindow;
        if (ids.len == 0) return;
        f.gated = false; // prompt segments read their rows on the old path
        if (f.leads()) {
            // [op, total, offset, ids...] frames (planlink takes up to 64M words a frame; a prompt is a few)
            const frame: usize = 1 << 20;
            const msg = try f.gpa.alloc(i64, 3 + @min(frame, ids.len));
            defer f.gpa.free(msg);
            var at: usize = 0;
            while (at < ids.len) : (at += frame) {
                const m = @min(frame, ids.len - at);
                msg[0] = op_prefill;
                msg[1] = @intCast(ids.len);
                msg[2] = @intCast(at);
                for (ids[at .. at + m], msg[3 .. 3 + m]) |t, *y| y.* = t;
                try f.link.?.send(msg[0 .. 3 + m]);
            }
        }
        f.quiet = true;
        defer f.quiet = false;
        return (try hooks.prefill()).prompt(f, ids);
    }

    /// One decode window: `ids` at the slot's position. This rank's logits stay in the runner's "w.logits".
    pub fn window(f: *Forward, ids_in: []const u32) !void {
        // the forward's own copy: keep() reads the ids (Engram's tail) after the caller's slice may be gone
        if (ids_in.ptr != f.ids_own.items.ptr) {
            f.ids_own.clearRetainingCapacity();
            try f.ids_own.appendSlice(f.gpa, ids_in);
        }
        const ids = f.ids_own.items;
        if (ids.len == 0 or f.slot.pending != null) return error.BadWindow;
        // no embed of a window fills image rows (graphed: never; eager: no keep or bias), so image ids fail here
        if (hooks.set.vision) |v| try v.checkWindow(ids);
        // the pool's pages for the window's positions, their table entries up before the launches (every rank alike)
        if (f.kv) |kx| try kx.reserve(f.slot.pos + ids.len);
        if (f.leads()) {
            var msg: [2 + 128]i64 = undefined;
            if (ids.len > 128) return error.BadWindow;
            msg[0] = op_window;
            msg[1] = @intCast(ids.len);
            for (ids, 0..) |t, i| msg[2 + i] = t;
            try f.link.?.send(msg[0 .. 2 + ids.len]);
        }
        const tf = ph.start(f.phases, .forward);
        defer tf.stop();
        f.runner.rows = @intCast(ids.len); // l2pf: wider windows skip the prefetch sites
        try f.transaction.begin(f, @intCast(ids.len));
        if (try f.graphedWindow(ids)) return f.gatedEnd();
        // the gate (its worker reads every layer at once) or every layer's reads in flight while the calls are emitted
        f.gated = try f.armEngram(&.{.{ .ids = ids, .tail = f.slot.tail[0..f.slot.tail_len] }}, @intCast(ids.len));
        if (!f.gated) f.issueEngram(ids, f.slot.tail[0..f.slot.tail_len]);
        // the window's calls at its position (the indexer's scores and top-k grow with the cache: NK = end / ratio)
        var wa = std.heap.ArenaAllocator.init(f.gpa);
        defer wa.deinit();
        const cs = try block.emit(wa.allocator(), f.cfg, f.widths, f.opts, try f.backbone(wa.allocator()), @intCast(ids.len), @intCast(f.slot.pos), true);
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        f.runner.glue = .{ .ctx = f, .run = glueFn };
        try f.runner.timedWindow(cs);
        try f.gatedEnd();
    }

    pub const greedy = fwd_sample.greedy;
    pub const greedyScratch = fwd_sample.greedyScratch;

    /// The last window's pick on the device: `out` int64 [n + 3] = picks, then (chain) accepted, bonus, next position.
    pub const PickDevice = struct { out: u64, n: u32, chain: bool, start: u64 = 0, host: ?[*]const i64 = null };

    /// Work enqueued from the device pick, on every rank, after the pick's copy back and before the host waits for it.
    pub const AfterPick = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, pick: PickDevice) anyerror!void };

    /// Work enqueued from sampled device choices (rows [row0, row0 + n) into `chosen` int64 [n]), every rank.
    pub const AfterSample = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void };

    pub fn pickDevice(f: *const Forward) ?PickDevice {
        return f.pick_last;
    }

    /// Commits the pending window's first row + `accepted` drafts.
    pub fn keep(f: *Forward, accepted: u32) !void {
        const p = f.slot.pending orelse return error.BadWindow;
        if (accepted >= p.n) return error.BadWindow;
        if (f.sess) |x| if (x.reserve) |reserve| try reserve(x.ptr, f.ids[0 .. accepted + 1]);
        try f.engramCheck(); // a gated window whose rows failed: its picks are not committed
        if (f.leads()) try f.link.?.send(&.{ op_keep, accepted });
        // the ratio-2 KV sources' carry: fp32 of the accepted row of the window's compressor projection
        const r = f.runner;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(f.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
            const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            try r.kernels.others(r.stream).carry(src, w, 0, accepted, w, dst);
        };
        try f.transaction.keep(accepted + 1);
        // Engram's tail: (tail + the committed ids)[-(max_ngram - 1):]
        if (f.engram) |h| {
            const keep_n = h.max_ngram - 1;
            var all: [2 * eh.max_ngram + 128]u32 = undefined;
            var m: usize = 0;
            for (f.slot.tail[0..f.slot.tail_len]) |t| {
                all[m] = t;
                m += 1;
            }
            const take = @min(accepted + 1, all.len - m);
            for (f.ids[0..take]) |t| {
                all[m] = t;
                m += 1;
            }
            const k = @min(m, keep_n);
            @memcpy(f.slot.tail[0..k], all[m - k .. m]);
            f.slot.tail_len = k;
        }
        f.slot.pos = p.start + accepted + 1;
        f.slot.pending = null;
        if (f.sess) |x| try x.commit(x.ptr, f.ids[0 .. accepted + 1]);
    }

    const op_window = fwd_follow.op_window;
    const op_greedy = fwd_follow.op_greedy;
    const op_keep = fwd_follow.op_keep;
    const op_stop = fwd_follow.op_stop;
    const op_ds_ingest = fwd_follow.op_ds_ingest;
    const op_ds_propose = fwd_follow.op_ds_propose;
    const op_prefill = fwd_follow.op_prefill;
    const op_ahead = fwd_follow.op_ahead;
    const op_sess_save = fwd_follow.op_sess_save;
    const op_sess_restore = fwd_follow.op_sess_restore;
    const op_release = fwd_follow.op_release;
    const op_ds_reset = fwd_follow.op_ds_reset;
    pub const op_sample = fwd_follow.op_sample;
    pub const op_sample_many = fwd_follow.op_sample_many;

    /// The sampler's side of a follower (sampling_gpu.zig): rank 0's op_sample message, run on this rank.
    pub const Sampler = struct {
        ptr: *anyopaque,
        follow: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void,
    };

    pub const sendSample = fwd_follow.sendSample;
    const op_tree_drop = fwd_follow.op_tree_drop;
    const op_ced_finish = fwd_follow.op_ced_finish;
    const op_sess_prompt = fwd_follow.op_sess_prompt;
    const op_sess_admit = fwd_follow.op_sess_admit;
    const op_pf_multi = fwd_follow.op_pf_multi;
    const op_ced_multi = fwd_follow.op_ced_multi;
    pub const replayMulti = fwd_follow.replayMulti;

    /// A slot's views made current on this rank (slots.zig SlotSet.view), for a multi-segment run's segments.
    pub const PfSwitch = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, slot: u32) anyerror!void };
    pub const PfStash = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

    pub const prefillMulti = fwd_follow.prefillMulti;
    pub const finishPrompt = fwd_follow.finishPrompt;
    pub const op_grammar = fwd_follow.op_grammar;

    /// Structured output's side of the forward: `apply` before a pick reads "w.logits", `follow` an op_grammar message.
    pub const Grammar = struct {
        ptr: *anyopaque,
        follow: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void,
        apply: *const fn (ptr: *anyopaque, row0: u32, n: u32) anyerror!void,
    };

    /// A workstream's follower side: rank 0's message of an operation number this file does not know.
    pub const Ext = struct { ptr: *anyopaque, run: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void };

    /// The session store's side: `commit` hears every committed id; `save` / `restore` run in the leader's order.
    pub const Sess = struct {
        ptr: *anyopaque,
        commit: *const fn (ptr: *anyopaque, ids: []const u32) anyerror!void,
        reserve: ?*const fn (ptr: *anyopaque, ids: []const u32) anyerror!void = null,
        save: *const fn (ptr: *anyopaque) anyerror!void,
        /// `ids`: an NVMe entry's ids (the leader's prompt prefix: its tokens are not indexed)
        restore: *const fn (ptr: *anyopaque, id: u32, ids: []const i64) anyerror!void,
        /// CED replay's prompt snapshot: an entry at the slot's position, the slot keeps going (sessions_gpu.zig)
        prompt: *const fn (ptr: *anyopaque) anyerror!void,
        /// the slot empty: its pages back, the committed ids cleared
        reset: *const fn (ptr: *anyopaque) anyerror!void,
        /// a request's pool admission (op_sess_admit's message): the leader's spills evicted, its pages reserved
        admit: ?*const fn (ptr: *anyopaque, msg: []const i64) anyerror!void = null,
    };

    pub const op_calib_mark = fwd_follow.op_calib_mark;
    pub const op_calib_gather = fwd_follow.op_calib_gather;
    pub const calibMark = fwd_follow.calibMark;
    pub const calibGather = fwd_follow.calibGather;
    pub const followCalib = fwd_follow.followCalib;
    pub const drop = fwd_follow.drop;
    pub const release = fwd_follow.release;
    pub const sendRestore = fwd_follow.sendRestore;
    pub const sendSess = fwd_follow.sendSess;
    pub const sendSessAdmit = fwd_follow.sendSessAdmit;

    /// The drafter's operations a follower runs (dspark_gpu.zig): its collectives pair up with the leader's.
    pub const Drafter = struct {
        ptr: *anyopaque,
        ingest: *const fn (ptr: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void,
        propose: *const fn (ptr: *anyopaque, anchor: u32, start: u64) anyerror!void,
        /// the slot's draft context restarts (a new request): every rank's, or their candidates differ
        reset: *const fn (ptr: *anyopaque) void,
    };

    pub const dsResetOp = fwd_follow.dsResetOp;
    pub const sendDrafter = fwd_follow.sendDrafter;
    pub const dsIngestOp = fwd_follow.dsIngestOp;
    pub const dsProposeOp = fwd_follow.dsProposeOp;

    pub fn leads(f: *const Forward) bool {
        return f.link != null and !f.quiet and f.comm.rank() == 0;
    }

    pub const stop = fwd_follow.stop;
    pub const follow = fwd_follow.follow;
    const glueMissing = fwd_glue.glueMissing;
    const selectionGlue = fwd_glue.selectionGlue;
    const numel = fwd_glue.numel;
    pub const sendBuffer = fwd_glue.sendBuffer;
    pub const glueFn = fwd_glue.glueFn;
    pub const engramRows = fwd_engram.engramRows;
    pub const engramCols = fwd_engram.engramCols;
    pub const fillEngramRows = fwd_engram.fillEngramRows;
    pub const engramSource = fwd_engram.engramSource;
    pub const armEngram = fwd_engram.armEngram;
    pub const gatedEnd = fwd_engram.gatedEnd;
    pub const engramCheck = fwd_engram.engramCheck;
    pub const gateCopy = fwd_engram.gateCopy;
    pub const issueEngram = fwd_engram.issueEngram;
    pub const prefetchEngram = fwd_engram.prefetchEngram;
    pub const engramGathered = fwd_graph.engramGathered;
    pub const activeGraph = fwd_graph.activeGraph;
    pub const graphState = fwd_graph.graphState;
    const graphedWindow = fwd_graph.graphedWindow;
    const graphBody = fwd_graph.graphBody;
    pub const fingerprint = fwd_graph.fingerprint;
};

pub const Graphed = fwd_graph.Graphed;

/// A world-2 peer for one GPU: the peer's half of every all-gather is zeros (only the forward's all-gather exists).
pub const ZeroPeer = struct {
    d: *const cuda.Driver,

    const C = tp.collective;

    fn rank(_: *anyopaque) u32 {
        return 0;
    }
    fn world(_: *anyopaque) u32 {
        return 2;
    }
    fn allGather(ptr: *anyopaque, send: C.DevicePtr, recv: C.DevicePtr, count: usize, t: C.DType, stream: C.Stream) C.Error!void {
        const z: *ZeroPeer = @ptrCast(@alignCast(ptr));
        const n = count * t.size();
        z.d.check(z.d.api.cuMemcpyDtoDAsync_v2(recv, send, n, stream), "cuMemcpyDtoDAsync") catch return error.BackendFailed;
        z.d.check(z.d.api.cuMemsetD8Async(recv + n, 0, n, stream), "cuMemsetD8Async") catch return error.BackendFailed;
    }
    fn unsupported(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: C.Op, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn exchange(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn p2p(_: *anyopaque, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn none(_: *anyopaque) C.Error!void {}
    fn abort(_: *anyopaque) void {}

    const vtable: C.Collective.VTable = .{
        .kind = .host,
        .rank = rank,
        .world = world,
        .all_reduce = unsupported,
        .all_gather = allGather,
        .exchange = exchange,
        .send = p2p,
        .recv = p2p,
        .barrier = none,
        .check = none,
        .abort = abort,
    };

    pub fn collective(z: *ZeroPeer) C.Collective {
        return .{ .ptr = z, .vtable = &vtable };
    }
};
