//! Batched decode windows over several slots: one forward over every slot's window, rows padded to a bucket.

// Eager and graphed run the same program keyed by (bucket rows, context bucket), so graphs on == off, batched == alone.

const std = @import("std");
const ph = @import("phases.zig");
const cuda = @import("cuda");
const kv = @import("kv");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const graphs = @import("graphs.zig");
const round_graph = @import("round_graph.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const slots_mod = @import("slots.zig");
const iface = @import("draft/iface.zig");
const buffers = @import("buffers.zig");
const samp = @import("sampling_gpu.zig");
const sessions_gpu = @import("sessions_gpu.zig");
const batch_engram = @import("batch_engram.zig");
const batch_graph = @import("batch_graph.zig");
const batch_target = @import("batch_target.zig");
const batch_prof = @import("batch_prof.zig");

pub const Error = error{ BadWindow, NotAChain, RowsCap, BadPlan, TreeWindow, Sampling, Slots };

/// With taps (drafts): each slot's taps rows of a round's earlier forwards, [S, seg_max, taps width] bf16.
pub const taps_stash_role = "s.rows.taps";

/// TF_DSV41_ROWS_CAP: rows one forward takes (a bucket; default 16).
pub fn capFromEnv() !u32 {
    const v = std.c.getenv("TF_DSV41_ROWS_CAP") orelse return 16;
    const n = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (rowtab.bucketRows(n, 64) != n) return error.RowsCap;
    return n;
}

/// A slot's rows in the last row window; `stashed`: its ratio-2 projection rows were copied aside, its taps gone.
const Pend = struct { a: u32, start: u64, n: u32, ids: [rowtab.seg_max]u32 = undefined, stashed: bool = false };

pub const Batch = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    ss: *slots_mod.SlotSet,
    cap: u32,
    settings: graphs.Settings,
    tab: Table,
    /// each ratio-2 kv source's projection rows of pending slots, [S, seg_max, 2D] bf16 (a round split over forwards)
    stash: std.ArrayList(Stash) = .empty,
    /// with taps: the pending windows' taps rows of a round split over forwards (taps_stash_role), null without
    taps_stash: ?cuda.DeviceBuffer = null,
    stages: std.ArrayList(graphs.EngramStage) = .empty,
    /// each key's row program (emitted once, the forward's arena)
    programs: std.AutoHashMapUnmanaged(graphs.Key, []const calls.Call) = .empty,
    taps_at: std.AutoHashMapUnmanaged(graphs.Key, ?usize) = .empty, // each key's program's last "w.taps" call, cached
    g: ?Graphs = null,
    pend: []?Pend,
    /// the last window's rows (real, padded)
    n: u32 = 0,
    R: u32 = 0,
    current: []const calls.Call = &.{},
    marked: bool = false, // TF_DSV41_DRAFT_OVERLAP: the last window recorded the taps mark; ingests may wait on it
    /// the calls of `current` the graph body issues (TF_DSV41_ROUND_GRAPH's `split`: the head's, then the tail's)
    span: [2]usize = .{ 0, 0 },
    /// TF_DSV41_ROUND_GRAPH (round_graph.zig)
    rg: round_graph.Settings = .{},
    stats: struct {
        windows: u64 = 0,
        rows: u64 = 0,
        padded: u64 = 0,
        split: u64 = 0,
        builds: u64 = 0, // first uses of a key: program emitted (host ms in build_ns); `eager`: windows without a graph
        build_ns: u64 = 0,
        captures: u64 = 0,
        capture_ns: u64 = 0,
        eager: u64 = 0,
        prep_ns: u64 = 0, // host time on the critical path: checks, plan send, pages, the program
        stage_ns: u64 = 0, // the row table, Engram's arm or rows
        launch_ns: u64 = 0, // the graph's launch or the eager window
        keep_ns: u64 = 0,
    } = .{},
    prof: ?*WinProf = null, // TF_DSV41_WIN_PROF=1: host ns per runRows step, logged at deinit; null: no clock reads
    /// the session store (model.zig): each slot's kept rows are its committed ids
    sess: ?*sessions_gpu.Sessions = null,

    /// After Runner.bind and SlotSet.bindBases (the stacked roles have addresses): the table, the Engram stages.
    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, ss: *slots_mod.SlotSet, cap: u32) !*Batch {
        const b = try gpa.create(Batch);
        var assembled = false;
        errdefer if (assembled) b.deinit() else gpa.destroy(b);
        const r = f.runner;
        // graphs as the one-slot forward's default (served: on unless TF_DSV41_GRAPHS=0; the gates' drivers: off)
        const s = try graphs.Settings.fromEnvOr(f.graphs_default);
        var tab = try Table.init(r.d, cap);
        errdefer if (!assembled) tab.deinit();
        const pend = try gpa.alloc(?Pend, ss.n);
        errdefer if (!assembled) gpa.free(pend);
        b.* = .{ .gpa = gpa, .f = f, .ss = ss, .cap = cap, .settings = s, .tab = tab, .pend = pend, .rg = try round_graph.current() };
        assembled = true;
        if (WinProf.fromEnv()) b.prof = try gpa.create(WinProf);
        if (b.prof) |p| p.* = .{};
        @memset(b.pend, null);
        {
            var la = std.heap.ArenaAllocator.init(gpa);
            defer la.deinit();
            const w = 2 * @as(usize, f.cfg.head_dim);
            for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
                try b.stash.ensureUnusedCapacity(gpa, 1);
                b.stash.appendAssumeCapacity(.{ .layer = L, .buf = try cuda.DeviceBuffer.alloc(r.d, 2 * w * rowtab.seg_max * @as(usize, ss.n)) });
            };
        }
        if (f.opts.taps) {
            b.taps_stash = try cuda.DeviceBuffer.alloc(r.d, tapsBytes(f) * rowtab.seg_max * @as(usize, ss.n));
        }
        // every buffer a row window bakes in, at its largest: the exchanges' send, Engram's gathered rows
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        const most: usize = @max(cap, f.planned_rows);
        _ = try f.sendBuffer(r, 2 * most * D);
        if (f.engram) |h| {
            var la = std.heap.ArenaAllocator.init(gpa);
            defer la.deinit();
            const cols = engramCols(f, h);
            for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) {
                try b.stages.ensureUnusedCapacity(gpa, 1);
                b.stages.appendAssumeCapacity(try graphs.EngramStage.init(r.d, L, cap, cols));
            };
            const bytes = W * 2 * most * cols;
            _ = try @import("grow_device.zig").grow(&f.gathered, r.d, bytes);
        }
        if (s.on) {
            b.g = @as(Graphs, undefined);
            b.g.?.init(f, s, b.rg.split) catch |e| {
                b.g = null;
                return e;
            };
            std.log.scoped(.dsv41).info("row graphs: on, buckets up to {d} rows, context buckets of {d}{s}", .{ cap, s.bucket, if (b.rg.split) " (split after the head's layers)" else "" });
        }
        const bindings = [_]run.Runner.External{
            .{ .role = rowmode.table_role, .addr = b.tab.dev.ptr },
            .{ .role = taps_stash_role, .addr = if (b.taps_stash) |buf| buf.ptr else 0 },
        };
        try r.externals(bindings[0..if (f.opts.taps) 2 else 1]);
        return b;
    }

    pub fn drain(b: *Batch) !void {
        try b.f.runner.stream.synchronize();
        try b.tab.drain();
        for (b.stages.items) |*st| try st.drain();
    }
    pub fn deinit(b: *Batch) void {
        if (b.prof) |p| {
            p.log(b.f.comm.rank());
            b.gpa.destroy(p);
        }
        if (b.g) |*g| g.deinit();
        for (b.stash.items) |*x| x.buf.free();
        b.stash.deinit(b.gpa);
        if (b.taps_stash) |*x| x.free();
        for (b.stages.items) |*st| st.deinit();
        b.stages.deinit(b.gpa);
        b.programs.deinit(b.gpa);
        b.taps_at.deinit(b.gpa);
        b.tab.deinit();
        b.gpa.free(b.pend);
        b.gpa.destroy(b);
    }

    /// The row programs' roles into the buffer plan (before Runner.bind): the stacked state sized for every slot.
    pub fn plan(f: *fwd.Forward, p: *buffers.Plan, nslots: u32, cap: u32) !void {
        const k = f.kv orelse return error.RowsNeedPool;
        // the row table is Batch's own buffer: Runner.bind leaves an external role alone (its address set in init)
        try f.runner.external(rowmode.table_role, 0);
        const a = f.arena.allocator();
        for (rowtab.buckets) |R| {
            if (R > cap) break;
            const cs = try block.emit(a, f.cfg, f.widths, rowOptions(f), try f.backbone(a), R, f.opts.limit - R, true);
            try p.add(try rowmode.transform(a, cs, R, .{ .slots = nslots, .rmax = cap, .pts = k.slot.max_pages }));
        }
    }

    pub fn pending(b: *const Batch, slot: u32) ?Pend {
        return b.pend[slot];
    }

    /// The slot's rows of the last window's taps (draft/iface.zig Taps); a stashed window's from the taps stash.
    pub fn tapsOf(b: *const Batch, slot: u32) ?struct { address: u64, rows: u32 } {
        const p = b.pend[slot] orelse return null;
        const row = tapsBytes(b.f);
        if (p.stashed) {
            const x = b.taps_stash orelse return null;
            return .{ .address = x.ptr + row * rowtab.seg_max * slot, .rows = p.n };
        }
        const base = b.f.runner.addressOf("w.taps") orelse return null;
        return .{ .address = base + row * p.a, .rows = p.n };
    }

    // -- the window -----------------------------------------------------------------------------------------------

    /// Every pending window kept whole before a one-slot operation or a new round; the leader sends op_rows_keep.
    pub fn settle(b: *Batch) !void {
        for (b.pend, 0..) |p, s| if (p) |x| try b.keep(@intCast(s), x.n - 1);
    }

    /// The leader's forward over `mix`; `more`: a later forward of the round (pending stashed), else keep all first.
    pub fn window(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        if (!more) try b.settle();
        return b.runRows(mix, ids, more);
    }

    fn runRows(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        const f = b.f;
        var tp = ph.nowNs();
        var pt = WinProf.start(b.prof);
        defer pt.end();
        try rowtab.check(mix, b.ss.n, @intCast(f.opts.limit));
        const n = rowtab.totalRows(mix);
        if (ids.len != n) return error.BadWindow;
        const R = rowtab.bucketRows(n, b.cap) orelse return error.RowsCap;
        try b.ss.deactivate();
        for (mix) |m| if (m.start != b.ss.states[m.slot].pos or b.pend[m.slot] != null) return error.NotAChain;
        if (b.ss.leads()) try b.send(mix, ids, more);
        if (more) try b.stashPending();
        pt.mark(.send);
        // every slot's pages for its positions, the changed table rows up before the launches
        const k = f.kv.?;
        for (mix) |m| try k.slots[m.slot].ensure(m.start + m.rows);
        try k.dev.syncTables();
        pt.mark(.tables);
        const s = b.settings;
        const key: graphs.Key = .{ .rows = R, .ctx = graphs.bucketOf(rowtab.lastPos(mix), s.bucket, s.grow) };
        const cs = try b.program(key, s);
        pt.mark(.program);
        const r = f.runner;
        b.lap(&tp, &b.stats.prep_ns);
        try b.tab.stage(r.stream, mix, ids, R, b.ss.n);
        pt.mark(.rowtab);
        try b.stageEngram(mix, ids, R);
        b.lap(&tp, &b.stats.stage_ns);
        defer b.lap(&tp, &b.stats.launch_ns);
        pt.mark(.engram);
        // Each slot owns its backup before any batched kernel writes its mutable state.
        for (mix) |m| {
            try b.ss.view(m.slot);
            try f.transaction.begin(f, m.rows);
        }
        try b.ss.deactivate();
        var a: u32 = 0;
        for (mix) |m| {
            b.pend[m.slot] = .{ .a = a, .start = m.start, .n = m.rows };
            @memcpy(b.pend[m.slot].?.ids[0..m.rows], ids[a..][0..m.rows]);
            b.ss.states[m.slot].pending = .{ .start = m.start, .n = m.rows };
            a += m.rows;
        }
        b.n = n;
        b.R = R;
        b.current = cs;
        // the forward's glue reads a pending window (its rows); the row path's own steps are handled here
        f.slot = .{ .pending = .{ .start = 0, .n = R } };
        defer f.slot = .{};
        r.glue = .{ .ctx = b, .run = glueFn };
        b.stats.windows += 1;
        b.stats.rows += n;
        b.stats.padded += R - n;
        r.rows = R; // l2pf: the window's rows (with its padding, the graph's key) past 16 skip the prefetch sites
        // the taps mark: armed for this window alone, so a key's graph holds it exactly when its program writes taps
        b.marked = false;
        if (r.taps_mark) |*m| {
            m.armed = true;
            b.marked = (b.taps_at.get(key) orelse null) != null;
        }
        defer if (r.taps_mark) |*m| {
            m.armed = false;
        };
        pt.mark(.bookkeep);
        if (b.g) |*g| {
            const fp = b.fingerprint();
            pt.mark(.fingerprint);
            // TF_DSV41_ROUND_GRAPH=split: the head's graph, then the tail's, launched while the GPU runs the head
            const at = if (b.rg.split) round_graph.splitAt(cs, b.rg.head) else null;
            const parts: [2][2]usize = if (at) |i| .{ .{ 0, i }, .{ i, cs.len } } else .{ .{ 0, cs.len }, .{ cs.len, cs.len } };
            for (parts, 0..) |sp, pi| {
                if (sp[0] == sp[1]) continue;
                b.span = sp;
                const pk: graphs.Key = .{ .rows = key.rows, .ctx = key.ctx, .part = if (at == null) 0 else @intCast(pi + 1) };
                const t0 = ph.nowNs();
                const out = try g.cache.run(pk, r.stream, fp, .{ .ctx = b, .run = body });
                pt.mark(if (pi == 0) .launch_head else .launch_tail);
                switch (out) {
                    .captured => {
                        const ns = ph.nowNs() - t0;
                        b.stats.captures += 1;
                        b.stats.capture_ns += ns;
                        if (graphLog()) std.log.scoped(.dsv41).info("row graphs: capture {d} ({d} rows, context bucket {d}, part {d}) in {d:.1} ms", .{ g.cache.stats.captured, pk.rows, pk.ctx, pk.part, @as(f64, @floatFromInt(ns)) / 1e6 });
                    },
                    .eager => b.stats.eager += 1,
                    .replayed => {},
                }
                if (outcomeLog(out, g)) std.log.scoped(.dsv41).info("row graphs: capture {d} ({d} rows, context bucket {d}, part {d}), {d} replayed", .{ g.cache.stats.captured, pk.rows, pk.ctx, pk.part, g.cache.stats.replayed });
            }
            if (at != null) b.stats.split += 1;
            return f.gatedEnd();
        }
        try r.timedWindow(cs);
        try f.gatedEnd();
    }

    const body = batch_graph.body;
    const program = batch_graph.program;

    /// Commits slot's pending window: its first row + `accepted` drafts (Forward.keep, by row).
    pub fn keep(b: *Batch, slot: u32, accepted: u32) !void {
        const p = b.pend[slot] orelse return error.BadWindow;
        var tk = ph.nowNs();
        defer b.lap(&tk, &b.stats.keep_ns);
        if (accepted >= p.n) return error.BadWindow;
        const f = b.f;
        if (b.sess) |x| try x.reserveAt(slot, p.ids[0 .. accepted + 1]);
        var pt = WinProf.start(b.prof);
        defer pt.endAs(.keep);
        if (b.ss.leads()) try f.link.?.send(&.{ slots_mod.op_rows_keep, slot, accepted });
        const r = f.runner;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(b.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const dst = b.ss.base(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            if (p.stashed) {
                const x = b.stashOf(L) orelse return error.Unbound;
                try r.kernels.others(r.stream).carry(x.buf.ptr + 2 * w * rowtab.seg_max * slot, w, 0, accepted, w, dst.base + slot * dst.bytes);
            } else {
                const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
                try r.kernels.others(r.stream).carry(src, w, 0, p.a + accepted, w, dst.base + slot * dst.bytes);
            }
        };
        try b.ss.transaction(slot).keep(accepted + 1);
        const st = b.ss.state(slot);
        if (b.sess) |x| try x.commitAt(slot, p.ids[0 .. accepted + 1]);
        if (f.engram) |h| tailAfter(st, h.max_ngram - 1, p.ids[0 .. accepted + 1]);
        st.pos = p.start + accepted + 1;
        st.pending = null;
        b.pend[slot] = null;
    }

    /// [op, n, (slot, start, rows, ids...) an ask]: the Engram warm (TF_DSV41_ENGRAM_WARM), every rank's own columns
    pub const op_engram_warm: i64 = 56;
    pub const max_warm = 16;
    pub const max_warm_ids = 16;

    pub const warmEngram = batch_engram.warmEngram;
    const parseWarm = batch_engram.parseWarm;
    pub const warmLocal = batch_engram.warmLocal;

    /// The slot's pending window dropped on every rank with no row committed (op_rows_drop): it stays put.
    pub fn drop(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.pend[slot] == null) return;
        if (b.ss.leads()) try b.f.link.?.send(&.{ slots_mod.op_rows_drop, slot });
        try b.dropPending(slot);
    }

    fn dropPending(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        const tx = b.ss.transaction(slot);
        if (tx.state.pending != null) try tx.rollback();
        b.pend[slot] = null;
        b.ss.state(slot).pending = null;
    }

    fn stashOf(b: *Batch, L: u32) ?*Stash {
        for (b.stash.items) |*x| if (x.layer == L) return x;
        return null;
    }

    /// The pending windows' ratio-2 projection rows copied aside on every rank, before a later forward overwrites them.
    fn stashPending(b: *Batch) !void {
        const r = b.f.runner;
        var nb: [64]u8 = undefined;
        const w = 2 * @as(usize, b.f.cfg.head_dim);
        for (b.pend, 0..) |*pp, s| if (pp.*) |*p| if (!p.stashed) {
            for (b.stash.items) |*x| {
                const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{x.layer})) orelse return error.Unbound;
                try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.buf.ptr + 2 * w * rowtab.seg_max * s, src + 2 * w * p.a, 2 * w * p.n, r.stream.handle), "cuMemcpyDtoDAsync");
            }
            if (b.taps_stash) |x| {
                const t = tapsBytes(b.f);
                const src = r.addressOf("w.taps") orelse return error.Unbound;
                try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.ptr + t * rowtab.seg_max * s, src + t * p.a, t * p.n, r.stream.handle), "cuMemcpyDtoDAsync");
            }
            p.stashed = true;
        };
    }

    /// A slot leaves (every rank, the leader's order): its pending window dropped, then Forward.release on it.
    pub fn release(b: *Batch, slot: u32) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.ss.leads()) try b.f.link.?.send(&.{ slots_mod.op_rows_drop, slot });
        try b.dropPending(slot);
        try b.ss.activate(slot);
        try b.f.release();
    }

    /// A slot leaves as a session turn: pending window kept, slot activated, entry saved, slot emptied (every rank).
    pub fn saveTurn(b: *Batch, slot: u32, ss: *sessions_gpu.Sessions) !void {
        if (slot >= b.ss.n) return error.Slots;
        if (b.pend[slot]) |p| try b.keep(slot, p.n - 1);
        try b.ss.activate(slot);
        _ = try ss.save();
    }

    // -- draft/iface.zig's Target over row windows ----------------------------------------------------------------

    pub const targetWindow = batch_target.targetWindow;
    pub const keepPath = batch_target.keepPath;
    pub const beginPrefill = batch_target.beginPrefill;

    fn send(b: *Batch, mix: []const rowtab.Seg, ids: []const u32, more: bool) !void {
        var msg: [2 + 3 * 16 + 64]i64 = undefined;
        if (mix.len > 16 or ids.len > 64) return error.BadWindow;
        msg[0] = slots_mod.op_rows;
        msg[1] = @as(i64, @intCast(mix.len)) | (@as(i64, @intFromBool(more)) << 8);
        for (mix, 0..) |m, i| {
            msg[2 + 3 * i] = m.slot;
            msg[3 + 3 * i] = @intCast(m.start);
            msg[4 + 3 * i] = m.rows;
        }
        const at = 2 + 3 * mix.len;
        for (ids, msg[at .. at + ids.len]) |t, *y| y.* = t;
        try b.f.link.?.send(msg[0 .. at + ids.len]);
    }

    /// A follower: the leader's op_rows / op_rows_keep / op_select (Forward.follow's extension).
    pub fn follow(b: *Batch, msg: []const i64) !void {
        switch (msg[0]) {
            slots_mod.op_select => try b.ss.activate(@intCast(msg[1])),
            slots_mod.op_rows_keep => try b.keep(@intCast(msg[1]), @intCast(msg[2])),
            slots_mod.op_rows_drop => try b.dropPending(@intCast(msg[1])),
            op_engram_warm => {
                var asks: [max_warm]iface.Warm = undefined;
                var ids: [max_warm * max_warm_ids]u32 = undefined;
                const n = try parseWarm(msg, &asks, &ids);
                b.warmLocal(asks[0..n]);
            },
            slots_mod.op_rows => {
                const k: usize = @intCast(msg[1] & 0xff);
                const more = (msg[1] >> 8) & 1 == 1;
                if (k == 0 or k > 16 or msg.len < 2 + 3 * k) return error.BadPlan;
                var mix: [16]rowtab.Seg = undefined;
                for (mix[0..k], 0..) |*m, i| m.* = .{ .slot = @intCast(msg[2 + 3 * i]), .start = @intCast(msg[3 + 3 * i]), .rows = @intCast(msg[4 + 3 * i]) };
                var ids: [64]u32 = undefined;
                const rest = msg[2 + 3 * k ..];
                if (rest.len > ids.len) return error.BadPlan;
                for (rest, ids[0..rest.len]) |t, *y| y.* = @intCast(t);
                try b.runRows(mix[0..k], ids[0..rest.len], more);
            },
            else => return error.BadPlan,
        }
    }

    pub fn ext(b: *Batch) fwd.Forward.Ext {
        return .{ .ptr = b, .run = followFn };
    }

    fn followFn(ptr: *anyopaque, msg: []const i64) anyerror!void {
        const b: *Batch = @ptrCast(@alignCast(ptr));
        return b.follow(msg);
    }

    // -- inputs and glue ------------------------------------------------------------------------------------------

    const stageEngram = batch_engram.stageEngram;
    const glueFn = batch_graph.glueFn;
    pub const exchange = batch_graph.exchange;

    /// Adds the time since `*t` to `*acc` and restarts `*t`.
    fn lap(_: *const Batch, t: *u64, acc: *u64) void {
        const now = ph.nowNs();
        acc.* += now - t.*;
        t.* = now;
    }

    const fingerprint = batch_graph.fingerprint;
};

/// The forward's options for a row-mode program (block.Options.rows: RowWin's selections).
pub fn rowOptions(f: *const fwd.Forward) block.Options {
    var o = f.opts;
    o.rows = true;
    return o;
}

fn tapsBytes(f: *const fwd.Forward) usize {
    return 2 * f.cfg.dspark_targets.items().len * @as(usize, f.cfg.hidden);
}

/// This rank's columns of an Engram row (its heads' share).
fn engramCols(f: *const fwd.Forward, h: *const eh.Host) usize {
    return h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep_n:].
fn tailAfter(st: *fwd.Slot, keep_n: usize, ids: []const u32) void {
    st.tail_len = eh.tailAfter(&st.tail, st.tail_len, keep_n, ids);
}

pub const Table = batch_graph.Table;

const Stash = struct { layer: u32, buf: cuda.DeviceBuffer };

const graphLog = batch_graph.graphLog;
const outcomeLog = batch_graph.outcomeLog;
const Graphs = batch_graph.Graphs;
pub const warmLookback = batch_engram.warmLookback;
pub const WinProf = batch_prof.WinProf;

test {
    _ = @import("batch_prof.zig");
}
