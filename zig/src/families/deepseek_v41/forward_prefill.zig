//! The forward's own prefill: a prompt in segments through block_prefill.zig's program, each committed as a keep.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const image_opts = @import("image_opts.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const ced = @import("ced.zig");
const ft = @import("forward_types.zig");
const graphs = @import("graphs.zig");
const serve_knobs = @import("serve_knobs.zig");
const pf_k4 = @import("forward_prefill_k4.zig");
const pf_finish = @import("forward_prefill_finish.zig");
const pf_glue = @import("forward_prefill_glue.zig");
const Forward = fwd.Forward;

pub const log = std.log.scoped(.dsv41);

pub const Error = error{ NoExpertTable, NoK2Table, BadGlue, StashGap };

/// The prefill's own state on the forward (forward.zig: `prefill_state`).
pub const State = struct {
    /// the persistent tables and grown buffers are in place (ready())
    ready: bool = false,
    /// the prompt's own copy (a caller's slice may alias the window's ids)
    ids: std.ArrayList(u32) = .empty,
    /// gm_plan's width-masked picks (int32 [P])
    masked: ?cuda.DeviceBuffer = null,
    /// the segment's layers and the MoE blocks seen: gm_ticket starts each layer's MoE, gm_plan reads its K2 table
    layers: []const u32 = &.{},
    moe: usize = 0,
    layer: u32 = 0,
    /// TF_DSV41_PREFETCH_AHEAD: the next segment's Engram reads issued before this one runs
    ahead: bool = false,
    /// long prompts' buffers (longpf.zig)
    long: longpf.State = .{},
    /// TF_DSV41_PF_TBO: micro-batch B's own long-prompt buffers, and the pair in flight
    long_b: longpf.State = .{},
    tbo: ?*Pair = null,
    /// TF_DSV41_PF_4K: 4,096-row segments in the prefill's own workspace (`K4`); the forward's plan stays 2,048 rows
    k4: bool = false,
    k4ws: K4 = .{},
    /// the prompt call may use the workspace (every rank's MemAvailable above the floor); false: 2,048-row segments
    ws_ok: bool = true,
    ws_fallbacks: u64 = 0,
    agreement: ?graphs.Agreement = null,
    /// the next prompt piece's ids (op 60): the call's last segment reads their Engram rows ahead; cleared by the call
    next: std.ArrayList(u32) = .empty,
    /// TF_DSV41_PREFILL: `.replay` = CED (segments are encoder passes, `finish` runs the decoder); `.full`: every layer
    mode: serve_knobs.PrefillMode = .full,
    /// the decoder replay's ids (the glue reads them through Forward.ids while it runs)
    tail_ids: [ced.keep]u32 = undefined,
    /// what "w.taps" holds after the prompt: positions [taps_at, taps_at + taps_rows) at rows 0 ..: the DSpark hand-off
    taps_at: u64 = 0,
    taps_rows: u32 = 0,

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        s.long.deinit(gpa);
        if (s.masked) |*b| b.free();
        s.ids.deinit(gpa);
        s.next.deinit(gpa);
        if (s.agreement) |*x| x.deinit();
        var it = s.k4ws.keep.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        s.k4ws.keep.deinit(gpa);
    }
};

/// The prefill's state on `f` (`Forward.prefill_state`, made by `attach` at engine open).
pub fn state(f: *const Forward) *State {
    return @ptrCast(@alignCast(f.prefill_state.?));
}

/// Engine open: the forward's prefill state, released by Forward.deinit (forward_hooks.Prefill.deinit).
pub fn attach(f: *Forward) !void {
    const s = try f.gpa.create(State);
    s.* = .{};
    f.prefill_state = s;
}

fn deinitFn(f: *Forward) void {
    const s = state(f);
    s.deinit(f.gpa);
    f.gpa.destroy(s);
    f.prefill_state = null;
}

fn aheadFn(f: *Forward) ?*std.ArrayList(u32) {
    const s = state(f);
    return if (s.ahead) &s.next else null;
}

fn replayingFn(f: *const Forward) bool {
    return state(f).mode == .replay;
}

fn planFn(f: *Forward, p: *buffers.Plan, rows: u32) anyerror!void {
    return plan(f, p, rows);
}

fn promptFn(f: *Forward, ids: []const u32) anyerror!void {
    return prompt(f, ids);
}

fn promptMultiFn(f: *Forward, segs: []const MultiSeg) anyerror!void {
    return promptMulti(f, segs);
}

fn finishFn(f: *Forward) anyerror!void {
    return finish(f);
}

fn finishMultiFn(f: *Forward, slots: []const u32, out: []Replayed) anyerror!void {
    return finishMulti(f, slots, out);
}

pub const hooks_table: @import("forward_hooks.zig").Prefill = .{ .plan = planFn, .prompt = promptFn, .promptMulti = promptMultiFn, .finish = finishFn, .finishMulti = finishMultiFn, .ahead = aheadFn, .replaying = replayingFn, .deinit = deinitFn };

/// Adds the prefill program's roles at `rows` (the largest segment) to `p`, at every start and segment size it uses.
pub fn plan(f: *Forward, p: *buffers.Plan, rows: u32) !void {
    if (rows <= 16) return error.BadWindow;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const layers = try f.backbone(la.allocator());
    try planPrograms(f, p, rows, layers);
    if (state(f).k4 or (state(f).mode == .replay and f.opts.pf_tbo)) {
        const ws = &state(f).k4ws;
        if (ws.plan == null) {
            ws.plan = .{ .a = f.gpa };
            try planWorkspace(f, &ws.plan.?, rows, layers);
            if (keepsShared(f)) {
                const before = planBytes(&ws.plan.?);
                try k4Trim(f.gpa, &ws.plan.?, p, &ws.keep);
                log.info("prefill 4K: rank {d}: {d} roles shared with the forward (TF_DSV41_PF_WS_SHARE): workspace {d} -> {d} MiB", .{ f.comm.rank(), ws.keep.count(), before >> 20, planBytes(&ws.plan.?) >> 20 });
            }
        }
    }
}

/// The forward's prefill programs at `rows` (`plan` without the workspace).
pub fn planPrograms(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const sizes = [_]u32{ rows, 17, 16 };
    for (sizes[if (rows > 17) 0 else 1..]) |n| {
        const top = try lastStart(f, layers, n, .whole);
        for ([_]i64{ 0, top }) |start| {
            var ea = std.heap.ArenaAllocator.init(f.gpa);
            defer ea.deinit();
            // replay's segments run the encoder (no head): its whole programs plan no head, w.logits not 2,048 rows
            const head = state(f).mode != .replay;
            try p.add(try block_prefill.emitPrefill(ea.allocator(), f.cfg, f.widths, image_opts.planOptions(f), layers, n, start, head));
        }
    }
    if (state(f).mode == .replay) try planReplay(f, p, rows, layers);
    // the multi-segment runs (TF_DSV41_PIECE_RUNS / _REPLAY_RUNS): their own roles and their rows' at `rows`
    var ra = std.heap.ArenaAllocator.init(f.gpa);
    defer ra.deinit();
    const replay = state(f).mode == .replay;
    const runs = try block_prefill.planRuns(ra.allocator(), f.cfg, f.widths, image_opts.planOptions(f), if (replay) try encoderLayers(f, layers) else layers, if (replay) try decoderLayers(f, layers) else null, rows, if (replay) .encoder else .whole);
    for (runs.items) |cs| try p.add(cs);
}

/// TF_DSV41_PF_WS_SHARE=1 with TF_DSV41_PF_4K alone: a 4K segment reuses forward roles that are already large enough.
pub fn keepsShared(f: *const Forward) bool {
    return state(f).k4 and !f.opts.pf_tbo and envOn("TF_DSV41_PF_WS_SHARE");
}

fn envOn(name: [:0]const u8) bool {
    const v = std.mem.span(std.c.getenv(name) orelse return false);
    return std.mem.eql(u8, v, "1");
}

pub const k4Trim = pf_k4.k4Trim;
const planWorkspace = pf_k4.planWorkspace;
pub const workspaceBytes = pf_k4.workspaceBytes;
const planBytes = pf_k4.planBytes;

/// CED replay's programs: an encoder segment and the decoder pass at each row count where its paths change.
fn planReplay(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const enc = try encoderLayers(f, layers);
    const dec = try decoderLayers(f, layers);
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    try p.add(try block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, image_opts.planOptions(f), enc, rows, 0, .encoder));
    for ([_]u32{ 16, 17, 32, 33, 64, 65, ced.keep }) |m| {
        if (m > rows) continue;
        const top = try lastStart(f, dec, m, .decoder);
        for ([_]i64{ 0, top }) |start| {
            var da = std.heap.ArenaAllocator.init(f.gpa);
            defer da.deinit();
            try p.add(try block_prefill.emitReplay(da.allocator(), f.cfg, f.widths, image_opts.planOptions(f), dec, m, start, .decoder));
        }
    }
}

/// The encoder pass's layers: the forward's up to the decoder's first (ced.decoderStart), inclusive.
pub fn encoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[0 .. i + 1];
}

/// The decoder pass's layers: the forward's from the decoder's first on.
pub fn decoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[i..];
}

/// Whether an `n`-row segment (`part` of it) at `start` is on the prefill path (block_prefill's emitter takes it).
fn emits(f: *Forward, o: block.Options, layers: []const u32, n: u32, start: i64, part: block_prefill.Part) !bool {
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    _ = block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, o, layers, n, start, part) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// The last start an `n`-row segment emits at; the decoder pass's is one lower.
fn lastStart(f: *Forward, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    return lastStartWith(f, f.opts, layers, n, part);
}

/// `lastStart` with options `o` (TF_DSV41_PF_4K's segments: k4Options).
pub fn lastStartWith(f: *Forward, o: block.Options, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    var hi: i64 = o.limit - n - @intFromBool(part == .decoder);
    if (hi < 0) return error.BadWindow;
    if (try emits(f, o, layers, n, hi, part)) return hi;
    if (!try emits(f, o, layers, n, 0, part)) {
        if (!quiet) log.warn("prefill: a {d}-row segment is not on the prefill path at any position (block_prefill: Unsupported)", .{n});
        return error.Unsupported;
    }
    var lo: i64 = 0;
    while (hi - lo > 1) {
        const mid = lo + @divFloor(hi - lo, 2);
        if (try emits(f, o, layers, n, mid, part)) lo = mid else hi = mid;
    }
    log.info("prefill: {d}-row segments emit up to start {d}", .{ n, lo });
    return lo;
}

pub const K4 = pf_k4.K4;
pub const segmentRows = pf_k4.segmentRows;
const wsAgreed = pf_k4.wsAgreed;
pub const k4Options = pf_k4.k4Options;
pub const wsPair = pf_k4.wsPair;
const wsArg = pf_k4.wsArg;
const k4Acquire = pf_k4.k4Acquire;
const nowMs = pf_k4.nowMs;
const wsFloorGiB = pf_k4.wsFloorGiB;
const hardFloorGiB = pf_k4.hardFloorGiB;
const k4Release = pf_k4.k4Release;

/// TF_DSV41_PF_TBO: one micro-batch's host side of a pair in flight: ids, window, Engram tail, MoE calls seen.
pub const Micro = struct {
    ids: []const u32,
    pending: @TypeOf(@as(fwd.Slot, undefined).pending),
    tail: Tail,
    moe: usize = 0,
    layer: u32 = 0,
};

pub const Pair = struct { a: Micro, b: Micro };

/// The prompt `ids` from the slot's position (forward.prefill: the leader has sent it to the followers).
pub fn prompt(f: *Forward, ids_in: []const u32) !void {
    const st = state(f);
    if (f.slot.pending != null) return error.BadWindow;
    if (ids_in.len == 0) return;
    if (f.slot.pos + ids_in.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
    if (ids_in.ptr != st.ids.items.ptr) {
        st.ids.clearRetainingCapacity();
        try st.ids.appendSlice(f.gpa, ids_in);
    }
    const ids = st.ids.items;
    st.taps_rows = 0;
    try ready(f);
    // TF_DSV41_PF_4K / _PF_TBO: the workspace for this call, decided on every rank alike (else 2,048-row segments)
    st.ws_ok = true;
    if (st.k4 or (st.mode == .replay and f.opts.pf_tbo)) st.ws_ok = try wsAgreed(f);
    const seg: usize = segmentRows(f);
    defer st.next.clearRetainingCapacity();
    var at: usize = 0;
    while (at < ids.len) {
        // TF_DSV41_PF_TBO: two segments at a time (the second at least tbo_min rows), else one
        if (pairOf(f, ids.len - at)) |pr| {
            const n = pr[0] + pr[1];
            if (st.ahead) ahead(f, ids, at, n, seg);
            if (try segmentPair(f, ids[at .. at + pr[0]], ids[at + pr[0] .. at + n])) {
                at += n;
                continue;
            }
        }
        const n = @min(seg, ids.len - at);
        if (st.ahead) ahead(f, ids, at, n, seg);
        try segment(f, ids[at .. at + n], at + n == ids.len and st.mode == .full);
        at += n;
    }
}

/// TF_DSV41_PF_TBO's split of the `left` rows ahead into two segments; null when off, full mode, or under tbo_min.
fn pairOf(f: *const Forward, left: usize) ?[2]usize {
    if (!f.opts.pf_tbo or state(f).mode != .replay or !state(f).ws_ok) return null;
    return pairSplit(segmentRows(f), left);
}

/// `pairOf`'s split of `left` rows at segments of `seg` rows (null: one segment at a time).
pub fn pairSplit(seg: usize, left: usize) ?[2]usize {
    const min: usize = @intCast(block_prefill.tbo_min);
    if (left >= 2 * seg) return .{ seg, seg };
    if (left < 2 * min) return null;
    const na = @min(seg, (left / 2) / 16 * 16);
    if (left - na > seg or na < min) return null;
    return .{ na, left - na };
}

/// Two encoder segments as one two-stream program, committed once; false: refused, run the segments one at a time.
fn segmentPair(f: *Forward, ia: []const u32, ib: []const u32) !bool {
    const st = state(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try encoderLayers(f, try f.backbone(a));
    const start = f.slot.pos;
    const na: u32 = @intCast(ia.len);
    const nb: u32 = @intCast(ib.len);
    // under TF_DSV41_PF_4K a micro-batch past the forward's rows is a 4K one (both in the workspace: wsPair)
    const big = st.k4 and @max(na, nb) > f.opts.prefill_rows;
    const o0 = image_opts.segmentOptions(f, ia.ptr[0 .. na + nb]);
    const o = if (big) k4Options(o0) else o0;
    const plain = block_prefill.emitTbo(a, f.cfg, f.widths, o, layers, na, nb, @intCast(start)) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    const cs = try wsPair(a, plain, big);
    try k4Acquire(f);
    if (f.kv) |kx| try kx.reserve(start + na + nb);
    // the micro-batches' host sides: B's Engram tail is the slot's after A's ids (tailOf, as A's commit leaves it)
    var pair: Pair = .{ .a = .{ .ids = ia, .pending = .{ .start = start, .n = na }, .tail = .{} }, .b = .{ .ids = ib, .pending = .{ .start = start + na, .n = nb }, .tail = .{} } };
    pair.a.tail.len = f.slot.tail_len;
    @memcpy(pair.a.tail.ids[0..pair.a.tail.len], f.slot.tail[0..f.slot.tail_len]);
    if (f.engram) |h| {
        var s2 = f.slot;
        tailOf(&s2, ia, h.max_ngram - 1);
        pair.b.tail.len = s2.tail_len;
        @memcpy(pair.b.tail.ids[0..s2.tail_len], s2.tail[0..s2.tail_len]);
    } else pair.b.tail = pair.a.tail;
    f.ids = ia;
    f.slot.pending = pair.a.pending;
    st.tbo = &pair;
    defer st.tbo = null;
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // the pair's send / gathered buffers are the workspace's while it runs (as a 4K segment's)
    std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
    std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    defer {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(na + nb), false);
    // the commit: both micro-batches' rows as one segment, the carries from B's last rows (its roles in the workspace)
    const both = ia.ptr[0 .. na + nb];
    f.ids = both;
    f.slot.pending = .{ .start = start, .n = na + nb };
    try commitRoles(f, nb, 0, K4.tag ++ "b~", na + nb);
    f.slot.ced.extend(start, both);
    f.last_rows = 0;
    st.taps_at = start;
    st.taps_rows = 0;
    return true;
}

/// Read-ahead of the Engram rows after segment ids[at .. at + n] (next segment or piece); a wrong guess costs a read.
fn ahead(f: *Forward, ids: []const u32, at: usize, n: usize, seg: usize) void {
    const st = state(f);
    if (at + n < ids.len) return readAhead(f, ids, at, n, ids[at + n .. at + n + @min(seg, ids.len - at - n)]);
    if (st.next.items.len > 0) readAhead(f, ids, at, n, st.next.items);
}

fn readAhead(f: *Forward, ids: []const u32, at: usize, n: usize, next: []const u32) void {
    const h = f.engram orelse return;
    if (!f.engram_prefetch) return;
    const src = (f.engramSource(h) catch return) orelse return;
    const keep: usize = h.max_ngram - 1;
    const b = at + n;
    var buf: [2 * eh.max_ngram]u32 = undefined;
    const lb: []const u32 = if (n >= keep) ids[b - keep .. b] else blk: {
        const old = @min(f.slot.tail_len, keep - n);
        @memcpy(buf[0..old], f.slot.tail[f.slot.tail_len - old .. f.slot.tail_len]);
        @memcpy(buf[old .. old + n], ids[at..b]);
        break :blk buf[0 .. old + n];
    };
    h.prefetch(f.gpa, src, next, lb, f.comm.rank(), f.comm.world());
}

/// One segment at the slot's position, run eagerly, then committed.
fn segment(f: *Forward, ids: []const u32, head: bool) !void {
    const st = state(f);
    const n: u32 = @intCast(ids.len);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    // CED replay: the encoder pass (layers up to the decoder's first, its site and compressor, the stash; no head)
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const start = f.slot.pos;
    if (f.kv) |kx| try kx.reserve(start + n);
    // TF_DSV41_PF_4K: a segment past the forward's rows runs in the workspace (its own roles: block_prefill.ownCalls)
    const big = st.k4 and st.ws_ok and n > f.opts.prefill_rows;
    const o = if (big) k4Options(image_opts.segmentOptions(f, ids)) else image_opts.segmentOptions(f, ids);
    const emitted = if (replay) block_prefill.emitReplay(a, f.cfg, f.widths, o, layers, n, @intCast(start), .encoder) else block_prefill.emitPrefill(a, f.cfg, f.widths, o, layers, n, @intCast(start), head);
    const plain = emitted catch |e| {
        if (e == error.Unsupported) if (!quiet) log.warn("prefill: the {d}-row segment at {d} is not on the prefill path (block_prefill: Unsupported)", .{ n, start });
        return e;
    };
    const cs = if (big) try block_prefill.ownCallsKeep(a, plain, K4.tag, &st.k4ws.keep) else plain;
    if (big) try k4Acquire(f);
    // the glue reads the ids (embedding, Engram) and the pending window (positions)
    f.ids = ids;
    f.slot.pending = .{ .start = start, .n = n };
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // TF_DSV41_PF_4K: the segment's send / gathered buffers are the workspace's while it runs
    if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    defer if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(n), false);
    if (big) try commitRoles(f, n, 0, K4.tag, n) else try commit(f, n);
    // replay.encode: the stash's ids follow its rows (a segment that does not continue it starts it over)
    if (replay) f.slot.ced.extend(start, ids);
    f.last_rows = if (replay) 0 else n;
    st.taps_at = start;
    st.taps_rows = if (replay) 0 else n;
}

pub const MultiSeg = ft.MultiSeg;

pub const Tail = ft.Tail;

pub const Multi = ft.Multi;

pub const promptMulti = pf_finish.promptMulti;
pub const runnable = pf_finish.runnable;
pub const Replayed = pf_finish.Replayed;
pub const replayRunnable = pf_finish.replayRunnable;
pub const finishMulti = pf_finish.finishMulti;
const enterRun = pf_finish.enterRun;
const segGlue = pf_glue.segGlue;
const runPositions = pf_glue.runPositions;
pub const finish = pf_finish.finish;
const stashCopy = pf_glue.stashCopy;
const commit = pf_finish.commit;
const commitAt = pf_finish.commitAt;
const commitRoles = pf_finish.commitRoles;
const tailOf = pf_finish.tailOf;
const ready = pf_glue.ready;
const grow = pf_glue.grow;
const copy = pf_glue.copy;
const zero = pf_glue.zero;
const numel = pf_glue.numel;
const bytesOf = pf_glue.bytesOf;
const dim = pf_glue.dim;
const ld = pf_glue.ld;
const glueFn = pf_glue.glueFn;
const glueOne = pf_glue.glueOne;
