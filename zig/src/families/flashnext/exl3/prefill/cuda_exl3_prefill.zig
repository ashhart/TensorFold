//! Expert-major prefill for an EXL3 Flash Next checkpoint whose routed experts live in host memory (the offload pool).

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const pk = @import("cuda_exl3_prefill_k.zig");
const prompt = @import("../../cuda_prompt.zig");
const window_ = @import("exl3_prefill_window.zig");
pub const state = @import("cuda_exl3_prefill_state.zig");
const chunks = @import("cuda_exl3_prefill_chunk.zig");
const ringmoe = @import("cuda_exl3_prefill_ringmoe.zig");
const heads = @import("cuda_exl3_prefill_head.zig");
const tiles = @import("cuda_exl3_prefill_tiles.zig");

const dims = wt.dims;
const wide = wt.wide;
const slots = wt.slots;
const vocab = wt.vocab;
const att_rows = batch.att_rows;

const Deep = state.Deep;
const HostLayout = state.HostLayout;
const Layout = state.Layout;
const Persistent = state.Persistent;
const Runtime = state.Runtime;
const chunk = chunks.chunk;
const selectWriteback = chunks.selectWriteback;
const twoChunk = chunks.twoChunk;
const ringOf = ringmoe.ringOf;
const warmTiles = tiles.warmTiles;

/// Frees the deep chunk's buffers (the engine's deinit, or a test).
pub fn release(runtime: *Runtime) void {
    if (runtime.deep) |dp| {
        @import("../graph/cuda_exl3_graph.zig").clearPrefillRuntime(@intFromPtr(runtime));
        if (dp.second) |second| {
            dp.aux.synchronize() catch @panic("two-chunk aux retirement synchronization failed");
            dp.dev.d.synchronize() catch @panic("two-chunk compute retirement synchronization failed");
            second.host.free();
            if (dp.red != null) {
                for (&second.red_ev) |*event| event.deinit();
                second.red_go.deinit();
            }
            if (second.sum_dev) |*buffer| buffer.free();
            second.dev.free();
            dp.gpa.destroy(second);
        }
        if (dp.ring) |r| r.deinit();
        if (dp.red) |*rs| {
            rs.synchronize() catch {};
            for (&dp.red_ev) |*ev| ev.deinit();
            dp.red_go.deinit();
            rs.deinit();
        }
        if (dp.unit_weights) |*buffer| buffer.free();
        dp.kern.unload();
        dp.aux.deinit();
        dp.host.free();
        dp.dev.free();
        dp.gpa.destroy(dp);
        runtime.deep = null;
    }
    if (runtime.agree_buf) |*b| b.free();
    runtime.agree_buf = null;
}

/// The server's sizing hook (cuda_lanes_load.zig, before the pools take the free memory): the deep chunk's buffers for
pub fn reserveFor(e: *prompt.Engine, context: usize, tp: usize) void {
    const runtime = &e.exl3_prefill;
    const o = runtime.opts();
    if (!o.enabled or (tp != 1 and tp != 2) or e.n_layers != wt.layers_n) return;
    for (e.layers[0..e.n_layers]) |*l| {
        const x = l.exl3 orelse return;
        if (x.bank == null) return;
        // expert parallel: every layer's experts carry this world's rank
        if ((if (x.tp) |r| r.world else 1) != tp) return;
    }
    const rows = @min(o.serve_rows, (context + att_rows - 1) / att_rows * att_rows);
    if (rows < o.min_rows) return;
    reserveAt(e.gpa, e.driver, runtime, e, rows, tp) catch |err| std.log.warn("flashnext exl3 prefill: deep chunks of {d} rows refused ({t}); prompts take the batched pass", .{ rows, err });
}

/// Rows a serving lane prefills a step (cuda_lanes.zig `step`): a deep chunk when this engine holds deep buffers.
pub fn stepRows(e: *const prompt.Engine, draft_rows: ?usize) usize {
    const runtime = &e.exl3_prefill;
    var n = e.chunk;
    if (runtime.deep) |dp| if (dp.owner == @as(*const anyopaque, e) and runtime.opts().enabled) {
        n = @max(n, stepCapacity(dp.rows, runtime.opts().two_chunk and dp.second != null));
    };
    if (draft_rows) |r| n = @min(n, @max(r, 1));
    return n;
}

pub fn stepCapacity(rows: usize, paired: bool) usize {
    return if (paired and rows <= window_.max_rows) rows * 2 else rows;
}

/// Rows of streams a lane's MTP draft keeps (cuda_lanes.zig, cuda_lanes_load.zig): a whole step's.
pub fn draftRows(e: *const prompt.Engine) usize {
    const runtime = &e.exl3_prefill;
    var n = stepRows(e, null);
    if (runtime.deep) |dp| if (dp.owner == @as(*const anyopaque, e)) {
        const o = runtime.opts();
        if (o.two_chunk and o.enabled and o.ring and !o.reduced_branch and dp.world == 2) {
            n = @max(n, stepCapacity(dp.rows, true));
        }
    };
    return n;
}

/// The engine's deinit: frees the deep chunk's buffers if `owner` made them.
pub fn releaseFor(runtime: *Runtime, owner: *const anyopaque) void {
    if (runtime.deep) |dp| if (dp.owner == owner) release(runtime);
}

/// Allocates the deep chunk's buffers for `rows` rows now (before the pool takes the free memory).
pub fn reserve(gpa: std.mem.Allocator, d: *const cuda.Driver, runtime: *Runtime, owner: *const anyopaque, rows_in: usize) !void {
    return reserveAt(gpa, d, runtime, owner, rows_in, 1);
}

/// `reserve` for a rank of `world` (expert parallel at 2: the ranks' sum, and per-slot outputs zeroed once so a slot
pub fn reserveAt(gpa: std.mem.Allocator, d: *const cuda.Driver, runtime: *Runtime, owner: *const anyopaque, rows_in: usize, world: usize) !void {
    const rows = rows_in / att_rows * att_rows;
    if (rows == 0 or (world != 1 and world != 2)) return error.Invalid;
    if (runtime.deep) |dp| {
        if (dp.owner == owner and dp.rows >= rows and dp.world == world) return;
        release(runtime);
    }
    const dp = try gpa.create(Deep);
    errdefer gpa.destroy(dp);
    dp.* = .{ .runtime = runtime, .gpa = gpa, .owner = owner, .rows = rows, .world = world, .lay = Layout.of(rows, world), .hl = HostLayout.of(rows), .dev = undefined, .host = undefined, .kern = undefined, .aux = undefined };
    dp.dev = try cuda.DeviceBuffer.alloc(d, dp.lay.total);
    errdefer dp.dev.free();
    dp.host = try cuda.HostBuffer.alloc(d, dp.hl.total);
    errdefer dp.host.free();
    dp.kern = try pk.Kernels.load(d);
    errdefer dp.kern.unload();
    dp.aux = try cuda.Stream.init(d, true);
    errdefer dp.aux.deinit();
    if (world > 1) {
        try d.check(d.api.cuMemsetD8Async(dp.dptr(dp.lay.y), 0, rows * slots * dims * 4, dp.aux.handle), "cuMemsetD8Async");
        try dp.aux.synchronize();
        var rs = try cuda.Stream.init(d, true);
        errdefer rs.deinit();
        var made: usize = 0;
        errdefer for (dp.red_ev[0..made]) |*ev| ev.deinit();
        for (&dp.red_ev) |*ev| {
            ev.* = try cuda.Event.init(d, false);
            made += 1;
        }
        dp.red_go = try cuda.Event.init(d, false);
        dp.red = rs;
    }
    runtime.deep = dp;
    std.log.info("flashnext exl3 prefill: deep chunks of {d} rows, {d:.2} GiB on the device, {d:.1} MiB pinned (world {d})", .{ rows, @as(f64, @floatFromInt(dp.lay.total)) / (1 << 30), @as(f64, @floatFromInt(dp.hl.total)) / (1 << 20), world });
}

pub fn reserveTwoChunk(optional: ?*Runtime, d: *const cuda.Driver, free: u64, protected: u64, graph_reservation: u64) !u64 {
    const runtime = optional orelse return 0;
    const o = runtime.opts();
    if (!o.two_chunk or !o.enabled or !o.ring or o.reduced_branch) return 0;
    const dp = runtime.deep orelse return 0;
    if (dp.dev.d != d or dp.world != 2 or dp.rows > window_.max_rows) return 0;
    if (dp.second != null) return 0;
    const graph_required = @import("../graph/cuda_exl3_graph_memory.zig").twoChunkReservationBytes();
    if (@import("../graph/cuda_exl3_graph.zig").enabled() and (graph_required == 0 or graph_reservation < graph_required)) {
        std.log.info("flashnext exl3 prefill: two-chunk window declined, incremental graph reservation unavailable (required={d} supplied={d})", .{ graph_required, graph_reservation });
        return 0;
    }
    const persistent = Persistent.of(dp.rows);
    const sum_bytes = if (dp.world > 1) state.sumBytes(dp.rows, dp.world) else 0;
    const device_bytes = try std.math.add(usize, persistent.total, sum_bytes);
    if (!window_.admits(free, protected, device_bytes)) {
        std.log.info("flashnext exl3 prefill: two-chunk window declined, extra_device={d} full_layout={d} extra_pinned=0 protected={d} free={d}", .{ device_bytes, dp.lay.total, protected, free });
        return 0;
    }
    const second = try dp.gpa.create(Deep);
    second.* = .{ .gpa = dp.gpa, .owner = dp.owner, .rows = dp.rows, .world = dp.world, .lay = dp.lay, .hl = dp.hl, .dev = undefined, .host = undefined, .kern = dp.kern, .aux = dp.aux, .red = dp.red, .red_ev = undefined, .red_go = undefined, .shared = dp };
    var device_allocated = false;
    var host_allocated = false;
    var made_events: usize = 0;
    var reduce_event_allocated = false;
    errdefer {
        if (reduce_event_allocated) second.red_go.deinit();
        for (second.red_ev[0..made_events]) |*event| event.deinit();
        if (host_allocated) second.host.free();
        if (second.sum_dev) |*buffer| buffer.free();
        if (device_allocated) second.dev.free();
        dp.gpa.destroy(second);
    }
    second.dev = try cuda.DeviceBuffer.alloc(d, persistent.total);
    device_allocated = true;
    if (sum_bytes != 0) {
        second.sum_dev = try cuda.DeviceBuffer.alloc(d, sum_bytes);
    }
    second.host = try cuda.HostBuffer.alloc(d, second.hl.total);
    host_allocated = true;
    if (dp.red != null) {
        for (&second.red_ev) |*event| {
            event.* = try cuda.Event.init(d, false);
            made_events += 1;
        }
        second.red_go = try cuda.Event.init(d, false);
        reduce_event_allocated = true;
    } else {
        second.red_ev = dp.red_ev;
        second.red_go = dp.red_go;
    }
    try d.check(d.api.cuMemsetD8Async(second.dptr(second.lay.y), 0, dp.rows * slots * dims * 4, dp.aux.handle), "cuMemsetD8Async");
    try dp.aux.synchronize();
    dp.second = second;
    dp.window_graph_reservation = graph_reservation;
    std.log.info("flashnext exl3 prefill: experimental two-chunk window, chunk_rows={d} hidden_bytes={d} full_layout={d} extra_device={d} extra_pinned={d}", .{ dp.rows, 2 * dp.rows * wide * 2, dp.lay.total, device_bytes, second.hl.total });
    return device_bytes;
}

pub fn twoChunkAllocated(optional: ?*const Runtime) bool {
    const runtime = optional orelse return false;
    return if (runtime.deep) |buffers| buffers.second != null else false;
}

fn applies(e: *prompt.Engine, seq: *prompt.Seq, ids: []const u32, hooks: batch.Hooks, o: Options) bool {
    if (!o.enabled or ids.len < o.min_rows) return false;
    if (hooks.keep != null or hooks.logits == .all) return false;
    if (seq.media != null or e.n_layers != wt.layers_n) return false;
    // expert parallel (cuda_exl3_tp.zig) at two ranks over the engine's collective.
    const world = worldOf(e);
    if (e.head_q.n != vocab and !(world == 2 and e.head_q.n == vocab / 2)) return false;
    for (e.layers[0..e.n_layers]) |*l| {
        const x = l.exl3 orelse return false;
        if ((if (x.tp) |r| r.world else 1) != world or x.needsPool()) return false;
        if (x.layer.act_mode != 1 and x.layer.act_mode != 0) return false;
    }
    return e.chunk >= att_rows;
}

fn worldOf(e: *const prompt.Engine) usize {
    return if (e.collective != null) 2 else 1;
}

/// Expert parallel: both ranks take the deep path or neither, with the same deep chunk (the smaller rank's): each
fn agree(e: *prompt.Engine, mine: usize) !usize {
    const runtime = &e.exl3_prefill;
    const c = e.collective orelse return mine;
    const st = e.stream.*;
    const rank = e.layers[0].exl3.?.tp.?.rank;
    if (runtime.agree_buf == null) runtime.agree_buf = try cuda.DeviceBuffer.alloc(e.driver, 256);
    const b = runtime.agree_buf.?;
    var v: [2]f32 = .{ 0, 0 };
    v[rank] = @floatFromInt(mine);
    try st.synchronize();
    try b.upload(0, std.mem.asBytes(&v));
    try c.sum(st, b.ptr, 2);
    try c.finish(st);
    try b.download(0, std.mem.asBytes(&v));
    if (v[0] < 1 or v[1] < 1) return 0;
    return @intFromFloat(@min(v[0], v[1]));
}
var tp_logged = false;

/// The hook (Engine.forward): runs `ids` from `seq.fed` and returns true, or returns false having done nothing.
pub fn forward(e: *prompt.Engine, comptime Tri: type, tri: Tri, seq: *prompt.Seq, ids: []const u32, hooks: batch.Hooks) !bool {
    const runtime = &e.exl3_prefill;
    const o = runtime.opts();
    if (!applies(e, seq, ids, hooks, o)) return false;
    try @import("../dplan/cuda_exl3_dplan.zig").toHost(&e.exl3_dplan); // the pools' residency on the host before planning a prompt
    if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
    const s = try e.reserve(@min(ids.len, e.chunk));
    const sub = s.cap / att_rows * att_rows;
    if (sub == 0) return false;
    // the deep chunk: as large as asked (whole attention blocks), no larger than the prompt needs; its sub-blocks are
    const rows = @max(att_rows, @min(o.rows, (ids.len + att_rows - 1) / att_rows * att_rows) / att_rows * att_rows);
    // the buffers reserved at load (reserveFor) are kept whatever their size; a first prompt without them allocates
    const world = worldOf(e);
    const have = if (runtime.deep) |d| d.owner == @as(*const anyopaque, e) and d.world == world else false;
    var room: usize = if (have) runtime.deep.?.rows else 0;
    if (!have) {
        if (reserveAt(e.gpa, e.driver, runtime, e, rows, world)) |_| {
            room = runtime.deep.?.rows;
        } else |err| std.log.warn("flashnext exl3 prefill: no room for deep chunks of {d} rows ({t}); the batched pass takes the prompt", .{ rows, err });
    }
    // both ranks the same way (a collective a rank skips would hang the other)
    room = try agree(e, room);
    if (room == 0) return false;
    if (world == 2 and !tp_logged) std.log.info("flashnext exl3 prefill: expert-parallel deep chunks of up to {d} rows (rank {d}: its own experts, one all-reduce a layer)", .{ room, e.layers[0].exl3.?.tp.?.rank });
    tp_logged = world == 2 or tp_logged;
    const dp = runtime.deep.?;
    try selectWriteback(dp, o.reduced_branch and world == 2);
    const scores = try e.reserveScores(seq.cap, s.cap);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = tri,
        .k = &e.batch_k,
        .s = s,
        .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram, .dplan = &e.exl3_dplan, .collective = e.collective },
        .seq = seq.view(scores),
        .hooks = hooks,
    };
    // equal deep chunks (a short last chunk would pay a whole chunk's expert copies for a few rows)
    const cap_rows = @min(rows, room);
    const n_chunks = (ids.len + cap_rows - 1) / cap_rows;
    const deep_rows = @min(cap_rows, ((ids.len + n_chunks - 1) / n_chunks + att_rows - 1) / att_rows * att_rows);
    const rg = if (o.ring) try ringOf(dp, e) else null;
    var window_room: usize = 0;
    if (o.two_chunk) {
        var supported = dp.second != null and rg != null and world == 2 and !dp.reduced_branch and deep_rows <= window_.max_rows and ids.len > deep_rows;
        if (@import("../graph/cuda_exl3_graph.zig").enabled()) {
            const required = @import("../graph/cuda_exl3_graph_memory.zig").twoChunkReservationBytes();
            if (required == 0 or dp.window_graph_reservation < required) supported = false;
        }
        window_room = try agree(e, if (supported) deep_rows else 0);
    }
    if (o.tiles and !o.slots_kernel) {
        try warmTiles(e);
        runtime.tiles_on = true;
    }
    defer runtime.tiles_on = false;
    if (rg) |r| try r.begin(p.stream);
    defer if (rg) |r| r.end(p.stream, o.reseed) catch |err| std.log.err("flashnext exl3 prefill: ring end: {t}", .{err});
    var at: usize = 0;
    while (at < ids.len) {
        if (window_room != 0) {
            const plan_ = try window_.Plan.init(ids.len - at, window_room, att_rows);
            try twoChunk(Tri, dp, &p, s, ids[at..][0..plan_.rows], seq.fed + at, sub, hooks, at, at + plan_.rows == ids.len, seq.logits, o, rg.?, plan_);
            at += plan_.rows;
            continue;
        }
        const n = @min(deep_rows, ids.len - at);
        const last = at + n == ids.len;
        const lg: batch.Logits = if (last) hooks.logits else .none;
        const st: u64 = if (hooks.streams != 0) hooks.streams + at * wide * 2 else 0;
        try chunk(Tri, dp, &p, s, ids[at..][0..n], seq.fed + at, sub, lg, st, if (last and hooks.logits != .none) seq.logits else null, o, rg);
        at += n;
    }
    if (hooks.logits == .none) try p.stream.synchronize();
    seq.fed += ids.len;
    return true;
}

pub const Options = state.Options;
pub const optionsFromEnv = state.optionsFromEnv;
pub const Stats = state.Stats;
pub const RingTelemetry = state.RingTelemetry;
pub fn ringTelemetry(e: *const prompt.Engine) ?RingTelemetry {
    return state.ringTelemetry(&e.exl3_prefill, e);
}
pub const deviceBytes = state.deviceBytes;
pub const deviceBytesAt = state.deviceBytesAt;
pub const head = heads.head;
pub const tilesPart = tiles.tilesPart;
