//! Two-rank expert parallel for an EXL3 Flash Next checkpoint (plan step 10 of docs/plans/exl3-on-upstream-1.1.0.md),
//! on upstream's rank engine (cuda_rank_engine.zig: one process a GPU, rank 1 replays rank 0's operations) and its
//! NCCL `Collective`.
//!
//! What a rank holds and runs:
//!  - routed experts by id (exl3_tp.zig `routed_set`): rank 0 ids 0..255, rank 1 ids 256..511. An EXL3 expert cannot be
//!    cut along its 640-wide intermediate (320 is not a multiple of the 128-wide Hadamard block), which is why
//!    upstream's ModelOpt `Cut` (intermediate halves) does not apply;
//!  - the shared expert (id 512) on rank 1, so each rank owns ONE contiguous id range: rank 0 [0, 256), rank 1
//!    [256, 513). Both ranks run the same kernel sequence (no rank branch in the pass: the other rank's slots are
//!    masked, not skipped by a different code path), so rank 1 replays rank 0's operations and a captured round graph
//!    has the same nodes on both. Upstream puts its shared expert on rank 0 behind an `if (rank == 0)` branch; here
//!    rank 0 already carries the MTP head and the sampled logits, so the shared expert's compute and its 2.9 MB a layer
//!    on rank 1 balance that;
//!  - every expert table keeps its 513 global ids (the offload pool's tables, `shared_id` 512 and its policy are
//!    unchanged); the stacked suh/svh scales hold only the rank's rows, their base pointers shifted by `lo` rows so the
//!    kernels' `scale[e * width]` still indexes by global id;
//!  - attention, GDN, hyper-connections, the router and KV are replicated, as upstream's ModelOpt tp does;
//!  - offloaded (the default): each rank's routed trellis in its own pinned bank (its 256 experts, about 21 GiB a
//!    process) and its own pool per layer (cuda_exl3_pass.attach sizes it from the rank's free memory), planning on the
//!    rank's masked picks: the same lines cover half the experts, so a rank's hit rate is higher than tp=1's;
//!  - the lm_head split by vocabulary (cuda_exl3_load.shardHead, default at tp=2): 124,160 columns a rank.
//!
//! The MoE branch (`moe`, the hook in cuda_exl3_pass.moe): the router and top-k run on both ranks (same rows, same
//! picks); `fn_exl3_tp_mask` turns the other rank's slots into pick -1, weight 0; `Experts.run` combines the rank's
//! slots in slot order (fp32 fmaf, as the write-back does) into [rows, 2560] fp32; ONE `Collective.sum` over that
//! (not upstream's per-slot [rows, 11, 2560]: 11 times less traffic); `fn_exl3_tp_place` writes the sum into slot 0 of
//! the pass's `y` with weights {1, 0, ...}, so the upstream write-back is unchanged and writes the sum's bf16.
//! tp=2 differs from tp=1 only in the fp32 grouping of the 11 terms: (rank 0's chain) + (rank 1's chain).
//! With two ranks the all-reduce is a single fp32 addition a + b, which is commutative, so both ranks get the same
//! bits whatever NCCL's algorithm (checked on the device by flashnext-exl3-tp).
//!
//! Masked slots and stale values: `Experts.run` leaves a skipped slot's per-slot output as it was in the scratch and
//! multiplies it by the slot's weight. The weight is 0 and the scratch is zeroed when allocated and only ever holds
//! finite expert outputs, so a skipped slot adds exactly 0.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_exl3_load.zig");
const memory = @import("cuda_exl3_memory.zig");
const projection = @import("cuda_projection.zig");
const cuts = @import("exl3_tp.zig");

const dims = wt.dims;
const slots = wt.slots;
const width = wt.moe_w;
/// Table entries: 512 routed experts and the shared one as id 512.
const table_n = wt.experts_n + 1;
/// The rank that holds the shared expert (id 512).
pub const shared_rank: usize = 1;
/// Rows an `Experts.run` call takes (cuda_exl3_pass.zig's block).
pub const moe_rows: usize = @import("cuda_exl3_pass.zig").moe_rows;

/// The ids [lo, hi) rank `rank` of `world` holds.
pub const Rank = struct {
    world: usize,
    rank: usize,
    lo: usize,
    hi: usize,

    pub fn init(world: usize, rank: usize) !Rank {
        if ((world != 1 and world != 2) or rank >= world) return error.BadRank;
        if (world == 1) return .{ .world = 1, .rank = 0, .lo = 0, .hi = table_n };
        const r = cuts.range(.routed_set, rank, world) catch return error.BadRank;
        return .{ .world = world, .rank = rank, .lo = r.lo, .hi = if (rank == shared_rank) r.hi + 1 else r.hi };
    }

    pub fn owns(r: Rank, id: usize) bool {
        return id >= r.lo and id < r.hi;
    }
};

// ---- loading -----------------------------------------------------------------------------------------------------

fn up(x: *load.Experts, d: *const cuda.Driver, bytes: []const u8) !u64 {
    try x.bufs.ensureUnusedCapacity(x.gpa, 1);
    const b = try cuda.DeviceBuffer.fromHost(d, bytes);
    x.bufs.appendAssumeCapacity(b);
    return b.ptr;
}

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

fn expertName(b: []u8, mlp: []const u8, e: usize, proj: []const u8) ![]const u8 {
    if (e == wt.experts_n) return std.fmt.bufPrint(b, "{s}.shared_expert.{s}", .{ mlp, proj });
    return std.fmt.bufPrint(b, "{s}.experts.{d}.{s}", .{ mlp, e, proj });
}

/// cuda_exl3_load.experts at world 2: the rank's experts, the 513-entry tables (other ids zero), the scales' rows
/// lo..hi with their bases shifted to global ids. Offloaded (the default, cuda_exl3_load.offloaded): only the rank's
/// own trellis words go to a pinned host bank (about half the tp=1 bank a process), the other rank's ids stay empty
/// in `routed` (the pool never stages them: their picks are masked to -1 before it plans), and the pool attached by
/// cuda_exl3_pass.attach owns the tables. Resident (TF_FLASHNEXT_EXL3_ROUTED=device): the rank's experts on the device.
pub fn experts(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, mlp: []const u8, world: usize, rank: usize) !*load.Experts {
    const r = try Rank.init(world, rank);
    const offload = load.offloaded();
    const x = try gpa.create(load.Experts);
    x.* = .{ .gpa = gpa, .ops = try load.opsOf(store, d), .layer = undefined, .tp = r };
    errdefer x.destroy();
    for (&x.trellis, &x.halves) |*t, *h| {
        t.* = try gpa.alloc(u64, table_n);
        @memset(t.*, 0);
        h.* = try gpa.alloc(i32, table_n);
        @memset(h.*, 0);
    }
    const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };
    const ks = [3]usize{ dims, dims, width };
    const ns = [3]usize{ width, width, dims };
    var scales: [6]std.ArrayList(u8) = @splat(.empty);
    defer for (&scales) |*l| l.deinit(gpa);
    var buf: [256]u8 = undefined;
    var at: usize = 0;
    if (offload) {
        // the bank's size from the headers of the rank's own experts, then one pinned region for the layer
        var total: usize = 0;
        for (r.lo..r.hi) |e| {
            for (projs) |proj| total += align256((try load.mat(store, try expertName(&buf, mlp, e, proj))).trellis.len);
            store.release();
        }
        x.bank = try cuda.HostBuffer.alloc(d, total);
        x.routed = try gpa.alloc([3][]const u8, wt.experts_n);
        for (x.routed) |*e| e.* = .{ &.{}, &.{}, &.{} };
    }
    const t0 = std.Io.Clock.awake.now(store.io).toNanoseconds();
    for (r.lo..r.hi) |e| {
        for (projs, 0..) |proj, j| {
            const m = try load.mat(store, try expertName(&buf, mlp, e, proj));
            if (m.k != ks[j] or m.n != ns[j]) return error.UnexpectedTensor;
            if (x.bank) |bank| {
                const dst = bank.bytes[at..][0..m.trellis.len];
                @memcpy(dst, m.trellis);
                at += align256(m.trellis.len);
                if (e == wt.experts_n) x.shared[j] = dst else x.routed[e][j] = dst;
            } else x.trellis[j][e] = try up(x, d, m.trellis);
            x.halves[j][e] = @intCast(m.halves);
            try scales[2 * j].appendSlice(gpa, m.suh);
            try scales[2 * j + 1].appendSlice(gpa, m.svh);
        }
        store.release();
    }
    x.bank_ns = @intCast(std.Io.Clock.awake.now(store.io).toNanoseconds() - t0);
    // a scale row's fp16 count: suh_g D, svh_g I, suh_u D, svh_u I, suh_d I, svh_d D
    const rows = [6]usize{ dims, width, dims, width, width, dims };
    var base: [6]u64 = undefined;
    for (&base, scales, rows) |*b, l, n| b.* = try up(x, d, l.items) - r.lo * n * 2;
    x.layer = .{
        // offloaded: the pool's tables replace these at attach (0 until then; a pass refuses an unattached layer)
        .gate = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.trellis[0])),
        .up = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.trellis[1])),
        .down = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.trellis[2])),
        .gate_k2 = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.halves[0])),
        .up_k2 = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.halves[1])),
        .down_k2 = if (offload) 0 else try up(x, d, std.mem.sliceAsBytes(x.halves[2])),
        .suh_g = base[0],
        .svh_g = base[1],
        .suh_u = base[2],
        .svh_u = base[3],
        .suh_d = base[4],
        .svh_d = base[5],
        .experts = table_n,
        .dims = dims,
        .width = width,
        .codebook = .mul1,
        .act_mode = 1,
    };
    return x;
}

// ---- the pass ----------------------------------------------------------------------------------------------------

/// The engine's expert-parallel scratch and kernels (held by the shared cuda_exl3_load.Ops, released with it).
pub const Scratch = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    mask_fn: cuda.Function,
    place_fn: cuda.Function,
    /// Masked picks (int32) and weights (fp32) [rows, 11]; the rank's combined branch, then the ranks' sum [rows, D].
    picks: ?cuda.DeviceBuffer = null,
    wts: ?cuda.DeviceBuffer = null,
    sum: ?cuda.DeviceBuffer = null,
    rows: usize = 0,
    /// Experts.run's scratch for `moe_cap` rows, zeroed when allocated (a skipped slot reads its stale, finite y).
    moe: ?cuda.DeviceBuffer = null,
    moe_cap: usize = 0,

    fn create(gpa: std.mem.Allocator, d: *const cuda.Driver) !*Scratch {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var m = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_tp);
        errdefer m.unload();
        const t = try gpa.create(Scratch);
        t.* = .{ .gpa = gpa, .d = d, .mod = m, .mask_fn = try m.function("fn_exl3_tp_mask"), .place_fn = try m.function("fn_exl3_tp_place") };
        return t;
    }

    pub fn destroy(t: *Scratch) void {
        inline for (.{ "picks", "wts", "sum", "moe" }) |f| if (@field(t, f)) |*b| b.free();
        t.mod.unload();
        t.gpa.destroy(t);
    }

    /// Room for `rows` rows; a growth waits for `s` (an earlier call may still read the old buffers).
    fn ensure(t: *Scratch, s: cuda.Stream, l: cuda.exl3_experts.Layer, rows: usize) !void {
        if (rows > t.rows) {
            try s.synchronize();
            inline for (.{ "picks", "wts", "sum" }) |f| if (@field(t, f)) |*b| {
                b.free();
                @field(t, f) = null;
            };
            t.rows = 0;
            t.picks = try cuda.DeviceBuffer.alloc(t.d, rows * slots * 4);
            t.wts = try cuda.DeviceBuffer.alloc(t.d, rows * slots * 4);
            t.sum = try cuda.DeviceBuffer.alloc(t.d, rows * dims * 4);
            t.rows = rows;
        }
        const cap = @min(rows, moe_rows);
        if (cap > t.moe_cap) {
            try s.synchronize();
            if (t.moe) |*b| b.free();
            t.moe = null;
            t.moe_cap = 0;
            const bytes = try cuda.exl3_experts.scratchBytes(l, cap, slots);
            t.moe = try cuda.DeviceBuffer.alloc(t.d, bytes);
            try s.d.check(s.d.api.cuMemsetD8Async(t.moe.?.ptr, 0, bytes, s.handle), "cuMemsetD8Async");
            t.moe_cap = cap;
        }
    }

    /// pick -> -1 and weight -> 0 outside [lo, hi), for `n` slots.
    pub fn mask(t: *const Scratch, s: cuda.Stream, picks: u64, wts: u64, mpicks: u64, mwts: u64, n: usize, r: Rank) !void {
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(wts);
        a.add(mpicks);
        a.add(mwts);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(r.lo)));
        a.add(@as(c_int, @intCast(r.hi)));
        try cuda.launch.launch(t.mask_fn, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The summed branch [rows, D] into y [rows, 11, D] slot 0 (others 0) and weights {1, 0, ...}.
    pub fn place(t: *const Scratch, s: cuda.Stream, sum: u64, y: u64, wts: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(sum);
        a.add(y);
        a.add(wts);
        a.add(@as(c_int, @intCast(dims)));
        a.add(@as(c_int, @intCast(slots)));
        try cuda.launch.launch(t.place_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(slots) }, .block = .{ .x = 256 } }, s, &a);
    }
};

/// The Ops' expert-parallel scratch, made on first use.
pub fn scratchOf(ops: *load.Ops) !*Scratch {
    if (ops.tp == null) ops.tp = try Scratch.create(ops.gpa, ops.d);
    return ops.tp.?;
}

/// Every expert-parallel scratch a pass of up to `rows` rows reads, allocated now (cuda_exl3_pass.attach, before the
/// pools size themselves from free memory), so no round grows a buffer (a growth synchronises the stream and moves an
/// address a captured graph would hold): the masked picks, weights and the ranks' sum for `rows` rows, Experts.run's
/// scratch for a block (zeroed), and the pinned picks of the pool's A/B cut.
pub fn reserve(ops: *load.Ops, s: cuda.Stream, l: cuda.exl3_experts.Layer, rows: usize) !void {
    const t = try scratchOf(ops);
    try t.ensure(s, l, rows);
    if (ops.picks == null) ops.picks = try cuda.HostBuffer.alloc(ops.d, moe_rows * slots * 4);
    try s.synchronize();
}

/// The expert-parallel buffers' addresses and sizes (cuda_exl3_load.Ops.fingerprint's tail).
pub fn fingerprint(t: *const Scratch) [4][2]u64 {
    var f: [4][2]u64 = @splat(.{ 0, 0 });
    for ([4]?cuda.DeviceBuffer{ t.picks, t.wts, t.sum, t.moe }, &f) |b, *e| if (b) |x| {
        e.* = .{ x.ptr, x.len };
    };
    return f;
}

/// Device bytes the expert-parallel scratch holds.
pub fn reservedBytes(t: *const Scratch) usize {
    var n: usize = 0;
    for ([4]?cuda.DeviceBuffer{ t.picks, t.wts, t.sum, t.moe }) |b| if (b) |x| {
        n += x.len;
    };
    return n;
}

/// The rank's half of the MoE branch, then the ranks' sum: cuda_exl3_pass.moe for a layer whose experts carry a rank.
/// `p` is the pass (`*cuda_batch.Pass(Tri)`); on return s.y / s.wts hold what the write-back reads.
pub fn moe(p: anytype, lw: anytype, x: *const load.Experts) !void {
    const r = x.tp orelse return error.UnexpectedTensor;
    const s = p.s;
    const n = p.rows;
    const st = p.stream;
    const c = p.m.collective orelse return error.TensorParallelUnavailable;
    try p.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
    try p.tri.topkMany(s.router_l, s.picks, s.wts, n);
    const ex = try x.ops.grouped();
    const t = try scratchOf(x.ops);
    try t.ensure(st, x.layer, n);
    const mp = t.picks.?.ptr;
    const mw = t.wts.?.ptr;
    const sum = t.sum.?.ptr;
    try t.mask(st, s.picks, s.wts, mp, mw, n * slots, r);
    if (x.needsPool()) return error.Exl3PoolNotAttached;
    const sc = try cuda.exl3_experts.scratch(t.moe.?.ptr, x.layer, t.moe_cap, slots);
    var r0: usize = 0;
    while (r0 < n) : (r0 += t.moe_cap) {
        const rows = @min(t.moe_cap, n - r0);
        const xin = s.mixed + r0 * dims * 2;
        // offloaded: the rank's own pool plans on the rank's masked picks (the other rank's slots are -1)
        if (x.pool) |pool| {
            const dplan = @import("cuda_exl3_dplan.zig");
            if (dplan.active) |dp| if (dp.indexOf(pool)) |i| { // a decode round planned on the device (no host cut)
                try dplan.runLayer(dp, st, i, ex, x.layer, xin, mp + r0 * slots * 4, rows, sc, mw + r0 * slots * 4, sum + r0 * dims * 4);
                continue;
            };
            try @import("cuda_exl3_pass.zig").pooled(st, x.ops, pool, ex, x.layer, xin, mp + r0 * slots * 4, rows, sc, mw + r0 * slots * 4, sum + r0 * dims * 4);
            continue;
        }
        try ex.run(st, x.layer, xin, dims, .bf16, mp + r0 * slots * 4, mw + r0 * slots * 4, rows, sum + r0 * dims * 4, sc);
    }
    try c.sum(st, sum, n * dims);
    try t.place(st, sum, s.y, s.wts, n);
}

// ---- memory ------------------------------------------------------------------------------------------------------

/// The expert id a tensor belongs to (routed N, or 512 for the shared expert), else null (replicated).
pub fn expertOf(name: []const u8) ?usize {
    if (std.mem.indexOf(u8, name, ".mlp.shared_expert.") != null) return wt.experts_n;
    const key = ".mlp.experts.";
    const at = std.mem.indexOf(u8, name, key) orelse return null;
    const rest = name[at + key.len ..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
}

/// One tensor's device bytes on rank `r`: memory.tensorBytes for what the rank holds, 0 for the other rank's experts.
pub fn tensorBytes(name: []const u8, e: core.safetensors.Entry, o: memory.Options, r: Rank) !u64 {
    if (expertOf(name)) |id| if (!r.owns(id)) return 0;
    // the vocabulary-split head (load.shardHead): half the trellis and svh a rank, suh whole
    if (r.world == 2 and load.shardHead(2) and (std.mem.eql(u8, name, "lm_head.trellis") or std.mem.eql(u8, name, "lm_head.svh")))
        return (try memory.tensorBytes(name, e, o)) / 2;
    return memory.tensorBytes(name, e, o);
}

/// Per layer: three 513-entry pointer tables (int64) and three width tables (int32), on every rank.
const tables_per_layer: u64 = 3 * table_n * 8 + 3 * table_n * 4;

/// The main model's device bytes on rank `r` (the MTP head is memory.headBytes, rank 0's).
pub fn deviceBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: memory.Options, r: Rank) !u64 {
    var store = try wt.Store.open(gpa, io, dir);
    defer store.deinit();
    const wm = store.parsed.value.object.get("weight_map") orelse return error.MissingTensor;
    var files: std.StringHashMapUnmanaged(void) = .empty;
    defer files.deinit(gpa);
    var it = wm.object.iterator();
    while (it.next()) |kv| try files.put(gpa, kv.value_ptr.string, {});
    var total: u64 = tables_per_layer * wt.layers_n;
    var fi = files.keyIterator();
    while (fi.next()) |rel| {
        const path = try std.fs.path.join(gpa, &.{ dir, rel.* });
        defer gpa.free(path);
        var file = try core.safetensors.File.open(gpa, io, path);
        defer file.close(io);
        var ni = file.names.iterator();
        while (ni.next()) |e| total = try std.math.add(u64, total, try tensorBytes(e.key_ptr.*, e.value_ptr.*, o, r));
    }
    return total;
}

/// cuda_exl3_memory.placement at world 2: the rank's experts, everything else replicated, the MTP head on rank 0
/// (cuda_lanes_load.zig loads it there only).
pub fn placement(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, world: usize, rank: usize, drafts: bool) !u64 {
    const r = try Rank.init(world, rank);
    const o = memory.options();
    const head = if (drafts and rank == 0) try memory.headBytes(gpa, io, dir, o) else 0;
    return try deviceBytes(gpa, io, dir, o, r) + head;
}

// ---- the head's vocabulary split ---------------------------------------------------------------------------------
//
// cuda_head.zig splits a BF16 lm_head by output rows and cuda_batch.zig projectHead joins the halves with
// ncclAllGather. An EXL3 lm_head [K 2560, N 248,320] splits along N at 128-column blocks (Hadamard), 124,160 columns
// a rank; Pass.lane() already runs an EXL3 Projection of n = vocab / 2, so projectHead needs no change. What is missing
// is the hook in cuda_prompt.Engine.initRank that reads `lm_head` as EXL3 at all (the serve stage's); with it,
// `headProjection(..., world, rank)` is that hook's tp=2 branch under TF_FLASHNEXT_SHARD_HEAD=1.

/// Rank `rank`'s output columns [lo, hi) of an `n`-column EXL3 matrix: whole 128-column blocks, equal shares.
pub fn headCut(n: usize, world: usize, rank: usize) ![2]usize {
    if ((world != 1 and world != 2) or rank >= world or n % (128 * world) != 0) return error.BadHeadSharding;
    const part = n / world;
    return .{ rank * part, (rank + 1) * part };
}

/// The trellis tiles of output columns [lo, hi), in the stored layout [K/16][N/16][tile words] made dense.
pub fn shardTrellis(gpa: std.mem.Allocator, m: load.Mat, lo: usize, hi: usize) ![]u8 {
    if (lo % 128 != 0 or hi % 128 != 0 or hi <= lo or hi > m.n) return error.BadHeadSharding;
    const tile = @as(usize, m.halves) * 16; // bytes a 16x16 tile: 256 values at halves/2 bits
    const row = (m.n / 16) * tile;
    if (m.trellis.len != (m.k / 16) * row) return error.UnexpectedTensor;
    const part = ((hi - lo) / 16) * tile;
    const out = try gpa.alloc(u8, (m.k / 16) * part);
    for (0..m.k / 16) |kt| @memcpy(out[kt * part ..][0..part], m.trellis[kt * row + (lo / 16) * tile ..][0..part]);
    return out;
}

/// The EXL3 head `prefix` (lm_head) as rank `rank`'s Projection of n / world columns (a dense shard on the device).
pub fn headProjection(gpa: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, world: usize, rank: usize) !projection.Projection {
    const m = try load.mat(store, prefix);
    const cut = try headCut(m.n, world, rank);
    const n = cut[1] - cut[0];
    const trellis = try shardTrellis(gpa, m, cut[0], cut[1]);
    defer gpa.free(trellis);
    const p = try gpa.create(load.Proj);
    p.* = .{ .gpa = gpa, .ops = try load.opsOf(store, d), .k = m.k, .n = n, .parts = &.{} };
    errdefer p.destroy();
    var bufs: [3]u64 = undefined;
    for (&bufs, [3][]const u8{ trellis, m.suh, m.svh[cut[0] * 2 .. cut[1] * 2] }) |*b, bytes| {
        try p.bufs.ensureUnusedCapacity(gpa, 1);
        const db = try cuda.DeviceBuffer.fromHost(d, bytes);
        p.bufs.appendAssumeCapacity(db);
        b.* = db.ptr;
    }
    const st = cuda.exl3.strides(m.k, n, m.halves, false);
    p.parts = try gpa.alloc(load.Part, 1);
    p.parts[0] = .{ .col = 0, .n = n, .kind = .{ .exl3 = .{ .trellis = bufs[0], .suh = bufs[1], .svh = bufs[2], .k = @intCast(m.k), .n = @intCast(n), .halves = m.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb } } };
    // the shard's Dense scratch (split partials, counters) is reserved with the rest at attach
    p.ops.noteWeight(p.parts[0].kind.exl3, n);
    store.release();
    var out = projection.Projection.empty(d);
    out.n = n;
    out.k = m.k;
    out.exl3 = p;
    return out;
}

// ---- host tests --------------------------------------------------------------------------------------------------

test "exl3: expert-parallel ranks own one contiguous id range each, the shared expert on rank 1" {
    const a = try Rank.init(2, 0);
    const b = try Rank.init(2, 1);
    try std.testing.expectEqual(@as(usize, 0), a.lo);
    try std.testing.expectEqual(@as(usize, 256), a.hi);
    try std.testing.expectEqual(a.hi, b.lo);
    try std.testing.expectEqual(@as(usize, 513), b.hi);
    for (0..table_n) |id| try std.testing.expect(a.owns(id) != b.owns(id)); // every id exactly once
    try std.testing.expect(b.owns(wt.experts_n) and !a.owns(wt.experts_n));
    const one = try Rank.init(1, 0);
    try std.testing.expect(one.owns(0) and one.owns(511) and one.owns(512));
    try std.testing.expectError(error.BadRank, Rank.init(2, 2));
    try std.testing.expectError(error.BadRank, Rank.init(4, 0));
}

test "exl3: expert tensors count on their own rank only; the router, the shared gate and HC on both" {
    try std.testing.expectEqual(@as(?usize, 7), expertOf("model.language_model.layers.0.mlp.experts.7.gate_proj.trellis"));
    try std.testing.expectEqual(@as(?usize, 511), expertOf("model.language_model.layers.47.mlp.experts.511.down_proj.svh"));
    try std.testing.expectEqual(@as(?usize, 512), expertOf("model.language_model.layers.3.mlp.shared_expert.up_proj.suh"));
    try std.testing.expectEqual(@as(?usize, null), expertOf("model.language_model.layers.3.mlp.shared_expert_gate.weight"));
    try std.testing.expectEqual(@as(?usize, null), expertOf("model.language_model.layers.3.mlp.gate.weight"));
    var e: core.safetensors.Entry = .{ .dtype = .f16, .rank = 1, .shape = .{ 2560, 1, 1, 1, 1 }, .begin = 0, .end = 5120 };
    const r0 = try Rank.init(2, 0);
    const r1 = try Rank.init(2, 1);
    const routed = "model.language_model.layers.0.mlp.experts.300.gate_proj.suh";
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes(routed, e, .{}, r0));
    try std.testing.expectEqual(@as(u64, 5120), try tensorBytes(routed, e, .{}, r1));
    const shared = "model.language_model.layers.0.mlp.shared_expert.gate_proj.suh";
    try std.testing.expectEqual(@as(u64, 0), try tensorBytes(shared, e, .{}, r0));
    try std.testing.expectEqual(@as(u64, 5120), try tensorBytes(shared, e, .{}, r1));
    e.dtype = .f16;
    e.rank = 2;
    e.shape = .{ 512, 2560, 1, 1, 1 };
    e.end = 512 * 2560 * 2;
    const router = "model.language_model.layers.0.mlp.gate.weight";
    try std.testing.expectEqual(try tensorBytes(router, e, .{}, r0), try tensorBytes(router, e, .{}, r1));
    try std.testing.expect(try tensorBytes(router, e, .{}, r0) > 0);
}

test "exl3: the split head counts half its trellis and svh a rank" {
    const r0 = try Rank.init(2, 0);
    const tr: core.safetensors.Entry = .{ .dtype = .i16, .rank = 3, .shape = .{ 160, 15520, 80, 1, 1 }, .begin = 0, .end = 160 * 15520 * 80 * 2 };
    if (load.shardHead(2)) try std.testing.expectEqual(tr.end / 2, try tensorBytes("lm_head.trellis", tr, .{}, r0));
    try std.testing.expectEqual(tr.end, try tensorBytes("lm_head.trellis", tr, .{}, try Rank.init(1, 0)));
}

test "exl3: tensor parallel placement opens at world 2 and still refuses other worlds" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = @embedFile("fixtures/exl3_config.json") });
    const dir = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(dir);
    // past the gate: no index in the fixture dir, so it fails reading the checkpoint, not at the quantization
    for ([_]usize{ 0, 1 }) |rank| {
        if (memory.placement(a, std.testing.io, dir, 2, rank, rank == 0)) |_| return error.TestUnexpectedResult else |err| try std.testing.expect(err != error.UnsupportedQuantization and err != error.BadRank);
    }
    try std.testing.expectError(error.BadRank, memory.placement(a, std.testing.io, dir, 3, 0, false));
    try std.testing.expectError(error.BadRank, memory.placement(a, std.testing.io, dir, 2, 2, false));
}

test "exl3: the head splits at 128-column blocks and a shard's tiles are the full matrix's columns" {
    try std.testing.expectEqual([2]usize{ 0, 124_160 }, try headCut(248_320, 2, 0));
    try std.testing.expectEqual([2]usize{ 124_160, 248_320 }, try headCut(248_320, 2, 1));
    try std.testing.expectError(error.BadHeadSharding, headCut(248_320 + 128, 2, 0));
    try std.testing.expectError(error.BadHeadSharding, headCut(256, 2, 2));
    // K 32 (2 tile rows), N 512 (32 tiles a row), 3 halves: 48 bytes a tile
    const k = 32;
    const n = 512;
    const halves = 3;
    const tile = halves * 16;
    var bytes: [(k / 16) * (n / 16) * tile]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i * 7 + i / 255);
    const m: load.Mat = .{ .trellis = &bytes, .suh = &.{}, .svh = &.{}, .k = k, .n = n, .halves = halves };
    const a = std.testing.allocator;
    for (0..2) |rank| {
        const cut = try headCut(n, 2, rank);
        const got = try shardTrellis(a, m, cut[0], cut[1]);
        defer a.free(got);
        try std.testing.expectEqual(bytes.len / 2, got.len);
        for (0..k / 16) |kt| for (0..(cut[1] - cut[0]) / 16) |j| {
            const src = (kt * (n / 16) + cut[0] / 16 + j) * tile;
            const dst = (kt * ((cut[1] - cut[0]) / 16) + j) * tile;
            try std.testing.expectEqualSlices(u8, bytes[src..][0..tile], got[dst..][0..tile]);
        };
    }
    try std.testing.expectError(error.BadHeadSharding, shardTrellis(a, m, 0, 100));
}
