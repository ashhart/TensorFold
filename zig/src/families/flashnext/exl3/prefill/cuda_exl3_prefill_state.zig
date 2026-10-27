//! Deep-prefill state: options, stats, the chunk's device and pinned layouts, and the buffers one engine owns.

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("../../cuda_batch.zig");
const wt = @import("../../cuda_prompt_wt.zig");
const pk = @import("cuda_exl3_prefill_k.zig");
const ring_ = @import("cuda_exl3_prefill_ring.zig");
const history_ = @import("cuda_exl3_prefill_history.zig");
const window_ = @import("exl3_prefill_window.zig");

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const table_n = wt.experts_n + 1;
const att_rows = batch.att_rows;
const routed_guess: usize = 400;
const pss_n = batch.pss_n;

pub const Options = struct {
    enabled: bool = true,
    /// Rows a deep chunk holds (whole attention blocks).
    rows: usize = 8192,
    /// The deep chunk the server reserves at load (reserveFor): its buffers come out of the pools' share of the card
    serve_rows: usize = 4096,
    /// The shortest prompt that takes this path.
    min_rows: usize = 512,
    /// New experts a pass stages at most, as a fraction of the layer's lines (0.5: double buffering).
    pass_lines_half: bool = true,
    /// Stage the next layer's first pass while this layer's blocks compute.
    prefetch: bool = true,
    /// Run each pass through Experts.runSlots (256-row blocks over masked picks) instead of the 64-member prefill
    slots_kernel: bool = false,
    /// Every decoder layer's pool lines as one ring the copy engine streams the coming layers' experts into
    ring: bool = true,
    /// Experts a ring pass computes at most.
    ring_pass: usize = 32,
    /// Layers ahead the ring prefetches (the guess: the hottest experts, as many as the current layer used).
    ring_ahead: usize = 1,
    /// Dense EXL3 projections of a deep chunk's sub-blocks on #806's prompt tiles (Dense.prompt, .tiles) instead of
    tiles: bool = true,
    /// After a ring prefill, refill every pool with its layer's hottest experts of the prefills (measured: decode hit
    reseed: bool = false,
    reduced_branch: bool = false,
    two_chunk: bool = false,
};

pub fn optionsFromEnv() Options {
    var o: Options = .{};
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL")) |v| o.enabled = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_ROWS")) |v| {
        o.rows = std.fmt.parseInt(usize, std.mem.span(v), 10) catch o.rows;
        o.serve_rows = o.rows;
    }
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH")) |v| o.reduced_branch = std.mem.eql(u8, std.mem.span(v), "1");
    o.two_chunk = window_.enabled(if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK")) |value| std.mem.span(value) else null);
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_RING")) |v| o.ring = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD")) |v| o.ring_ahead = std.fmt.parseInt(usize, std.mem.span(v), 10) catch o.ring_ahead;
    return o;
}

/// The options every call reads (a test may change them between calls).
pub var options: ?Options = null;

pub fn opts() Options {
    if (options == null) options = optionsFromEnv();
    return options.?;
}

/// What the last deep forward did (read by tf-cuda-test exl3-prefill engine).
pub const Stats = struct {
    chunks: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    copies: u64 = 0,
    prefetched: u64 = 0,
    members: u64 = 0,
    host_wait_ns: u64 = 0, // the picks' D2H and its wait, every layer
    plan_ns: u64 = 0, // host plan of the passes (members, items, pool rounds)
    reduces: u64 = 0, // expert parallel: the ranks' all-reduces (one a layer a deep chunk)
    reduce_ns: u64 = 0, // their combine + all-reduce + place, measured only with TF_FLASHNEXT_PROFILE=1 (a wait each)
    head_rows: u64 = 0, // MTP head rows run expert-major (`head`)
};
pub var stats: Stats = .{};

/// Device bytes of a deep chunk's own rows: streams, sums, gates, mixed rows, slot outputs, picks, n-gram rows, and
pub fn deviceBytes(rows: usize) usize {
    return Layout.of(rows, 1).total;
}

/// The same at `world` ranks (expert parallel adds the ranks' sum, rows x 2560 fp32).
pub fn deviceBytesAt(rows: usize, world: usize) usize {
    return Layout.of(rows, world).total;
}

pub const Layout = struct {
    h: usize,
    pss: usize,
    inj: usize,
    mixed: usize,
    xsm: usize,
    y: usize,
    wts: usize,
    picks: usize,
    ple_emb: usize,
    ple_xs: usize,
    ids: usize,
    pos: usize,
    members: usize,
    items: usize,
    kscratch: usize,
    /// Expert parallel only: a rank's combined branch, then the ranks' sum [rows, D] fp32 (0 bytes at world 1).
    sum: usize,
    total: usize,
    member_cap: usize,

    pub fn of(rows: usize, world: usize) Layout {
        var l: Layout = undefined;
        var at: usize = 0;
        const take = struct {
            fn f(a: *usize, n: usize) usize {
                const r = a.*;
                a.* = r + (n + 255) / 256 * 256;
                return r;
            }
        }.f;
        const pairs = rows * slots;
        l.member_cap = 2 * rows;
        l.h = take(&at, rows * wide * 2);
        l.pss = take(&at, rows * pss_n * 4);
        l.inj = take(&at, rows * streams * 2);
        l.mixed = take(&at, rows * dims * 2);
        l.xsm = take(&at, rows * (dims / 32) * 4);
        l.y = take(&at, pairs * dims * 4);
        l.wts = take(&at, pairs * 4);
        l.picks = take(&at, pairs * 4);
        l.ple_emb = take(&at, rows * dims * 2);
        l.ple_xs = take(&at, rows * (dims / 32) * 4);
        l.ids = take(&at, rows * 4);
        l.pos = take(&at, (rows / att_rows + 1) * 16);
        l.members = take(&at, pairs * 4);
        l.items = take(&at, 3 * pk.maxItems(pairs, table_n) * 4);
        l.kscratch = take(&at, pk.scratchBytes(l.member_cap, dims, wt.moe_w));
        l.sum = take(&at, if (world > 1) rows * dims * 4 else 0);
        l.total = at;
        return l;
    }
};

/// Host side: pinned uploads (ids, positions, n-gram rows, members and items) and the picks' download.
pub const HostLayout = struct {
    ids: usize,
    pos: usize,
    pos_stride: usize,
    emb: usize,
    xs: usize,
    picks: usize,
    members: usize,
    items: usize,
    total: usize,

    pub fn of(rows: usize) HostLayout {
        var l: HostLayout = undefined;
        var at: usize = 0;
        const take = struct {
            fn f(a: *usize, n: usize) usize {
                const r = a.*;
                a.* = r + (n + 63) / 64 * 64;
                return r;
            }
        }.f;
        const pairs = rows * slots;
        l.ids = take(&at, rows * 4);
        l.pos_stride = (rows / att_rows + 1) * 16;
        l.pos = take(&at, l.pos_stride * (wt.layers_n + 1));
        l.emb = take(&at, rows * dims * 2);
        l.xs = take(&at, rows * (dims / 32) * 4);
        l.picks = take(&at, pairs * 4);
        l.members = take(&at, pairs * 4);
        l.items = take(&at, 3 * pk.maxItems(pairs, table_n) * 4);
        l.total = at;
        return l;
    }
};

/// One pass of a layer: its experts (the pool round's routed ids), its members and items.
pub const Pass = struct { id_lo: u32, id_hi: u32, m_lo: u32, m_n: u32, i_lo: u32, i_n: u32 };

/// Sub-blocks a deep chunk holds at most (chunk's view array).
pub const max_sub = 64;

pub const Persistent = struct {
    h: usize,
    pss: usize,
    inj: usize,
    y: usize,
    wts: usize,
    total: usize,

    pub fn of(rows: usize) Persistent {
        const h = rows * wide * 2;
        const pss = std.mem.alignForward(usize, rows * pss_n * 4, 256);
        const inj = std.mem.alignForward(usize, rows * streams * 2, 256);
        const y = rows * slots * dims * 4;
        const wts = std.mem.alignForward(usize, rows * slots * 4, 256);
        return .{ .h = 0, .pss = h, .inj = h + pss, .y = h + pss + inj, .wts = h + pss + inj + y, .total = h + pss + inj + y + wts };
    }
};

pub const Deep = struct {
    gpa: std.mem.Allocator,
    owner: *const anyopaque,
    rows: usize,
    world: usize = 1,
    lay: Layout,
    hl: HostLayout,
    dev: cuda.DeviceBuffer,
    sum_dev: ?cuda.DeviceBuffer = null,
    host: cuda.HostBuffer,
    kern: pk.Kernels,
    aux: cuda.Stream,
    /// Expert parallel: the ranks' all-reduces run on `red`, one a sub-block.
    red: ?cuda.Stream = null,
    red_ev: [max_sub]cuda.Event = undefined,
    red_go: cuda.Event = undefined,
    red_n: usize = 0,
    reduced_branch: bool = false,
    unit_weights: ?cuda.DeviceBuffer = null,
    /// The layer's pass plan: expert ids in pass order, the passes, member counts and starts.
    order: [table_n]u16 = undefined,
    passes: [table_n + 1]Pass = undefined,
    n_passes: usize = 0,
    cnt: [table_n]u32 = undefined,
    start: [table_n]u32 = undefined,
    /// Experts the previous layer's prefetch staged for each layer (held at its MoE).
    fake: [48 * slots]i32 = undefined,
    /// The prefill line ring over the pools (made at the first prefill, once the pools exist).
    ring: ?*ring_.Ring = null,
    ring_failed: bool = false,
    /// Experts each layer used in the current chunk (the next layers' prefetch size).
    used: [wt.layers_n]usize = @splat(routed_guess),
    history: [wt.layers_n]history_.Slot = @splat(.{}),
    history_shape: ?history_.Shape = null,
    second: ?*Deep = null,
    shared: ?*const Deep = null,

    pub fn dptr(dp: *const Deep, off: usize) u64 {
        if (dp.shared) |source| {
            const persistent = Persistent.of(dp.rows);
            inline for (.{ "h", "pss", "inj", "y", "wts" }) |field| {
                if (off == @field(dp.lay, field)) return dp.dev.ptr + @field(persistent, field);
            }
            if (off == dp.lay.sum) if (dp.sum_dev) |buffer| return buffer.ptr;
            return source.dptr(off);
        }
        return dp.dev.ptr + off;
    }
    pub fn hbytes(dp: *const Deep, off: usize, n: usize) []u8 {
        return dp.host.bytes[off..][0..n];
    }
};

pub var deep: ?*Deep = null;

pub const RingTelemetry = struct { owner: u64, copies: u64, dma_requested_bytes: u64, unused_copies: u64, hit_picks: u64, miss_picks: u64, hit_expert_requests: u64, miss_expert_requests: u64, attribution: ?@import("cuda_exl3_prefill_attribution.zig").Totals };

pub fn ringTelemetry(owner: *const anyopaque) ?RingTelemetry {
    const buffers = deep orelse return null;
    if (buffers.owner != owner) return null;
    const ring = buffers.ring orelse return null;
    return .{ .owner = ring.telemetry_id, .copies = ring.stats.copies, .dma_requested_bytes = ring.stats.dma_requested_bytes, .unused_copies = ring.stats.wasted, .hit_picks = ring.stats.hit_picks, .miss_picks = ring.stats.miss_picks, .hit_expert_requests = ring.stats.hit_expert_requests, .miss_expert_requests = ring.stats.miss_expert_requests, .attribution = if (ring.attribution) |state| state.totals else null };
}

pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn upload(st: cuda.Stream, dst: u64, bytes: []const u8) !void {
    try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, st.handle), "cuMemcpyHtoDAsync");
}

pub fn profile() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROFILE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
