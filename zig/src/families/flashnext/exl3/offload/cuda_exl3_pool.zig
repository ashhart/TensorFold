//! One layer's EXL3 routed-expert offload pool on upstream's `Experts` (plan step 6 of

const std = @import("std");
const cuda = @import("cuda");
const round_ = @import("../model/exl3_round.zig");
const exl3e = cuda.exl3_experts;
const Lines = cuda.exl3_pool.Lines;
const telemetry = @import("../model/exl3_telemetry.zig");

pub const offload = @import("exl3_offload.zig");
pub const round = round_;
pub const Round = round_.Round;
pub const linesFor = round_.linesFor;
pub const shape = offload.shape;
pub const shared_id = offload.shared_id;
/// The fewest lines a layer may have: one row's ten routed picks always fit one pass.
pub const min_lines: usize = 10;

test {
    _ = @import("exl3_frequency_seed.zig");
}

/// The layer's experts in host memory: routed [512] gate/up/down trellis bytes (stored layout), the shared expert's.
pub const Source = struct {
    routed: []const [3][]const u8,
    shared: ?[3][]const u8 = null,
    pinned_bank: ?[]const u8 = null,
};

/// Device addresses of the resident, stacked fp16 scales ([E, D] or [E, I], E = 513).
pub const Scales = struct { suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64 };

pub const Options = struct {
    layer: u8,
    lines: usize,
    dims: usize = 2560,
    width: usize = 640,
    /// Rows and slots a round may hold (sizes the masked-pick buffers).
    max_rows: usize = 16,
    slots: usize = 11,
    staging: usize = 8,
    /// Copy from the source bytes directly (they are pinned already); otherwise through pinned staging lines.
    direct: bool = false,
    codebook: cuda.exl3.Codebook = .mul1,
    act_mode: i32 = 1,
    swap_every: u16 = 4,
    swap_cap: u16 = 96,
    /// Bytes a swap round may prefetch (the copy budget); 0: no prefetch.
    prefetch_bytes: u64 = 0,
    /// Double buffering: a round that fits without them does not evict the previous round's lines.
    double_buffer: bool = true,
    /// Masked-pick buffers for this many passes of a split round (0: the round's own bound, uniq + 1).
    max_passes: usize = 0,
};

/// Passes a round of a `lines`-line pool can split into when nothing prefetches (`prefetch_bytes` 0): pass 0 and one
pub fn passesFor(lines: usize) usize {
    return 2 + (shape.experts + lines - 1) / lines;
}

pub const Stats = struct {
    rounds: u64 = 0,
    passes: u64 = 0,
    hit_slots: u64 = 0,
    miss_slots: u64 = 0,
    unique: u64 = 0,
    misses: u64 = 0, // mandatory copies
    prefetches: u64 = 0,
    evictions: u64 = 0,
    busy: u64 = 0, // plans that had to drain in-flight copies first
    protected: u64 = 0, // rounds that spared the previous round's lines
};

pub const Pool = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    opt: Options,
    src: Source,
    scales: Scales,
    lines: Lines,
    k2: [][3]i32, // per routed expert, from its trellis sizes
    slots_tab: []u16,
    line_tab: []offload.Line,
    stamp_tab: []u32,
    layer_tab: []u8,
    v: offload.Vram,
    counts: *[shape.layers][shape.experts]u32,
    own_counts: bool,
    round: Round,
    now: u32 = 0,
    round_no: u32 = 0,
    picks_host: cuda.HostBuffer,
    picks_dev: cuda.DeviceBuffer,
    picks_ev: cuda.Event,
    picks_pending: bool = false,
    picks_n: usize = 0, // the planned round's picks (rows * slots)
    prev: []u16, // the previous round's routed experts (double buffering)
    prev_n: usize = 0,
    stats: Stats = .{},
    telemetry_id: u64 = 0,
    telemetry_valid: bool = true,
    telemetry_device: telemetry.Device = .{},

    /// `counts`: the family's lifetime use table shared by every layer's pool (null: the pool owns one).
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, src: Source, scales: Scales, opt: Options, counts: ?*[shape.layers][shape.experts]u32) !Pool {
        if (opt.lines < min_lines) return error.PoolTooSmall;
        if (src.routed.len != shape.experts) return error.Invalid;
        const tiles = (opt.dims / 16) * (opt.width / 16);
        var bytes: [3]usize = .{ 0, 0, 0 };
        const k2 = try gpa.alloc([3]i32, shape.experts);
        errdefer gpa.free(k2);
        var held: usize = 0;
        for (src.routed, k2) |e, *w| {
            // an expert another rank holds (expert parallel, cuda_exl3_tp.zig): no bytes, never picked here
            if (e[0].len == 0 and e[1].len == 0 and e[2].len == 0) {
                w.* = .{ 0, 0, 0 };
                continue;
            }
            held += 1;
            for (0..3) |p| {
                // trellis bytes = tiles * 16 * k2 (k2 half-bits a value, 8 * k2 int16 words a 16x16 tile)
                if (e[p].len == 0 or e[p].len % (tiles * 16) != 0) return error.Invalid;
                w[p] = @intCast(e[p].len / (tiles * 16));
                bytes[p] = @max(bytes[p], e[p].len);
            }
        }
        if (held == 0) return error.Invalid;
        const table_e: usize = if (src.shared != null) shape.experts + 1 else shape.experts;
        var lines = try Lines.init(gpa, d, .{ .experts = table_e, .bytes = bytes, .lines = opt.lines, .staging = if (opt.direct) 0 else opt.staging });
        errdefer lines.deinit();
        lines.pinned_bank = if (opt.direct) src.pinned_bank else null;
        if (src.shared) |sh| {
            var w: [3]i32 = undefined;
            for (0..3) |p| {
                if (sh[p].len == 0 or sh[p].len % (tiles * 16) != 0) return error.Invalid;
                w[p] = @intCast(sh[p].len / (tiles * 16));
            }
            // the shared expert gets its own resident region (it may be wider than the routed experts)
            try lines.setResident(shared_id, sh, w);
        }
        const n = opt.lines;
        const slots_tab = try gpa.alloc(u16, n);
        errdefer gpa.free(slots_tab);
        const line_tab = try gpa.alloc(offload.Line, n);
        errdefer gpa.free(line_tab);
        const stamp_tab = try gpa.alloc(u32, n);
        errdefer gpa.free(stamp_tab);
        const layer_tab = try gpa.alloc(u8, n);
        errdefer gpa.free(layer_tab);
        const own = counts == null;
        const c = counts orelse blk: {
            const t = try gpa.create([shape.layers][shape.experts]u32);
            t.* = @splat(@splat(0));
            break :blk t;
        };
        errdefer if (own) gpa.destroy(c);
        var rnd = try Round.init(gpa, n, opt.max_rows * opt.slots);
        errdefer rnd.deinit(gpa);
        const passes_n = if (opt.max_passes == 0) rnd.passes.len else @min(opt.max_passes, rnd.passes.len);
        if (opt.max_passes != 0 and opt.prefetch_bytes != 0) return error.Invalid; // in-flight prefetches lengthen rounds
        const pick_bytes = passes_n * opt.max_rows * opt.slots * 4;
        var picks_host = try cuda.HostBuffer.alloc(d, pick_bytes);
        errdefer picks_host.free();
        var picks_dev = try cuda.DeviceBuffer.alloc(d, pick_bytes);
        errdefer picks_dev.free();
        var picks_ev = try cuda.Event.init(d, false);
        errdefer picks_ev.deinit();
        const prev = try gpa.alloc(u16, shape.experts);
        errdefer gpa.free(prev);
        return .{
            .telemetry_id = try telemetry.identity(),
            .prev = prev,
            .gpa = gpa,
            .io = io,
            .opt = opt,
            .src = src,
            .scales = scales,
            .lines = lines,
            .k2 = k2,
            .slots_tab = slots_tab,
            .line_tab = line_tab,
            .stamp_tab = stamp_tab,
            .layer_tab = layer_tab,
            .v = offload.Vram.init(slots_tab, line_tab, stamp_tab, layer_tab),
            .counts = c,
            .own_counts = own,
            .round = rnd,
            .picks_host = picks_host,
            .picks_dev = picks_dev,
            .picks_ev = picks_ev,
        };
    }

    pub fn deinit(p: *Pool) void {
        p.lines.deinit();
        p.picks_ev.deinit();
        p.picks_dev.free();
        p.picks_host.free();
        p.round.deinit(p.gpa);
        if (p.own_counts) p.gpa.destroy(p.counts);
        p.gpa.free(p.prev);
        p.gpa.free(p.layer_tab);
        p.gpa.free(p.stamp_tab);
        p.gpa.free(p.line_tab);
        p.gpa.free(p.slots_tab);
        p.gpa.free(p.k2);
        p.* = undefined;
    }

    /// The `Experts` layer: the pool's device tables (fixed addresses) and the resident scales.
    pub fn tables(p: *const Pool) exl3e.Layer {
        return .{
            .gate = p.lines.table(0),
            .up = p.lines.table(1),
            .down = p.lines.table(2),
            .gate_k2 = p.lines.widths(0),
            .up_k2 = p.lines.widths(1),
            .down_k2 = p.lines.widths(2),
            .suh_g = p.scales.suh_g,
            .suh_u = p.scales.suh_u,
            .svh_g = p.scales.svh_g,
            .svh_u = p.scales.svh_u,
            .suh_d = p.scales.suh_d,
            .svh_d = p.scales.svh_d,
            .experts = p.lines.spec.experts,
            .dims = p.opt.dims,
            .width = p.opt.width,
            .codebook = p.opt.codebook,
            .act_mode = p.opt.act_mode,
        };
    }

    /// Promotes prefetch lines whose copy landed (the policy's reap); never blocks.
    fn reap(p: *Pool) !void {
        for (p.line_tab, p.slots_tab, 0..) |state, id, i| {
            if (state != .inflight) continue;
            if (try p.lines.landed(i)) offload.reapAt(&p.v, p.opt.layer, id, p.now) catch unreachable;
        }
    }

    /// Host step of a round: `picks` [rows * slots] (int32, -1 skipped, 512 the shared expert), as the device holds
    pub fn plan(p: *Pool, picks: []const i32, rows: usize) !*const Round {
        return p.planInternal(picks, rows, true);
    }

    fn planInternal(p: *Pool, picks: []const i32, rows: usize, observe: bool) !*const Round {
        if (rows == 0 or rows > p.opt.max_rows or picks.len != rows * p.opt.slots) return error.Invalid;
        p.now += 1;
        p.round_no += 1;
        try p.reap();
        const ropt: round_.Options = .{
            .swap_every = p.opt.swap_every,
            .swap_cap = p.opt.swap_cap,
            .prefetch_bytes = p.opt.prefetch_bytes,
            .line_bytes = p.lines.layout.line_bytes,
            .protect = if (p.opt.double_buffer) p.prev[0..p.prev_n] else &.{},
            .observe = observe,
        };
        round_.plan(&p.v, p.opt.layer, picks, p.counts, p.now, p.round_no, ropt, &p.round) catch |e| switch (e) {
            error.Busy => {
                // every line is held by a copy in flight: let them land, then plan again (nothing was written)
                p.stats.busy += 1;
                try p.lines.drain();
                try p.reap();
                try round_.plan(&p.v, p.opt.layer, picks, p.counts, p.now, p.round_no, ropt, &p.round);
            },
            else => return e,
        };
        const r = &p.round;
        if (r.n_passes * p.opt.max_rows * p.opt.slots * 4 > p.picks_dev.len) return error.PoolPasses;
        p.picks_n = picks.len;
        p.prev_n = r.n_ids;
        @memcpy(p.prev[0..r.n_ids], r.ids[0..r.n_ids]);
        if (r.protected) p.stats.protected += 1;
        if (r.masked()) {
            if (p.picks_pending) try p.picks_ev.synchronize(); // the previous round's uploads read these bytes
            p.picks_pending = false;
            const stride = p.opt.max_rows * p.opt.slots;
            const host = p.picks_host.slice(i32);
            var member: [shape.experts + 1]bool = undefined;
            for (0..r.n_passes) |j| {
                @memset(&member, false);
                for (r.passIds(j)) |id| member[id] = true;
                member[shared_id] = j == 0;
                for (picks, host[j * stride ..][0..picks.len]) |pk, *o| {
                    o.* = if (pk >= 0 and pk <= shared_id and member[@intCast(pk)]) pk else -1;
                }
            }
        }
        p.stats.rounds += 1;
        p.stats.passes += r.n_passes;
        p.stats.hit_slots += r.hit_slots;
        p.stats.miss_slots += r.miss_slots;
        p.stats.unique += r.unique;
        p.stats.misses += r.misses_();
        p.stats.evictions += r.evictions;
        return r;
    }

    fn loadCopy(p: *Pool, c: round_.Copy) !void {
        if (p.k2[c.id][0] == 0) return error.Invalid; // another rank's expert reached the planner unmasked
        const ev: ?usize = if (c.evicted == offload.Cache.empty_id) null else c.evicted;
        try p.lines.load(c.line, c.id, p.src.routed[c.id], p.k2[c.id], ev, p.opt.direct, p.io);
    }

    /// Device step of pass `j`: its copies and table rewrites on the copy stream, its masked picks (if any) on `s`
    pub fn stage(p: *Pool, r: *const Round, j: usize, s: cuda.Stream) !void {
        for (r.passCopies(j)) |c| try p.loadCopy(c);
        if (r.masked()) {
            const stride = p.opt.max_rows * p.opt.slots * 4;
            const n = p.picks_n * 4;
            try p.picks_dev.d.check(p.picks_dev.d.api.cuMemcpyHtoDAsync_v2(p.picks_dev.ptr + j * stride, p.picks_host.bytes[j * stride ..].ptr, n, s.handle), "cuMemcpyHtoDAsync");
            try p.picks_ev.record(s);
            p.picks_pending = true;
        }
        for (r.passLines(j)) |line| try p.lines.gate(s, line);
    }

    /// The picks pass `j` reads: the round's own device picks, or the pool's masked copy for a split round.
    pub fn picksFor(p: *const Pool, r: *const Round, j: usize, own: u64) u64 {
        if (!r.masked()) return own;
        return p.picks_dev.ptr + j * p.opt.max_rows * p.opt.slots * 4;
    }

    /// After pass `j` was enqueued on `s`: its lines are protected until it finishes.
    pub fn release(p: *Pool, r: *const Round, j: usize, s: cuda.Stream) !void {
        try p.lines.release(s, r.passLines(j));
        if (j + 1 == r.n_passes) {
            for (r.prefetch()) |c| try p.loadCopy(c);
            p.stats.prefetches += r.n_prefetch;
        }
    }

    pub fn warmProfile(pool: *Pool) !usize {
        return @import("exl3_frequency_seed.zig").warm(pool);
    }

    pub fn deviceBytes(p: *const Pool) usize {
        return p.lines.deviceBytes() + p.picks_dev.len;
    }

    pub fn lineBytes(p: *const Pool) usize {
        return p.lines.layout.line_bytes;
    }
};
