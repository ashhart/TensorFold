//! The offload pool planned on the device (flashnext_exl3_dplan.cu): a decode round's MoE needs no host wait between
//! the router and the experts, so the whole step (48 layers, the head, at tp=2 the NCCL sums) is one stream of device
//! work that a CUDA graph can hold (cuda_exl3_graph.zig).
//!
//! Each decoder layer's pool keeps its residency twice: the host's (exl3_offload.Vram + the lifetime counts, which
//! prefill, the MTP head and any eager round plan with) and this device copy. One of them owns it at a time:
//!   toDevice: the host's lines, stamps, counts and clock go up (one H2D on the compute stream, after the pools'
//!     copy streams drained), then every decode round plans on the device;
//!   toHost: the device copy comes back (one D2H after the stream finished) and rebuilds each pool's Vram, counts and
//!     statistics, before any host plan (cuda_exl3_pass.pooled, the expert-major prefill).
//! The device planner follows the host planner's rules for a one-pass round without prefetch or double-buffer
//! protection (exl3_round.zig `plan` with `protect` empty), so the two produce the same lines for the same picks
//! (tf-cuda-test flashnext-exl3-graph plan). The tables are the pool's own (fixed addresses): the device rewrites them
//! in place exactly where the host's Lines.load would.
//!
//! A miss is copied by fn_exl3_dcopy, a grid reading the pinned host bank through its mapped address (unified
//! addressing), instead of a DMA on the pool's copy stream: the copy is device work ordered on the compute stream.

const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("cuda_exl3_pool.zig");
const offload = pool_.offload;

const experts_n: usize = 512;
const misc_words: usize = 8;
const empty_id = offload.Cache.empty_id;

/// u32 words of one layer's device state.
pub fn stateWords(lines: usize) usize {
    return 2 * lines + 2 * experts_n + misc_words;
}

/// The planned copies a round can hold: a row's ten routed picks each.
fn workCap(max_rows: usize) usize {
    return max_rows * 10;
}

pub const Owner = enum { host, device };

/// Device statistics of a layer (the misc words after now).
pub const Misc = extern struct { now: u32, rounds: u32, hit_slots: u32, miss_slots: u32, copies: u32, evictions: u32, overflow: u32, inplace: u32 };

pub const DevPlan = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    plan_fn: cuda.Function,
    copy_fn: cuda.Function,
    restore_fn: cuda.Function,
    pools: []*pool_.Pool,
    lines: usize,
    words: usize,
    /// Rows a planned round may hold: rows * 10 distinct experts must fit the lines.
    max_rows: usize,
    state: cuda.DeviceBuffer,
    mirror: cuda.HostBuffer,
    /// Per pool: src [512 * 3] int64, lens [512 * 3] int64, k2 [512 * 3] int32 (static).
    statics: cuda.DeviceBuffer,
    work: cuda.DeviceBuffer,
    owner: Owner = .host,
    /// The stream the last device round ran on (toHost waits for it).
    stream: ?cuda.Stream = null,
    copy_blocks: u32 = 144,
    /// Overlap (TF_FLASHNEXT_EXL3_OVERLAP): the held experts compute while the misses' copies run on `side`, then the
    /// misses' experts (and the combine) follow; the per-slot outputs are the single pass's (each slot is written by
    /// the one pass that holds its expert, as the host cut's split rounds do).
    overlap: bool = false,
    side: cuda.Stream,
    ev_plan: cuda.Event,
    ev_copy: cuda.Event,
    /// The split's masked picks [max_rows * 11] each: held (and the shared expert), then the misses.
    split: cuda.DeviceBuffer,
    /// Device statistics already added to the pools' (a pull adds the difference).
    seen: []Misc,
    pushes: u64 = 0,
    pulls: u64 = 0,
    /// Experts read in place from the host bank (more misses than free lines in a round), summed at each pull.
    inplace: u64 = 0,

    const static_bytes = experts_n * 3 * (8 + 8 + 4);

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, pools: []const *pool_.Pool) !*DevPlan {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        if (pools.len == 0) return error.Invalid;
        const lines = pools[0].opt.lines;
        for (pools) |p| if (p.opt.lines != lines or p.lines.spec.experts < experts_n) return error.Invalid;
        var mod = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_dplan);
        errdefer mod.unload();
        const dp = try gpa.create(DevPlan);
        errdefer gpa.destroy(dp);
        const words = stateWords(lines);
        const own = try gpa.dupe(*pool_.Pool, pools);
        errdefer gpa.free(own);
        var state = try cuda.DeviceBuffer.alloc(d, pools.len * words * 4);
        errdefer state.free();
        var mirror = try cuda.HostBuffer.alloc(d, pools.len * words * 4);
        errdefer mirror.free();
        var statics = try cuda.DeviceBuffer.alloc(d, pools.len * static_bytes);
        errdefer statics.free();
        // rows a planned round may hold (MTP verify windows reach 16): more distinct misses than lines free of the
        // round's experts are read in place from the mapped bank for that round (fn_exl3_dplan), so any row count
        // up to this bound plans in one pass
        const max_rows = maxRows();
        var work = try cuda.DeviceBuffer.alloc(d, workCap(max_rows) * 9 * 8 + 256 + (experts_n + 64) * 4);
        errdefer work.free();
        const seen = try gpa.alloc(Misc, pools.len);
        errdefer gpa.free(seen);
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var ev_plan = try cuda.Event.init(d, false);
        errdefer ev_plan.deinit();
        var ev_copy = try cuda.Event.init(d, false);
        errdefer ev_copy.deinit();
        var split = try cuda.DeviceBuffer.alloc(d, 2 * max_rows * 11 * 4);
        errdefer split.free();
        // every expert's three host-bank addresses (mapped: unified addressing), byte counts and widths
        const host = try gpa.alloc(u8, static_bytes);
        defer gpa.free(host);
        for (pools, 0..) |p, i| {
            const src = std.mem.bytesAsSlice(u64, host[0 .. experts_n * 24]);
            const lens = std.mem.bytesAsSlice(u64, host[experts_n * 24 .. experts_n * 48]);
            const k2 = std.mem.bytesAsSlice(i32, host[experts_n * 48 ..]);
            for (0..experts_n) |e| for (0..3) |w| {
                const b = p.src.routed[e][w];
                src[e * 3 + w] = if (b.len == 0) 0 else @intFromPtr(b.ptr);
                lens[e * 3 + w] = b.len;
                k2[e * 3 + w] = p.k2[e][w];
                if (b.len != 0 and (b.len % 16 != 0 or @intFromPtr(b.ptr) % 16 != 0)) return error.Invalid;
            };
            try statics.upload(i * static_bytes, host);
        }
        dp.* = .{
            .gpa = gpa,
            .d = d,
            .mod = mod,
            .plan_fn = try mod.function("fn_exl3_dplan"),
            .copy_fn = try mod.function("fn_exl3_dcopy"),
            .restore_fn = try mod.function("fn_exl3_drestore"),
            .pools = own,
            .lines = lines,
            .words = words,
            .max_rows = max_rows,
            .state = state,
            .mirror = mirror,
            .statics = statics,
            .work = work,
            .seen = seen,
            .side = side,
            .ev_plan = ev_plan,
            .ev_copy = ev_copy,
            .split = split,
            .overlap = overlapFromEnv(pools[0]),
        };
        if (std.c.getenv("TF_FLASHNEXT_EXL3_DCOPY_BLOCKS")) |v| dp.copy_blocks = std.fmt.parseInt(u32, std.mem.span(v), 10) catch dp.copy_blocks;
        return dp;
    }

    pub fn deinit(dp: *DevPlan) void {
        dp.side.synchronize() catch {};
        dp.split.free();
        dp.ev_copy.deinit();
        dp.ev_plan.deinit();
        dp.side.deinit();
        dp.work.free();
        dp.statics.free();
        dp.mirror.free();
        dp.state.free();
        dp.mod.unload();
        dp.gpa.free(dp.seen);
        dp.gpa.free(dp.pools);
        dp.gpa.destroy(dp);
    }

    fn mirrorOf(dp: *DevPlan, i: usize) []u32 {
        const all = std.mem.bytesAsSlice(u32, dp.mirror.bytes[0 .. dp.pools.len * dp.words * 4]);
        return @alignCast(all[i * dp.words ..][0..dp.words]);
    }

    /// The host's residency up to the device, on `s` (a no-op if the device owns it). Waits for every pool's copies.
    pub fn toDevice(dp: *DevPlan, s: cuda.Stream) !void {
        dp.stream = s;
        if (dp.owner == .device) return;
        const lines = dp.lines;
        for (dp.pools, 0..) |p, i| {
            // copies in flight land, prefetched lines count as resident (the device knows no in-flight state)
            try p.lines.drain();
            for (p.line_tab, p.slots_tab) |*state, id| if (state.* == .inflight) {
                offload.reapAt(&p.v, p.opt.layer, id, p.now) catch {};
            };
            const m = dp.mirrorOf(i);
            const line_id: []i32 = @ptrCast(m[0..lines]);
            const stamp = m[lines .. 2 * lines];
            const line_of: []i32 = @ptrCast(m[2 * lines ..][0..experts_n]);
            const counts = m[2 * lines + experts_n ..][0..experts_n];
            @memset(line_of, -1);
            for (0..lines) |ln| {
                const id = p.slots_tab[ln];
                const held = p.line_tab[ln] == .resident and id != empty_id and id < experts_n;
                line_id[ln] = if (held) id else -1;
                stamp[ln] = p.stamp_tab[ln];
                if (held) line_of[id] = @intCast(ln);
            }
            @memcpy(counts, &p.counts[p.opt.layer]);
            const misc: *Misc = @ptrCast(m[2 * lines + 2 * experts_n ..][0..misc_words]);
            misc.* = .{ .now = p.now, .rounds = 0, .hit_slots = 0, .miss_slots = 0, .copies = 0, .evictions = 0, .overflow = 0, .inplace = 0 };
            dp.seen[i] = misc.*;
        }
        try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dp.state.ptr, dp.mirror.bytes.ptr, dp.state.len, s.handle), "cuMemcpyHtoDAsync");
        dp.owner = .device;
        dp.pushes += 1;
    }

    /// The device's residency back to the host pools (a no-op if the host owns it): waits for the last round's stream.
    pub fn toHost(dp: *DevPlan) !void {
        if (dp.owner == .host) return;
        if (dp.stream) |s| try s.synchronize();
        try dp.side.synchronize();
        try dp.d.check(dp.d.api.cuMemcpyDtoH_v2(dp.mirror.bytes.ptr, dp.state.ptr, dp.state.len), "cuMemcpyDtoH");
        const lines = dp.lines;
        var overflow: u64 = 0;
        for (dp.pools, 0..) |p, i| {
            const m = dp.mirrorOf(i);
            const line_id: []const i32 = @ptrCast(m[0..lines]);
            const stamp = m[lines .. 2 * lines];
            const counts = m[2 * lines + experts_n ..][0..experts_n];
            const misc: *const Misc = @ptrCast(m[2 * lines + 2 * experts_n ..][0..misc_words]);
            for (0..lines) |ln| {
                const id = line_id[ln];
                p.slots_tab[ln] = if (id < 0) empty_id else @intCast(id);
                p.line_tab[ln] = if (id < 0) .empty else .resident;
                p.stamp_tab[ln] = stamp[ln];
                p.layer_tab[ln] = p.opt.layer;
            }
            @memcpy(&p.counts[p.opt.layer], counts);
            const was = dp.seen[i];
            p.now = misc.now;
            p.stats.rounds += misc.rounds - was.rounds;
            p.stats.passes += misc.rounds - was.rounds;
            p.stats.hit_slots += misc.hit_slots - was.hit_slots;
            p.stats.miss_slots += misc.miss_slots - was.miss_slots;
            p.stats.misses += misc.copies - was.copies;
            p.stats.evictions += misc.evictions - was.evictions;
            overflow += misc.overflow;
            dp.inplace += misc.inplace - was.inplace;
            dp.seen[i] = misc.*;
            // the device's rounds finished (synchronized above): no line waits for a reader, nothing is in flight
            @memset(p.lines.line_epoch, 0);
            @memset(p.lines.line_pending, false);
            p.prev_n = 0;
            p.picks_pending = false;
        }
        dp.owner = .host;
        dp.pulls += 1;
        if (overflow != 0) {
            std.log.err("flashnext exl3 dplan: {d} rounds had more distinct experts than the planner holds (256): results of those rounds are wrong", .{overflow});
            return error.PoolPasses;
        }
    }

    /// Device statistics so far, summed over the layers (synchronizes; does not change the owner).
    pub fn peek(dp: *DevPlan) !Misc {
        var t: Misc = std.mem.zeroes(Misc);
        if (dp.owner == .host) return t;
        if (dp.stream) |s| try s.synchronize();
        try dp.d.check(dp.d.api.cuMemcpyDtoH_v2(dp.mirror.bytes.ptr, dp.state.ptr, dp.state.len), "cuMemcpyDtoH");
        for (dp.pools, 0..) |_, i| {
            const m = dp.mirrorOf(i);
            const misc: *const Misc = @ptrCast(m[2 * dp.lines + 2 * experts_n ..][0..misc_words]);
            inline for (.{ "rounds", "hit_slots", "miss_slots", "copies", "evictions", "overflow", "inplace" }) |f| @field(t, f) += @field(misc, f) - @field(dp.seen[i], f);
        }
        return t;
    }

    /// The in-place experts of the last plan ([512] int32) and their count after it.
    fn ovfList(dp: *const DevPlan) u64 {
        return dp.work.ptr + workCap(dp.max_rows) * 9 * 8 + 256;
    }

    /// After the round's experts (on `s`): the in-place experts' table entries cleared.
    pub fn restore(dp: *DevPlan, s: cuda.Stream, i: usize) !void {
        const p = dp.pools[i];
        var a: cuda.Args = .{};
        a.add(dp.ovfList());
        a.add(dp.ovfList() + experts_n * 4);
        for (0..3) |w| a.add(p.lines.table(w));
        for (0..3) |w| a.add(p.lines.widths(w));
        try cuda.launch.launch(dp.restore_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// `s` waits for the last plan's copies (runLayer joins them itself; a caller of `plan` alone needs this).
    pub fn join(dp: *DevPlan, s: cuda.Stream) !void {
        if (dp.overlap) try s.wait(dp.ev_copy);
    }

    /// The index of `p` among the planned pools, or null (the MTP layer's pool, another engine's).
    pub fn indexOf(dp: *const DevPlan, p: *const pool_.Pool) ?usize {
        const i = p.opt.layer;
        if (i < dp.pools.len and dp.pools[i] == p) return i;
        for (dp.pools, 0..) |q, j| if (q == p) return j;
        return null;
    }

    /// Plans pool `i`'s round of `rows` rows on the device and copies its misses into their lines, on `s`: after this
    /// the pool's tables hold every expert `picks` [rows * 11] names. Graph-capturable (no host work depends on it).
    pub fn plan(dp: *DevPlan, s: cuda.Stream, i: usize, picks: u64, rows: usize) !void {
        if (rows == 0 or rows > dp.max_rows) return error.PoolPasses;
        const p = dp.pools[i];
        const st = dp.state.ptr + i * dp.words * 4;
        const sb = dp.statics.ptr + i * static_bytes;
        const work = dp.work.ptr;
        const work_n = dp.work.ptr + workCap(dp.max_rows) * 9 * 8;
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(@as(c_int, @intCast(rows * 11)));
        a.add(st);
        a.add(@as(c_int, @intCast(dp.lines)));
        for (0..3) |w| a.add(p.lines.table(w));
        for (0..3) |w| a.add(p.lines.widths(w));
        a.add(sb);
        a.add(sb + experts_n * 24);
        a.add(sb + experts_n * 48);
        a.add(@as(u64, p.lines.mem.ptr));
        a.add(@as(u64, p.lines.layout.line_bytes));
        for (p.lines.layout.off) |o| a.add(@as(u64, o));
        a.add(work);
        a.add(work_n);
        a.add(if (dp.overlap) dp.split.ptr else @as(u64, 0));
        a.add(if (dp.overlap) dp.split.ptr + dp.max_rows * 11 * 4 else @as(u64, 0));
        a.add(dp.ovfList());
        a.add(dp.ovfList() + experts_n * 4);
        try cuda.launch.launch(dp.plan_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &a);
        var c: cuda.Args = .{};
        c.add(work);
        c.add(work_n);
        if (!dp.overlap) return cuda.launch.launch(dp.copy_fn, .{ .grid = .{ .x = dp.copy_blocks }, .block = .{ .x = 256 } }, s, &c);
        // the copies on the side stream (a fork in a captured graph); `s` joins them before the misses' experts
        try dp.ev_plan.record(s);
        try dp.side.wait(dp.ev_plan);
        try cuda.launch.launch(dp.copy_fn, .{ .grid = .{ .x = dp.copy_blocks }, .block = .{ .x = 256 } }, dp.side, &c);
        try dp.ev_copy.record(dp.side);
    }
};

// ---- the engine's planner -------------------------------------------------------------------------------------------

/// The engine's device planner (one engine a process: a rank), made on the first device round.
var global: ?*DevPlan = null;
var global_owner: ?*const anyopaque = null;
/// Set while a decode round runs with device plans (cuda_exl3_graph.zig): cuda_exl3_pass.moe takes `layer` instead
/// of the host cut.
pub var active: ?*DevPlan = null;

/// TF_FLASHNEXT_EXL3_DPLAN: 1 plans decode rounds on the device (implied by TF_FLASHNEXT_EXL3_GRAPH=1).
pub fn enabled() bool {
    if (@import("cuda_exl3_graph.zig").force) |f| return f != .off;
    if (std.c.getenv("TF_FLASHNEXT_EXL3_DPLAN")) |v| return std.mem.eql(u8, std.mem.span(v), "1");
    return @import("cuda_exl3_graph.zig").enabled();
}

/// The device planner over `layers`' pools (the decoder layers only), made once for `owner`; null when a layer has no
/// pool (resident experts) or the pools differ in lines.
pub fn forLayers(gpa: std.mem.Allocator, d: *const cuda.Driver, owner: *const anyopaque, layers: []const @import("cuda_prompt_load.zig").Layer) !?*DevPlan {
    if (global) |g| {
        if (global_owner == owner) return g;
        try g.toHost();
        g.deinit();
        global = null;
    }
    var pools: [64]*pool_.Pool = undefined;
    var n: usize = 0;
    for (layers) |*l| {
        const x = l.exl3 orelse return null;
        const p = x.pool orelse return null;
        if (n == pools.len) return null;
        pools[n] = p;
        n += 1;
    }
    if (n == 0) return null;
    // the MTP layer's pool last (its own lifetime counts; the draft chain plans it on the device)
    if (mtp_pool) |mp| if (mp.opt.lines == pools[0].opt.lines and n < pools.len) {
        pools[n] = mp;
        n += 1;
    };
    global = try DevPlan.init(gpa, d, pools[0..n]);
    global_owner = owner;
    return global;
}

/// The draft head's (MTP layer's) pool, registered by cuda_exl3_pass.attach.
pub var mtp_pool: ?*pool_.Pool = null;

/// The engine's planner if made (any owner).
pub fn current() ?*DevPlan {
    return global;
}

/// Before any host plan: the device's residency back to the pools (no-op without a device planner).
pub fn toHost() !void {
    if (global) |g| try g.toHost();
}

pub fn releaseFor(owner: *const anyopaque) void {
    if (global) |g| if (global_owner == owner) {
        g.deinit();
        global = null;
        global_owner = null;
        active = null;
    };
}

/// One offloaded layer's experts on the device plan: plan + copies, then `Experts.run` on the pool's tables, as the
/// host cut's single pass runs it (cuda_exl3_pass.pooled): same kernels, same inputs, so the same bits.
pub fn runLayer(dp: *DevPlan, st: cuda.Stream, i: usize, ex: *const cuda.exl3_experts.Experts, layer: cuda.exl3_experts.Layer, xin: u64, picks: u64, rows: usize, sc: cuda.exl3_experts.Scratch, wts: u64, out: u64) !void {
    try dp.plan(st, i, picks, rows);
    if (!dp.overlap) {
        try ex.run(st, layer, xin, 2560, .bf16, picks, wts, rows, out, sc);
    } else {
        // held experts while the copies run, then the misses' experts and (expert parallel) the combine of every slot
        try ex.run(st, layer, xin, 2560, .bf16, dp.split.ptr, 0, rows, 0, sc);
        try st.wait(dp.ev_copy);
        try ex.run(st, layer, xin, 2560, .bf16, dp.split.ptr + dp.max_rows * 11 * 4, wts, rows, out, sc);
    }
    try dp.restore(st, i);
}

/// TF_FLASHNEXT_EXL3_GRAPH_ROWS: rows a device-planned round holds at most (default 16, the MTP window).
fn maxRows() usize {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_ROWS") orelse return 16;
    return @max(1, std.fmt.parseInt(usize, std.mem.span(v), 10) catch 16);
}

/// TF_FLASHNEXT_EXL3_OVERLAP=1 turns the split on (default off: compiled, not yet run on the real pack, unmeasured).
fn overlapFromEnv(p: *const pool_.Pool) bool {
    _ = p;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_OVERLAP") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}
