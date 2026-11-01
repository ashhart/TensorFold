//! The batched pass's EXL3 branches (cuda_batch.zig):

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const load = @import("../pack/cuda_exl3_load.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const memory = @import("../pack/exl3_startup_memory.zig");
const line_deltas = @import("../offload/exl3_line_deltas.zig");
const profile_line_budget = @import("../offload/exl3_profile_line_budget.zig");

const dims = wt.dims;
const slots = wt.slots;

/// Rows a routed-experts call takes: its scratch grows with rows * 11 (fp16 inputs, fp32 split sums).
pub const moe_rows: usize = 256;

fn copy2d(s: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&m, s.handle), "cuMemcpy2DAsync");
}

/// One part's rows [m, part.n] into `y` (row pitch part.n), bf16 or fp32.
fn part(p: anytype, ops: *load.Ops, pt: load.Part, x: u64, k: usize, m: usize, y: u64, f32_out: bool) !void {
    switch (pt.kind) {
        .exl3 => |w| {
            const de = try ops.linear();
            if (try @import("../prefill/cuda_exl3_prefill.zig").tilesPart(de, p.stream, w, x, m, y, f32_out)) return; // prefill rows on #806's tiles
            try de.decode(p.stream, w, x, .bf16, m, y, if (f32_out) .f32 else .bf16);
        },
        .native => |w| {
            if (f32_out) return p.k.native_lane.matmulF32(p.stream, x, k, m, w, y, p.s.native_part);
            try p.k.native_lane.matmul(p.stream, x, k, m, w, y, p.s.native_part);
        },
    }
}

/// Pass.lane() for an EXL3 projection: out bf16 [m, n] (sk 1), or fp32 [m.
pub fn lane(p: anytype, x: u64, e: *const load.Proj, out: u64, slices: u64, m: usize, sk: usize) !void {
    const s = p.stream;
    if (e.x == .f16) return laneF16(p, x, e, out, slices, m, sk);
    if (sk > 1) {
        if (e.parts.len != 1) return error.UnexpectedTensor;
        try part(p, e.ops, e.parts[0], x, e.k, m, slices, true);
        const bytes = m * e.n * 4;
        try s.d.check(s.d.api.cuMemsetD8Async(slices + bytes, 0, bytes * (sk - 1), s.handle), "cuMemsetD8Async");
        return;
    }
    if (try @import("../tp/cuda_exl3_fork.zig").lane(p, x, e, out, m, onePart)) return; // the parts on parallel streams
    for (e.parts) |pt| try onePart(p, e, pt, x, m, out);
}

/// One part of a bf16 sk-1 lane() on the pass's stream: one part over the whole row, or a single row, writes its
fn onePart(p: anytype, e: *const load.Proj, pt: load.Part, x: u64, m: usize, out: u64) !void {
    const s = p.stream;
    if (pt.n == e.n or m == 1) return part(p, e.ops, pt, x, e.k, m, out + pt.col * 2, false);
    const tmp = try e.ops.scratch(s, &e.ops.tmp, m * pt.n * 2);
    try part(p, e.ops, pt, x, e.k, m, tmp, false);
    try copy2d(s, out + pt.col * 2, e.n * 2, tmp, pt.n * 2, pt.n * 2, m);
}

/// fp16 rows through a dense BF16 lane weight: x = hi + lo exactly (fn_exl3_split_f16), hi . w and lo . w in fp32.
fn laneF16(p: anytype, x: u64, e: *const load.Proj, out: u64, slices: u64, m: usize, sk: usize) !void {
    const s = p.stream;
    if (e.parts.len != 1 or e.parts[0].kind != .native) return error.UnexpectedTensor;
    const w = e.parts[0].kind.native;
    const count = m * e.k;
    const halves = try e.ops.scratch(s, &e.ops.halves, 2 * count * 2);
    var a: cuda.Args = .{};
    a.add(x);
    a.add(halves);
    a.add(halves + count * 2);
    a.add(@as(c_longlong, @intCast(count)));
    try cuda.launch.launch(try e.ops.split(), .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    const bytes = m * e.n * 4;
    const sums = if (sk == 1) try e.ops.scratch(s, &e.ops.sums, 2 * bytes) else slices;
    try p.k.native_lane.matmulF32(s, halves, e.k, m, w, sums, p.s.native_part);
    try p.k.native_lane.matmulF32(s, halves + count * 2, e.k, m, w, sums + bytes, p.s.native_part);
    if (sk == 1) return p.sumSlices(sums, out, true, m * e.n, 2);
    if (sk > 2) try s.d.check(s.d.api.cuMemsetD8Async(slices + 2 * bytes, 0, bytes * (sk - 2), s.handle), "cuMemsetD8Async");
}

/// Pass.moeBlock() for EXL3 experts: the router's 513 logits (the BF16 rows the Store widened exactly), each row's ten
pub fn moe(p: anytype, lw: anytype, x: *const load.Experts) !void {
    if (x.tp != null) return @import("../tp/cuda_exl3_tp.zig").moe(p, lw, x); // expert parallel (plan step 10)
    const s = p.s;
    const n = p.rows;
    try p.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, n);
    try p.tri.topkMany(s.router_l, s.picks, s.wts, n);
    if (x.needsPool()) return error.Exl3PoolNotAttached;
    const ex = try x.ops.grouped();
    const block = @min(n, moe_rows);
    const bytes = try cuda.exl3_experts.scratchBytes(x.layer, block, slots);
    const base = try x.ops.scratch(p.stream, &x.ops.moe, bytes);
    var r0: usize = 0;
    while (r0 < n) : (r0 += block) {
        const rows = @min(block, n - r0);
        var sc = try cuda.exl3_experts.scratch(base, x.layer, block, slots);
        // runSlots leaves each slot's fp32 output in sc.y [rows * slots, dims]: the pass's y rows are that layout
        sc.y = s.y + r0 * slots * dims * 4;
        const xin = s.mixed + r0 * dims * 2;
        const picks = s.picks + r0 * slots * 4;
        if (x.pool) |pool| {
            try pooled(p.stream, x.ops, pool, ex, x.layer, xin, picks, rows, sc, 0, 0);
            continue;
        }
        try ex.runSlots(p.stream, x.layer, xin, dims, .bf16, picks, rows, sc);
    }
    try @import("../tp/cuda_exl3_tp.zig").canonical(p);
}

/// Host time of the A/B cut (the picks D2H and its wait) and of Pool.plan, summed over every pooled block (read by
pub var pool_time: struct { cut_ns: u64 = 0, plan_ns: u64 = 0, blocks: u64 = 0 } = .{};

/// One block of `rows` rows through `pool`: per-slot outputs in sc.y, and with `wts` != 0 the last pass also combines
pub fn pooled(st: cuda.Stream, ops: *load.Ops, pool: *pool_.Pool, ex: *const cuda.exl3_experts.Experts, layer: cuda.exl3_experts.Layer, xin: u64, picks: u64, rows: usize, sc: cuda.exl3_experts.Scratch, wts: u64, out: u64) !void {
    const host = (ops.picks orelse return error.Exl3ScratchNotReserved).slice(i32)[0 .. rows * slots];
    const io = pool.io;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    // end of segment A: the block's picks reach the host (the only host wait of the layer)
    try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(host.ptr, picks, rows * slots * 4, st.handle), "cuMemcpyDtoHAsync");
    try st.synchronize();
    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    const r = try pool.plan(host, rows);
    pool_time.cut_ns += @intCast(t1 - t0);
    pool_time.plan_ns += @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t1);
    pool_time.blocks += 1;
    // segment B, a pass at a time: its copies and table rewrites, then the experts on the pool's tables
    for (0..r.n_passes) |j| {
        try pool.stage(r, j, st);
        const last = j + 1 == r.n_passes;
        try ex.run(st, layer, xin, dims, .bf16, pool.picksFor(r, j, picks), if (last) wts else 0, rows, if (last) out else 0, sc);
        try pool.release(r, j, st);
    }
}

/// Every offloaded experts layer of a loaded engine (the 48 decoder layers, then the MTP layer if any).
pub fn offloadedLayers(layers: []const @import("../../cuda_prompt_load.zig").Layer, mtp: ?*const @import("../../cuda_prompt_load.zig").Layer, out: []*load.Experts) usize {
    var n: usize = 0;
    for (layers) |*l| if (l.exl3) |x| if (x.needsPool()) {
        out[n] = x;
        n += 1;
    };
    if (mtp) |l| if (l.exl3) |x| if (x.needsPool()) {
        out[n] = x;
        n += 1;
    };
    return n;
}

pub const Budget = struct {
    /// Device bytes left free after the pools; null: the native server's own reserve (TENSORFOLD_MEMORY_RESERVE_GIB).
    reserve: ?u64 = null,
    /// What the pools leave beyond the server's reserve: allocator rounding and the copy streams' own memory.
    margin: u64 = 256 << 20,
    /// Bytes the family still allocates after the pools: each admitted stream's caches.
    other: u64 = 0,
    /// Lines a layer, or null: from free memory (cuda_exl3_round.linesFor).
    lines: ?usize = null,
    /// Rows a planned round holds (the pass's block of routed rows).
    max_rows: usize = moe_rows,
};

/// TF_FLASHNEXT_EXPERT_LINES and TF_FLASHNEXT_EXPERT_RESERVE_MB override pool sizing.
pub fn budgetFromEnv(stream_bytes: u64) !Budget {
    return budgetFromEnvWithStreams(stream_bytes, null);
}

pub fn budgetForStreams(stream_bytes: u64, requested_streams: u32) !Budget {
    return budgetFromEnvWithStreams(stream_bytes, requested_streams);
}

fn budgetFromEnvWithStreams(stream_bytes: u64, requested_streams: ?u32) !Budget {
    const streams = requested_streams orelse 1;
    var b: Budget = .{ .other = try streamReservation(stream_bytes, streams) };
    if (std.c.getenv("TF_FLASHNEXT_EXPERT_LINES")) |v| b.lines = try std.fmt.parseInt(usize, std.mem.span(v), 10);
    if (std.c.getenv("TF_FLASHNEXT_EXPERT_RESERVE_MB")) |v| b.reserve = try memory.mebibytes(std.mem.span(v));
    if (requested_streams == null) if (std.c.getenv("TF_FLASHNEXT_EXL3_STREAMS")) |v| {
        const env_streams = try std.fmt.parseInt(u64, std.mem.span(v), 10);
        if (env_streams == 0) return error.InvalidStreamCount;
        b.other = try std.math.mul(u64, stream_bytes, env_streams);
    };
    return b;
}

fn streamReservation(stream_bytes: u64, streams: u32) !u64 {
    if (streams == 0) return error.InvalidStreamCount;
    return std.math.mul(u64, stream_bytes, streams);
}

test "EXL3 pool reserves the requested number of concurrent stream caches" {
    try std.testing.expectEqual(@as(u64, 200), try streamReservation(100, 2));
    try std.testing.expectError(error.InvalidStreamCount, streamReservation(100, 0));
}

pub const Attached = struct { layers: usize = 0, lines: usize = 0, line_bytes: usize = 0, device_bytes: u64 = 0, free_before: u64 = 0, free_after: u64 = 0, scratch_bytes: u64 = 0, bank_bytes: u64 = 0, bank_ns: u64 = 0, prefill_window_bytes: u64 = 0 };

pub fn freeBytes(d: *const cuda.Driver) !u64 {
    var free: usize = 0;
    var total: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

/// The sizing hook (cuda_lanes_load.zig, once everything else of the family is allocated).
pub fn attach(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, st: cuda.Stream, layers: []const @import("../../cuda_prompt_load.zig").Layer, mtp: ?*const @import("../../cuda_prompt_load.zig").Layer, rows: usize, b: Budget, profile_counts: ?*[pool_.shape.layers][pool_.shape.experts]u32) !Attached {
    var all: [wt.layers_n + 1]*load.Experts = undefined;
    const n = offloadedLayers(layers, mtp, &all);
    const delta_text = if (std.c.getenv("TF_FLASHNEXT_EXL3_LINE_DELTAS")) |value| std.mem.span(value) else null;
    var out: Attached = .{};
    const scratch_before = try freeBytes(d);
    // every distinct Ops (the engine's store and the MTP head's) reserves its scratch now
    var seen: [4]*load.Ops = undefined;
    var n_seen: usize = 0;
    for (layers) |*l| if (l.exl3) |x| try reserveOnce(x.ops, &seen, &n_seen, st, rows);
    if (mtp) |l| if (l.exl3) |x| try reserveOnce(x.ops, &seen, &n_seen, st, rows);
    // expert parallel (cuda_exl3_tp.zig): its masks, sums and block scratch, before the pools take what is free
    for (layers) |*l| if (l.exl3) |x| if (x.tp != null) {
        try @import("../tp/cuda_exl3_tp.zig").reserve(x.ops, st, x.layer, rows, x.tp.?.world);
        break;
    };
    if (n == 0) {
        if (delta_text != null) return error.InvalidLineDeltaIndex;
        return out;
    }
    for (all[0..n]) |x| {
        out.line_bytes = @max(out.line_bytes, x.lineBytes());
        out.bank_bytes += x.bank.?.bytes.len;
        out.bank_ns += x.bank_ns;
    }
    var free: usize = 0;
    var total: usize = 0;
    try d.check(d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    out.free_before = free;
    out.scratch_bytes = scratch_before -| free;
    const reserve_floor = b.reserve orelse serverReserve(total);
    const reserve = b.reserve orelse try std.math.add(u64, reserve_floor, b.margin);
    // beside its lines a layer holds the shared expert's region, its six tables and its masked picks (sized for the
    var fixed: u64 = 0;
    for (all[0..n]) |x| {
        for (x.shared) |t| fixed = try std.math.add(u64, fixed, try std.math.mul(u64, (try std.math.add(u64, t.len, 255)) / 256, 256));
        const picks_bytes = try std.math.mul(u64, try std.math.mul(u64, pool_.passesFor(pool_.min_lines), b.max_rows), slots * 4);
        fixed = try std.math.add(u64, fixed, try std.math.add(u64, cuda.exl3_pool.Tables.init(wt.experts_n + 1).bytes + (64 << 10), picks_bytes));
    }
    const future = b.other;
    const prefill = @import("../prefill/cuda_exl3_prefill.zig");
    const minimum_pools = try memory.poolBytes(b.lines orelse pool_.min_lines, out.line_bytes, n, fixed);
    const protected = try std.math.add(u64, try std.math.add(u64, reserve, future), minimum_pools);
    out.prefill_window_bytes = prefill.reserveTwoChunk(d, free, protected) catch |err| declined: {
        std.log.warn("flashnext exl3 prefill: two-chunk reservation refused ({t}); keeping single-chunk prefill", .{err});
        break :declined 0;
    };
    if (out.prefill_window_bytes != 0) {
        free = @intCast(try freeBytes(d));
        out.free_before = free;
    }
    const pool_room = try memory.available(free, reserve, future);
    const lines = b.lines orelse try pool_.linesFor(free, reserve, try std.math.add(u64, future, fixed), out.line_bytes, n, pool_.min_lines);
    if (lines < pool_.min_lines) return error.PoolTooSmall;
    const startup_bytes = try memory.poolBytes(lines, out.line_bytes, n, fixed);
    if (startup_bytes > pool_room) return error.InsufficientStartupMemory;
    out.lines = lines;
    var pool_lines: [wt.layers_n + 1]usize = @splat(lines);
    var line_bytes: [wt.layers_n + 1]usize = undefined;
    var profile_sized = false;
    var profile_min_lines: usize = std.math.maxInt(usize);
    var profile_max_lines: usize = 0;
    var decoder_pool_count: usize = 0;
    for (all[0..n], 0..) |x, index| {
        line_bytes[index] = x.lineBytes();
        const is_mtp = if (mtp) |l| (if (l.exl3) |m| m == x else false) else false;
        if (!is_mtp) decoder_pool_count += 1;
    }
    if (delta_text == null and shouldProfileSize(profileSizingEnabled(), b.lines)) if (profile_counts) |prior| {
        var count_rows: [wt.layers_n][]const u32 = undefined;
        var allocation_costs: [wt.layers_n]usize = undefined;
        var allocation_lines: [wt.layers_n]usize = undefined;
        var decoder_index: usize = 0;
        var has_profile_signal = false;
        for (all[0..n], 0..) |x, pool_index| {
            const is_mtp = if (mtp) |l| (if (l.exl3) |m| m == x else false) else false;
            if (is_mtp) continue;
            for (prior[decoder_index]) |count| {
                has_profile_signal = has_profile_signal or count != 0;
            }
            count_rows[decoder_index] = &prior[decoder_index];
            allocation_costs[decoder_index] = line_bytes[pool_index];
            decoder_index += 1;
        }
        if (has_profile_signal and decoder_index == wt.layers_n) {
            const rank = all[0].tp orelse try @import("../tp/cuda_exl3_tp.zig").Rank.init(1, 0);
            const maybe_plan: ?profile_line_budget.Budget = profile_line_budget.allocate(gpa, .{
                .counts = &count_rows,
                .rank = rank.rank,
                .world = rank.world,
                .line_bytes = &allocation_costs,
                .uniform_lines = lines,
                .min_lines = pool_.min_lines,
                .max_rows = b.max_rows,
                .slots = slots,
            }, &allocation_lines) catch |err| blk: {
                std.log.info("flashnext exl3: profile-sized pools unavailable ({s}); retaining uniform decoder pools", .{@errorName(err)});
                break :blk null;
            };
            if (maybe_plan) |plan| {
                decoder_index = 0;
                for (all[0..n], 0..) |x, pool_index| {
                    const is_mtp = if (mtp) |l| (if (l.exl3) |m| m == x else false) else false;
                    if (is_mtp) continue;
                    pool_lines[pool_index] = allocation_lines[decoder_index];
                    profile_min_lines = @min(profile_min_lines, allocation_lines[decoder_index]);
                    profile_max_lines = @max(profile_max_lines, allocation_lines[decoder_index]);
                    decoder_index += 1;
                }
                if (plan.uniform_fallback) {
                    @memset(pool_lines[0..n], lines);
                    std.log.info("flashnext exl3: profile candidate changed total variable CUDA bytes; retaining uniform decoder pools", .{});
                } else {
                    profile_sized = true;
                }
                std.log.info("flashnext exl3: profile-weighted decoder allocation used {d}/{d} variable CUDA bytes (expert lines and device picks); saved lifetime picks are a proxy, not latency; MTP remains uniform", .{ plan.allocated_bytes, plan.baseline_bytes });
                std.log.info("flashnext exl3: profile decoder pool sizes range {d}..{d} lines; uniform-equivalent baseline was {d}", .{ profile_min_lines, profile_max_lines, lines });
            }
        }
    };
    if (delta_text) |text| {
        const plan = try line_deltas.apply(text, lines, decoder_pool_count, line_bytes[0..n], pool_.min_lines, fixed, startup_bytes, pool_lines[0..n]);
        std.log.info("flashnext exl3: experimental rank-local decoder pool line deltas {s}; base={d} startup_bytes={d}", .{ text, lines, plan.pool_bytes });
    }
    // the first decoder layer's pool owns the lifetime use table the other decoder layers share
    var counts = profile_counts;
    var decoder: u8 = 0;
    for (all[0..n], pool_lines[0..n]) |x, count| {
        const is_mtp = if (mtp) |l| (if (l.exl3) |m| m == x else false) else false;
        try x.attach(io, d, if (is_mtp) 0 else decoder, count, b.max_rows, if (is_mtp) null else counts);
        if (!is_mtp and counts == null) counts = x.pool.?.counts;
        if (!is_mtp) decoder += 1;
        out.device_bytes += x.pool.?.deviceBytes();
        out.layers += 1;
    }
    out.free_after = try freeBytes(d);
    _ = memory.available(out.free_after, reserve_floor, future) catch |err| {
        std.log.err("EXL3 pool reservation shortfall: free_before={d} free_after={d} pool_bytes={d} floor={d} allocator_margin={d} future={d}", .{ free, out.free_after, out.device_bytes, reserve_floor, b.margin, future });
        return err;
    };
    const allocation = if (delta_text != null) "uniform-equivalent lines (experimental deltas applied)" else if (profile_sized) "uniform-equivalent lines (profile-sized decoder pools)" else "lines a layer";
    std.log.info("flashnext exl3: {d} offloaded expert layers, {d} {s} of {d:.2} MiB ({d:.2} GiB on the device, {d:.2} GiB free before); pinned host bank {d:.2} GiB copied in {d:.1} s", .{ out.layers, out.lines, allocation, @as(f64, @floatFromInt(out.line_bytes)) / (1 << 20), @as(f64, @floatFromInt(out.device_bytes)) / (1 << 30), @as(f64, @floatFromInt(out.free_before)) / (1 << 30), @as(f64, @floatFromInt(out.bank_bytes)) / (1 << 30), @as(f64, @floatFromInt(out.bank_ns)) / 1e9 });
    return out;
}

test {
    _ = line_deltas;
    _ = profile_line_budget;
}

fn profileSizingEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_PROFILE_SIZING") orelse return false;
    return shouldProfileSize(std.mem.eql(u8, std.mem.span(value), "1"), null);
}

fn shouldProfileSize(enabled: bool, explicit_lines: ?usize) bool {
    return enabled and explicit_lines == null;
}

test "exl3: profile sizing requires explicit opt-in and does not override fixed line budgets" {
    try std.testing.expect(!shouldProfileSize(false, null));
    try std.testing.expect(!shouldProfileSize(true, 32));
    try std.testing.expect(shouldProfileSize(true, null));
}

/// native/cuda_memory.zig reserveBytes for a discrete card: TENSORFOLD_MEMORY_RESERVE_GIB, else max(1 GiB, total / 32).
fn serverReserve(total: u64) u64 {
    if (std.c.getenv("TENSORFOLD_MEMORY_RESERVE_GIB")) |v| {
        const g = std.fmt.parseFloat(f64, std.mem.trim(u8, std.mem.span(v), " ")) catch 0;
        if (g >= 0.5) return @intFromFloat(g * (1 << 30));
    }
    return @max(@as(u64, 1 << 30), total / 32);
}

fn reserveOnce(o: *load.Ops, seen: []*load.Ops, n: *usize, st: cuda.Stream, rows: usize) !void {
    for (seen[0..n.*]) |x| if (x == o) return;
    if (n.* == seen.len) return error.Invalid;
    seen[n.*] = o;
    n.* += 1;
    try o.reserve(st, rows, moe_rows, slots);
}

/// Pool statistics summed over every attached layer.
pub fn poolStats(layers: []const @import("../../cuda_prompt_load.zig").Layer, mtp: ?*const @import("../../cuda_prompt_load.zig").Layer) pool_.Stats {
    var t: pool_.Stats = .{};
    var all: [wt.layers_n + 1]*load.Experts = undefined;
    const n = offloadedPools(layers, mtp, &all);
    for (all[0..n]) |x| {
        const s = x.pool.?.stats;
        inline for (@typeInfo(pool_.Stats).@"struct".field_names) |f| {
            @field(t, f) += @field(s, f);
        }
    }
    return t;
}

pub fn offloadedPools(layers: []const @import("../../cuda_prompt_load.zig").Layer, mtp: ?*const @import("../../cuda_prompt_load.zig").Layer, out: []*load.Experts) usize {
    var n: usize = 0;
    for (layers) |*l| if (l.exl3) |x| if (x.pool != null) {
        out[n] = x;
        n += 1;
    };
    if (mtp) |l| if (l.exl3) |x| if (x.pool != null) {
        out[n] = x;
        n += 1;
    };
    return n;
}
