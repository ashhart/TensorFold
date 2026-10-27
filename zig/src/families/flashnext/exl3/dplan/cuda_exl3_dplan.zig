//! The offload pool planned on the device (flashnext_exl3_dplan.cu): a decode round's MoE needs no host wait between

const std = @import("std");
const cuda = @import("cuda");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const offload = pool_.offload;
const telemetry = @import("../model/exl3_telemetry.zig");
const recent_demand = @import("../offload/exl3_recent_demand.zig");
const layout = @import("exl3_dplan_layout.zig");
pub const demand_trace = @import("exl3_dplan_trace.zig");
pub const replay = @import("exl3_dplan_replay.zig");
pub const policy_eval = @import("exl3_dplan_policy_eval.zig");

const experts_n = layout.experts_n;
const misc_words = layout.misc_words;
const recentCountsOffset = layout.recentCountsOffset;
pub const miscOffset = layout.miscOffset;
pub const PoolState = layout.PoolState;
const empty_id = offload.Cache.empty_id;

comptime {
    if (recent_demand.expert_count != experts_n) @compileError("recent expert history shape mismatch");
}

test {
    _ = recent_demand;
    _ = layout;
    _ = demand_trace;
    _ = replay;
    _ = @import("exl3_dplan_trace_test.zig");
    _ = @import("exl3_dplan_trace_replay_test.zig");
    _ = @import("cuda_exl3_dplan_test.zig");
    _ = @import("exl3_dplan_policy_eval_test.zig");
    _ = policy_eval;
}

/// u32 words of one layer's device state.
pub const stateWords = layout.stateWords;

/// The planned copies a round can hold: a row's ten routed picks each.
fn workCap(max_rows: usize) usize {
    return max_rows * 10;
}

pub const Owner = enum { host, device };

/// Device statistics of a layer (the misc words after now).
pub const Misc = extern struct { now: u32, rounds: u32, hit_slots: u32, miss_slots: u32, copies: u32, evictions: u32, overflow: u32, inplace: u32, recent_cursor: u32, recent_filled: u32 };

pub const DevPlan = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    plan_fn: cuda.Function,
    copy_fn: cuda.Function,
    restore_fn: cuda.Function,
    pools: []*pool_.Pool,
    pool_states: []PoolState,
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
    /// Overlap: the held experts compute while the misses' copies run on `side`, then the misses' experts.
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
    history_initialized: bool = false,
    /// Experts read in place from the host bank (more misses than free lines in a round), summed at each pull.
    inplace: u64 = 0,
    trace_mem: ?cuda.DeviceBuffer = null,
    trace_checkpoint_fn: ?cuda.Function = null,

    const static_bytes = layout.static_bytes;

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, pools: []const *pool_.Pool) !*DevPlan {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        if (pools.len == 0) return error.Invalid;
        const pool_states = try gpa.alloc(PoolState, pools.len);
        errdefer gpa.free(pool_states);
        var total_words: usize = 0;
        for (pools, pool_states) |p, *entry| {
            if (p.lines.spec.experts < experts_n) return error.Invalid;
            entry.* = try PoolState.init(p.opt.lines, total_words);
            total_words = entry.endWords();
        }
        var mod = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_dplan);
        errdefer mod.unload();
        const dp = try gpa.create(DevPlan);
        errdefer gpa.destroy(dp);
        const own = try gpa.dupe(*pool_.Pool, pools);
        errdefer gpa.free(own);
        var state = try cuda.DeviceBuffer.alloc(d, total_words * @sizeOf(u32));
        errdefer state.free();
        var mirror = try cuda.HostBuffer.alloc(d, state.len);
        errdefer mirror.free();
        var statics = try cuda.DeviceBuffer.alloc(d, pools.len * static_bytes);
        errdefer statics.free();
        // rows a planned round may hold (MTP verify windows reach 16): more distinct misses than lines free of the
        const max_rows = maxRows();
        var trace_mem: ?cuda.DeviceBuffer = null;
        errdefer if (trace_mem) |*buffer| buffer.free();
        if (traceEnabled()) {
            for (pools, pool_states, 0..) |p, entry, index| _ = try demand_trace.header(index, p.opt.layer, entry.lines, max_rows);
            trace_mem = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, pools.len, demand_trace.pool_bytes));
            try trace_mem.?.fill32(0, null);
            for (pools, pool_states, 0..) |p, entry, index| {
                const header = try demand_trace.header(index, p.opt.layer, entry.lines, max_rows);
                try trace_mem.?.upload(index * demand_trace.pool_bytes, std.mem.asBytes(&header));
            }
        }
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
            .pool_states = pool_states,
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
            .overlap = true, // the held experts compute beside the misses' copies: +3-4% decode on two RTX 5060 Ti (see the report)
            .trace_mem = trace_mem,
            .trace_checkpoint_fn = if (trace_mem != null) try mod.function("fn_exl3_dcheckpoint") else null,
        };
        if (dp.overlap) dp.copy_blocks = overlapCopyBlocks(d) catch dp.copy_blocks;
        if (std.c.getenv("TF_FLASHNEXT_EXL3_DCOPY_BLOCKS")) |v| dp.copy_blocks = std.fmt.parseInt(u32, std.mem.span(v), 10) catch dp.copy_blocks;
        return dp;
    }

    /// With the trace on and TF_FLASHNEXT_EXL3_DPLAN_TRACE_DIR set: each pool's raw buffer as dplan-<pid>-pool<N>.bin.
    fn dumpTrace(dp: *DevPlan) void {
        const dir = std.c.getenv("TF_FLASHNEXT_EXL3_DPLAN_TRACE_DIR") orelse return;
        const buffer = dp.trace_mem orelse return;
        if (dp.stream) |s| s.synchronize() catch return;
        const bytes = std.heap.page_allocator.alloc(u8, demand_trace.pool_bytes) catch return;
        defer std.heap.page_allocator.free(bytes);
        for (0..dp.pools.len) |index| {
            buffer.download(index * demand_trace.pool_bytes, bytes) catch return;
            var path: [256]u8 = undefined;
            const name = std.fmt.bufPrint(&path, "{s}/dplan-{d}-pool{d}.bin\x00", .{ std.mem.span(dir), std.c.getpid(), index }) catch return;
            const fd = std.c.open(@ptrCast(name.ptr), .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
            if (fd < 0) return;
            defer _ = std.c.close(fd);
            var sent: usize = 0;
            while (sent < bytes.len) {
                const n = std.c.write(fd, bytes[sent..].ptr, bytes.len - sent);
                if (n <= 0) break;
                sent += @intCast(n);
            }
        }
    }

    pub fn deinit(dp: *DevPlan) void {
        dp.side.synchronize() catch {};
        dumpTrace(dp);
        if (dp.trace_mem) |*buffer| buffer.free();
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
        dp.gpa.free(dp.pool_states);
        dp.gpa.free(dp.pools);
        dp.gpa.destroy(dp);
    }

    fn mirrorOf(dp: *DevPlan, i: usize) []u32 {
        const all: []u32 = @alignCast(std.mem.bytesAsSlice(u32, dp.mirror.bytes[0..dp.state.len]));
        return dp.pool_states[i].slice(all);
    }

    /// The host's residency up to the device, on `s` (a no-op if the device owns it). Waits for every pool's copies.
    pub fn toDevice(dp: *DevPlan, s: cuda.Stream) !void {
        dp.stream = s;
        if (dp.owner == .device) return;
        for (dp.pools, 0..) |p, i| {
            const lines = dp.pool_states[i].lines;
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
            if (!dp.history_initialized) @memset(m[recentCountsOffset(lines)..miscOffset(lines)], 0);
            @memset(line_of, -1);
            for (0..lines) |ln| {
                const id = p.slots_tab[ln];
                const held = p.line_tab[ln] == .resident and id != empty_id and id < experts_n;
                line_id[ln] = if (held) id else -1;
                stamp[ln] = p.stamp_tab[ln];
                if (held) line_of[id] = @intCast(ln);
            }
            @memcpy(counts, &p.counts[p.opt.layer]);
            const misc: *Misc = @ptrCast(m[miscOffset(lines)..][0..misc_words]);
            const recent_cursor = if (dp.history_initialized) misc.recent_cursor else 0;
            const recent_filled = if (dp.history_initialized) misc.recent_filled else 0;
            misc.* = .{ .now = p.now, .rounds = 0, .hit_slots = 0, .miss_slots = 0, .copies = 0, .evictions = 0, .overflow = 0, .inplace = 0, .recent_cursor = recent_cursor, .recent_filled = recent_filled };
            dp.seen[i] = misc.*;
        }
        try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dp.state.ptr, dp.mirror.bytes.ptr, dp.state.len, s.handle), "cuMemcpyHtoDAsync");
        try dp.traceOwnership(s, true);
        dp.history_initialized = true;
        dp.owner = .device;
        dp.pushes += 1;
    }

    /// The device's residency back to the host pools (a no-op if the host owns it): waits for the last round's stream.
    pub fn toHost(dp: *DevPlan) !void {
        if (dp.owner == .host) return;
        if (dp.stream) |s| try dp.traceOwnership(s, false);
        if (dp.stream) |s| try s.synchronize();
        try dp.side.synchronize();
        try dp.d.check(dp.d.api.cuMemcpyDtoH_v2(dp.mirror.bytes.ptr, dp.state.ptr, dp.state.len), "cuMemcpyDtoH");
        var overflow: u64 = 0;
        for (dp.pools, 0..) |p, i| {
            const lines = dp.pool_states[i].lines;
            const m = dp.mirrorOf(i);
            const line_id: []const i32 = @ptrCast(m[0..lines]);
            const stamp = m[lines .. 2 * lines];
            const counts = m[2 * lines + experts_n ..][0..experts_n];
            const misc: *const Misc = @ptrCast(m[miscOffset(lines)..][0..misc_words]);
            for (0..lines) |ln| {
                const id = line_id[ln];
                p.slots_tab[ln] = if (id < 0) empty_id else @intCast(id);
                p.line_tab[ln] = if (id < 0) .empty else .resident;
                p.stamp_tab[ln] = stamp[ln];
                p.layer_tab[ln] = p.opt.layer;
            }
            @memcpy(&p.counts[p.opt.layer], counts);
            const was = dp.seen[i];
            telemetry.Device.record(&p.telemetry_device, &p.telemetry_valid, misc.*, was, dp.max_rows * 10);
            p.now = misc.now;
            p.stats.rounds += misc.rounds -% was.rounds;
            p.stats.passes += misc.rounds -% was.rounds;
            p.stats.hit_slots += misc.hit_slots -% was.hit_slots;
            p.stats.miss_slots += misc.miss_slots -% was.miss_slots;
            p.stats.misses += misc.copies -% was.copies;
            p.stats.evictions += misc.evictions -% was.evictions;
            overflow += misc.overflow;
            dp.inplace += misc.inplace -% was.inplace;
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
            const misc: *const Misc = @ptrCast(m[miscOffset(dp.pool_states[i].lines)..][0..misc_words]);
            inline for (.{ "rounds", "hit_slots", "miss_slots", "copies", "evictions", "overflow", "inplace" }) |f| @field(t, f) += @field(misc, f) - @field(dp.seen[i], f);
        }
        return t;
    }

    pub fn samplePoolAfterConsumers(dp: *const DevPlan, pool: *const pool_.Pool) !telemetry.Device {
        if (!pool.telemetry_valid) return error.TelemetryCounterRegression;
        var total = pool.telemetry_device;
        if (dp.owner != .device) return total;
        for (dp.pools, 0..) |candidate, index| {
            if (candidate != pool) continue;
            var observed: Misc = undefined;
            const entry = dp.pool_states[index];
            const offset = entry.byteOffset() + miscOffset(entry.lines) * @sizeOf(u32);
            try dp.state.download(offset, std.mem.asBytes(&observed));
            try telemetry.add(telemetry.Device, &total, try telemetry.Device.pendingBounded(observed, dp.seen[index], dp.max_rows * 10));
            return total;
        }
        return total;
    }

    /// Exact additional device allocation; tracing is fixed at init and costs zero bytes when disabled.
    pub fn traceDeviceBytes(dp: *const DevPlan) usize {
        return if (dp.trace_mem) |buffer| buffer.len else 0;
    }

    pub fn traceOwnership(dp: *const DevPlan, stream: cuda.Stream, transfer: bool) !void {
        const buffer = dp.trace_mem orelse return;
        for (dp.pool_states, 0..) |entry, index| {
            var args: cuda.launch.Args = .{};
            args.add(dp.state.ptr + entry.byteOffset());
            args.add(@as(c_int, @intCast(entry.lines)));
            args.add(buffer.ptr + index * demand_trace.pool_bytes);
            args.add(@as(c_int, @intFromBool(transfer)));
            try cuda.launch.launch(dp.trace_checkpoint_fn.?, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, stream, &args);
        }
    }

    /// Caller must join all planner/expert consumers and stop submissions before this blocking diagnostic read.
    pub fn readTraceAfterConsumers(dp: *const DevPlan, index: usize, out: []u32) !demand_trace.View {
        const buffer = dp.trace_mem orelse return error.TraceDisabled;
        if (index >= dp.pools.len or out.len != demand_trace.pool_words) return error.Invalid;
        try buffer.download(index * demand_trace.pool_bytes, std.mem.sliceAsBytes(out));
        return demand_trace.View.init(out);
    }

    /// Allocating diagnostic read with the same joined/quiescent contract.
    pub fn downloadTraceAfterConsumers(dp: *const DevPlan, allocator: std.mem.Allocator, index: usize) !demand_trace.Snapshot {
        if (dp.trace_mem == null) return error.TraceDisabled;
        if (index >= dp.pools.len) return error.Invalid;
        const words = try allocator.alloc(u32, demand_trace.pool_words);
        errdefer allocator.free(words);
        _ = try dp.readTraceAfterConsumers(index, words);
        return .{ .allocator = allocator, .words = words };
    }

    /// The in-place experts of the last plan ([512] int32) and their count after it.
    fn ovfList(dp: *const DevPlan) u64 {
        return dp.work.ptr + workCap(dp.max_rows) * 9 * 8 + 256;
    }

    /// After the round's experts (on `s`): the in-place experts' table entries cleared.
    pub fn restore(dp: *DevPlan, s: cuda.Stream, i: usize) !void {
        if (i >= dp.pools.len) return error.Invalid;
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
    pub fn plan(dp: *DevPlan, s: cuda.Stream, i: usize, picks: u64, rows: usize) !void {
        if (rows == 0 or rows > dp.max_rows) return error.PoolPasses;
        if (i >= dp.pools.len) return error.Invalid;
        const p = dp.pools[i];
        const entry = dp.pool_states[i];
        const st = dp.state.ptr + entry.byteOffset();
        const sb = dp.statics.ptr + i * static_bytes;
        const work = dp.work.ptr;
        const work_n = dp.work.ptr + workCap(dp.max_rows) * 9 * 8;
        var a: cuda.Args = .{};
        a.add(picks);
        a.add(@as(c_int, @intCast(rows * 11)));
        a.add(st);
        a.add(@as(c_int, @intCast(entry.lines)));
        a.add(@as(c_int, @intFromBool(recentEvictionEnabled())));
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
        a.add(if (dp.trace_mem) |buffer| buffer.ptr + i * demand_trace.pool_bytes else @as(u64, 0));
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
/// Set while a decode round runs with device plans (cuda_exl3_graph.zig).
pub var active: ?*DevPlan = null;

/// TF_FLASHNEXT_EXL3_DPLAN: 1 plans decode rounds on the device (implied by TF_FLASHNEXT_EXL3_GRAPH=1).
pub fn enabled() bool {
    if (@import("../graph/cuda_exl3_graph.zig").force) |f| return f != .off;
    if (std.c.getenv("TF_FLASHNEXT_EXL3_DPLAN")) |v| return std.mem.eql(u8, std.mem.span(v), "1");
    return @import("../graph/cuda_exl3_graph.zig").enabled();
}

/// Experimental recent-demand eviction policy. Lifetime counts still persist and seed startup placement.
pub fn recentEvictionEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_RECENT_EVICTION") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

/// TF_FLASHNEXT_EXL3_DPLAN_TRACE=1 enables bounded per-pool demand capture at planner initialization.
fn traceEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_DPLAN_TRACE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

/// The device planner over `layers`' pools (the decoder layers only), made once for `owner`; null when a layer has no
pub fn forLayers(gpa: std.mem.Allocator, d: *const cuda.Driver, owner: *const anyopaque, layers: []const @import("../../cuda_prompt_load.zig").Layer) !?*DevPlan {
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
    if (mtp_pool) |mp| if (n < pools.len) {
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

/// With the split the copies run beside the held experts: a quarter of the SMs, at least 8 blocks, keeps the link busy.
fn overlapCopyBlocks(d: *const cuda.Driver) !u32 {
    var device: c_int = 0;
    var sms: c_int = 0;
    try d.check(d.api.cuDeviceGet(&device, 0), "cuDeviceGet");
    try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "cuDeviceGetAttribute");
    return @max(8, @as(u32, @intCast(sms)) / 4);
}
