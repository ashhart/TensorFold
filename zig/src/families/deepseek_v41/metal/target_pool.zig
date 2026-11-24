//! DeepSeek-V4.1 Metal placement and release of a slot's shared KV run.
const std = @import("std");
const mtl = @import("metal");
const Target = @import("target.zig").Target;
const pool_ = @import("pool.zig");
const pool_bufs = @import("pool_bufs.zig");
const pipelines = @import("pipelines.zig");

pub fn init(t: *Target) !void {
    const setting = t.opt.pool orelse return;
    t.max_new = try t.a.alloc(?u64, t.opt.slots);
    errdefer t.a.free(t.max_new);
    @memset(t.max_new, null);
    t.pool = try pool_.Pool.init(t.a, t.opt.slots, setting.blocks);
    errdefer t.pool.?.deinit();
    t.pbufs = try pool_bufs.Bufs.init(t.a, t.env.device, t.env.c, setting.blocks);
}

pub fn deinit(t: *Target) void {
    if (t.pbufs) |*b| b.deinit();
    if (t.pool) |*p| p.deinit();
    t.a.free(t.max_new);
}

/// Shared buffers join the residency set once, regardless of the number of slots that bind them.
pub fn appendBuffers(t: *const Target, out: *std.ArrayList(mtl.Buffer)) !void {
    const b = t.pbufs orelse return;
    for (b.pools) |pbuf| if (pbuf) |buf| try out.append(t.a, buf);
    for (b.keys) |kbuf| if (kbuf) |buf| try out.append(t.a, buf);
}

/// Before `prefill`: the request's max new tokens size the slot's run (without it, the run is the whole window).
pub fn admit(ptr: *anyopaque, slot: u32, max_new: u64) void {
    const t = Target.self(ptr);
    if (slot < t.max_new.len) t.max_new[slot] = max_new;
}

/// The slot's run for a prompt of `n` ids, its max new tokens and a window of drafts; LaneMemory: not now.
pub fn place(t: *Target, slot: u32, n: usize) !void {
    const p = if (t.pool) |*x| x else return;
    const s = &t.slots[slot];
    const mn = t.max_new[slot];
    t.max_new[slot] = null;
    giveBack(t, slot);
    const want = @min(t.opt.limit, @as(u64, n) +| (mn orelse t.opt.limit) +| pipelines.max_rows);
    const plan = p.take(slot, want) catch |e| return if (e == error.LaneMemory) error.LaneMemory else error.PastLimit;
    const c = t.env.c;
    if (plan.moves.len > 0) {
        const t0 = mtl.clock.seconds();
        t.pbufs.?.applyMoves(c, plan.moves);
        for (plan.moves) |m| t.slots[m.seat].kv.bind(c, @as(u64, m.to) * pool_.block, t.slots[m.seat].kv.limit);
        const move_ms = (mtl.clock.seconds() - t0) * 1e3;
        t.pbufs.?.applyZero(c, plan.zero);
        const total_ms = (mtl.clock.seconds() - t0) * 1e3;
        var blocks: u64 = 0;
        for (plan.moves) |m| blocks += m.blocks;
        const mib = @as(f64, @floatFromInt(blocks * pool_bufs.bytesPerBlock(c))) / (1 << 20);
        if (!@import("builtin").is_test) std.log.info("dsv41 pool: compacted {d} run(s) ({d} blocks, {d:.1} MiB) in {d:.2} ms ({d:.2} ms with zero)", .{ plan.moves.len, blocks, mib, move_ms, total_ms });
    } else {
        t.pbufs.?.apply(c, plan);
    }
    s.kv.bind(c, @as(u64, plan.run.start) * pool_.block, @intCast(@min(t.opt.limit, pool_.tokensOf(plan.run.blocks))));
    s.high = 0; // the run reads as zeros past what this request writes
}

/// The slot's run goes back to the pool (the blocks its writes reached are zeroed for the next run).
pub fn giveBack(t: *Target, slot: u32) void {
    const p = if (t.pool) |*x| x else return;
    const s = &t.slots[slot];
    if (p.runs[slot] == null) return;
    p.give(slot, s.high);
    s.kv.bind(t.env.c, 0, 0);
}

/// `giveBack` after a failed prompt: every block of the run counts as written.
pub fn giveBackAll(t: *Target, slot: u32) void {
    const p = if (t.pool) |*x| x else return;
    const r = p.runs[slot] orelse return;
    t.slots[slot].high = pool_.tokensOf(r.blocks);
    giveBack(t, slot);
}
