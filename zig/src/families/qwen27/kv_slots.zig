//! Grow-as-used K/V for a runner's slots (TF_Q27_KV_GROW): grow before rows are written, give back once a slot is free.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const Runner = @import("decode_round.zig").Runner;
const gpuFailed = @import("decode_round.zig").gpuFailed;
const attn = @import("attention.zig");
const kv_grow = @import("kv_grow.zig");

/// Slot `slot` holds `need` K/V rows: layer by layer, a wider pair takes every old byte at its place (waited blit).
pub fn reserve(r: *Runner, slot: u32, need: u32) !void {
    const s = r.kv_step orelse return;
    if (slot >= r.ready) return error.SlotNotReady;
    const held = r.heldRows(slot);
    if (need <= held) return;
    if (need > r.capacity) return error.AttentionCapacity;
    r.settleSlot(slot);
    if (r.failed or r.active != null) return error.RoundNotReady;
    const rows = kv_grow.rows(held, need, s, r.capacity);
    const layers = r.caches.len / r.slots;
    const bytes = @as(usize, rows) * r.model.config.kvDim() * 2;
    // one layer at a time, so growth holds at most one layer's old pair beside the new rows
    for (0..layers) |layer| {
        const at = layer * r.slots + slot;
        const c = r.caches[at];
        if (c.capacity >= rows) continue; // grown before an earlier failure
        if (c.keys.offset != 0 or c.values.offset != 0 or c.key_stride != 0 or c.value_stride != 0) return error.KvGrowLayout;
        var fresh: [2]mtl.Buffer = undefined;
        fresh[0] = try r.model.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer fresh[0].deinit();
        fresh[1] = try r.model.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer fresh[1].deinit();
        const old = [2]mtl.Buffer{ c.keys.buffer, c.values.buffer };
        {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = r.model.queue.commandBuffer();
            const e = cb.blit();
            var moves: [attn.planning.max_streams]kv_grow.Move = undefined;
            for (old, fresh) |from, to| {
                for (kv_grow.moves(&moves, r.model.config.kv_heads, r.model.config.head_dim, c.capacity, rows)) |mv| e.copy(from, mv.from, to, mv.to, mv.len);
            }
            e.end();
            cb.commit();
            cb.wait();
            if (gpuFailed(cb, "a slot's K/V growth")) return error.RoundGpuFailure;
        }
        swapPair(r, at, fresh, rows);
    }
    std.log.info("qwen27: slot {d} K/V grew from {d} to {d} rows ({d} of {d} MiB held)", .{ slot, held, rows, kv_grow.slotBytes(layers, r.model.config.kvDim(), rows) >> 20, r.slotBytes() >> 20 });
}

/// Slot `slot` back to its first rows once no stream holds it and no blit reads it: its rows are gone (offset 0).
pub fn release(r: *Runner, slot: u32) !void {
    const s = r.kv_step orelse return;
    return refit(r, slot, kv_grow.first(s, r.capacity));
}

/// An emptied slot takes the rows a prompt of `need` tokens fills, at once and without a copy (a fresh pass).
pub fn admit(r: *Runner, slot: u32, need: u32) !void {
    const s = r.kv_step orelse return;
    if (need > r.capacity) return error.AttentionCapacity;
    return refit(r, slot, kv_grow.admitted(need, s, r.capacity));
}

/// Every layer of empty slot `slot` holds exactly `first` rows: new buffers where it holds any other count.
fn refit(r: *Runner, slot: u32, first: u32) !void {
    if (slot >= r.ready) return;
    var held: u32 = 0;
    for (0..r.caches.len / r.slots) |layer| held = @max(held, r.caches[layer * r.slots + slot].capacity);
    if (held == first and r.heldRows(slot) == first) return;
    r.settleSlot(slot);
    if (r.failed) return error.RoundNotReady;
    const bytes = @as(usize, first) * r.model.config.kvDim() * 2;
    for (0..r.caches.len / r.slots) |layer| {
        const at = layer * r.slots + slot;
        const c = r.caches[at];
        if (c.capacity == first) continue;
        var fresh: [2]mtl.Buffer = undefined;
        fresh[0] = try r.model.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer fresh[0].deinit();
        fresh[1] = try r.model.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        swapPair(r, at, fresh, first);
    }
    r.offsets[slot] = 0;
    std.log.info("qwen27: slot {d} K/V refit from {d} to {d} rows", .{ slot, held, first });
}

/// Cache `at` takes `fresh` (`rows` each); its old pair leaves the residency set and is freed.
fn swapPair(r: *Runner, at: usize, fresh: [2]mtl.Buffer, rows: u32) void {
    const c = r.caches[at];
    for ([_]mtl.Buffer{ c.keys.buffer, c.values.buffer }, fresh) |from, to| {
        for (r.owned.items) |*b| if (b.id == from.id) {
            b.* = to;
        };
        if (r.residency) |f| {
            f.leave(f.ctx, from);
            f.join(f.ctx, to);
        }
        from.deinit();
    }
    r.caches[at] = .{ .keys = .{ .buffer = fresh[0] }, .values = .{ .buffer = fresh[1] }, .capacity = rows };
    if (r.grown) |g| g.added(g.ctx, &fresh);
}

/// One slot's K/V rows across the attention layers.
pub fn slotBytes(r: *const Runner) usize {
    return r.caches.len / r.slots * 2 * @as(usize, r.capacity) * r.model.config.kvDim() * 2;
}

/// Slot `slot`'s K/V rows (the next unready slot, as free slots are taken lowest first); its new buffers.
pub fn ensure(r: *Runner, slot: u32) ![]const mtl.Buffer {
    if (slot < r.ready) return &.{};
    if (slot != r.ready) return error.SlotOutOfOrder;
    const first = r.owned.items.len;
    errdefer {
        for (r.owned.items[first..]) |b| b.deinit();
        r.owned.shrinkRetainingCapacity(first);
    }
    const layers = r.caches.len / r.slots;
    const rows = if (r.kv_step) |s| kv_grow.first(s, r.capacity) else r.capacity;
    for (0..layers) |layer| r.caches[layer * r.slots + slot] = try slotCache(r.allocator, r.model, &r.owned, rows);
    r.ready += 1;
    if (r.residency) |f| for (r.owned.items[first..]) |b| f.join(f.ctx, b);
    return r.owned.items[first..];
}

/// The K/V rows slot `slot` holds now: the fewest any attention layer holds.
pub fn heldRows(r: *const Runner, slot: u32) u32 {
    var rows: u32 = std.math.maxInt(u32);
    for (0..r.caches.len / r.slots) |layer| rows = @min(rows, r.caches[layer * r.slots + slot].capacity);
    return rows;
}

/// Whether a blit still reads slot `slot` (without waiting for it).
pub fn busy(r: *const Runner, slot: u32) bool {
    const cb = r.reading[slot] orelse return false;
    return switch (cb.status()) {
        .completed, .@"error" => false,
        else => true,
    };
}

/// The K/V buffers of the ready slots (for a residency set).
pub fn readyBuffers(r: *const Runner, a: std.mem.Allocator, out: *std.ArrayList(mtl.Buffer)) !void {
    if (r.kv) |k| return out.appendSlice(a, &.{ k.arena, k.tables });
    for (0..r.caches.len / r.slots) |layer| for (0..r.ready) |slot| {
        const c = r.caches[layer * r.slots + slot];
        try out.appendSlice(a, &.{ c.keys.buffer, c.values.buffer });
    };
}

pub fn slotCache(a: std.mem.Allocator, m: *Model, owned: *std.ArrayList(mtl.Buffer), capacity: u32) !attn.Cache {
    const bytes = @as(usize, capacity) * m.config.kvDim() * 2;
    const key = try allocation(a, m.device, owned, bytes);
    const value = try allocation(a, m.device, owned, bytes);
    return .{ .keys = .{ .buffer = key }, .values = .{ .buffer = value }, .capacity = capacity };
}

fn allocation(a: std.mem.Allocator, device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(a, b) catch |err| {
        b.deinit();
        return err;
    };
    return b;
}
