//! Physical snapshots for core.prompt_cache; policy, identity, borrowing, and eviction stay in core.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const api = @import("engine_api");
const Backend = @import("hip_backend.zig").Backend;
const Cache = @import("hip_text.zig").Cache;

const Saved = struct { cache: Cache, logits: hip.DeviceBuffer, at: usize };

pub fn family(h: *Backend) api.prompt_cache.Snapshots {
    return .{ .ptr = h, .vtable = &vtable };
}
const vtable: api.prompt_cache.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restoreFn, .drop = drop };

fn bytes(ptr: *anyopaque, at: u32) u64 {
    const h: *Backend = @ptrCast(@alignCast(ptr));
    return (Cache.bytes(h.e.c, at) catch return std.math.maxInt(u64)) + @as(u64, h.e.c.vocab) * 2;
}

fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const h: *Backend = @ptrCast(@alignCast(ptr));
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoSnapshotOwner));
    const l = try h.lane(s);
    if (l.filled != at or l.logits_at != at) return error.BadSnapshotPosition;
    if (try h.e.room() < bytes(ptr, at)) return error.DotsMemoryBudget;
    const saved = try h.e.gpa.create(Saved);
    errdefer h.e.gpa.destroy(saved);
    saved.cache = try Cache.init(&h.e.driver, h.e.c, at);
    errdefer saved.cache.deinit();
    saved.logits = try hip.DeviceBuffer.alloc(&h.e.driver, @as(usize, h.e.c.vocab) * 2);
    errdefer saved.logits.free();
    errdefer h.e.stream.synchronize() catch {};
    saved.at = at;
    try h.e.slots[l.slot].cache.copyPrefix(&saved.cache, at, h.e.stream);
    try hip.raw.copy(saved.logits, 0, h.e.slots[l.slot].logits.base(), saved.logits.len, h.e.stream.handle);
    try h.e.stream.synchronize();
    return saved;
}

pub fn restore(h: *Backend, s: *lanes.Stream, ptr: *anyopaque) !void {
    const saved: *Saved = @ptrCast(@alignCast(ptr));
    const l = try h.lane(s);
    if (saved.at != s.reuse.at or saved.at > h.e.p.window) return error.BadSnapshotPosition;
    try saved.cache.copyPrefix(&h.e.slots[l.slot].cache, saved.at, h.e.stream);
    try hip.raw.copy(h.e.slots[l.slot].logits, 0, saved.logits.base(), saved.logits.len, h.e.stream.handle);
    try h.e.stream.synchronize();
    l.filled = saved.at;
    l.logits_at = saved.at;
}
fn restoreFn(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    return restore(@ptrCast(@alignCast(ptr)), @ptrCast(@alignCast(owner orelse return error.NoSnapshotOwner)), saved);
}
fn drop(ptr: *anyopaque, saved_ptr: *anyopaque) void {
    const h: *Backend = @ptrCast(@alignCast(ptr));
    const saved: *Saved = @ptrCast(@alignCast(saved_ptr));
    h.e.stream.synchronize() catch {};
    saved.logits.free();
    saved.cache.deinit();
    h.e.gpa.destroy(saved);
}
