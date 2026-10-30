//! The native 27B host inventories nested owners once and follows later allocations on the same shared set.
const std = @import("std");
const mtl = @import("metal");
const fields = mtl.resident.fields;

pub fn appendRunner(r: anytype, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
    try fields.append(a, list, &r.gdn);
    try fields.append(a, list, &r.scratch);
    for (r.bindings) |*binding| try fields.append(a, list, binding);
    try fields.append(a, list, &r.taps);
    try r.readyBuffers(a, list);
}

fn runnerBuffers(r: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    fields.apply(&r.scratch, hook, joining);
    for (r.bindings) |*binding| fields.apply(binding, hook, joining);
    fields.apply(&r.taps, hook, joining);
    for (r.owned.items) |b| fields.one(b, hook, joining);
}

pub fn setRunner(r: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (r.residency) |old| runnerBuffers(r, old, false);
    r.gdn.setResident(hook);
    r.residency = hook;
    if (hook) |f| runnerBuffers(r, f, true);
}

pub fn appendDraft(b: anytype, a: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
    try fields.append(a, list, &b.attention);
    try fields.append(a, list, &b.topk);
    try list.appendSlice(a, b.head.owned.items);
    try list.appendSlice(a, b.frame.pool.items);
    for (b.caches) |*c| try fields.append(a, list, c);
    var adapters = b.prepared.valueIterator();
    while (adapters.next()) |adapter| {
        try fields.append(a, list, adapter);
        try fields.append(a, list, &adapter.projection);
    }
    var tiled = b.tiled.valueIterator();
    while (tiled.next()) |t| try fields.append(a, list, t);
}

fn draftBuffers(b: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    fields.apply(&b.attention, hook, joining);
    fields.apply(&b.topk, hook, joining);
    for (b.head.owned.items) |buffer| fields.one(buffer, hook, joining);
    for (b.caches) |*c| fields.apply(c, hook, joining);
    var adapters = b.prepared.valueIterator();
    while (adapters.next()) |adapter| prepared(adapter, hook, joining);
    var tiled = b.tiled.valueIterator();
    while (tiled.next()) |t| fields.apply(t, hook, joining);
}

pub fn prepared(adapter: anytype, hook: mtl.ResidencySet.Follow, joining: bool) void {
    fields.apply(adapter, hook, joining);
    fields.apply(&adapter.projection, hook, joining);
}

pub fn putPrepared(b: anytype, key: usize, adapter: anytype) !void {
    const old = b.prepared.fetchPut(b.allocator, key, adapter) catch |err| {
        adapter.deinit();
        return err;
    };
    if (old) |entry| {
        if (b.residency) |f| prepared(&entry.value, f, false);
        entry.value.deinit();
    }
    if (b.residency) |f| prepared(&adapter, f, true);
}

pub fn putTiled(b: anytype, key: usize, t: anytype) !void {
    const old = b.tiled.fetchPut(b.allocator, key, t) catch |err| {
        t.deinit();
        return err;
    };
    if (old) |entry| {
        if (b.residency) |f| fields.apply(&entry.value, f, false);
        entry.value.deinit();
    }
    if (b.residency) |f| fields.apply(&t, f, true);
}

pub fn setDraft(b: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (b.residency) |old| draftBuffers(b, old, false);
    b.frame.setResident(hook);
    b.residency = hook;
    if (hook) |f| draftBuffers(b, f, true);
}

pub fn setBack(b: anytype, hook: ?mtl.ResidencySet.Follow) void {
    if (b.residency) |old| fields.apply(b, old, false);
    for (b.contexts) |context| if (context) |x| x.setResident(hook);
    b.residency = hook;
    if (hook) |f| fields.apply(b, f, true);
}

const share_owner = @import("ane_mlp.zig");
pub const ShareBuffers = [1 + 2 * share_owner.max_slices]mtl.Buffer;

/// One inventory for an immediate share and the deferred share's handoff to an existing owner.
pub fn shareBuffers(share: anytype, storage: *ShareBuffers) ![]const mtl.Buffer {
    const n = share.slices;
    if (n == 0 or n > share_owner.max_slices or share.inputs.len != n or share.outputs.len != n)
        return error.BadAneShare;
    storage[0] = share.partial;
    for (share.inputs, share.outputs, 0..) |input, output, i| {
        storage[1 + 2 * i] = input.buffer;
        storage[2 + 2 * i] = output.buffer;
    }
    return storage[0 .. 1 + 2 * n];
}

/// A deferred optional share needs the stable residency owner; refusal unwinds before the model frees buffers.
pub fn finishShare(h: anytype, room: u64) !void {
    if (!h.model.share_pending) return;
    try h.model.loadShare(if (h.resident != null) room else 0);
    if (h.model.ane) |share| {
        var buffers: ShareBuffers = undefined;
        const resident = if (h.resident) |*r| r else return error.AneShareResidencyRefused;
        if (!resident.tryAddNow(try shareBuffers(share, &buffers))) return error.AneShareResidencyRefused;
    }
}

pub fn hold(h: anytype, io: std.Io) !void {
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(h.gpa);
    for (h.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
    try list.appendSlice(h.gpa, h.model.weights.owned.items);
    try h.model.frame.appendBuffers(h.gpa, &list);
    if (h.model.ane) |share| {
        var buffers: ShareBuffers = undefined;
        try list.appendSlice(h.gpa, try shareBuffers(share, &buffers));
    }
    var projections = h.model.kernels.lanes.valueIterator();
    while (projections.next()) |projection| try fields.append(h.gpa, &list, projection);
    try appendRunner(&h.runner, h.gpa, &list);
    try fields.append(h.gpa, &list, h.back);
    for (h.back.contexts) |c| if (c) |x| try x.buffers(&list, h.gpa);
    if (h.disk_stage) |b| try list.append(h.gpa, b);
    if (h.draft) |d| {
        for (d.model.checkpoint.shards.items) |s| try list.append(h.gpa, s.buffer);
        try list.append(h.gpa, d.model.pending);
        try appendDraft(&d.model.backend, h.gpa, &list);
    }
    if (h.back.vision) |v| {
        const held = v.buffers();
        try list.appendSlice(h.gpa, &held);
    }
    h.resident = try mtl.resident.Resident.hold(h.gpa, io, h.model.device, h.model.queue, list.items, "qwen27");
    if (h.resident) |*r| {
        h.model.kernels.setResident(r.follow());
        h.runner.setResident(r.follow());
        setBack(h.back, r.follow());
        if (h.draft) |d| d.model.backend.setResident(r.follow());
    }
    h.warm = if (h.resident) |*r| r.target() else .{ .queue = h.model.queue };
}

pub fn drop(h: anytype) void {
    h.model.kernels.setResident(null);
    h.runner.setResident(null);
    setBack(h.back, null);
    if (h.draft) |d| d.model.backend.setResident(null);
    if (h.resident) |*r| r.deinit();
    h.resident = null;
    h.warm = .{ .queue = h.model.queue };
}

pub fn stage(h: anytype, n: usize) !mtl.Buffer {
    if (h.disk_stage) |b| if (b.length() >= n) return b;
    if (h.disk_stage) |b| {
        if (h.resident) |*r| r.remove(b);
        b.deinit();
    }
    h.disk_stage = null;
    const b = try h.model.device.buffer(@max(n, 32 << 20), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    h.disk_stage = b;
    if (h.resident) |*r| r.add(&.{b});
    return b;
}
