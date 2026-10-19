//! Deferred share buffers use the real bounded owner, with allocation and reference lifetimes recorded on the host.
const std = @import("std");
const mtl = @import("metal");
const residency = @import("src/families/qwen27/residency.zig");
const share_owner = @import("src/families/qwen27/ane_mlp.zig");
const Config = @import("src/families/qwen27/config.zig").Config;
const share_fit = @import("src/native/qwen27_share_fit.zig");
const t = std.testing;
const a = t.allocator;

const Share = struct {
    partial: mtl.Buffer,
    inputs: []mtl.ane.Surface,
    outputs: []mtl.ane.Surface,
    slices: u32,

    fn init(device: mtl.Device, slices: u32) !*Share {
        const s = try a.create(Share);
        errdefer a.destroy(s);
        const partial = try device.buffer(slices * 128 * 5120 * 4, mtl.ResourceOptions.shared);
        errdefer partial.deinit();
        const inputs = try a.alloc(mtl.ane.Surface, slices);
        errdefer a.free(inputs);
        const outputs = try a.alloc(mtl.ane.Surface, slices);
        errdefer a.free(outputs);
        var made: usize = 0;
        errdefer for (0..made) |i| {
            inputs[i].buffer.deinit();
            outputs[i].buffer.deinit();
        };
        while (made < slices) : (made += 1) {
            inputs[made] = .{ .buffer = try device.buffer(128 * 5120 * 2, mtl.ResourceOptions.shared) };
            outputs[made] = .{ .buffer = device.buffer(128 * 5120 * 2, mtl.ResourceOptions.shared) catch |err| {
                inputs[made].buffer.deinit();
                return err;
            } };
        }
        s.* = .{ .partial = partial, .inputs = inputs, .outputs = outputs, .slices = slices };
        return s;
    }
    fn deinit(s: *Share) void {
        for (s.inputs, s.outputs) |i, o| {
            i.buffer.deinit();
            o.buffer.deinit();
        }
        s.partial.deinit();
        a.free(s.inputs);
        a.free(s.outputs);
        a.destroy(s);
    }
};
const Hook = struct {
    pub fn setResident(_: *Hook, _: ?mtl.ResidencySet.Follow) void {}
};
const Back = struct { residency: ?mtl.ResidencySet.Follow = null, contexts: [0]?*Hook = .{} };
const Draft = struct { model: *struct { backend: Hook } };
const Model = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: Hook = .{},
    slices: u32,
    ane: ?*Share = null,
    share_pending: bool = true,
    pub fn loadShare(m: *Model, room: u64) !void {
        m.share_pending = false;
        if (room != 0) m.ane = try Share.init(m.device, m.slices);
    }
    fn deinit(m: *Model) void {
        if (m.ane) |s| s.deinit();
    }
};
const Host = struct {
    model: *Model,
    back: *Back,
    runner: Hook = .{},
    draft: ?*Draft = null,
    resident: ?mtl.resident.Resident = null,
    warm: mtl.keepalive.Target = undefined,
};
const End = enum { shutdown, cache_failure, start_failure, join_refusal };

fn lifetime(record: *mtl.Record, slices: u32, end: End) !void {
    var model: Model = .{ .device = record.device(), .queue = record.queue(), .slices = slices };
    defer model.deinit();
    var back: Back = .{};
    var h: Host = .{ .model = &model, .back = &back };
    var inventory: residency.ShareBuffers = undefined;
    const initial: []const mtl.Buffer = if (model.ane) |s| try residency.shareBuffers(s, &inventory) else &.{};
    h.resident = try mtl.resident.Resident.hold(a, t.io, model.device, record.queue(), initial, "qwen27");
    defer residency.drop(&h);
    try t.expect(model.ane == null);
    try t.expectEqual(@as(u64, 64), h.resident.?.held.bytes);
    if (end == .join_refusal) h.resident.?.held.limit = 64 + mtl.ResidencySet.margin;
    try residency.finishShare(&h, std.math.maxInt(u64));
    const loaded = model.ane orelse return error.MissingShare;
    const buffers = try residency.shareBuffers(loaded, &inventory);
    try t.expectEqual(@as(usize, 1 + 2 * slices), buffers.len);
    try t.expectEqual(@as(u64, slices) * (5 << 20) + 64, h.resident.?.held.bytes);
    try t.expectEqual(@as(u32, 2 + 2 * slices), h.resident.?.held.members.count());
    try t.expectEqual(@as(usize, 2), record.set_commits);
    try record.check();
    try t.expect(h.resident.?.tryAddNow(buffers));
    try residency.finishShare(&h, std.math.maxInt(u64));
    try t.expectEqual(@as(usize, 2), record.set_commits);
    try t.expectEqual(@as(u64, slices) * (5 << 20) + 64, h.resident.?.held.bytes);
    switch (end) {
        .shutdown => {},
        .cache_failure => return error.InjectedCacheFailure,
        .start_failure => return error.InjectedStartFailure,
        .join_refusal => unreachable,
    }
}

test "residency owner Qwen27 share: late handoff commits every buffer once and drops references before free" {
    for ([_]u32{ 1, 4, 16 }) |slices| for ([_]End{ .shutdown, .cache_failure, .start_failure }) |end| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        switch (end) {
            .shutdown => try lifetime(&record, slices, end),
            .cache_failure => try t.expectError(error.InjectedCacheFailure, lifetime(&record, slices, end)),
            .start_failure => try t.expectError(error.InjectedStartFailure, lifetime(&record, slices, end)),
            .join_refusal => unreachable,
        }
        try record.allFreed();
    };
}

test "residency owner Qwen27 share: a refused bounded handoff unwinds and a missing owner declines the share" {
    for ([_]u32{ 1, 4, 16 }) |slices| {
        var record: mtl.Record = .{ .gpa = a };
        defer record.deinit();
        try t.expectError(error.AneShareResidencyRefused, lifetime(&record, slices, .join_refusal));
        try record.allFreed();
        var model: Model = .{ .device = record.device(), .queue = record.queue(), .slices = slices };
        defer model.deinit();
        var back: Back = .{};
        var h: Host = .{ .model = &model, .back = &back };
        try residency.finishShare(&h, std.math.maxInt(u64));
        try t.expect(model.ane == null and !model.share_pending);
        try record.allFreed();
    }
}

test "residency owner Qwen27 share: program-only room refuses and exact payload edges include every slice" {
    var c: Config = undefined;
    c.hidden = 5120;
    c.layers = 64;
    const program = share_owner.Share.bytes(c, 256);
    try t.expectEqual(@as(u64, 480 << 20), program);
    const ram: u64 = 32 << 30;
    const ceiling = ram / 100 * 70 - mtl.ResidencySet.margin;
    for ([_]u32{ 1, 4, 16 }) |slices| {
        const total = try share_owner.Share.admissionBytes(c, 256, slices);
        try t.expectEqual(program + @as(u64, slices) * (5 << 20), total);
        try t.expectError(error.MissingAneOverhead, share_owner.Share.totalBytes(c, 256, slices, null));
        try t.expectEqual(total, try share_owner.Share.totalBytes(c, 256, slices, 0));
        try t.expectEqual(total + 1234, try share_owner.Share.totalBytes(c, 256, slices, 1234));
        try t.expectError(error.Overflow, share_owner.Share.totalBytes(c, 256, slices, std.math.maxInt(u64)));
        for ([_]u64{ program, total, total - 1 }, [_]bool{ false, true, false }) |room, fits| {
            const measured = share_fit.room(ram, ceiling - room, 0, .{});
            try t.expectEqual(fits, total <= measured);
        }
    }
    try t.expectError(error.BadAneShare, share_owner.Share.admissionBytes(c, 256, 0));
    try t.expectError(error.BadAneShare, share_owner.Share.admissionBytes(c, 256, 17));
    c.hidden = std.math.maxInt(u32);
    c.layers = std.math.maxInt(u32);
    try t.expectError(error.Overflow, share_owner.Share.admissionBytes(c, std.math.maxInt(u32), 16));
}
