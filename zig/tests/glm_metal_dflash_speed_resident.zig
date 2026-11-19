//! The harness's stand-in for a served target's residency set: dummy buffers plus the target's embedding and head.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
pub const Mode = enum { join, none, own, attach };

/// `gib` GiB in `count` buffers (into `out`), with the target's weights in a set on the target's queue unless `unset`.
pub fn target(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, weights: []const mtl.Buffer, gib: u32, count: u32, unset: bool, out: *std.ArrayList(mtl.Buffer)) !?mtl.ResidencySet {
    if (gib == 0) return null;
    const each = (@as(usize, gib) << 30) / count;
    for (0..count) |_| try out.append(gpa, try device.buffer(each, mtl.ResourceOptions.private));
    std.debug.print("  target stand-in: {d} GiB in {d} buffers{s}\n", .{ gib, count, if (unset) ", no set" else ", in a set on the target's queue" });
    if (unset) return null;
    const set = try device.residencySet(count + weights.len);
    for (out.items) |b| set.add(b);
    for (weights) |w| set.add(w);
    set.commit();
    set.requestResidency();
    queue.addResidencySet(set);
    return set;
}

/// A target window's stand-in before a pass: `mb` MiB copied among the dummy buffers on the target's queue, waited on.
pub fn window(queue: mtl.Queue, dummies: []const mtl.Buffer, mb: u32, round: usize) void {
    if (mb == 0 or dummies.len < 2) return;
    const cb = queue.commandBuffer();
    const e = cb.blit();
    var left: usize = @as(usize, mb) << 20;
    var i = round;
    while (left > 0) : (i += 1) {
        const from = dummies[i % dummies.len];
        const to = dummies[(i + 1) % dummies.len];
        const n = @min(left, @min(from.length(), to.length()));
        e.copy(from, 0, to, 0, n);
        left -= n;
    }
    e.end();
    cb.commit();
    cb.wait();
}

/// The drafter's buffers toward `set` as the benchmark `mode` places them.
pub fn place(gpa: std.mem.Allocator, device: mtl.Device, model: anytype, set: ?mtl.ResidencySet, mode: Mode) !void {
    const s = set orelse return;
    var list: std.ArrayList(mtl.Buffer) = .empty;
    defer list.deinit(gpa);
    for (model.checkpoint.shards.items) |sh| try list.append(gpa, sh.buffer);
    if (model.backend.copies) |*c| try c.appendBuffers(&list);
    for (model.backend.caches) |c| try list.appendSlice(gpa, &.{ c.keys, c.values });
    for (model.backend.frame.pool.items) |b| try list.append(gpa, b);
    try list.append(gpa, model.pending);
    switch (mode) {
        .none => {},
        .own => {
            const own = try device.residencySet(list.items.len);
            for (list.items) |b| own.add(b);
            own.commit();
            own.requestResidency();
            model.backend.queue.addResidencySet(own);
        },
        .join => {
            for (list.items) |b| s.add(b);
            s.commit();
        },
        .attach => model.backend.queue.addResidencySet(s),
    }
    std.debug.print("  drafter residency {s} ({d} buffers)\n", .{ @tagName(mode), list.items.len });
}
