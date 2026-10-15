//! GLM_RESIDENCY's opt-in residency set (weights, arena and image tower, within the load bound) and the keepalive.
const std = @import("std");
const mtl = @import("metal");
const Engine = @import("engine.zig").Engine;

/// The set when GLM_RESIDENCY asks for one and its bytes fit the bound, then the idle keepalive's target either way.
pub fn wire(e: *Engine) void {
    // opt-in: wiring 181 GB leaves macOS nothing to reclaim if another model shares the Mac (Flash Next runs without)
    const want = std.c.getenv("GLM_RESIDENCY") != null and mtl.ResidencySet.within("glm", 0, bytes(e));
    if (!want) {} else if (e.device.residencySet(e.w.buffers.items.len + e.arena.buffers.items.len + if (e.vision) |v| v.buffers().len else 0)) |set| {
        for (e.w.buffers.items) |b| set.add(b);
        for (e.arena.buffers.items) |b| set.add(b);
        if (e.vision) |v| for (v.buffers()) |b| set.add(b);
        set.commit();
        set.requestResidency();
        e.queue.addResidencySet(set);
        e.residency = set;
    } else |_| {}
    // every Metal engine offers its queue to the idle keepalive; the sets list only when the model is resident
    if (e.residency) |set| e.keepalive_sets[0] = set;
    e.keepalive_target = .{ .queue = e.queue, .sets = if (e.residency != null) &e.keepalive_sets else &.{} };
}

/// What the set would wire: every weight and arena buffer, and the image tower's when it is loaded.
fn bytes(e: *const Engine) u64 {
    var n: u64 = 0;
    for (e.w.buffers.items) |b| n += b.length();
    for (e.arena.buffers.items) |b| n += b.length();
    if (e.vision) |v| {
        for (v.buffers()) |b| n += b.length();
    }
    return n;
}
