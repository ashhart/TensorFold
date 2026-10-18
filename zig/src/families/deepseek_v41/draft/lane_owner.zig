// Allocation and destruction for the family backend. Scheduling stays in core.lanes.Engine.
const std = @import("std");
const ln = @import("lanes.zig");
const iface = @import("iface.zig");
const costs = @import("costs.zig");
const Policy = @import("policy.zig").Policy;
const Spec = @import("spec.zig").Spec;
pub fn init(gpa: std.mem.Allocator, target: iface.Target, pass: iface.Pass, prices: costs.Costs, opts: ln.Options) !*ln.Lanes {
    const b = opts.shape.block;
    const c = opts.shape.candidates;
    if (b == 0 or b > ln.max_block or c == 0 or c > 1024 or opts.slots == 0 or opts.slots > 16 or opts.siblings > @import("tree.zig").max_siblings) return error.BadOptions;
    const x = try gpa.create(ln.Lanes);
    errdefer gpa.destroy(x);
    var policy = try Policy.init(gpa, prices, opts);
    errdefer policy.deinit();
    const drafts = try gpa.alloc(u32, opts.slots * b);
    errdefer gpa.free(drafts);
    const conf = try gpa.alloc(f32, opts.slots * b);
    errdefer gpa.free(conf);
    const cand = try gpa.alloc(i32, opts.slots * b * c);
    errdefer gpa.free(cand);
    const base = try gpa.alloc(f32, opts.slots * b * c);
    errdefer gpa.free(base);
    const by_slot = try gpa.alloc(?*ln.St, opts.slots);
    errdefer gpa.free(by_slot);
    @memset(by_slot, null);
    x.* = .{ .gpa = gpa, .target = target, .pass = pass, .opts = opts, .policy = policy, .scratch = std.heap.ArenaAllocator.init(gpa), .drafts = drafts, .conf = conf, .cand = cand, .base = base, .by_slot = by_slot };
    x.policy.owner = x;
    if (opts.spec.on) x.spec = try Spec.init(gpa, opts.spec, opts.slots, b, c);
    errdefer if (x.spec) |*sp| sp.deinit();
    errdefer x.free.deinit(gpa);
    try x.free.ensureTotalCapacity(gpa, opts.slots);
    var slot = opts.slots;
    while (slot > 0) : (slot -= 1) x.free.appendAssumeCapacity(slot - 1);
    return x;
}
pub fn deinit(x: *ln.Lanes) void {
    for (x.by_slot) |st| if (st) |owner| {
        owner.shown.deinit(x.gpa);
        x.gpa.destroy(owner);
    };
    x.streams.deinit(x.gpa);
    x.free.deinit(x.gpa);
    x.drawn.deinit(x.gpa);
    x.policy.deinit();
    x.scratch.deinit();
    x.gpa.free(x.drafts);
    x.gpa.free(x.conf);
    x.gpa.free(x.cand);
    x.gpa.free(x.base);
    if (x.spec) |*sp| sp.deinit();
    x.gpa.free(x.by_slot);
    x.gpa.destroy(x);
}
