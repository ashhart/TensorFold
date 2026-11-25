//! Inkling 2 Mini's HIP routed experts (zig/kernels/hip/inkling2/experts.hip) under the contract, with controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const K = @import("inkling2_experts_hip");
const Profile = @import("contract_entry.zig").Profile;
const factors = @import("contract_inkling2_experts_factors.zig");

const g = core.contract.gates;

/// The `inkling2_experts` contract entry (contract_families.zig): the kernels, then the controls.
pub fn run(a: std.mem.Allocator, _: std.Io, profile: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var k = try K.Kernels.load(&r, hip.kernels.images[@backingInt(hip.kernels.Group.inkling2_experts)]);
    defer k.unload();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    var set: Set = .{};
    try set.init(a, &r, s, &k, profile, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6b65, &t);
    return t;
}

/// Every registration, each with its own buffers: the kernels first and the controls after.
pub const Set = struct {
    factors: [2]factors.Factors = undefined,
    made: bool = false,

    pub fn init(set: *Set, a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, k: *const K.Kernels, _: Profile, list: *std.ArrayList(g.Kernel)) !void {
        for (&set.factors, 0..) |*x, i| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .native = i == 1 };
        set.made = true;
        try list.append(a, set.factors[0].kernel());
        try list.append(a, set.factors[1].kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.made) return;
        for (&set.factors) |*x| x.deinit();
        set.made = false;
    }
};

test {
    _ = @import("contract_inkling2_experts_ref.zig");
}
