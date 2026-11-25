//! Inkling 2 Mini's HIP routed experts (zig/kernels/hip/inkling2/experts.hip) under the contract, with controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const K = @import("inkling2_experts_hip");
const Profile = @import("contract_entry.zig").Profile;
const factors = @import("contract_inkling2_experts_factors.zig");
const moe = @import("contract_inkling2_experts_moe.zig");

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
    cells: [forms.len]moe.Moe = undefined,
    made: bool = false,

    /// Each cell's kind, planted fault and whether it covers every decode pair; the controls register last.
    const forms = [_]struct { moe.Kind, moe.Plant, bool }{
        .{ .nv_down, .none, false },
        .{ .mx_down, .none, false },
        .{ .mx_down, .none, true },
        .{ .nv_down, .none, true },
        .{ .nv_down, .late_scale, false },
        .{ .nv_gu, .none, false },
        .{ .mx_gu, .none, false },
        .{ .nv_gu, .halves, false },
        .{ .mx_gu, .x_ulp, false },
        .{ .combine, .none, false },
        .{ .combine, .bf16_weights, false },
    };

    pub fn init(set: *Set, a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, k: *const K.Kernels, profile: Profile, list: *std.ArrayList(g.Kernel)) !void {
        for (&set.factors, 0..) |*x, i| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .native = i == 1 };
        for (&set.cells, forms) |*x, f| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .kind = f[0], .plant = f[1], .decode = f[2], .profile = profile };
        set.made = true;
        try list.append(a, set.factors[0].kernel());
        for (&set.cells) |*x| if (x.plant == .none) try list.append(a, x.kernel());
        try list.append(a, set.factors[1].kernel());
        for (&set.cells) |*x| if (x.plant != .none) try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.made) return;
        for (&set.factors) |*x| x.deinit();
        for (&set.cells) |*x| x.deinit();
        set.made = false;
    }
};

test {
    _ = @import("contract_inkling2_experts_ref.zig");
}
