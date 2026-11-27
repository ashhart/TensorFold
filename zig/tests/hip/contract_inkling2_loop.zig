//! Inkling 2's HIP loop kernels (zig/kernels/hip/inkling2/loop.hip) at both widths under the contract, with controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const K = @import("inkling2_loop_hip");
const Profile = @import("contract_entry.zig").Profile;
const math = @import("contract_inkling2_loop_math.zig");
const msg = @import("contract_inkling2_loop_msg.zig");
const gate = @import("contract_inkling2_loop_gate.zig");
const ref = @import("contract_inkling2_loop_ref.zig");

const g = core.contract.gates;

/// The `inkling2_loop` contract entry (contract_families.zig): the kernels, then the controls.
pub fn run(a: std.mem.Allocator, _: std.Io, profile: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var k = try K.Kernels.load(&r, hip.kernels.images[@backingInt(hip.kernels.Group.inkling2_loop)]);
    defer k.unload();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    var set: Set = .{};
    defer set.deinit();
    try set.init(a, &r, s, &k, profile, &list);
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6c70, &t);
    return t;
}

/// Every registration, each with its own buffers: the kernels first and the controls after, Mini's width before 256.
pub const Set = struct {
    math: [2]math.Math = undefined,
    msgs: [K.widths.len][6]msg.Msg = undefined,
    gates: [K.widths.len][6]gate.Gate = undefined,
    made: bool = false,

    pub fn init(set: *Set, a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, k: *const K.Kernels, profile: Profile, list: *std.ArrayList(g.Kernel)) !void {
        for (&set.math, 0..) |*x, i| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .native = i == 1 };
        for (&set.msgs, ref.geometries) |*row, geo| for (row, 0..) |*x, i| {
            x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .geo = geo, .which = @fromBackingInt(@intCast(i)), .profile = profile };
        };
        for (&set.gates, ref.geometries) |*row, geo| for (row, 0..) |*x, i| {
            x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .geo = geo, .which = @fromBackingInt(@intCast(i)), .profile = profile };
        };
        set.made = true;
        for (&set.math) |*x| try x.init();
        try list.append(a, set.math[0].kernel());
        for (&set.msgs, &set.gates) |*m, *gt| {
            try list.append(a, m[0].kernel());
            try list.append(a, gt[0].kernel());
        }
        try list.append(a, set.math[1].kernel());
        for (&set.msgs) |*row| for (row[1..]) |*x| try list.append(a, x.kernel());
        for (&set.gates) |*row| for (row[1..]) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.made) return;
        for (&set.math) |*x| x.deinit();
        for (&set.msgs) |*row| for (row) |*x| x.deinit();
        for (&set.gates) |*row| for (row) |*x| x.deinit();
        set.made = false;
    }
};

test {
    _ = @import("contract_inkling2_loop_ref.zig");
}
