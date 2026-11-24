//! Inkling 2's HIP attention (zig/kernels/hip/inkling2) under the contract at Mini's shapes, with planted controls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const K = @import("inkling2_hip");
const Profile = @import("contract_entry.zig").Profile;
const rows = @import("contract_inkling2_rows.zig");
const prep = @import("contract_inkling2_prep.zig");

const g = core.contract.gates;

/// The `inkling2` contract entry (contract_families.zig): prep, the table, each attention path, then the controls.
pub fn run(a: std.mem.Allocator, _: std.Io, profile: Profile) anyerror!g.Tally {
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var k = try K.Kernels.load(&r, hip.kernels.images[@backingInt(hip.kernels.Group.inkling2)]);
    defer k.unload();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    var set: Set = .{};
    try set.init(a, &r, s, &k, profile, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x6b63, &t);
    return t;
}

/// The launches as the forward makes them: prep, the bias table, then the chunks and their merge.
pub const Set = struct {
    attns: [12]rows.Attn = undefined,
    exacts: [2]rows.Attn = undefined,
    preps: [5]prep.Prep = undefined,
    tables: [2]prep.Table = undefined,
    made: bool = false,

    const forms = [_]struct { rows.Path, bool, rows.Plant }{
        .{ .sliding, false, .none }, .{ .fold, false, .none },          .{ .split, false, .none },
        .{ .prompt, false, .none },  .{ .sliding, true, .none },        .{ .fold, true, .none },
        .{ .split, true, .none },    .{ .sliding, false, .late_bias },  .{ .fold, false, .no_sink },
        .{ .prompt, false, .q_ulp }, .{ .sliding, false, .window_one }, .{ .sliding, false, .pre_sigmoid },
    };
    const prep_forms = [_]struct { bool, bool, prep.Fault }{ .{ false, false, .none }, .{ true, false, .none }, .{ true, true, .none }, .{ true, false, .tau_off }, .{ true, false, .cross_write } };

    /// Every registration, its kernels first and its controls after, each with its own buffers.
    pub fn init(set: *Set, a: std.mem.Allocator, r: *const hip.Runtime, s: hip.Stream, k: *const K.Kernels, profile: Profile, list: *std.ArrayList(g.Kernel)) !void {
        for (&set.attns, forms) |*x, f| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .path = f[0], .apart = f[1], .plant = f[2], .profile = profile };
        for (&set.preps, prep_forms) |*x, f| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .global = f[0], .apart = f[1], .fault = f[2] };
        for (&set.tables, 0..) |*x, i| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .late = i == 1 };
        for (&set.exacts, [_]rows.Path{ .sliding, .fold }) |*x, p| x.* = .{ .rig = .{ .a = a, .r = r, .s = s }, .k = k, .path = p, .exact_p = true, .profile = profile };
        set.made = true;
        for (set.preps[0..3]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.tables[0].kernel());
        for (set.attns[0..7]) |*x| try list.append(a, x.kernel());
        for (&set.exacts) |*x| try list.append(a, x.kernel());
        for (set.preps[3..]) |*x| try list.append(a, x.kernel());
        try list.append(a, set.tables[1].kernel());
        for (set.attns[7..]) |*x| try list.append(a, x.kernel());
    }

    pub fn deinit(set: *Set) void {
        if (!set.made) return;
        for (&set.attns) |*x| x.deinit();
        for (&set.exacts) |*x| x.deinit();
        for (&set.preps) |*x| x.deinit();
        for (&set.tables) |*x| x.deinit();
        set.made = false;
    }
};

test {
    _ = @import("contract_inkling2_fill.zig");
}
