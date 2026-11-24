//! Mini's five-way projection groups under the contract: tf_exl3_group5_* and tf_exl3_rot_group5 against linear_body.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const x3 = @import("contract_exl3.zig");
const pg = @import("inkling2").proj_group;

const c = core.contract.spec;
const g = core.contract.gates;
const exl3 = cuda.exl3;

fn sh(rows: usize, n: usize, k: usize) c.Shape {
    return .{ .rows = rows, .n = n, .k = k };
}

/// n names the members at K 2304: a TP=2 rank's q, gate, r (held whole), k, v; one rank's; two wide layers (8 warps).
fn members(n: usize) []const usize {
    return switch (n) {
        4480 => &.{ 1536, 1536, 384, 512, 512 },
        8576 => &.{ 3072, 3072, 384, 1024, 1024 },
        16384 => &.{ 8192, 8192 },
        else => &.{ 1536, 512 },
    };
}

const narrow = [_]c.Shape{ sh(1, 4480, 2304), sh(16, 4480, 2304), sh(7, 8576, 2304), sh(16, 2048, 2304), sh(3, 2048, 2304) };
const wide = [_]c.Shape{ sh(1, 16384, 2304), sh(4, 16384, 2304) };

fn fits(ws: []const exl3.Weight, rows: usize) bool {
    var ms: [pg.max_members]pg.Member = undefined;
    for (ws, 0..) |w, i| ms[i] = .{ .w = w, .x = 0, .y = 0, .dt = .bf16 };
    return pg.Group.fits(ms[0..ws.len], rows);
}

/// The projections' one group (the set's), shared by every case: Group.run as the forward calls it.
var group: ?pg.Group = null;

fn runLinear(cs: *x3.Case, ws: []const exl3.Weight, ys: []const u64) !void {
    var ms: [pg.max_members]pg.Member = undefined;
    for (ws, ys, 0..) |w, y, i| ms[i] = .{ .w = w, .x = cs.xd, .y = y, .dt = .bf16 };
    try group.?.run(cs.rig.s, ms[0..ws.len], cs.shape.rows);
}

/// proj_group's RotC: each member's rotated input straight into the case's output, member after member.
const RotC = extern struct { x: [pg.max_members]u64, suh: [pg.max_members]u64, xh: [pg.max_members]u64, x_dtype: i32, m: i32, k: i32 };

fn runRot(cs: *x3.Case, ws: []const exl3.Weight, _: []const u64) !void {
    const rows = cs.shape.rows;
    const k = cs.shape.k;
    var r: RotC = .{ .x = @splat(0), .suh = @splat(0), .xh = @splat(0), .x_dtype = @backingInt(exl3.DType.bf16), .m = @intCast(rows), .k = @intCast(k) };
    for (ws, 0..) |w, i| {
        r.x[i] = cs.xd;
        r.suh[i] = w.suh;
        r.xh[i] = cs.out + i * rows * k * 2;
    }
    var a: cuda.Args = .{};
    a.add(r);
    try cuda.launch.launch(group.?.rot, .{ .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(rows), .z = @intCast(ws.len) }, .block = .{ .x = 128 } }, cs.rig.s, &a);
}

const lin_ext: x3.Ext = .{ .members = members, .fits = fits, .run = runLinear };
const rot_ext: x3.Ext = .{ .members = members, .fits = fits, .run = runRot };

/// A registration: a width (halves) and warp count's group5 kernel, or the rotation, or a control posing as one.
const Reg = struct { rot: bool = false, h: u32 = 12, wk: u32 = 4, ctl: bool = false };
const registrations = [_]Reg{
    .{ .h = 8 },      .{ .h = 8, .wk = 8 }, .{ .h = 10 },                  .{ .h = 10, .wk = 8 }, .{ .h = 12 }, .{ .h = 12, .wk = 8 },
    .{ .rot = true }, .{ .ctl = true },     .{ .rot = true, .ctl = true },
};

fn spec(r: Reg) c.Spec {
    var s = x3.spec(if (r.rot) .rot_group else .group, 8, r.ctl);
    s.served = if (r.wk == 8) &wide else &narrow;
    inline for (.{ 8, 10, 12 }) |h| inline for (.{ 4, 8 }) |wk| if (!r.rot and r.h == h and r.wk == wk) {
        s.name = if (r.ctl) "control: exl3 group5 with a member's scales swapped posing as its members' linear_body" else std.fmt.comptimePrint("cuda exl3 tf_exl3_group5_{d}_2_{d} (a visit's projections, 1-16 rows)", .{ h, wk });
    };
    if (r.rot) s.name = if (r.ctl) "control: exl3 rot_group5 with a member's suh swapped posing as rot_in" else "cuda exl3 tf_exl3_rot_group5 (a visit's rotated inputs, 1-16 rows)";
    return s;
}

pub const Set = struct {
    lin: ?exl3.Linear = null,
    cases: [registrations.len]x3.Case = undefined,
    n: usize = 0,

    /// The cases on one Linear and one Group, appended to `list` (the cases point into the set: it must not move).
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        self.lin = try exl3.Linear.load(gpu.d);
        group = try pg.Group.init(gpu.d, &self.lin.?);
        for (registrations) |r| {
            self.cases[self.n] = .{ .rig = .{ .gpu = gpu, .s = s }, .lin = &self.lin.?, .kind = if (r.rot) .rot_group else .group, .halves = r.h, .control = r.ctl, .ext = if (r.rot) &rot_ext else &lin_ext };
            try list.append(a, .{ .spec = spec(r), .ctx = &self.cases[self.n], .vt = &x3.Case.vt, .control = if (r.ctl) .shapes else null });
            self.n += 1;
        }
    }

    pub fn deinit(self: *Set) void {
        for (self.cases[0..self.n]) |*cs| cs.deinit();
        if (group) |*x| x.deinit();
        group = null;
        if (self.lin) |*l| l.unload();
        self.lin = null;
    }
};

test "contract_inkling2_group5: each registration's shapes reach its kernel's width and warps" {
    for (registrations) |r| if (!r.rot) for (spec(r).served) |s| {
        var ws: [pg.max_members]exl3.Weight = undefined;
        const ns = members(s.n);
        for (ns, 0..) |n, i| ws[i] = .{ .trellis = 0, .suh = 0, .svh = 0, .k = @intCast(s.k), .n = @intCast(n), .halves = r.h, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
        try std.testing.expect(fits(ws[0..ns.len], s.rows));
        try std.testing.expectEqual(r.wk, exl3.planOf(ws[0]).wk);
    };
}
