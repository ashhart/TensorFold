//! Mini's five-way projection groups under the contract: each member on its own inputs against its own linear_body.
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

/// A control's planted fault in the launch's arguments: members 0 and 1's inputs, or the last two's outputs, swapped.
const Ctl = enum { none, scales, inputs, outputs };

fn runLinear(comptime ctl: Ctl) fn (*x3.Case, []const exl3.Weight, []const u64) anyerror!void {
    return struct {
        fn run(cs: *x3.Case, ws: []const exl3.Weight, ys: []const u64) !void {
            var ms: [pg.max_members]pg.Member = undefined;
            for (ws, ys, 0..) |w, y, i| ms[i] = .{ .w = w, .x = cs.xAt(i), .y = y, .dt = cs.dt };
            if (ctl == .inputs) std.mem.swap(u64, &ms[0].x, &ms[1].x);
            if (ctl == .outputs) std.mem.swap(u64, &ms[ws.len - 2].y, &ms[ws.len - 1].y);
            try group.?.run(cs.rig.s, ms[0..ws.len], cs.shape.rows);
        }
    }.run;
}

/// proj_group's RotC: each member's rotated input straight into the case's output, member after member.
const RotC = extern struct { x: [pg.max_members]u64, suh: [pg.max_members]u64, xh: [pg.max_members]u64, x_dtype: i32, m: i32, k: i32 };

fn runRot(comptime ctl: Ctl) fn (*x3.Case, []const exl3.Weight, []const u64) anyerror!void {
    return struct {
        fn run(cs: *x3.Case, ws: []const exl3.Weight, _: []const u64) !void {
            const rows = cs.shape.rows;
            const k = cs.shape.k;
            var r: RotC = .{ .x = @splat(0), .suh = @splat(0), .xh = @splat(0), .x_dtype = @backingInt(exl3.DType.bf16), .m = @intCast(rows), .k = @intCast(k) };
            for (ws, 0..) |w, i| {
                r.x[i] = cs.xAt(i);
                r.suh[i] = w.suh;
                r.xh[i] = cs.out + i * rows * k * 2;
            }
            if (ctl == .inputs) std.mem.swap(u64, &r.x[0], &r.x[1]);
            var a: cuda.Args = .{};
            a.add(r);
            try cuda.launch.launch(group.?.rot, .{ .grid = .{ .x = @intCast((k / 128 + 3) / 4), .y = @intCast(rows), .z = @intCast(ws.len) }, .block = .{ .x = 128 } }, cs.rig.s, &a);
        }
    }.run;
}

fn ext(comptime rot: bool, comptime ctl: Ctl) *const x3.Ext {
    return &struct {
        const e: x3.Ext = .{ .members = members, .fits = fits, .run = if (rot) runRot(ctl) else runLinear(ctl) };
    }.e;
}

/// A registration: a width (halves), warp count and output type's group5 kernel, or the rotation, or a control.
const Reg = struct { rot: bool = false, h: u32 = 12, wk: u32 = 4, f32: bool = false, ctl: Ctl = .none };
const registrations = [_]Reg{
    .{ .h = 8 },                       .{ .h = 8, .wk = 8 },               .{ .h = 10 },                     .{ .h = 10, .wk = 8 },
    .{ .h = 12 },                      .{ .h = 12, .wk = 8 },              .{ .h = 8, .f32 = true },         .{ .h = 8, .wk = 8, .f32 = true },
    .{ .h = 10, .f32 = true },         .{ .h = 10, .wk = 8, .f32 = true }, .{ .h = 12, .f32 = true },        .{ .h = 12, .wk = 8, .f32 = true },
    .{ .rot = true },                  .{ .ctl = .scales },                .{ .rot = true, .ctl = .scales }, .{ .ctl = .inputs },
    .{ .f32 = true, .ctl = .outputs }, .{ .rot = true, .ctl = .inputs },
};

fn spec(r: Reg) c.Spec {
    var s = x3.spec(if (r.rot) .rot_group else .group, 8, r.ctl == .scales);
    s.served = if (r.wk == 8) &wide else if (r.ctl == .outputs) narrow[0..3] else &narrow;
    if (r.f32) s.out = .f32;
    inline for (.{ 8, 10, 12 }) |h| inline for (.{ 4, 8 }) |wk| inline for (.{ false, true }) |f| if (!r.rot and r.h == h and r.wk == wk and r.f32 == f) {
        s.name = std.fmt.comptimePrint("cuda exl3 tf_exl3_group5_{d}_2_{d}{s} (a visit's projections, 1-16 rows)", .{ h, wk, if (f) " fp32 out" else "" });
    };
    if (r.rot) s.name = "cuda exl3 tf_exl3_rot_group5 (a visit's rotated inputs, 1-16 rows)";
    s.name = switch (r.ctl) {
        .none => s.name,
        .scales => if (r.rot) "control: exl3 rot_group5 with a member's suh swapped posing as rot_in" else "control: exl3 group5 with a member's scales swapped posing as its members' linear_body",
        .inputs => if (r.rot) "control: exl3 rot_group5 with two members' inputs swapped posing as rot_in" else "control: exl3 group5 with two members' inputs swapped posing as their linear_body",
        .outputs => "control: exl3 group5 fp32 out with two members' outputs swapped posing as their linear_body",
    };
    return s;
}

const exts = blk: {
    var es: [registrations.len]*const x3.Ext = undefined;
    for (registrations, &es) |r, *e| e.* = ext(r.rot, r.ctl);
    break :blk es;
};

pub const Set = struct {
    lin: ?exl3.Linear = null,
    cases: [registrations.len]x3.Case = undefined,
    n: usize = 0,

    /// The cases on one Linear and one Group, appended to `list` (the cases point into the set: it must not move).
    pub fn add(self: *Set, gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !void {
        const at = list.items.len;
        errdefer list.shrinkRetainingCapacity(at); // a failed add leaves the list as it found it
        self.lin = try exl3.Linear.load(gpu.d);
        errdefer self.deinit(); // the module, the group and every case made so far
        group = try pg.Group.init(gpu.d, &self.lin.?);
        for (registrations) |r| {
            self.cases[self.n] = .{ .rig = .{ .gpu = gpu, .s = s }, .lin = &self.lin.?, .kind = if (r.rot) .rot_group else .group, .halves = r.h, .control = r.ctl == .scales, .ext = exts[self.n], .dt = if (r.f32) .f32 else .bf16 };
            try list.append(a, .{ .spec = spec(r), .ctx = &self.cases[self.n], .vt = &x3.Case.vt, .control = if (r.ctl != .none) .shapes else null });
            self.n += 1;
        }
    }

    pub fn deinit(self: *Set) void {
        for (self.cases[0..self.n]) |*cs| cs.deinit();
        self.n = 0;
        if (group) |*x| x.deinit();
        group = null;
        if (self.lin) |*l| l.unload();
        self.lin = null;
    }
};

test "contract_inkling2_group5: the output swap control's last two members are one width at every shape" {
    for (registrations) |r| if (r.ctl == .outputs) for (spec(r).served) |s| {
        const ns = members(s.n);
        try std.testing.expect(ns.len == 5 and ns[3] == ns[4]);
    };
}

test "contract_inkling2_group5: each control launches through its own planted fault, every kernel through none" {
    for (registrations, exts) |r, e| for (registrations, exts) |q, f| {
        try std.testing.expectEqual(r.rot == q.rot and r.ctl == q.ctl, e == f);
    };
    _ = &x3.Case.vt;
    _ = &Set.add;
}

test "contract_inkling2_group5: each registration's shapes reach its kernel's width and warps" {
    for (registrations) |r| if (!r.rot) for (spec(r).served) |s| {
        var ws: [pg.max_members]exl3.Weight = undefined;
        const ns = members(s.n);
        for (ns, 0..) |n, i| ws[i] = .{ .trellis = 0, .suh = 0, .svh = 0, .k = @intCast(s.k), .n = @intCast(n), .halves = r.h, .codebook = .mul1, .stride_k = 0, .stride_nb = 0 };
        try std.testing.expect(fits(ws[0..ns.len], s.rows));
        try std.testing.expectEqual(r.wk, exl3.planOf(ws[0]).wk);
    };
}
