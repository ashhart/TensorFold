//! Paired-edit sensitivity through real nonidentity mHC posts; emitter placement is a separate IR/model check.
//! The paired site (Matt Suiche/weightless' DeepSeek concept): the writer's edit before the mHC post and the fold's
//! edit after it, both adding alpha d; additive convention from llama.cpp.

const std = @import("std");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");

const D = 5120;
const R = 4; // mHC's boundary kernel takes at most 16 rows
const world = 2;
const alpha: f32 = 0.75;
const eps: f32 = 1e-6;

const Prog = struct {
    steer: bool = true, // false: the unsteered program
    writer: bool = true,
    fold: bool = true,
    shift: bool = false, // layer 1 takes layer 0's direction
};

const Inputs = struct {
    dirs: [2][]f32,
    x0: []u16,
    g: [2][]u16, // [world, R, D] a layer's gathered partials
    post: [2][]f32, // [R, 4]
    comb: [2][]f32, // [R, 16], comb[i, j] at 4 i + j
    pre: []f32,
    nw: []f32,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator, seed: u64) !Inputs {
        var rng = ref.Rng.init(seed);
        var s: Inputs = undefined;
        s.a = a;
        for (&s.dirs) |*d| {
            d.* = try a.alloc(f32, D);
            ref.fillDir(.random, &rng, d.*);
        }
        s.x0 = try a.alloc(u16, R * 4 * D);
        for (s.x0) |*v| v.* = ref.spread(&rng);
        for (&s.g, &s.post, &s.comb) |*g, *post, *comb| {
            g.* = try a.alloc(u16, world * R * D);
            for (g.*) |*v| v.* = ref.spread(&rng);
            post.* = try a.alloc(f32, R * 4);
            for (post.*) |*v| v.* = @floatCast(0.5 + rng.unit());
            comb.* = try a.alloc(f32, R * 16);
            for (comb.*) |*v| v.* = @floatCast(0.05 + 0.5 * rng.unit()); // not the identity
        }
        s.pre = try a.alloc(f32, R * 4);
        for (s.pre) |*q| q.* = @floatCast(rng.unit() * 1.5);
        s.nw = try a.alloc(f32, D);
        for (s.nw) |*v| v.* = @floatCast(0.5 + rng.unit());
        return s;
    }

    fn deinit(s: Inputs) void {
        for (s.dirs) |d| s.a.free(d);
        for (s.g) |g| s.a.free(g);
        for (s.post) |p| s.a.free(p);
        for (s.comb) |m| s.a.free(m);
        s.a.free(s.x0);
        s.a.free(s.pre);
        s.a.free(s.nw);
    }
};

/// The device state of one run: the streams (current and next), mHC's scratch and the coefficients.
const Dev = struct {
    cur: rig.Buf,
    nxt: rig.Buf,
    part: rig.Buf,
    coll: rig.Buf,
    cnt: rig.Buf,
    outm: rig.Buf,
    pre: rig.Buf,
    nw: rig.Buf,
    post: [2]rig.Buf,
    comb: [2]rig.Buf,

    fn init(c: rig.Ctx, in: Inputs) !Dev {
        const a = c.gpu.gpa;
        const zeros = try a.alloc(u8, R * 4 * 40 * 32 * 4);
        defer a.free(zeros);
        @memset(zeros, 0);
        var s: Dev = undefined;
        s.cur = try rig.Buf.init(c.gpu, rig.bytes(in.x0));
        s.nxt = try rig.Buf.init(c.gpu, rig.bytes(in.x0));
        s.part = try rig.Buf.init(c.gpu, zeros);
        s.coll = try rig.Buf.init(c.gpu, zeros[0 .. R * D * 2]);
        s.cnt = try rig.Buf.init(c.gpu, zeros[0..16]);
        s.outm = try rig.Buf.init(c.gpu, zeros[0 .. R * D * 2]);
        s.pre = try rig.Buf.init(c.gpu, rig.bytes(in.pre));
        s.nw = try rig.Buf.init(c.gpu, rig.bytes(in.nw));
        for (&s.post, &s.comb, in.post, in.comb) |*p, *m, hp, hm| {
            p.* = try rig.Buf.init(c.gpu, rig.bytes(hp));
            m.* = try rig.Buf.init(c.gpu, rig.bytes(hm));
        }
        return s;
    }

    fn free(s: *Dev) void {
        inline for (.{ "cur", "nxt", "part", "coll", "cnt", "outm", "pre", "nw" }) |f| @field(s, f).free();
        for (&s.post) |*p| p.free();
        for (&s.comb) |*m| m.free();
    }

    /// mHC's post of layer `l` over the gathered partials `gb`: the new streams land in `nxt`, then become `cur`.
    fn post_(s: *Dev, c: rig.Ctx, k: *const dsv41.Kernels, gb: rig.Buf, l: usize) !void {
        const args: dsv41.ops.mhc.Args = .{
            .x = s.cur.ptr(),
            .xs = 4 * D,
            .xout = s.nxt.ptr(),
            .g = gb.ptr(),
            .gr = R * D,
            .world = world,
            .gbf16 = 1,
            .post = s.post[l].ptr(),
            .comb = s.comb[l].ptr(),
            .pre = s.pre.ptr(),
            .part = s.part.ptr(),
            .c = s.coll.ptr(),
            .nw = s.nw.ptr(),
            .nwbf16 = 0,
            .out = s.outm.ptr(),
            .cnt = s.cnt.ptr(),
            .R = R,
            .eps = eps,
        };
        try k.others(c.stream).mhcBoundary(args, 3, false);
        try c.sync();
        std.mem.swap(rig.Buf, &s.cur, &s.nxt);
    }
};

/// The program's final normed rows (R * D words).
fn program(c: rig.Ctx, k: *const dsv41.Kernels, in: Inputs, p: Prog) ![]u16 {
    const a = c.gpu.gpa;
    var s = try Dev.init(c, in);
    defer s.free();
    for (0..2) |l| {
        const d = if (p.shift and l == 1) in.dirs[0] else in.dirs[l];
        var db = try rig.Buf.init(c.gpu, rig.bytes(d));
        defer db.free();
        var gb = try rig.Buf.init(c.gpu, rig.bytes(in.g[l]));
        defer gb.free();
        if (p.steer and p.writer) try rig.writer(c.ops, gb.ptr(), R * D, D, db.ptr(), alpha, R, D, world);
        try s.post_(c, k, gb, l);
        if (p.steer and p.fold) try rig.fold(c.ops, s.cur.ptr(), 4 * D, 0, 0, db.ptr(), alpha, R, D);
        try c.sync();
    }
    var ob = try rig.Buf.init(c.gpu, rig.bytes(in.x0[0 .. R * D]));
    defer ob.free();
    try c.ops.final(.{ .x = s.cur.ptr(), .xs = 4 * D, .pre = s.pre.ptr(), .nw = s.nw.ptr(), .out = ob.ptr(), .eps = eps, .nwbf16 = 0, .R = R, .width = D });
    try c.sync();
    const got = try ob.read(a);
    defer a.free(got);
    return a.dupe(u16, rig.words(got));
}

fn differ(a: []const u16, b: []const u16) usize {
    var n: usize = 0;
    for (a, b) |p, q| {
        if (!ref.same(p, q)) n += 1;
    }
    return n;
}

const Variant = struct { name: []const u8, arith: ref.Arith };

/// Each reduced-precision arithmetic an implementation could use in place of the documented fp32 edit.
const variants = [_]Variant{
    .{ .name = "bf16 direction", .arith = .{ .dir_bf16 = true } },
    .{ .name = "fp16 edit accumulation", .arith = .{ .edit_acc = .f16 } },
    .{ .name = "bf16 edit accumulation", .arith = .{ .edit_acc = .bf16 } },
    .{ .name = "unrounded writer sum", .arith = .{ .sum_rounded = false } },
};

/// Return layer-zero writer, real-post and folded intermediates; the caller owns all three slices.
const Layer0 = struct { o: []u16, posted: []u16, folded: []u16 };

fn layer0(c: rig.Ctx, k: *const dsv41.Kernels, in: Inputs) !Layer0 {
    const a = c.gpu.gpa;
    var s = try Dev.init(c, in);
    defer s.free();
    var db = try rig.Buf.init(c.gpu, rig.bytes(in.dirs[0]));
    defer db.free();
    var gb = try rig.Buf.init(c.gpu, rig.bytes(in.g[0]));
    defer gb.free();
    try rig.writer(c.ops, gb.ptr(), R * D, D, db.ptr(), alpha, R, D, world);
    try c.sync();
    const gk = try gb.read(a);
    defer a.free(gk);
    const o = try a.dupe(u16, rig.words(gk)[0 .. R * D]);
    errdefer a.free(o);
    try s.post_(c, k, gb, 0);
    const pk = try s.cur.read(a);
    defer a.free(pk);
    const posted = try a.dupe(u16, rig.words(pk));
    errdefer a.free(posted);
    try rig.fold(c.ops, s.cur.ptr(), 4 * D, 0, 0, db.ptr(), alpha, R, D);
    try c.sync();
    const fk = try s.cur.read(a);
    defer a.free(fk);
    return .{ .o = o, .posted = posted, .folded = try a.dupe(u16, rig.words(fk)) };
}

pub fn run(c: rig.Ctx, k: *const dsv41.Kernels) !void {
    const a = c.gpu.gpa;
    const in = try Inputs.init(a, 0x7001);
    defer in.deinit();
    const paired = try program(c, k, in, .{});
    defer a.free(paired);
    const again = try program(c, k, in, .{});
    defer a.free(again);
    if (differ(paired, again) != 0) return rig.fail("program: the paired device program repeated differs in {d} words", .{differ(paired, again)});
    check.pass("glp program: the paired writer, real mHC post, fold program over 2 layers (width {d}, {d} rows) repeats bit-identically", .{ D, R });
    const wrong = [_]struct { name: []const u8, p: Prog }{
        .{ .name = "no steering at all", .p = .{ .steer = false } },
        .{ .name = "missing companion: the fold half only", .p = .{ .writer = false } },
        .{ .name = "missing companion: the writer half only", .p = .{ .fold = false } },
        .{ .name = "layer shift: layer 0's direction at layer 1", .p = .{ .shift = true } },
    };
    for (wrong) |w| {
        const got = try program(c, k, in, w.p);
        defer a.free(got);
        const n = differ(got, paired);
        if (n == 0) return rig.fail("program control '{s}' left the final bits equal to the paired result", .{w.name});
        check.pass("glp program control '{s}': {d} of {d} final words differ from the paired result, as planted", .{ w.name, n, paired.len });
    }
    // reduced precision, on the real intermediates of layer 0
    const l0 = try layer0(c, k, in);
    defer {
        a.free(l0.o);
        a.free(l0.posted);
        a.free(l0.folded);
    }
    const ym = try a.alloc(u16, 4 * D);
    defer a.free(ym);
    const ow = try a.alloc(u16, D);
    defer a.free(ow);
    var ranks: [world][]const u16 = undefined;
    for (0..variants.len + 1) |vi| { // 0: the documented arithmetic, then each reduced-precision variant
        const maybe: ?Variant = if (vi == 0) null else variants[vi - 1];
        const arith: ref.Arith = if (maybe) |v| v.arith else .{};
        const name = if (maybe) |v| v.name else "the documented fp32 arithmetic";
        var wdiff: usize = 0;
        var fdiff: usize = 0;
        for (0..R) |r| {
            for (0..world) |rk| ranks[rk] = in.g[0][rk * R * D + r * D ..][0..D];
            ref.writerRow(world, &ranks, in.dirs[0], alpha, arith, ow);
            wdiff += differ(l0.o[r * D ..][0..D], ow);
            ref.foldRow(l0.posted[r * 4 * D ..][0 .. 4 * D], in.dirs[0], alpha, arith, ym, null);
            fdiff += differ(l0.folded[r * 4 * D ..][0 .. 4 * D], ym);
        }
        if (maybe == null) {
            if (wdiff + fdiff != 0) return rig.fail("program: the device writer and fold are off the host model on the real post's streams ({d} + {d} words)", .{ wdiff, fdiff });
            check.pass("glp program: on layer 0's real intermediates the device writer and fold equal the host model of the documented arithmetic ({d} + {d} words)", .{ R * D, R * 4 * D });
            continue;
        }
        const seen = wdiff + fdiff;
        if (seen == 0) return rig.fail("reduced-precision control '{s}' equals the kernels' bits on the real intermediates: the kernels would not be told from it", .{name});
        check.pass("glp reduced-precision control '{s}': the host model of it differs from the kernels in {d} words of layer 0, as planted", .{ name, seen });
    }
}
