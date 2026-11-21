//! CUDA GLPv3 additive contracts and fault controls: each output equals glp_ref's fp32 order model and the FP64 bound.

// Format and paired site: Matt Suiche/weightless (spec b181a07d3980fc48abe39d95ddb044459dd0b868).
const std = @import("std");
// Additive convention: llama.cpp's; Arditi et al. (NeurIPS 2024) is background only.
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const images = @import("contract_deepseek_images.zig");
const ref = @import("deepseek_v41/glp_ref.zig");
const fix = @import("contract_glp_fixture.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows_ = core.contract.rows;

// Mirror glp_ops.zig's by-value launch arguments (mode-less, 56 / 48 / 56 bytes) without importing the family module.
const StreamArgs = extern struct { x: u64, xs: i64, tap: u64 = 0, ts: i64 = 0, dir: u64, alpha: f32, R: c_int, width: c_int };
const WriterArgs = extern struct { g: u64, gr: i64, gs: i64, dir: u64, alpha: f32, R: c_int, width: c_int, world: c_int };
const FinalArgs = extern struct { x: u64, xs: i64, pre: u64, nw: u64, out: u64, eps: f32, nwbf16: c_int, R: c_int, width: c_int };

comptime {
    std.debug.assert(@sizeOf(StreamArgs) == 56 and @offsetOf(StreamArgs, "dir") == 32 and @offsetOf(StreamArgs, "alpha") == 40 and @offsetOf(StreamArgs, "R") == 44 and @offsetOf(StreamArgs, "width") == 48);
    std.debug.assert(@sizeOf(WriterArgs) == 48 and @offsetOf(WriterArgs, "dir") == 24 and @offsetOf(WriterArgs, "alpha") == 32 and @offsetOf(WriterArgs, "R") == 36 and @offsetOf(WriterArgs, "world") == 44);
    std.debug.assert(@sizeOf(FinalArgs) == 56 and @offsetOf(FinalArgs, "eps") == 40 and @offsetOf(FinalArgs, "R") == 48);
}

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    fx: fix.Fixture,
    name: []const u8,
    control: ?g.Gate = null,
    x: u64 = 0,
    dptr: u64 = 0,
    pre_ptr: u64 = 0,
    nw_ptr: u64 = 0,
    out: u64 = 0,
    msg: [192]u8 = undefined,

    pub fn spec(t: *const Case) c.Spec {
        return fix.specOf(t.fx.kind, t.name);
    }

    pub fn kernel(t: *Case) g.Kernel {
        return .{ .spec = t.spec(), .ctx = t, .vt = &vt, .control = t.control };
    }
    const vt = g.hooks(Case);

    pub fn deinit(t: *Case) void {
        t.fx.free(t.rig.gpu.gpa);
        t.rig.deinit();
    }

    pub fn prepare(t: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!t.fx.admits(shape)) return false;
        t.rig.reset();
        const a = t.rig.gpu.gpa;
        try t.fx.build(a, shape, fill, seed);
        const w = shape.n;
        const R = shape.rows;
        if (t.fx.kind == .final) {
            t.x = try t.rig.input(std.mem.sliceAsBytes(t.fx.words));
            t.pre_ptr = try t.rig.input(std.mem.sliceAsBytes(t.fx.pre));
            t.nw_ptr = try t.rig.input(t.fx.nw);
            t.out = try t.rig.output(R * w * 2);
        } else {
            t.dptr = try t.rig.input(std.mem.sliceAsBytes(t.fx.d));
            t.x = try t.rig.output(t.fx.count() * 2); // in place: the copies, then the tap right after them
            try t.upload();
        }
        return true;
    }

    fn upload(t: *Case) !void {
        const b: cuda.DeviceBuffer = .{ .d = t.rig.gpu.d, .ptr = t.x, .len = t.fx.count() * 2 };
        try b.upload(0, std.mem.sliceAsBytes(t.fx.words));
    }

    pub fn run(t: *Case) !void {
        if (t.fx.kind != .final) try t.upload(); // in place: every run starts from the prepared bytes
        const w: c_int = @intCast(t.fx.shape.n);
        const R: c_int = @intCast(t.fx.shape.rows);
        var args: cuda.Args = .{};
        switch (t.fx.kind) {
            .stream => {
                const tap: u64 = if (t.fx.tap) t.x + @as(u64, t.fx.shape.rows) * 4 * t.fx.shape.n * 2 else 0;
                args.add(StreamArgs{ .x = t.x, .xs = 4 * w, .tap = tap, .ts = w, .dir = t.dptr, .alpha = t.fx.alpha, .R = R, .width = w });
            },
            .writer => args.add(WriterArgs{ .g = t.x, .gr = @as(i64, R) * w, .gs = w, .dir = t.dptr, .alpha = t.fx.alpha, .R = R, .width = w, .world = @intCast(t.fx.world) }),
            .final => args.add(FinalArgs{ .x = t.x, .xs = 4 * w, .pre = t.pre_ptr, .nw = t.nw_ptr, .out = t.out, .eps = fix.eps, .nwbf16 = @intFromBool(t.fx.nw16), .R = R, .width = w }),
        }
        const shared: u32 = if (t.fx.kind == .final) @intCast(2 * t.fx.shape.n) else 0;
        const block: u32 = @intCast(ref.nt);
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(t.fx.shape.rows) }, .block = .{ .x = block }, .shared = shared }, t.rig.s, &args);
        try t.rig.s.synchronize();
    }

    pub fn count(t: *Case) usize {
        return t.fx.count();
    }

    pub fn reference(t: *Case, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        t.fx.reference(want);
    }

    /// Preserve exact stress allowances (the final's gate 4); ordinary bounds are the fixture's.
    pub fn bounds(t: *Case, want: []const f64, _: []const f64, allow: []f64) !void {
        if (t.fx.stress) return;
        t.fx.bounds(want, allow);
    }

    pub fn simulate(t: *Case, acc: c.Accum, order: rows_.Order, out: []f64) !void {
        try t.fx.simulate(t.rig.gpu.gpa, acc, order, out);
    }

    pub fn check(t: *Case, at: g.Gate) !?[]const u8 {
        const a = t.rig.gpu.gpa;
        const n = t.fx.count();
        const out = try t.rig.outputBytes();
        const got = try a.alloc(u16, n);
        defer a.free(got);
        for (got, 0..) |*v, i| v.* = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
        const j = try t.fx.judge(a, got);
        if (at == .nonfinite and (j.planted == 0 or j.planted == n))
            return std.fmt.bufPrint(&t.msg, "the planted values make {d} of {d} reference outputs non-finite: no proof", .{ j.planted, n }) catch "no proof";
        if (j.off_model + j.off_fp64 == 0) return null;
        return std.fmt.bufPrint(&t.msg, "{d} of {d} words differ from the fp32 order model and {d} leave the FP64 bound, first at {?d}", .{ j.off_model, n, j.off_fp64, j.first }) catch "outputs differ";
    }
};

/// One registration: production kernels, then each control's fold and writer; `sum` breaks the tap, `rank_order` ranks.
const Reg = struct { name: []const u8, kind: fix.Kind, tap: bool = false, world: usize = 1, cancel: bool = false, nw16: bool = false, ctl: ?fix.Ctl = null };

const regs = [_]Reg{
    .{ .name = "cuda dsv41 glp add fold", .kind = .stream },
    .{ .name = "cuda dsv41 glp add fold with tap", .kind = .stream, .tap = true },
    .{ .name = "cuda dsv41 glp add writer, 3 ranks", .kind = .writer, .world = 3 },
    .{ .name = "cuda dsv41 glp add writer, 3 cancelling ranks", .kind = .writer, .world = 3, .cancel = true },
    .{ .name = "cuda dsv41 glp add final, fp32 weights", .kind = .final },
    .{ .name = "cuda dsv41 glp add final, bf16 weights", .kind = .final, .nw16 = true },
    .{ .name = "control: dsv41 glp add fold tap sum", .kind = .stream, .tap = true, .ctl = .sum },
    .{ .name = "control: dsv41 glp add fold half", .kind = .stream, .ctl = .half },
    .{ .name = "control: dsv41 glp add writer half", .kind = .writer, .world = 2, .ctl = .half },
    .{ .name = "control: dsv41 glp add fold drop_tail", .kind = .stream, .ctl = .drop_tail },
    .{ .name = "control: dsv41 glp add writer drop_tail", .kind = .writer, .world = 3, .ctl = .drop_tail },
    .{ .name = "control: dsv41 glp add fold drop_nonfinite", .kind = .stream, .ctl = .drop_nonfinite },
    .{ .name = "control: dsv41 glp add writer drop_nonfinite", .kind = .writer, .world = 2, .ctl = .drop_nonfinite },
    .{ .name = "control: dsv41 glp add writer rank_order", .kind = .writer, .world = 3, .cancel = true, .ctl = .rank_order },
    .{ .name = "control: dsv41 glp add fold truncate", .kind = .stream, .ctl = .truncate },
    .{ .name = "control: dsv41 glp add writer truncate", .kind = .writer, .world = 2, .ctl = .truncate },
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    prod: cuda.Module,
    ctl: cuda.Module,
    cases: [regs.len]Case = undefined,
    made: usize = 0,

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var prod = try cuda.Module.load(gpu.d, images.image("dsv41_glp_add") orelse return error.UnknownContractImage);
        errdefer prod.unload();
        var ctl = try cuda.Module.load(gpu.d, images.image("dsv41_glp_add_controls") orelse return error.UnknownContractImage);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .prod = prod, .ctl = ctl };
        errdefer set.release();
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (regs) |r| {
            var sym: [64:0]u8 = undefined;
            const symbol = try std.fmt.bufPrintSentinel(&sym, "dsv41_glp_add_{s}{s}{s}", .{ @tagName(r.kind), if (r.ctl != null) "_" else "", if (r.ctl) |v| @tagName(v) else "" }, 0);
            const f = try (if (r.ctl != null) set.ctl else set.prod).function(symbol);
            set.cases[set.made] = .{
                .rig = rig,
                .f = f,
                .fx = .{ .kind = r.kind, .tap = r.tap, .world = r.world, .nw16 = r.nw16, .cancel = r.cancel },
                .name = r.name,
                .control = if (r.ctl) |v| v.gate() else null,
            };
            set.made += 1;
            try list.append(a, set.cases[set.made - 1].kernel());
        }
        std.debug.assert(set.made == regs.len);
        return set;
    }

    fn release(set: *Set) void {
        for (set.cases[0..set.made]) |*x| x.deinit();
    }

    pub fn deinit(set: *Set) void {
        set.release();
        set.ctl.unload();
        set.prod.unload();
        set.gpa.destroy(set);
    }
};

const sum_orders = [_]rows_.Order{ .sequential, .pairwise, .strided };

// Host witness: each planted fault is emulated by host arithmetic and judged like the device cases, with no device.

fn trunc16(f: f32) u16 {
    if (f != f) return ref.nan16;
    return @intCast(@as(u32, @bitCast(f)) >> 16);
}

/// What the kernel of `ctl` (null: production) would write for the fixture's inputs.
fn emulate(fx: *const fix.Fixture, ctl: ?fix.Ctl, got: []u16) void {
    const w = fx.shape.n;
    const R = fx.shape.rows;
    fx.model(got);
    const k = ctl orelse return;
    for (0..R) |r| switch (fx.kind) {
        .stream => {
            const x = fx.words[r * 4 * w ..];
            const y = got[r * 4 * w ..];
            switch (k) {
                .sum => if (fx.tap) {
                    for (0..w) |col| got[R * 4 * w + r * w + col] = ref.rne32(ref.widen(y[col]) + ref.widen(y[w + col]) + ref.widen(y[2 * w + col]) + ref.widen(y[3 * w + col]));
                },
                .half => ref.foldRow(x[0 .. 4 * w], fx.d, fx.alpha, .{ .edit_acc = .f16 }, y[0 .. 4 * w], null),
                .drop_tail => for (0..4) |cc| {
                    y[cc * w + w - 1] = x[cc * w + w - 1];
                },
                .drop_nonfinite => for (0..4 * w) |i| {
                    const xv = ref.widen(x[i]);
                    y[i] = ref.rne32((if (std.math.isFinite(xv)) xv else 0) + fx.alpha * fx.d[i % w]);
                },
                .rank_order => unreachable, // the fold has no ranks
                .truncate => for (0..4 * w) |i| {
                    y[i] = trunc16(ref.widen(x[i]) + fx.alpha * fx.d[i % w]);
                },
            }
        },
        .writer => {
            const y = got[r * w ..];
            for (0..w) |col| {
                var s = ref.widen(fx.words[r * w + col]);
                for (1..fx.world) |rk| s += ref.widen(fx.words[rk * R * w + r * w + col]);
                var rev = ref.widen(fx.words[(fx.world - 1) * R * w + r * w + col]);
                var rj = fx.world - 1;
                while (rj > 0) {
                    rj -= 1;
                    rev += ref.widen(fx.words[rj * R * w + r * w + col]);
                }
                const add = fx.alpha * fx.d[col];
                y[col] = switch (k) {
                    .sum => unreachable, // the writer has no tap
                    .half => y[col], // set below, as a whole row
                    .drop_tail => if (col == w - 1) ref.rne32(s) else y[col],
                    .drop_nonfinite => blk: {
                        const o = ref.widen(ref.rne32(s));
                        break :blk ref.rne32((if (std.math.isFinite(o)) o else 0) + add);
                    },
                    .rank_order => ref.rne32(ref.widen(ref.rne32(rev)) + add),
                    .truncate => trunc16(ref.widen(ref.rne32(s)) + add), // only the edit's store: the rank sum's round is the canonical pack
                };
            }
            if (k == .half) {
                var ranks: [ref.max_world][]const u16 = undefined;
                for (0..fx.world) |rk| ranks[rk] = fx.words[rk * R * w + r * w ..][0..w];
                ref.writerRow(fx.world, ranks[0..fx.world], fx.d, fx.alpha, .{ .edit_acc = .f16 }, y[0..w]);
            }
        },
        .final => unreachable,
    };
}

test "the registered fixtures tell production from each planted fault on the host" {
    const a = std.testing.allocator;
    var production: usize = 0;
    var red: usize = 0;
    for (regs) |r| {
        if (r.kind == .final) continue;
        var fx: fix.Fixture = .{ .kind = r.kind, .tap = r.tap, .world = r.world, .cancel = r.cancel };
        defer fx.free(a);
        const gate_fill: g.Fill = if (r.ctl) |v| (if (v.gate() == .nonfinite) .nonfinite else .random) else .random;
        for (fix.edit_served) |shape| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
            if (fill == .nonfinite and shape.rows < 5) continue; // gate 1's shape has 5 to 8 rows
            const shaped: c.Shape = if (fill == .nonfinite) .{ .rows = 5, .n = shape.n, .k = shape.k } else shape;
            try fx.build(a, shaped, fill, 0x676c70 + shape.n);
            const got = try a.alloc(u16, fx.count());
            defer a.free(got);
            emulate(&fx, null, got);
            const clean = try fx.judge(a, got);
            if (clean.off_model + clean.off_fp64 != 0) {
                std.debug.print("production {s} at {d}x{d}, {t} fill: {d} off the model, {d} off FP64 (first {?d})\n", .{ r.name, shaped.rows, shaped.n, fill, clean.off_model, clean.off_fp64, clean.first });
                return error.ProductionNotGreen;
            }
            production += 1;
            const v = r.ctl orelse continue;
            if (fill != gate_fill) continue; // each control is held at its own gate's fill
            emulate(&fx, v, got);
            const bad = try fx.judge(a, got);
            const seen = if (v.gate() == .nonfinite) bad.off_fp64 > 0 else bad.off_model + bad.off_fp64 > 0;
            if (!seen) {
                std.debug.print("control {s} at {d}x{d} stays green on the host fixture\n", .{ r.name, shaped.rows, shaped.n });
                return error.ControlNotRed;
            }
            red += 1;
        };
    }
    try std.testing.expect(production > 0 and red > 0);
}

test "the cancelling writer fixture is the only one that tells the rank order" {
    const a = std.testing.allocator;
    var plain: fix.Fixture = .{ .kind = .writer, .world = 3 };
    defer plain.free(a);
    var cancelling: fix.Fixture = .{ .kind = .writer, .world = 3, .cancel = true };
    defer cancelling.free(a);
    var told: [2]bool = undefined;
    for ([_]*fix.Fixture{ &plain, &cancelling }, &told) |fx, *t| {
        try fx.build(a, .{ .rows = 4, .n = 1032, .k = 1032 }, .random, 0x77);
        const got = try a.alloc(u16, fx.count());
        defer a.free(got);
        emulate(fx, .rank_order, got);
        const j = try fx.judge(a, got);
        t.* = j.off_model + j.off_fp64 > 0;
    }
    try std.testing.expect(!told[0] or told[1]);
    try std.testing.expect(told[1]);
}

// Host witnesses separate both reduced types and FP16 RMS sums from exact FP32 at the declared width.
test "gate 4 stress fixture of the final: squares exact in fp32 under 2^24, overflowing fp16 in every order" {
    const w = 5120;
    for ([_]f64{ 16 * 16, 57 * 57 }) |square| {
        const terms: [w]f64 = @splat(square);
        const total = square * w;
        try std.testing.expect(total < 0x1p24 and total > 65504);
        for (sum_orders) |o| {
            try std.testing.expectEqual(total, rows_.sumIn(.f32, o, &terms));
            try std.testing.expect(!std.math.isFinite(rows_.sumIn(.f16, o, &terms))); // rn reads 0: the output misses by all of it
        }
    }
    const sp = fix.specOf(.final, "witness");
    try std.testing.expect(1 / c.boundExact(sp, 1) >= 100); // an output of 0 against any finite want
}

test "the final's stress fixture is exact in the model: the host final equals its simulation in every fp32 order" {
    const a = std.testing.allocator;
    var fx: fix.Fixture = .{ .kind = .final };
    defer fx.free(a);
    try fx.build(a, .{ .rows = 2, .n = 5120, .k = 5120 }, .stress, 0x55);
    const out = try a.alloc(f64, fx.count());
    defer a.free(out);
    const want = try a.alloc(f64, fx.count());
    defer a.free(want);
    fx.reference(want);
    for (sum_orders) |o| {
        try fx.simulate(a, .f32, o, out);
        for (out, want) |p, q| try std.testing.expect(@abs(p - q) <= c.boundExact(fix.specOf(.final, "stress"), q));
    }
}

test "the registered cases' device-side code and the set's registration compile" {
    // never taken (no device in a host test), but analyzed: a reference to a function does not analyze its body
    var device: ?check_.Gpu = null;
    _ = &device;
    if (device) |dev| {
        const a = std.testing.allocator;
        var s = try cuda.Stream.init(dev.d, false);
        defer s.deinit();
        var list: std.ArrayList(g.Kernel) = .empty;
        defer list.deinit(a);
        const set = try Set.init(dev, s, a, &list);
        defer set.deinit();
        var t: g.Tally = .{};
        try g.runAll(a, list.items, 1, &t);
    }
    try std.testing.expectEqual(@as(usize, 16), regs.len);
}
