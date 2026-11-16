//! NVFP4_NZ5's block 32 experts against E2M1's: each +5 written as a 4 under a scale 1.25 times as large, equal bytes.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const Y = cuda.experts32.Nvfp4b32Experts;
const Matrix = cuda.experts32.Matrix;
const Layer = cuda.experts.Layer;
const d = 4096;
const widths = [_]usize{ 1024, 704, 640 }; // TP=2's rank and TP=3's two
const experts = 6;
const slots = 3; // two routed picks and the skipped shared slot
const rows_set = [_]usize{ 1, 7, 17, 22, 64, 65, 130 };
const most = rows_set[rows_set.len - 1];

/// A launch's form: the decode or the prompt kernels, and the down's sums.
const Form = struct { prompt: bool, down: cuda.experts.Epi };
const forms = [_]Form{ .{ .prompt = false, .down = .down_f32 }, .{ .prompt = false, .down = .down_bf16 }, .{ .prompt = true, .down = .down_bf16 } };

fn put(codes: []u8, i: usize, v: u8) void {
    const b = &codes[i / 2];
    b.* = if (i % 2 == 0) (b.* & 0xf0) | v else (b.* & 0x0f) | v << 4;
}

/// A matrix as NVFP4_NZ5 holds it (`nz`) and its values in E2M1 (`eq`): a planted group's +5 is a 4 at scale x1.25.
const Pair = struct {
    nz: Matrix,
    eq: Matrix,

    fn init(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) !Pair {
        const codes = try a.alloc(u8, n * k / 2);
        rng.bytes(codes);
        for (codes) |*b| { // no 0b1000 outside a planted group: E2M1 reads it as -0 and NVFP4_NZ5 as +5
            if (b.* & 0x0f == 8) b.* &= 0xf0;
            if (b.* >> 4 == 8) b.* &= 0x0f;
        }
        const bases = try a.alloc(u8, n);
        for (bases) |*b| b.* = 0x20 + rng.uintLessThan(u8, 0x18);
        const offsets = try a.alloc(u8, n * (k / 64));
        rng.bytes(offsets);
        const eq_codes = try a.dupe(u8, codes);
        const eq_offsets = try a.dupe(u8, offsets);
        var planted: usize = 0;
        for (0..n) |r| for (0..k / 32) |g| {
            const byte = r * (k / 64) + g / 2;
            const shift: u3 = if (g & 1 == 0) 0 else 4;
            const off = (offsets[byte] >> shift) & 15;
            if ((bases[r] + off) & 7 != 0 or off > 13) continue; // a power of two, with room for two steps (x1.25)
            const at = r * k / 2 + g * 16;
            @memset(codes[at..][0..16], 0);
            @memset(eq_codes[at..][0..16], 0);
            for (0..1 + rng.uintLessThan(usize, 3)) |_| {
                const i = rng.uintLessThan(usize, 32);
                put(codes[at..], i, 8);
                put(eq_codes[at..], i, 6); // e2m1's 4
            }
            eq_offsets[byte] += @as(u8, 2) << shift;
            planted += 1;
        };
        if (planted == 0) return error.NothingPlanted;
        return .{ .nz = .{ .codes = codes, .bases = bases, .offsets = offsets }, .eq = .{ .codes = eq_codes, .bases = bases, .offsets = eq_offsets } };
    }
};

/// One width's experts on the device, both ways.
const Layers = struct {
    eq: Layer,
    nz: Layer,

    fn init(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, width: usize) !Layers {
        const up = Y.bytes(1, width, d, 2) / 4;
        const down = Y.bytes(1, d, width, 1) / 4;
        const w = [_][]u32{ try a.alloc(u32, experts * up), try a.alloc(u32, experts * down), try a.alloc(u32, experts * up), try a.alloc(u32, experts * down) };
        for (0..experts) |e| {
            const g: Pair = try .init(a, rng, width, d);
            const u: Pair = try .init(a, rng, width, d);
            const dn: Pair = try .init(a, rng, d, width);
            Y.packGateUp(w[0][e * up ..][0..up], g.eq, u.eq, width, d);
            Y.packOne(w[1][e * down ..][0..down], dn.eq, d, width);
            Y.packGateUp(w[2][e * up ..][0..up], g.nz, u.nz, width, d);
            Y.packOne(w[3][e * down ..][0..down], dn.nz, d, width);
        }
        const dg = try r.dev(std.mem.sliceAsBytes(r.f32s((experts + 1) * 2, 0.01, 0.03)));
        const common: Layer = .{ .format = .nvfp4_csf0, .up = 0, .down = 0, .up_scale = dg, .down_scale = dg, .width = width, .dims = d, .experts = experts, .limit = 10.0 };
        var l: Layers = .{ .eq = common, .nz = common };
        l.eq.up = try r.dev(std.mem.sliceAsBytes(w[0]));
        l.eq.down = try r.dev(std.mem.sliceAsBytes(w[1]));
        l.nz.up = try r.dev(std.mem.sliceAsBytes(w[2]));
        l.nz.down = try r.dev(std.mem.sliceAsBytes(w[3]));
        return l;
    }
};

/// `x` and `xd`, the gate-up's and the down's inputs: bf16 integers (ints), so a planted group's sum is exact.
const Bufs = struct { act: [2]u64, out: [2]u64, plan: cuda.grouped.Plan, x: u64, xd: u64, picks: u64 };

/// bf16 integers in [-8, 8]: each group's sum is exact before its scale, so a +5 under s equals a 4 under 1.25 s.
fn ints(a: std.mem.Allocator, rng: std.Random, n: usize) ![]u16 {
    const out = try a.alloc(u16, n);
    for (out) |*v| {
        const f: f32 = @floatFromInt(rng.intRangeAtMost(i32, -8, 8));
        v.* = @intCast(@as(u32, @bitCast(f)) >> 16); // an integer's bf16 is its f32's top half
    }
    return out;
}

fn clear(gpu: check.Gpu, r: *rig.Rig, p: u64, bytes: usize) !void {
    try gpu.d.check(gpu.d.api.cuMemsetD8Async(p, 0xff, bytes, r.s.handle), "cuMemsetD8Async");
}

/// One form's gate-up and down, `ka` on `la` into the first buffers and `kb` on `lb` into the second: their bytes.
fn launch(gpu: check.Gpu, r: *rig.Rig, router: cuda.grouped.Router, b: Bufs, ka: *const Y, la: Layer, kb: *const Y, lb: Layer, rows: usize, f: Form) ![4][]u8 {
    const pairs = rows * slots;
    const width = la.width;
    const t: usize = if (f.prompt) 64 else 16;
    const items = cuda.grouped.maxItems(pairs, experts + 1, t);
    try router.route(r.s, b.picks, pairs, experts + 1, t, b.plan);
    for (b.act) |p| try clear(gpu, r, p, pairs * width * 2);
    for (b.out) |p| try clear(gpu, r, p, pairs * d * 4);
    const gate: cuda.experts.Rows = .{ .x = b.x, .stride = d, .slots = slots };
    const dn: cuda.experts.Rows = .{ .x = b.xd, .stride = width };
    for ([2]*const Y{ ka, kb }, [2]Layer{ la, lb }, b.act) |k, l, act| {
        if (f.prompt) try k.runPrompt(r.s, .up, gate, l, b.plan, act, items, experts) else try k.run(r.s, .up, gate, l, b.plan, act, items, experts);
    }
    for ([2]*const Y{ ka, kb }, [2]Layer{ la, lb }, b.out) |k, l, out| {
        if (f.prompt) try k.runPrompt(r.s, f.down, dn, l, b.plan, out, items, experts) else try k.run(r.s, f.down, dn, l, b.plan, out, items, experts);
    }
    return .{ try r.back(u8, b.act[0], pairs * width * 2), try r.back(u8, b.act[1], pairs * width * 2), try r.back(u8, b.out[0], pairs * d * 4), try r.back(u8, b.out[1], pairs * d * 4) };
}

/// Each routed pair of `bytes` holds a written row (a launch's outputs start as all ones).
fn written(bytes: []const u8, row: usize, rows: usize) !void {
    for (0..rows * slots) |i| {
        if (i % slots == slots - 1) continue; // the shared slot, skipped
        try check.expect(std.mem.indexOfNone(u8, bytes[i * row ..][0..row], &.{0xff}) != null, "nvfp4-nz5: pair {d} of {d} rows unwritten", .{ i, rows });
    }
}

/// A form at a row count: every routed pair written, NVFP4_NZ5's bytes equal to E2M1's on the same values.
fn compare(gpu: check.Gpu, r: *rig.Rig, router: cuda.grouped.Router, b: Bufs, e2m1: *const Y, nz5: *const Y, l: Layers, rows: usize, f: Form) !void {
    const o = try launch(gpu, r, router, b, e2m1, l.eq, nz5, l.nz, rows, f);
    try written(o[1], l.nz.width * 2, rows);
    try written(o[3], if (f.down == .down_bf16) d * 2 else d * 4, rows);
    try check.sameBytes("nvfp4-nz5 gate-up", o[1], o[0]);
    try check.sameBytes("nvfp4-nz5 down", o[3], o[2]);
}

/// The control: E2M1's kernels on NVFP4_NZ5's bytes read each +5 as -0, and both outputs part from the nz5 kernels'.
fn control(gpu: check.Gpu, r: *rig.Rig, router: cuda.grouped.Router, b: Bufs, e2m1: *const Y, nz5: *const Y, l: Layers, f: Form) !void {
    const o = try launch(gpu, r, router, b, e2m1, l.nz, nz5, l.nz, most, f);
    try check.expect(!std.mem.eql(u8, o[0], o[1]), "control: E2M1's gate-up on NVFP4_NZ5's codes parts from the nz5 kernel's", .{});
    try check.expect(!std.mem.eql(u8, o[2], o[3]), "control: E2M1's down on NVFP4_NZ5's codes parts from the nz5 kernel's", .{});
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x25);
    const rng = prng.random();
    var r = try rig.Rig.init(gpu, a, rng);
    defer r.deinit();
    const dx = try r.dev(std.mem.sliceAsBytes(try ints(a, rng, most * d)));
    const picks = try a.alloc(i32, most * slots);
    for (0..most) |row| {
        // expert 0 takes most pairs (several 64-pair items), the others a few each
        picks[row * slots] = if (row % 3 == 0) @intCast(1 + row % (experts - 1)) else 0;
        picks[row * slots + 1] = @intCast(1 + (row * 7) % (experts - 1));
        picks[row * slots + 2] = experts; // the shared expert, skipped
    }
    const pairs_most = most * slots;
    const plan: cuda.grouped.Plan = .{
        .members = try r.out(pairs_most * 4),
        .items = try r.out(cuda.grouped.maxItems(pairs_most, experts + 1, 16) * 12),
        .counts = try r.out(8),
        .rank = try r.out(pairs_most * 4),
        .hist = try r.out(((pairs_most + 1023) / 1024) * (experts + 1) * 4),
    };
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const sms: u32 = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var e2m1 = try Y.load(gpu.d, sms, .e2m1);
    defer e2m1.unload();
    var nz5 = try Y.load(gpu.d, sms, .nz5);
    defer nz5.unload();
    const router = try cuda.grouped.Router.resolve(mod);
    const bufs: Bufs = .{
        .act = .{ try r.out(pairs_most * widths[0] * 2), try r.out(pairs_most * widths[0] * 2) },
        .out = .{ try r.out(pairs_most * d * 4), try r.out(pairs_most * d * 4) },
        .plan = plan,
        .x = dx,
        .xd = try r.dev(std.mem.sliceAsBytes(try ints(a, rng, pairs_most * widths[0]))),
        .picks = try r.dev(std.mem.sliceAsBytes(picks)),
    };
    for (widths) |width| {
        const l: Layers = try .init(&r, a, rng, width);
        for (forms) |f| {
            for (rows_set) |rows| try compare(gpu, &r, router, bufs, &e2m1, &nz5, l, rows, f);
            try control(gpu, &r, router, bufs, &e2m1, &nz5, l, f);
        }
    }
    r.report("nvfp4-nz5: gate-up, fp32 and bf16 down, decode and prompt, at 1,024, 704 and 640: E2M1's bytes, each control red");
}
