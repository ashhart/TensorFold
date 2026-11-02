//! Block 32 experts with csf0 scales against nvfp4_experts.cu on equal scales: equal bytes in every form and width.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");
const X = cuda.experts.Nvfp4Experts;
const Y = cuda.experts32.Nvfp4b32Experts;
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

/// A matrix in both forms: as a checkpoint holds it, and its scales [n, k/16] two by two.
const Both = struct {
    m: cuda.experts32.Matrix,
    scales: []u8,

    fn init(a: std.mem.Allocator, rng: std.Random, n: usize, k: usize) !Both {
        const codes = try a.alloc(u8, n * k / 2);
        rng.bytes(codes);
        const bases = try a.alloc(u8, n);
        for (bases) |*b| b.* = 0x20 + rng.uintLessThan(u8, 0x18);
        const offsets = try a.alloc(u8, n * (k / 64));
        rng.bytes(offsets);
        const scales = try a.alloc(u8, n * k / 16);
        for (0..n) |r| for (0..k / 32) |g| {
            const byte = offsets[r * (k / 64) + g / 2];
            const v = bases[r] + if (g & 1 == 0) byte & 15 else byte >> 4;
            @memset(scales[r * (k / 16) + 2 * g ..][0..2], v);
        };
        return .{ .m = .{ .codes = codes, .bases = bases, .offsets = offsets }, .scales = scales };
    }
};

/// One width's experts on the device: block 16's, block 32's, and block 16's with one of a scale's two bytes moved.
const Layers = struct {
    x: Layer,
    y: Layer,
    moved: Layer,

    fn init(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, width: usize) !Layers {
        const up16 = X.bytes(1, width, d, 2) / 4;
        const down16 = X.bytes(1, d, width, 1) / 4;
        const up32 = Y.bytes(1, width, d, 2) / 4;
        const down32 = Y.bytes(1, d, width, 1) / 4;
        const w = [_][]u32{ try a.alloc(u32, experts * up16), try a.alloc(u32, experts * down16), try a.alloc(u32, experts * up32), try a.alloc(u32, experts * down32) };
        const moved = [_][]u32{ try a.alloc(u32, experts * up16), try a.alloc(u32, experts * down16) };
        for (0..experts) |e| {
            const g: Both = try .init(a, rng, width, d);
            const u: Both = try .init(a, rng, width, d);
            const dn: Both = try .init(a, rng, d, width);
            X.packGateUp(w[0][e * up16 ..][0..up16], .{ g.m.codes, g.scales }, .{ u.m.codes, u.scales }, width, d);
            X.packOne(w[1][e * down16 ..][0..down16], dn.m.codes, dn.scales, d, width);
            Y.packGateUp(w[2][e * up32 ..][0..up32], g.m, u.m, width, d);
            Y.packOne(w[3][e * down32 ..][0..down32], dn.m, d, width);
            g.scales[0] +%= 1; // row 0's first 16 inputs: block 32 has no such scale, so the kernels must part
            dn.scales[0] +%= 1;
            X.packGateUp(moved[0][e * up16 ..][0..up16], .{ g.m.codes, g.scales }, .{ u.m.codes, u.scales }, width, d);
            X.packOne(moved[1][e * down16 ..][0..down16], dn.m.codes, dn.scales, d, width);
        }
        const dg = try r.dev(std.mem.sliceAsBytes(r.f32s((experts + 1) * 2, 0.01, 0.03)));
        const common: Layer = .{ .format = .nvfp4, .up = 0, .down = 0, .up_scale = dg, .down_scale = dg, .width = width, .dims = d, .experts = experts, .limit = 10.0 };
        var l: Layers = .{ .x = common, .y = common, .moved = common };
        l.x.up = try r.dev(std.mem.sliceAsBytes(w[0]));
        l.x.down = try r.dev(std.mem.sliceAsBytes(w[1]));
        l.y.format = .nvfp4_csf0;
        l.y.up = try r.dev(std.mem.sliceAsBytes(w[2]));
        l.y.down = try r.dev(std.mem.sliceAsBytes(w[3]));
        l.moved.up = try r.dev(std.mem.sliceAsBytes(moved[0]));
        l.moved.down = try r.dev(std.mem.sliceAsBytes(moved[1]));
        return l;
    }
};

const Kernels = struct { x: *const X, y: *const Y, router: cuda.grouped.Router };

const Bufs = struct { act: [2]u64, out: [2]u64, plan: cuda.grouped.Plan, x: u64, picks: u64 };

fn clear(gpu: check.Gpu, r: *rig.Rig, p: u64, bytes: usize) !void {
    try gpu.d.check(gpu.d.api.cuMemsetD8Async(p, 0xff, bytes, r.s.handle), "cuMemsetD8Async");
}

/// One form's gate-up and down of `x` under block 16's kernels and `y` under block 32's: their act and out bytes.
fn launch(gpu: check.Gpu, r: *rig.Rig, k: Kernels, b: Bufs, x: Layer, y: Layer, rows: usize, f: Form) ![4][]u8 {
    const pairs = rows * slots;
    const width = x.width;
    const t: usize = if (f.prompt) 64 else 16;
    const items = cuda.grouped.maxItems(pairs, experts + 1, t);
    try k.router.route(r.s, b.picks, pairs, experts + 1, t, b.plan);
    for (b.act) |p| try clear(gpu, r, p, pairs * width * 2);
    for (b.out) |p| try clear(gpu, r, p, pairs * d * 4);
    const gate: cuda.experts.Rows = .{ .x = b.x, .stride = d, .slots = slots };
    const dn: cuda.experts.Rows = .{ .x = b.act[0], .stride = width };
    if (f.prompt) {
        try k.x.runPrompt(r.s, .up, gate, x, b.plan, b.act[0], items, experts);
        try k.y.runPrompt(r.s, .up, gate, y, b.plan, b.act[1], items, experts);
        try k.x.runPrompt(r.s, f.down, dn, x, b.plan, b.out[0], items, experts);
        try k.y.runPrompt(r.s, f.down, dn, y, b.plan, b.out[1], items, experts);
    } else {
        try k.x.run(r.s, .up, gate, x, b.plan, b.act[0], items, experts);
        try k.y.run(r.s, .up, gate, y, b.plan, b.act[1], items, experts);
        try k.x.run(r.s, f.down, dn, x, b.plan, b.out[0], items, experts);
        try k.y.run(r.s, f.down, dn, y, b.plan, b.out[1], items, experts);
    }
    return .{ try r.back(u8, b.act[0], pairs * width * 2), try r.back(u8, b.act[1], pairs * width * 2), try r.back(u8, b.out[0], pairs * d * 4), try r.back(u8, b.out[1], pairs * d * 4) };
}

/// Each routed pair of `bytes` holds a written row (a launch's outputs start as all ones).
fn written(bytes: []const u8, row: usize, rows: usize) !void {
    for (0..rows * slots) |i| {
        if (i % slots == slots - 1) continue; // the shared slot, skipped
        try check.expect(std.mem.indexOfNone(u8, bytes[i * row ..][0..row], &.{0xff}) != null, "nvfp4-b32: pair {d} of {d} rows unwritten", .{ i, rows });
    }
}

/// A form at a row count: every routed pair written, the two kernels' bytes equal.
fn compare(gpu: check.Gpu, r: *rig.Rig, k: Kernels, b: Bufs, l: Layers, rows: usize, f: Form) !void {
    const o = try launch(gpu, r, k, b, l.x, l.y, rows, f);
    try written(o[1], l.x.width * 2, rows);
    try written(o[3], if (f.down == .down_bf16) d * 2 else d * 4, rows);
    try check.sameBytes("nvfp4-b32 gate-up", o[1], o[0]);
    try check.sameBytes("nvfp4-b32 down", o[3], o[2]);
}

/// The control: a moved scale byte parts both of a form's outputs from block 32's, so `compare` can see a difference.
fn control(gpu: check.Gpu, r: *rig.Rig, k: Kernels, b: Bufs, l: Layers, f: Form) !void {
    const o = try launch(gpu, r, k, b, l.moved, l.y, most, f);
    try check.expect(!std.mem.eql(u8, o[0], o[1]), "control: a moved gate scale parts gate-up's bytes", .{});
    try check.expect(!std.mem.eql(u8, o[2], o[3]), "control: a moved down scale parts down's bytes", .{});
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xb32);
    const rng = prng.random();
    var r = try rig.Rig.init(gpu, a, rng);
    defer r.deinit();
    const dx = try r.dev(std.mem.sliceAsBytes(r.bfs(most * d, 0.5)));
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
    var x = try X.load(gpu.d, sms);
    defer x.unload();
    var y = try Y.load(gpu.d, sms);
    defer y.unload();
    const k: Kernels = .{ .x = &x, .y = &y, .router = try cuda.grouped.Router.resolve(mod) };
    const bufs: Bufs = .{
        .act = .{ try r.out(pairs_most * widths[0] * 2), try r.out(pairs_most * widths[0] * 2) },
        .out = .{ try r.out(pairs_most * d * 4), try r.out(pairs_most * d * 4) },
        .plan = plan,
        .x = dx,
        .picks = try r.dev(std.mem.sliceAsBytes(picks)),
    };
    for (widths) |width| {
        const l: Layers = try .init(&r, a, rng, width);
        for (forms) |f| {
            for (rows_set) |rows| try compare(gpu, &r, k, bufs, l, rows, f);
            try control(gpu, &r, k, bufs, l, f);
        }
    }
    r.report("nvfp4-b32: gate-up, fp32 and bf16 down, decode and prompt, at 1,024, 704 and 640: block 16's bytes, each control red");
}
