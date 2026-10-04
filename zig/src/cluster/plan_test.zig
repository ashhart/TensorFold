//! Kimi K3 on four 512 GiB M3 Ultras: the default layout, layouts from a config, determinism, and refusals.
const std = @import("std");
const node = @import("node.zig");
const model = @import("model.zig");
const estimate = @import("estimate.zig");
const checkpoint = @import("checkpoint.zig");
const plan = @import("plan.zig");
const budget = @import("budget.zig");

const gib = node.gib;

pub fn studio(i: usize, memory: u64, wired_mb: u64) node.Inventory {
    var buf: [16]u8 = undefined;
    return .{
        .id = 1000 + i,
        .name = .of(std.fmt.bufPrint(&buf, "studio{d}", .{i + 1}) catch unreachable),
        .chip = .of("Apple M3 Ultra"),
        .gpu_cores = 80,
        .memory = memory,
        .wired_limit_mb = wired_mb,
        .gpu_limit = node.gpuLimitFor(memory, wired_mb),
        .bandwidth = node.bandwidthOf("Apple M3 Ultra"),
        .free = memory / 2,
        .disk_free = 2 << 40,
    };
}

/// K3's tensor table from its literal shape and real shard sizes (TF_K3_DIR's files.tsv when set).
pub fn k3(a: std.mem.Allocator) !checkpoint.Checkpoint {
    const s = model.k3();
    var files: []checkpoint.File = try a.alloc(checkpoint.File, s.layers + 3);
    for (files, 0..) |*f, i| {
        const kda = i < s.layers and s.kind(@intCast(i)) == .linear;
        const size: u64 = if (i == 0) 2_341_216_112 else if (i < s.layers) (if (kda) 16_990_911_504 else 16_567_501_776) else if (i == s.layers) 4_697_664_072 else if (i == s.layers + 1) 92_289_328 else 802_448_352;
        f.* = .{ .name = try std.fmt.allocPrint(a, "model-{d:0>5}-of-{d:0>6}.safetensors", .{ i + 1, s.layers + 3 }), .size = size };
    }
    if (std.testing.environ.getPosix("TF_K3_DIR")) |dir| {
        const text = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, try std.fs.path.join(a, &.{ dir, "files.tsv" }), a, .limited(1 << 20));
        files = try estimate.manifest(a, text);
    }
    return estimate.kimi(a, &s, files);
}

fn four(memory: u64, wired_mb: [4]u64) [4]node.Inventory {
    var out: [4]node.Inventory = undefined;
    for (&out, 0..) |*n, i| n.* = studio(i, memory, wired_mb[i]);
    return out;
}

test "K3's default plan on four Studios: tensor 4 x expert 4, a quarter of every expert byte each, all fit by default" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try k3(a);
    const s = model.k3();
    const nodes = four(512 * gib, @splat(0));
    const p = try plan.plan(a, c, &s, &nodes, .{ .drafter_bytes = 4 * gib });
    try std.testing.expectEqual(plan.Layout{ .tensor = 4, .pipeline = 1, .expert = 4 }, p.layout);
    const ns = try budget.needs(a, &p, c, &s);
    var experts: u64 = 0;
    for (c.tensors) |t| experts += if (t.class.role == .expert) t.bytes else 0;
    for (ns) |n| try std.testing.expectEqual(experts / 4, n.of(.experts));
    for (ns) |n| try std.testing.expect(budget.gb(n.weightBytes()) > 365 and budget.gb(n.weightBytes()) < 368);
    try std.testing.expectEqual(ns[1].of(.attention), ns[3].of(.attention));
    try std.testing.expect(ns[0].drafter > 4 * gib);
    const fs = try budget.fits(a, &p, ns);
    try std.testing.expect(budget.allFit(fs));
    const same = try plan.plan(a, c, &s, &nodes, .{ .layout = .{ .tensor = 4, .expert = 4 }, .drafter_bytes = 4 * gib });
    try std.testing.expectEqual(p.digest, same.digest);
    const shuffled = try a.dupe(checkpoint.Tensor, c.tensors);
    var prng: std.Random.DefaultPrng = .init(3);
    prng.random().shuffle(checkpoint.Tensor, shuffled);
    const again = try plan.plan(a, .{ .files = c.files, .tensors = shuffled }, &s, &nodes, .{ .drafter_bytes = 4 * gib });
    try std.testing.expectEqual(p.digest, again.digest);
    const other = try plan.plan(a, c, &s, &nodes, .{ .slices = 16 });
    try std.testing.expect(other.digest != p.digest);
}

test "K3 as four pipeline stages, and with every expert split four ways" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try k3(a);
    const s = model.k3();
    const nodes = four(512 * gib, @splat(0));
    const pp = try plan.plan(a, c, &s, &nodes, .{ .layout = .{ .pipeline = 4 } });
    try std.testing.expectEqual(@as(u32, 4), pp.layout.pipeline);
    var covered: u32 = 0;
    for (pp.stages) |st| {
        try std.testing.expectEqual(covered, st.layers.begin);
        covered = st.layers.end;
    }
    try std.testing.expectEqual(@as(u32, 93), covered);
    const ns = try budget.needs(a, &pp, c, &s);
    for (ns) |n| try std.testing.expect(budget.gb(n.weightBytes()) > 330 and budget.gb(n.weightBytes()) < 400);
    try std.testing.expect(ns[0].of(.embed) > 0 and ns[3].of(.embed) == 0 and ns[3].of(.head) > 0);
    const tp = try plan.plan(a, c, &s, &nodes, .{ .layout = .{ .tensor = 4, .expert = 1 } });
    const nt = try budget.needs(a, &tp, c, &s);
    try std.testing.expect(nt[0].of(.experts) == nt[3].of(.experts));
}

test "a node whose wired limit is too low is refused with the exact raise; 128 GiB nodes cannot fit at all" {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try k3(a);
    const s = model.k3();
    const low = four(512 * gib, .{ 0, 0, 0, 360_000 });
    const p = try plan.plan(a, c, &s, &low, .{});
    const fs = try budget.fits(a, &p, try budget.needs(a, &p, c, &s));
    try std.testing.expect(fs[0].ok and !fs[3].ok and fs[3].possible);
    try std.testing.expect(fs[3].raise_mb > 360_000 and fs[3].raise_mb * (1 << 20) >= fs[3].need);
    var out: std.Io.Writer.Allocating = .init(a);
    try budget.refusal(&out.writer, &p, fs);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "sudo sysctl iogpu.wired_limit_mb=") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "node-d") != null);
    const small = four(128 * gib, @splat(0));
    const q = try plan.plan(a, c, &s, &small, .{});
    const gs = try budget.fits(a, &q, try budget.needs(a, &q, c, &s));
    for (gs) |g| try std.testing.expect(!g.ok and !g.possible);
    out.clearRetainingCapacity();
    try budget.refusal(&out.writer, &q, gs);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "add nodes") != null);
    try std.testing.expectError(error.BadLayout, plan.plan(a, c, &s, &low, .{ .layout = .{ .tensor = 3 } }));
    try std.testing.expectError(error.BadSlices, plan.plan(a, c, &s, &low, .{ .slices = 7 }));
}
