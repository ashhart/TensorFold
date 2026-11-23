//! The 1-row algorithm on the device: 1 row gives a batch's bits; a planted split-K candidate is refused.
const std = @import("std");
const cuda = @import("cuda");
const gemm = @import("gemm.zig");
const one_row = cuda.one_row;

const cublaslt = cuda.cublaslt;
const t = std.testing;
const gpa = t.allocator;

/// Qwen3-0.6B's (n, k) pairs: q, k and v, o, gate and up, down, the tied head.
const pairs = [_][2]usize{ .{ 2048, 1024 }, .{ 1024, 1024 }, .{ 1024, 2048 }, .{ 3072, 1024 }, .{ 1024, 3072 }, .{ 151936, 1024 } };

const Device = struct {
    d: cuda.Driver,
    ctx: cuda.Context,
    s: cuda.Stream,

    fn open(dev: *Device) !void {
        dev.d = cuda.Driver.open() catch return error.SkipZigTest;
        errdefer dev.d.close();
        dev.ctx = cuda.Context.init(&dev.d, 0) catch return error.SkipZigTest;
        errdefer dev.ctx.deinit();
        dev.s = try cuda.Stream.init(&dev.d, false);
    }

    fn close(dev: *Device) void {
        dev.s.deinit();
        dev.ctx.deinit();
        dev.d.close();
    }
};

/// Rows [0, rows) of `x` through `g` at 1 row each into `y1`, and all at once into `yb`; whether every bit agrees.
fn rowsAgree(g: *gemm.Gemms, dev: *Device, pr: *one_row.Probe, n: usize, k: usize) !bool {
    const rows = one_row.rows;
    try g.run(dev.s, pr.x.ptr, pr.w.ptr, pr.ref.ptr, rows, n, k);
    for (0..rows) |r| try g.run(dev.s, pr.x.ptr + r * k * 2, pr.w.ptr, pr.y.ptr + r * n * 2, 1, n, k);
    try dev.s.synchronize();
    try pr.ref.download(0, pr.want);
    try pr.y.download(0, pr.got);
    return std.mem.eql(u8, pr.want, pr.got);
}

/// New input rows for a probe: the same classes on another seed than the admission's.
fn reseed(pr: *one_row.Probe, k: usize, seed: u64) !void {
    const x = try gpa.alloc(u16, one_row.rows * k);
    defer gpa.free(x);
    one_row.inputs(x, k, seed);
    try pr.x.upload(0, std.mem.sliceAsBytes(x));
}

test "qwen3 gemm: a 1-row call gives each row's bits in a batch of 64, for every Qwen3-0.6B (n, k), fresh inputs" {
    var dev: Device = undefined;
    try dev.open();
    defer dev.close();
    var g = try gemm.Gemms.init(gpa, &dev.d);
    defer g.deinit();
    for (pairs) |nk| {
        const n, const k = nk;
        const p = try g.plan(n, k);
        std.debug.print("(n {d}, k {d}): 1-row algorithm {s}\n", .{ n, k, if (p.one != null) "taken" else "none" });
        var pr = try one_row.Probe.init(gpa, &dev.d, n, k);
        defer pr.deinit(gpa);
        try reseed(&pr, k, 0x51ab + n + k);
        try t.expect(try rowsAgree(&g, &dev, &pr, n, k));
    }
}

/// The 1-row heuristic's first candidate cuBLASLt runs with K split `ways` (in-place reduction); null if none does.
fn plantSplitK(lt: *const cublaslt.Library, one: *const cublaslt.Linear, ways: c_int) !?cublaslt.Heuristic {
    var found: [one_row.candidates]cublaslt.Heuristic = undefined;
    var count: c_int = 0;
    try lt.check(lt.api.cublasLtMatmulAlgoGetHeuristic(one.handle, one.desc, one.a, one.b, one.c, one.c, one.pref, one_row.candidates, &found, &count), "heuristic");
    const inplace: u32 = 1; // CUBLASLT_REDUCTION_SCHEME_INPLACE
    for (found[0..@intCast(@max(count, 0))]) |h| {
        var algo = h.algo;
        if (lt.api.cublasLtMatmulAlgoConfigSetAttribute(&algo, cublaslt.algo_config_splitk, &ways, @sizeOf(c_int)) != 0) continue;
        if (lt.api.cublasLtMatmulAlgoConfigSetAttribute(&algo, cublaslt.algo_config_reduction, &inplace, @sizeOf(u32)) != 0) continue;
        var checked: cublaslt.Heuristic = undefined;
        if (lt.api.cublasLtMatmulAlgoCheck(one.handle, one.desc, one.a, one.b, one.c, one.c, &algo, &checked) != 0) continue;
        if (checked.state != 0 or checked.workspace_size > gemm.workspace_bytes) continue;
        checked.algo = algo;
        return checked;
    }
    return null;
}

test "qwen3 gemm control: a planted split-K 1-row candidate gives other bits, is refused, and 1 row runs the plan's" {
    var dev: Device = undefined;
    try dev.open();
    defer dev.close();
    var g = try gemm.Gemms.init(gpa, &dev.d);
    defer g.deinit();
    const n = 1024;
    const k = 3072;
    const p = try g.plan(n, k);
    var one = try cublaslt.Linear.initEpilogue(&g.lt, 1, n, k, .bf16, gemm.workspace_bytes, .none);
    defer one.deinit();
    const planted = try plantSplitK(&g.lt, &one, 4) orelse return error.NoSplitKCandidate;
    const c = gemm.config(&g.lt, planted.algo).?;
    try t.expectEqual(@as(c_int, 4), c.splitk);
    try t.expect(!gemm.oneOrder(c));
    // red: past the K-order check, the probe alone sees the split sums' other bits
    try t.expect(!try g.probeSame(p, n, k, planted));
    // green beside it: the candidate the plan took passes the same probe
    if (p.one) |o| {
        var taken = std.mem.zeroes(cublaslt.Heuristic);
        taken.algo = o.algo;
        taken.workspace_size = o.workspace_need;
        try t.expect(try g.probeSame(p, n, k, taken));
    }
    // offered alone, the planted candidate is refused: no 1-row algorithm
    try t.expect(!try g.oneRowFrom(p, &one, n, k, &.{planted}));
    // and with none, 1-row calls run the plan's algorithm, its bits a batch's
    if (p.one) |*o| o.deinit();
    p.one = null;
    var pr = try one_row.Probe.init(gpa, &dev.d, n, k);
    defer pr.deinit(gpa);
    try t.expect(try rowsAgree(&g, &dev, &pr, n, k));
}

test "qwen3 gemm: a plan whose every allocation fails in turn leaks nothing, its checked rows included" {
    var dev: Device = undefined;
    try dev.open();
    defer dev.close();
    var failed: usize = 0;
    while (true) : (failed += 1) {
        var failing = t.FailingAllocator.init(gpa, .{ .fail_index = failed });
        var g = try gemm.Gemms.init(failing.allocator(), &dev.d);
        defer g.deinit();
        _ = g.plan(1024, 1024) catch |e| {
            try t.expectEqual(error.OutOfMemory, e);
            continue;
        };
        break;
    }
    // past the checked-row put: getOrPut, that put, then the probe's own allocations fail
    try t.expect(failed > 2);
}
