//! The routed chain in a captured bucket: rows past the params' word 2 are padding, never routed, planned or combined.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const moe = @import("moe_tests.zig");
const Gpu = check.Gpu;

const routed = 64;
const count = routed + 1;
const top_k = 8;
const slots = top_k + 1;
const dims = 256;
const ni = 128;
const most = 32;
const unset: u8 = 0xab;

/// The chain's kernels, weights and scratch for buckets of up to `most` rows.
const Chain = struct {
    gpu: Gpu,
    s: cuda.Stream,
    f: cuda.moe.Fns,
    router: cuda.grouped.Router,
    grouped: cuda.experts.Grouped,
    layer: cuda.experts.Layer,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    w: u64 = 0,
    logits: u64 = 0,
    picks: u64 = 0,
    wts: u64 = 0,
    plan: cuda.grouped.Plan = undefined,
    act: u64 = 0,
    y: u64 = 0,

    fn buf(c: *Chain, len: usize) !u64 {
        var b = try cuda.DeviceBuffer.alloc(c.gpu.d, len);
        try b.fill8(0, null);
        try c.bufs.append(c.gpu.gpa, b);
        return b.ptr;
    }

    fn put(c: *Chain, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(c.gpu.d, bytes);
        try c.bufs.append(c.gpu.gpa, b);
        return b.ptr;
    }

    fn get(c: *Chain, ptr: u64, out: []u8) !void {
        try c.s.synchronize();
        try c.gpu.d.check(c.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, out.len), "cuMemcpyDtoH");
    }

    /// The engine's decode chain over a bucket of `rows` rows at x into out (cuda_routed.rows, wide off).
    fn run(c: *Chain, x: u64, out: u64, rows: usize, params: u64) !void {
        const pairs = rows * slots;
        const items = cuda.grouped.maxItems(pairs, count, 16);
        try c.f.route(c.s, x, dims, rows, c.w, c.logits, dims, count, params);
        try c.f.pick(c.s, c.logits, c.picks, c.wts, rows, routed, top_k, params);
        try c.router.routeRows(c.s, c.picks, pairs, count, 16, c.plan, if (params == 0) 0 else params + 8, slots);
        try c.grouped.decode(c.s, .up, .{ .x = x, .stride = dims, .slots = slots }, c.layer, c.plan, c.act, items, -1);
        try c.grouped.decode(c.s, .down_f32, .{ .x = c.act, .stride = ni }, c.layer, c.plan, c.y, items, -1);
        try c.f.combine(c.s, c.y, true, c.wts, out, rows, slots, dims, params);
    }

    /// The pairs the last plan grouped: the items' counts summed, and the largest member.
    fn planned(c: *Chain, a: std.mem.Allocator) !struct { pairs: usize, top: i32 } {
        var counts: [2]i32 = undefined;
        try c.get(c.plan.counts, std.mem.sliceAsBytes(&counts));
        const items = try a.alloc(i32, 3 * @as(usize, @intCast(counts[0])));
        defer a.free(items);
        try c.get(c.plan.items, std.mem.sliceAsBytes(items));
        var pairs: usize = 0;
        for (0..items.len / 3) |i| pairs += @intCast(items[3 * i + 2]);
        const members = try a.alloc(i32, pairs);
        defer a.free(members);
        try c.get(c.plan.members, std.mem.sliceAsBytes(members));
        return .{ .pairs = pairs, .top = if (pairs > 0) std.mem.max(i32, members) else -1 };
    }
};

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x11fe);
    const r = prng.random();
    var mods: [4]cuda.Module = undefined;
    for (&mods, [_][]const u8{ cuda.kernels.moe, cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack }) |*m, img| m.* = try cuda.Module.load(gpu.d, img);
    defer for (&mods) |*m| m.unload();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const ex = try cuda.experts.Affine4Experts.resolve(mods[1], mods[2], mods[3], sms);
    var c: Chain = .{
        .gpu = gpu,
        .s = try cuda.Stream.init(gpu.d, true),
        .f = try cuda.moe.Fns.resolve(mods[0]),
        .router = try cuda.grouped.Router.resolve(mods[1]),
        .grouped = .{ .affine4 = &ex },
        .layer = undefined,
    };
    defer c.s.deinit();
    defer {
        for (c.bufs.items) |*b| b.free();
        c.bufs.deinit(a);
    }
    // a bf16 router (routed experts, then the shared gate), affine-4 SwiGLU experts with the shared one last
    const w = try a.alloc(u16, count * dims);
    defer a.free(w);
    for (w) |*v| v.* = moe.randBf16(r, 0.1);
    c.w = try c.put(std.mem.sliceAsBytes(w));
    const up = try c.buf(cuda.experts.Affine4Experts.pairBytes(count, ni, dims));
    const down = try c.buf(cuda.experts.Affine4Experts.bytes(count, dims, ni));
    for ([_]struct { u64, usize, usize, usize, usize }{ .{ up, ni, dims, 2, 0 }, .{ up, ni, dims, 2, 1 }, .{ down, dims, ni, 1, 0 } }) |t| {
        const tb = try moe.table(a, r, count, t[1], t[2]);
        defer {
            a.free(tb.words);
            a.free(tb.scales);
            a.free(tb.biases);
        }
        try ex.packInto(c.s, try c.put(std.mem.sliceAsBytes(tb.words)), try c.put(std.mem.sliceAsBytes(tb.scales)), try c.put(std.mem.sliceAsBytes(tb.biases)), t[0], count, t[1], t[2], t[3], t[4]);
        try c.s.synchronize();
    }
    c.layer = .{ .format = .affine4, .up = up, .down = down, .width = ni, .dims = dims, .experts = count, .swiglu = true };
    const pairs = most * slots;
    c.logits = try c.buf(most * count * 4);
    c.picks = try c.buf(pairs * 4);
    c.wts = try c.buf(pairs * 4);
    c.plan = .{ .members = try c.buf(pairs * 4), .items = try c.buf(3 * cuda.grouped.maxItems(pairs, count, 16) * 4), .counts = try c.buf(8), .rank = 0, .hist = 0 };
    c.act = try c.buf(pairs * ni * 2);
    c.y = try c.buf(pairs * dims * 4);
    // rows of normed activations; every row is non-zero, so a padding row routed would plan its own experts
    const x = try a.alloc(u16, most * dims);
    defer a.free(x);
    for (x) |*v| v.* = moe.randBf16(r, 1.0);
    const dx = try c.put(std.mem.sliceAsBytes(x));
    // each row alone, uncaptured and without params: the bits every live row of a bucket must equal
    const want = try a.alloc(u8, most * dims * 2);
    defer a.free(want);
    const one = try c.buf(dims * 2);
    for (0..most) |row| {
        try c.run(dx + row * dims * 2, one, 1, 0);
        try c.get(one, want[row * dims * 2 ..][0 .. dims * 2]);
    }
    const out = try c.buf(most * dims * 2);
    const got = try a.alloc(u8, most * dims * 2);
    defer a.free(got);
    var params = [8]i32{ 0, 0, 0, 0, 0, 0, 0, 0 };
    const dparams = try c.put(std.mem.sliceAsBytes(&params));
    // each bucket captured once, then replayed at live counts in this order (the count is read at each replay)
    const sweeps = [_]struct { bucket: usize, live: []const usize }{
        .{ .bucket = 1, .live = &.{1} },
        .{ .bucket = 16, .live = &.{ 1, 2, 3, 7, 15, 16, 3, 14, 2 } },
        .{ .bucket = most, .live = &.{ 17, 18, 31, 32, 3, 30, 17 } },
    };
    for (sweeps) |sw| {
        try cuda.graph.beginCapture(c.s, .thread_local);
        c.run(dx, out, sw.bucket, dparams) catch |err| {
            var broken = cuda.graph.endCapture(c.s) catch return err;
            broken.deinit();
            return err;
        };
        var graph = try cuda.graph.endCapture(c.s);
        defer graph.deinit();
        var exec = try graph.instantiate();
        defer exec.deinit();
        for (sw.live) |live| {
            params[2] = @intCast(live);
            try c.s.synchronize();
            try c.gpu.d.check(c.gpu.d.api.cuMemcpyHtoD_v2(dparams, &params, @sizeOf(@TypeOf(params))), "cuMemcpyHtoD");
            try c.gpu.d.check(c.gpu.d.api.cuMemsetD8_v2(out, unset, most * dims * 2), "cuMemsetD8");
            try exec.launchOn(c.s);
            const p = try c.planned(a);
            try check.expect(p.pairs == live * slots and p.top < live * slots, "bucket {d} at {d} live rows planned {d} pairs (largest member {d}), want the live rows' {d}", .{ sw.bucket, live, p.pairs, p.top, live * slots });
            try c.get(out, got);
            try check.sameBytes("live rows against one-row runs", got[0 .. live * dims * 2], want[0 .. live * dims * 2]);
            for (got[live * dims * 2 ..]) |byte| try check.expect(byte == unset, "bucket {d} at {d} live rows wrote a padding row", .{ sw.bucket, live });
        }
        check.pass("moe live rows: bucket {d} replayed at {any} live rows plans only their pairs, equals one-row runs, skips padding", .{ sw.bucket, sw.live });
    }
}
