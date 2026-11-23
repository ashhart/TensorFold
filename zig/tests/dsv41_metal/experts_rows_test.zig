//! The decode experts' rows in one launch equal each row launched alone, whatever experts the rows share.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");

const testing = std.testing;
const ex = pkg.pipelines.experts;
const topk = ex.topk;
const experts = 96;
/// Bytes between two experts' codes in one buffer (scales: a sixteenth), so the table's experts overlap but differ.
const stride = 4096;
const shared = mtl.ResourceOptions.shared;

const Routing = enum { same, disjoint, mixed };

/// Expert ids a row: every row the same six, no two rows one, or six of the first nine at random.
fn route(ids: []u32, rows: usize, how: Routing, rng: std.Random) void {
    for (0..rows) |r| {
        var pool: [experts]u32 = undefined;
        for (&pool, 0..) |*p, i| p.* = @intCast(i);
        if (how == .mixed) rng.shuffle(u32, pool[0..9]);
        for (0..topk) |i| ids[r * topk + i] = switch (how) {
            .same => @intCast((i + r) % topk),
            .disjoint => @intCast(r * topk + i),
            .mixed => pool[i],
        };
    }
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    pipes: ex.Pipelines,
    weights: [6]mtl.Buffer,
    table: mtl.Buffer,
    bufs: std.ArrayList(mtl.Buffer) = .empty,

    fn buffer(g: *Gpu, bytes: []const u8) !mtl.Buffer {
        const b = try g.device.buffer(@max(bytes.len, 16), shared);
        @memcpy(b.contents()[0..bytes.len], bytes);
        try g.bufs.append(testing.allocator, b);
        return b;
    }

    fn scratch(g: *Gpu, len: usize) !mtl.Buffer {
        const b = try g.device.buffer(@max(len, 16), shared);
        @memset(b.contents()[0..@max(len, 16)], 0xAB);
        try g.bufs.append(testing.allocator, b);
        return b;
    }

    fn free(g: *Gpu) void {
        for (g.bufs.items) |b| b.deinit();
        g.bufs.clearRetainingCapacity();
    }
};

/// act, out (bf16 by pair position), y (fp32 by row) and where (by pair) of one launch of `ids.len / topk` rows.
fn launch(g: *Gpu, x: []const u16, ids: []const u32, wts: []const f32) ![4][]const u8 {
    const rows: u32 = @intCast(ids.len / topk);
    const call: ex.Call = .{ .rows = rows, .experts = experts };
    const b = ex.bytes(call);
    var sc: [6]mtl.Buffer = undefined;
    for (&sc, b) |*s, len| s.* = try g.scratch(len);
    var ones: [16 * topk]f32 = @splat(1);
    const r: ex.Route = .{ .ids = .{ .buf = try g.buffer(std.mem.sliceAsBytes(ids)) }, .weights = .{ .buf = try g.buffer(std.mem.sliceAsBytes(wts)) }, .slots = .{ .buf = sc[0] }, .nslots = .{ .buf = sc[1] }, .pairs = .{ .buf = sc[2] }, .where = .{ .buf = sc[3] } };
    const xr: ex.Ref = .{ .buf = try g.buffer(std.mem.sliceAsBytes(x)) };
    const xq: ex.Ref = .{ .buf = try g.scratch(2 * rows * ex.hidden) };
    const xqs: ex.Ref = .{ .buf = try g.scratch(rows * ex.hidden / 8) };
    const y = try g.scratch(4 * rows * ex.hidden);
    const unit: ex.Ref = .{ .buf = try g.buffer(std.mem.sliceAsBytes(ones[0..ids.len])) };
    const t: ex.Ref = .{ .buf = g.table };
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = g.queue.commandBuffer();
    const e = cb.compute(.serial);
    ex.use(e, &g.weights);
    ex.plan(&g.pipes, e, r, call);
    ex.quant(&g.pipes, e, xr, xq, xqs, call);
    e.barrier();
    ex.upCoded(&g.pipes, e, xq, xqs, t, r, .{ .buf = sc[4] }, call);
    e.barrier();
    ex.down(&g.pipes, e, .{ .buf = sc[4] }, t, r, .{ .buf = sc[5] }, call);
    e.barrier();
    ex.combine(&g.pipes, e, .{ .buf = sc[5] }, r, unit, .{ .buf = y }, true, call);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.CommandBufferFailed;
    return .{ sc[4].contents()[0..b[4]], sc[5].contents()[0..b[5]], y.contents()[0 .. 4 * rows * ex.hidden], sc[3].contents()[0..b[3]] };
}

/// Values of every row that differ from its launch alone.
fn compare(g: *Gpu, rows: usize, how: Routing, nonfinite: bool, rng: std.Random) !usize {
    defer g.free();
    const H = ex.hidden;
    const W = ex.width;
    var ids: [16 * topk]u32 = undefined;
    var wts: [16 * topk]f32 = undefined;
    route(&ids, rows, how, rng);
    for (&wts) |*w| w.* = 0.1 + 1.9 * rng.float(f32);
    const x = try testing.allocator.alloc(u16, rows * H);
    defer testing.allocator.free(x);
    for (x) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 4 - 2)) >> 16);
    if (nonfinite) {
        x[H / 3] = 0x7fc0;
        if (rows > 1) x[H + H - 1] = 0x7f80;
        if (rows > 2) x[2 * H] = 0xff80;
    }
    const np = rows * topk;
    const whole = try launch(g, x, ids[0..np], wts[0..np]);
    const wh: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, whole[3]));
    var bad: usize = 0;
    for (0..rows) |r| {
        const one = try launch(g, x[r * H ..][0..H], ids[r * topk ..][0..topk], wts[r * topk ..][0..topk]);
        const oh: []const u32 = @alignCast(std.mem.bytesAsSlice(u32, one[3]));
        for (0..topk) |i| {
            const pw: usize = wh[r * topk + i];
            const po: usize = oh[i];
            bad += differ(whole[0][2 * W * pw ..][0 .. 2 * W], one[0][2 * W * po ..][0 .. 2 * W]);
            bad += differ(whole[1][2 * H * pw ..][0 .. 2 * H], one[1][2 * H * po ..][0 .. 2 * H]);
        }
        bad += differ(whole[2][4 * H * r ..][0 .. 4 * H], one[2][0 .. 4 * H]);
    }
    return bad;
}

fn differ(x: []const u8, y: []const u8) usize {
    var n: usize = 0;
    for (x, y) |p, q| n += @intFromBool(p != q);
    return n;
}

test "dsv41 metal experts: a launch's rows equal each row launched alone at same, disjoint and mixed routing" {
    const a = testing.allocator;
    const device = try mtl.Device.init();
    defer device.deinit();
    var g: Gpu = .{ .device = device, .queue = try device.queue(), .pipes = try ex.Pipelines.load(a, device), .weights = undefined, .table = undefined };
    defer {
        g.free();
        g.bufs.deinit(a);
        g.pipes.deinit();
        for (g.weights) |w| w.deinit();
        g.table.deinit();
        g.queue.deinit();
    }
    var prng = std.Random.DefaultPrng.init(83);
    const rng = prng.random();
    const sizes = [3][2]usize{ .{ ex.width, ex.hidden }, .{ ex.width, ex.hidden }, .{ ex.hidden, ex.width } };
    for (sizes, 0..) |s, i| {
        const len = s[0] * s[1] / 2 + (experts - 1) * stride;
        g.weights[2 * i] = try device.buffer(len, shared);
        rng.bytes(g.weights[2 * i].contents()[0..len]);
        g.weights[2 * i + 1] = try device.buffer(len / 16, shared);
        for (g.weights[2 * i + 1].contents()[0 .. len / 16]) |*v| v.* = 119 + rng.uintLessThan(u8, 7);
    }
    var table: [experts]ex.ExpertPtrs = undefined;
    for (&table, 0..) |*t, e| {
        var v: [6]u64 = undefined;
        for (&v, g.weights, 0..) |*p, w, i| p.* = w.gpuAddress() + e * (if (i % 2 == 0) @as(usize, stride) else stride / 16);
        t.* = .{ .w1p = v[0], .w1s = v[1], .w3p = v[2], .w3s = v[3], .w2p = v[4], .w2s = v[5] };
    }
    g.table = try device.buffer(@sizeOf(@TypeOf(table)), shared);
    @memcpy(g.table.contents()[0..@sizeOf(@TypeOf(table))], std.mem.asBytes(&table));
    var bad: usize = 0;
    var cases: usize = 0;
    for ([_]Routing{ .same, .disjoint, .mixed }) |how| for ([_]usize{ 2, 3, 4, 5, 6, 8, 16 }) |rows| for ([_]bool{ false, true }) |nf| {
        const n = try compare(&g, rows, how, nf, rng);
        if (n > 0) std.debug.print("FAIL dsv41 metal experts rows: {t} routing, {d} rows{s}: {d} bytes differ\n", .{ how, rows, if (nf) ", non-finite" else "", n });
        bad += n;
        cases += 1;
    };
    std.debug.print("{s} dsv41 metal experts rows: same/disjoint/mixed routing at 2-16 rows, {d} cases: act, out and y equal one row a launch\n", .{ if (bad == 0) "PASS" else "FAIL", cases });
    try testing.expectEqual(@as(usize, 0), bad);
}
