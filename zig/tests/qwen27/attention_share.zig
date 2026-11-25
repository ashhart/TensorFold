//! KV-page attention, streams on one prefix's pages: Q27_JOIN's pairs against each row alone, bit for bit, and timed.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const attn = q.attention;
const planning = attn.planning;

const dim = 256;
const page = attn.page_rows;
const prefix_pages = 136; // 34,816 keys: the prefix ends on a page, each stream's rows on a page of its own
const start: u32 = prefix_pages * page;
const capacity: u32 = start + page;
const most_streams = 8;
const most_rows = 64; // 8 streams of a drafted window of 8
const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const reps = 15;
const passes = 16; // encodes a command buffer, as a forward's attention layers
const q_row = 6144; // halves a row's query (24 heads of 256), out the same
const kv_row = 1024;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s >> 12;
        r.s ^= r.s << 25;
        r.s ^= r.s >> 27;
        return r.s *% 0x2545f4914f6cdd1d;
    }
    /// A bf16 in [-2, 2): the upper half of an f32.
    fn bf16(r: *Rng) u16 {
        const unit: f32 = @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
        return @truncate(@as(u32, @bitCast(unit * 4.0 - 2.0)) >> 16);
    }
};

fn random(device: mtl.Device, rng: *Rng, halves: usize) !mtl.Buffer {
    const b = try device.buffer(halves * 2, options);
    for (b.slice(u16, halves)) |*x| x.* = rng.bf16();
    return b;
}

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    op: *const attn.Attention,
    scratch: *attn.Scratch,
    limits: attn.Limits,
    query: mtl.Buffer,
    key: mtl.Buffer,
    value: mtl.Buffer,
    gate: mtl.Buffer,

    /// One pass of `plan` over `caches`, its rows' inputs from row `row0`, out at `out` row 0; seconds on the GPU.
    fn run(r: *Rig, plan: attn.Plan, caches: []const attn.Cache, row0: usize, out: mtl.Buffer, n: usize) !f64 {
        try r.scratch.upload(plan);
        var b = try attn.Binding.init(r.device, r.limits);
        defer b.deinit();
        try b.upload(plan, caches);
        const cb = r.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        for (0..n) |_| try r.op.encode(e, r.scratch, &b, plan, .{ .buffer = r.query, .offset = row0 * q_row * 2 }, .{ .buffer = r.key, .offset = row0 * kv_row * 2 }, .{ .buffer = r.value, .offset = row0 * kv_row * 2 }, .{ .buffer = out }, caches, .{ .buffer = r.gate, .offset = row0 * 2 * q_row * 2 }, false, false);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        return cb.gpuSeconds();
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const units = device.tensorUnits();
    const geometry = planning.Geometry{ .shared_prefix = units };
    const kvh = geometry.kv_heads;
    if (!units) try q.simd_attention.check(device, queue);
    const limits = attn.Limits{ .rows = most_rows, .streams = most_streams, .keys = capacity, .shared_prefix = units, .geometry = geometry };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    // main's pipelines, and the join's beside them (taken only for a plan with pairs)
    const text = try q.simd_attention.grouped(a, units, true, geometry.group(), kvh, true);
    const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer library.deinit();
    var op = try attn.Attention.init(device, library);
    const joined = try q.simd_attention.joined(a, units, false, geometry.group(), kvh, true);
    const join_library = try mtl.Library.fromSource(device, joined, mtl.CompileOptions.mlx());
    defer join_library.deinit();
    op.join = try q.attention_join.pipelines(device, join_library);
    defer op.deinit();

    // pages: the prefix's, each stream's copy of it, then each stream's own page for its rows, and one planted copy
    const part = @as(u64, kvh) * page * dim * 2;
    const page_bytes = 2 * part;
    const per = attn.pageCount(capacity);
    const copies = prefix_pages;
    const own = copies + most_streams * prefix_pages;
    const planted = own + most_streams;
    const total = planted + 1;
    const arena = try device.buffer(@intCast(total * page_bytes), options);
    defer arena.deinit();
    var rng = Rng{ .s = 0x5eed_0e1 };
    const halves: usize = @intCast(page_bytes / 2);
    const all = arena.slice(u16, total * halves);
    for (all[0 .. prefix_pages * halves]) |*x| x.* = rng.bf16();
    for (0..most_streams) |st| @memcpy(all[(copies + st * prefix_pages) * halves ..][0 .. prefix_pages * halves], all[0 .. prefix_pages * halves]);
    @memcpy(all[planted * halves ..][0..halves], all[5 * halves ..][0..halves]);
    all[planted * halves + 77] ^= 0x0100; // one value of prefix page 5's keys changed in its planted copy
    // tables: 0 stream st on the prefix's ids, 1 on its copy's, 2 as 0 with stream 1's page 5 the planted copy
    var tables: [3]mtl.Buffer = undefined;
    for (&tables, 0..) |*t, v| {
        t.* = try device.buffer(most_streams * per * 4, options);
        const ids = t.slice(u32, most_streams * per);
        for (0..most_streams) |st| {
            for (0..prefix_pages) |p| ids[st * per + p] = @intCast(if (v == 1) copies + st * prefix_pages + p else p);
            ids[st * per + prefix_pages] = @intCast(own + st);
        }
        if (v == 2) ids[per + 5] = @intCast(planted);
    }
    defer for (tables) |t| t.deinit();
    var rig = Rig{ .device = device, .queue = queue, .op = &op, .scratch = &scratch, .limits = limits, .query = try random(device, &rng, most_rows * q_row), .key = try random(device, &rng, most_rows * kv_row), .value = try random(device, &rng, most_rows * kv_row), .gate = try random(device, &rng, most_rows * 2 * q_row) };
    var outs: [3]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = try device.buffer(most_rows * q_row * 2, options);
    defer for ([_]mtl.Buffer{ rig.query, rig.key, rig.value, rig.gate, outs[0], outs[1], outs[2] }) |b| b.deinit();
    std.debug.print("attention share on {s}, tensor units {s}: {d} prefix pages of {d} bytes, {d} encodes a pass\n", .{ std.mem.span(device.name()), if (units) "yes" else "no", prefix_pages, page_bytes, passes });
    if (units) return error.JoinNeedsTheTailPlan; // Plan.pair takes no pairs with the prefix kernel's plan (the M5's)

    var caches: [3][most_streams]attn.Cache = undefined;
    for (&caches, 0..) |*cs, v| for (cs, 0..) |*c, st| {
        c.* = .{ .keys = .{ .buffer = arena }, .values = .{ .buffer = arena, .offset = part }, .capacity = capacity, .pages = .{ .table = .{ .buffer = tables[v], .offset = st * per * 4 }, .page_bytes = page_bytes } };
    };
    var parents: [8]i32 = undefined;
    for (&parents, 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1; // a drafted window's chain
    for ([_][2]usize{ .{ 8, 1 }, .{ 4, 1 }, .{ 4, 8 } }) |shape| {
        const n = shape[0];
        const rows = shape[1];
        var windows: [most_streams]planning.Window = undefined;
        for (windows[0..n]) |*w| w.* = .{ .parents = parents[0..rows], .start = start, .capacity = capacity };
        var shares: [most_streams]planning.Share = undefined;
        for (shares[0..n]) |*s| s.* = .{ .group = 1, .keys = start };
        var plan = try attn.Plan.init(a, windows[0..n], geometry);
        defer plan.deinit();
        var paired = try attn.Plan.init(a, windows[0..n], geometry);
        defer paired.deinit();
        paired.pair(shares[0..n]);
        if (paired.pairs != n * rows / 2) return error.PairsMissing;
        const bytes = n * rows * q_row * 2;
        // exact: the pairs' rows against each stream alone through main's kernels, and against main's unpaired plan
        _ = try rig.run(paired, caches[0][0..n], 0, outs[0], 1);
        _ = try rig.run(plan, caches[0][0..n], 0, outs[1], 1);
        if (!std.mem.eql(u8, outs[0].contents()[0..bytes], outs[1].contents()[0..bytes])) return error.PairsDiffer;
        for (0..n) |st| {
            var one = try attn.Plan.init(a, windows[st .. st + 1], geometry);
            defer one.deinit();
            _ = try rig.run(one, caches[0][st .. st + 1], st * rows, outs[2], 1);
            const at = st * rows * q_row * 2;
            if (!std.mem.eql(u8, outs[0].contents()[at..][0 .. rows * q_row * 2], outs[2].contents()[0 .. rows * q_row * 2])) return error.StreamDiffers;
        }
        // control (streams paired across, one row each): a share claimed over a page stream 1 holds differently shows
        if (rows == 1) {
            _ = try rig.run(paired, caches[2][0..n], 0, outs[2], 1);
            _ = try rig.run(plan, caches[2][0..n], 0, outs[1], 1);
            if (std.mem.eql(u8, outs[2].contents()[q_row * 2 ..][0 .. q_row * 2], outs[1].contents()[q_row * 2 ..][0 .. q_row * 2])) return error.ControlMissed;
        }
        // time: main's kernels on shared pages and on copies, and the pairs on shared pages
        var us: [3][reps]f64 = undefined;
        for (0..reps) |rep| for (0..3) |w| {
            const v = (w + rep) % 3;
            us[v][rep] = 1e6 / @as(f64, passes) * try rig.run(if (v == 2) paired else plan, caches[if (v == 1) 1 else 0][0..n], 0, outs[v], passes);
        };
        for (&us) |*t| std.mem.sort(f64, t, {}, std.sort.asc(f64));
        std.debug.print("RESULT attention share {d} streams x {d} rows at {d} keys: main's kernels {d:.1} us a pass on shared pages, {d:.1} on copies; pairs {d:.1}; median of {d}; pairs == alone\n", .{ n, rows, start + 1, us[0][reps / 2], us[1][reps / 2], us[2][reps / 2], reps });
    }
}
