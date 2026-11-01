//! A REAL release layer's decode timed at 1 and 4 rows: TF_DSV41_REAL (config + shard), TF_DSV41_REAL_LAYER.
const std = @import("std");
const mtl = @import("metal");
const pkg = @import("dsv41_metal");
const host = @import("dsv41");
const fixture = @import("fixture.zig");
const recorder = @import("recorder.zig");
const mv_sweep = @import("mv_sweep.zig");

const counters = mtl.counters;
const limit = 1280;
const p0 = 1101;

fn env(name: [*:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

/// A rig of the release's config with only `layer` loaded (no embed or head).
fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, layer: u32) !*fixture.Rig {
    const r = try a.create(fixture.Rig);
    errdefer a.destroy(r);
    r.a = a;
    r.c = try host.config.Config.read(a, io, dir);
    r.device = try mtl.Device.init();
    r.queue = try r.device.queue();
    r.p = try pkg.pipelines.Pipelines.load(a, r.device, &r.c);
    var pack = try host.pack.Pack.open(a, io, dir);
    defer pack.deinit();
    r.map = null;
    r.layers = try a.alloc(pkg.weights.Layer, 1);
    r.layers[0] = try pkg.weights.loadFrom(a, io, r.device, &pack, &r.c, layer, null);
    r.top = null;
    r.tables = try pkg.rope.Tables.init(r.device, &r.c, limit);
    r.s = try pkg.scratch.Scratch.init(r.device, &r.c, limit);
    r.slot = try pkg.kv.Slot.init(a, r.device, &r.c, limit);
    const sizes = [4]usize{ 16 * 4 * r.c.hidden * 2, 16 * 4 * r.c.hidden * 2, 16 * 16, 16 * 16 };
    for (&r.io, sizes) |*b, n| b.* = try r.device.buffer(n, mtl.ResourceOptions.shared);
    return r;
}

/// Bytes written between timed launches so no weight stays in the GPU's caches, as in a 40-layer token.
const flush_bytes = 1 << 30;

fn flush(r: *fixture.Rig, junk: mtl.Buffer) void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const b = cb.blit();
    b.fill(junk, 0, flush_bytes, 0x5a);
    b.end();
    cb.commit();
    cb.wait();
}

fn lessThan(_: void, x: f64, y: f64) bool {
    return x < y;
}

fn median(v: []f64) f64 {
    std.mem.sort(f64, v, {}, lessThan);
    return v[v.len / 2];
}

/// The fastest of the runs: other GPU work on this Mac only ever adds time.
fn least(v: []f64) f64 {
    std.mem.sort(f64, v, {}, lessThan);
    return v[0];
}

/// Pipeline ids to names, by walking the Pipelines struct (glue and mv arrays by their registered names).
const Names = struct {
    ids: std.ArrayList(usize) = .empty,
    names: std.ArrayList([]const u8) = .empty,

    fn add(n: *Names, a: std.mem.Allocator, p: mtl.Pipeline, name: []const u8) !void {
        try n.ids.append(a, @intFromPtr(p.id));
        try n.names.append(a, name);
    }

    fn walk(n: *Names, a: std.mem.Allocator, comptime T: type, v: *const T, prefix: []const u8) !void {
        switch (T) {
            mtl.Pipeline => return n.add(a, v.*, prefix),
            ?mtl.Pipeline => return if (v.*) |p| n.add(a, p, prefix),
            pkg.pipelines.glue.Pipelines => for (v.p, 0..) |p, i| try n.add(a, p, pkg.pipelines.glue.names[i]),
            pkg.mv.Pipelines => {
                for (v.p, 0..) |p, i| try n.add(a, p, pkg.mv.names[i]);
                return;
            },
            else => {},
        }
        switch (@typeInfo(T)) {
            .@"struct" => |info| inline for (info.field_names, info.field_types) |fname, ft| {
                try n.walk(a, ft, &@field(v, fname), try std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, fname }));
            },
            .array => |ar| for (v, 0..) |*x, i| try n.walk(a, ar.child, x, try std.fmt.allocPrint(a, "{s}[{d}]", .{ prefix, i })),
            else => {},
        }
    }

    fn of(n: *const Names, id: ?mtl.objc.Id) []const u8 {
        const key = if (id) |i| @intFromPtr(i) else return "?";
        for (n.ids.items, n.names.items) |k, v| if (k == key) return v;
        return "?";
    }
};

/// Calibrated GPU seconds of samples [i] .. [j].
const Clock = struct {
    spt: f64,
    fn span(c: Clock, v: []const counters.TimestampResult, i: usize, j: usize) f64 {
        return @as(f64, @floatFromInt(v[j].timestamp -% v[i].timestamp)) * c.spt;
    }
};

fn clockOf(before: counters.ClockSample, after: counters.ClockSample) Clock {
    const cpu: f64 = @floatFromInt(after.cpu - before.cpu);
    const gpu: f64 = @floatFromInt(after.gpu - before.gpu);
    return .{ .spt = cpu / gpu * 1e-9 };
}

const Run = struct { host: f64, gpu: f64, enc: f64, each: []f64 };

/// The direct launch (host encode, GPU span), then its recording replayed whole and one dispatch an encoder.
fn once(a: std.mem.Allocator, r: *fixture.Rig, junk: mtl.Buffer, rec: *recorder.Recording, rows: u32, samples: counters.SampleBuffer, pass: counters.ComputePass, conc: counters.ComputePass) !Run {
    const w = &r.layers[0];
    const ctx = r.ctx(&r.slot, p0, rows);
    try r.s.writeRows(&r.c, p0, rows);
    const s = r.streams();
    const pr = r.pres();
    const io: pkg.layer.Io = .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] };
    var out: Run = undefined;
    {
        flush(r, junk);
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const before = counters.sampleClocks(r.device);
        const t0 = mtl.clock.seconds();
        const cb = r.queue.commandBuffer();
        const e = try conc.compute(cb, samples, 0, 1);
        try pkg.layer.encode(&ctx, e, w, io);
        e.end();
        out.host = mtl.clock.seconds() - t0;
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| std.debug.print("command buffer failed: {s}\n", .{msg});
        const res = try samples.resolveCompleted(2);
        defer res.deinit();
        out.gpu = clockOf(before, counters.sampleClocks(r.device)).span(res.values(), 0, 1);
    }
    const re = recorder.start(rec);
    try pkg.layer.encode(&ctx, re, w, io);
    recorder.stop(re);
    const n = rec.dispatches();
    out.each = try a.alloc(f64, n);
    flush(r, junk);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const before = counters.sampleClocks(r.device);
    const cb = r.queue.commandBuffer();
    const e = try pass.compute(cb, samples, 0, 1);
    rec.replay(e);
    e.end();
    for (0..n) |i| {
        const ei = try pass.compute(cb, samples, 2 + 2 * i, 3 + 2 * i);
        _ = rec.replayOne(ei, i);
        ei.end();
    }
    cb.commit();
    cb.wait();
    const after = counters.sampleClocks(r.device);
    const clk = clockOf(before, after);
    const res = try samples.resolveCompleted(2 + 2 * n);
    defer res.deinit();
    const v = res.values();
    out.enc = clk.span(v, 0, 1);
    for (out.each, 0..) |*x, i| x.* = clk.span(v, 2 + 2 * i, 3 + 2 * i);
    return out;
}

fn report(a: std.mem.Allocator, r: *fixture.Rig, junk: mtl.Buffer, names: *const Names, rows: u32, n: usize) !void {
    var rec: recorder.Recording = .{ .a = a };
    defer rec.deinit();
    const samples = try counters.SampleBuffer.init(r.device, 4096);
    defer samples.deinit();
    const pass = try counters.ComputePass.init();
    defer pass.deinit();
    // the serving path's encoder: concurrent, barriers where the layer puts them
    const conc = try counters.ComputePass.init();
    defer conc.deinit();
    mtl.objc.msg(void, conc.id, "setDispatchType:", .{@as(usize, @backingInt(mtl.DispatchType.concurrent))});
    const runs = try a.alloc(Run, n);
    for (runs) |*x| x.* = try once(a, r, junk, &rec, rows, samples, pass, conc);
    const v = try a.alloc(f64, n);
    for (runs, v) |x, *y| y.* = x.gpu;
    const gpu_med = median(v);
    const gpu = least(v);
    for (runs, v) |x, *y| y.* = x.host;
    const hst = median(v);
    for (runs, v) |x, *y| y.* = x.enc;
    const enc = least(v);
    const enc_med = median(v);
    const d = runs[0].each.len;
    // per dispatch medians, then summed by kernel name in first-seen order
    var cls: std.ArrayList([]const u8) = .empty;
    var tot: std.ArrayList(f64) = .empty;
    var cnt: std.ArrayList(usize) = .empty;
    var sum: f64 = 0;
    var pipes = try a.alloc(?mtl.objc.Id, d);
    {
        const re = recorder.start(&rec);
        const ctx = r.ctx(&r.slot, p0, rows);
        const s = r.streams();
        const pr = r.pres();
        try pkg.layer.encode(&ctx, re, &r.layers[0], .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
        recorder.stop(re);
        var k: usize = 0;
        var pipe: ?mtl.objc.Id = null;
        for (rec.ops.items) |op| switch (op) {
            .pipe => |p| pipe = p,
            .groups, .threads => {
                pipes[k] = pipe;
                k += 1;
            },
            else => {},
        };
    }
    for (0..d) |i| {
        for (runs, v) |x, *y| y.* = x.each[i];
        const m = least(v);
        sum += m;
        const name = names.of(pipes[i]);
        const at = for (cls.items, 0..) |c, j| {
            if (std.mem.eql(u8, c, name)) break j;
        } else blk: {
            try cls.append(a, name);
            try tot.append(a, 0);
            try cnt.append(a, 0);
            break :blk cls.items.len - 1;
        };
        tot.items[at] += m;
        cnt.items[at] += 1;
    }
    std.debug.print("layer {d}, {d} row(s), fastest of {d}, {d} dispatches:\n", .{ r.layers[0].layer, rows, n, d });
    std.debug.print("  concurrent (serving) encoder {d:.1} us (median {d:.1}), host encode {d:.1} us; one serial encoder {d:.1} us (median {d:.1}); dispatches alone {d:.1} us\n", .{ gpu * 1e6, gpu_med * 1e6, hst * 1e6, enc * 1e6, enc_med * 1e6, sum * 1e6 });
    std.debug.print("  x40: {d:.2} ms a round, {d:.1} tok/s at {d} row(s) a round (serial {d:.2} ms)\n", .{ 40 * gpu * 1e3, @as(f64, @floatFromInt(rows)) / (40 * gpu), rows, 40 * enc * 1e3 });
    for (cls.items, tot.items, cnt.items) |c, t, k| std.debug.print("    {s:<34} x{d:<3} {d:>9.1} us\n", .{ c, k, t * 1e6 });
    if (env("TF_DSV41_TIMING_EACH") != null) for (0..d) |i| {
        for (runs, v) |x, *y| y.* = x.each[i];
        std.debug.print("      #{d:<3} {s:<34} {d:>9.1} us\n", .{ i, names.of(pipes[i]), least(v) * 1e6 });
    };
}

/// Each dense FP8 projection of the layer alone at 1, 4 and 5 rows, by block kind, interleaved.
fn fp8Shapes(a: std.mem.Allocator, r: *fixture.Rig, junk: mtl.Buffer, n: usize) !void {
    const fp8 = pkg.pipelines.fp8;
    const w = &r.layers[0];
    const Shape = struct { name: []const u8, w: pkg.weights.Fp8 };
    const shapes = [_]Shape{ .{ .name = "wq_a", .w = w.wq_a }, .{ .name = "wkv", .w = w.wkv }, .{ .name = "wq_b", .w = w.wq_b }, .{ .name = "wo_b", .w = w.wo_b }, .{ .name = "shared w1", .w = w.shared[0] }, .{ .name = "shared w2", .w = w.shared[1] }, .{ .name = "ix_wq_b", .w = w.ix_wq_b.? } };
    const samples = try counters.SampleBuffer.init(r.device, 64);
    defer samples.deinit();
    const pass = try counters.ComputePass.init();
    defer pass.deinit();
    const kinds = [_]fp8.Kind{ .b, .c };
    const t = try a.alloc(f64, n * kinds.len);
    for ([_]u32{ 1, 4, 5 }) |rows| for (shapes) |sh| {
        const call: fp8.Call = .{ .k = sh.w.k, .n = sh.w.n, .rows = rows, .split = fp8.splitOf(sh.w.k), .out = .bf16 };
        for (0..n) |i| for (kinds, 0..) |kind, j| {
            flush(r, junk);
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const before = counters.sampleClocks(r.device);
            const cb = r.queue.commandBuffer();
            const e = try pass.compute(cb, samples, 0, 1);
            try fp8.encodeWholeAs(&r.p.fp8, e, .{ .buf = r.s.hq.buf, .off = r.s.hq.off }, .{ .buf = sh.w.w.buf, .off = sh.w.w.off }, .{ .buf = sh.w.s.buf, .off = sh.w.s.off }, .{ .buf = r.s.q.buf, .off = r.s.q.off }, call, .{}, kind);
            e.end();
            cb.commit();
            cb.wait();
            const res = try samples.resolveCompleted(2);
            defer res.deinit();
            t[j * n + i] = clockOf(before, counters.sampleClocks(r.device)).span(res.values(), 0, 1);
        };
        std.debug.print("  fp8 {s:<10} {d}x{d} at {d} row(s), launched as {s}:", .{ sh.name, sh.w.k, sh.w.n, rows, @tagName(fp8.kindOf(call)) });
        for (kinds, 0..) |kind, j| std.debug.print("  {s} {d:.1} us", .{ @tagName(kind), least(t[j * n ..][0..n]) * 1e6 });
        std.debug.print("\n", .{});
    };
}

/// One timed launch of `f` (a dispatch sequence) on a flushed cache, in a sampled concurrent encoder.
fn timeOnce(r: *fixture.Rig, junk: mtl.Buffer, samples: counters.SampleBuffer, pass: counters.ComputePass, ctx: anytype, comptime f: anytype) !f64 {
    flush(r, junk);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const before = counters.sampleClocks(r.device);
    const cb = r.queue.commandBuffer();
    const e = try pass.compute(cb, samples, 0, 1);
    try f(ctx, e);
    e.end();
    cb.commit();
    cb.wait();
    const res = try samples.resolveCompleted(2);
    defer res.deinit();
    return clockOf(before, counters.sampleClocks(r.device)).span(res.values(), 0, 1);
}

/// Bytes that no page or cache line repeats, bounded so bf16 and E4M3 stay finite (`mask` clears the top bits).
fn noise(out: []u8, mask: u8) void {
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().bytes(out);
    for (out) |*b| b.* &= mask | 0x80;
}

/// The rest of a token beside its layers, on synthetic weights of the release's shapes: bf16 head, Engram FP8 wkv.
fn rest(a: std.mem.Allocator, r: *fixture.Rig, junk: mtl.Buffer, n: usize) !void {
    const c = &r.c;
    const H: usize = c.hidden;
    const V: usize = c.vocab;
    const K: usize = @as(usize, c.hashCols()) * c.engram_head_dim;
    const N: usize = (@as(usize, c.hc_mult) + 1) * H;
    const head = try r.device.buffer(V * H * 2, mtl.ResourceOptions.shared);
    defer head.deinit();
    noise(head.contents()[0 .. V * H * 2], 0x3c);
    const logits = try r.device.buffer(16 * V * 4, mtl.ResourceOptions.private);
    defer logits.deinit();
    const ew = try r.device.buffer(N * K, mtl.ResourceOptions.shared);
    defer ew.deinit();
    noise(ew.contents()[0 .. N * K], 0x77);
    const es = try r.device.buffer(N / 32 * (K / 32), mtl.ResourceOptions.shared);
    defer es.deinit();
    @memset(es.contents()[0 .. N / 32 * (K / 32)], 120);
    const ex = try r.device.buffer(16 * K * 2, mtl.ResourceOptions.shared);
    defer ex.deinit();
    @memset(ex.contents()[0 .. 16 * K * 2], 0);
    const ekv = try r.device.buffer(16 * N * 2, mtl.ResourceOptions.private);
    defer ekv.deinit();
    const samples = try counters.SampleBuffer.init(r.device, 64);
    defer samples.deinit();
    const pass = try counters.ComputePass.init();
    defer pass.deinit();
    const Args = struct { r: *fixture.Rig, rows: u32, head: mtl.Buffer, logits: mtl.Buffer, ew: mtl.Buffer, es: mtl.Buffer, ex: mtl.Buffer, ekv: mtl.Buffer, k: u32, n: u32 };
    const F = struct {
        fn headMv(x: Args, e: mtl.ComputeEncoder) !void {
            pkg.pipelines.glue.mv(&x.r.p.glue, e, x.r.s.hn, .{ .buf = x.head }, .{ .buf = x.logits }, x.rows, x.r.c.hidden, x.r.c.vocab, true);
        }
        fn wkv(x: Args, e: mtl.ComputeEncoder) !void {
            const fp8 = pkg.pipelines.fp8;
            const call: fp8.Call = .{ .k = x.k, .n = x.n, .rows = x.rows, .split = fp8.splitOf(x.k), .out = .bf16 };
            try fp8.encodeWhole(&x.r.p.fp8, e, .{ .buf = x.ex }, .{ .buf = x.ew }, .{ .buf = x.es }, .{ .buf = x.ekv }, call, .{});
        }
    };
    const t = try a.alloc(f64, n);
    for ([_]u32{ 1, 4 }) |rows| {
        const x: Args = .{ .r = r, .rows = rows, .head = head, .logits = logits, .ew = ew, .es = es, .ex = ex, .ekv = ekv, .k = @intCast(K), .n = @intCast(N) };
        for (t) |*v| v.* = try timeOnce(r, junk, samples, pass, x, F.headMv);
        const th = least(t);
        for (t) |*v| v.* = try timeOnce(r, junk, samples, pass, x, F.wkv);
        const te = least(t);
        std.debug.print("rest of a token at {d} row(s): head {d}x{d} bf16 {d:.1} us, Engram wkv {d}x{d} FP8 {d:.1} us (x2)\n", .{ rows, H, V, th * 1e6, K, N, te * 1e6 });
    }
}

/// A hash of one launch's outputs (streams, pre mix, the layer's KV state) from a fresh past, for before/after checks.
fn hashOf(r: *fixture.Rig, rows: u32, kind: mtl.DispatchType) ![2]u64 {
    fixture.fillPast(&r.c, &r.slot, p0, 21);
    fixture.fillInput(r, rows, 31);
    const ctx = r.ctx(&r.slot, p0, rows);
    try r.s.writeRows(&r.c, p0, rows);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(kind);
    const s = r.streams();
    const pr = r.pres();
    try pkg.layer.encode(&ctx, e, &r.layers[0], .{ .x = s[0], .pre = pr[0], .x_out = s[1], .pre_out = pr[1] });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| std.debug.print("command buffer failed: {s}\n", .{msg});
    var h = std.hash.Wyhash.init(0);
    h.update(r.io[1].contents()[0 .. rows * 4 * r.c.hidden * 2]);
    h.update(r.io[3].contents()[0 .. rows * 16]);
    var k = std.hash.Wyhash.init(0);
    const reg = r.slot.regions[r.layers[0].layer];
    for ([_]?mtl.Buffer{ reg.window, reg.pool, reg.keys, reg.carry, reg.inputs }) |b| if (b) |x| k.update(x.contents()[0..x.length()]);
    return .{ h.final(), k.final() };
}

/// The serial encoder's hashes, and whether every one of `n` concurrent launches gives the same bits.
fn bits(r: *fixture.Rig, rows: u32, n: usize) !void {
    const want = try hashOf(r, rows, .serial);
    var same: usize = 0;
    for (0..n) |_| same += @intFromBool(std.mem.eql(u64, &want, &(try hashOf(r, rows, .concurrent))));
    std.debug.print("bits: layer {d}, {d} row(s): outputs {x:0>16}, KV state {x:0>16}; concurrent equal {d} of {d}\n", .{ r.layers[0].layer, rows, want[0], want[1], same, n });
    if (same != n) return error.ConcurrentBitsDiffer;
}

test "dsv41 metal REAL layer decode timing at 1 and 4 rows" {
    const dir = env("TF_DSV41_REAL") orelse return error.SkipZigTest;
    const layer = if (env("TF_DSV41_REAL_LAYER")) |v| try std.fmt.parseInt(u32, v, 10) else 2;
    const n = if (env("TF_DSV41_TIMING_N")) |v| try std.fmt.parseInt(usize, v, 10) else 21;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try open(std.heap.page_allocator, std.testing.io, dir, layer);
    defer r.deinit();
    const junk = try r.device.buffer(flush_bytes, mtl.ResourceOptions.private);
    defer junk.deinit();
    var names: Names = .{};
    try names.walk(a, pkg.pipelines.Pipelines, &r.p, "p");
    if (env("TF_DSV41_TIMING_MV") != null) return mv_sweep.run(r.device, r.queue, junk, n);
    for ([_]u32{ 1, 4 }) |rows| try bits(r, rows, 20);
    fixture.fillPast(&r.c, &r.slot, p0, 21);
    // TF_DSV41_TIMING_ROWS: a comma list of row counts to time (default 1,4; at most 16)
    var it = std.mem.tokenizeScalar(u8, env("TF_DSV41_TIMING_ROWS") orelse "1,4", ',');
    while (it.next()) |t| {
        const rows = try std.fmt.parseInt(u32, t, 10);
        fixture.fillInput(r, rows, 31);
        // TF_DSV41_TIMING_SAME: every row's input is row 0's, so the rows mostly share their experts (a draft block)
        if (env("TF_DSV41_TIMING_SAME") != null) {
            const x = r.io[0].slice(u16, rows * 4 * r.c.hidden);
            for (1..rows) |i| @memcpy(x[i * 4 * r.c.hidden ..][0 .. 4 * r.c.hidden], x[0 .. 4 * r.c.hidden]);
            const pre = r.io[2].slice(f32, rows * 4);
            for (1..rows) |i| @memcpy(pre[i * 4 ..][0..4], pre[0..4]);
        }
        try report(a, r, junk, &names, rows, n);
        std.debug.print("  routed experts a round: {d} for {d} picks\n", .{ std.mem.bytesToValue(u32, r.s.bytes(r.s.plan[1], 4)), rows * r.c.topk });
    }
    if (env("TF_DSV41_TIMING_FP8") != null) try fp8Shapes(a, r, junk, n);
    if (env("TF_DSV41_TIMING_REST") != null) try rest(a, r, junk, n);
    // the same weights as a window-only layer (ratio 0): the cost of a layer that neither compresses nor indexes
    if (env("TF_DSV41_TIMING_WINDOW") == null) return;
    r.c.compress_ratios.buf[layer] = 0;
    std.debug.print("as a window-only layer (compress ratio 0):\n", .{});
    for ([_]u32{ 1, 4 }) |rows| {
        fixture.fillInput(r, rows, 31);
        try report(a, r, junk, &names, rows, n);
    }
}
