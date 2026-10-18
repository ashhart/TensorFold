//! Bolt's ggml formats on the GPU (ggml.cu): real tensors' repack, grouped IQ2_XXS experts and dense rows against fp64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const ggml = cuda.ggml;
const grouped = cuda.grouped;
const ex_ = cuda.experts;

fn arr(fx: Fixture, comptime fmt: []const u8, args: anytype) ![]u8 {
    var name: [48]u8 = undefined;
    return fx.bytes(try std.fmt.bufPrint(&name, fmt, args));
}

fn bf(v: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
}

fn relu2(v: f64) f64 {
    const u = @max(v, 0);
    return u * u;
}

/// Max |got - want| and max |want| over outputs fp32 or bf16; fails past `tol` of the largest reference.
fn compare(gpu: Gpu, what: []const u8, b: cuda.DeviceBuffer, ref: []const u8, out: ggml.Out, tol: f64) !void {
    const got = try check.download(gpu, b);
    defer gpu.gpa.free(got);
    const want = std.mem.bytesAsSlice(f64, @as([]align(1) const u8, ref));
    var err: f64 = 0;
    var top: f64 = 0;
    for (want, 0..) |w0, i| {
        const w = if (out == .relu2) relu2(w0) else w0;
        const g: f64 = if (out == .f32) @floatCast(std.mem.bytesToValue(f32, got[4 * i ..][0..4])) else bf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]));
        if (!(g == g)) return check.expect(false, "{s}: NaN at {d}", .{ what, i });
        err = @max(err, @abs(g - w));
        top = @max(top, @abs(w));
    }
    try check.expect(err <= tol * top, "{s}: max abs error {e:.3} is {e:.3} of max |ref| {e:.3}", .{ what, err, err / top, top });
    check.pass("{s}: max abs error {e:.3}, {e:.3} of max |ref| {e:.3}", .{ what, err, err / top, top });
}

/// fp32 sums; bf16 rounds once (2^-8 relative); relu^2 rounds its input to bf16 too, then squares it.
fn tolerance(out: ggml.Out) f64 {
    return switch (out) {
        .f32 => 1e-5,
        .bf16 => 1.0 / 256.0,
        .relu2 => 3.0 / 256.0,
    };
}

/// Raw blocks packed on the device equal the host repack.
fn packed_(gpu: Gpu, k: *const ggml.Kernels, s: cuda.Stream, kind: ggml.Kind, raw: []const u8, n: usize, kk: usize, count: usize) !cuda.DeviceBuffer {
    var draw = try cuda.DeviceBuffer.fromHost(gpu.d, raw);
    defer draw.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, raw.len);
    errdefer out.free();
    try k.pack(s, kind, draw.ptr, out.ptr, n, kk, count);
    try s.synchronize();
    const got = try check.download(gpu, out);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(u8, raw.len);
    defer gpu.gpa.free(want);
    const one = ggml.bytes(kind, n, kk);
    for (0..count) |m| ggml.packHost(kind, want[m * one ..][0..one], raw[m * one ..][0..one], n, kk);
    try check.sameBytes("ggml pack", got, want);
    return out;
}

pub fn run(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    var k = try ggml.Kernels.load(gpu.d);
    defer k.unload();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    try experts(gpu, fx, &k, stream);
    try dense(gpu, fx, &k, stream);
}

fn experts(gpu: Gpu, fx: Fixture, k: *const ggml.Kernels, s: cuda.Stream) !void {
    const a = gpu.gpa;
    const e: usize = @intCast(try fx.int("experts"));
    const d: usize = @intCast(try fx.int("dims"));
    const ni: usize = @intCast(try fx.int("width"));
    const slots: usize = @intCast(try fx.int("slots"));
    const up_raw = try fx.bytes("up_raw");
    defer a.free(up_raw);
    const down_raw = try fx.bytes("down_raw");
    defer a.free(down_raw);
    var up = try packed_(gpu, k, s, .iq2xxs, up_raw, ni, d, e);
    defer up.free();
    var down = try packed_(gpu, k, s, .iq2xxs, down_raw, d, ni, e);
    defer down.free();
    check.pass("ggml pack: {d} real IQ2_XXS experts' up [{d}, {d}] and down on the device equal the host repack", .{ e, ni, d });
    const layer: ex_.Layer = .{ .format = .iq2xxs, .up = up.ptr, .down = down.ptr, .width = ni, .dims = d, .experts = e, .relu2 = true };
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    const g: ex_.Grouped = .{ .iq2xxs = k };
    inline for (.{ .{ "decode", false }, .{ "prompt", true } }) |set| {
        var it = std.mem.tokenizeScalar(u8, try fx.string(set[0]), ',');
        while (it.next()) |tok| {
            const rows = try std.fmt.parseInt(usize, tok, 10);
            try expertCase(gpu, fx, router, g, k, s, layer, rows, slots, set[1]);
            if (set[1]) try expertCase(gpu, fx, router, g, k, s, layer, rows, slots, false); // decode form at prompt width
        }
    }
}

fn expertCase(gpu: Gpu, fx: Fixture, router: grouped.Router, g: ex_.Grouped, k: *const ggml.Kernels, s: cuda.Stream, l: ex_.Layer, rows: usize, slots: usize, prompt: bool) !void {
    const a = gpu.gpa;
    const pairs = rows * slots;
    const tile: usize = if (prompt) ex_.promptTile(.iq2xxs) else 16;
    const picks = try arr(fx, "picks{d}", .{rows});
    defer a.free(picks);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, picks);
    defer dpicks.free();
    const big = pairs > grouped.small;
    const bound = grouped.maxItems(pairs, l.experts, tile);
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const sizes = [5]usize{ pairs * 4, bound * 12, 8, (if (big) pairs else 1) * 4, (if (big) (pairs + 1023) / 1024 * l.experts else 1) * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
        made += 1;
    }
    const plan: grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    try router.route(s, dpicks.ptr, pairs, l.experts, tile, plan);

    const x = try arr(fx, "x{d}", .{rows});
    defer a.free(x);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
    defer dx.free();
    const h = try arr(fx, "h{d}", .{rows});
    defer a.free(h);
    var dh = try cuda.DeviceBuffer.fromHost(gpu.d, h);
    defer dh.free();
    const ref_up = try arr(fx, "up{d}", .{rows});
    defer a.free(ref_up);
    const ref_down = try arr(fx, "down{d}", .{rows});
    defer a.free(ref_down);
    var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * @max(l.width, l.dims) * 4);
    defer y.free();
    const form = if (prompt) "prompt" else "decode";
    var name: [96]u8 = undefined;

    // up: relu^2 through experts.Grouped, then fp32 sums by the same kernels with no activation
    try y.fill8(0xff, s.handle); // a call that writes nothing fails rather than pass on old bytes
    const call = if (prompt) &ex_.Grouped.prompt else &ex_.Grouped.decode;
    try call(g, s, .up, .{ .x = dx.ptr, .stride = l.dims, .slots = slots }, l, plan, y.ptr, bound, -1);
    try s.synchronize();
    try compare(gpu, try std.fmt.bufPrint(&name, "iq2 up relu^2 {s} {d} rows x {d}", .{ form, rows, slots }), y, ref_up, .relu2, tolerance(.relu2));
    try y.fill8(0xff, s.handle);
    try k.experts(s, prompt, .iq2xxs, .f32, dx.ptr, l.dims, slots, l.up, l.width, l.dims, plan, y.ptr, bound, -1);
    try s.synchronize();
    try compare(gpu, try std.fmt.bufPrint(&name, "iq2 up fp32 {s} {d} rows x {d}", .{ form, rows, slots }), y, ref_up, .f32, tolerance(.f32));
    // down from pair rows, fp32 and bf16
    inline for (.{ .{ ex_.Epi.down_f32, ggml.Out.f32 }, .{ ex_.Epi.down_bf16, ggml.Out.bf16 } }) |c| {
        try y.fill8(0xff, s.handle);
        try call(g, s, c[0], .{ .x = dh.ptr, .stride = l.width }, l, plan, y.ptr, bound, -1);
        try s.synchronize();
        try compare(gpu, try std.fmt.bufPrint(&name, "iq2 down {t} {s} {d} rows x {d}", .{ c[1], form, rows, slots }), y, ref_down, c[1], tolerance(c[1]));
    }
}

fn dense(gpu: Gpu, fx: Fixture, k: *const ggml.Kernels, s: cuda.Stream) !void {
    const a = gpu.gpa;
    var cases = std.mem.tokenizeScalar(u8, try fx.string("dense"), ',');
    while (cases.next()) |case| {
        var name: [96]u8 = undefined;
        const n: usize = @intCast(try fx.int(try std.fmt.bufPrint(&name, "{s}_n", .{case})));
        const kk: usize = @intCast(try fx.int(try std.fmt.bufPrint(&name, "{s}_k", .{case})));
        const kind: ggml.Kind = if (std.mem.eql(u8, try fx.string(try std.fmt.bufPrint(&name, "{s}_kind", .{case})), "Q8_0")) .q8_0 else .iq2xxs;
        const raw = try arr(fx, "{s}_raw", .{case});
        defer a.free(raw);
        var w_buf = try packed_(gpu, k, s, kind, raw, n, kk, 1);
        defer w_buf.free();
        const w = ggml.Weight{ .kind = kind, .w = w_buf.ptr, .n = n, .k = kk };
        var rows_it = std.mem.tokenizeScalar(u8, try fx.string("dense_rows"), ',');
        while (rows_it.next()) |tok| {
            const rows = try std.fmt.parseInt(usize, tok, 10);
            const x = try arr(fx, "{s}_x{d}", .{ case, rows });
            defer a.free(x);
            var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
            defer dx.free();
            const ref = try arr(fx, "{s}_y{d}", .{ case, rows });
            defer a.free(ref);
            var y = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 4);
            defer y.free();
            const forms: []const bool = if (rows <= ggml.decode_rows) &.{ false, true } else &.{true};
            for (forms) |prompt| for ([_]ggml.Out{ .f32, .bf16 }) |out| {
                try y.fill8(0xff, s.handle);
                if (prompt) try k.promptRows(s, w, dx.ptr, 0, rows, y.ptr, out) else try k.decode(s, w, dx.ptr, 0, rows, y.ptr, out);
                try s.synchronize();
                try compare(gpu, try std.fmt.bufPrint(&name, "{t} {s} [{d}, {d}] {s} {d} rows {t}", .{ kind, case, n, kk, if (prompt) "prompt" else "decode", rows, out }), y, ref, out, tolerance(out));
            };
        }
    }
}

/// Decode GB/s: 9 IQ2_XXS experts' up and down at width 2560 (fresh experts each pass, past L2), and lm_head rows.
pub fn bench(gpu: Gpu) !void {
    var k = try ggml.Kernels.load(gpu.d);
    defer k.unload();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();

    // experts: 16 sets of 9 (144 experts, 194 MB), a pass on each set in turn
    const width = 2560;
    const dims = 1024;
    const sets = 16;
    const e = 9 * sets;
    const one = ggml.bytes(.iq2xxs, width, dims);
    var up = try cuda.DeviceBuffer.alloc(gpu.d, e * one);
    defer up.free();
    var down = try cuda.DeviceBuffer.alloc(gpu.d, e * one);
    defer down.free();
    try up.fill8(0x11, s.handle);
    try down.fill8(0x11, s.handle);
    var x = try cuda.DeviceBuffer.alloc(gpu.d, 512 * 4096 * 2);
    defer x.free();
    try x.fill8(0x3c, s.handle); // bf16 ~0.01
    var h = try cuda.DeviceBuffer.alloc(gpu.d, 9 * width * 2);
    defer h.free();
    try h.fill8(0x3c, s.handle);
    var y = try cuda.DeviceBuffer.alloc(gpu.d, 512 * 131072 * 2);
    defer y.free();
    var plans = try cuda.DeviceBuffer.alloc(gpu.d, sets * 256);
    defer plans.free();
    var picks: [sets * 9]i32 = undefined;
    for (&picks, 0..) |*p, i| p.* = @intCast(i);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&picks));
    defer dpicks.free();
    var ps: [sets]grouped.Plan = undefined;
    for (&ps, 0..) |*p, i| {
        const base = plans.ptr + i * 256;
        p.* = .{ .members = base, .items = base + 64, .counts = base + 192, .rank = 0, .hist = 0 };
        try router.route(s, dpicks.ptr + i * 9 * 4, 9, e, 16, p.*);
    }
    const l: ex_.Layer = .{ .format = .iq2xxs, .up = up.ptr, .down = down.ptr, .width = width, .dims = dims, .experts = e, .relu2 = true };
    const g: ex_.Grouped = .{ .iq2xxs = &k };
    const reps = 20;
    const bound = grouped.maxItems(9, e, 16);
    const parts = [_][]const u8{ "up", "down", "up+down" };
    for (parts, 0..) |what, pi| {
        for (0..2) |warm| {
            try t0.record(s);
            for (0..reps) |_| for (ps) |p| {
                if (pi != 1) try g.decode(s, .up, .{ .x = x.ptr, .stride = dims, .slots = 9 }, l, p, h.ptr, bound, -1);
                if (pi != 0) try g.decode(s, .down_f32, .{ .x = h.ptr, .stride = width }, l, p, y.ptr, bound, -1);
            };
            try t1.record(s);
            try t1.synchronize();
            if (warm == 0) continue;
            const us = try cuda.Event.elapsedMs(t0, t1) * 1e3 / (reps * sets);
            const moved: f64 = @floatFromInt(9 * one * @as(usize, if (pi == 2) 2 else 1));
            std.debug.print("RESULT iq2xxs experts decode, 1 token x 9 experts {s} at width {d}: {d:.2} us, {d:.0} GB/s\n", .{ what, width, us, moved / us / 1e3 });
        }
    }

    // prompt rows: 2048 tokens x 9 slots over the 144 experts, up then down, as the prompt plan groups them
    {
        const tokens = 2048;
        const pairs = tokens * 9;
        const many = try gpu.gpa.alloc(i32, pairs);
        defer gpu.gpa.free(many);
        var prng = std.Random.DefaultPrng.init(7);
        for (0..tokens) |tk| {
            const first = prng.random().uintLessThan(usize, e - 9);
            for (0..9) |j| many[tk * 9 + j] = @intCast(first + j);
        }
        var dmany = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(many));
        defer dmany.free();
        const pb = grouped.maxItems(pairs, e, 64);
        const hist = (pairs + 1023) / 1024 * e * 4;
        var scratch = try cuda.DeviceBuffer.alloc(gpu.d, pairs * 8 + pb * 12 + 16 + hist);
        defer scratch.free();
        const base = scratch.ptr;
        const pp: grouped.Plan = .{ .members = base, .items = base + pairs * 4, .counts = base + pairs * 4 + pb * 12, .rank = base + pairs * 4 + pb * 12 + 16, .hist = base + pairs * 8 + pb * 12 + 16 };
        try router.route(s, dmany.ptr, pairs, e, 64, pp);
        var xp = try cuda.DeviceBuffer.alloc(gpu.d, tokens * dims * 2);
        defer xp.free();
        try xp.fill8(0x3c, s.handle);
        var hp = try cuda.DeviceBuffer.alloc(gpu.d, pairs * width * 2);
        defer hp.free();
        var yp = try cuda.DeviceBuffer.alloc(gpu.d, pairs * dims * 2);
        defer yp.free();
        var t2 = try cuda.Event.init(gpu.d, true);
        defer t2.deinit();
        for (0..2) |warm| {
            try t0.record(s);
            try g.prompt(s, .up, .{ .x = xp.ptr, .stride = dims, .slots = 9 }, l, pp, hp.ptr, pb, -1);
            try t1.record(s);
            try g.prompt(s, .down_bf16, .{ .x = hp.ptr, .stride = width }, l, pp, yp.ptr, pb, -1);
            try t2.record(s);
            try t2.synchronize();
            if (warm == 0) continue;
            const up_us = try cuda.Event.elapsedMs(t0, t1) * 1e3;
            const down_us = try cuda.Event.elapsedMs(t1, t2) * 1e3;
            const flops: f64 = 2.0 * pairs * width * dims;
            std.debug.print("RESULT iq2xxs experts prompt, {d} tokens x 9 over {d} experts at width {d}: up {d:.0} us ({d:.0} TFLOP/s), down {d:.0} us ({d:.0} TFLOP/s)\n", .{ tokens, e, width, up_us, flops / up_us / 1e6, down_us, flops / down_us / 1e6 });
        }
    }

    // lm_head: Q8_0 [131072, 4096]
    const n = 131072;
    const kk = 4096;
    const lb = ggml.bytes(.q8_0, n, kk);
    var lm = try cuda.DeviceBuffer.alloc(gpu.d, lb);
    defer lm.free();
    try lm.fill8(0x11, s.handle);
    const w = ggml.q8Weight(lm.ptr, n, kk);
    for ([_]usize{ 1, 4, 16, 64, 512 }) |rows| for ([_]ggml.Out{ .f32, .bf16 }) |out| {
        if (rows > 16 and out == .f32) continue;
        const r2: usize = if (rows > 16) 3 else 10;
        for (0..2) |warm| {
            try t0.record(s);
            for (0..r2) |_| try k.matmul(s, w, x.ptr, 0, rows, y.ptr, out);
            try t1.record(s);
            try t1.synchronize();
            if (warm == 0) continue;
            const us = try cuda.Event.elapsedMs(t0, t1) * 1e3 / @as(f32, @floatFromInt(r2));
            std.debug.print("RESULT q8_0 lm_head [{d}, {d}] {d} rows {t}: {d:.1} us, {d:.0} GB/s\n", .{ n, kk, rows, out, us, @as(f64, @floatFromInt(lb)) / us / 1e3 });
        }
    };
}
