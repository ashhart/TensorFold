//! BF16 K32 prompt staging against the parent 128x128 K64 tile, including forced tails and GPU event timing.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const q = cuda.qmmf;
const rows = [_]usize{ 1, 16, 64, 65, 127, 128, 129, 511, 512, 1536, 4218 };
pub const Projection = struct { name: []const u8, n: usize, k: usize, out: q.Out, calls: usize };
/// BF16 projections in the serving snapshot fc694b54: one normal main-model chunk on TP1 or rank 0.
pub const normal = [_]Projection{
    .{ .name = "GDN input", .n = 16480, .k = 2560, .out = .bf16, .calls = 36 },
    .{ .name = "attention input", .n = 13952, .k = 2560, .out = .bf16, .calls = 12 },
    .{ .name = "mixer output", .n = 2560, .k = 6144, .out = .f32, .calls = 48 },
    .{ .name = "shared gate/up", .n = 640, .k = 2560, .out = .bf16, .calls = 96 },
    .{ .name = "shared down", .n = 2560, .k = 640, .out = .f32, .calls = 48 },
    .{ .name = "HC up", .n = 10240, .k = 320, .out = .bf16, .calls = 96 },
    .{ .name = "HC down/inject", .n = 324, .k = 10240, .out = .f32, .calls = 96 },
    .{ .name = "PLE key", .n = 10240, .k = 2560, .out = .bf16, .calls = 1 },
    .{ .name = "PLE value", .n = 2560, .k = 2560, .out = .f32, .calls = 1 },
};
/// .all logits uses these additional shapes; HC up is already in normal, and these do not enter its total.
pub const all_logits = [_]Projection{
    .{ .name = "final HC down", .n = 320, .k = 10240, .out = .f32, .calls = 1 },
    .{ .name = "full vocabulary", .n = 248320, .k = 2560, .out = .bf16, .calls = 1 },
    .{ .name = "sharded vocabulary", .n = 124160, .k = 2560, .out = .bf16, .calls = 1 },
};
pub const mtp_hidden: Projection = .{ .name = "MTP hidden", .n = 2560, .k = 2560, .out = .f32, .calls = 1 };
const guard = 128;

fn fixturePartBytes(m: usize, n: usize) usize {
    const scratch_rows: usize = @min(m, q.fused_rows - 1);
    return 8 * scratch_rows * n * @sizeOf(f32);
}

const Fixture = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    lane: *q.Lane,
    w: q.Weight,
    x: cuda.DeviceBuffer,
    codes: cuda.DeviceBuffer,
    old: cuda.DeviceBuffer,
    new: cuda.DeviceBuffer,
    part: cuda.DeviceBuffer,
    ldx: usize,

    fn init(gpu: check.Gpu, s: cuda.Stream, lane: *q.Lane, n: usize, k: usize, m: usize) !Fixture {
        return initSized(gpu, s, lane, n, k, m, fixturePartBytes(m, n));
    }

    // Bounded all-logit probes run at M>=512, so neither baseline nor candidate needs a split partial plane.
    fn initBounded(gpu: check.Gpu, s: cuda.Stream, lane: *q.Lane, n: usize, k: usize) !Fixture {
        return initSized(gpu, s, lane, n, k, 1536, 16);
    }

    fn initSized(gpu: check.Gpu, s: cuda.Stream, lane: *q.Lane, n: usize, k: usize, m: usize, part_bytes: usize) !Fixture {
        const a = gpu.gpa;
        const npad = cuda.modelopt.padded(n);
        const ldx = k + 64;
        var prng = std.Random.DefaultPrng.init(0x4218_bf16 + n);
        const rng = prng.random();
        const host_w = try a.alloc(u16, n * k);
        defer a.free(host_w);
        const tiled = try a.alloc(u16, npad * k);
        defer a.free(tiled);
        const host_x = try a.alloc(u16, m * ldx);
        defer a.free(host_x);
        for (host_w) |*v| v.* = finite(rng);
        for (host_x) |*v| v.* = finite(rng);
        q.packBf16(tiled, host_w, n, k);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_x));
        errdefer x.free();
        var codes = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(tiled));
        errdefer codes.free();
        var old = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 4 + 2 * guard);
        errdefer old.free();
        var new = try cuda.DeviceBuffer.alloc(gpu.d, old.len);
        errdefer new.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, part_bytes);
        errdefer part.free();
        return .{ .gpu = gpu, .s = s, .lane = lane, .w = .{ .mode = .bf16, .codes = codes.ptr, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad), .scale = 0.8123 }, .x = x, .codes = codes, .old = old, .new = new, .part = part, .ldx = ldx };
    }

    fn deinit(f: *Fixture) void {
        f.part.free();
        f.new.free();
        f.old.free();
        f.codes.free();
        f.x.free();
    }

    fn baseline(f: *Fixture, m: usize, out: q.Out) !void {
        try f.product(m, out, false, f.old.ptr + guard);
    }

    fn candidate(f: *Fixture, m: usize, out: q.Out) !void {
        try f.product(m, out, true, f.new.ptr + guard);
    }

    // Force the parent's K64 or new unsplit K32 tile even below the production threshold.
    fn product(f: *Fixture, m: usize, out: q.Out, candidate_: bool, dest: u64) !void {
        const sk = f.w.sk;
        const k32 = candidate_ and sk == 1;
        const fun = if (k32) f.lane.prompt_k32.?[@backingInt(out)] else f.lane.prompt.?[@backingInt(out)][@intFromBool(sk > 1)];
        const band = @max(1, @min((m + 127) / 128, (12 << 20) / (128 * f.w.k * 2)));
        var args: cuda.Args = .{};
        args.add(f.x.ptr);
        args.add(f.w.codes);
        args.add(f.w.scales);
        args.add(f.w.scale);
        args.add(dest);
        args.add(@as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(f.w.n)));
        args.add(@as(i32, @intCast(f.w.k)));
        args.add(@as(i32, @intCast(sk)));
        args.add(@as(i32, @intCast(f.w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) f.w.k else f.ldx)));
        args.add(@as(i32, @intCast(band)));
        try cuda.launch.launch(fun, .{
            .grid = .{ .x = @intCast(((m + 127) / 128) * ((f.w.n + 127) / 128)), .y = 1, .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
            .shared = if (k32) q.prompt_k32_shared else q.prompt_shared,
        }, f.s, &args);
    }

    fn compare(f: *Fixture, m: usize, out: q.Out, repeats: usize) !void {
        const bytes = m * f.w.n * @as(usize, if (out == .f32) 4 else 2);
        try check.expect(f.x.len >= ((m - 1) * f.ldx + f.w.k) * 2, "prompt fixture input span", .{});
        try check.expect(f.codes.len >= @as(usize, f.w.npad) * f.w.k * 2, "prompt fixture weight span", .{});
        try check.expect(f.old.len >= bytes + 2 * guard and f.new.len >= bytes + 2 * guard, "prompt fixture output spans", .{});
        try check.expect(f.part.len >= f.lane.partBytes(m, f.w), "prompt fixture split scratch: have {d}, need {d}", .{ f.part.len, f.lane.partBytes(m, f.w) });
        const a = f.gpu.gpa;
        const want = try a.alloc(u8, bytes);
        defer a.free(want);
        const got = try a.alloc(u8, bytes);
        defer a.free(got);
        try f.old.fill8(0xa5, f.s.handle);
        try f.part.fill8(0xff, f.s.handle);
        try f.baseline(m, out);
        try f.s.synchronize();
        try f.old.download(guard, want);
        for (0..repeats) |_| {
            try f.new.fill8(0x5a, f.s.handle);
            try f.candidate(m, out);
            try f.s.synchronize();
            try f.new.download(guard, got);
            try check.sameBytes("K32 versus parent K64 prompt tile", got, want);
            try f.guards(bytes);
        }
        try f.new.fill8(0x5a, f.s.handle);
        try f.part.fill8(0xff, f.s.handle);
        try f.lane.matmulOut(f.s, f.x.ptr, f.ldx, m, f.w, f.new.ptr + guard, out, f.part.ptr);
        try f.s.synchronize();
        try f.new.download(guard, got);
        try check.sameBytes("production prompt versus parent K64 tile", got, want);
        try f.guards(bytes);
    }

    fn guards(f: *Fixture, bytes: usize) !void {
        var buf: [guard]u8 = undefined;
        for ([_]usize{ 0, guard + bytes }) |at| {
            try f.new.download(at, &buf);
            for (buf) |v| try check.expect(v == 0x5a, "prompt tile output guard", .{});
        }
    }

    fn timed(f: *Fixture, m: usize, prompt: bool, out: q.Out) !void {
        try f.product(m, out, prompt, f.new.ptr + guard);
    }

    fn time(f: *Fixture, m: usize, prompt: bool, out: q.Out) !f64 {
        var t0 = try cuda.Event.init(f.gpu.d, true);
        defer t0.deinit();
        var t1 = try cuda.Event.init(f.gpu.d, true);
        defer t1.deinit();
        for (0..3) |_| {
            try f.timed(m, prompt, out);
        }
        try t0.record(f.s);
        for (0..20) |_| {
            try f.timed(m, prompt, out);
        }
        try t1.record(f.s);
        try t1.synchronize();
        return @as(f64, try cuda.Event.elapsedMs(t0, t1)) / 20;
    }
};

fn finite(rng: std.Random) u16 {
    const bits = rng.int(u16);
    return (bits & 0x807f) | (@as(u16, 118 + rng.uintLessThan(u16, 16)) << 7);
}

fn load(gpu: check.Gpu) !q.Lane {
    const cap = try gpu.ctx.capability();
    try check.expect(cap / 10 == 12, "qmmf prompt tiles require sm_12x, got sm_{d}", .{cap});
    return q.Lane.load(gpu.d, cap / 10);
}

pub fn run(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try load(gpu);
    defer lane.unload();
    for (normal) |shape| {
        var f = try Fixture.init(gpu, s, &lane, shape.n, shape.k, 4218);
        defer f.deinit();
        for ([_]u32{ 1, q.splitK(shape.n, shape.k) }, 0..) |sk, si| {
            if (si == 1 and sk == 1) continue;
            f.w = try q.pinned(f.w, sk);
            for ([_]bool{ false, true }) |clusters| {
                lane.clusters = clusters;
                for ([_]q.Out{ .bf16, .f32 }) |out| for (rows) |m| {
                    try f.compare(m, out, if (m == 4218) 5 else 1);
                    check.pass("qmmf-prompt K32/K64 M={d} N={d} K={d} SK={d} out={t} clusters={}", .{ m, f.w.n, f.w.k, sk, out, clusters });
                };
            }
        }
    }
    for ([_]usize{ 512, 1024 }) |m| {
        var f = try Fixture.init(gpu, s, &lane, mtp_hidden.n, mtp_hidden.k, m);
        defer f.deinit();
        f.w = try q.pinned(f.w, q.splitK(f.w.n, f.w.k));
        try f.compare(m, mtp_hidden.out, 5);
        check.pass("qmmf-prompt MTP hidden M={d} N=2560 K=2560 SK=4 fp32", .{m});
    }
    for (all_logits) |shape| {
        var f = try Fixture.initBounded(gpu, s, &lane, shape.n, shape.k);
        defer f.deinit();
        f.w = try q.pinned(f.w, q.splitK(shape.n, shape.k));
        for ([_]usize{ 512, 1536 }) |m| {
            try f.compare(m, shape.out, 1);
            check.pass("qmmf-prompt all-logit {s} M={d} N={d} K={d} SK={d} out={t}", .{ shape.name, m, shape.n, shape.k, f.w.sk, shape.out });
        }
    }
    try special(gpu, s, &lane);
}

/// Reproduce the first reported fault without running the preceding matrix of cases.
pub fn repro(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try load(gpu);
    defer lane.unload();
    lane.clusters = false;
    var f = try Fixture.init(gpu, s, &lane, 8240, 2560, 4218);
    defer f.deinit();
    f.w = try q.pinned(f.w, 4);
    try f.compare(65, .bf16, 1);
}

fn special(gpu: check.Gpu, s: cuda.Stream, lane: *q.Lane) !void {
    var f = try Fixture.init(gpu, s, lane, 129, 2048, 129);
    defer f.deinit();
    const plants = [_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80, 0x0001, 0x8001, 0, 0x8000 };
    for (plants, 0..) |value, row| try f.x.upload(row * f.ldx * 2, std.mem.asBytes(&value));
    const infinities = [_]u16{ 0x7f80, 0xff80 };
    try f.x.upload(8 * f.ldx * 2, std.mem.sliceAsBytes(&infinities));
    const x_before = try check.download(gpu, f.x);
    defer gpu.gpa.free(x_before);
    const w_before = try check.download(gpu, f.codes);
    defer gpu.gpa.free(w_before);
    for ([_]u32{ 1, 4 }) |sk| {
        f.w = try q.pinned(f.w, sk);
        for ([_]q.Out{ .bf16, .f32 }) |out| try f.compare(129, out, 5);
    }
    const x_after = try check.download(gpu, f.x);
    defer gpu.gpa.free(x_after);
    const w_after = try check.download(gpu, f.codes);
    defer gpu.gpa.free(w_after);
    try check.sameBytes("prompt inputs unchanged", x_after, x_before);
    try check.sameBytes("prompt weights unchanged", w_after, w_before);
    check.pass("qmmf-prompt N=129: nonfinite, subnormal and signed-zero input bytes, immutable inputs", .{});
}

fn timings(f: *Fixture, m: usize, out: q.Out) ![2]f64 {
    var samples: [3][3]f64 = undefined;
    for (&samples) |*sample| {
        sample[0] = try f.time(m, false, out);
        sample[1] = try f.time(m, true, out);
        sample[2] = try f.time(m, false, out);
    }
    var old: [3]f64 = undefined;
    var new: [3]f64 = undefined;
    for (samples, 0..) |sample, j| {
        old[j] = (sample[0] + sample[2]) / 2;
        new[j] = sample[1];
    }
    return .{ check.median(&old), check.median(&new) };
}

pub fn bench(gpu: check.Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var lane = try load(gpu);
    defer lane.unload();
    var totals = [2]f64{ 0, 0 };
    for (normal) |shape| {
        var f = try Fixture.init(gpu, s, &lane, shape.n, shape.k, 4218);
        defer f.deinit();
        f.w.scale = 1;
        f.w = try q.pinned(f.w, q.splitK(shape.n, shape.k));
        const out = shape.out;
        try f.compare(4218, out, 1);
        const k32 = f.w.sk == 1;
        const fun = if (k32) lane.prompt_k32.?[@backingInt(out)] else lane.prompt.?[@backingInt(out)][1];
        std.debug.print("LIMITS qmmf-prompt candidate regs={d} static_shared={d} blocks_per_sm={d}\n", .{ try fun.attribute(.num_regs), try fun.attribute(.shared_size_bytes), try fun.occupancy(256, if (k32) q.prompt_k32_shared else q.prompt_shared) });
        const sample = try timings(&f, 4218, out);
        const calls: f64 = @floatFromInt(shape.calls);
        totals[0] += sample[0] * calls;
        totals[1] += sample[1] * calls;
        std.debug.print("BENCH qmmf-prompt rank0 normal-main {s} M=4218 N={d} K={d} SK={d} out={t} calls={d} parent_k64={d:.4} ms candidate={d:.4} ms speedup={d:.3}x\n", .{ shape.name, f.w.n, f.w.k, f.w.sk, out, shape.calls, sample[0], sample[1], sample[0] / sample[1] });
    }
    std.debug.print("BENCH qmmf-prompt rank0 normal-main weighted 36/12/48/96/48/96/96/1/1: parent_k64={d:.3} ms candidate={d:.3} ms speedup={d:.3}x; excludes MTP and all-logit head\n", .{ totals[0], totals[1], totals[0] / totals[1] });
    for ([_]usize{ 512, 1024 }) |m| {
        var f = try Fixture.init(gpu, s, &lane, mtp_hidden.n, mtp_hidden.k, m);
        defer f.deinit();
        f.w.scale = 1;
        f.w = try q.pinned(f.w, q.splitK(f.w.n, f.w.k));
        try f.compare(m, mtp_hidden.out, 1);
        const sample = try timings(&f, m, mtp_hidden.out);
        std.debug.print("BENCH qmmf-prompt MTP hidden M={d} N=2560 K=2560 SK=4 out=f32 parent_k64={d:.4} ms candidate={d:.4} ms speedup={d:.3}x; separate from normal-main\n", .{ m, sample[0], sample[1], sample[0] / sample[1] });
    }
}

test "qmmf prompt scratch fixture uses usize before multiplication and covers every split row count" {
    try std.testing.expectEqual(@as(usize, 17_139_200), fixturePartBytes(65, 8240));
    try std.testing.expectEqual(@as(usize, 67_238_400), fixturePartBytes(4218, 8240));
    var lane: q.Lane = undefined;
    lane.clusters = false;
    for (normal) |shape| {
        const n = shape.n;
        const w = try q.pinned(cuda.modelopt.bf16Weight(0, n, shape.k), q.splitK(n, shape.k));
        for (1..16385) |m| {
            try std.testing.expect(fixturePartBytes(m, n) >= lane.partBytes(m, w));
            try std.testing.expect(fixturePartBytes(4218, n) >= lane.partBytes(m, w));
        }
    }
}

test "qmmf prompt scratch keeps the current normal weights and bounds all-logit probes" {
    var calls: usize = 0;
    for (normal) |projection| calls += projection.calls;
    try std.testing.expectEqual(@as(usize, 434), calls);
    const final = cuda.modelopt.bf16Weight(0, 320, 10240);
    try std.testing.expectEqual(@as(u32, 384), final.npad);
    try std.testing.expect(q.promptEligible(512, final));
    var lane: q.Lane = undefined;
    lane.clusters = false;
    for (all_logits) |projection| {
        const w = cuda.modelopt.bf16Weight(0, projection.n, projection.k);
        if (projection.n >= 124160) try std.testing.expectEqual(q.Out.bf16, projection.out);
        for ([_]usize{ 512, 1536 }) |m| {
            try std.testing.expect(q.promptEligible(m, w));
            try std.testing.expectEqual(@as(usize, 0), lane.partBytes(m, w));
        }
    }
    const hidden = cuda.modelopt.bf16Weight(0, mtp_hidden.n, mtp_hidden.k);
    try std.testing.expectEqual(q.Out.f32, mtp_hidden.out);
    try std.testing.expectEqual(@as(u32, 4), q.splitK(hidden.n, hidden.k));
    try std.testing.expect(!q.promptEligible(4 * 127, hidden));
    try std.testing.expect(q.promptEligible(4 * 128, hidden));
    try std.testing.expect(q.promptEligible(4 * 256, hidden));
}

test "qmmf prompt scratch timed fused control shares one kernel, arguments and output" {
    const Trace = struct {
        var calls: usize = 0;
        var ids: [2]usize = undefined;
        var args: [2][13]u64 = undefined;
        var geometry: [2][7]u32 = undefined;
        fn arg(comptime T: type, raw: ?[*]?*anyopaque, i: usize) T {
            return @as(*const T, @ptrCast(@alignCast(raw.?[i].?))).*;
        }
        fn launch(f: cuda.abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, sm: c_uint, _: cuda.abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const at = calls;
            calls += 1;
            ids[at] = @intFromPtr(f);
            geometry[at] = .{ gx, gy, gz, bx, by, bz, sm };
            for (0..3) |i| args[at][i] = arg(u64, raw, i);
            args[at][3] = @as(u32, @bitCast(arg(f32, raw, 3)));
            for (4..6) |i| args[at][i] = arg(u64, raw, i);
            for (6..13) |i| args[at][i] = @intCast(arg(i32, raw, i));
            return cuda.abi.success;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuLaunchKernel = Trace.launch;
    const old: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(10) };
    const prompt: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(20) };
    var lane: q.Lane = .{
        .mod = undefined,
        .fns = @splat(@splat(old)),
        .f32_fns = @splat(@splat(old)),
        .reduce = @splat(old),
        .half = @splat(@splat(@splat(old))),
        .half_below = 0,
        .prompt = @splat(@splat(prompt)),
        .clusters = false,
    };
    var f: Fixture = .{
        .gpu = undefined,
        .s = .{ .d = &driver, .handle = null },
        .lane = &lane,
        .w = try q.pinned(cuda.modelopt.bf16Weight(0x2000, 2560, 2560), 4),
        .x = .{ .d = &driver, .ptr = 0x1000, .len = 1 },
        .codes = .{ .d = &driver, .ptr = 0x2000, .len = 1 },
        .old = .{ .d = &driver, .ptr = 0x3000, .len = 1 },
        .new = .{ .d = &driver, .ptr = 0x4000, .len = 1 },
        .part = .{ .d = &driver, .ptr = 0x5000, .len = 1 },
        .ldx = 2560,
    };
    for ([_]usize{ 512, 1024 }) |m| for ([_]q.Out{ .bf16, .f32 }) |out| {
        Trace.calls = 0;
        try f.timed(m, false, out);
        try f.timed(m, true, out);
        try std.testing.expectEqual(@as(usize, 2), Trace.calls);
        try std.testing.expectEqualSlices(usize, &.{ 20, 20 }, &Trace.ids);
        try std.testing.expectEqualDeep(Trace.args[0], Trace.args[1]);
        try std.testing.expectEqualDeep(Trace.geometry[0], Trace.geometry[1]);
        for (Trace.args) |a| try std.testing.expectEqual(f.new.ptr + guard, a[4]);
        Trace.calls = 0;
        try f.baseline(m, out);
        try f.candidate(m, out);
        try std.testing.expectEqual(f.old.ptr + guard, Trace.args[0][4]);
        try std.testing.expectEqual(f.new.ptr + guard, Trace.args[1][4]);
        try std.testing.expect(Trace.args[0][4] != Trace.args[1][4]);
    };
}

test "qmmf prompt scratch production envelope covers TP1 and TP2 through the chunk cap" {
    const dims: usize = 2560;
    const chunk_cap: usize = 16384;
    const weights = [_][3]u32{
        .{ 16480, 2560, 1 },  .{ 13952, 2560, 1 }, .{ 8240, 2560, 1 },  .{ 7296, 2560, 1 },
        .{ 2560, 6144, 0 },   .{ 2560, 3072, 0 },  .{ 10240, 2560, 0 }, .{ 2560, 2560, 0 },
        .{ 324, 10240, 0 },   .{ 320, 10240, 0 },  .{ 10240, 320, 0 },  .{ 640, 2560, 0 },
        .{ 320, 2560, 0 },    .{ 2560, 640, 0 },   .{ 2560, 320, 0 },   .{ 248320, 2560, 0 },
        .{ 124160, 2560, 0 }, .{ 79591, 2560, 0 },
    };
    var lane: q.Lane = undefined;
    lane.clusters = false;
    for (weights) |shape| {
        const w: q.Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = shape[0], .k = shape[1], .npad = (shape[0] + 127) / 128 * 128, .sk = shape[2] };
        for (1..chunk_cap + 1) |m| {
            const native_bytes: usize = 8 * m * dims * 4;
            try std.testing.expect(lane.partBytes(m, w) <= native_bytes);
            if (q.promptEligible(m, w)) try std.testing.expectEqual(@as(usize, 0), lane.partBytes(m, w));
            if (w.n == dims) {
                const p = q.planFor(m, w, false);
                try std.testing.expect(@as(usize, p.sk) * m * w.n * 4 <= native_bytes);
            }
        }
    }
    // MTP reserves at least two rows per token before fc_hidden projects all four streams.
    const fc: q.Weight = .{ .mode = .bf16, .codes = 0, .scales = 0, .n = 2560, .k = 2560, .npad = 2560 };
    for (1..257) |m| {
        const cap: usize = @max(32, 2 * m);
        const native_bytes: usize = 8 * cap * dims * 4;
        try std.testing.expect(lane.partBytes(4 * m, fc) <= native_bytes);
        try std.testing.expect(4 * (4 * m) * dims * 4 <= native_bytes);
    }
}
