//! Small synthetic device checks, independent of models, captures, Python and the pair.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core_lane");
const frozen = @import("fixtures/fn_lane/golden.zig");
const a = std.heap.page_allocator;
const guard = 16;
const rows_max = 16;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn f32of(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

const Data = struct {
    x: []u16,
    codes: []u8,
    words: []u32,
    scales: []u16,
    biases: []u16,

    fn init(l: core.Layout) !Data {
        const d = Data{ .x = try a.alloc(u16, rows_max * l.k), .codes = try a.alloc(u8, l.n * l.k), .words = try a.alloc(u32, l.weightWords()), .scales = try a.alloc(u16, l.n * l.k / l.format.group), .biases = try a.alloc(u16, l.n * l.k / l.format.group) };
        @memset(d.words, 0);
        var prng: std.Random.DefaultPrng = .init(0x1a9e + @as(u64, l.format.bits) * 1000 + l.format.group);
        const rnd = prng.random();
        for (d.x, 0..) |*v, i| v.* = if (i % 29 == 0) 0 else bf16(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -500, 500))) * 0.0317);
        const mask: u16 = (@as(u16, 1) << @intCast(l.format.bits)) - 1;
        for (d.codes, 0..) |*v, i| {
            v.* = if (i % 31 == 0) @intCast(mask) else if (i % 17 == 0) 0 else @intCast(rnd.intRangeLessThan(u16, 0, mask + 1));
            // Independent bit-by-bit MLX packing; the CPU oracle reads the original codes directly.
            for (0..l.format.bits) |bit| if (v.* & (@as(u8, 1) << @intCast(bit)) != 0) {
                const at = i * l.format.bits + bit;
                d.words[at / 32] |= @as(u32, 1) << @intCast(at % 32);
            };
        }
        for (d.scales) |*v| v.* = bf16(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -20, 21))) * 0.0078125);
        for (d.biases) |*v| v.* = bf16(@as(f32, @floatFromInt(rnd.intRangeLessThan(i32, -40, 41))) * 0.0053);
        return d;
    }
    fn deinit(d: Data) void {
        a.free(d.x);
        a.free(d.codes);
        a.free(d.words);
        a.free(d.scales);
        a.free(d.biases);
    }
};

fn buffer(device: mtl.Device, bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes + 2 * guard, opts);
    @memset(b.contents()[0..b.length()], 0xa5);
    return b;
}
fn ref(b: mtl.Buffer, off: usize) core.Ref {
    return .{ .buf = b, .off = guard + off };
}
fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |message| {
        std.debug.print("GPU command failed: {s}\n", .{message});
        return error.GpuFailure;
    }
}
fn checkGuard(b: mtl.Buffer) !void {
    for (b.contents()[0..guard]) |v| if (v != 0xa5) return error.PrefixOverwrite;
    for (b.contents()[b.length() - guard .. b.length()]) |v| if (v != 0xa5) return error.SuffixOverwrite;
}
fn selected(l: core.Layout, n: usize) bool {
    if (l.ranges.len == 0) return true;
    for (l.ranges) |r| if (n / 32 >= r[0] and n / 32 < r[0] + r[1]) return true;
    return false;
}

const Oracle = struct { value: f64, bound: f64 };
fn oracle(l: core.Layout, d: Data, row: usize, n: usize) Oracle {
    const cut = l.groupCut();
    const ng = l.k / l.format.group;
    var sum: f64 = 0;
    var magnitude: f64 = 0;
    for (cut[0]..cut[0] + cut[1]) |g| {
        const s: f64 = f32of(d.scales[n * ng + g]);
        const b: f64 = f32of(d.biases[n * ng + g]);
        for (g * l.format.group..(g + 1) * l.format.group) |k| {
            const x: f64 = f32of(d.x[row * l.k + k]);
            const q: f64 = @floatFromInt(d.codes[n * l.k + k]);
            sum += x * (s * q + b);
            magnitude += @abs(x * s * q) + @abs(x * b);
        }
    }
    // Roundoff bound per dot/xsum, two group FMAs and ordered split-K adds; inputs and codes are exact in f64.
    const operations: f64 = @floatFromInt(l.format.group + 2 * cut[1] + l.sk);
    const u = 0x1p-24;
    const gamma = operations * u / (1 - operations * u);
    return .{ .value = sum, .bound = gamma * magnitude + 0x1p-120 };
}

const Counts = struct { layouts: usize = 0, row_pairs: usize = 0, golden_windows: usize = 0, cpu_values: usize = 0, max_error: f64 = 0, max_fraction: f64 = 0 };

fn check(device: mtl.Device, queue: mtl.Queue, l: core.Layout, counts: *Counts) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const p = try core.Projection.init(a, device, l);
    defer p.deinit();
    if (counts.layouts == 0) for ([_]u32{ 0, 17 }) |bad_rows| {
        // These errors must precede any encoder or buffer access.
        p.encode(undefined, undefined, undefined, undefined, undefined, bad_rows) catch |err| {
            if (err != error.InvalidRows) return err;
            continue;
        };
        return error.AcceptedInvalidRowCount;
    };
    const d = try Data.init(l);
    defer d.deinit();
    const x = try buffer(device, d.x.len * 2);
    defer x.deinit();
    const w = try buffer(device, l.weightWords() * 4);
    defer w.deinit();
    const sb = try buffer(device, l.metadataElements() * 2);
    defer sb.deinit();
    const elem: usize = if (l.output == .f32) 4 else 2;
    const one = try buffer(device, rows_max * l.n * elem);
    defer one.deinit();
    const window_bytes = rows_max * l.n * elem;
    const windows = try buffer(device, rows_max * window_bytes);
    defer windows.deinit();
    @memcpy(x.contents()[guard..][0 .. d.x.len * 2], std.mem.sliceAsBytes(d.x));
    const packed_w = @as([*]u32, @ptrCast(@alignCast(w.contents() + guard)))[0..l.weightWords()];
    const packed_sb = @as([*]u16, @ptrCast(@alignCast(sb.contents() + guard)))[0..l.metadataElements()];
    try core.pack(l, d.words, d.scales, d.biases, packed_w, packed_sb);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    for (0..rows_max) |r| try p.encode(enc, ref(x, r * l.k * 2), ref(w, 0), ref(sb, 0), ref(one, r * l.n * elem), 1);
    for (1..rows_max + 1) |m| try p.encode(enc, ref(x, 0), ref(w, 0), ref(sb, 0), ref(windows, (m - 1) * window_bytes), @intCast(m));
    enc.end();
    try finish(cb);
    for (1..rows_max + 1) |m| {
        const out = windows.contents()[guard + (m - 1) * window_bytes ..][0..window_bytes];
        for (0..m) |r| {
            const expected = one.contents()[guard + r * l.n * elem ..][0 .. l.n * elem];
            if (!std.mem.eql(u8, expected, out[r * l.n * elem ..][0 .. l.n * elem])) {
                std.debug.print("row mismatch b{d}g{d} sk{d} pf{d} {s}, width{d} row{d}\n", .{ l.format.bits, l.format.group, l.sk, l.pf, @tagName(l.output), m, r });
                return error.RowInvariantMismatch;
            }
            counts.row_pairs += 1;
        }
        for (m * l.n * elem..window_bytes) |i| if (out[i] != 0xa5) return error.InactiveRowOverwrite;
        for (0..m) |r| for (0..l.n) |n| if (!selected(l, n)) {
            for (out[(r * l.n + n) * elem ..][0..elem]) |v| if (v != 0xa5) return error.UnselectedTileOverwrite;
        };
    }
    if (l.output == .f32) {
        const values = @as([*]const f32, @ptrCast(@alignCast(one.contents() + guard)))[0 .. rows_max * l.n];
        for (0..rows_max) |r| for (0..l.n) |n| if (selected(l, n)) {
            const want = oracle(l, d, r, n);
            const got = values[r * l.n + n];
            const err = @abs(@as(f64, got) - want.value);
            if (!std.math.isFinite(got) or err > want.bound) {
                std.debug.print("CPU mismatch b{d}g{d} sk{d} pf{d} r{d} n{d}: gpu{d} ref{d} error{d} bound{d}\n", .{ l.format.bits, l.format.group, l.sk, l.pf, r, n, got, want.value, err, want.bound });
                return error.CpuReferenceMismatch;
            }
            counts.max_error = @max(counts.max_error, err);
            counts.max_fraction = @max(counts.max_fraction, err / want.bound);
            counts.cpu_values += 1;
        };
    }
    if (l.format.bits == 6 and l.format.group == 32) {
        for (1..rows_max + 1) |m| {
            const count = m * l.n * elem;
            const got = windows.contents()[guard + (m - 1) * window_bytes ..][0..count];
            try frozen.compareWindow(l, m, got);
        }
        counts.golden_windows += 16;
    }
    for ([_]mtl.Buffer{ x, w, sb, one, windows }) |b| try checkGuard(b);
    counts.layouts += 1;
    std.debug.print("PASS b{d}g{d} sk{d} pf{d} {s} groups{any} tiles{d}: widths1-16, guards{s}\n", .{
        l.format.bits,                                                                          l.format.group, l.sk, l.pf, @tagName(l.output), l.groupCut(), l.tiles(),
        if (l.format.bits == 6 and l.format.group == 32) ", frozen fn_lane byte-equal" else "",
    });
}

fn packBench(io: std.Io, l: core.Layout) !void {
    try l.validate();
    const w = try a.alloc(u32, l.weightWords());
    defer a.free(w);
    const s = try a.alloc(u16, l.n * l.k / l.format.group);
    defer a.free(s);
    const b = try a.alloc(u16, s.len);
    defer a.free(b);
    const pw = try a.alloc(u32, w.len);
    defer a.free(pw);
    const sb = try a.alloc(u16, 2 * s.len);
    defer a.free(sb);
    for (w, 0..) |*v, i| v.* = @as(u32, @truncate(i)) *% 0x9e3779b9;
    @memset(s, 0x3b80);
    @memset(b, 0xbb80);
    try core.pack(l, w, s, b, pw, sb); // touch all source and destination pages before timing
    var ns: [5]i96 = undefined;
    for (&ns) |*dt| {
        const start = std.Io.Clock.awake.now(io);
        try core.pack(l, w, s, b, pw, sb);
        dt.* = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        std.mem.doNotOptimizeAway(pw.ptr);
        std.mem.doNotOptimizeAway(sb.ptr);
    }
    std.mem.sort(i96, &ns, {}, std.sort.asc(i96));
    const ms = @as(f64, @floatFromInt(ns[2])) / 1e6;
    std.debug.print("CPU pack only: synthetic b{d}g{d} [{d},{d}], output {d} bytes, read+write {d} bytes, median5 {d:.3} ms, {d:.3} GB/s, peak additional {d} bytes\n", .{
        l.format.bits, l.format.group, l.n, l.k, l.packedBytes(), 2 * l.packedBytes(), ms, @as(f64, @floatFromInt(2 * l.packedBytes())) / (ms * 1e6), l.packedBytes(),
    });
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len == 2 and std.mem.eql(u8, argv[1], "--pack-bench")) return packBench(init.io, .{ .n = 4096, .k = 24576, .format = .{ .bits = 4, .group = 64 } });
    if (argv.len == 6 and std.mem.eql(u8, argv[1], "--pack-bench")) return packBench(init.io, .{
        .format = .{ .bits = try std.fmt.parseInt(u8, argv[2], 10), .group = try std.fmt.parseInt(usize, argv[3], 10) },
        .n = try std.fmt.parseInt(usize, argv[4], 10),
        .k = try std.fmt.parseInt(usize, argv[5], 10),
    });
    if (argv.len != 1) return error.UnknownArgument;
    try frozen.verify();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) return error.TensorUnitsRequired;
    const queue = try device.queue();
    defer queue.deinit();
    std.debug.print("Small synthetic projection checks on {s}; no model files\n", .{device.name()});
    var counts: Counts = .{};
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        for ([_]usize{ 1, 2, 4 }) |sk| for ([_]usize{ 1, 2 }) |pf| {
            const l = core.Layout{ .n = 64, .k = 7 * group, .format = .{ .bits = bits, .group = group }, .sk = sk, .pf = pf, .output = .f32 };
            try check(device, queue, l, &counts);
            if ((sk == 1 and pf == 1) or (bits == 6 and group == 32)) {
                var bf = l;
                bf.output = .bf16;
                try check(device, queue, bf, &counts);
            }
        };
    };
    for ([_]usize{ 32, 64, 128 }) |group| for ([_]usize{ 1, 2, 4, 8 }) |sk| for ([_]usize{ 1, 2 }) |pf| {
        const layout = core.Layout{ .n = 96, .k = 9 * group, .format = .{ .bits = 4, .group = group }, .sk = sk, .pf = pf, .precompute_sums = true, .cooperative = true, .output = .f32 };
        layout.validate() catch |err| {
            if (err == error.ThreadgroupMemoryLimit) continue;
            return err;
        };
        try check(device, queue, layout, &counts);
        var bf = layout;
        bf.output = .bf16;
        try check(device, queue, bf, &counts);
    };
    for ([_]u8{ 4, 6, 8 }) |bits| {
        const group: usize = if (bits == 4) 64 else if (bits == 6) 32 else 128;
        try check(device, queue, .{ .n = 96, .k = 7 * group, .format = .{ .bits = bits, .group = group }, .sk = 4, .pf = 2, .ranges = &.{.{ 1, 1 }}, .groups = .{ 1, 3 }, .output = .f32 }, &counts);
    }
    std.debug.print("PASS {d} layouts, {d} bit-equal row/window pairs, {d} frozen fn_lane byte-equal windows, {d} CPU-reference values; max fp32 error {d:.9}, max roundoff-bound fraction {d:.6}\n", .{
        counts.layouts, counts.row_pairs, counts.golden_windows, counts.cpu_values, counts.max_error, counts.max_fraction,
    });
}
