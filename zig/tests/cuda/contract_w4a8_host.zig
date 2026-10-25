//! CPU checks of the production normalization helper, independent references, envelopes and accumulator controls.
const std = @import("std");
const core = @import("core");
const c = core.contract.spec;
const g = core.contract.gates;
const ref = @import("w4a8_ref.zig");
const cases = @import("contract_w4a8.zig");
const T = std.testing;
extern fn tf_w4a8_normalize_host(f32, f32) f32;

test "w4a8 host: exceptional sentinels fit one admitted row and preserve its guards" {
    var guarded: [130]u16 = @splat(c.toBf16(1));
    cases.plantExceptional(guarded[1..129]);
    try T.expectEqual(c.Class.nan, c.classOf(c.fromBf16(guarded[1])));
    try T.expectEqual(c.Class.pos_inf, c.classOf(c.fromBf16(guarded[2])));
    try T.expectEqual(c.Class.neg_inf, c.classOf(c.fromBf16(guarded[3])));
    try T.expectEqual(c.toBf16(1), guarded[0]);
    for (guarded[4..]) |v| try T.expectEqual(c.toBf16(1), v);
}

test "w4a8 host: finite reciprocal overflow normalizes extrema and preserves ordinary multiplication bits" {
    for ([_]f32{ 0x1p-133, 0x1p-127, 0x1p-122, 0x1p-121, 0x1p-120, 1, 127, 65536 }) |top| {
        try T.expect(std.math.isFinite(tf_w4a8_normalize_host(top, top)));
        try T.expect(std.math.isFinite(tf_w4a8_normalize_host(-top, top)));
        try T.expectEqual(@as(i8, 127), ref.nearestEven(tf_w4a8_normalize_host(top, top)));
        try T.expectEqual(@as(i8, -127), ref.nearestEven(tf_w4a8_normalize_host(-top, top)));
        const inv: f32 = 127 / top;
        if (!std.math.isFinite(inv)) {
            try T.expect(!std.math.isFinite(top * inv)); // the original reciprocal path is a planted numerical control
        } else for ([_]f32{ -1, -0.5, 0, 0.25, 0.5, 1 }) |ratio| {
            const v: f32 = ratio * top;
            try T.expectEqual(@as(u32, @bitCast(v * inv)), @as(u32, @bitCast(tf_w4a8_normalize_host(v, top))));
        }
    }
    try T.expectEqual(@as(f32, 0), tf_w4a8_normalize_host(0, 0));
    for ([_]f32{ -2.5, -1.5, -0.5, 0.5, 1.5, 2.5 }, [_]i8{ -2, -2, 0, 0, 2, 2 }) |v, want|
        try T.expectEqual(want, ref.nearestEven(tf_w4a8_normalize_host(v, 127)));
}

test "w4a8 host: independent group reference covers zero, tiny, ties and signed extremes" {
    var x: [64]u16 = @splat(0);
    var out = try ref.group(&x);
    try T.expectEqual(@as(f32, 1), out.scale);
    try T.expectEqual(@as(f64, 0), out.sum);
    for (out.codes) |v| try T.expectEqual(@as(i8, 0), v);
    for ([_]f32{ 0x1p-133, 0x1p-122, 127 }) |top| {
        x[0] = c.toBf16(top);
        x[1] = c.toBf16(-top);
        out = try ref.group(&x);
        try T.expectEqual(@as(i8, 127), out.codes[0]);
        try T.expectEqual(@as(i8, -127), out.codes[1]);
        try T.expectEqual(@as(f64, 0), out.sum);
    }
    x[2] = c.toBf16(0.5);
    x[3] = c.toBf16(1.5);
    out = try ref.group(&x);
    try T.expectEqual(@as(i8, 0), out.codes[2]);
    try T.expectEqual(@as(i8, 2), out.codes[3]);
    try T.expectEqual(@as(f64, 2), out.sum);
    x[0] = c.toBf16(std.math.nan(f32));
    try T.expectError(error.ExceptionalQuantization, ref.group(&x));
}

test "w4a8 host: packed layout covers every input and zero-padded output column" {
    var seen: [32]bool = @splat(false);
    for (0..32) |i| {
        try T.expect(!seen[ref.inputAt(i)]);
        seen[ref.inputAt(i)] = true;
    }
    const n = 65;
    const k = 192;
    const padded = 128;
    var codes: [n * k]u8 = undefined;
    for (&codes, 0..) |*v, i| v.* = @intCast(i % 16);
    var words: [padded * k / 8]u32 = undefined;
    ref.pack(&words, &codes, n, k);
    const offsets = [_]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
    for (words, 0..) |word, idx| {
        const step = idx & 1;
        const lane = (idx >> 1) & 31;
        const j = (idx >> 6) & 7;
        const group = (idx >> 9) % (k / 64);
        const tile = (idx >> 9) / (k / 64);
        const row = tile * 64 + j * 8 + (lane >> 2);
        for (offsets, 0..) |offset, p| {
            const input = group * 64 + 32 * step + 2 * (lane & 3) + offset;
            const want: u32 = if (row < n) codes[row * k + input] else 0;
            try T.expectEqual(want, (word >> @intCast(4 * p)) & 15);
        }
    }
}

test "w4a8 host: both real registrations expose row, column and group edges and require an executed exceptional rule" {
    for (std.enums.values(cases.Kind)) |kind| {
        const s = cases.spec(kind);
        try T.expectEqual(c.Accum.f32, s.accum);
        try T.expectEqual(@as(usize, 128), s.tiles[0].size);
        const shapes = try c.shapes(T.allocator, s);
        defer T.allocator.free(shapes);
        for ([_]usize{ 64, 128, 192, 256, 32768 }) |k| {
            var found = false;
            for (shapes) |shape| found = found or shape.k == k;
            try T.expect(found);
        }
        for ([_]usize{ 1, 17, 127, 128, 129, 4096 }) |rows| {
            var found = false;
            for (shapes) |shape| found = found or shape.rows == rows;
            try T.expect(found);
        }
        var case: cases.Case = .{ .rig = undefined, .kind = kind, .kernels = undefined, .arena = .init(T.allocator) };
        defer case.arena.deinit();
        const registration = case.kernel();
        try T.expect(registration.vt.simulate != null and registration.vt.bounds != null);
        try T.expect((try case.check(.nonfinite)) != null);
        try T.expect(s.known == null); // the settled rule has no known waiver
    }
}

const Cpu = struct {
    kind: cases.Kind,
    narrow: bool = false,
    terms: [512]f64 = undefined,
    bytes: [8]u8 = @splat(0),
    fn n(p: *Cpu) usize {
        return if (p.kind == .quantize) 64 else 512;
    }
    fn of(p: *anyopaque) *Cpu {
        return @ptrCast(@alignCast(p));
    }
    fn prepare(p: *anyopaque, _: c.Shape, fill: g.Fill, _: u64) anyerror!bool {
        const self = of(p);
        if (fill != .stress) return error.StressOnly;
        for (self.terms[0..self.n()], 0..) |*v, i| v.* = if (self.kind == .quantize) (if (i % 2 == 0) @as(f64, 65536) else 1) else switch (i / 128) {
            0 => 32768,
            1, 3 => 1,
            else => -32768,
        };
        return true;
    }
    fn run(p: *anyopaque) anyerror!void {
        const self = of(p);
        const v = ref.sumTerms(if (self.narrow) .bf16 else .f32, .sequential, self.terms[0..self.n()]);
        if (self.kind == .quantize) std.mem.writeInt(u32, self.bytes[0..4], @bitCast(@as(f32, @floatCast(v))), .little) else std.mem.writeInt(u16, self.bytes[0..2], c.toBf16(@floatCast(v)), .little);
    }
    fn output(p: *anyopaque) anyerror![]const u8 {
        return &of(p).bytes;
    }
    fn inputs(_: *anyopaque) anyerror![]const u8 {
        return &.{};
    }
    fn intact(_: *anyopaque) anyerror!bool {
        return true;
    }
    fn overlap(_: *anyopaque) bool {
        return false;
    }
    fn count(_: *anyopaque) usize {
        return 1;
    }
    fn reference(p: *anyopaque, want: []f64, mag: []f64) anyerror!void {
        const self = of(p);
        want[0] = 0;
        mag[0] = 0;
        for (self.terms[0..self.n()]) |v| {
            want[0] += v;
            mag[0] += @abs(v);
        }
    }
    fn simulate(p: *anyopaque, acc: c.Accum, order: g.Order, out: []f64) anyerror!void {
        const self = of(p);
        out[0] = ref.sumTerms(acc, order, self.terms[0..self.n()]);
    }
    const vt: g.VTable = .{ .prepare = prepare, .run = run, .output = output, .inputs = inputs, .intact = intact, .overlap = overlap, .count = count, .reference = reference, .simulate = simulate };
    fn kernel(self: *Cpu) g.Kernel {
        return .{ .spec = cases.spec(self.kind), .ctx = self, .vt = &vt };
    }
};

test "w4a8 host: gate four rejects narrow group and epilogue sums in all three orders" {
    const quiet = g.quiet;
    g.quiet = true;
    defer g.quiet = quiet;
    for (std.enums.values(cases.Kind)) |kind| {
        var cpu: Cpu = .{ .kind = kind };
        try T.expectEqual(g.Verdict.pass, try g.sums(T.allocator, cpu.kernel(), 29));
        cpu.narrow = true;
        try T.expectEqual(g.Verdict.fail, try g.sums(T.allocator, cpu.kernel(), 29));
    }
    const codes: [64]u8 = @splat(15);
    const xq: [64]i8 = @splat(127);
    var terms: [1]f64 = undefined;
    ref.terms(&xq, &.{1}, &.{0}, &codes, &.{c.toBf16(1)}, &.{0}, &terms);
    try T.expectEqual(@as(f64, 121920), terms[0]);
    try T.expect(terms[0] < 0x1p23); // the per-group int32 dot and its float-offset representation are exact
}

extern fn tf_w4a8_store_host([*]u16, c_int, f32, f32) c_int;

test "w4a8 host: repeat row one column zero never issues a misaligned BF16 pair" {
    // repeat clamps served[0] to 64 rows: N=129 puts row 1 at byte 258, still inside the payload.
    const shape = cases.spec(.prefill).served[0];
    try T.expectEqual(@as(usize, 129), shape.n);
    try T.expectEqual(@as(usize, 256), shape.k);
    var output: [64 * 129 + 256]u16 align(256) = @splat(0xa5a5);
    const dst = output[128 + 129 ..].ptr;
    try T.expectEqual(@as(usize, 2), @intFromPtr(dst) % 4);
    try T.expectEqual(@as(c_int, 0), tf_w4a8_store_host(dst, 129, 1.00390625, -1.01171875));
    try T.expectEqual(c.toBf16(1.00390625), dst[0]);
    try T.expectEqual(c.toBf16(-1.01171875), dst[1]);
    try T.expectEqual(@as(u16, 0xa5a5), dst[2]);
}

test "w4a8 host: actual epilogue stores cover every tile edge exactly once with intact guards" {
    for (cases.spec(.prefill).served) |shape| try stores(shape.rows, shape.n);
    // The repeat shape, the nonfinite shape (no GPU launch), and compute-sanitizer's prompt sizes.
    try stores(64, 129);
    try stores(8, 129);
    for ([_]usize{ 5120, 17408, 6144, 200 }) |n| try stores(200, n);
}

fn value(row: usize, col: usize) f32 {
    const edges = [_]f32{ 0, -0.0, 1.00390625, -1.01171875, 0x1p-133, -0x1p-133, 0x1.fffffep127, -0x1.fffffep127 };
    return edges[(row + col) % edges.len];
}

fn stores(rows: usize, n: usize) !void {
    const all = try T.allocator.alignedAlloc(u16, .fromByteUnits(256), rows * n + 256);
    defer T.allocator.free(all);
    @memset(all, 0xa5a5);
    const out = all[128..][0 .. rows * n];
    const seen = try T.allocator.alloc(u8, rows * n);
    defer T.allocator.free(seen);
    @memset(seen, 0);
    // Enumerate the real 128x128, 2x4-warp MMA epilogue, independently of the store helper.
    for (0..(rows + 127) / 128) |bm| for (0..(n + 127) / 128) |bn| {
        for (0..256) |tid| for (0..4) |i| for (0..4) |j| for (0..2) |h| {
            const warp = tid / 32;
            const lane = tid % 32;
            const row = bm * 128 + warp / 4 * 64 + i * 16 + lane / 4 + h * 8;
            const col = bn * 128 + warp % 4 * 32 + j * 8 + lane % 4 * 2;
            if (row >= rows or col >= n) continue;
            try T.expectEqual(@as(c_int, 0), tf_w4a8_store_host(out[row * n + col ..].ptr, @intCast(n - col), value(row, col), value(row, col + 1)));
            for (col..@min(col + 2, n)) |c_| seen[row * n + c_] += 1;
        };
    };
    for (out, seen, 0..) |v, count, idx| {
        try T.expectEqual(@as(u8, 1), count);
        try T.expectEqual(c.toBf16(value(idx / n, idx % n)), v);
    }
    for (all[0..128]) |v| try T.expectEqual(@as(u16, 0xa5a5), v);
    for (all[128 + out.len ..]) |v| try T.expectEqual(@as(u16, 0xa5a5), v);
}
comptime {
    _ = @import("w4a8_exception_source_host.zig");
}
comptime {
    _ = @import("w4a8_exception_gate_host.zig");
}
