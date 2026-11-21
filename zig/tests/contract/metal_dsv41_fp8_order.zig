//! Pin the released K32 pair tree and its ascending diagnostic on exact, signed cancellation inputs.
const std = @import("std");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const c = @import("contract").spec;
const gpu = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");
const mma = tf.dsv41_metal.fp8_mma;
const tile = tf.dsv41_metal.fp8_tile;
const rows = tf.dsv41_metal.fp8_rows;

const k = 5120;
const n = 512;
const groups = k / 32;
const max_rows = mma.row_tile + 1;
const Order = enum { pair, ascending };
const Path = enum { prompt, verifier, serial };

fn negativeGroup(column: usize) usize {
    return if (column / 32 % 2 == 0) 2 else 128;
}

fn plant(f: ref.Fp8) !void {
    @memset(f.x, 0);
    @memset(f.codes, 0);
    @memset(f.scales, 127);
    for (0..max_rows) |r| for ([_]usize{ 0, 1, 2, 3, 128, 129 }) |g| {
        const last = g == 3 or g == 129;
        f.x[r * k + g * 32] = c.toBf16(if (last and r % 2 == 1) -1 else 1);
    };
    // Alternate scale tiles place the cancelling pair on opposite sides of a verifier's K split.
    for (0..n) |j| {
        const neg = negativeGroup(j);
        const w = f.codes[j * k ..][0..k];
        w[0] = 0x38;
        w[32] = 0x38;
        w[neg * 32] = 0xb8;
        w[(neg + 1) * 32] = if (j % 2 == 1) 0xb8 else 0x38;
    }
    for (0..n / 32) |j| {
        f.scales[j * groups] = 152;
        f.scales[j * groups + negativeGroup(j * 32)] = 152;
    }
    // One product per block; FP64 represents every intermediate of these two exact integer sums.
    f.dequant();
    var leaves: [ref.levelsOf(groups)]f32 = undefined;
    for (0..2) |r| for ([_]usize{ 0, 1, 32, 33 }) |j| {
        const want: f64 = if ((r + j) % 2 == 0) 2 else 0;
        if (f.one(r, j, &leaves).exact != want) return error.WrongCancellationFixture;
    };
}

fn view(comptime T: type, b: gpu.Ref) T {
    return .{ .buf = b.buf, .off = b.off };
}

const Buffers = struct { x: gpu.Ref, w: gpu.Ref, s: gpu.Ref, y: gpu.Ref };
const Pipes = struct { prompt: *const mma.Pipelines, verifier: *const tile.Pipelines, serial: *const rows.Pipelines };

const Report = struct {
    launches: usize = 0,
    failed: usize = 0,
    damaged_guard: bool = false,

    fn fail(r: *Report, comptime fmt: []const u8, args: anytype) void {
        r.failed += 1;
        if (r.failed <= 8) std.debug.print("FAIL dsv41_fp8 block order " ++ fmt ++ "\n", args);
    }
};

fn check(rig: *gpu.Rig, b: Buffers, p: Pipes, path: Path, order: Order, out: mma.Out, count: u32, report: *Report) !void {
    if (report.damaged_guard) return;
    const size: usize = if (out == .f32) 4 else 2;
    const bytes = rig.bytes(b.y, max_rows * n * 4);
    @memset(bytes, 0xa5);
    const pass = rig.begin();
    switch (path) {
        .prompt => try mma.encode(p.prompt, pass.e, view(mma.Ref, b.x), view(mma.Ref, b.w), view(mma.Ref, b.s), view(mma.Ref, b.y), .{ .k = k, .n = n, .rows = count, .out = out }),
        .verifier => {
            const call: tile.Call = .{ .k = k, .n = n, .rows = count, .out = if (out == .f32) .f32 else .bf16 };
            var form = tile.formOf(k, n);
            // A single group owns all K32 blocks; multiple groups would sum separate chains.
            if (order == .ascending) form.sgs = 1;
            try tile.encodeAs(p.verifier, pass.e, view(tile.Ref, b.x), view(tile.Ref, b.w), view(tile.Ref, b.s), view(tile.Ref, b.y), call, .{}, form);
        },
        .serial => {
            const call: rows.Call = .{ .k = k, .n = n, .rows = 1, .split = rows.splitOf(k), .out = if (out == .f32) .f32 else .bf16 };
            try rows.encodeWhole(p.serial, pass.e, view(rows.Ref, b.x), view(rows.Ref, b.w), view(rows.Ref, b.s), view(rows.Ref, b.y), call, .{});
        },
    }
    try rig.finish(pass);
    report.launches += 1;
    if (!try rig.intact()) {
        report.damaged_guard = true;
        report.fail("{s} {s} {s} rows={d}: damaged buffer guard; stopping this probe", .{ @tagName(path), @tagName(order), @tagName(out), count });
        return;
    }
    if (!std.mem.allEqual(u8, bytes[count * n * size ..], 0xa5)) {
        report.fail("{s} {s} {s} rows={d}: wrote past the requested rows", .{ @tagName(path), @tagName(order), @tagName(out), count });
        return;
    }
    for (0..count) |r| for (0..n) |j| {
        const last: f32 = if ((r + j) % 2 == 0) 1 else -1;
        const expected: f32 = if (order == .pair) 0 else last;
        const offset = (r * n + j) * size;
        const got: u32 = if (out == .f32) std.mem.readInt(u32, bytes[offset..][0..4], .little) else std.mem.readInt(u16, bytes[offset..][0..2], .little);
        const want: u32 = if (out == .f32) @bitCast(expected) else c.toBf16(expected);
        if (got != want) {
            report.fail("{s} {s} {s} rows={d} row={d} col={d}: got 0x{x}, want 0x{x}", .{ @tagName(path), @tagName(order), @tagName(out), count, r, j, got, want });
            return;
        }
    };
}

pub fn run(rig: *gpu.Rig, stock_prompt: *const mma.Pipelines) !usize {
    rig.reset();
    defer rig.reset();
    const f = try ref.Fp8.init(rig.a, .{ .k = k, .n = n, .rows = max_rows });
    defer f.deinit();
    try plant(f);
    const b: Buffers = .{
        .x = try rig.input(std.mem.sliceAsBytes(f.x)),
        .w = try rig.input(f.codes),
        .s = try rig.input(f.scales),
        .y = try rig.output(max_rows * n * 4),
    };
    var asc_prompt = try mma.Pipelines.loadSource(rig.device, "#define TF_CTL 2\n" ++ sources.dsv41.fp8_mma);
    defer asc_prompt.deinit();
    var stock_tile = try tile.Pipelines.load(rig.device);
    defer stock_tile.deinit();
    var asc_tile = try tile.Pipelines.loadSource(rig.device, "#define TF_CTL 2\n" ++ sources.dsv41.fp8_tile);
    defer asc_tile.deinit();
    var serial = try rows.Pipelines.load(rig.device);
    defer serial.deinit();
    var report: Report = .{};
    for ([_]Order{ .pair, .ascending }) |order| {
        const p: Pipes = .{
            .prompt = if (order == .pair) stock_prompt else &asc_prompt,
            .verifier = if (order == .pair) &stock_tile else &asc_tile,
            .serial = &serial,
        };
        for ([_]mma.Out{ .f32, .bf16 }) |out| {
            if (order == .pair) try check(rig, b, p, .serial, order, out, 1, &report);
            for (1..tile.max_rows + 1) |width| try check(rig, b, p, .verifier, order, out, @intCast(width), &report);
            for ([_]u32{ 1, 16, 17, mma.row_tile - 1, mma.row_tile, max_rows }) |width| try check(rig, b, p, .prompt, order, out, width, &report);
        }
    }
    std.debug.print("{s} dsv41_fp8 block order: signed four-block cancellation, FP32/BF16, pair row, verifier 1-16 and prompt 1/16/17/31/32/33; {d} launches run, {d} launches differ\n", .{ if (report.failed == 0) "PASS" else "FAIL", report.launches, report.failed });
    return report.failed;
}
