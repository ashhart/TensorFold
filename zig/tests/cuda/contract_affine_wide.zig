//! Full-byte 8192-row prepared-affine comparison with independent sampled quantized FP64 references.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const Base = @import("contract_fp8_prompt.zig");
const Rig = @import("contract_rig.zig").Rig;
const Gpu = @import("check.zig").Gpu;

fn decode(bits: u8) f64 {
    const code = bits & 127;
    if (code == 127) return std.math.nan(f64);
    const exponent: i32 = @intCast(code >> 3);
    const mantissa: f64 = @floatFromInt(code & 7);
    const value = if (exponent == 0) std.math.ldexp(mantissa, -9) else std.math.ldexp(8 + mantissa, exponent - 10);
    return if (bits & 128 == 0) value else -value;
}
fn reference(f: *const Base.Case, row: usize, column: usize) struct { value: f64, magnitude: f64 } {
    const k = f.shape.k;
    const groups = k / 64;
    var value: f64 = 0;
    var magnitude: f64 = 0;
    for (0..groups) |group| {
        var dot: f64 = 0;
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..64) |i| {
            const x = decode(f.q[row * k + group * 64 + i]);
            const term = x * @as(f64, @floatFromInt(f.codes[column * k + group * 64 + i]));
            dot += term;
            sum += x;
            mag += @abs(term);
        }
        const scale = c.fromBf16(f.scales[column * groups + group]);
        const bias = c.fromBf16(f.biases[column * groups + group]);
        const activation = f.ascales[row * groups + group];
        value += (dot * scale + sum * bias) * activation;
        magnitude += (mag * @abs(scale) + @abs(sum * bias)) * @abs(activation);
    }
    return .{ .value = value, .magnitude = magnitude };
}

pub fn run(gpu: Gpu) !void {
    if (@import("qwen_prefill_config").max_rows != 8192) return error.WideAffineGateNeeds8192Build;
    if (!cuda.fp8_prefill.affine_packing.selectedRows(8192)) return error.WideAffineGateNotDispatched;
    try runCase(gpu, 8192, 10240, 5120, .bf16);
    try runCase(gpu, 8192, 48, 5120, .f32);
}

fn runCase(gpu: Gpu, rows: usize, columns: usize, k: usize, kind: cuda.qmmf.Out) !void {
    std.debug.print("START wide affine M={d} N={d} K={d} out={t}\n", .{ rows, columns, k, kind });
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var fixture: Base.Case = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = .{ .kind = .affine4, .out = kind } };
    defer fixture.deinit();
    if (!try fixture.prepare(.{ .rows = 17, .n = columns, .k = k }, .random, 0xa48192))
        return error.WideAffineFixtureRejected;
    var rig: Rig = .{ .gpu = gpu, .s = stream };
    defer rig.deinit();
    const x = try gpu.gpa.alloc(u16, rows * k);
    defer gpu.gpa.free(x);
    for (0..rows) |row| @memcpy(x[row * k ..][0..k], fixture.x[(row % 17) * k ..][0..k]);
    const dx = try rig.input(std.mem.sliceAsBytes(x));
    const bytes = rows * columns * c.size(Base.spec(fixture.config).out);
    const output = try rig.output(bytes);
    const baseline_out = try rig.scratch(bytes);
    var native = try cuda.fp8_prefill.Native.load(gpu.d, rows, k);
    defer native.unload();
    const scratch_len = native.scratch.len;
    const scratch = try rig.scratch(scratch_len);
    native.scratch.free();
    native.scratch = .{ .d = gpu.d, .ptr = scratch, .len = scratch_len, .borrowed = true };
    if (native.affine_prepared == null) native.affine_prepared = try cuda.fp8_prefill.affine_packing.Prepared.load(gpu.d);
    const packed_len = cuda.fp8_prefill.affine_packing.required(fixture.affine).?;
    native.affine_prepared.?.buffer = .{ .d = gpu.d, .ptr = try rig.scratch(packed_len), .len = packed_len, .borrowed = true };
    var original = native;
    original.affine_prepared = null;
    try original.affine4(stream, dx, rows, fixture.affine, baseline_out, kind);
    const baseline = try gpu.gpa.dupe(u8, try rig.bytes(baseline_out, bytes));
    defer gpu.gpa.free(baseline);
    const original_weights = try gpu.gpa.dupe(u8, try fixture.rig.inputBytes());
    defer gpu.gpa.free(original_weights);
    for (0..5) |_| {
        const out_view: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = output, .len = bytes, .borrowed = true };
        try out_view.fill8(0xff, stream.handle);
        try native.affine4(stream, dx, rows, fixture.affine, output, kind);
        const actual = try rig.outputBytes();
        if (!std.mem.eql(u8, actual, baseline)) return error.WideAffineOriginalBytesDiffer;
    }
    if (native.affine_prepared.?.last_stream == null) return error.WideAffinePreparationNotUsed;
    const actual = try rig.outputBytes();
    for (0..256) |i| {
        const row = if (i == 0) 0 else if (i == 1) rows - 1 else (i * 2654435761 + 7) % rows;
        const column = if (i == 0) 0 else if (i == 1) columns - 1 else (i * 2246822519 + 13) % columns;
        const want = reference(&fixture, row % 17, column);
        const got = c.read(Base.spec(fixture.config).out, actual, row * columns + column);
        const bound = c.bound(Base.spec(fixture.config), .{ .rows = rows, .n = columns, .k = k }, want.value, want.magnitude);
        if (!std.math.isFinite(got) or @abs(got - want.value) > bound) return error.WideAffineFp64Mismatch;
    }
    if (!std.mem.eql(u8, try rig.bytes(dx, x.len * 2), std.mem.sliceAsBytes(x)) or
        !std.mem.eql(u8, try fixture.rig.inputBytes(), original_weights)) return error.WideAffineInputChanged;
    if (!try rig.intact() or !try fixture.rig.intact()) return error.WideAffineGuardChanged;
    std.debug.print("PASS wide affine M={d} N={d} K={d} out={t}: original bytes, 256 FP64 samples, five runs, guards\n", .{
        rows, columns, k, kind,
    });
}

test "FP8 prompt wide affine sampled formula retains the original FP64 reference" {
    std.mem.doNotOptimizeAway(&run);
    var fixture: Base.Case = undefined;
    fixture.shape = .{ .rows = 1, .n = 1, .k = 64 };
    var q: [64]u8 = @splat(0x38);
    var codes: [64]u8 = @splat(3);
    var scales = [_]u16{c.toBf16(0.25)};
    var biases = [_]u16{c.toBf16(-0.5)};
    var ascales = [_]f32{0.125};
    fixture.q = &q;
    fixture.codes = &codes;
    fixture.scales = &scales;
    fixture.biases = &biases;
    fixture.ascales = &ascales;
    const value = reference(&fixture, 0, 0);
    try std.testing.expectEqual(@as(f64, 2), value.value);
    try std.testing.expectEqual(@as(f64, 10), value.magnitude);
}
