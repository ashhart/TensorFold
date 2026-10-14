//! FP64 ideal-reference isolation for the CSA2 compressed/sparse-attention stages: rounding level,
//! Metal deltas and cancellation amplification. Receipt lines print with a "CSA64" prefix.
//! No tolerance is relaxed and no source coefficient is injected: synthetic tensors only, the
//! Metal kernel, the FP32 source-contract reference and the FP64 ideal reference on identical rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const math = @import("math.zig");
const attention_math = @import("attention_math.zig");

const kv = tf.deepseek_v41.kv;
const csa2 = tf.deepseek_v41.csa2;
const View = tf.deepseek_v41.attention.View;

const width: usize = 512;
const ratio: usize = 2;
const eps: f32 = 0.000001;

/// The local FP32 spacing at the FP64 reference value: the ULP unit for kernel deltas.
fn nextUpF32(x: f32) f32 {
    if (std.math.isNan(x) or std.math.isInf(x)) return x;
    const bits: u32 = @bitCast(x);
    if (bits == 0 or bits == 0x80000000) return @bitCast(@as(u32, 1));
    return @bitCast(if (bits >> 31 == 0) bits + 1 else bits - 1);
}

fn ulp32At(v: f64) f64 {
    const f: f32 = @floatCast(v);
    if (std.math.isNan(f) or std.math.isInf(f)) return @abs(v);
    const a = @abs(f);
    if (a == 0) return 0x1p-149;
    const up = nextUpF32(a);
    if (std.math.isInf(up)) return @as(f64, a) - @as(f64, -nextUpF32(-a));
    return @as(f64, up) - @as(f64, a);
}

/// A BF16-cell comparison: NRMS against the ideal, and the count of cells the two sides disagree on.
const BfDelta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    unequal: usize = 0,
    worst: usize = 0,
    worst_relative: f64 = 0,
};

fn deltaBf16(actual: []const u16, ideal: []const f64) BfDelta {
    var d: BfDelta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |cell, r, i| {
        const a: f64 = math.f(cell);
        const diff = a - r;
        squared += diff * diff;
        magnitude += r * r;
        const rel = @abs(diff) / @max(@abs(r), 0x1p-126);
        if (cell != math.b64(r)) {
            d.unequal += 1;
            if (rel > d.worst_relative) {
                d.worst_relative = rel;
                d.worst = i;
            }
        }
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

/// An FP32-cell comparison, used for pre-rounding stage boundaries that stay in FP32.
const Delta = struct {
    count: usize = 0,
    nrms: f64 = 0,
    max_abs: f64 = 0,
    max_ulp: f64 = 0,
    worst: usize = 0,
};

fn deltaF32VsF64(actual: []const f32, ideal: []const f64) Delta {
    var d: Delta = .{ .count = actual.len };
    var squared: f64 = 0;
    var magnitude: f64 = 0;
    for (actual, ideal, 0..) |a, r, i| {
        const diff = @as(f64, a) - r;
        squared += diff * diff;
        magnitude += r * r;
        const ad = @abs(diff);
        if (ad > d.max_abs) {
            d.max_abs = ad;
            d.worst = i;
        }
        const u = ad / ulp32At(r);
        if (u > d.max_ulp) d.max_ulp = u;
    }
    d.nrms = if (magnitude == 0) @sqrt(squared) else @sqrt(squared / magnitude);
    return d;
}

fn printBf(case_name: []const u8, stage: []const u8, label: []const u8, d: BfDelta) void {
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"{s}","side":"{s}","cells":{d},"nrms_vs_ideal":{d},"bf16_unequal_vs_ideal":{d},"worst_index":{d},"worst_relative":{d}}}
    , .{ case_name, stage, label, d.count, d.nrms, d.unequal, d.worst, d.worst_relative });
}

fn printF32(case_name: []const u8, stage: []const u8, label: []const u8, d: Delta) void {
    std.debug.print("CSA64 {{\"case\":\"{s}\",\"stage\":\"{s}\",\"side\":\"{s}\",\"cells\":{d}," ++
        "\"nrms_vs_ideal\":{d},\"max_abs_vs_ideal\":{d},\"max_ulp_vs_ideal\":{d:.3}," ++
        "\"worst_index\":{d}}}\n", .{ case_name, stage, label, d.count, d.nrms, d.max_abs, d.max_ulp, d.worst });
}

const Lcg = struct {
    state: u64 = 0x853c49e6748fea9b,
    fn next(s: *Lcg) u64 {
        s.state = s.state *% 6364136223846793005 +% 1442695040888963407;
        return s.state >> 11;
    }
    /// Uniform [-1, 1) with 24-bit granularity, exactly representable in FP32.
    fn uniform(s: *Lcg) f32 {
        const v: f64 = @as(f64, @floatFromInt(s.next() & 0xffffff)) / 16777216.0;
        return @floatCast(v * 2 - 1);
    }
    fn scale(s: *Lcg, amplitude: f32) f32 {
        return s.uniform() * amplitude;
    }
};

const Buffers = struct {
    device: mtl.Device,
    items: [16]mtl.Buffer = undefined,
    count: usize = 0,
    fn get(b: *Buffers, comptime T: type, n: usize) !mtl.Buffer {
        const count = @max(n, 1);
        const x = try b.device.buffer(try std.math.mul(usize, @sizeOf(T), count), mtl.ResourceOptions.shared);
        b.items[b.count] = x;
        b.count += 1;
        @memset(x.slice(T, n), 0);
        return x;
    }
    fn deinit(b: *Buffers) void {
        for (b.items[0..b.count]) |x| x.deinit();
    }
};

fn view(buffer: mtl.Buffer) View {
    return .{ .buffer = buffer };
}

/// One compressor-pool group: the two projected positions of a ratio-2 group, pooled then RMS-normed.
/// `values[i][d]`/`gates[i][d]` are the FP32 outputs of the compressor's wkv/wgate projections; the
/// kernel and both references read exactly these numbers.
fn poolCase(device: mtl.Device, queue: mtl.Queue, p: *const kv.Pipelines,
    case_name: []const u8, values: []const f32, gates: []const f32, weight: []const u16) !void
{
    var b = Buffers{ .device = device };
    defer b.deinit();
    const values_gpu = try b.get(f32, 2 * width);
    const gates_gpu = try b.get(f32, 2 * width);
    const partial_gpu = try b.get(f32, 2 * ratio * width);
    const weight_gpu = try b.get(u16, width);
    const latent_gpu = try b.get(u16, width);
    @memcpy(values_gpu.slice(f32, 2 * width), values);
    @memcpy(gates_gpu.slice(f32, 2 * width), gates);
    @memcpy(weight_gpu.slice(u16, width), weight);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encodePool(enc, view(values_gpu), view(gates_gpu), view(partial_gpu), view(weight_gpu), view(latent_gpu),
        .{ .rows = 2, .width = @intCast(width), .ratio = @intCast(ratio), .position = 2, .stride = @intCast(width), .input_f32 = 1, .eps = eps });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;
    const metal = latent_gpu.slice(u16, width);

    // Both references: the same pool, then the same RMS boundary, each at its own precision.
    var ref32: [width]u16 = undefined;
    var ref64: [width]u16 = undefined;
    var pooled32: [width]u16 = undefined;
    var pooled64: [width]u16 = undefined;
    attention_math.poolOf(f32, .{ values[0..width], values[width .. 2 * width] },
        .{ gates[0..width], gates[width .. 2 * width] }, &pooled32);
    var wide_values: [2][width]f64 = undefined;
    var wide_gates: [2][width]f64 = undefined;
    for (0..2) |slot| wide_values[slot] = widen(values[slot * width ..][0..width]);
    for (0..2) |slot| wide_gates[slot] = widen(gates[slot * width ..][0..width]);
    attention_math.poolOf(f64, .{ &wide_values[0], &wide_values[1] }, .{ &wide_gates[0], &wide_gates[1] }, &pooled64);
    var cells: [width]f32 = undefined;
    for (weight, 0..) |cell, i| cells[i] = math.f(cell);
    math.rmsOfValues(f32, &pooled32, &cells, eps, &ref32);
    math.rmsOfValues(f64, &pooled64, &cells, eps, &ref64);

    const ideal = widenCells(&ref64);
    var from_f32 = widenCells(&ref32);
    var magnitude: f64 = 0;
    for (ideal) |v| magnitude += v * v;
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"pool","side":"scale","ideal_rms":{d}}}
    , .{ case_name, @sqrt(magnitude / @as(f64, width)) });
    printBf(case_name, "pool", "metal", deltaBf16(metal, &ideal));
    printBf(case_name, "pool", "f32_reference", deltaBf16(&ref32, &ideal));
    printBf(case_name, "pool", "metal_vs_f32_reference", deltaBf16(metal, &from_f32));
}

fn widen(cells: []const f32) [width]f64 {
    var out: [width]f64 = undefined;
    for (cells, 0..) |v, i| out[i] = @as(f64, v);
    return out;
}

fn widenCells(cells: *const [width]u16) [width]f64 {
    var out: [width]f64 = undefined;
    for (cells, 0..) |cell, i| out[i] = math.f(cell);
    return out;
}

test "compressor pool stage sits at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try kv.Pipelines.load(device);
    defer p.deinit();

    var rng = Lcg{ .state = 0x9e3779b97f4a7c15 };
    // Case 1: typical magnitudes; gates straddle each other so the pool weights split.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| values[i] = rng.scale(1.0);
        for (0..2 * width) |i| gates[i] = rng.scale(8.0);
        try poolCase(device, queue, &p, "typical", values, gates, try weightCells(gpa, &rng));
    }
    // Case 2: cancellation - the two pooled terms cancel to a few ULPs of the larger one.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..width) |d| {
            const large = rng.scale(64.0);
            const small = rng.scale(0.001);
            values[d] = large + small;
            values[width + d] = -large + rng.scale(0.001);
        }
        for (0..2 * width) |i| gates[i] = rng.scale(0.5);
        try poolCase(device, queue, &p, "cancellation", values, gates, try weightCells(gpa, &rng));
    }
    // Case 3: wide dynamic range, so one slot dominates and the other is far below it.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| {
            const exponent = @as(f32, @floatFromInt(rng.next() % 60)) - 30;
            values[i] = std.math.pow(f32, 2, exponent) * rng.uniform();
            gates[i] = rng.scale(3.0);
        }
        try poolCase(device, queue, &p, "wide_range", values, gates, try weightCells(gpa, &rng));
    }
    // Case 4: deep cancellation, 1e-7 of the terms it cancels: FP32 error is a visible fraction.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..width) |d| {
            const large = rng.scale(64.0);
            values[d] = large * (1 + rng.scale(1e-7));
            values[width + d] = -large * (1 + rng.scale(1e-7));
        }
        for (0..2 * width) |i| gates[i] = rng.scale(0.5);
        try poolCase(device, queue, &p, "cancel_1e7", values, gates, try weightCells(gpa, &rng));
    }
    // Case 5: gate skew that pushes one softmax weight into FP32 subnormal territory.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        for (0..2 * width) |i| values[i] = rng.scale(1.0);
        for (0..width) |d| {
            const base = rng.scale(4.0);
            gates[d] = base;
            gates[width + d] = base - 85 - @as(f32, @floatFromInt(rng.next() % 8));
        }
        try poolCase(device, queue, &p, "gate_skew", values, gates, try weightCells(gpa, &rng));
    }
    // Case 6: near-equal gates so the weights are both near one half, with large cancelling values.
    {
        const values = try gpa.alloc(f32, 2 * width);
        const gates = try gpa.alloc(f32, 2 * width);
        const shared = try gpa.alloc(f32, width);
        for (0..width) |d| shared[d] = rng.scale(32.0);
        for (0..width) |d| {
            values[d] = shared[d] + rng.scale(1e-4);
            values[width + d] = -shared[d] + rng.scale(1e-4);
            gates[d] = rng.scale(0.01);
            gates[width + d] = rng.scale(0.01);
        }
        try poolCase(device, queue, &p, "balanced_cancel", values, gates, try weightCells(gpa, &rng));
    }
}

/// The packed-cache code tables. Codes are generated directly, so the FP64 ideal and the kernel
/// read bit-identical BF16 keys and the stage delta isolates the attention arithmetic itself.
fn e4(code: u8) f32 {
    const magnitude = code & 127;
    if (magnitude == 127) return std.math.nan(f32);
    const exponent: i32 = @intCast(magnitude >> 3);
    const mantissa = magnitude & 7;
    const base: f32 = if (exponent == 0) @as(f32, @floatFromInt(mantissa)) / 512 else std.math.ldexp(1 + @as(f32, @floatFromInt(mantissa)) / 8, exponent - 7);
    return if ((code & 128) != 0) -base else base;
}

fn e2(code: u8) f32 {
    const levels = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const v = levels[code & 7];
    return if ((code & 8) != 0) -v else v;
}

/// One ring row: 512 FP8 codes plus the FP32 scale of each 32-column block, exactly as dsv41_unpack reads it.
fn decodeWindowRow(bytes: []const u8) [width]u16 {
    var row: [width]u16 = undefined;
    for (0..width) |d| {
        const scale: f32 = @bitCast(@as(u32, bytes[width + d / 32]) << 23);
        row[d] = math.b(e4(bytes[d]) * scale);
    }
    return row;
}

/// One compressed row: 256 FP4 nibble bytes plus the FP8 scale of each 16-column block.
fn decodeMainRow(bytes: []const u8) [width]u16 {
    var row: [width]u16 = undefined;
    for (0..width) |d| {
        const nibble: u8 = if ((d & 1) != 0) bytes[d / 2] >> 4 else bytes[d / 2] & 15;
        row[d] = math.b(e2(nibble) * e4(bytes[width / 2 + d / 16]));
    }
    return row;
}

/// The FP8 code of a value under the FP8 scale the pack kernel would choose for its block.
/// Metal's `rint`: nearest integer, ties to even. The exponent-only code rounding needs it exactly.
fn roundHalfEven(x: f32) f32 {
    const toward: f32 = @trunc(x);
    const magnitude: f32 = if (@abs(x - toward) > 0.5) toward + std.math.sign(x) else toward;
    if (@abs(x - magnitude) != 0.5) return magnitude;
    const as_int: i64 = @intFromFloat(magnitude);
    if (@mod(as_int, 2) == 0) return magnitude;
    return if (x < 0) magnitude + 1 else magnitude - 1;
}

fn e4Code(x: f32) u8 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return if (std.math.signbit(x)) 128 else 0;
    const bits: u32 = @bitCast(a);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127;
    const e = @max(unbiased, -6);
    const q: i32 = @intFromFloat(roundHalfEven(std.math.ldexp(a, 3 - @as(i32, @intCast(e)))));
    const code: u32 = if (e == -6 and q < 8) @intCast(q) else @intCast((e + 7) * 8 + q - 8);
    return @as(u8, @intCast(@min(code, 126))) | @as(u8, if (std.math.signbit(x)) 128 else 0);
}

fn e4Scale(x: f32) u8 {
    const bits: u32 = @bitCast(x);
    const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127 + @intFromBool((bits & 0x7fffff) != 0);
    return @intCast(unbiased + 127);
}

fn e2Code(x: f32) u8 {
    const a = @min(@abs(x), 6.0);
    const pick: u8 = if (a <= 0.25) 0 else if (a < 0.75) 1 else if (a <= 1.25) 2 else if (a < 1.75) 3
        else if (a <= 2.5) 4 else if (a < 3.5) 5 else if (a <= 5.0) 6 else 7;
    return pick | @as(u8, if (std.math.signbit(x)) 8 else 0);
}

/// Pack a window row exactly as dsv41_pack kind 0 would: per 32-column block, a power-of-two FP8
/// scale and FP8 codes. The decoded cells are what both the kernel and the references will read.
fn packWindowRow(values: []const f32, out: []u8) void {
    @memset(out[0..width + width / 32], 0);
    for (0..width / 32) |block| {
        var a: f32 = 1e-4;
        for (values[block * 32 ..][0..32]) |v| a = @max(a, @abs(v));
        const z = a / 448;
        const zbits: u32 = @bitCast(z);
        const scale = std.math.ldexp(@as(f32, 1.0), @as(i32, @intCast((zbits >> 23) & 255)) - 127 + @intFromBool((zbits & 0x7fffff) != 0));
        out[width + block] = e4Scale(scale);
        for (0..32) |j| out[block * 32 + j] = e4Code(values[block * 32 + j] / scale);
    }
}

/// Pack a compressed row exactly as dsv41_pack kind 1 would: per 16-column block, an FP8 scale and
/// FP4 codes; the decoded cells are what both sides read.
fn packMainRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. width / 2 + width / 16], 0);
    for (0..width / 16) |block| {
        var a: f32 = 6.0 * 0x1p-9;
        for (values[block * 16 ..][0..16]) |v| a = @max(a, @abs(v));
        const scale = e4(e4Code(@max(a, 6.0 * 0x1p-9) / 6.0));
        out[width / 2 + block] = e4Code(@max(a, 6.0 * 0x1p-9) / 6.0);
        for (0..16) |j| {
            const code = e2Code(values[block * 16 + j] / scale);
            const at = block * 8 + j / 2;
            if ((j & 1) != 0) out[at] |= code << 4 else out[at] |= code;
        }
    }
}

const SparseSpec = struct {
    name: []const u8,
    heads: usize,
    rows: usize,
    position: usize,
    ratio: usize,
    compressed_rows: usize,
    topk: usize,
};

/// The kernel's own column addressing, so the reference reads the identical key per column slot.
fn columnKey(spec: SparseSpec, row: usize, slot: usize, indices: []const i32) ?KeySource {
    const query = spec.position + row;
    const window_columns: usize = if (spec.position + row == 0) 1 else 128;
    if (slot < window_columns) {
        const pos: usize = if (query == 0) 0 else if (query + 1 < 128) (query + 1 + slot) % 128 else blk: {
            const begin = if (query + 1 > 128) query + 1 - 128 else 0;
            break :blk begin + slot;
        };
        if (pos > query) return null;
        return .{ .row = pos % 128 };
    }
    if (spec.ratio == 0) return null;
    const pos_i = indices[row * spec.topk + slot - window_columns];
    const limit = @min(spec.compressed_rows, (query + 1) / spec.ratio);
    if (pos_i < 0 or @as(usize, @intCast(pos_i)) >= limit) return null;
    return .{ .compressed = true, .row = @intCast(pos_i) };
}

const KeySource = struct { compressed: bool = false, row: usize };

fn sparseCase(device: mtl.Device, queue: mtl.Queue, p: *const csa2.Pipelines, spec: SparseSpec,
    q: []const u16, window_rows: []const []const f32, compressed_rows: []const []const f32,
    indices: []const i32, sinks: []const f32,
) !void {
    var b = Buffers{ .device = device };
    defer b.deinit();
    const q_stride = spec.heads * width;
    const q_gpu = try b.get(u16, spec.rows * q_stride);
    const ring_gpu = try b.get(u8, 128 * 528);
    const chunk_gpu = try b.get(u8, @max(spec.rows, 1) * 528);
    const main_gpu = try b.get(u8, @max(spec.compressed_rows, 1) * 288);
    const positions_gpu = try b.get(i32, spec.rows * spec.topk);
    const sink_gpu = try b.get(f32, spec.heads);
    const out_gpu = try b.get(u16, spec.rows * q_stride);
    @memcpy(q_gpu.slice(u16, spec.rows * q_stride), q);
    @memcpy(positions_gpu.slice(i32, spec.rows * spec.topk), indices);
    @memcpy(sink_gpu.slice(f32, spec.heads), sinks);

    // The reference's key table is the kernel's decoded cache: window columns by slot, then the
    // selected compressed rows in the order the index array selects them.
    var window_table = try std.testing.allocator.alloc(u16, 128 * width);
    defer std.testing.allocator.free(window_table);
    @memset(window_table, 0);
    // The reference table is indexed 128 + row, so it needs one row per compressed row.
    const compressed_cells = @max(compressed_rows.len, 1);
    var compressed_table = try std.testing.allocator.alloc(u16, compressed_cells * width);
    defer std.testing.allocator.free(compressed_table);
    @memset(compressed_table, 0);
    // Rows are laid out by ring slot: a logical position reads ring[(pos % 128)] unless it belongs to
    // the chunk this call publishes, which the kernel reads from `chunk` instead.
    for (window_rows, 0..) |values, logical| {
        const bytes = try std.testing.allocator.alloc(u8, 528);
        defer std.testing.allocator.free(bytes);
        packWindowRow(values, bytes);
        const slot = logical % 128;
        @memcpy(ring_gpu.slice(u8, 128 * 528)[slot * 528 ..][0..528], bytes);
        const decoded = decodeWindowRow(bytes);
        @memcpy(window_table[slot * width ..][0..width], decoded[0..width]);
        if (logical >= spec.position and logical < spec.position + spec.rows)
            @memcpy(chunk_gpu.slice(u8, @max(spec.rows, 1) * 528)[(logical - spec.position) * 528 ..][0..528], bytes);
    }
    for (compressed_rows, 0..) |values, row| {
        const bytes = try std.testing.allocator.alloc(u8, 288);
        defer std.testing.allocator.free(bytes);
        packMainRow(values, bytes);
        @memcpy(main_gpu.slice(u8, @max(spec.compressed_rows, 1) * 288)[row * 288 ..][0..288], bytes);
        const decoded = decodeMainRow(bytes);
        @memcpy(compressed_table[row * width ..][0..width], decoded[0..width]);
    }

    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q_gpu), view(ring_gpu), view(chunk_gpu), view(main_gpu), view(positions_gpu),
        view(sink_gpu), view(out_gpu), .{
        .rows = @intCast(spec.rows),
        .heads = @intCast(spec.heads),
        .q_stride = @intCast(q_stride),
        .position = @intCast(spec.position),
        .chunk_rows = @intCast(spec.rows),
        .ratio = @intCast(spec.ratio),
        .compressed_rows = @intCast(spec.compressed_rows),
        .topk = @intCast(spec.topk),
        .index_stride = @intCast(spec.topk),
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;
    const metal = out_gpu.slice(u16, spec.rows * q_stride);

    var column_indices = try std.testing.allocator.alloc(i32, 640);
    defer std.testing.allocator.free(column_indices);
    var wide32: [width]u16 = undefined;
    var wide64: [width]u16 = undefined;
    for (0..spec.rows) |row| {
        const window_columns: usize = if (spec.position + row == 0) 1 else 128;
        const compressed_columns: usize = if (spec.ratio == 0) 0 else @min(spec.topk, @min(spec.compressed_rows, (spec.position + row + 1) / spec.ratio));
        const count = window_columns + compressed_columns;
        for (0..count) |slot| {
            if (columnKey(spec, row, slot, indices)) |key| {
                column_indices[slot] = if (key.compressed) @intCast(128 + key.row) else @intCast(key.row);
            } else column_indices[slot] = -1;
        }
        const keys = attention_math.KeyTable{
            .window = window_table,
            .window_stride = width,
            .compressed = compressed_table,
            .compressed_stride = width,
        };
        for (0..spec.heads) |head| {
            const head_q = q[row * q_stride + head * width ..][0..width];
            var pre32: [width]f32 = undefined;
            var pre64: [width]f64 = undefined;
            attention_math.sparseHeadPre(f32, head_q, keys, column_indices[0..count], sinks[head], wide32[0..width],
                &pre32);
            attention_math.sparseHeadPre(f64, head_q, keys, column_indices[0..count], sinks[head], wide64[0..width],
                &pre64);
            var ideal: [width]f64 = undefined;
            for (wide64, 0..) |cell, d| ideal[d] = math.f(cell);
            var energy: f64 = 0;
            var peak: f64 = 0;
            var live: usize = 0;
            for (column_indices[0..count]) |at| live += @intFromBool(at >= 0);
            for (ideal) |v| {
                energy += v * v;
                peak = @max(peak, @abs(v));
            }
            std.debug.print(
                \\CSA64 {{"case":"{s}","stage":"scale","side":"{s}","columns":{d},"live":{d},"ideal_rms":{d},"ideal_peak":{d}}}
            , .{ spec.name, "sparse", count, live, @sqrt(energy / @as(f64, width)), peak });
            var stage_buffer: [32]u8 = undefined;
            const stage = std.fmt.bufPrint(&stage_buffer, "row{d}_head{d}", .{ row, head }) catch unreachable;
            // The intrinsic FP32-versus-FP64 level of the stage, before the BF16 output quantum hides
            // it. This is the rounding level the stage verdicts are measured against.
            printF32(spec.name, stage, "f32_vs_ideal_pre_rounding", deltaF32VsF64(&pre32, &pre64));
            printBf(spec.name, stage, "metal", deltaBf16(metal[row * q_stride + head * width ..][0..width], &ideal));
            printBf(spec.name, stage, "f32_reference", deltaBf16(wide32[0..width], &ideal));
            // Decisive for the stage verdict: the kernel and the source-contract reference on the
            // identical decoded rows. Anything nonzero here is the kernel, not the contract.
            var contract_cells: [width]f64 = undefined;
            for (wide32, 0..) |cell, d| contract_cells[d] = math.f(cell);
            printBf(spec.name, stage, "metal_vs_f32_reference", deltaBf16(
                metal[row * q_stride + head * width ..][0..width], &contract_cells));
        }
    }
}

const KeyPattern = enum { random, cancelling, near_tie, one_sided };

fn sparseCaseFrom(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, p: *const csa2.Pipelines,
    spec: SparseSpec, rng: *Lcg, key_amplitude: f32, scale_exponent: i32, indices_mode: usize,
    sink: f32, pattern: KeyPattern,
) !void {
    try sparseCaseScaled(gpa, device, queue, p, spec, rng, key_amplitude, scale_exponent, scale_exponent,
        indices_mode, sink, pattern);
}

/// The same rows with the window and the compressed cache on independent scale grids: the stage
/// output then carries a large and a tiny contribution in one accumulator, which is what the
/// mixing step actually rounds.
fn sparseCaseScaled(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, p: *const csa2.Pipelines,
    spec: SparseSpec, rng: *Lcg, key_amplitude: f32, window_exponent: i32, compressed_exponent: i32,
    indices_mode: usize, sink: f32, pattern: KeyPattern,
) !void {
    const q_stride = spec.heads * width;
    const q = try gpa.alloc(u16, spec.rows * q_stride);
    defer gpa.free(q);
    const q_scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(window_exponent)));
    for (q) |*cell| cell.* = math.b(rng.uniform() * q_scale);
    const logical_window = spec.position + spec.rows;
    const window_rows = try gpa.alloc([]const f32, logical_window);
    defer gpa.free(window_rows);
    for (window_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, width);
        fillPatternRow(cells, rng, i, key_amplitude, window_exponent, pattern);
        row.* = cells;
    }
    const compressed_rows = try gpa.alloc([]const f32, spec.compressed_rows);
    defer gpa.free(compressed_rows);
    for (compressed_rows, 0..) |*row, i| {
        const cells = try gpa.alloc(f32, width);
        fillPatternRow(cells, rng, i, key_amplitude, compressed_exponent, pattern);
        row.* = cells;
    }
    const indices = try gpa.alloc(i32, spec.rows * spec.topk);
    defer gpa.free(indices);
    const limit = @max(spec.compressed_rows, 1);
    for (0..spec.rows) |r| for (0..spec.topk) |k| {
        indices[r * spec.topk + k] = switch (indices_mode) {
            0 => @intCast(k * limit / @max(spec.topk, 1)), // spread and ascending
            1 => @intCast((limit - 1) - k), // descending (the reference sorts; the kernel does not)
            else => @as(i32, @intCast((r *% 7 +% 13 * k) % limit)), // arbitrary order, unsorted
        };
    };
    const sinks = try gpa.alloc(f32, spec.heads);
    defer gpa.free(sinks);
    for (sinks) |*v| v.* = sink;
    try sparseCase(device, queue, p, spec, q, window_rows, compressed_rows, indices, sinks);
}

fn fillPatternRow(row: []f32, rng: *Lcg, index: usize, amplitude: f32, exponent: i32, pattern: KeyPattern) void {
    const scale = std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)));
    if (pattern == .random) {
        for (row) |*v| v.* = rng.uniform() * amplitude * scale;
        return;
    }
    for (row, 0..) |*v, d| {
        if (pattern == .cancelling) {
            // Same shape in every key: all scores tie, and the value sum cancels to the ULPs of
            // the large terms that make it up.
            const flip: f32 = if ((d & 1) == 0) 1 else -1;
            v.* = flip * amplitude * scale + rng.uniform() * amplitude * scale * 1e-3;
        } else if (pattern == .near_tie) {
            // Keys differ only far above the score rounding boundary: probabilities tie at the top.
            v.* = @as(f32, @floatFromInt(index)) * amplitude * scale + rng.uniform() * amplitude * scale * 1e-7;
        } else {
            // One dominant key against keys orders of magnitude smaller.
            v.* = rng.uniform() * amplitude * scale * @as(f32, if (index == 0) @as(f32, 1e4) else @as(f32, 1e-4));
        }
    }
}

test "sparse attention stages sit at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();

    var rng = Lcg{ .state = 0x2545f4914f6cdd1d };

    // Window only, the decode-step boundary at a wrapped ring.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "window_decode", .heads = 4, .rows = 1,
        .position = 255, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Compressed rows with a small selection, ascending indices.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "compressed_asc", .heads = 4, .rows = 1,
        .position = 255, .ratio = 2, .compressed_rows = 128, .topk = 16 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Unsorted indices: the kernel consumes the array in the given order; the reference must too.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "compressed_unsorted", .heads = 2, .rows = 1,
        .position = 9, .ratio = 2, .compressed_rows = 40, .topk = 12 }, &rng, 1.0, 0, 2, 0, KeyPattern.random);
    // Prefill rows, multi-row chunk with a wrapped ring.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "prefill_rows", .heads = 2, .rows = 4,
        .position = 250, .ratio = 2, .compressed_rows = 100, .topk = 8 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
    // Wide dynamic range keys: the score sum runs across many orders of magnitude.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "wide_range", .heads = 1, .rows = 1,
        .position = 200, .ratio = 2, .compressed_rows = 32, .topk = 8 }, &rng, 1.0, -20, 0, 0, KeyPattern.random);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "wide_range_hi", .heads = 1, .rows = 1,
        .position = 200, .ratio = 2, .compressed_rows = 32, .topk = 8 }, &rng, 1.0, 20, 0, 0, KeyPattern.random);
    // Denormal-scale sink: the sink mass underflows FP32 while it is exact in FP64.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "sink_denormal", .heads = 1, .rows = 1,
        .position = 100, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, -16.0, KeyPattern.random);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "sink_large", .heads = 1, .rows = 1,
        .position = 100, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 12.0, KeyPattern.random);
    // Value-product cancellation: identical-shaped keys, so every score ties and the P·V sum
    // cancels to the ULPs of the terms that make it up.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_values", .heads = 1, .rows = 1,
        .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 0, 0, 0,
        KeyPattern.cancelling);
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "cancel_wide", .heads = 1, .rows = 1,
        .position = 300, .ratio = 2, .compressed_rows = 64, .topk = 64 }, &rng, 32.0, 6, 0, 0,
        KeyPattern.cancelling);
    // Scores that tie: the softmax denominator is the sum of equal exponentials, and the
    // accumulation order decides which probability lands on which side of its rounding boundary.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "score_tie", .heads = 1, .rows = 1,
        .position = 300, .ratio = 2, .compressed_rows = 32, .topk = 32 }, &rng, 1.0, 0, 0, 0,
        KeyPattern.near_tie);
    // One dominant key against keys orders of magnitude smaller: denominator and accumulator
    // run at the extreme ratio.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "one_dominant", .heads = 1, .rows = 1,
        .position = 300, .ratio = 2, .compressed_rows = 16, .topk = 16 }, &rng, 1.0, 0, 0, 0,
        KeyPattern.one_sided);
    // Many tiles, so the online correction multiplies the accumulator once per 64 columns.
    try sparseCaseFrom(gpa, device, queue, &p, .{ .name = "many_tiles", .heads = 1, .rows = 1,
        .position = 511, .ratio = 2, .compressed_rows = 512, .topk = 384 }, &rng, 1.0, 0, 0, 0,
        KeyPattern.random);
    // Cancellation in the value product: keys of alternating sign, so the P·V sum is small next
    // to the terms that make it up.
    {
        const spec = SparseSpec{ .name = "pv_cancellation", .heads = 1, .rows = 1, .position = 64,
            .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows) |*row| {
            const cells = try gpa.alloc(f32, width);
            for (cells, 0..) |*v, d| v.* = if ((d & 1) == 0) rng.scale(64.0) else rng.scale(64.0) * 0.987654321;
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &p, spec, q, window_rows, empty, indices, sinks);
    }
    // Large cancelling keys beside small ones: score dot products run through near-cancelling terms.
    {
        const spec = SparseSpec{ .name = "score_cancellation", .heads = 1, .rows = 1, .position = 96,
            .ratio = 0, .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        for (q, 0..) |*cell, d| cell.* = if ((d & 1) == 0) math.b(1.0) else math.b(-1.0);
        const window_rows = try gpa.alloc([]const f32, spec.position + spec.rows);
        defer gpa.free(window_rows);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, width);
            const bias: f32 = @as(f32, @floatFromInt(@rem(@as(i32, @intCast(i)), 8))) * 4.0;
            for (cells, 0..) |*v, d| v.* = bias + (if ((d & 1) == 0) rng.scale(0.001) else rng.scale(0.001) * 0.99999);
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &p, spec, q, window_rows, empty, indices, sinks);
    }
}

/// One live column pins the running maximum to a single known score, so `sink - max` is the whole
/// normalization. Sweeping it across the FP32 exp() overflow and underflow limits is the only path
/// by which the sink term can move the stage output, and it is also where an FP32 exp() would leave
/// the FP64 equation behind.
fn sinkCase(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, p: *const csa2.Pipelines,
    name: []const u8, rng: *Lcg, sink: f32,
) !void {
    const spec = SparseSpec{ .name = name, .heads = 1, .rows = 1, .position = 0, .ratio = 0,
        .compressed_rows = 0, .topk = 0 };
    const q = try gpa.alloc(u16, width);
    defer gpa.free(q);
    for (q) |*cell| cell.* = math.b(rng.scale(1.0));
    const window_rows = try gpa.alloc([]const f32, 1);
    defer gpa.free(window_rows);
    const cells = try gpa.alloc(f32, width);
    defer gpa.free(cells);
    for (cells) |*v| v.* = rng.scale(1.0);
    window_rows[0] = cells;
    const sinks = try gpa.alloc(f32, 1);
    defer gpa.free(sinks);
    sinks[0] = sink;
    const empty = try gpa.alloc([]const f32, 0);
    defer gpa.free(empty);
    const indices = try gpa.alloc(i32, 0);
    defer gpa.free(indices);
    try sparseCase(device, queue, p, spec, q, window_rows, empty, indices, sinks);
}

test "sink normalization sits at rounding level across the exp overflow boundaries" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x51c0de51c0de51c0 };

    // One case per boundary FP32 exp() has: underflow to zero, the overflow edge, and the
    // mass-balanced middle where the sink decides the output on its own. From about 83 upward the
    // sink exponent reaches the FP32 overflow boundary, so the cells that differ from the reference
    // sit near 1e-38 and a one-ULP difference in the running maximum decides whether either side
    // sees a finite denominator at all.
    const sweep = [_]struct { sink: f32, name: []const u8 }{
        .{ .sink = -104, .name = "sink_underflow" },
        .{ .sink = -90, .name = "sink_near_underflow" },
        .{ .sink = -60, .name = "sink_tiny" },
        .{ .sink = -12, .name = "sink_small" },
        .{ .sink = -1, .name = "sink_minus_one" },
        .{ .sink = 0, .name = "sink_zero" },
        .{ .sink = 1, .name = "sink_one" },
        .{ .sink = 12, .name = "sink_medium" },
        .{ .sink = 60, .name = "sink_large" },
        .{ .sink = 80, .name = "sink_edge80" },
        .{ .sink = 83, .name = "sink_edge83" },
        .{ .sink = 84, .name = "sink_edge84" },
        .{ .sink = 85, .name = "sink_edge85" },
        .{ .sink = 86, .name = "sink_over_edge" },
        .{ .sink = 90, .name = "sink_overflow" },
    };
    for (sweep) |entry| try sinkCase(gpa, device, queue, &pipelines, entry.name, &rng, entry.sink);

    // Per-head sinks: the kernel indexes the sink array by head, so heads whose sink masses differ
    // by orders of magnitude check that indexing as well as the arithmetic.
    {
        const spec = SparseSpec{ .name = "sink_per_head", .heads = 4, .rows = 1, .position = 0,
            .ratio = 0, .compressed_rows = 0, .topk = 0 };
        var head_rng = Lcg{ .state = 0xabcdef0123456789 };
        const q = try gpa.alloc(u16, spec.heads * width);
        defer gpa.free(q);
        for (q) |*cell| cell.* = math.b(head_rng.scale(1.0));
        const window_rows = try gpa.alloc([]const f32, 1);
        defer gpa.free(window_rows);
        const cells = try gpa.alloc(f32, width);
        defer gpa.free(cells);
        for (cells) |*v| v.* = head_rng.scale(1.0);
        window_rows[0] = cells;
        const sinks = try gpa.alloc(f32, spec.heads);
        defer gpa.free(sinks);
        for (0..spec.heads) |h| sinks[h] = @as(f32, @floatFromInt(@as(i32, @intCast(h)) * 24 - 36));
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}

// Tile denominators: the online correction, the per-tile sum and the running maximum. A flat
// score set makes the denominator a pure sum of equal exponentials; a ramped score set drives the
// correction through the FP32 underflow limit; a descending set never applies a correction at all.
test "tile denominators sit at rounding level for every tile shape" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x1a2b3c4d5e6f7081 };

    // Ten tiles, so the denominator accumulates 639 exponentials and the accumulator takes 639
    // products. Identical key shapes make every score tie: the denominator is exactly the count.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_equal_scores", .heads = 1, .rows = 1,
        .position = 1022, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 0, 0, KeyPattern.cancelling);
    // Ascending scores by tile: the correction falls one power of two per step until FP32 flushes
    // it, and the dropped mass is the whole question.
    for ([_]i32{ -2, 0, 1, 2 }) |step| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "tiles_ramp{d}", .{step}) catch unreachable;
        try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = label, .heads = 1, .rows = 1,
            .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, step, 0, 0,
            KeyPattern.near_tie);
    }
    // Descending scores: the running maximum never moves, so every tile lands at full weight and
    // the denominator is a plain sum.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_descending", .heads = 1, .rows = 1,
        .position = 1000, .ratio = 2, .compressed_rows = 512, .topk = 512 }, &rng, 1.0, 0, 1, 0,
        KeyPattern.near_tie);
    // One tile only, the boundary case of the online loop: the correction is applied zero times.
    try sparseCaseFrom(gpa, device, queue, &pipelines, .{ .name = "tiles_single", .heads = 1, .rows = 1,
        .position = 20, .ratio = 0, .compressed_rows = 0, .topk = 0 }, &rng, 1.0, 0, 0, 0, KeyPattern.random);
}

// The output mix: window rows and compressed rows meeting in one accumulator on different scale
// grids, and two mirror rows whose shared large shape cancels out of the weighted sum. The mirror
// case is the stage's real amplification probe - whatever the score arithmetic gets wrong shows up
// multiplied by the shape that the sum is cancelling.
test "the output mix sits at rounding level when window and compressed rows cancel" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var pipelines = try csa2.Pipelines.load(device);
    defer pipelines.deinit();
    var rng = Lcg{ .state = 0x77a1c0de_5b5e1a2b };

    // Large window rows against tiny compressed rows, then the reverse: one accumulator holding
    // terms six orders of magnitude apart.
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_window_dominates", .heads = 1, .rows = 1,
        .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 14, -12, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_compressed_dominates", .heads = 1, .rows = 1,
        .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, -12, 14, 0, 0, KeyPattern.random);
    try sparseCaseScaled(gpa, device, queue, &pipelines, .{ .name = "mix_balanced_magnitudes", .heads = 1, .rows = 1,
        .position = 100, .ratio = 2, .compressed_rows = 100, .topk = 64 }, &rng, 1.0, 6, 6, 0, 0,
        KeyPattern.cancelling);

    // The stage's real amplification probe: 128 window rows that carry a large alternating shape
    // which cancels out of both the score (a uniform query against a sign-alternating shape) and
    // the weighted sum, so every accumulator term is large and their total is small. Whatever the
    // arithmetic gets wrong arrives multiplied by that shape.
    for ([_]i32{ 8, 16, 23 }) |shape_exponent| {
        var name_buf: [24]u8 = undefined;
        const label = std.fmt.bufPrint(&name_buf, "mirror_shape_e{d}", .{shape_exponent}) catch unreachable;
        const spec = SparseSpec{ .name = label, .heads = 1, .rows = 1, .position = 127, .ratio = 0,
            .compressed_rows = 0, .topk = 0 };
        const q = try gpa.alloc(u16, width);
        defer gpa.free(q);
        // q dot shape == 0 exactly: a uniform query against a shape that alternates sign per
        // column, so the large part drops out of both scores and the accumulator alike.
        for (q) |*cell| cell.* = math.b(1.0);
        const window_rows = try gpa.alloc([]const f32, 128);
        defer gpa.free(window_rows);
        const shape = std.math.pow(f32, 2, @floatFromInt(shape_exponent));
        const shared = try gpa.alloc(f32, width);
        defer gpa.free(shared);
        // One 64th of the cancelling shape: near the FP8 window grid's floor, so the accumulator's
        // partial sums run 2^16 times the sum they end at. Below that the code grid cancels the
        // small part away and FP32 needs no help to be exact.
        for (shared) |*v| v.* = rng.scale(shape * 0x1p-16);
        for (window_rows, 0..) |*row, i| {
            const cells = try gpa.alloc(f32, width);
            const side: f32 = if ((i & 1) == 0) 1 else -1;
            for (0..width) |d| {
                const flip: f32 = if ((d & 1) == 0) 1 else -1;
                cells[d] = side * flip * shape + shared[d];
            }
            row.* = cells;
        }
        const sinks = try gpa.alloc(f32, 1);
        defer gpa.free(sinks);
        sinks[0] = 0;
        const empty = try gpa.alloc([]const f32, 0);
        defer gpa.free(empty);
        const indices = try gpa.alloc(i32, 0);
        defer gpa.free(indices);
        try sparseCase(device, queue, &pipelines, spec, q, window_rows, empty, indices, sinks);
    }
}

/// Both sides of one rotary call: `contract` is the pinned source's FP32 path (FP32 frequency, FP32
/// angle, FP32 trig), `ideal` keeps frequency, angle, both unaries and the products in FP64. The
/// kernel differs from the contract by the factor table's own unaries, which is what this checks.
fn ropeFrequency64(index: usize, theta: f64, original: f64, factor: f64, beta_fast: f64,
    beta_slow: f64,
) f64 {
    const frequency: f64 = 1 / std.math.pow(f64, theta, @as(f64, @floatFromInt(index * 2)) / 64);
    if (original <= 0) return frequency;
    const denominator = 2 * @log(theta);
    const low = @max(@floor(64 * @log(original / (beta_fast * 2 * std.math.pi)) / denominator), 0);
    const high = @min(@ceil(64 * @log(original / (beta_slow * 2 * std.math.pi)) / denominator), 63);
    const ramp = std.math.clamp((@as(f64, @floatFromInt(index)) - low) / @max(high - low, 1e-3), 0, 1);
    return frequency / factor * ramp + frequency * (1 - ramp);
}

fn ropeCase(device: mtl.Device, queue: mtl.Queue, p: *const tf.deepseek_v41.rope.Pipelines,
    name: []const u8, rng: *Lcg, position: usize, rows: usize, heads: usize,
    spectrum: tf.deepseek_v41.rope.Spectrum, inverse: bool, theta: f32, original: u32, factor: f32,
) !void {
    const stride = heads * width;
    const cells = rows * stride;
    var b = Buffers{ .device = device };
    defer b.deinit();
    const x_gpu = try b.get(u16, cells);
    const x = x_gpu.slice(u16, cells);
    const before = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(before);
    for (x, 0..) |*cell, i| {
        cell.* = math.b(rng.scale(1.0));
        before[i] = cell.*;
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(x_gpu), .{ .rows = @intCast(rows), .heads = @intCast(heads),
        .width = width, .rotary = 64, .stride = @intCast(stride), .position = @intCast(position),
        .step = 1, .inverse = @intFromBool(inverse), .spectrum = spectrum });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;

    const theta64: f64 = @as(f64, theta);
    const original64: f64 = if (original == 0) 0 else @floatFromInt(original);
    const ideal_cells = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(ideal_cells);
    const contract_cells = try std.testing.allocator.alloc(u16, cells);
    defer std.testing.allocator.free(contract_cells);
    for (0..rows) |row| for (0..heads) |head| {
        const base = row * stride + head * width;
        // Everything outside the rotary window is copied through, so both sides must agree with the
        // input there too.
        for (0..width - 64) |d| {
            ideal_cells[base + d] = before[base + d];
            contract_cells[base + d] = before[base + d];
        }
        for (0..32) |i| {
            const at = base + width - 64 + 2 * i;
            const a = math.f(before[at]);
            const z = math.f(before[at + 1]);
            const frequency = math.ropeFrequency(i, theta, original, factor, 32, 1);
            const angle: f32 = @as(f32, @floatFromInt(position + row)) * frequency;
            const cosine32 = @cos(angle);
            const sine32 = @sin(angle) * (if (inverse) @as(f32, -1) else 1);
            contract_cells[at] = math.b(a * cosine32 - z * sine32);
            contract_cells[at + 1] = math.b(a * sine32 + z * cosine32);
            const frequency64 = ropeFrequency64(i, theta64, original64, @as(f64, factor), 32, 1);
            const angle64: f64 = @as(f64, @floatFromInt(position + row)) * frequency64;
            const cosine64 = @cos(angle64);
            const sine64 = @sin(angle64) * (if (inverse) @as(f64, -1) else @as(f64, 1));
            ideal_cells[at] = math.b64(@as(f64, a) * cosine64 - @as(f64, z) * sine64);
            ideal_cells[at + 1] = math.b64(@as(f64, a) * sine64 + @as(f64, z) * cosine64);
        }
    };

    var stage_buffer: [32]u8 = undefined;
    for (0..rows) |row| for (0..heads) |head| {
        const stage = std.fmt.bufPrint(&stage_buffer, "row{d}_head{d}", .{ row, head }) catch unreachable;
        const base = row * stride + head * width;
        var ideal: [width]f64 = undefined;
        var contract: [width]f64 = undefined;
        for (0..width) |d| {
            ideal[d] = math.f(ideal_cells[base + d]);
            contract[d] = math.f(contract_cells[base + d]);
        }
        printBf(name, stage, "metal", deltaBf16(x[base..][0..width], &ideal));
        printBf(name, stage, "f32_reference", deltaBf16(contract_cells[base..][0..width], &ideal));
        printBf(name, stage, "metal_vs_f32_reference", deltaBf16(x[base..][0..width], &contract));
    };
}

test "the rotary stage sits at rounding level for both spectra and at long positions" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    // The production shape: 64 rotary columns, YaRN on the compressed spectrum only, positions out
    // to the resident capacity.
    var p = try tf.deepseek_v41.rope.Pipelines.load(device, 16384, 64,
        .{ .theta = 10000, .original = 0 },
        .{ .theta = 10000, .factor = 16, .original = 65536, .beta_fast = 32, .beta_slow = 1 });
    defer p.deinit();

    const positions = [_]usize{ 1, 7, 63, 64, 4095, 8191, 16383 };
    for (positions) |position| {
        var name_buf: [32]u8 = undefined;
        var scoped = Lcg{ .state = 0x9e3779b97f4a7c15 ^ @as(u64, position) };
        const label = std.fmt.bufPrint(&name_buf, "rope_normal_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label, &scoped, position, 1, 2, .normal, false, 10000, 0, 16);
        var scoped2 = Lcg{ .state = 0x85ebca77c0b98731 ^ @as(u64, position) };
        const label2 = std.fmt.bufPrint(&name_buf, "rope_compressed_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label2, &scoped2, position, 1, 2, .compressed, false, 10000, 65536, 16);
        var scoped3 = Lcg{ .state = 0xc2b2ae3d27d4eb4f ^ @as(u64, position) };
        const label3 = std.fmt.bufPrint(&name_buf, "rope_inverse_p{d}", .{position}) catch unreachable;
        try ropeCase(device, queue, &p, label3, &scoped3, position, 1, 1, .compressed, true, 10000, 65536, 16);
    }
    // A prefill row set: the kernel indexes the factor table by absolute position per row.
    var multi = Lcg{ .state = 0x2545f4914f6cdd1d };
    try ropeCase(device, queue, &p, "rope_prefill_rows", &multi, 8100, 4, 2, .compressed, false, 10000, 65536, 16);
}

/// One indexer-score row set: 32 heads of 128 FP4-index columns each, scored against packed keys.
const IndexSpec = struct { name: []const u8, rows: usize, keys: usize, topk: usize };

fn packIndexRow(values: []const f32, out: []u8) void {
    @memset(out[0 .. 128 / 2 + 128 / 32], 0);
    for (0..128 / 32) |block| {
        var a: f32 = 6.0 * 0x1p-126;
        for (values[block * 32 ..][0..32]) |v| a = @max(a, @abs(v));
        const bits: u32 = @bitCast(a / 6.0);
        const unbiased: i32 = @as(i32, @intCast((bits >> 23) & 255)) - 127 + @intFromBool((bits & 0x7fffff) != 0);
        const scale = std.math.ldexp(@as(f32, 1.0), unbiased);
        out[64 + block] = @intCast(unbiased + 127);
        for (0..32) |j| {
            const code = e2Code(values[block * 32 + j] / scale);
            const at = block * 16 + j / 2;
            if ((j & 1) != 0) out[at] |= code << 4 else out[at] |= code;
        }
    }
}

fn decodeIndexRow(bytes: []const u8) [128]u16 {
    var row: [128]u16 = undefined;
    for (0..128) |d| {
        const nibble: u8 = if ((d & 1) != 0) bytes[d / 2] >> 4 else bytes[d / 2] & 15;
        const scale: f32 = @bitCast(@as(u32, bytes[64 + d / 32]) << 23);
        row[d] = math.b(e2(nibble) * scale);
    }
    return row;
}

fn indexCase(device: mtl.Device, queue: mtl.Queue, p: *const tf.deepseek_v41.csa2_index.Pipelines,
    spec: IndexSpec, q: []const u16, weights: []const u16, key_values: []const []const f32,
) !void {
    var b = Buffers{ .device = device };
    defer b.deinit();
    var scratch = try tf.deepseek_v41.csa2_index.Scratch.init(device, @intCast(spec.rows));
    defer scratch.deinit();
    const q_gpu = try b.get(u16, spec.rows * 32 * 128);
    const weights_gpu = try b.get(u16, spec.rows * 32);
    const keys_gpu = try b.get(u8, spec.keys * 68);
    @memcpy(q_gpu.slice(u16, spec.rows * 32 * 128), q);
    @memcpy(weights_gpu.slice(u16, spec.rows * 32), weights);

    // Both sides read the same decoded keys; the references differ only in their arithmetic.
    const decoded = try std.testing.allocator.alloc(u16, spec.keys * 128);
    defer std.testing.allocator.free(decoded);
    for (key_values, 0..) |values, row| {
        const bytes = try std.testing.allocator.alloc(u8, 68);
        defer std.testing.allocator.free(bytes);
        packIndexRow(values, bytes);
        @memcpy(keys_gpu.slice(u8, spec.keys * 68)[row * 68 ..][0..68], bytes);
        const decoded_row = decodeIndexRow(bytes);
        @memcpy(decoded[row * 128 ..][0..128], &decoded_row);
    }
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    try p.encode(enc, view(q_gpu), view(weights_gpu), view(keys_gpu), scratch, .{
        .rows = @intCast(spec.rows),
        .keys = @intCast(spec.keys),
        .q_stride = 32 * 128,
        .weight_stride = 32,
        .position = @intCast(spec.keys),
        .ratio = 1,
    });
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.Csa2Fp64GpuFailure;

    // Positions live behind the scores/heap/candidates regions; read them through the view's offset,
    // otherwise this compares the score words against themselves and the check is vacuous.
    const kernel_scores = scratch.scores().buffer.slice(f32, spec.rows * 4096);
    const positions_offset = scratch.positions().offset / @sizeOf(i32);
    const kernel_positions = scratch.positions().buffer.slice(i32, positions_offset + spec.rows * 512);
    var ref32 = try std.testing.allocator.alloc(f32, spec.keys);
    defer std.testing.allocator.free(ref32);
    var ref64 = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(ref64);
    // The same scores with only the documented quantizations below the final one: the BF16 quantum
    // that no change of arithmetic can remove.
    var exact = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(exact);
    const w_scale: f32 = 1 / @sqrt(@as(f32, 128 * 32));
    for (0..spec.keys) |row| {
        const key = decoded[row * 128 ..][0..128];
        var ideal: f64 = 0;
        var narrow32: f32 = 0;
        for (0..32) |head| {
            var dot: f64 = 0;
            for (0..128) |d| dot += @as(f64, math.f(key[d])) * @as(f64, math.f(q[head * 128 + d]));
            // The pinned source's head einsum lands in BF16, so the rectified score is a quantized
            // algorithm boundary like the cache codes are; only the reduction above it is widened.
            const rectified: f64 = @max(math.f(math.b(@min(@as(f32, @floatCast(dot)), 1e38))), 0);
            const weight: f32 = math.f(weights[head]);
            const scaled_weight = math.f(math.b(weight * w_scale));
            const weighted = math.f(math.b(@as(f32, @floatCast(rectified)) * scaled_weight));
            ideal += weighted;
            narrow32 += @as(f32, @floatCast(weighted));
        }
        exact[row] = ideal;
        // The score is written back through BF16 by the kernel and by the source contract, so the
        // ideal carries that final boundary too: the stage then differs from its output by nothing
        // but the arithmetic being compared.
        ref64[row] = math.f(math.b64(ideal));
        ref32[row] = math.f(math.b(narrow32));
    }
    var ideal_cells: [1024]f64 = undefined;
    const shown = @min(spec.keys, ideal_cells.len);
    for (0..shown) |row| ideal_cells[row] = ref64[row];
    printF32(spec.name, "index_score", "bf16_score_boundary", deltaF32VsF64(ref32[0..shown], exact[0..shown]));
    printF32(spec.name, "index_score", "metal_vs_ideal", deltaF32VsF64(
        kernel_scores[0..shown], ideal_cells[0..shown]));
    printF32(spec.name, "index_score", "f32_vs_ideal", deltaF32VsF64(ref32[0..shown], ideal_cells[0..shown]));
    // The kernel against the source-contract FP32 reference on identical keys: the stage delta that
    // selection can actually see, independent of how far the reference itself sits from the ideal.
    var contract: [1024]f64 = undefined;
    for (0..shown) |row| contract[row] = @as(f64, ref32[row]);
    printF32(spec.name, "index_score", "metal_vs_f32_reference", deltaF32VsF64(
        kernel_scores[0..shown], contract[0..shown]));

    // Selection: which keys land in the top-k. Both sides break ties by the lowest position, the
    // kernel's heap comparator and the reference's strict-greater scan agree on that.
    var chosen = try std.testing.allocator.alloc(bool, spec.keys);
    defer std.testing.allocator.free(chosen);
    @memset(chosen, false);
    var scores_copy = try std.testing.allocator.alloc(f32, spec.keys);
    defer std.testing.allocator.free(scores_copy);
    @memcpy(scores_copy, ref32[0..spec.keys]);
    var reference_picks: [512]u32 = undefined;
    const picks = @min(spec.topk, spec.keys);
    for (0..picks) |pick| {
        var best: usize = 0;
        for (0..spec.keys) |i| {
            if (scores_copy[i] > scores_copy[best]) best = i;
        }
        reference_picks[pick] = @intCast(best);
        scores_copy[best] = -std.math.inf(f32);
    }
    std.mem.sort(u32, reference_picks[0..picks], {}, std.sort.asc(u32));
    var overlap: usize = 0;
    for (reference_picks[0..picks]) |at| chosen[at] = true;
    var min_selected: f32 = std.math.inf(f32);
    var max_unselected: f32 = -std.math.inf(f32);
    for (0..spec.keys) |row| {
        if (chosen[row]) {
            overlap += 1;
            min_selected = @min(min_selected, ref32[row]);
        } else max_unselected = @max(max_unselected, ref32[row]);
    }
    var kernel_only: usize = 0;
    var out_of_range: usize = 0;
    for (0..picks) |i| {
        const slot = kernel_positions[positions_offset + i];
        if (slot < 0 or slot >= @as(i32, @intCast(spec.keys))) {
            out_of_range += 1;
            continue;
        }
        if (!chosen[@intCast(slot)]) kernel_only += 1;
    }
    // How much arithmetic noise the selection can absorb before the *set* changes. The score is only
    // worth what the storage keeps, so a pick is knife-edged when the boundary margin is comparable
    // to one score quantum; if it is not, no 1-ULP arithmetic difference can flip this row.
    var exact_picks: [512]u32 = undefined;
    var exact_scores = try std.testing.allocator.alloc(f64, spec.keys);
    defer std.testing.allocator.free(exact_scores);
    @memcpy(exact_scores, exact[0..spec.keys]);
    for (0..picks) |pick| {
        var best: usize = 0;
        for (0..spec.keys) |i| {
            if (exact_scores[i] > exact_scores[best]) best = i;
        }
        exact_picks[pick] = @intCast(best);
        exact_scores[best] = -std.math.inf(f64);
    }
    var flips: usize = 0;
    var exact_chosen = try std.testing.allocator.alloc(bool, spec.keys);
    defer std.testing.allocator.free(exact_chosen);
    @memset(exact_chosen, false);
    for (exact_picks[0..picks]) |at| exact_chosen[at] = true;
    for (0..spec.keys) |at| {
        if (chosen[at] != exact_chosen[at]) flips += 1;
    }
    var quantum: f32 = 0;
    var margin_in_quanta: f64 = 0;
    if (min_selected != std.math.inf(f32) and max_unselected != -std.math.inf(f32)) {
        const boundary = min_selected * 0.5 + max_unselected * 0.5;
        const exponent: i32 = if (@abs(boundary) > 0) std.math.ilogb(@abs(boundary)) - 7 else -149;
        quantum = std.math.pow(f32, 2.0, @as(f32, @floatFromInt(exponent)));
        margin_in_quanta = @as(f64, min_selected - max_unselected) / @as(f64, quantum);
    }
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"index_select","side":"noise_tolerance","keys":{d},"picks":{d},"precision_flips":{d},"selection_margin":{d},"score_quantum":{d},"margin_in_quanta":{d:.1}}}
    , .{ spec.name, spec.keys, picks, flips, min_selected - max_unselected, quantum, margin_in_quanta });
    std.debug.print(
        \\CSA64 {{"case":"{s}","stage":"index_select","side":"metal","keys":{d},"picks":{d},"reference_only":{d},"kernel_only":{d},"invalid_positions":{d},"lowest_selected":{d},"highest_unselected":{d}}}
    , .{ spec.name, spec.keys, picks, picks - overlap, kernel_only, out_of_range, min_selected, max_unselected });
}

test "indexer scoring and selection sit at rounding level against the FP64 ideal" {
    var arena_instance = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_instance.deinit();
    const gpa = arena_instance.allocator();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try tf.deepseek_v41.csa2_index.Pipelines.load(device);
    defer p.deinit();

    var rng = Lcg{ .state = 0x1234_5678_9abc_def0 };
    var amplitude: f32 = 1;
    for ([_]IndexSpec{
        .{ .name = "index_random", .rows = 1, .keys = 1024, .topk = 512 },
        .{ .name = "index_random_4k", .rows = 1, .keys = 4096, .topk = 512 },
        .{ .name = "index_ties", .rows = 1, .keys = 1024, .topk = 512 },
        .{ .name = "index_boundary", .rows = 1, .keys = 1024, .topk = 512 },
    }, 0..) |spec, case_index| {
        _ = case_index;
        const q = try gpa.alloc(u16, 32 * 128);
        defer gpa.free(q);
        const weights = try gpa.alloc(u16, 32);
        defer gpa.free(weights);
        const cells = try gpa.alloc(f32, spec.keys * 128);
        defer gpa.free(cells);
        const keys = try gpa.alloc([]const f32, spec.keys);
        defer gpa.free(keys);
        for (q) |*cell| cell.* = math.b(rng.scale(1.0));
        for (weights) |*cell| cell.* = math.b(rng.scale(1.0));
        for (cells, 0..) |*v, i| {
            v.* = rng.scale(1.0);
            keys[i / 128] = cells[i / 128 * 128 ..][0..128];
        }
        // Identical keys: every score is the same value, so the whole selection is tie-driven.
        if (std.mem.eql(u8, spec.name, "index_ties")) {
            for (cells[128..], cells[0..128]) |*dst, src| dst.* = src;
        }
        // A score cliff just inside the selection boundary: half the keys sit on the boundary.
        if (std.mem.eql(u8, spec.name, "index_boundary")) {
            for (cells, 0..) |*v, i| {
                if ((i % 128) == 0) amplitude = (if (i / 128 < 512) @as(f32, 1.000001) else @as(f32, 1.0));
                v.* = rng.scale(amplitude);
            }
        }
        try indexCase(device, queue, &p, spec, q, weights, keys);
    }
}

fn weightCells(gpa: std.mem.Allocator, rng: *Lcg) ![]u16 {
    const out = try gpa.alloc(u16, width);
    for (out) |*cell| {
        const amplitude: f32 = 0.5 + 0.25 * @as(f32, @floatFromInt(rng.next() % 1000)) / 1000;
        cell.* = math.b(amplitude);
    }
    return out;
}
