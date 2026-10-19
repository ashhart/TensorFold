//! Hyperconnection retained comparisons against the FP64 ideal.
const support = @import("hc_fp64_fixture.zig");
const std = support.std;
const mtl = support.mtl;
const math = support.math;
const mhc = support.mhc;
const ops = support.ops;
const Hc = support.Hc;
const Config = support.Config;
const width = support.width;
const copies = support.copies;
const stream_width = support.stream_width;
const mix_count = support.mix_count;
const released_config = support.released_config;
const fixture_path = support.fixture_path;
const Fixture = support.Fixture;
const decodeU16 = support.decodeU16;
const decodeF32 = support.decodeF32;
const Lcg = support.Lcg;
const upload = support.upload;
const runCase = support.runCase;
const syntheticWeight = support.syntheticWeight;
const fourCopies = support.fourCopies;
const mixesLegacy = support.mixesLegacy;

test "mHC fp64 rounding level and Metal deltas on retained streams" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try mhc.Pipelines.load(device);
    defer p.deinit();
    var o = try ops.Pipelines.load(device);
    defer o.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);

    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_stream = try decodeU16(gpa, fixture.value.ffn_stream_bf16);
    const ffn_branch = try decodeU16(gpa, fixture.value.ffn_branch_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const token671 = eight_rows[1 * width ..][0..width]; // hc_attention_rms.json provenance: token 671.

    const weight = try syntheticWeight(gpa);

    // Token 671 expands to four streams; all sides share retained branch data and real FFN scale/base.
    {
        const stream = try fourCopies(gpa, token671);
        const branch = try gpa.alloc(u16, width);
        @memcpy(branch, ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "token671_attention", .rows = 1, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    {
        const branch = try gpa.alloc(u16, width);
        @memcpy(branch, ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "token1_ffn", .rows = 1, .stream = ffn_stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    {
        const rows = 8;
        const stream = try gpa.alloc(u16, rows * stream_width);
        for (0..rows) |r| {
            const row = eight_rows[r * width ..][0..width];
            for (0..copies) |c| @memcpy(stream[r * stream_width + c * width ..][0..width], row);
        }
        const branch = try gpa.alloc(u16, rows * width);
        for (0..rows) |r| @memcpy(branch[r * width ..][0..width], ffn_branch);
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "eight_teacher_rows", .rows = rows, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
    // Wide dynamic range: copy magnitudes 2^24..2^-12 with rotating signs, branch 2^8..2^-4.
    {
        const stream = try gpa.alloc(u16, stream_width);
        const exponents = [_]i32{ 24, 12, 0, -12 };
        for (0..copies) |c| {
            for (0..width) |d| {
                const sign: f32 = if ((d + c) % 2 == 0) 1 else -1;
                stream[c * width + d] = math.b(sign * @as(f32, @floatCast(std.math.pow(f64, 2, @floatFromInt(exponents[c])))));
            }
        }
        const branch = try gpa.alloc(u16, width);
        const branch_exponents = [_]i32{ 8, 4, 0, -4 };
        for (0..width) |d| {
            const sign: f32 = if (d % 2 == 0) 1 else -1;
            branch[d] = math.b(sign * @as(f32, @floatCast(std.math.pow(f64, 2, @floatFromInt(branch_exponents[d % 4])))));
        }
        try runCase(gpa, device, queue, &p, &o, cfg, .{ .name = "wide_dynamic_range", .rows = 1, .stream = stream, .weight = weight, .branch = branch, .scale = ffn_scale, .base = ffn_base });
    }
}

// Compare math.mixesOf(f32) bitwise with the verbatim pre-parameterization algorithm on a retained row.
test "mHC fp32 reference path stays bit-identical under the comptime parameterization" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const cfg = try Config.parse(gpa, released_config);
    const fixture = try std.json.parseFromSlice(Fixture, gpa, @embedFile(fixture_path), .{ .ignore_unknown_fields = true });
    const eight_rows = try decodeU16(gpa, fixture.value.eight_rows_bf16);
    const ffn_scale = try decodeF32(gpa, fixture.value.ffn_scale_f32);
    const ffn_base = try decodeF32(gpa, fixture.value.ffn_base_f32);
    const scale = try device.buffer(12, mtl.ResourceOptions.shared);
    defer scale.deinit();
    upload(f32, scale, ffn_scale);
    const base = try device.buffer(mix_count * 4, mtl.ResourceOptions.shared);
    defer base.deinit();
    upload(f32, base, ffn_base);
    var lcg = Lcg{};
    const weight = try device.buffer(mix_count * stream_width * 4, mtl.ResourceOptions.shared);
    defer weight.deinit();
    const weight_host = try gpa.alloc(f32, mix_count * stream_width);
    for (weight_host) |*v| v.* = lcg.uniform() * 0.03;
    upload(f32, weight, weight_host);
    const hc = Hc{
        .projection = .{ .buffer = weight, .offset = 0, .bytes = mix_count * stream_width * 4, .dtype = .f32 },
        .scale = .{ .buffer = scale, .offset = 0, .bytes = 12, .dtype = .f32 },
        .base = .{ .buffer = base, .offset = 0, .bytes = mix_count * 4, .dtype = .f32 },
    };
    for (0..8) |r| {
        const row = eight_rows[r * width ..][0..width];
        var stream: [stream_width]u16 = undefined;
        for (0..copies) |c| @memcpy(stream[c * width ..][0..width], row);
        const parameterized = math.mixesOf(f32, &stream, hc, cfg.text_config);
        const legacy = mixesLegacy(&stream, hc, cfg);
        try std.testing.expect(std.meta.eql(parameterized, legacy));
    }
}
