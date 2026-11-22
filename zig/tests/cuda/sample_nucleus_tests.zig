//! Adversarial and timing cases for CUDA nucleus sampling.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const check = @import("check.zig");
const sample_tests = @import("sample_tests.zig");
const Gpu = check.Gpu;
const vocab = sample_tests.vocab;
const Rule = sample_tests.Rule;
const Kind = sample_tests.Kind;
const fill = sample_tests.fill;
const bf16 = sample_tests.bf16;
const ruleOf = sample_tests.ruleOf;
const widen = sample_tests.widen;
const order = sample_tests.order;
const share = sample_tests.share;

const extended_vocab = 248320;
const extended_mapped = 32768;
const extended_padding = 128;

const sample_bench_warm = 3;
const sample_bench_reps = 11;
const sample_bench_calls = 16;
const sample_bench_rows = 4;
const sample_bench_pitch = vocab + 128;
const BenchPath = enum { sample, pitched_draw };

const ExtendedCase = struct {
    name: []const u8,
    kind: Kind,
    top_k: u32,
    top_p: f64,
    pitched: bool,
    mapped: bool,
};

/// Boundary cases at Qwen's vocabulary size, also used for exact old/new kernel comparisons.
pub fn adversarial(gpu: Gpu) !void {
    const cases = [_]ExtendedCase{
        .{ .name = "flat-v248320-k0-p95", .kind = .flat, .top_k = 0, .top_p = 0.95, .pitched = false, .mapped = false },
        .{ .name = "flat-v248320-k0-p999999-pitched", .kind = .flat, .top_k = 0, .top_p = 0.999999, .pitched = true, .mapped = false },
        .{ .name = "flat-v248320-k1500-p95-pitched", .kind = .flat, .top_k = 1500, .top_p = 0.95, .pitched = true, .mapped = false },
        .{ .name = "flat-v248320-k0-p000001", .kind = .flat, .top_k = 0, .top_p = 0.000001, .pitched = false, .mapped = false },
        .{ .name = "spread-v248320-k0-p95-pitched", .kind = .spread, .top_k = 0, .top_p = 0.95, .pitched = true, .mapped = false },
        .{ .name = "mapped-v32768-k0-p95", .kind = .flat, .top_k = 0, .top_p = 0.95, .pitched = false, .mapped = true },
        .{ .name = "mapped-v32768-k1500-p95-pitched", .kind = .flat, .top_k = 1500, .top_p = 0.95, .pitched = true, .mapped = true },
    };
    const gpa = gpu.gpa;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer module.unload();
    const full = try module.function("tf_draw");
    const ids_fn = try module.function("tf_draw_ids");
    const pitched_fn = try module.function("tf_draw_pitched");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const host = try gpa.alloc(u16, extended_vocab + extended_padding);
    defer gpa.free(host);
    const wide = try gpa.alloc(f32, extended_vocab);
    defer gpa.free(wide);
    const ids = try gpa.alloc(u32, extended_mapped);
    defer gpa.free(ids);
    const wide_ids = try gpa.alloc(i64, extended_mapped);
    defer gpa.free(wide_ids);
    for (ids, 0..) |*id, i| id.* = @intCast(i * 7 + 1);
    for (wide_ids, ids) |*id, token| id.* = token;
    var ids_dev = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wide_ids));
    defer ids_dev.free();
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, (extended_vocab + extended_padding) * 2);
    defer logits.free();
    var rule_dev = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(Rule));
    defer rule_dev.free();
    var meta = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(i32));
    defer meta.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(u32));
    defer tok.free();
    var prob = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(f32));
    defer prob.free();
    const base: i32 = 100;
    try meta.upload(0, std.mem.asBytes(&base));
    for (cases, 0..) |case, case_index| {
        const n: usize = if (case.mapped) extended_mapped else extended_vocab;
        const pitch: usize = if (case.pitched) n + extended_padding else n;
        fill(case.kind, 0x3713 + @as(u64, @intCast(case_index)), host[0..n], wide[0..n]);
        @memset(host[n..pitch], bf16(0));
        try logits.upload(0, std.mem.sliceAsBytes(host[0..pitch]));
        const sampling: lanes.Sampling = .{
            .seed = 0x3713_0000 + @as(u64, @intCast(case_index)),
            .temperature = 1.0,
            .top_k = case.top_k,
            .top_p = case.top_p,
            .min_p = 0.0,
        };
        const rule_value = ruleOf(sampling);
        try rule_dev.upload(0, std.mem.asBytes(&rule_value));
        const offset: i32 = if (case.mapped) 2 else 0;
        if (case.pitched) {
            var args: cuda.Args = .{};
            args.add(logits.ptr);
            args.add(@as(u32, @intCast(n)));
            args.add(@as(u32, @intCast(pitch)));
            args.add(rule_dev.ptr);
            args.add(meta.ptr);
            args.add(offset);
            args.add(tok.ptr);
            args.add(if (case.mapped) ids_dev.ptr else @as(u64, 0));
            args.add(prob.ptr);
            try cuda.launch.launch(pitched_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        } else if (case.mapped) {
            var args: cuda.Args = .{};
            args.add(logits.ptr);
            args.add(@as(u32, @intCast(n)));
            args.add(rule_dev.ptr);
            args.add(meta.ptr);
            args.add(offset);
            args.add(tok.ptr);
            args.add(ids_dev.ptr);
            args.add(prob.ptr);
            try cuda.launch.launch(ids_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        } else {
            var args: cuda.Args = .{};
            args.add(logits.ptr);
            args.add(@as(u32, @intCast(n)));
            args.add(rule_dev.ptr);
            args.add(meta.ptr);
            args.add(offset);
            args.add(tok.ptr);
            args.add(prob.ptr);
            try cuda.launch.launch(full, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &args);
        }
        try stream.synchronize();
        var got: [1]u32 = undefined;
        var probability: [1]f32 = undefined;
        try tok.download(0, std.mem.sliceAsBytes(&got));
        try prob.download(0, std.mem.sliceAsBytes(&probability));
        const position: u32 = @intCast(base + 1 + offset);
        const expected = try lanes.gpu_full.draw(gpa, wide[0..n], sampling, position, if (case.mapped) ids else null);
        try check.expect(got[0] == expected, "sample-adversarial {s}: got {d}, host {d}", .{ case.name, got[0], expected });
        const col: usize = if (case.mapped) @intCast((got[0] - 1) / 7) else @intCast(got[0]);
        const sorted = try order(gpa, wide[0..n]);
        defer gpa.free(sorted);
        const reference_probability = share(wide[0..n], sorted, sampling, col);
        const relative_error = @abs(@as(f64, probability[0]) - reference_probability) / reference_probability;
        try check.expect(relative_error < 1e-3, "sample-adversarial {s}: picked share differs by {e}", .{ case.name, relative_error });
        std.debug.print("RESULT sample-adversarial {s} token {d} probability_bits 0x{x}\n", .{ case.name, got[0], @as(u32, @bitCast(probability[0])) });
    }
    check.pass("sample-adversarial: {d} boundary cases equal the host reference", .{cases.len});
}

fn launchDrawRows(f: anytype, stream: cuda.Stream, logits: u64, rule: u64, meta: u64, tok: u64, n_rows: usize) !void {
    var args: cuda.Args = .{};
    args.add(logits);
    args.add(@as(u32, @intCast(vocab)));
    args.add(rule);
    args.add(meta);
    args.add(@as(i32, 0));
    args.add(tok);
    args.add(@as(u64, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n_rows) }, .block = .{ .x = 1024 } }, stream, &args);
}

fn launchPitchedDraw(f: anytype, stream: cuda.Stream, logits: u64, rule: u64, meta: u64, tok: u64, n_rows: usize) !void {
    var args: cuda.Args = .{};
    args.add(logits);
    args.add(@as(u32, @intCast(vocab)));
    args.add(@as(u32, @intCast(sample_bench_pitch)));
    args.add(rule);
    args.add(meta);
    args.add(@as(i32, 0));
    args.add(tok);
    args.add(@as(u64, 0));
    args.add(@as(u64, 0));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n_rows) }, .block = .{ .x = 1024 } }, stream, &args);
}

/// Amortized per-row times for full-row and small-k draws, with nucleus off and on.
pub fn bench(gpu: Gpu) !void {
    const gpa = gpu.gpa;
    var module = try cuda.Module.load(gpu.d, cuda.kernels.sample);
    defer module.unload();
    const draw_fn = try module.function("tf_draw");
    const pitched_fn = try module.function("tf_draw_pitched");
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const host = try gpa.alloc(u16, sample_bench_rows * vocab);
    defer gpa.free(host);
    const pitched_host = try gpa.alloc(u16, sample_bench_rows * sample_bench_pitch);
    defer gpa.free(pitched_host);
    const wide = try gpa.alloc(f32, sample_bench_rows * vocab);
    defer gpa.free(wide);
    var logits = try cuda.DeviceBuffer.alloc(gpu.d, sample_bench_rows * vocab * 2);
    defer logits.free();
    var pitched_logits = try cuda.DeviceBuffer.alloc(gpu.d, sample_bench_rows * sample_bench_pitch * 2);
    defer pitched_logits.free();
    var rule = try cuda.DeviceBuffer.alloc(gpu.d, @sizeOf(Rule));
    defer rule.free();
    var meta = try cuda.DeviceBuffer.alloc(gpu.d, 4);
    defer meta.free();
    var tok = try cuda.DeviceBuffer.alloc(gpu.d, sample_bench_rows * 4);
    defer tok.free();
    try meta.upload(0, std.mem.asBytes(&[_]i32{100}));
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    for ([_]Kind{ .flat, .spread, .peaked }, 0..) |kind, ki| {
        for (0..sample_bench_rows) |r| {
            const row = host[r * vocab ..][0..vocab];
            fill(kind, ki + 17 + r, row, wide[r * vocab ..][0..vocab]);
            @memcpy(pitched_host[r * sample_bench_pitch ..][0..vocab], row);
            @memset(pitched_host[r * sample_bench_pitch + vocab ..][0 .. sample_bench_pitch - vocab], 0);
        }
        try logits.upload(0, std.mem.sliceAsBytes(host));
        try pitched_logits.upload(0, std.mem.sliceAsBytes(pitched_host));
        for ([_]u32{ 0, 20 }) |top_k| for ([_]f32{ 1.0, 0.95 }) |top_p| {
            const sampling: lanes.Sampling = .{ .seed = 0x3713 + ki, .temperature = 1.0, .top_k = top_k, .top_p = top_p };
            const rule_value = ruleOf(sampling);
            try rule.upload(0, std.mem.asBytes(&rule_value));
            inline for (.{ BenchPath.sample, BenchPath.pitched_draw }) |path| {
                for (0..sample_bench_warm) |_| {
                    if (path == .sample) {
                        try launchDrawRows(draw_fn, stream, logits.ptr, rule.ptr, meta.ptr, tok.ptr, sample_bench_rows);
                    } else {
                        try launchPitchedDraw(pitched_fn, stream, pitched_logits.ptr, rule.ptr, meta.ptr, tok.ptr, sample_bench_rows);
                    }
                }
                try stream.synchronize();
                var times: [sample_bench_reps]f64 = undefined;
                for (&times) |*time| {
                    try start.record(stream);
                    for (0..sample_bench_calls) |_| {
                        if (path == .sample) {
                            try launchDrawRows(draw_fn, stream, logits.ptr, rule.ptr, meta.ptr, tok.ptr, sample_bench_rows);
                        } else {
                            try launchPitchedDraw(pitched_fn, stream, pitched_logits.ptr, rule.ptr, meta.ptr, tok.ptr, sample_bench_rows);
                        }
                    }
                    try end.record(stream);
                    try stream.synchronize();
                    time.* = @as(f64, try start.elapsedMs(end)) * 1000.0 / (sample_bench_calls * sample_bench_rows);
                }
                std.mem.sort(f64, &times, {}, std.sort.asc(f64));
                std.debug.print("RESULT sample-bench {s} {s} top_k {d} top_p {d:.2}: {d:.2} us per row\n", .{
                    @tagName(kind), @tagName(path), top_k, top_p, times[times.len / 2],
                });
            }
        };
    }
}
