//! tf_sample_full and the two-pass top_k draw against their host reference on synthetic rows: exit 0 when all match.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");

const full = tf.lanes.gpu_full;
const Sampling = tf.lanes.Sampling;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Rows a dispatch: one row at each position from `position`, as a decode window draws them.
const rows = 16;
const position: u32 = 4096;

const Case = struct { name: []const u8, s: Sampling, fill: *const fn ([]f32) void, vocab: usize = 131072, map: bool = false };

fn flat(l: []f32) void {
    @memset(l, 0.0);
}

fn peaked(l: []f32) void {
    for (l, 0..) |*x, i| x.* = 12.0 - 0.05 * @as(f32, @floatFromInt(i % 8192)) - @as(f32, @floatFromInt(i / 8192));
}

fn spread(l: []f32) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15;
    for (l) |*v| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        v.* = -3.0 * @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
    }
}

fn ties(l: []f32) void {
    for (l, 0..) |*v, i| v.* = -0.5 * @as(f32, @floatFromInt((i * 2654435761) % 7));
}

fn twoLevels(l: []f32) void {
    @memset(l, -10.0);
    @memset(l[0..2000], 0.0);
}

const cases = [_]Case{
    .{ .name = "flat, top_k off, top_p 1", .s = .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, .fill = flat },
    .{ .name = "peaked, top_p 0.9 (nucleus inside 1,024)", .s = .{ .seed = 2, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }, .fill = peaked },
    .{ .name = "spread, top_p 0.9 (nucleus past 1,024)", .s = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = spread },
    .{ .name = "7 tied levels, top_p 0.5", .s = .{ .seed = 4, .temperature = 1.0, .top_k = 0, .top_p = 0.5 }, .fill = ties },
    .{ .name = "spread, top_k 3000", .s = .{ .seed = 5, .temperature = 1.0, .top_k = 3000, .top_p = 1.0 }, .fill = spread },
    .{ .name = "spread, top_k 3000, top_p 0.5", .s = .{ .seed = 6, .temperature = 1.0, .top_k = 3000, .top_p = 0.5 }, .fill = spread },
    .{ .name = "two levels, min_p 0.01", .s = .{ .seed = 7, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.01 }, .fill = twoLevels },
    .{ .name = "draft vocabulary through an id map", .s = .{ .seed = 8, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = spread, .vocab = 49152, .map = true },
    .{ .name = "spread, top_k 20, top_p 0.95 (Flash Next's vocabulary)", .s = .{ .seed = 9, .temperature = 1.0, .top_k = 20, .top_p = 0.95 }, .fill = spread, .vocab = 248320 },
    .{ .name = "peaked, top_k 20, top_p 0.95, T 0.7", .s = .{ .seed = 10, .temperature = 0.7, .top_k = 20, .top_p = 0.95 }, .fill = peaked },
    .{ .name = "7 tied levels, top_k 40, top_p 0.5", .s = .{ .seed = 11, .temperature = 1.0, .top_k = 40, .top_p = 0.5 }, .fill = ties },
    .{ .name = "two levels, top_k 64, min_p 0.01", .s = .{ .seed = 12, .temperature = 1.0, .top_k = 64, .top_p = 1.0, .min_p = 0.01 }, .fill = twoLevels },
    .{ .name = "flat, top_k 1", .s = .{ .seed = 13, .temperature = 1.0, .top_k = 1, .top_p = 1.0 }, .fill = flat },
    .{ .name = "draft vocabulary through an id map, top_k 20", .s = .{ .seed = 14, .temperature = 1.0, .top_k = 20, .top_p = 0.95 }, .fill = spread, .vocab = 79616, .map = true },
};

/// tf_topk_slices' slices a row and threads a slice, and the widest top_k it draws (nemotron_sample.metal).
const slices = 32;
const slice_threads = 512;
const kmax = 64;

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

/// The generated tf_gpu_sample for the full vocabulary, for timing beside tf_sample_full.
fn generated(key: []const u8) sources.nemotron.Kernel {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k;
    unreachable;
}

/// One dispatch of `rows` rows through `p`; its GPU milliseconds.
fn dispatch(queue: mtl.Queue, p: mtl.Pipeline, logits: mtl.Buffer, s: Sampling, tok: mtl.Buffer, ids: mtl.Buffer, vocab: usize, whole: bool) !f64 {
    var seeds: [2 * rows]u32 = undefined;
    var positions: [rows]u32 = undefined;
    var cfgs: [4 * rows]f32 = undefined;
    var caps: [rows]u32 = undefined;
    const rule = tf.lanes.gpu_rule.rule(s);
    for (0..rows) |r| {
        seeds[2 * r ..][0..2].* = rule.seedHalves();
        positions[r] = position + @as(u32, @intCast(r));
        cfgs[4 * r ..][0..4].* = rule.cfg();
        caps[r] = rule.top_k;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(logits, 0, 0);
    enc.setBytes(std.mem.sliceAsBytes(&seeds), 1);
    enc.setBytes(std.mem.sliceAsBytes(&positions), 2);
    enc.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
    enc.setBytes(std.mem.sliceAsBytes(&caps), 4);
    enc.setBuffer(tok, 0, 5);
    if (whole) {
        enc.setBuffer(ids, 0, 6);
        enc.setBytes(std.mem.asBytes(&@as(u32, @intCast(vocab))), 7);
    }
    enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("command buffer failed: {s}\n", .{text});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e3;
}

/// The two-pass draw of `rows` rows (tf_topk_slices into ck / ci, then `race`); its GPU milliseconds.
fn twoPass(queue: mtl.Queue, first: mtl.Pipeline, race: mtl.Pipeline, logits: mtl.Buffer, s: Sampling, tok: mtl.Buffer, ids: mtl.Buffer, vocab: usize, ck: mtl.Buffer, ci: mtl.Buffer) !f64 {
    var seeds: [2 * rows]u32 = undefined;
    var positions: [rows]u32 = undefined;
    var cfgs: [4 * rows]f32 = undefined;
    var caps: [rows]u32 = undefined;
    const rule = tf.lanes.gpu_rule.rule(s);
    for (0..rows) |r| {
        seeds[2 * r ..][0..2].* = rule.seedHalves();
        positions[r] = position + @as(u32, @intCast(r));
        cfgs[4 * r ..][0..4].* = rule.cfg();
        caps[r] = rule.top_k;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setBuffer(logits, 0, 0);
    enc.setBytes(std.mem.sliceAsBytes(&seeds), 1);
    enc.setBytes(std.mem.sliceAsBytes(&positions), 2);
    enc.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
    enc.setBytes(std.mem.sliceAsBytes(&caps), 4);
    enc.setBuffer(tok, 0, 5);
    enc.setBuffer(ids, 0, 6);
    enc.setBytes(std.mem.asBytes(&@as(u32, @intCast(vocab))), 7);
    enc.setBuffer(ck, 0, 8);
    enc.setBuffer(ci, 0, 9);
    enc.setPipeline(first);
    enc.dispatchGroups(mtl.Size.of(slices, rows, 1), mtl.Size.of(slice_threads, 1, 1));
    enc.setPipeline(race);
    enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("command buffer failed: {s}\n", .{text});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e3;
}

const F32Case = struct {
    name: []const u8,
    s: Sampling,
    fill: *const fn ([]f32, usize) void,
    pos: u32 = position,
    vocab: usize = 200058,
    n_rows: usize = rows,
    map: bool = false,
};

fn fillFine(l: []f32, row_idx: usize) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15 +% row_idx *% 0xD1B5_4A32_D192_ED03;
    for (l) |*v| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        v.* = -2.0 * @as(f32, @floatFromInt(x >> 33)) / @as(f32, 1 << 31);
    }
    l[1234] = 5.0;
    l[5678] = 5.0 + 0x1p-20;
    l[9999] = 5.0 + 0x1p-20;
}

fn fillSharp(l: []f32, _: usize) void {
    for (l, 0..) |*v, i| v.* = -8.0 - 0.01 * @as(f32, @floatFromInt(i % 4096));
    l[77] = 10.0;
    l[200001 % l.len] = 9.5;
}

fn fillNonfinite(l: []f32, row_idx: usize) void {
    fillFine(l, row_idx);
    if (row_idx % 4 == 0) l[l.len / 3] = std.math.nan(f32);
    if (row_idx % 4 == 1) {
        l[0] = std.math.nan(f32);
        l[10] = 50.0;
    }
    if (row_idx % 4 == 2) l[l.len / 2] = std.math.inf(f32);
    if (row_idx % 4 == 3) {
        l[0] = -0.0;
        l[1] = 0.0;
        l[l.len - 1] = -std.math.inf(f32);
    }
}

fn fillNormBoundary(l: []f32, _: usize) void {
    @memset(l, -50.0);
    l[9] = 0.0;
    l[1035] = -16.7;
    @memset(l[1056..1088], -16.7);
}

fn fillUniformOneBoundary(l: []f32, _: usize) void {
    @memset(l, -50.0);
    l[3122] = 0.0;
    l[1035] = -3.7;
    @memset(l[1040..1058], -3.7);
}

fn fillNeverFloorBoundary(l: []f32, _: usize) void {
    @memset(l, -50.0);
    l[3122] = 0.0;
    l[1035] = -30.0;
}

const f32_cases = [_]F32Case{
    .{ .name = "f32 200k broad top_p 0.9, fine ulp", .s = .{ .seed = 101, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = fillFine, .n_rows = 8 },
    .{ .name = "f32 200k sharp top_p 0.05, T 0.7", .s = .{ .seed = 102, .temperature = 0.7, .top_k = 0, .top_p = 0.05 }, .fill = fillSharp, .n_rows = 4 },
    .{ .name = "f32 200k top_k 1", .s = .{ .seed = 103, .temperature = 1.0, .top_k = 1, .top_p = 1.0 }, .fill = fillFine, .n_rows = 3 },
    .{ .name = "f32 200k top_k 20, top_p 0.95, T 0.7", .s = .{ .seed = 104, .temperature = 0.7, .top_k = 20, .top_p = 0.95 }, .fill = fillFine, .n_rows = 8 },
    .{ .name = "f32 200k top_k 64, min_p 0.01", .s = .{ .seed = 105, .temperature = 1.0, .top_k = 64, .top_p = 1.0, .min_p = 0.01 }, .fill = fillSharp, .n_rows = 5 },
    .{ .name = "f32 200k top_k 100, top_p 0.9", .s = .{ .seed = 112, .temperature = 1.0, .top_k = 100, .top_p = 0.9 }, .fill = fillFine, .n_rows = 6 },
    .{ .name = "f32 200k top_k 1024", .s = .{ .seed = 106, .temperature = 1.0, .top_k = 1024, .top_p = 0.9 }, .fill = fillFine, .n_rows = 6 },
    .{ .name = "f32 200k top_k 2048", .s = .{ .seed = 107, .temperature = 0.7, .top_k = 2048, .top_p = 0.9 }, .fill = fillFine, .n_rows = 7 },
    .{ .name = "f32 200k nonfinite NaN/inf/signed-zero, top_k 0", .s = .{ .seed = 108, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = fillNonfinite, .n_rows = 8 },
    .{ .name = "f32 200k nonfinite NaN/inf, top_k 20", .s = .{ .seed = 109, .temperature = 1.0, .top_k = 20, .top_p = 0.95 }, .fill = fillNonfinite, .n_rows = 8 },
    .{ .name = "f32 200k nonfinite NaN/inf, top_k 100", .s = .{ .seed = 113, .temperature = 1.0, .top_k = 100, .top_p = 0.95 }, .fill = fillNonfinite, .n_rows = 8 },
    .{ .name = "f32 200k ids map, top_k 0", .s = .{ .seed = 110, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = fillFine, .n_rows = 4, .map = true },
    .{ .name = "f32 200k ids map, top_k 20", .s = .{ .seed = 111, .temperature = 1.0, .top_k = 20, .top_p = 0.95 }, .fill = fillFine, .n_rows = 4, .map = true },
    .{ .name = "f32 200k ids map, top_k 100", .s = .{ .seed = 114, .temperature = 1.0, .top_k = 100, .top_p = 0.95 }, .fill = fillFine, .n_rows = 4, .map = true },
    .{ .name = "f32 norm boundary, top_k 40", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 40, .top_p = 0.9999995 }, .fill = fillNormBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 norm boundary, top_k 100", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9999995 }, .fill = fillNormBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 norm boundary, top_k 0", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9999995 }, .fill = fillNormBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 uniform=1.0 boundary, top_k 20", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 20, .top_p = 0.9 }, .fill = fillUniformOneBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 uniform=1.0 boundary, top_k 100", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 0.9 }, .fill = fillUniformOneBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 uniform=1.0 boundary, top_k 0", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = fillUniformOneBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 never-24 boundary, top_k 20", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 20, .top_p = 1.0 }, .fill = fillNeverFloorBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 never-24 boundary, top_k 100", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 100, .top_p = 1.0 }, .fill = fillNeverFloorBoundary, .pos = 42, .n_rows = 1 },
    .{ .name = "f32 never-24 boundary, top_k 0", .s = .{ .seed = 18508, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, .fill = fillNeverFloorBoundary, .pos = 42, .n_rows = 1 },
};

const Times = struct { wall: f64, gpu: f64 };

fn msSince(t0: std.Io.Timestamp, io: std.Io, iters: usize) f64 {
    const ns: f64 = @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
    return ns / 1e6 / @as(f64, @floatFromInt(iters));
}

fn dispatchF32(queue: mtl.Queue, p: mtl.Pipeline, logits: mtl.Buffer, s: Sampling, pos0: u32, tok: mtl.Buffer, ids: mtl.Buffer, vocab: usize, n_rows: usize, iters: usize, io: std.Io) !Times {
    var launch: tf.lanes.gpu_rule.Launch(rows) = .{};
    for (0..n_rows) |r| launch.put(s, pos0 + r);
    const padded = launch.padded();
    const t0 = std.Io.Clock.awake.now(io);
    var gpu_ms: f64 = 0;
    for (0..iters) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(p);
        enc.setBuffer(logits, 0, 0);
        enc.setBytes(std.mem.sliceAsBytes(launch.seeds[0 .. 2 * padded]), 1);
        enc.setBytes(std.mem.sliceAsBytes(launch.positions[0..padded]), 2);
        enc.setBytes(std.mem.sliceAsBytes(launch.cfgs[0 .. 4 * padded]), 3);
        enc.setBytes(std.mem.sliceAsBytes(launch.caps[0..padded]), 4);
        enc.setBuffer(tok, 0, 5);
        enc.setBuffer(ids, 0, 6);
        enc.setValue(@as(u32, @intCast(vocab)), 7);
        enc.dispatchGroups(mtl.Size.of(n_rows, 1, 1), mtl.Size.of(1024, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        gpu_ms += cb.gpuSeconds() * 1e3;
    }
    return .{ .wall = msSince(t0, io, iters), .gpu = gpu_ms / @as(f64, @floatFromInt(iters)) };
}

fn twoPassF32(queue: mtl.Queue, first: mtl.Pipeline, rule_p: mtl.Pipeline, logits: mtl.Buffer, s: Sampling, pos0: u32, tok: mtl.Buffer, ids: mtl.Buffer, vocab: usize, n_rows: usize, ck: mtl.Buffer, ci: mtl.Buffer, iters: usize, io: std.Io) !Times {
    var launch: tf.lanes.gpu_rule.Launch(rows) = .{};
    for (0..n_rows) |r| launch.put(s, pos0 + r);
    const padded = launch.padded();
    const t0 = std.Io.Clock.awake.now(io);
    var gpu_ms: f64 = 0;
    for (0..iters) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setBuffer(logits, 0, 0);
        enc.setBytes(std.mem.sliceAsBytes(launch.seeds[0 .. 2 * padded]), 1);
        enc.setBytes(std.mem.sliceAsBytes(launch.positions[0..padded]), 2);
        enc.setBytes(std.mem.sliceAsBytes(launch.cfgs[0 .. 4 * padded]), 3);
        enc.setBytes(std.mem.sliceAsBytes(launch.caps[0..padded]), 4);
        enc.setBuffer(tok, 0, 5);
        enc.setBuffer(ids, 0, 6);
        enc.setValue(@as(u32, @intCast(vocab)), 7);
        enc.setBuffer(ck, 0, 8);
        enc.setBuffer(ci, 0, 9);
        enc.setPipeline(first);
        enc.dispatchGroups(mtl.Size.of(slices, n_rows, 1), mtl.Size.of(slice_threads, 1, 1));
        enc.setPipeline(rule_p);
        enc.dispatchGroups(mtl.Size.of(n_rows, 1, 1), mtl.Size.of(64, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |text| {
            std.debug.print("command buffer failed: {s}\n", .{text});
            return error.GpuFailed;
        }
        gpu_ms += cb.gpuSeconds() * 1e3;
    }
    return .{ .wall = msSince(t0, io, iters), .gpu = gpu_ms / @as(f64, @floatFromInt(iters)) };
}

fn checkArgmax(device: mtl.Device, queue: mtl.Queue) !usize {
    const glue_lib = try mtl.Library.fromSource(device, sources.inkling2_glue, mtl.CompileOptions.mlx());
    defer glue_lib.deinit();
    const argmax_pipe = try mtl.Pipeline.init(device, glue_lib, "tf_ink2_argmax", false);
    defer argmax_pipe.deinit();
    const vocab: usize = 200058;
    const n_rows: usize = 8;
    const logits = try device.buffer(n_rows * vocab * 4, opts);
    defer logits.deinit();
    const tok = try device.buffer(n_rows * 4, opts);
    defer tok.deinit();
    const host_rows = logits.slice(f32, n_rows * vocab);
    for (0..n_rows) |r| {
        const row = host_rows[r * vocab ..][0..vocab];
        fillFine(row, r);
        switch (r) {
            0 => {},
            1 => row[0] = std.math.nan(f32),
            2 => {
                row[0] = -10.0;
                row[777] = std.math.nan(f32);
                row[1234] = 5.0;
            },
            3 => {
                @memset(row, -5.0);
                row[33] = -0.0;
                row[44] = 0.0;
            },
            4 => {
                row[200000] = std.math.inf(f32);
                row[200057] = std.math.inf(f32);
            },
            5 => {
                @memset(row, std.math.nan(f32));
                row[0] = -std.math.inf(f32);
            },
            else => fillNonfinite(row, r),
        }
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(argmax_pipe);
    enc.setBuffer(logits, 0, 0);
    enc.setBuffer(tok, 0, 1);
    enc.setValue(@as(u32, vocab), 2);
    enc.dispatchGroups(mtl.Size.of(n_rows, 1, 1), mtl.Size.of(1024, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("argmax cb failed: {s}\n", .{text});
        return n_rows;
    }
    var failed: usize = 0;
    for (tok.slice(u32, n_rows), 0..) |got, r| {
        const want = tf.inkling2_metal_host.draw.greedyArgmax(host_rows[r * vocab ..][0..vocab]);
        if (got != want) {
            std.debug.print("tf_ink2_argmax row {d}: GPU {d}, host {d}\n", .{ r, got, want });
            failed += 1;
        }
    }
    std.debug.print("tf_ink2_argmax 200k: {d}/{d} rows match\n", .{ n_rows - failed, n_rows });
    return failed;
}

fn checkF32(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, ink_lib: mtl.Library, ck: mtl.Buffer, ci: mtl.Buffer, io: std.Io) !usize {
    const first_f32 = try mtl.Pipeline.init(device, lib, "tf_topk_slices_f32", false);
    defer first_f32.deinit();
    var failed: usize = try checkArgmax(device, queue);
    for (f32_cases) |c| {
        const logits = try device.buffer(c.n_rows * c.vocab * 4, opts);
        defer logits.deinit();
        const frows = logits.slice(f32, c.n_rows * c.vocab);
        for (0..c.n_rows) |r| c.fill(frows[r * c.vocab ..][0..c.vocab], r);
        const map = try gpa.alloc(u32, c.vocab);
        defer gpa.free(map);
        for (map, 0..) |*m, i| m.* = @intCast(2 * i + 1);
        const ids = try device.buffer(c.vocab * 4, opts);
        defer ids.deinit();
        @memcpy(ids.slice(u32, c.vocab), map);
        const tok = try device.buffer(c.n_rows * 4, opts);
        defer tok.deinit();
        @memset(tok.slice(u32, c.n_rows), 0xFFFF_FFFF);
        if (full.fullVocabulary(c.s)) {
            const p = try mtl.Pipeline.init(device, lib, if (c.map) "tf_sample_full_ids_f32" else "tf_sample_full_f32", false);
            defer p.deinit();
            _ = try dispatchF32(queue, p, logits, c.s, c.pos, tok, ids, c.vocab, c.n_rows, 1, io);
        } else if (c.s.top_k <= kmax) {
            const p = try mtl.Pipeline.init(device, ink_lib, if (c.map) "tf_ink2_topk_rule_ids_f32" else "tf_ink2_topk_rule_f32", false);
            defer p.deinit();
            _ = try twoPassF32(queue, first_f32, p, logits, c.s, c.pos, tok, ids, c.vocab, c.n_rows, ck, ci, 1, io);
        } else {
            const p = try mtl.Pipeline.init(device, ink_lib, if (c.map) "tf_ink2_rule_1024_ids_f32" else "tf_ink2_rule_1024_f32", false);
            defer p.deinit();
            _ = try dispatchF32(queue, p, logits, c.s, c.pos, tok, ids, c.vocab, c.n_rows, 1, io);
        }
        const scratch = try gpa.alloc(f32, c.vocab);
        defer gpa.free(scratch);
        var same: usize = 0;
        for (tok.slice(u32, c.n_rows), 0..) |got, r| {
            const row = frows[r * c.vocab ..][0..c.vocab];
            const pos = c.pos + @as(u32, @intCast(r));
            const want = try full.draw(gpa, row, c.s, pos, if (c.map) map else null);
            if (!c.map) {
                const host_want = try tf.inkling2_metal_host.draw.draw(gpa, row, scratch, c.s, pos, &.{});
                if (want != host_want) std.debug.print("  {s} row {d}: draw.draw {d} vs gpu_full.draw {d}\n", .{ c.name, r, host_want, want });
            }
            same += @intFromBool(got == want);
            if (got != want) std.debug.print("  {s} row {d}: GPU {d}, host {d}\n", .{ c.name, r, got, want });
        }
        failed += c.n_rows - same;
        std.debug.print("{s}: {d}/{d} rows match\n", .{ c.name, same, c.n_rows });
    }
    return failed;
}

fn timeCpuDraw(gpa: std.mem.Allocator, frows: []const f32, scratch: []f32, s: ?Sampling, vocab: usize, n_rows: usize, iters: usize, io: std.Io) !f64 {
    const t0 = std.Io.Clock.awake.now(io);
    var acc: u32 = 0;
    for (0..iters) |it| {
        for (0..n_rows) |r| {
            acc +%= try tf.inkling2_metal_host.draw.draw(gpa, frows[r * vocab ..][0..vocab], scratch, s, position + r + it, &.{});
        }
    }
    std.mem.doNotOptimizeAway(acc);
    return msSince(t0, io, iters);
}

fn timeGpuArgmax(queue: mtl.Queue, p: mtl.Pipeline, logits: mtl.Buffer, tok: mtl.Buffer, vocab: usize, n_rows: usize, iters: usize, io: std.Io) !Times {
    const t0 = std.Io.Clock.awake.now(io);
    var gpu_ms: f64 = 0;
    for (0..iters) |_| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(p);
        enc.setBuffer(logits, 0, 0);
        enc.setBuffer(tok, 0, 1);
        enc.setValue(@as(u32, @intCast(vocab)), 2);
        enc.dispatchGroups(mtl.Size.of(n_rows, 1, 1), mtl.Size.of(1024, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        gpu_ms += cb.gpuSeconds() * 1e3;
    }
    return .{ .wall = msSince(t0, io, iters), .gpu = gpu_ms / @as(f64, @floatFromInt(iters)) };
}

fn benchF32(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, lib: mtl.Library, ink_lib: mtl.Library, ck: mtl.Buffer, ci: mtl.Buffer, io: std.Io) !void {
    const vocab: usize = 200058;
    const full_p = try mtl.Pipeline.init(device, lib, "tf_sample_full_f32", false);
    defer full_p.deinit();
    const slices_p = try mtl.Pipeline.init(device, lib, "tf_topk_slices_f32", false);
    defer slices_p.deinit();
    const topk_p = try mtl.Pipeline.init(device, ink_lib, "tf_ink2_topk_rule_f32", false);
    defer topk_p.deinit();
    const glue_lib = try mtl.Library.fromSource(device, sources.inkling2_glue, mtl.CompileOptions.mlx());
    defer glue_lib.deinit();
    const argmax_p = try mtl.Pipeline.init(device, glue_lib, "tf_ink2_argmax", false);
    defer argmax_p.deinit();
    const logits = try device.buffer(rows * vocab * 4, opts);
    defer logits.deinit();
    const tok = try device.buffer(rows * 4, opts);
    defer tok.deinit();
    const ids = try device.buffer(vocab * 4, opts);
    defer ids.deinit();
    const frows = logits.slice(f32, rows * vocab);
    for (0..rows) |r| fillFine(frows[r * vocab ..][0..vocab], r);
    const scratch = try gpa.alloc(f32, vocab);
    defer gpa.free(scratch);
    const s_full: Sampling = .{ .seed = 101, .temperature = 1.0, .top_k = 0, .top_p = 0.9 };
    const s_topk: Sampling = .{ .seed = 104, .temperature = 1.0, .top_k = 20, .top_p = 0.95 };
    _ = try dispatchF32(queue, full_p, logits, s_full, position, tok, ids, vocab, 1, 2, io);
    _ = try twoPassF32(queue, slices_p, topk_p, logits, s_topk, position, tok, ids, vocab, 1, ck, ci, 2, io);
    _ = try timeGpuArgmax(queue, argmax_p, logits, tok, vocab, 1, 2, io);
    std.debug.print("\ninkling2 f32 sampling microbench (vocab={d}, ms/batch):\n", .{vocab});
    for ([_]usize{ 1, 4, 16 }) |n_rows| {
        const cpu_full = try timeCpuDraw(gpa, frows, scratch, s_full, vocab, n_rows, 5, io);
        const gpu_full = try dispatchF32(queue, full_p, logits, s_full, position, tok, ids, vocab, n_rows, 20, io);
        const cpu_topk = try timeCpuDraw(gpa, frows, scratch, s_topk, vocab, n_rows, 20, io);
        const gpu_topk = try twoPassF32(queue, slices_p, topk_p, logits, s_topk, position, tok, ids, vocab, n_rows, ck, ci, 20, io);
        const cpu_greedy = try timeCpuDraw(gpa, frows, scratch, null, vocab, n_rows, 20, io);
        const gpu_greedy = try timeGpuArgmax(queue, argmax_p, logits, tok, vocab, n_rows, 20, io);
        std.debug.print(
            "  rows={d:>2} | full: CPU {d:.3} vs GPU {d:.3} (w {d:.3}) | topk20: CPU {d:.3} vs GPU {d:.3} (w {d:.3})\n",
            .{ n_rows, cpu_full, gpu_full.gpu, gpu_full.wall, cpu_topk, gpu_topk.gpu, gpu_topk.wall },
        );
        std.debug.print("          | greedy: CPU {d:.3} vs GPU {d:.3} (w {d:.3})\n", .{ cpu_greedy, gpu_greedy.gpu, gpu_greedy.wall });
    }
}

pub fn main(init: std.process.Init) !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const ink_lib = try mtl.Library.fromSource(device, sources.inkling2_sample, mtl.CompileOptions.mlx());
    defer ink_lib.deinit();
    const old_kernel = generated("sample");
    const old_lib = try mtl.Library.fromSource(device, old_kernel.source, mtl.CompileOptions.mlx());
    defer old_lib.deinit();
    const old = try mtl.Pipeline.init(device, old_lib, old_kernel.function, false);
    defer old.deinit();
    const first = try mtl.Pipeline.init(device, lib, "tf_topk_slices", false);
    defer first.deinit();
    const ck = try device.buffer(rows * slices * kmax * 4, opts);
    defer ck.deinit();
    const ci = try device.buffer(rows * slices * kmax * 4, opts);
    defer ci.deinit();
    std.debug.print("device: {s}\n", .{device.name()});
    var failed: usize = 0;
    for (cases) |c| {
        const p = try mtl.Pipeline.init(device, lib, if (c.map) "tf_sample_full_ids" else "tf_sample_full", false);
        defer p.deinit();
        const row = try gpa.alloc(f32, c.vocab);
        defer gpa.free(row);
        c.fill(row);
        const logits = try device.buffer(rows * c.vocab * 2, opts);
        defer logits.deinit();
        const words = logits.slice(u16, rows * c.vocab);
        for (row, 0..) |*x, i| {
            const h = bf16(x.*);
            x.* = @bitCast(@as(u32, h) << 16);
            for (0..rows) |r| words[r * c.vocab + i] = h;
        }
        const map = try gpa.alloc(u32, c.vocab);
        defer gpa.free(map);
        for (map, 0..) |*m, i| m.* = @intCast(2 * i + 1);
        const ids = try device.buffer(c.vocab * 4, opts);
        defer ids.deinit();
        @memcpy(ids.slice(u32, c.vocab), map);
        const tok = try device.buffer(rows * 4, opts);
        defer tok.deinit();
        _ = try dispatch(queue, p, logits, c.s, tok, ids, c.vocab, true);
        const ms = try dispatch(queue, p, logits, c.s, tok, ids, c.vocab, true);
        const plan = try full.plan(gpa, row, c.s);
        var same: usize = 0;
        for (tok.slice(u32, rows), 0..) |got, r| {
            const want = full.race(row, plan, c.s.seed, position + @as(u32, @intCast(r)), if (c.map) map else null);
            same += @intFromBool(got == want);
            if (got != want) std.debug.print("  row {d}: GPU {d}, host {d}\n", .{ r, got, want });
        }
        failed += rows - same;
        var old_ms: f64 = 0.0;
        if (!c.map and c.vocab == 131072) for (0..2) |_| {
            old_ms = try dispatch(queue, old, logits, c.s, tok, ids, c.vocab, false);
        };
        std.debug.print("{s}: {d}/{d} rows match, tf_sample_full {d:.3} ms, tf_gpu_sample {d:.3} ms ({d} rows)\n", .{ c.name, same, rows, ms, old_ms, rows });
        if (c.s.top_k >= 1 and c.s.top_k <= kmax) {
            const race = try mtl.Pipeline.init(device, lib, if (c.map) "tf_topk_race_ids" else "tf_topk_race", false);
            defer race.deinit();
            @memset(tok.slice(u32, rows), 0xFFFF_FFFF);
            _ = try twoPass(queue, first, race, logits, c.s, tok, ids, c.vocab, ck, ci);
            const two_ms = try twoPass(queue, first, race, logits, c.s, tok, ids, c.vocab, ck, ci);
            var two_same: usize = 0;
            for (tok.slice(u32, rows), 0..) |got, r| {
                const want = full.race(row, plan, c.s.seed, position + @as(u32, @intCast(r)), if (c.map) map else null);
                two_same += @intFromBool(got == want);
                if (got != want) std.debug.print("  two-pass row {d}: GPU {d}, host {d}\n", .{ r, got, want });
            }
            failed += rows - two_same;
            std.debug.print("  two-pass: {d}/{d} rows match, {d:.3} ms\n", .{ two_same, rows, two_ms });
        }
    }
    failed += try checkF32(gpa, device, queue, lib, ink_lib, ck, ci, init.io);
    try benchF32(gpa, device, queue, lib, ink_lib, ck, ci, init.io);
    std.debug.print("{s}\n", .{if (failed == 0) "all draws match" else "draws differ"});
    if (failed > 0) std.process.exit(1);
}
