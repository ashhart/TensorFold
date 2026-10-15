//! Standalone deterministic EXL3 fixtures, independent FP64-dot verification and relative GPU operator timings.
const std = @import("std");
const mtl = @import("metal");
const ex = @import("exl3_projection");
const sources = @import("kernel_sources");
const oracle = @import("exl3_projection_oracle.zig");
const Decode = enum { arithmetic, hoisted_decode };
const codebook_entries = 3 * 65536;
const codebook_bytes = codebook_entries * @sizeOf(f16);

const Shape = struct { k: usize, n: usize };
const shapes = [_]Shape{
    .{ .k = 5120, .n = 2304 },  .{ .k = 2304, .n = 5120 },
    .{ .k = 1280, .n = 32768 }, .{ .k = 5120, .n = 129280 },
    .{ .k = 4096, .n = 1024 },  .{ .k = 8192, .n = 5120 },
    .{ .k = 5120, .n = 1280 },  .{ .k = 5120, .n = 512 },
    .{ .k = 6144, .n = 25600 },
};
const Mode = enum { one, routed6, prefill };
const Settings = struct {
    warmup: usize = 2,
    repeats: usize = 5,
    prefill: usize = 32,
    candidate: Decode = .hoisted_decode,
    mode: ?Mode = null,
    qualify_only: bool = false,
    width: u16 = 6,
    book: ex.format.Codebook = .mul1,
    output: ?ex.Output = null,
};
const limit = 2 * 1024 * 1024 * 1024;

const Owned = struct {
    matrix: ex.Matrix,
    buffers: [4]mtl.Buffer,

    fn bytes(s: Shape, halves: u16) usize {
        const ns = @as(usize, halves) * 16 + 2;
        return 16 + (s.k / 16) * ((s.n / 16) * ns + 16) + (s.k + 2 * s.n) * 2 + 48;
    }
    fn init(device: mtl.Device, s: Shape, halves: u16, book: ex.format.Codebook, seed: u32) !Owned {
        const ns = @as(usize, halves) * 16 + 2;
        const ks = (s.n / 16) * ns + 16;
        const t = try device.buffer(16 + (s.k / 16) * ks, mtl.ResourceOptions.shared);
        errdefer t.deinit();
        const su = try device.buffer(s.k * 2 + 16, mtl.ResourceOptions.shared);
        errdefer su.deinit();
        const sv = try device.buffer(s.n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer sv.deinit();
        const bias = try device.buffer(s.n * 2 + 16, mtl.ResourceOptions.shared);
        errdefer bias.deinit();
        // Any circular packed bitstream is valid; independent deterministic bytes also fill padding.
        var state = seed;
        for (t.slice(u16, t.length() / 2)) |*v| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            v.* = @truncate(state);
        }
        for (su.slice(f16, s.k + 8)[8..], 0..) |*v, i| v.* = if (i % 3 == 0) -0.75 else 1.125;
        for (sv.slice(f16, s.n + 8)[8..], 0..) |*v, i| v.* = if (i % 5 == 0) -0.25 else 0.875;
        for (bias.slice(f16, s.n + 8)[8..], 0..) |*v, i|
            v.* = @floatCast(@as(f32, @floatFromInt(i % 7)) / 16);
        var matrix: ex.Matrix = .{
            .trellis = .{ .buffer = t, .offset = 16 },
            .suh = .{ .buffer = su, .offset = 16 },
            .svh = .{ .buffer = sv, .offset = 16 },
            .bias = .{ .buffer = bias, .offset = 16 },
            .k = s.k,
            .n = s.n,
            .width = .{ .halves = halves },
            .codebook = book,
            .tile_k_stride = ks,
            .tile_n_stride = ns,
        };
        if (seed % 2 == 0) {
            matrix.suh_format = .packed_signs;
            matrix.svh_format = .packed_signs;
            @memset(su.slice(u16, s.k + 8)[8..][0 .. s.k / 16], 0x9635);
            @memset(sv.slice(u16, s.n + 8)[8..][0 .. s.n / 16], 0x4a17);
        }
        return .{ .matrix = matrix, .buffers = .{ t, su, sv, bias } };
    }
    fn deinit(w: Owned) void {
        for (w.buffers) |b| b.deinit();
    }
};

fn finish(cb: mtl.CommandBuffer, e: mtl.ComputeEncoder) !f64 {
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("EXL3 GPU failure: {s}", .{msg});
        return error.Exl3GpuFailure;
    }
    const seconds = cb.gpuSeconds();
    if (!std.math.isFinite(seconds) or seconds <= 0) return error.GpuTimestampUnavailable;
    return seconds;
}

fn checkCodebooks(device: mtl.Device, queue: mtl.Queue) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const lib = try mtl.Library.fromSource(device, sources.core_exl3_projection, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const check = try mtl.Pipeline.init(device, lib, "tf_exl3_codebook_check", false);
    defer check.deinit();
    const arithmetic = try device.buffer(codebook_bytes, mtl.ResourceOptions.shared);
    defer arithmetic.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    e.setPipeline(check);
    e.setBuffer(arithmetic, 0, 0);
    e.dispatchThreads(.{ .width = codebook_entries }, .{ .width = 256 });
    _ = try finish(cb, e);
    var wrong: usize = 0;
    const books = [_]ex.format.Codebook{ .inst3, .mcg, .mul1 };
    for (arithmetic.slice(u16, codebook_entries), 0..) |actual, i| {
        const expected = ex.format.codebookValue(books[i / 65536], @intCast(i % 65536));
        if (actual != @as(u16, @bitCast(expected))) wrong += 1;
    }
    std.debug.print("EXL3 codebook_states={d} fp16_bit_mismatches={d} sample_bytes={d}\n", .{ codebook_entries, wrong, arithmetic.length() });
    if (wrong != 0) return error.CodebookParityFailure;
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

const Work = struct {
    matrix: ex.Matrix,
    experts: ex.Experts,
    indexed: bool,
    x: ex.View,
    ys: [2]ex.View,
    scratch: ex.View,
    rows: ex.Rows,

    fn run(w: Work, queue: mtl.Queue, pipelines: *const [2]ex.Pipelines, decode: Decode) !f64 {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const index: usize = if (decode == .arithmetic) 0 else 1;
        const p = &pipelines[index];
        const y = w.ys[index];
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        if (w.indexed) try ex.encodeIndexed(p, e, w.experts, w.x, y, w.scratch, w.rows) else try ex.encode(p, e, w.matrix, w.x, y, w.scratch, w.rows);
        return finish(cb, e);
    }
    fn parity(w: Work, n: usize) !void {
        var wrong: usize = 0;
        var max_abs: f32 = 0;
        const size: usize = if (w.rows.output == .f32) 4 else 2;
        const length = (w.rows.count - 1) * w.rows.y_stride + n;
        const a = w.ys[0].buffer.contents() + w.ys[0].offset;
        const b = w.ys[1].buffer.contents() + w.ys[1].offset;
        for (0..w.rows.count) |row| for (0..n) |col| {
            const i = row * w.rows.y_stride + col;
            if (size == 4) {
                const av = @as([*]const f32, @ptrCast(@alignCast(a)))[0..length][i];
                const bv = @as([*]const f32, @ptrCast(@alignCast(b)))[0..length][i];
                if (!std.math.isFinite(av) or !std.math.isFinite(bv)) return error.NonFiniteOutput;
                if (@as(u32, @bitCast(av)) != @as(u32, @bitCast(bv))) wrong += 1;
                max_abs = @max(max_abs, @abs(av - bv));
            } else {
                const av = @as([*]const u16, @ptrCast(@alignCast(a)))[0..length][i];
                const bv = @as([*]const u16, @ptrCast(@alignCast(b)))[0..length][i];
                const af: f32 = @bitCast(@as(u32, av) << 16);
                const bf: f32 = @bitCast(@as(u32, bv) << 16);
                if (!std.math.isFinite(af) or !std.math.isFinite(bf)) return error.NonFiniteOutput;
                if (av != bv) wrong += 1;
                max_abs = @max(max_abs, @abs(af - bf));
            }
        };
        std.debug.print(" parity_bit_mismatches={d} max_abs={d:.9}\n", .{ wrong, max_abs });
        if (wrong != 0) return error.OutputParityFailure;
    }
};

fn benchmark(device: mtl.Device, queue: mtl.Queue, p: *const [2]ex.Pipelines, s: Shape, mode: Mode, settings: Settings) !void {
    const count: usize = switch (mode) {
        .one => 1,
        .routed6 => 6,
        .prefill => settings.prefill,
    };
    const books = [_]ex.format.Codebook{ settings.book, .inst3, .mul1, .mcg, .mul1, .inst3 };
    const halves = [_]u16{ settings.width, 4, 3, 6, 5, 6 };
    const expert_count: usize = if (mode == .routed6) 6 else 1;
    const output: ex.Output = settings.output orelse (if (s.n == 129280) ex.Output.f32 else ex.Output.bf16);
    const element_bytes: usize = if (output == .f32) 4 else 2;
    const x_stride = s.k + 16;
    const y_stride = s.n + 8;
    const scratch_bytes = try ex.scratchBytes(s.k, s.n, count);
    var total = count * x_stride * 2 + 32 + 2 * (count * y_stride * element_bytes + 32) +
        scratch_bytes + expert_count * @sizeOf(ex.ExpertDescriptor) + count * 12;
    for (0..expert_count) |i| total += Owned.bytes(s, halves[i]);
    if (total >= limit) return error.BenchmarkAllocationLimit;
    var owned: [6]Owned = undefined;
    var initialized: usize = 0;
    defer for (owned[0..initialized]) |w| w.deinit();
    for (0..expert_count) |i| {
        owned[i] = try Owned.init(device, s, halves[i], books[i], @intCast(0x173819 + i));
        initialized += 1;
    }
    const x = try device.buffer(count * x_stride * 2 + 32, mtl.ResourceOptions.shared);
    defer x.deinit();
    for (x.slice(u16, count * x_stride + 16), 0..) |*v, i|
        v.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 32);
    var ys: [2]mtl.Buffer = undefined;
    ys[0] = try device.buffer(count * y_stride * element_bytes + 32, mtl.ResourceOptions.shared);
    defer ys[0].deinit();
    ys[1] = try device.buffer(count * y_stride * element_bytes + 32, mtl.ResourceOptions.shared);
    defer ys[1].deinit();
    for (ys) |y| @memset(y.slice(u8, y.length()), 0xa5);
    const scratch = try device.buffer(scratch_bytes, mtl.ResourceOptions.shared);
    defer scratch.deinit();
    const descriptors = try device.buffer(expert_count * @sizeOf(ex.ExpertDescriptor), mtl.ResourceOptions.shared);
    defer descriptors.deinit();
    for (owned[0..expert_count], descriptors.slice(ex.ExpertDescriptor, expert_count)) |w, *d|
        d.* = try ex.descriptor(w.matrix);
    const routes = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer routes.deinit();
    const gather = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer gather.deinit();
    const scatter = try device.buffer(count * 4, mtl.ResourceOptions.shared);
    defer scatter.deinit();
    for (0..count) |i| {
        routes.slice(u32, count)[i] = @intCast(i % expert_count);
        gather.slice(u32, count)[i] = 0;
        scatter.slice(u32, count)[i] = @intCast(count - 1 - i);
    }
    const set = try device.residencySet(40);
    defer set.deinit();
    for (owned[0..expert_count]) |w| for (w.buffers) |b| set.add(b);
    for ([_]mtl.Buffer{ x, ys[0], ys[1], scratch, descriptors, routes, gather, scatter }) |b| set.add(b);
    set.commit();
    set.requestResidency();
    queue.addResidencySet(set);
    defer queue.removeResidencySet(set);
    const w: Work = .{
        .matrix = owned[0].matrix,
        .experts = .{ .descriptors = .{ .buffer = descriptors }, .count = expert_count, .k = s.k, .n = s.n },
        .indexed = mode == .routed6,
        .x = .{ .buffer = x, .offset = 32 },
        .ys = .{ .{ .buffer = ys[0], .offset = 32 }, .{ .buffer = ys[1], .offset = 32 } },
        .scratch = .{ .buffer = scratch },
        .rows = .{ .count = count, .x_stride = x_stride, .y_stride = y_stride, .output = output, .routes = if (mode == .routed6) .{ .ids = .{ .buffer = routes }, .repeat = 6 } else null, .input_ids = if (mode == .routed6) .{ .buffer = gather } else null, .output_ids = if (mode == .routed6) .{ .buffer = scatter } else null, .input_rows = if (mode == .routed6) 1 else 0, .output_rows = if (mode == .routed6) count else 0 },
    };
    for (0..settings.warmup) |_| {
        _ = try w.run(queue, p, .arithmetic);
        _ = try w.run(queue, p, settings.candidate);
    }
    var totals = [_]f64{ 0, 0 };
    var best = [_]f64{ std.math.inf(f64), std.math.inf(f64) };
    for (0..settings.repeats) |r| {
        // Alternate order to avoid consistently favoring either path's cache/thermal state.
        for (0..2) |j| {
            const index = (r + j) % 2;
            const seconds = try w.run(queue, p, if (index == 0) .arithmetic else settings.candidate);
            totals[index] += seconds;
            best[index] = @min(best[index], seconds);
        }
        try w.parity(s.n);
    }
    var references: [6]ex.Matrix = undefined;
    for (owned[0..expert_count], references[0..expert_count]) |weight, *reference| reference.* = weight.matrix;
    try oracle.verify(std.heap.page_allocator, references[0..expert_count], w.x, w.ys, w.rows);
    const repeats: f64 = @floatFromInt(settings.repeats);
    std.debug.print("EXL3 k={d} n={d} mode={s} rows={d} output={s} candidate={s} allocated_bytes={d} " ++
        "warmup={d} repeats={d} baseline_mean_gpu_s={d:.9} candidate_mean_gpu_s={d:.9} " ++
        "baseline_best_gpu_s={d:.9} candidate_best_gpu_s={d:.9} mean_speedup={d:.4}\n", .{ s.k, s.n, @tagName(mode), count, @tagName(output), @tagName(settings.candidate), total, settings.warmup, settings.repeats, totals[0] / repeats, totals[1] / repeats, best[0], best[1], totals[0] / totals[1] });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var settings: Settings = .{};
    var shape_index: ?usize = null;
    var i: usize = 1;
    while (i < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--help")) {
            std.debug.print("tf-exl3-bench [--qualify-only 0|1] [--shape 0..8] [--mode one|routed6|prefill]\n" ++
                "  [--warmup 0..100] [--repeats 1..100] [--prefill 1..128] [--candidate hoisted_decode]\n" ++
                "Always verifies all widths/codebooks on complete 256x128 fixtures in BF16 and F32.\n" ++
                "Large-shape oracle samples first/middle/last 128 columns and at most two rows.\n" ++
                "Allocated Metal buffers are capped below 2 GiB; GPU times exclude oracle/allocation.\n", .{});
            return;
        }
        if (i + 1 >= args.len) return error.OptionValueRequired;
        if (std.mem.eql(u8, args[i], "--candidate")) {
            settings.candidate = std.meta.stringToEnum(Decode, args[i + 1]) orelse return error.UnknownCandidate;
            if (settings.candidate == .arithmetic) return error.InvalidBenchmarkSettings;
            continue;
        }
        if (std.mem.eql(u8, args[i], "--mode")) {
            settings.mode = std.meta.stringToEnum(Mode, args[i + 1]) orelse return error.UnknownMode;
            continue;
        }
        const v = try std.fmt.parseInt(usize, args[i + 1], 10);
        if (std.mem.eql(u8, args[i], "--repeats")) settings.repeats = v else if (std.mem.eql(u8, args[i], "--warmup")) settings.warmup = v else if (std.mem.eql(u8, args[i], "--prefill")) settings.prefill = v else if (std.mem.eql(u8, args[i], "--shape")) shape_index = v else if (std.mem.eql(u8, args[i], "--qualify-only") and v <= 1) settings.qualify_only = v == 1 else return error.UnknownOption;
    }
    if (settings.repeats == 0 or settings.repeats > 100 or settings.warmup > 100 or
        settings.prefill == 0 or settings.prefill > 128 or
        (shape_index != null and shape_index.? >= shapes.len)) return error.InvalidBenchmarkSettings;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p: [2]ex.Pipelines = undefined;
    p[0] = try ex.Pipelines.load(device);
    defer p[0].deinit();
    p[1] = try ex.Pipelines.load(device);
    defer p[1].deinit();
    const lib = try mtl.Library.fromSource(device, sources.core_exl3_projection, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const baseline = try mtl.Pipeline.init(device, lib, "tf_exl3_matmul", false);
    p[0].matmul.deinit();
    p[0].matmul = baseline;
    try checkCodebooks(device, queue);
    for ([_]ex.format.Codebook{ .inst3, .mcg, .mul1 }) |book| {
        for ([_]u16{ 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16 }) |width| {
            if (width % 2 == 1 and book != .mul1) continue;
            for ([_]ex.Output{ .bf16, .f32 }) |output| {
                var fixture = settings;
                fixture.width = width;
                fixture.book = book;
                fixture.output = output;
                fixture.prefill = 9;
                try benchmark(device, queue, &p, .{ .k = 256, .n = 128 }, .prefill, fixture);
            }
        }
    }
    try benchmark(device, queue, &p, .{ .k = 256, .n = 128 }, .routed6, settings);
    if (settings.qualify_only) return;
    for (shapes, 0..) |s, si| {
        if (shape_index != null and shape_index.? != si) continue;
        for ([_]Mode{ .one, .routed6, .prefill }) |mode| {
            if (settings.mode != null and settings.mode.? != mode) continue;
            try benchmark(device, queue, &p, s, mode, settings);
        }
    }
}

test {
    _ = oracle;
}
