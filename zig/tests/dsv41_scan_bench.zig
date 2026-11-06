//! The prompt scan at its real shape, kernel by kernel — the instrument ROUND-275 named.
//!
//! Dispatches exactly what `csa2.Pipelines.encode` dispatches for a prompt chunk (the dense-key pass and the
//! tensor-unit sparse attention `dsv41_sparse_nax`, plus the split merge), with the real buffer geometry and
//! no engine around them, so the scan's cost stops being an inference from the 4.57 s prompt.
//!
//! Variants price the three suspects ROUND-275 listed: the packed-key decode (dense_kv alone, and the scan
//! against an already-dense table), the exp/online-softmax and MMA work (the scan alone, swept by key count),
//! and occupancy/latency (the split count, which is the same knob the decode layer's scan uses).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");

/// csa2.Params, field for field (extern, so the kernel sees the engine's layout).
const Params = extern struct {
    rows: u32,
    heads: u32 = 64,
    width: u32 = 512,
    q_stride: u32,
    position: u32,
    chunk_rows: u32,
    window: u32 = 128,
    ring_stride: u32 = 528,
    chunk_stride: u32 = 528,
    main_stride: u32 = 288,
    main_block: u32 = 16,
    ratio: u32,
    compressed_rows: u32,
    topk: u32 = 512,
    index_stride: u32 = 512,
    scale: f32 = 0.04419417382415922,
    splits: u32 = 1,
};

/// csa2.DenseKv, field for field.
const DenseKv = extern struct { rows: u32, width: u32, stride: u32, first_row: u32, source: u32 };

const BLOCK_W = 32; // the scan's threadgroup is (32, 4, 1); 4 simdgroups, 128 width each

const Case = struct {
    stream_rows: u32,
    heads: u32 = 64,
    width: u32 = 512,
    window: u32 = 128,
    ratio: u32,
    compressed_rows: u32,
    topk: u32,
    parts_bytes: u64,

    q: mtl.Buffer,
    ring: mtl.Buffer,
    chunk: mtl.Buffer,
    main: mtl.Buffer,
    idx: mtl.Buffer,
    sink: mtl.Buffer,
    out: mtl.Buffer,
    dense: mtl.Buffer,

    dense_kv: mtl.Pipeline,
    scan: mtl.Pipeline,
    merge: mtl.Pipeline,
    parts: mtl.Buffer,
    alias: bool = false,

    fn params(c: *const Case, ratio: u32, compressed_rows: u32, topk: u32, splits: u32) Params {
        return .{
            .rows = c.stream_rows,
            .heads = c.heads,
            .width = c.width,
            .q_stride = c.heads * c.width,
            .position = 0,
            .chunk_rows = c.stream_rows,
            .window = c.window,
            .ring_stride = 528,
            .chunk_stride = 528,
            .main_stride = 288,
            .main_block = 16,
            .ratio = ratio,
            .compressed_rows = compressed_rows,
            .topk = topk,
            .index_stride = @max(topk, 1),
            .scale = 1.0 / @sqrt(512.0),
            .splits = splits,
        };
    }

    /// The dense-key pre-pass over the three sections encode() decodes: the ring, this chunk, the compressed pool.
    fn densePass(c: *const Case, e: mtl.ComputeEncoder, ratio: u32, compressed_rows: u32) void {
        const sections = [_]struct { rows: u32, stride: u32, first: u32, source: u32, buf: mtl.Buffer }{
            .{ .rows = 0, .stride = 528, .first = 0, .source = 0, .buf = c.ring }, // position 0: no ring rows
            .{ .rows = c.stream_rows, .stride = 528, .first = c.window, .source = 0, .buf = c.chunk },
            .{ .rows = if (ratio == 0 or compressed_rows == 0) 0 else compressed_rows, .stride = 288, .first = c.window + c.stream_rows, .source = 1, .buf = c.main },
        };
        e.setPipeline(c.dense_kv);
        for (sections) |sec| {
            if (sec.rows == 0) continue;
            e.setBuffer(sec.buf, 0, 0);
            e.setBuffer(c.dense, 0, 1);
            e.setValue(DenseKv{ .rows = sec.rows, .width = c.width, .stride = sec.stride, .first_row = sec.first, .source = sec.source }, 2);
            e.dispatchThreads(mtl.Size.of(sec.rows, c.width / 4, 1), mtl.Size.of(BLOCK_W, 4, 1));
        }
    }

    /// dsv41_sparse_nax (+ its merge when split), the dispatch encode() makes.
    fn scanPass(ct: *const Case, e: mtl.ComputeEncoder, p: Params) void {
        e.setPipeline(ct.scan);
        e.setBuffer(ct.q, 0, 0);
        e.setBuffer(ct.ring, 0, 1);
        e.setBuffer(ct.chunk, 0, 2);
        e.setBuffer(ct.main, 0, 3);
        e.setBuffer(ct.idx, 0, 4);
        e.setBuffer(ct.sink, 0, 5);
        e.setBuffer(if (ct.alias) ct.q else ct.out, 0, 6);
        e.setValue(p, 7);
        e.setBuffer(ct.dense, 0, 8);
        e.setBuffer(ct.parts, 0, 9);
        e.dispatchGroups(mtl.Size.of(ct.heads / 16, p.rows, p.splits), mtl.Size.of(BLOCK_W, 4, 1));
        if (p.splits > 1) {
            e.setPipeline(ct.merge);
            e.setBuffer(ct.parts, 0, 0);
            e.setBuffer(ct.out, 0, 1);
            e.setBuffer(ct.sink, 0, 2);
            e.setValue(p, 3);
            e.dispatchGroups(mtl.Size.of(ct.heads / 16, p.rows, 1), mtl.Size.of(BLOCK_W, 4, 1));
        }
    }
};

const Variant = enum {
    dense_pool, // the pre-pass alone, at the pool layer's ratio
    scan_pool, // the scan alone, pool layer (ratio 2, 512 picks)
    pair_pool, // what a pool layer dispatches: pre-pass, then scan
    pair_pool_split2,
    pair_pool_split4,
    pair_pool_split8,
    pair_full, // a compressed layer (ratio 1, 512 picks; the pool is the chunk itself)
    pair_swa, // layers 0-1 and the MTP layers: window only, no pool
    scan_pool_keys256,
    scan_pool_keys128,
    scan_swa,
};

fn encode(c: *const Case, v: Variant, e: mtl.ComputeEncoder) void {
    const pr = c.params(c.ratio, c.compressed_rows, c.topk, 1);
    switch (v) {
        .dense_pool => c.densePass(e, c.ratio, c.compressed_rows),
        .scan_pool => c.scanPass(e, pr),
        .pair_pool => {
            c.densePass(e, c.ratio, c.compressed_rows);
            c.scanPass(e, pr);
        },
        .pair_pool_split2 => {
            c.densePass(e, c.ratio, c.compressed_rows);
            c.scanPass(e, c.params(c.ratio, c.compressed_rows, c.topk, 2));
        },
        .pair_pool_split4 => {
            c.densePass(e, c.ratio, c.compressed_rows);
            c.scanPass(e, c.params(c.ratio, c.compressed_rows, c.topk, 4));
        },
        .pair_pool_split8 => {
            c.densePass(e, c.ratio, c.compressed_rows);
            c.scanPass(e, c.params(c.ratio, c.compressed_rows, c.topk, 8));
        },
        .pair_full => {
            c.densePass(e, 1, c.stream_rows);
            c.scanPass(e, c.params(1, c.stream_rows, c.topk, 1));
        },
        .pair_swa => {
            c.densePass(e, 0, 0);
            c.scanPass(e, c.params(0, 0, 0, 1));
        },
        .scan_pool_keys256 => c.scanPass(e, c.params(c.ratio, c.compressed_rows, 256, 1)),
        .scan_pool_keys128 => c.scanPass(e, c.params(c.ratio, c.compressed_rows, 128, 1)),
        .scan_swa => c.scanPass(e, c.params(0, 0, 0, 1)),
    }
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

/// Median ms a dispatch, over `trials` command buffers of `reps` dependent dispatches each.
fn time(c: *const Case, queue: mtl.Queue, v: Variant, reps: usize, trials: usize, scratch: []f64) !f64 {
    var n: usize = 0;
    for (0..trials) |_| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..reps) |_| encode(c, v, e);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.debug.print("{s}: failed: {s}\n", .{ @tagName(v), msg });
            return error.DispatchFailed;
        }
        scratch[n] = (cb.gpuEnd() - cb.gpuStart()) * 1e3 / @as(f64, @floatFromInt(reps));
        n += 1;
    }
    return median(scratch[0..n]);
}

pub fn main() !void {
    const gpa = std.heap.page_allocator;
    const scatter = std.c.getenv("TF_BENCH_SCATTER") != null;
    const alias = std.c.getenv("TF_BENCH_ALIAS") != null;
    const bigdense = std.c.getenv("TF_BENCH_BIGDENSE") != null;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const tensor_units = device.tensorUnits();

    var options = mtl.CompileOptions.mlx();
    options.math_functions = .precise;
    // The engine's own source text (csa2.Pipelines.load), nax.h inlined for this GPU.
    const nax_text = try std.mem.replaceOwned(u8, gpa, ks.deepseek_v41_csa2_nax, "#include \"../nax.h\"", ks.nax);
    defer gpa.free(nax_text);
    const nax_src: []const u8 = if (tensor_units) nax_text else blk: {
        break :blk try std.mem.concat(gpa, u8, &.{ "#define TF_SIMD_FRAGS 1\n", nax_text });
    };
    defer if (!tensor_units) gpa.free(nax_src);
    const text = try std.mem.concat(gpa, u8, &.{ ks.deepseek_v41_kv, ks.deepseek_v41_csa2, nax_src });
    defer gpa.free(text);
    const lib = try mtl.Library.fromSource(device, text, options);
    const dense_kv = try mtl.Pipeline.init(device, lib, "dsv41_dense_kv", false);
    const scan = try mtl.Pipeline.init(device, lib, "dsv41_sparse_nax", false);
    const merge = try mtl.Pipeline.init(device, lib, "dsv41_sparse_merge", false);

    // The prompt, and the pool layer's geometry: 3,619 tokens, ratio 2, 512 picks, window 128.
    const rows: u32 = 3619;
    const heads: u32 = 64;
    const width: u32 = 512;
    const window: u32 = 128;
    const ratio: u32 = 2;
    const compressed: u32 = rows / ratio;
    const topk: u32 = 512;
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

    const q = try device.buffer(@as(usize, rows) * heads * width * 2, opts);
    defer q.deinit();
    const ring = try device.buffer(@as(usize, window) * 528, opts);
    defer ring.deinit();
    const chunk = try device.buffer(@as(usize, rows) * 528, opts);
    defer chunk.deinit();
    const pool_main = try device.buffer(@as(usize, compressed) * 288, opts);
    defer pool_main.deinit();
    const idx = try device.buffer(@as(usize, rows) * topk * 4, opts);
    defer idx.deinit();
    const sink = try device.buffer(@as(usize, heads) * 4, opts);
    defer sink.deinit();
    const out = try device.buffer(@as(usize, rows) * heads * width * 2, opts);
    defer out.deinit();
    const dense_rows: usize = if (bigdense) 128 + 15872 + 65536 else window + rows + compressed;
    const dense = try device.buffer(dense_rows * width * 2, opts);
    defer dense.deinit();
    // splits' partials: rows x splits x (heads/16) x 4 simdgroups x 32 lanes x 68 floats
    const parts_bytes: usize = @as(usize, rows) * 8 * (heads / 16) * 4 * 32 * 68 * 4;
    const parts = try device.buffer(parts_bytes, opts);
    defer parts.deinit();

    // Values: in-range bf16 queries and sinks; packed keys any bytes do (the decoders are total, and the
    // timing does not read the outputs); indices name visible compressed rows, newest first.
    for (q.slice(u16, @as(usize, rows) * heads * width), 0..) |*v, i| {
        const x: i32 = @intCast((i * 2654435761) % 2048);
        v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(x)) / 1024.0 - 1.0)) >> 16);
    }
    for (ring.slice(u8, @as(usize, window) * 528), 0..) |*v, i| v.* = @truncate((i * 37) % 251);
    for (chunk.slice(u8, @as(usize, rows) * 528), 0..) |*v, i| v.* = @truncate((i * 53) % 251);
    for (pool_main.slice(u8, @as(usize, compressed) * 288), 0..) |*v, i| v.* = @truncate((i * 29) % 251);
    for (sink.slice(f32, heads), 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i % 5));
    const is = idx.slice(i32, @as(usize, rows) * topk);
    for (0..rows) |r| {
        const visible: i32 = @intCast(@min(compressed, (r + 1) / ratio));
        for (0..topk) |s| {
            if (scatter) {
                const p: i32 = if (@as(i32, @intCast(s)) >= visible) -1 else @intCast((@as(u64, s) * 7919 + @as(u64, r) * 104729) % @as(u64, @intCast(@max(visible, 1))));
                is[r * topk + s] = p;
            } else {
                const p = visible - 1 - @as(i32, @intCast(s));
                is[r * topk + s] = if (p >= 0) p else -1;
            }
        }
    }

    const c = Case{
        .stream_rows = rows,
        .ratio = ratio,
        .compressed_rows = compressed,
        .topk = topk,
        .parts_bytes = parts_bytes,
        .q = q,
        .ring = ring,
        .chunk = chunk,
        .main = pool_main,
        .idx = idx,
        .sink = sink,
        .out = out,
        .dense = dense,
        .parts = parts,
        .dense_kv = dense_kv,
        .scan = scan,
        .merge = merge,
        .alias = alias,
    };

    const variants = [_]Variant{ .dense_pool, .scan_pool, .pair_pool, .scan_pool_keys256, .scan_pool_keys128, .pair_pool_split2, .pair_pool_split4, .pair_pool_split8, .pair_full, .pair_swa };
    const reps: usize = 3;
    const trials: usize = 5;
    var scratch: [32]f64 = undefined;
    std.debug.print("scan bench: rows {d}, heads {d}, width {d}, window {d}, topk {d}, ratio {d}, compressed_rows {d}, tensor units {}, scatter {} alias {} bigdense {}\n", .{ rows, heads, width, window, topk, ratio, compressed, tensor_units, scatter, alias, bigdense });
    std.debug.print("            {d} keys/row at the pool layer, {d} at the SWA layers; 4,570 threadgroups a layer at splits 1\n", .{ window + topk, window });
    std.debug.print("  {s:<20} {s:>10} {s:>12} {s:>12}\n", .{ "variant", "ms", "us/row", "us/key" });
    var got: [variants.len]f64 = undefined;
    for (variants, 0..) |v, i| {
        _ = try time(&c, queue, v, 1, 1, &scratch); // warm
        got[i] = try time(&c, queue, v, reps, trials, &scratch);
        const keys: f64 = switch (v) {
            .scan_pool_keys256 => window + 256,
            .scan_pool_keys128 => window + 128,
            .scan_swa, .pair_swa => window,
            else => if (v == .pair_full) window + topk else window + topk,
        };
        std.debug.print("  {s:<20} {d:>10.3} {d:>12.2} {d:>12.3}\n", .{ @tagName(v), got[i], got[i] * 1e3 / @as(f64, @floatFromInt(rows)), got[i] * 1e6 / (@as(f64, @floatFromInt(rows)) * keys) });
    }

    // The 40 layers' weighted total, from the measured variants (2 SWA + 18 pool + 20 compressed).
    const pair_pool_ms = got[2];
    const pair_full_ms = got[8];
    const pair_swa_ms = got[9];
    const total = 2.0 * pair_swa_ms + 18.0 * pair_pool_ms + 20.0 * pair_full_ms;
    std.debug.print("\n  a 3,619-token prompt's attention, 40 layers by the measured variants: {d:.1} ms = {d:.3} s\n", .{ total, total * 1e-3 });
    std.debug.print("  split sweep at the pool layer: 1 {d:.3} / 2 {d:.3} / 4 {d:.3} / 8 {d:.3} ms\n", .{ got[2], got[5], got[6], got[7] });
    std.debug.print("  key sweep (scan alone): {d} keys {d:.3} / {d} {d:.3} / {d} {d:.3} / {d} {d:.3} ms\n", .{ window, got[9], window + 128, got[4], window + 256, got[3], window + 512, got[1] });
    const macs = 40.0 * @as(f64, @floatFromInt(rows)) * 64.0 * 640.0 * 512.0 * 2.0;
    std.debug.print("  scan MACs over the 40 layers {d:.3} TMAC at {d:.1} TMAC/s (tensor peak 41.5)\n", .{ macs / 1e12, macs / (total * 1e-3) / 1e12 });
}
