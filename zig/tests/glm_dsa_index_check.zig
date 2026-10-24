//! The DSA indexer's Metal scan on synthetic keys: the blocked kernel's scores and top-k against the row kernel's.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const kn = g.kernels;
const wt = g.weights;
const a = std.heap.page_allocator;

const heads: u32 = 32;
const dim: u32 = 128;

fn bf16(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}

fn ref(b: mtl.Buffer) kn.Ref {
    return .{ .buf = b };
}

var rows: u32 = 4;
var context: u32 = 4096;
var reps: u32 = 30;

const Data = struct {
    q: mtl.Buffer,
    proj: mtl.Buffer,
    cache: mtl.Buffer,
    rows: mtl.Buffer,
    score: mtl.Buffer,
    keys: mtl.Buffer,
    ia: kn.IndexArgs,
    last: u32,
};

/// The score kernel alone, the blocked one or the row one, and the top-k that follows.
fn scan(d: *const Data, K: *const g.layer.Kernels, queue: mtl.Queue, blocked: bool, topk: bool) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    if (blocked) kn.indexScoresBlock(&K.k, e, ref(d.q), ref(d.proj), ref(d.cache), ref(d.rows), d.ia, ref(d.score), ref(K.inv), d.last) else kn.indexScoresRows(&K.k, e, ref(d.q), ref(d.proj), ref(d.cache), ref(d.rows), d.ia, ref(d.score), ref(K.inv), d.last);
    if (topk) {
        e.barrier();
        kn.indexTopkOnly(&K.k, e, ref(d.rows), d.ia, ref(d.score), ref(d.keys));
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("the GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1000;
}

/// The median of `reps` scans, in ms.
fn median(d: *const Data, K: *const g.layer.Kernels, queue: mtl.Queue, blocked: bool, topk: bool) !f64 {
    var ms: [256]f64 = undefined;
    const n = @min(reps, 256);
    for (0..n) |i| ms[i] = try scan(d, K, queue, blocked, topk);
    std.mem.sort(f64, ms[0..n], {}, std.sort.asc(f64));
    return ms[n / 2];
}

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len > 1) rows = try std.fmt.parseInt(u32, argv[1], 10);
    if (argv.len > 2) context = try std.fmt.parseInt(u32, argv[2], 10);
    if (argv.len > 3) reps = try std.fmt.parseInt(u32, argv[3], 10);
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var c: g.config.Config = .{};
    var K = try g.layer.Kernels.load(a, device, &c);
    defer K.deinit();

    const q_stride = heads * dim;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const q = try device.buffer(rows * q_stride * 4, mtl.ResourceOptions.shared);
    const proj = try device.buffer(rows * wt.proj_width * 4, mtl.ResourceOptions.shared);
    const cache = try device.buffer(context * dim * 2, mtl.ResourceOptions.shared);
    const rows_buf = try device.buffer(rows * @sizeOf(kn.Row), mtl.ResourceOptions.shared);
    const score = try device.buffer(rows * context * 4, mtl.ResourceOptions.shared);
    const keys = try device.buffer(rows * c.i_topk * 4, mtl.ResourceOptions.shared);

    for (q.slice(f32, rows * q_stride)) |*v| v.* = rnd.floatNorm(f32) * 0.5;
    for (proj.slice(f32, rows * wt.proj_width)) |*v| v.* = rnd.floatNorm(f32) * 0.5;
    for (cache.slice(u16, context * dim)) |*v| v.* = bf16(rnd.floatNorm(f32) * 0.1);
    for (rows_buf.slice(kn.Row, rows), 0..) |*row, r| {
        row.* = .{ .slot = 0, .pos = context - 1 - @as(u32, @intCast(r)), .list = @as(u32, @intCast(r)) * c.i_topk };
    }
    const d: Data = .{
        .q = q,
        .proj = proj,
        .cache = cache,
        .rows = rows_buf,
        .score = score,
        .keys = keys,
        .ia = .{
            .rows = rows,
            .slot_keys = context,
            .proj_stride = wt.proj_width,
            .k_col = wt.wk_col,
            .w_col = wt.wp_col,
            .q_stride = q_stride,
            .score_stride = context,
            .key_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim))),
            .head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(heads))),
        },
        .last = context - 1,
    };

    // the row kernel's scores and lists, then the blocked kernel's, on the same data
    _ = try scan(&d, &K, queue, false, true);
    const old_scores = try a.dupe(f32, d.score.slice(f32, rows * context));
    const old_keys = try a.dupe(u32, d.keys.slice(u32, rows * c.i_topk));
    _ = try scan(&d, &K, queue, true, true);
    const new_scores = try a.dupe(f32, d.score.slice(f32, rows * context));
    const new_keys = try a.dupe(u32, d.keys.slice(u32, rows * c.i_topk));

    // the scores are the same bits, the lists the same indices in the same order
    var diff: usize = 0;
    for (0..rows) |r| {
        const pos = context - 1 - @as(usize, @intCast(r));
        for (0..pos + 1) |k| {
            if (@as(u32, @bitCast(old_scores[r * context + k])) != @as(u32, @bitCast(new_scores[r * context + k]))) diff += 1;
        }
    }
    if (diff != 0) {
        std.debug.print("the blocked scan's scores differ in {d} places\n", .{diff});
        return error.ScoresDiffer;
    }
    for (old_keys, new_keys) |o, n| {
        if (o != n) {
            std.debug.print("the blocked scan's key lists differ: {d} and {d}\n", .{ o, n });
            return error.KeysDiffer;
        }
    }
    for (rows_buf.slice(kn.Row, rows)) |row| {
        if (row.nkeys != c.i_topk) {
            std.debug.print("a row kept {d} keys, not {d}\n", .{ row.nkeys, c.i_topk });
            return error.BadKeyCount;
        }
    }

    const row_ms = try median(&d, &K, queue, false, false);
    const block_ms = try median(&d, &K, queue, true, false);
    const row_full = try median(&d, &K, queue, false, true);
    const block_full = try median(&d, &K, queue, true, true);
    std.debug.print("the DSA scan over {d} keys for {d} rows: the row kernel {d:.3} ms, the blocked kernel {d:.3} ms ({d:.2}x)\n", .{ context, rows, row_ms, block_ms, row_ms / @max(block_ms, 1e-9) });
    std.debug.print("the scan and the top-k together: the row kernel {d:.3} ms, the blocked kernel {d:.3} ms\n", .{ row_full, block_full });
    std.debug.print("the scores are the same bits, the key lists the same indices in the same order\n", .{});
}
