//! The tensor-unit prompt kernel against the lane kernel on a real layer's expert gate halves: error and GPU time.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g = tf.glm_dsa;
const ex = tf.exlr8;
const op = tf.exlr8_projection;
const pp = tf.exlr8_prompt;
const a = std.heap.page_allocator;

const usage = "tf-exlr8-prompt-check MODEL_DIR [LAYER 40] [ROWS 32] [REPS 20]: every expert's gate halves of rank 0 of 2";

fn arg(argv: []const [:0]const u8, i: usize, d: u32) !u32 {
    return if (argv.len > i) try std.fmt.parseInt(u32, argv[i], 10) else d;
}

fn run(queue: mtl.Queue, comptime f: anytype, ctx: anytype) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    try f(ctx, e);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |m| {
        std.debug.print("GPU command failed: {s}\n", .{m});
        return error.GpuFailure;
    }
    return (cb.gpuEnd() - cb.gpuStart()) * 1000;
}

const Ctx = struct {
    P: *const op.Pipelines,
    Q: *const pp.Pipelines,
    t: op.Tables,
    x: mtl.Buffer,
    y: mtl.Buffer,
    members: mtl.Buffer,
    rows: u32,
    width: u32,
};

fn lane16(c: Ctx, e: mtl.ComputeEncoder) !void {
    // the lane kernel takes 16 rows a job: the jobs' second halves run as a second set of member lists
    var t = c.t;
    try op.encodeProject(c.P, e, .{ .buf = c.x }, t, .{ .buf = c.y }, @min(c.rows, 16), 6144, c.width, true, .{ .buf = c.members });
    if (c.rows > 16) {
        t.jobs_off = t.count * @sizeOf(op.Job);
        try op.encodeProject(c.P, e, .{ .buf = c.x }, t, .{ .buf = c.y }, c.rows - 16, 6144, c.width, true, .{ .buf = c.members });
    }
}

fn prompt(c: Ctx, e: mtl.ComputeEncoder) !void {
    var t = c.t;
    t.jobs_off = 2 * t.count * @sizeOf(op.Job);
    try pp.encode(c.Q, e, .{ .buf = c.x }, t, .{ .buf = c.y }, 32, 6144, c.width, true, .{ .buf = c.members });
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 2) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    const model = argv[1];
    const L_n = try arg(argv, 2, 40);
    const rows = @min(try arg(argv, 3, 32), 32);
    const reps = try arg(argv, 4, 20);
    var c = try g.config.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{model}), a, .limited(1 << 20)));
    try g.config.split(&c, 0, 2);
    var manifest = try ex.manifest.Manifest.load(a, io, try std.fmt.allocPrint(a, "{s}/manifests/exlr8-manifest-home.json", .{model}));
    defer manifest.deinit();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    var xl = try ex.experts.Layer.open(a, io, model, @intCast(L_n));
    defer xl.close(io);
    var pk = ex.pack.Pack.init(a);
    const w = try g.weights.load(a, io, model, &c, &manifest, L_n, &pk, &xl);
    const L = try g.layer.Layer.init(a, device, w, &pk, 1, 64, mtl.ResourceOptions.shared);
    const P = try op.Pipelines.load(device);
    const Q = try pp.Pipelines.load(a, device);
    const halves = c.localHalves();
    const experts = w.mlp.moe.experts;
    const njobs: u32 = @intCast(experts.len * halves);
    const width = halves * 128;
    // jobs three times over: the lane kernel's rows 0-15, its rows 16-31, the prompt kernel's 32 rows
    const jobs = try device.buffer(3 * njobs * @sizeOf(op.Job), mtl.ResourceOptions.shared);
    const members = try device.buffer(3 * 32 * 4 + 16, mtl.ResourceOptions.shared);
    const m = members.slice(u32, 96);
    for (0..32) |i| {
        // row i reads input row i and writes output row i of its expert's block
        m[i] = @intCast((i << 16) | i);
        m[32 + i] = @intCast(((16 + i) << 16) | (16 + i));
        m[64 + i] = @intCast((i << 16) | i);
    }
    const js = jobs.slice(op.Job, 3 * njobs);
    for (experts, 0..) |x, ei| for (0..halves) |h| {
        const k = ei * halves + h;
        const base: op.Job = .{ .k = x.gate[h].k, .blocks = 48, .base = x.gate_base[h], .in_scale = x.gate[h].block, .out_scale = x.gate[h].chan, .y_col = @intCast(ei * width + 128 * h) };
        js[k] = base;
        js[k].members = 0;
        js[k].count = @min(rows, 16);
        js[njobs + k] = base;
        js[njobs + k].members = 32;
        js[njobs + k].count = if (rows > 16) rows - 16 else 0;
        js[2 * njobs + k] = base;
        js[2 * njobs + k].members = 64;
        js[2 * njobs + k].count = rows;
    };
    const x = try device.buffer(32 * 6144 * 4, mtl.ResourceOptions.shared);
    var prng = std.Random.DefaultPrng.init(11);
    for (x.slice(f32, 32 * 6144)) |*v| v.* = prng.random().floatNorm(f32);
    const y_lane = try device.buffer(32 * @as(usize, width) * 4 * experts.len, mtl.ResourceOptions.shared);
    const y_tu = try device.buffer(32 * @as(usize, width) * 4 * experts.len, mtl.ResourceOptions.shared);
    const tt: op.Tables = .{ .weights = L.k.bytes, .bases = L.k.bases, .halves = L.k.halves, .signs = L.k.signs, .jobs = jobs, .count = njobs };
    const stride: u32 = @intCast(experts.len * width);
    var worst: f64 = 0;
    var sq: f64 = 0;
    var ref_sq: f64 = 0;
    var lane_ms: f64 = 0;
    var tu_ms: f64 = 0;
    for (0..reps) |rep| {
        const c1: Ctx = .{ .P = &P, .Q = &Q, .t = tt, .x = x, .y = y_lane, .members = members, .rows = rows, .width = stride };
        var c2 = c1;
        c2.y = y_tu;
        const ml = try run(queue, lane16, c1);
        const mt = try run(queue, prompt, c2);
        if (rep > 0) {
            lane_ms += ml;
            tu_ms += mt;
        }
    }
    const yl = y_lane.slice(f32, 32 * stride);
    const yt = y_tu.slice(f32, 32 * stride);
    for (0..rows) |r| for (0..stride) |col| {
        const d = @abs(@as(f64, yt[r * stride + col]) - yl[r * stride + col]);
        worst = @max(worst, d);
        sq += d * d;
        ref_sq += @as(f64, yl[r * stride + col]) * yl[r * stride + col];
    };
    const n: f64 = @floatFromInt(@max(reps - 1, 1));
    std.debug.print("layer {d}: {d} gate jobs (every expert's {d} halves) over {d} rows\n", .{ L_n, njobs, halves, rows });
    std.debug.print("  lane kernel {d:.3} ms, tensor-unit kernel {d:.3} ms ({d:.2}x)\n", .{ lane_ms / n, tu_ms / n, lane_ms / @max(tu_ms, 1e-9) });
    std.debug.print("  rms(d)/rms(lane) {e:.3}, max |d| / rms(lane) {e:.3}\n", .{ @sqrt(sq / ref_sq), worst / @sqrt(ref_sq / @as(f64, @floatFromInt(rows * stride))) });
}
