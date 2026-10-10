//! Sliding Weights on CUDA: the change's gradient against the loss's slope along it, where no expert choice moves.

const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const dims = nemotron.slide_dims;
const block = dims.block;

/// MODEL with IDS_FILE's tokens (the answer from START): each probe's analytic slope against its central differences.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8, start_text: []const u8, reach_text: ?[]const u8, restarts_text: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    const text = try std.Io.Dir.cwd().readFileAlloc(io, ids_path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |w| try ids.append(gpa, try std.fmt.parseInt(u32, w, 10));
    const start = try std.fmt.parseInt(usize, start_text, 10);
    // the loss change the furthest step predicts
    const reach: f64 = if (reach_text) |x| try std.fmt.parseFloat(f64, x) else 0.16;
    const e = try nemotron.Engine.init(gpa, io, gpu.ctx, model, null, .{ .context = 2048, .mtp = false, .graphs = false, .sampling = null, .segments = 1, .slide = true });
    defer e.deinit();
    const t = try nemotron.train.Trainer.init(gpa, e);
    defer t.deinit(gpa);
    const s = &t.sites;
    const restarts: usize = if (restarts_text) |x| try std.fmt.parseInt(usize, x, 10) else 1;
    // the top layers: below them a step in the change moves expert choices downstream, each a step in the loss
    const spans = [_][2]usize{ .{ 51, 52 }, .{ 50, 51 }, .{ 49, 50 } };
    var sum: [spans.len]f64 = @splat(0);
    var sq: [spans.len]f64 = @splat(0);
    var fits: [spans.len]usize = @splat(0);
    for (0..restarts) |restart| {
        var prng = std.Random.DefaultPrng.init(7 + restart);
        const r = prng.random();
        // one block at every layer: random directions a_norm long, open on every row, small random outputs
        for (s.list) |*site| {
            const a = site.a.slice(f32, block * site.in);
            for (0..block) |q| {
                const row = a[q * site.in ..][0..site.in];
                var n: f32 = 0;
                for (row) |*v| {
                    v.* = r.floatNorm(f32);
                    n += v.* * v.*;
                }
                for (row) |*v| v.* *= dims.a_norm / @sqrt(n);
            }
            site.gate(0).* = -std.math.inf(f32);
            for (site.b.slice(f32, block * site.out)) |*v| v.* = 1e-3 * r.floatNorm(f32);
        }
        s.rank = block;
        t.attach(true);
        const base = try t.step(ids.items, start, .loss);
        for (s.list) |*site| @memset(site.gb.slice(f32, block * site.out), 0);
        const graded = try t.step(ids.items, start, .grad);
        try check.expect(@abs(base.loss - graded.loss) < 1e-4 * @max(1, base.loss), "a grad step's loss {d} equals the loss step's {d}", .{ graded.loss, base.loss });
        // the probes' grad steps add into the gradients: each probe starts from the base's, and its expert choices
        const was = try picks(t, ids.items.len - 1);
        defer std.heap.page_allocator.free(was);
        var grads: std.ArrayList(f32) = .empty;
        defer grads.deinit(gpa);
        for (s.list) |*site| try grads.appendSlice(gpa, site.gb.slice(f32, block * site.out));
        for (spans, 0..) |span, si| {
            var at: usize = 0;
            for (s.list) |*site| {
                const g = site.gb.slice(f32, block * site.out);
                @memcpy(g, grads.items[at..][0..g.len]);
                at += g.len;
            }
            const rel = try probe(t, ids.items, start, span, reach, base.loss, was) orelse continue;
            sum[si] += rel;
            sq[si] += rel * rel;
            fits[si] += 1;
        }
    }
    var worst: f64 = 0;
    for (spans, sum, sq, fits) |span, x, x2, count| {
        if (count == 0) continue;
        const n: f64 = @floatFromInt(count);
        const mean = x / n;
        const sd = @sqrt(@max(x2 / n - mean * mean, 0));
        worst = @max(worst, @abs(mean));
        std.debug.print("RESULT layer {d} ({t}): fitted over analytic slope {d:.4} (sd {d:.4} over {d} starts)\n", .{ span[0], e.c.kinds[span[0]], 1 + mean, sd, count });
    }
    try check.expect(worst < 0.05, "every probe's mean fitted slope within 5% of its analytic slope (worst {d:.4})", .{worst});
    check.pass("slide-grad: every probe's mean within {d:.2}% of its analytic slope", .{100 * worst});
}

/// The slope along the span's gradient, fitted where no expert choice moved, over its length less 1 (null: no fit).
fn probe(t: *nemotron.train.Trainer, ids: []const u32, start: usize, span: [2]usize, reach: f64, base: f32, was: []const i32) !?f64 {
    const gpa = std.heap.page_allocator;
    const s = &t.sites;
    var dirs: std.ArrayList(f32) = .empty;
    defer dirs.deinit(gpa);
    var keep: std.ArrayList(f32) = .empty;
    defer keep.deinit(gpa);
    var norm: f64 = 0;
    for (s.list[span[0]..span[1]]) |*site| for (site.gb.slice(f32, block * site.out)) |g| {
        try dirs.append(gpa, g);
        norm += g * g;
    };
    const slope = @sqrt(norm);
    for (dirs.items) |*v| v.* /= @floatCast(slope);
    for (s.list[span[0]..span[1]]) |*site| try keep.appendSlice(gpa, site.b.slice(f32, block * site.out));
    const eps: f64 = reach / 8 / @max(slope, 1e-12);
    var xs: [17]f64 = undefined;
    var ys: [17]f64 = undefined;
    var moved: [17]usize = @splat(0);
    for (0..17) |i| {
        const k = @as(f64, @floatFromInt(i)) - 8;
        xs[i] = k * eps;
        if (k == 0) {
            ys[i] = base;
            continue;
        }
        place(s, span, keep.items, dirs.items, @floatCast(xs[i]));
        ys[i] = (try t.step(ids, start, .grad)).loss;
        const now = try picks(t, ids.len - 1);
        defer gpa.free(now);
        for (was, now) |a, b| moved[i] += @intFromBool(a != b);
    }
    place(s, span, keep.items, dirs.items, 0);
    var lo: usize = 8;
    var hi: usize = 8;
    while (lo > 0 and moved[lo - 1] == 0) lo -= 1;
    while (hi < 16 and moved[hi + 1] == 0) hi += 1;
    const clean = @min(8 - lo, hi - 8);
    // the odd part's least-squares slope over the clean steps: sum k (y(+k) - y(-k)) / (2 eps sum k^2)
    var num: f64 = 0;
    var den: f64 = 0;
    for (1..clean + 1) |kk| {
        const kf: f64 = @floatFromInt(kk);
        num += kf * (ys[8 + kk] - ys[8 - kk]);
        den += 2 * eps * kf * kf;
    }
    var all: usize = 0;
    for (moved) |m| all += m;
    std.debug.print("INFO layers {d}..{d}: {d} expert choices moved across the steps; {d} clean steps each side\n", .{ span[0], span[1] - 1, all, clean });
    if (clean < 2) return null;
    return num / den / @max(slope, 1e-12) - 1;
}

/// Every MoE layer's expert choices as the last grad step kept them.
fn picks(t: *nemotron.train.Trainer, rows: usize) ![]i32 {
    const gpa = std.heap.page_allocator;
    const c = t.e.c;
    var out: std.ArrayList(i32) = .empty;
    for (0..c.layers) |l| if (c.kinds[l] == .moe) {
        const n = rows * c.slots();
        const at = out.items.len;
        try out.resize(gpa, at + n);
        try t.e.stream.synchronize();
        try t.e.ctx.d.check(t.e.ctx.d.api.cuMemcpyDtoH_v2(out.items[at..].ptr, t.parts(l)[0], n * 4), "cuMemcpyDtoH");
    };
    return out.toOwnedSlice(gpa);
}

/// The span's open block outputs at keep + x dirs.
fn place(s: *nemotron.sites.Sites, span: [2]usize, keep: []const f32, dirs: []const f32, x: f32) void {
    var at: usize = 0;
    for (s.list[span[0]..span[1]]) |*site| {
        const b = site.b.slice(f32, block * site.out);
        for (b, keep[at..][0..b.len], dirs[at..][0..b.len]) |*v, k0, d| v.* = k0 + x * d;
        at += b.len;
    }
}
