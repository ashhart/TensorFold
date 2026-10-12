//! glm_rows_check's projection classes: each GLM product at the oQ checkpoints' widths, decode and prompt, vs fp64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const wide = tf.glm.wide;
const affine = tf.glm.affine;
const affine_mm = tf.affine_mm;

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const rows = 16; // a decode window
const prompt_rows = 33; // a prompt chunk: two 16-row steps and one more

/// A class's shape [n, k] (a head's for embed_q / unembed_out) and the widths oQ4, oQ2 and the 4-bit build store it at.
const Class = struct { name: []const u8, k: u32, n: u32, bits: []const u8 };
const classes = [_]Class{
    .{ .name = "kda_in q/k/v", .k = 4096, .n = 8192, .bits = &.{ 5, 8 } },
    .{ .name = "kda_in f_a/g_a", .k = 4096, .n = 128, .bits = &.{ 5, 8 } },
    .{ .name = "kda_in b", .k = 4096, .n = 64, .bits = &.{ 5, 8 } },
    .{ .name = "kda f_b/g_b", .k = 128, .n = 8192, .bits = &.{ 5, 8 } },
    .{ .name = "kda o_proj", .k = 8192, .n = 4096, .bits = &.{ 6, 8 } },
    .{ .name = "mla q_a", .k = 4096, .n = 1536, .bits = &.{ 6, 8 } },
    .{ .name = "mla kv_a", .k = 4096, .n = 512, .bits = &.{ 6, 8 } },
    .{ .name = "indexer wk", .k = 4096, .n = 128, .bits = &.{ 5, 8 } },
    .{ .name = "indexer weights_proj", .k = 4096, .n = 32, .bits = &.{ 5, 8 } },
    .{ .name = "mla x_proj (one width)", .k = 4096, .n = 2208, .bits = &.{8} },
    .{ .name = "mla q_b", .k = 1536, .n = 16384, .bits = &.{ 6, 8 } },
    .{ .name = "indexer wq_b", .k = 1536, .n = 4096, .bits = &.{ 5, 8 } },
    .{ .name = "embed_q (a head)", .k = 256, .n = 512, .bits = &.{ 5, 8 } },
    .{ .name = "unembed_out (a head)", .k = 512, .n = 256, .bits = &.{ 5, 8 } },
    .{ .name = "mla o_proj", .k = 16384, .n = 4096, .bits = &.{ 6, 8 } },
    .{ .name = "dense gate/up", .k = 4096, .n = 12288, .bits = &.{ 5, 8 } },
    .{ .name = "dense down", .k = 12288, .n = 4096, .bits = &.{ 6, 8 } },
    .{ .name = "shared gate/up", .k = 4096, .n = 2048, .bits = &.{8} },
    .{ .name = "shared down", .k = 2048, .n = 4096, .bits = &.{8} },
    .{ .name = "routed gate/up", .k = 4096, .n = 2048, .bits = &.{ 2, 4 } },
    .{ .name = "routed down", .k = 2048, .n = 4096, .bits = &.{ 2, 4 } },
    .{ .name = "lm_head", .k = 4096, .n = 154880, .bits = &.{8} },
};

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}
fn fromBf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// Code j of a row's little-endian bit stream at `bits` (MLX's packing at every width).
pub fn code(row: []const u8, j: usize, bits: u8) u32 {
    const at = j * bits;
    var v: u32 = 0;
    for (0..4) |b| {
        if (at / 8 + b < row.len) v |= @as(u32, row[at / 8 + b]) << @intCast(8 * b);
    }
    return (v >> @intCast(at % 8)) & ((@as(u32, 1) << @intCast(bits)) - 1);
}

const Tally = struct { values: usize = 0, worst: f64 = 0, worst_col: usize = 0, pairs: usize = 0, err: f64 = 0, ymax: f64 = 0, caught: usize = 0 };

/// fp32 sums' error per unit of |terms|: under k / 8 + 64 additions deep (row kernels and affine_mm), 2^-24 each.
pub fn gamma(k: usize) f64 {
    return @as(f64, @floatFromInt(k / 8 + 64)) * 0x1p-24;
}

/// out row i vs fp64 on every column within the fp32 sums' bound; `scale` != 1: a control, columns past it counted.
fn against(q: affine.Affine, w: []const u8, s: []const u16, b: []const u16, x: []const u16, out: []const u16, pitch: usize, i: usize, t: *Tally, scale: f64) !void {
    const k: usize = q.k;
    const g_n = k / q.group;
    const rb = k * q.bits / 8;
    const gm = gamma(k);
    for (0..q.n) |c| {
        var y: f64 = 0;
        var terms: f64 = 0;
        for (0..g_n) |g| {
            const sc = scale * fromBf(s[c * g_n + g]);
            const bi = fromBf(b[c * g_n + g]);
            var dot: f64 = 0;
            var sum: f64 = 0;
            for (0..q.group) |j| {
                const kk = g * q.group + j;
                const xv = fromBf(x[i * k + kk]);
                const cv: f64 = @floatFromInt(code(w[c * rb ..][0..rb], kk, q.bits));
                y += xv * (sc * cv + bi);
                dot += @abs(xv * cv);
                sum += @abs(xv);
            }
            terms += @abs(sc) * dot + @abs(bi) * sum;
        }
        const bound = std.math.ldexp(@abs(y) + gm * terms, -8) + gm * terms;
        const err = @abs(fromBf(out[i * pitch + c]) - y);
        if (scale != 1) {
            t.caught += @intFromBool(err > bound);
            continue;
        }
        t.err = @max(t.err, err);
        t.ymax = @max(t.ymax, @abs(y));
        if (err / bound > t.worst) {
            t.worst = err / bound;
            t.worst_col = c;
        }
        if (err > bound) {
            std.debug.print("col {d} row {d}: {d} vs fp64 {d} (bound {e})\n", .{ c, i, fromBf(out[i * pitch + c]), y, bound });
            return error.PastBound;
        }
        t.values += 1;
    }
}

fn submit(queue: mtl.Queue, comptime f: anytype, args: anytype) !void {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    @call(.auto, f, .{e} ++ args);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("GPU failed: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn rowsLaunch(e: mtl.ComputeEncoder, p: *const wide.Pipes, q: affine.Affine, x: mtl.Buffer, at: usize, out: mtl.Buffer, m: u32) void {
    wide.dense(p, e, q, .{ .in = (affine.Ref{ .buf = x }).at(at), .ldx = q.k, .out = .{ .buf = out }, .ldo = q.n, .rows = m });
}

fn mmLaunch(e: mtl.ComputeEncoder, mm: affine_mm.Pipes, q: affine.Affine, x: mtl.Buffer, sums: mtl.Buffer, out: mtl.Buffer, m: u32) void {
    const xr: affine.Ref = .{ .buf = x };
    const sr: affine.Ref = .{ .buf = sums };
    affine_mm.rowSums(e, mm, 64, xr, sr, m, q.k);
    affine_mm.dense(e, mm, xr, sr, q, affine.Ref{ .buf = out }, m);
}

/// Each class at its widths (g64): decode rows (alone == windowed) and the prompt's product vs fp64, every column.
pub fn check(p: *const wide.Pipes, device: mtl.Device, queue: mtl.Queue) !Tally {
    var all = Tally{};
    var prng = std.Random.DefaultPrng.init(716);
    const r = prng.random();
    for (classes) |cl| for (cl.bits) |bits| {
        const k: usize = cl.k;
        const n: usize = cl.n;
        const np = std.mem.alignForward(usize, n, 64); // the loader's padded rows (affine_mm reads whole 64-row tiles)
        const w_bytes = np * k * bits / 8;
        const tables = np * (k / 64);
        const bufs = [_]mtl.Buffer{ try device.buffer(w_bytes, opts), try device.buffer(tables * 2, opts), try device.buffer(tables * 2, opts), try device.buffer(prompt_rows * k * 2, opts), try device.buffer(prompt_rows * np * 2, opts), try device.buffer(prompt_rows * np * 2, opts), try device.buffer(prompt_rows * (k / 64) * 4, opts) };
        defer for (bufs) |x| x.deinit();
        r.bytes(bufs[0].slice(u8, w_bytes));
        for (bufs[1].slice(u16, tables)) |*v| v.* = toBf(r.float(f32) * 0.02 + 0.001);
        for (bufs[2].slice(u16, tables)) |*v| v.* = toBf((r.float(f32) - 0.5) * 0.1);
        for (bufs[3].slice(u16, prompt_rows * k)) |*v| v.* = toBf((r.float(f32) - 0.5) * 2);
        const q: affine.Affine = .{ .w = .{ .buf = bufs[0] }, .s = .{ .buf = bufs[1] }, .b = .{ .buf = bufs[2] }, .n = cl.n, .k = cl.k, .bits = bits, .group = 64 };
        const w = bufs[0].slice(u8, w_bytes);
        const s = bufs[1].slice(u16, tables);
        const b = bufs[2].slice(u16, tables);
        const x = bufs[3].slice(u16, prompt_rows * k);
        var t = Tally{};
        // decode: a 16-row window, then rows 0 and 15 alone
        try submit(queue, rowsLaunch, .{ p, q, bufs[3], 0, bufs[4], rows });
        for ([_]usize{ 0, rows - 1 }) |i| {
            try submit(queue, rowsLaunch, .{ p, q, bufs[3], i * k * 2, bufs[5], 1 });
            if (!std.mem.eql(u16, bufs[5].slice(u16, n), bufs[4].slice(u16, rows * n)[i * n ..][0..n])) return error.WindowDiffersFromRowAlone;
            t.pairs += n;
        }
        for ([_]usize{ 0, rows - 1 }) |i| try against(q, w, s, b, x, bufs[4].slice(u16, rows * n), n, i, &t, 1);
        const dec = t.worst;
        try against(q, w, s, b, x, bufs[4].slice(u16, rows * n), n, 0, &t, 1.01); // scales 1% off must be seen
        if (t.caught == 0) return error.ControlNotCaught;
        // prompt: the chunk's product at rows 0 and 32; on the row kernels every row's bits equal the decode's
        const mm: ?affine_mm.Pipes = if (bits == 8) p.mm8 else if (bits == 2) p.mm2 else null; // prompt.zig's (2: routed)
        const on_mm = mm != null;
        if (mm) |pipes| try submit(queue, mmLaunch, .{ pipes, q, bufs[3], bufs[6], bufs[5], prompt_rows }) else try submit(queue, rowsLaunch, .{ p, q, bufs[3], 0, bufs[5], prompt_rows });
        t.worst = 0;
        const pitch = if (on_mm) np else n;
        for ([_]usize{ 0, prompt_rows - 1 }) |i| try against(q, w, s, b, x, bufs[5].slice(u16, prompt_rows * pitch), pitch, i, &t, 1);
        if (!on_mm and !std.mem.eql(u16, bufs[5].slice(u16, rows * n), bufs[4].slice(u16, rows * n))) return error.PromptDiffersFromDecode;
        std.debug.print("{s} [{d}, {d}] q{d}: decode worst err/bound {d:.4}, prompt ({s}) {d:.4} at col {d}; max |err| {e:.3} = {e:.3} of max |y|; scales 1% off caught in {d} of {d} columns\n", .{ cl.name, n, k, bits, dec, if (on_mm) "affine_mm" else "row kernels", t.worst, t.worst_col, t.err, t.err / t.ymax, t.caught, n });
        all.values += t.values;
        all.pairs += t.pairs;
        all.worst = @max(all.worst, @max(dec, t.worst));
    };
    return all;
}
