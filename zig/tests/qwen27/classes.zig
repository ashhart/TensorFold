//! GPU time of each 27B projection class over every layer at a step's row counts, against its weight bytes' roofline.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Linear = q.projection.Linear;
const Ref = q.projection.Ref;
const help = "tf-qwen27-classes --model DIR [--gbs 800] [--reps 4]";

const Class = enum { gdn_qkv, gdn_zba, gdn_out, attn_q, attn_kv, attn_out, gate_up, down, head };
const Input = enum { input, activated, mixer_out };
const row_counts = [_]u32{ 1, 2, 4, 8, 16, 128 };

fn source(f: *q.gpu_frame.Frame, c: Class) Ref {
    return switch (c) {
        .gdn_out, .attn_out => f.get(.mixer_out),
        .down => f.get(.activated),
        else => f.get(.input),
    };
}

/// The class's matrices over every layer, as forward.zig and the mixers run them.
fn linears(w: q.gpu_weights.Weights, c: Class, out: []Linear) []Linear {
    if (c == .head) {
        out[0] = w.head;
        return out[0..1];
    }
    var n: usize = 0;
    for (w.layers) |l| {
        const x: ?Linear = switch (c) {
            .gate_up => l.gu,
            .down => l.down,
            .gdn_qkv, .gdn_zba, .gdn_out => switch (l.mixer) {
                .linear => |g| if (c == .gdn_qkv) g.qkv else if (c == .gdn_zba) g.zba else g.out,
                .attention => null,
            },
            .attn_q, .attn_kv, .attn_out => switch (l.mixer) {
                .attention => |g| if (c == .attn_q) g.q else if (c == .attn_kv) g.kv else g.out,
                .linear => null,
            },
            .head => unreachable,
        };
        if (x) |v| {
            out[n] = v;
            n += 1;
        }
    }
    return out[0..n];
}

/// Weight bytes a call reads (codes, bf16 scales and biases), and its n*k by code width.
fn bytes(l: Linear, by_bits: *[9]usize) usize {
    if (l.runs.len > 0) {
        var sum: usize = 0;
        for (l.runs) |r| sum += bytes(r.linear, by_bits);
        return sum;
    }
    const bits: usize = if (l.raw) |r| r.bits else l.bits;
    const group: usize = if (l.raw) |r| r.group else l.group;
    by_bits[bits] += @as(usize, l.n) * l.k;
    return @as(usize, l.n) * l.k * bits / 8 + @as(usize, l.n) * (l.k / group) * 4;
}

/// Every matrix of the class `reps` times in one command buffer: GPU seconds a step.
fn time(m: *q.model.Model, c: Class, ls: []const Linear, rows: u32, reps: usize) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const f = &m.frame;
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..reps) |_| for (ls) |l| try m.kernels.quant(e, l, source(f, c), null, f.get(.dims), f.get(.logits), rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.BenchGpuFailure;
    return cb.gpuSeconds() / @as(f64, @floatFromInt(reps));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var dir: ?[]const u8 = null;
    var gbs: f64 = 800;
    var reps: usize = 4;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help")) return std.debug.print("{s}\n", .{help});
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, args[i - 1], "--model")) dir = args[i] else if (std.mem.eql(u8, args[i - 1], "--gbs")) gbs = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, args[i - 1], "--reps")) reps = try std.fmt.parseInt(usize, args[i], 10) else return error.UnknownFlag;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(gpa, io, dir orelse return error.MissingModel, 128);
    defer m.deinit();
    const f = &m.frame;
    for ([_]Ref{ f.get(.input), f.get(.activated), f.get(.mixer_out) }) |r| {
        const words: [*]u16 = @ptrCast(@alignCast(r.buffer.contents() + r.offset));
        const n = @min(128 * @as(usize, m.config.intermediate), (r.buffer.length() - r.offset) / 2);
        for (words[0..n], 0..) |*v, j| v.* = @truncate(0x3c00 + (j * 7) % 96 + (j / 1000) % 16);
    }
    std.debug.print("device {s}; roofline at {d:.0} GB/s; ms a step = every layer's matrices of a class, {d} reps\n", .{ m.device.name(), gbs, reps });
    var store: [64]Linear = undefined;
    for (row_counts) |rows| {
        var total_ms: f64 = 0;
        var total_bytes: usize = 0;
        for (std.enums.values(Class)) |c| {
            const ls = linears(m.weights, c, &store);
            var by_bits: [9]usize = @splat(0);
            var b: usize = 0;
            for (ls) |l| b += bytes(l, &by_bits);
            _ = try time(m, c, ls, rows, 1); // pipelines and clocks
            const s = try time(m, c, ls, rows, reps);
            var all: usize = 0;
            for (by_bits) |x| all += x;
            var mix: [64]u8 = undefined;
            var len: usize = 0;
            for (by_bits, 0..) |x, bits| if (x > 0) {
                len += (try std.fmt.bufPrint(mix[len..], "q{d}:{d:.0}% ", .{ bits, 100 * @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(all)) })).len;
            };
            const roof = @as(f64, @floatFromInt(b)) / (gbs * 1e9);
            std.debug.print("CLASS rows={d} {s} calls={d} {s}MB={d:.1} ms={d:.3} GB/s={d:.0} roofline_ms={d:.3} x{d:.2}\n", .{ rows, @tagName(c), ls.len, mix[0..len], @as(f64, @floatFromInt(b)) / 1e6, s * 1e3, @as(f64, @floatFromInt(b)) / s / 1e9, roof * 1e3, s / roof });
            total_ms += s * 1e3;
            total_bytes += b;
        }
        const roof_ms = @as(f64, @floatFromInt(total_bytes)) / (gbs * 1e9) * 1e3;
        std.debug.print("TOTAL rows={d} MB={d:.0} ms={d:.2} roofline_ms={d:.2} x{d:.2} ({d:.0} rows/s through projections)\n", .{ rows, @as(f64, @floatFromInt(total_bytes)) / 1e6, total_ms, roof_ms, total_ms / roof_ms, @as(f64, @floatFromInt(rows)) * 1e3 / total_ms });
    }
}
