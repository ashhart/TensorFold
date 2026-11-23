//! Times the narrow and wide row kernels at each window width on this GPU, checks equal bytes, prints the cutover.
const std = @import("std");
const mtl = @import("metal");
const row = @import("row_projection");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const n = 2688;
const k = 4096;
const max_rows = 32;
const warm = 3;
const repeats = 15;

const Mat = struct { w: row.Weights, x: mtl.Buffer };

fn matrix(device: mtl.Device, bits: u8) !Mat {
    var prng = std.Random.DefaultPrng.init(@as(u64, 0x3396) + bits);
    const r = prng.random();
    const words = n * k * @as(usize, bits) / 32;
    const tables = n * (k / 64);
    const m = Mat{
        .w = .{ .w = try device.buffer(words * 4, opts), .scales = try device.buffer(tables * 2, opts), .biases = try device.buffer(tables * 2, opts), .n = n, .k = k, .bits = bits },
        .x = try device.buffer(max_rows * k * 2, opts),
    };
    for (m.w.w.slice(u32, words)) |*v| v.* = r.int(u32);
    for (m.w.scales.slice(u16, tables)) |*v| v.* = bf(r.float(f32) * 0.02 + 0.001);
    for (m.w.biases.slice(u16, tables)) |*v| v.* = bf((r.float(f32) - 0.5) * 0.1);
    for (m.x.slice(u16, max_rows * k)) |*v| v.* = bf((r.float(f32) - 0.5) * 2);
    return m;
}

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn launch(queue: mtl.Queue, c: row.Call, m: Mat, out: mtl.Buffer) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(c.pipeline);
    e.setBuffer(m.x, 0, 0);
    e.setBuffer(m.w.w, 0, 1);
    e.setBuffer(m.w.scales, 0, 2);
    e.setBuffer(m.w.biases, 0, 3);
    e.setValue(c.dims, 4);
    e.setBuffer(out, 0, 5);
    e.dispatchGroups(mtl.Size.of(c.groups, 1, 1), mtl.Size.of(c.threads, 1, 1));
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    return cb.gpuSeconds();
}

fn median(queue: mtl.Queue, c: row.Call, m: Mat, out: mtl.Buffer) !f64 {
    for (0..warm) |_| _ = try launch(queue, c, m, out);
    var t: [repeats]f64 = undefined;
    for (&t) |*s| s.* = try launch(queue, c, m, out);
    std.mem.sort(f64, &t, {}, std.sort.asc(f64));
    return t[repeats / 2];
}

pub fn main() !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try row.Pipelines.load(device);
    defer p.deinit();
    if (!p.wide) return error.NoTensorUnits;
    var narrow = p;
    narrow.wide_from = @splat(max_rows + 1);
    var wide = p;
    wide.wide_from = @splat(1);
    const a = try device.buffer(max_rows * n * 2, opts);
    defer a.deinit();
    const b = try device.buffer(max_rows * n * 2, opts);
    defer b.deinit();
    var line: [256]u8 = undefined;
    const name = std.mem.span(device.name()); // the name the key reads, printed beside it
    std.debug.print("{s}\n", .{try row.keyLine(&line, name, row.chipOf(device.generation(), name))});
    for (row.bit_widths) |bits| {
        const m = try matrix(device, bits);
        defer for ([_]mtl.Buffer{ m.w.w, m.w.scales, m.w.biases, m.x }) |x| x.deinit();
        var from: usize = max_rows + 1;
        var rows: usize = max_rows;
        var unbroken = true;
        while (rows >= 1) : (rows -= 1) {
            const tn = try median(queue, try row.call(&narrow, m.w, rows, false), m, a);
            const tw = try median(queue, try row.call(&wide, m.w, rows, false), m, b);
            if (!std.mem.eql(u8, a.slice(u8, rows * n * 2), b.slice(u8, rows * n * 2))) return error.KernelBytesDiffer;
            std.debug.print("ROW_CUTOVER bits={d} rows={d} narrow_us={d:.1} wide_us={d:.1} equal=true\n", .{ bits, rows, tn * 1e6, tw * 1e6 });
            unbroken = unbroken and tw < tn;
            if (unbroken) from = rows;
        }
        std.debug.print("ROW_CUTOVER_RESULT bits={d} wide_from={d}\n", .{ bits, from });
    }
}
