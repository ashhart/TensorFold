//! Gemma 4 E4B's per-layer inputs on the GPU (`inputs`, then `apply` per layer), dumped for tf-gemma4-e4b-ple-host.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;

fn bf(b16: u16) f32 {
    return @bitCast(@as(u32, b16) << 16);
}

/// The bf16 of a float, rounded to nearest even.
fn bits(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// A checkpoint matrix of `count` bf16 values.
fn raw(r: g4.weights.Ref, count: usize) []const u16 {
    const p: [*]const u16 = @ptrCast(@alignCast(r.buf.contents() + r.off));
    return p[0..count];
}

/// One table row [layers * width] dequantized to bf16 (4-bit codes in groups of 64: scale * code + bias, once rounded).
fn tableRow(t: g4.weights.Ple, id: usize, wide: usize, out: []f32) void {
    const words: [*]const u8 = @ptrCast(@alignCast(t.table_w.buf.contents() + t.table_w.off));
    const scales: [*]const u16 = @ptrCast(@alignCast(t.table_s.buf.contents() + t.table_s.off));
    const biases: [*]const u16 = @ptrCast(@alignCast(t.table_b.buf.contents() + t.table_b.off));
    for (0..wide) |c| {
        const byte = words[id * (wide / 2) + c / 2];
        const code: f32 = @floatFromInt(if (c % 2 == 0) byte & 0xf else byte >> 4);
        const g = id * (wide / 64) + c / 64;
        out[c] = bf(bits(@mulAdd(f32, bf(scales[g]), code, bf(biases[g]))));
    }
}

/// Buffers made untracked are not ordered between encoders: a copy is a command buffer of its own, waited on.
fn blit(m: *g4.Model, from: mtl.Buffer, to: mtl.Buffer, bytes: usize) !void {
    const cb = m.queue.commandBuffer();
    const b = cb.blit();
    b.copy(from, 0, to, 0, bytes);
    b.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

fn put(w: *std.Io.Writer, comptime T: type, v: []const T) !void {
    try w.writeAll(std.mem.sliceAsBytes(v));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const out_path = args.next() orelse return error.Usage;
    // a few tokens: the start of sequence, then ordinary ones
    var ids = [_]u32{ 2, 105, 2364, 107, 3689, 563, 496, 3333, 235, 19841, 1 };
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    const geo = m.config.geo;
    if (geo.layer_input == 0) return error.NotAnESeries;
    const rows = ids.len;
    const width = geo.layer_input;
    const wide = geo.layers * width;
    var s = try g4.state.Scratch.init(m.device, geo, 256);
    defer s.deinit();
    const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const stage_h = try m.device.buffer(rows * geo.hidden * 2, shared);
    defer stage_h.deinit();
    const stage_p = try m.device.buffer(rows * wide * 2, shared);
    defer stage_p.deinit();
    @memcpy(s.ids.slice(u32, rows), &ids);

    // the GPU's embeddings and per-layer inputs
    {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        g4.forward.embedRows(m, e, &s, rows);
        g4.forward.pleInputs(m, e, &s, &m.kernels.ple.?, rows, m.config.geo.layers);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("gpu failed: {s}\n", .{why});
            return error.GpuFailed;
        }
        try blit(m, s.h[0], stage_h, rows * geo.hidden * 2);
        try blit(m, s.ple.?.pli, stage_p, rows * wide * 2);
    }
    const t = m.weights.ple.?;
    const emb = try gpa.dupe(u16, stage_h.slice(u16, rows * geo.hidden)); // kept: the staging buffer is reused below
    defer gpa.free(emb);
    const gpu_p = stage_p.slice(u16, rows * wide);
    const table = try gpa.alloc(f32, rows * wide);
    defer gpa.free(table);
    for (ids, 0..) |id, r| tableRow(t, id, wide, table[r * wide ..][0..wide]);

    const cwd = std.Io.Dir.cwd();
    const file = try cwd.createFile(io, out_path, .{});
    defer file.close(io);
    var wbuf: [1 << 20]u8 = undefined;
    var fw = file.writer(io, &wbuf);
    const w = &fw.interface;
    try put(w, u32, &.{ @intCast(rows), @intCast(geo.layers), @intCast(width), @intCast(geo.hidden) });
    try put(w, u16, emb);
    try put(w, f32, table);
    try put(w, u16, raw(t.model_projection.w, wide * geo.hidden));
    try put(w, u16, raw(t.projection_norm, width));
    try put(w, u16, gpu_p);

    // apply: each layer's residual rows through the GPU's block, fed the GPU's per-layer input
    var prng = std.Random.DefaultPrng.init(0x4e34b);
    const start = try gpa.alloc(u16, rows * geo.hidden);
    defer gpa.free(start);
    for (0..geo.layers) |l| {
        const ld = m.weights.layers[l];
        const scale: f32 = 1.0 + 0.5 * @as(f32, @floatFromInt(l));
        const up = stage_h.slice(u16, rows * geo.hidden);
        for (start, emb, up) |*o, e0, *u| {
            const noise = (prng.random().float(f32) * 2 - 1) * 2.0;
            o.* = bits(bf(e0) * scale + noise);
            u.* = o.*;
        }
        try blit(m, stage_h, s.h[1], rows * geo.hidden * 2);
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        g4.forward.pleTail(m, e, &s, l, 0, rows);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |why| {
            std.debug.print("gpu failed in layer {d}: {s}\n", .{ l, why });
            return error.GpuFailed;
        }
        try blit(m, s.h[1], stage_h, rows * geo.hidden * 2);
        try put(w, u16, raw(ld.ple_gate.?.w, width * geo.hidden));
        try put(w, u16, raw(ld.ple_projection.?.w, geo.hidden * width));
        try put(w, u16, raw(ld.ple_norm.?, geo.hidden));
        try put(w, u16, raw(ld.scalar, 1));
        try put(w, u16, start);
        try put(w, u16, stage_h.slice(u16, rows * geo.hidden));
    }
    try w.flush();
    std.debug.print("wrote {s}: {d} tokens, {d} layers\n", .{ out_path, rows, geo.layers });
}
