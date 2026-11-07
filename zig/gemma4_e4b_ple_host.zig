//! tf-gemma4-e4b-ple's dump against layer_input.zig: the host's `inputs` and `apply` on the GPU's inputs, per layer.
const std = @import("std");
const li = @import("src/families/gemma4/layer_input.zig");
const cfg = @import("src/families/gemma4/config.zig");

fn bf(b16: u16) f32 {
    return @bitCast(@as(u32, b16) << 16);
}

fn bits(v: f32) u16 {
    return @truncate(@as(u32, @bitCast(v)) >> 16);
}

const Tally = struct {
    differ: usize = 0,
    total: usize = 0,
    worst: f32 = 0,

    fn add(t: *Tally, gpu: u16, host: f32) void {
        t.total += 1;
        if (gpu != bits(host)) t.differ += 1;
        t.worst = @max(t.worst, @abs(bf(gpu) - host));
    }
};

const Reader = struct {
    bytes: []const u8,
    at: usize = 0,

    fn take(r: *Reader, comptime T: type, n: usize) []const T {
        const len = n * @sizeOf(T);
        const out = std.mem.bytesAsSlice(T, r.bytes[r.at..][0..len]);
        r.at += len;
        return @alignCast(out);
    }

    fn floats(r: *Reader, gpa: std.mem.Allocator, n: usize) ![]f32 {
        const v = r.take(u16, n);
        const out = try gpa.alloc(f32, n);
        for (out, v) |*o, b| o.* = bf(b);
        return out;
    }
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const path = args.next() orelse return error.Usage;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    var r: Reader = .{ .bytes = bytes };
    const head = r.take(u32, 4);
    const rows: usize = head[0];
    const layers: usize = head[1];
    const width: usize = head[2];
    const hidden: usize = head[3];
    const geo = cfg.e4b;
    if (geo.layers != layers or geo.layer_input != width or geo.hidden != hidden) return error.NotTheE4B;
    const wide = layers * width;
    const emb = try r.floats(gpa, rows * hidden);
    defer gpa.free(emb);
    const table = r.take(f32, rows * wide);
    const project = try r.floats(gpa, wide * hidden);
    defer gpa.free(project);
    const norm = try r.floats(gpa, width);
    defer gpa.free(norm);
    const gpu_p = r.take(u16, rows * wide);
    const model: li.Model = .{ .project = project, .norm = norm };

    const input_tally = try gpa.alloc(Tally, layers);
    defer gpa.free(input_tally);
    @memset(input_tally, .{});
    const host_p = try gpa.alloc(f32, wide);
    defer gpa.free(host_p);
    for (0..rows) |row| {
        li.inputs(geo, model, emb[row * hidden ..][0..hidden], table[row * wide ..][0..wide], host_p);
        for (0..layers) |l| for (0..width) |j|
            input_tally[l].add(gpu_p[row * wide + l * width + j], host_p[l * width + j]);
    }

    const apply_tally = try gpa.alloc(Tally, layers);
    defer gpa.free(apply_tally);
    @memset(apply_tally, .{});
    const host_in = try gpa.alloc(f32, width);
    defer gpa.free(host_in);
    const scratch = try gpa.alloc(f32, width + hidden);
    defer gpa.free(scratch);
    const h = try gpa.alloc(f32, hidden);
    defer gpa.free(h);
    for (0..layers) |l| {
        const gate = try r.floats(gpa, width * hidden);
        defer gpa.free(gate);
        const proj = try r.floats(gpa, hidden * width);
        defer gpa.free(proj);
        const lnorm = try r.floats(gpa, hidden);
        defer gpa.free(lnorm);
        const scalar = r.take(u16, 1)[0];
        const start = r.take(u16, rows * hidden);
        const out = r.take(u16, rows * hidden);
        const layer: li.Layer = .{ .gate = gate, .project = proj, .norm = lnorm, .scalar = bf(scalar) };
        for (0..rows) |row| {
            for (host_in, gpu_p[row * wide + l * width ..][0..width]) |*o, b| o.* = bf(b);
            for (h, start[row * hidden ..][0..hidden]) |*o, b| o.* = bf(b);
            li.apply(geo, layer, host_in, h, scratch);
            for (h, out[row * hidden ..][0..hidden]) |v, g| apply_tally[l].add(g, v);
        }
    }

    std.debug.print("E4B per-layer inputs, GPU against layer_input.zig, {d} tokens\n", .{rows});
    std.debug.print("layer  inputs differ/total   max|diff|    apply differ/total   max|diff|\n", .{});
    var ti: Tally = .{};
    var ta: Tally = .{};
    for (0..layers) |l| {
        std.debug.print("{d:5}  {d:6}/{d:<8}  {e:10.3}  {d:6}/{d:<8}  {e:10.3}\n", .{ l, input_tally[l].differ, input_tally[l].total, input_tally[l].worst, apply_tally[l].differ, apply_tally[l].total, apply_tally[l].worst });
        ti.differ += input_tally[l].differ;
        ti.total += input_tally[l].total;
        ti.worst = @max(ti.worst, input_tally[l].worst);
        ta.differ += apply_tally[l].differ;
        ta.total += apply_tally[l].total;
        ta.worst = @max(ta.worst, apply_tally[l].worst);
    }
    std.debug.print("all    {d:6}/{d:<8}  {e:10.3}  {d:6}/{d:<8}  {e:10.3}\n", .{ ti.differ, ti.total, ti.worst, ta.differ, ta.total, ta.worst });
}
