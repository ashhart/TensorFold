//! Synthetic large-matrix timings and byte checks compare candidate windows with a retained baseline shader.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core_lane");
const a = std.heap.page_allocator;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const Args = struct { baseline: []const u8, n: usize, k: usize, sk: usize, pf: usize = 1 };
fn execute(queue: mtl.Queue, pipe: mtl.Pipeline, l: core.Layout, x: mtl.Buffer, w: mtl.Buffer, sb: mtl.Buffer, y: mtl.Buffer, rows: u32, repeats: usize, prepared: ?core.Projection) !f64 {
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(x, 0, 0);
    e.setBuffer(w, 0, 1);
    e.setBuffer(sb, 0, 2);
    e.setValue(@as(i32, @intCast(rows)), 3);
    e.setBuffer(y, 0, 4);
    for (0..repeats) |_| {
        if (prepared) |projection| try projection.encode(e, .{ .buf = x }, .{ .buf = w }, .{ .buf = sb }, .{ .buf = y }, rows) else e.dispatchGroups(.{ .width = l.n / 32 }, .{ .width = 32 * l.sk });
        e.barrier();
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailure;
    return cb.gpuSeconds() / @as(f64, @floatFromInt(repeats));
}
fn median(v: *[5]f64) f64 {
    std.mem.sort(f64, v, {}, std.sort.asc(f64));
    return v[2];
}
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 6 or args.len > 9) return error.Usage;
    if (args.len >= 7 and !std.mem.eql(u8, args[6], "sums") and !std.mem.eql(u8, args[6], "inline")) return error.Usage;
    if (args.len >= 8 and !std.mem.eql(u8, args[7], "pair")) return error.Usage;
    if (args.len == 9 and !std.mem.eql(u8, args[8], "f32")) return error.Usage;
    const config = Args{ .baseline = args[1], .n = try std.fmt.parseInt(usize, args[2], 10), .k = try std.fmt.parseInt(usize, args[3], 10), .sk = try std.fmt.parseInt(usize, args[4], 10), .pf = try std.fmt.parseInt(usize, args[5], 10) };
    const l = core.Layout{ .n = config.n, .k = config.k, .sk = config.sk, .pf = config.pf, .format = .{ .bits = 4, .group = 64 }, .precompute_sums = args.len >= 7 and !std.mem.eql(u8, args[6], "inline"), .cooperative = args.len >= 8, .output = if (args.len == 9) .f32 else .bf16 };
    try l.validate();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const current = try core.source(arena, l);
    const shader_start = std.mem.indexOf(u8, current, "#include <metal_stdlib>") orelse return error.BadSource;
    const old = try std.Io.Dir.cwd().readFileAlloc(init.io, config.baseline, arena, .limited(1 << 20));
    const control_source = try std.mem.concat(arena, u8, &.{ current[0..shader_start], old });
    const control_lib = try mtl.Library.fromSource(device, control_source, mtl.CompileOptions.mlx());
    defer control_lib.deinit();
    const control = try mtl.Pipeline.init(device, control_lib, "tf_lane", false);
    defer control.deinit();
    const candidate = try core.Projection.init(arena, device, l);
    defer candidate.deinit();
    const x = try device.buffer(16 * l.k * 2, opts);
    defer x.deinit();
    const w = try device.buffer(l.weightWords() * 4, opts);
    defer w.deinit();
    const sb = try device.buffer(l.metadataElements() * 2, opts);
    defer sb.deinit();
    const output_bytes: usize = if (l.output == .f32) 4 else 2;
    const expected = try device.buffer(16 * l.n * output_bytes, opts);
    defer expected.deinit();
    const actual = try device.buffer(16 * l.n * output_bytes, opts);
    defer actual.deinit();
    var random = std.Random.DefaultPrng.init(28);
    const rng = random.random();
    for (w.slice(u32, l.weightWords())) |*v| v.* = rng.int(u32);
    for (x.slice(u16, 16 * l.k)) |*v| {
        const value = @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -127, 128))) * 0.015625;
        v.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    for (sb.slice(u16, l.metadataElements()), 0..) |*v, i| v.* = if (i % 2 == 0) 0x3bc0 else 0xbd34;
    _ = try execute(queue, control, l, x, w, sb, expected, 16, 1, null);
    for (1..17) |rows| {
        _ = try execute(queue, candidate.pipe, l, x, w, sb, actual, @intCast(rows), 1, candidate);
        if (!std.mem.eql(u8, expected.contents()[0 .. rows * l.n * output_bytes], actual.contents()[0 .. rows * l.n * output_bytes])) return error.BaselineBytesDiffer;
    }
    _ = try execute(queue, control, l, x, w, sb, expected, 1, 500, null);
    _ = try execute(queue, candidate.pipe, l, x, w, sb, actual, 1, 500, candidate);
    for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| {
        var old_time: [5]f64 = undefined;
        var new_time: [5]f64 = undefined;
        for (&old_time, &new_time) |*before, *after| {
            before.* = try execute(queue, control, l, x, w, sb, expected, rows, 100, null);
            after.* = try execute(queue, candidate.pipe, l, x, w, sb, actual, rows, 100, candidate);
        }
        const before = median(&old_time);
        const after = median(&new_time);
        std.debug.print("N{d} K{d} SK{d} PF{d} rows{d} baseline_us{d:.3} candidate_us{d:.3} weight_GBs{d:.1} payload_GBs{d:.1} byte_equal1-16\n", .{ l.n, l.k, l.sk, l.pf, rows, before * 1e6, after * 1e6, @as(f64, @floatFromInt(l.weightWords() * 4)) / after / 1e9, @as(f64, @floatFromInt(l.packedBytes())) / after / 1e9 });
    }
}
