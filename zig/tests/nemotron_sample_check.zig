//! tf_sample_full on this GPU against its host reference on synthetic rows, timed beside tf_gpu_sample: exit 0 when all match.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");

const full = tf.lanes.gpu_full;
const Sampling = tf.lanes.Sampling;
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// Rows a dispatch: one row at each position from `position`, as a decode window draws them.
const rows = 16;
const position: u32 = 4096;

const Case = struct { name: []const u8, s: Sampling, fill: *const fn ([]f32) void, vocab: usize = 131072, map: bool = false };

fn flat(l: []f32) void {
    @memset(l, 0.0);
}

fn peaked(l: []f32) void {
    for (l, 0..) |*x, i| x.* = 12.0 - 0.05 * @as(f32, @floatFromInt(i % 8192)) - @as(f32, @floatFromInt(i / 8192));
}

fn spread(l: []f32) void {
    var x: u64 = 0x9E37_79B9_7F4A_7C15;
    for (l) |*v| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        v.* = -3.0 * @as(f32, @floatFromInt(x >> 40)) / 16777216.0;
    }
}

fn ties(l: []f32) void {
    for (l, 0..) |*v, i| v.* = -0.5 * @as(f32, @floatFromInt((i * 2654435761) % 7));
}

fn twoLevels(l: []f32) void {
    @memset(l, -10.0);
    @memset(l[0..2000], 0.0);
}

const cases = [_]Case{
    .{ .name = "flat, top_k off, top_p 1", .s = .{ .seed = 1, .temperature = 1.0, .top_k = 0, .top_p = 1.0 }, .fill = flat },
    .{ .name = "peaked, top_p 0.9 (nucleus inside 1,024)", .s = .{ .seed = 2, .temperature = 0.7, .top_k = 0, .top_p = 0.9 }, .fill = peaked },
    .{ .name = "spread, top_p 0.9 (nucleus past 1,024)", .s = .{ .seed = 3, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = spread },
    .{ .name = "7 tied levels, top_p 0.5", .s = .{ .seed = 4, .temperature = 1.0, .top_k = 0, .top_p = 0.5 }, .fill = ties },
    .{ .name = "spread, top_k 3000", .s = .{ .seed = 5, .temperature = 1.0, .top_k = 3000, .top_p = 1.0 }, .fill = spread },
    .{ .name = "spread, top_k 3000, top_p 0.5", .s = .{ .seed = 6, .temperature = 1.0, .top_k = 3000, .top_p = 0.5 }, .fill = spread },
    .{ .name = "two levels, min_p 0.01", .s = .{ .seed = 7, .temperature = 1.0, .top_k = 0, .top_p = 1.0, .min_p = 0.01 }, .fill = twoLevels },
    .{ .name = "draft vocabulary through an id map", .s = .{ .seed = 8, .temperature = 1.0, .top_k = 0, .top_p = 0.9 }, .fill = spread, .vocab = 49152, .map = true },
};

fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    return @truncate((b + 0x7FFF + ((b >> 16) & 1)) >> 16);
}

/// The generated tf_gpu_sample for the full vocabulary, for timing beside tf_sample_full.
fn generated(key: []const u8) sources.nemotron.Kernel {
    for (sources.nemotron.all) |k| if (std.mem.eql(u8, k.key, key)) return k;
    unreachable;
}

/// One dispatch of `rows` rows through `p`; its GPU milliseconds.
fn dispatch(queue: mtl.Queue, p: mtl.Pipeline, logits: mtl.Buffer, s: Sampling, tok: mtl.Buffer, ids: mtl.Buffer, vocab: usize, whole: bool) !f64 {
    var seeds: [2 * rows]u32 = undefined;
    var positions: [rows]u32 = undefined;
    var cfgs: [4 * rows]f32 = undefined;
    var caps: [rows]u32 = undefined;
    for (0..rows) |r| {
        seeds[2 * r] = @truncate(s.seed);
        seeds[2 * r + 1] = @truncate(s.seed >> 32);
        positions[r] = position + @as(u32, @intCast(r));
        cfgs[4 * r ..][0..4].* = .{ @floatCast(1.0 / @max(s.temperature, 1e-6)), @floatCast(s.top_p), 20.0, @floatCast(s.minLog()) };
        caps[r] = s.top_k;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p);
    enc.setBuffer(logits, 0, 0);
    enc.setBytes(std.mem.sliceAsBytes(&seeds), 1);
    enc.setBytes(std.mem.sliceAsBytes(&positions), 2);
    enc.setBytes(std.mem.sliceAsBytes(&cfgs), 3);
    enc.setBytes(std.mem.sliceAsBytes(&caps), 4);
    enc.setBuffer(tok, 0, 5);
    if (whole) {
        enc.setBuffer(ids, 0, 6);
        enc.setBytes(std.mem.asBytes(&@as(u32, @intCast(vocab))), 7);
    }
    enc.dispatchThreads(mtl.Size.of(1024 * rows, 1, 1), mtl.Size.of(1024, 1, 1));
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |text| {
        std.debug.print("command buffer failed: {s}\n", .{text});
        return error.GpuFailed;
    }
    return cb.gpuSeconds() * 1e3;
}

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.nemotron_sample, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const old_kernel = generated("sample");
    const old_lib = try mtl.Library.fromSource(device, old_kernel.source, mtl.CompileOptions.mlx());
    defer old_lib.deinit();
    const old = try mtl.Pipeline.init(device, old_lib, old_kernel.function, false);
    defer old.deinit();
    std.debug.print("device: {s}\n", .{device.name()});
    var failed: usize = 0;
    for (cases) |c| {
        const p = try mtl.Pipeline.init(device, lib, if (c.map) "tf_sample_full_ids" else "tf_sample_full", false);
        defer p.deinit();
        const row = try gpa.alloc(f32, c.vocab);
        defer gpa.free(row);
        c.fill(row);
        const logits = try device.buffer(rows * c.vocab * 2, opts);
        defer logits.deinit();
        const words = logits.slice(u16, rows * c.vocab);
        for (row, 0..) |*x, i| {
            const h = bf16(x.*);
            x.* = @bitCast(@as(u32, h) << 16);
            for (0..rows) |r| words[r * c.vocab + i] = h;
        }
        const map = try gpa.alloc(u32, c.vocab);
        defer gpa.free(map);
        for (map, 0..) |*m, i| m.* = @intCast(2 * i + 1);
        const ids = try device.buffer(c.vocab * 4, opts);
        defer ids.deinit();
        @memcpy(ids.slice(u32, c.vocab), map);
        const tok = try device.buffer(rows * 4, opts);
        defer tok.deinit();
        _ = try dispatch(queue, p, logits, c.s, tok, ids, c.vocab, true);
        const ms = try dispatch(queue, p, logits, c.s, tok, ids, c.vocab, true);
        const plan = try full.plan(gpa, row, c.s);
        var same: usize = 0;
        for (tok.slice(u32, rows), 0..) |got, r| {
            const want = full.race(row, plan, c.s.seed, position + @as(u32, @intCast(r)), if (c.map) map else null);
            same += @intFromBool(got == want);
            if (got != want) std.debug.print("  row {d}: GPU {d}, host {d}\n", .{ r, got, want });
        }
        failed += rows - same;
        var old_ms: f64 = 0.0;
        if (!c.map and c.vocab == 131072) for (0..2) |_| {
            old_ms = try dispatch(queue, old, logits, c.s, tok, ids, c.vocab, false);
        };
        std.debug.print("{s}: {d}/{d} rows match, tf_sample_full {d:.3} ms, tf_gpu_sample {d:.3} ms ({d} rows)\n", .{ c.name, same, rows, ms, old_ms, rows });
    }
    std.debug.print("{s}\n", .{if (failed == 0) "all draws match" else "draws differ"});
    if (failed > 0) std.process.exit(1);
}
