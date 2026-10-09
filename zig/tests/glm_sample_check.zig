//! Check the real GLM sampler and its production bindings against the host reference without a checkpoint.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const dispatch = @import("draw_dispatch");
const draw = dispatch.rules;
const sources = @import("kernel_sources");
const sentinel: u32 = 0xDEAD_BEEF;
const rows = 4;

const Case = struct {
    name: []const u8,
    vocab: usize,
    top_k: u32 = 0,
    top_p: f64 = 1,
    min_p: f64 = 0,
    temperature: f64 = 1,
    shape: enum { flat, levels, min_p, shuffled } = .flat,
};

const cases = [_]Case{
    .{ .name = "min_p normalization", .vocab = 3, .top_p = 0.5, .min_p = 0.8, .shape = .min_p },
    .{ .name = "arrival order", .vocab = 2049, .top_p = 0.5, .shape = .levels },
    .{ .name = "large top_k", .vocab = 4096, .top_k = 3000, .top_p = 0.5 },
    .{ .name = "large top_k unfiltered", .vocab = 4096, .top_k = 3000 },
    .{ .name = "large nucleus", .vocab = 2048, .top_p = 0.75 },
    .{ .name = "tiny vocabulary", .vocab = 1 },
    .{ .name = "top_k one", .vocab = 17, .top_k = 1, .shape = .shuffled },
    .{ .name = "small filtered top_k", .vocab = 1031, .top_k = 41, .top_p = 0.7, .temperature = 0.7, .shape = .shuffled },
    .{ .name = "top_k 1024", .vocab = 4097, .top_k = 1024, .top_p = 0.5 },
    .{ .name = "top_k 1025", .vocab = 4097, .top_k = 1025, .top_p = 0.5 },
    .{ .name = "top_k at vocabulary", .vocab = 2048, .top_k = 2048, .top_p = 0.75 },
    .{ .name = "top_k beyond vocabulary", .vocab = 2048, .top_k = 9999, .top_p = 0.75 },
    .{ .name = "shuffled ties", .vocab = 4097, .top_p = 0.9, .shape = .shuffled },
    .{ .name = "large vocabulary", .vocab = 65539, .top_p = 0.9, .temperature = 0.7, .shape = .shuffled },
    .{ .name = "hot distribution", .vocab = 1031, .top_p = 0.9, .temperature = 1.5, .shape = .shuffled },
};

fn value(c: Case, i: usize) f32 {
    return switch (c.shape) {
        .flat => 0,
        .levels => if (i < 1024) 1 else if (i < 2048) 10 else -10,
        .min_p => ([_]f32{ 0, -0.10009765625, -1 })[i],
        .shuffled => -0.25 * @as(f32, @floatFromInt((i * 37 + 11) % 19)),
    };
}

fn run(queue: mtl.Queue, pipeline: mtl.Pipeline, logits: mtl.Buffer, offset: usize, picks: mtl.Buffer, payload: *const draw.Payload) !void {
    @memset(picks.slice(u32, rows + 2), sentinel);
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    dispatch.encode(enc, pipeline, logits, offset, picks, 4, payload);
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |reason| {
        std.debug.print("Metal failure: {s}\n", .{reason});
        return error.GpuFailed;
    }
    const got = picks.slice(u32, rows + 2);
    if (got[0] != sentinel or got[payload.header.n + 1] != sentinel) return error.OutputBounds;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const lib = try mtl.Library.fromSource(device, sources.glm_sample, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipeline = try mtl.Pipeline.init(device, lib, "glm_sample", false);
    defer pipeline.deinit();
    std.debug.print("GLM sampling device: {s}\n", .{device.name()});
    for (cases) |c| {
        const row = try gpa.alloc(f32, c.vocab);
        defer gpa.free(row);
        for (row, 0..) |*v, i| v.* = value(c, i);
        const logits = try device.buffer(rows * c.vocab * 2, mtl.ResourceOptions.shared);
        defer logits.deinit();
        const words = logits.slice(u16, rows * c.vocab);
        for (0..rows) |r| for (row, 0..) |v, i| {
            words[r * c.vocab + i] = @truncate(@as(u32, @bitCast(v)) >> 16);
        };
        const picks = try device.buffer((rows + 2) * 4, mtl.ResourceOptions.shared);
        defer picks.deinit();
        const configs = [_]?lanes.Sampling{
            .{ .seed = 0, .temperature = c.temperature, .top_k = c.top_k, .top_p = c.top_p, .min_p = c.min_p },
            null,
            .{ .seed = 0x8000_0000_0000_0001, .temperature = c.temperature, .top_k = c.top_k, .top_p = c.top_p, .min_p = c.min_p },
            null,
        };
        var expected: [rows]u32 = undefined;
        var d: draw.Draws = .{};
        d.n = rows - 1;
        for (configs, 0..) |cfg, r| {
            const position: u32 = 17 + @as(u32, @intCast(r));
            expected[r] = if (cfg) |s| try lanes.gpu_full.sample(gpa, row, s, position, null) else lanes.gpu_rule.argmax(row);
            d.rules[r] = if (cfg) |s| draw.ruleOf(s, position, @intCast(c.vocab)) else draw.Rule.greedy(@intCast(c.vocab));
        }
        for (0..8) |repeat| {
            const reverse = repeat % 2 == 1;
            const payload = draw.Payload.init(&d, rows, @intCast(c.vocab));
            var ordered = payload;
            if (reverse) std.mem.reverse(draw.Rule, ordered.rules[0..rows]);
            try run(queue, pipeline, logits, 0, picks, &ordered);
            for (picks.slice(u32, rows + 2)[1 .. rows + 1], 0..) |got, r| {
                const want = expected[if (reverse) rows - 1 - r else r];
                if (got != want) {
                    std.debug.print("FAIL {s}: repeat {d} row {d}, got {d}, expected {d}\n", .{ c.name, repeat, r, got, want });
                    return error.SampleMismatch;
                }
            }
        }
        for (0..rows) |r| {
            var solo: draw.Draws = .{};
            solo.rules[0] = d.rules[r];
            solo.n = 1;
            const payload = draw.Payload.init(&solo, 1, @intCast(c.vocab));
            try run(queue, pipeline, logits, r * c.vocab * 2, picks, &payload);
            if (picks.slice(u32, rows + 2)[1] != expected[r]) return error.SoloMismatch;
        }
        std.debug.print("PASS {s}: repeated, reordered, mixed, trailing and solo rows\n", .{c.name});
    }
}
