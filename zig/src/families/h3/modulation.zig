//! A run's modulation tables from the checkpoint: the timestep embedder on the host, each block's AdaLN on the GPU.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3.zig");
const schedule = @import("schedule.zig");

const Model = h3.Model;

/// Each step's Euler pairs for the tool: the sigma the model saw and the grid's ratio, video then audio.
pub const Steps = struct { video: [][2]f32, audio: [][2]f32 };

const Names = struct { in_w: []const u8, in_b: []const u8, out_w: []const u8, out_b: []const u8, block: []const u8, final: []const u8 };

/// The embedder's and projections' names: FastH3's diffusers names, or the released MiniMax-H3's.
fn names(m: *const Model) Names {
    if (m.base) return .{ .in_w = "time_embedder.proj_in.weight", .in_b = "time_embedder.proj_in.bias", .out_w = "time_embedder.proj_out.weight", .out_b = "time_embedder.proj_out.bias", .block = "blocks.", .final = "final_layer.adaln_proj.linear." };
    return .{ .in_w = "time_embedder.linear_1.weight", .in_b = "time_embedder.linear_1.bias", .out_w = "time_embedder.linear_2.weight", .out_b = "time_embedder.linear_2.bias", .block = "transformer_blocks.", .final = "norm_out.linear." };
}

/// y (rows, n) = x (rows, k) w^T + b on the host, summed in float64 and rounded once to float32.
fn linear(a: std.mem.Allocator, x: []const f32, w: []const f32, b: []const f32, rows: usize, k: usize) ![]f32 {
    const n = b.len;
    const y = try a.alloc(f32, rows * n);
    for (0..rows) |r| for (0..n) |o| {
        var s: f64 = 0;
        for (x[r * k ..][0..k], w[o * k ..][0..k]) |p, q| s += @as(f64, p) * q;
        y[r * n + o] = @floatCast(s + b[o]);
    };
    return y;
}

fn silu(v: f32) f32 {
    return v / (1.0 + @exp(-v));
}

/// silu(time_embedder(sinusoid(t))) in bf16 for every timestep of `table`: the AdaLN projections' input.
fn embed(m: *Model, a: std.mem.Allocator, table: []const f32) ![]u16 {
    const ck = &m.checkpoint;
    const n = names(m);
    const w_in = try ck.floats(a, n.in_w);
    const b_in = try ck.floats(a, n.in_b);
    const w_out = try ck.floats(a, n.out_w);
    const b_out = try ck.floats(a, n.out_b);
    const dim = w_in.len / b_in.len;
    const half = dim / 2;
    // cosine half first, angles and frequencies in float32 as MLX computes them
    const sinusoid = try a.alloc(f32, table.len * dim);
    for (table, 0..) |t, r| for (0..half) |i| {
        const freq = @exp(@as(f32, @floatCast(-@log(10000.0))) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half)));
        sinusoid[r * dim + i] = @cos(t * freq);
        sinusoid[r * dim + half + i] = @sin(t * freq);
    };
    const hidden = try linear(a, sinusoid, w_in, b_in, table.len, dim);
    for (hidden) |*v| v.* = silu(v.*);
    const temb = try linear(a, hidden, w_out, b_out, table.len, b_in.len);
    const out = try a.alloc(u16, temb.len);
    for (out, temb) |*o, v| o.* = bf16(silu(v));
    return out;
}

/// float32 to bf16, round to nearest even.
fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

/// One projection's tables into `out`: rows of `groups` x `parts` x `width`, laid out (part, timestep x group, width).
fn project(m: *const Model, enc: mtl.ComputeEncoder, x: mtl.Buffer, prefix: []const u8, t: usize, k: usize, groups: usize, parts: usize, width: usize, out: mtl.Buffer) !void {
    var name: [128]u8 = undefined;
    const n = groups * parts * width;
    const w = try m.checkpoint.ref(try std.fmt.bufPrint(&name, "{s}weight", .{prefix}), n * k);
    const b = try m.checkpoint.ref(try std.fmt.bufPrint(&name, "{s}bias", .{prefix}), n);
    enc.setPipeline(m.pipeline("h3_modulation"));
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(w.buffer, w.offset, 1);
    enc.setBuffer(b.buffer, b.offset, 2);
    enc.setValue([6]i32{ @intCast(t), @intCast(k), @intCast(n), @intCast(groups), @intCast(parts), @intCast(width) }, 3);
    enc.setBuffer(out, 0, 4);
    enc.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
}

/// The case's schedule replaced (plan, lines, tables); `hold` keeps all but the generated audio at the keyframe level.
pub fn reschedule(m: *Model, a: std.mem.Allocator, video: schedule.Schedule, audio: schedule.Schedule, hold: bool) !Steps {
    const rows = m.rows;
    // a row's modality is the case's, fixed by the packing: its first step's line modulo the modalities
    const tags = try a.alloc(i32, rows);
    for (tags, m.adaln[0..rows]) |*t, line| t.* = @mod(line, schedule.modalities);
    const p = try schedule.plan(a, .{ .text = m.text, .condition = m.condition, .audio = m.audio, .video = m.video, .hold = hold }, video.timesteps, audio.timesteps);
    const adaln = try a.alloc(i32, p.index.len);
    for (adaln, p.index, 0..) |*o, i, at| o.* = i * schedule.modalities + tags[at % rows];
    m.adaln = adaln;
    m.times = p.index;
    const t = p.table.len;
    m.lines = t * schedule.modalities;
    const x = try embed(m, a, p.table);
    const k = x.len / t;
    const shared = mtl.ResourceOptions.shared;
    const input = try m.device.buffer(x.len * 2, shared);
    @memcpy(input.slice(u16, x.len), x);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    var name: [128]u8 = undefined;
    for (m.blocks, 0..) |*b, i| {
        b.tables = .{ .buffer = try m.device.buffer(6 * m.lines * m.hidden * 2, shared) };
        try project(m, enc, input, try std.fmt.bufPrint(&name, "{s}{d}.adaln_proj.linear.", .{ names(m).block, i }), t, k, schedule.modalities, 6, m.hidden, b.tables.buffer);
    }
    m.final_table = .{ .buffer = try m.device.buffer(t * 2 * m.hidden * 2, shared) };
    try project(m, enc, input, names(m).final, t, k, 1, 1, 2 * m.hidden, m.final_table.buffer);
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("modulation failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    const steps = Steps{ .video = try a.alloc([2]f32, video.timesteps.len), .audio = try a.alloc([2]f32, audio.timesteps.len) };
    for (steps.video, 0..) |*s, i| s.* = if (hold) .{ 0.0, 1.0 } else video.euler(i);
    for (steps.audio, 0..) |*s, i| s.* = audio.euler(i);
    return steps;
}
