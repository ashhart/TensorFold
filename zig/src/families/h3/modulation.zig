//! A run's modulation tables from the checkpoint: the timestep embedder on the host, each block's AdaLN on the GPU.
const std = @import("std");
const mtl = @import("metal");
const h3 = @import("h3.zig");
const schedule = @import("schedule.zig");
const shape = @import("modulation_shape.zig");

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
    const y = try a.alloc(f32, try shape.linear(x.len, w.len, n, rows, k));
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
    const dim = try shape.embedding(ck.entry(n.in_w) orelse return error.MissingTensor, ck.entry(n.in_b) orelse return error.MissingTensor, ck.entry(n.out_w) orelse return error.MissingTensor, ck.entry(n.out_b) orelse return error.MissingTensor, table.len);
    const w_in = try ck.floats(a, n.in_w);
    const b_in = try ck.floats(a, n.in_b);
    const w_out = try ck.floats(a, n.out_w);
    const b_out = try ck.floats(a, n.out_b);
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
    const n = try shape.projection(t, k, groups, parts, width);
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
    try schedule.validate(video);
    try schedule.validate(audio);
    if (video.timesteps.len != audio.timesteps.len) return error.H3BadSchedule;
    const rows = m.rows;
    // a row's modality is the case's, fixed by the packing: its first step's line modulo the modalities
    const tags = try a.alloc(i32, rows);
    for (tags, m.adaln[0..rows]) |*t, line| t.* = @mod(line, schedule.modalities);
    const p = try schedule.plan(a, .{ .text = m.text, .condition = m.condition, .audio = m.audio, .held_audio = m.held_audio, .video = m.video, .hold = hold }, video.timesteps, audio.timesteps);
    const adaln = try a.alloc(i32, p.index.len);
    for (adaln, p.index, 0..) |*o, i, at| o.* = i * schedule.modalities + tags[at % rows];
    const t = p.table.len;
    const lines = try shape.tables(t, m.hidden);
    const x = try embed(m, a, p.table);
    const k = x.len / t;
    _ = try shape.projection(t, k, schedule.modalities, 6, m.hidden);
    _ = try shape.projection(t, k, 1, 1, 2 * m.hidden);
    const shared = mtl.ResourceOptions.shared;
    const input = try m.device.buffer(x.len * 2, shared);
    defer input.deinit();
    @memcpy(input.slice(u16, x.len), x);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    var op = Replacement{ .allocator = a, .model = m, .input = input, .cb = cb, .enc = enc, .t = t, .k = k, .adaln = adaln, .times = p.index, .lines = lines, .video = video, .audio = audio, .hold = hold };
    return @import("schedule_replace.zig").run(m, &op);
}

const Replacement = struct {
    allocator: std.mem.Allocator,
    model: *Model,
    input: mtl.Buffer,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    t: usize,
    k: usize,
    adaln: []i32,
    times: []i32,
    lines: usize,
    video: schedule.Schedule,
    audio: schedule.Schedule,
    hold: bool,
    const Ref = @FieldType(Model, "final_table");
    pub fn makeSteps(o: *@This()) !Steps {
        const v = try o.allocator.alloc([2]f32, o.video.timesteps.len);
        errdefer o.allocator.free(v);
        const a = try o.allocator.alloc([2]f32, o.audio.timesteps.len);
        for (v, 0..) |*s, i| s.* = if (o.hold) .{ 0.0, 1.0 } else o.video.euler(i);
        for (a, 0..) |*s, i| s.* = o.audio.euler(i);
        return .{ .video = v, .audio = a };
    }
    pub fn freeSteps(o: *@This(), s: Steps) void {
        o.allocator.free(s.video);
        o.allocator.free(s.audio);
    }
    pub fn block(o: *@This(), i: usize) !Ref {
        const b = try o.model.device.buffer(6 * o.lines * o.model.hidden * 2, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        var name: [128]u8 = undefined;
        try project(o.model, o.enc, o.input, try std.fmt.bufPrint(&name, "{s}{d}.adaln_proj.linear.", .{ names(o.model).block, i }), o.t, o.k, schedule.modalities, 6, o.model.hidden, b);
        return .{ .buffer = b };
    }
    pub fn final(o: *@This()) !Ref {
        const b = try o.model.device.buffer(o.t * 2 * o.model.hidden * 2, mtl.ResourceOptions.shared);
        errdefer b.deinit();
        try project(o.model, o.enc, o.input, names(o.model).final, o.t, o.k, 1, 1, 2 * o.model.hidden, b);
        return .{ .buffer = b };
    }
    pub fn finish(o: *@This()) !void {
        o.enc.end();
        o.cb.commit();
        o.cb.wait();
        if (o.cb.failure()) |_| return error.GpuFailed;
    }
    pub fn release(_: *@This(), ref: Ref) void {
        ref.buffer.deinit();
    }
};
