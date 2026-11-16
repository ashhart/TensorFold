//! The two-forward stage diff over an N-row window, localising the first differing row, layer and stage.

const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const Model = tf.deepseek_v41.Model;
const attention = tf.deepseek_v41.attention;
const Stage = tf.deepseek_v41_profiler.Stage;

pub const attention_widths = [_]usize{ 1280, 1280, 32768, 32768, 512, 512, 512, 512, 32768, 32768, 8192, 5120 };
pub const attention_elements = blk: {
    var total: usize = 0;
    for (attention_widths) |width| total += width;
    break :blk total;
};
pub fn attentionOffset(stage: attention.DiagnosticStage) usize {
    var offset: usize = 0;
    for (attention_widths[0..@backingInt(stage)]) |width| offset += width;
    return offset;
}
/// One per-layer slot in the model observer's record: stream (20480), normalized (5120), branch (5120) as bf16.
pub const coarse_stride = 61488;
pub const layer_count = 40;
pub const logits_width = 129280;
pub const max_rows = 4;

pub const coarse_field_offsets = [_]usize{ 0, 20480 * 2, 20480 * 2 + 5120 * 2, 61440, 61464 };
pub const coarse_field_lengths = [_]usize{ 20480 * 2, 5120 * 2, 5120 * 2, 24, 24 };
pub const coarse_names = [_][]const u8{
    "boundary_attention.stream", "boundary_attention.normalized", "boundary_attention.branch", "boundary_attention.route_ids", "boundary_attention.route_weights",
    "boundary_moe.stream",       "boundary_moe.normalized",       "boundary_moe.branch",       "boundary_moe.route_ids",       "boundary_moe.route_weights",
};

pub const Capture = struct {
    attention_buffer: mtl.Buffer,
    coarse_buffer: mtl.Buffer,
    pipeline: mtl.Pipeline,
    /// The window's width (the slot count) and how many source rows this forward carries.
    rows: u32 = 2,
    /// The slot of the forward's first row: 0 for a window forward, the row index for a sequential one.
    row_base: u32 = 0,
    /// Source rows to copy: the window's width, or 1 for a sequential 1-row forward.
    active: u32 = 1,

    pub fn attentionGpu(ctx: *anyopaque, enc: mtl.ComputeEncoder, layer: u32, stage: attention.DiagnosticStage, source: attention.View) !void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        // The packed window key is a different layout (fp8, 528-stride); it is a pure function of kv_rope.
        if (stage == .kv_quantized) return;
        if (layer >= layer_count) return error.InvalidLayer;
        const width = attention_widths[@backingInt(stage)];
        for (0..self.active) |r| {
            const slot = self.row_base + r;
            const destination = ((@as(usize, layer) * self.rows + slot) * attention_elements + attentionOffset(stage)) * 2;
            enc.setPipeline(self.pipeline);
            enc.setBuffer(source.buffer, source.offset + r * width * 2, 0);
            enc.setBuffer(self.attention_buffer, destination, 1);
            enc.setValue(@as(u32, @intCast(width / 2)), 2);
            enc.dispatchThreads(.{ .width = width / 2 }, .{ .width = 256 });
            enc.barrier();
        }
    }

    pub fn coarseGpu(ctx: *anyopaque, enc: mtl.ComputeEncoder, layer: u32, stage: Stage, model: *const Model) void {
        const self: *Capture = @ptrCast(@alignCast(ctx));
        if (layer >= layer_count) return;
        const ws = &model.workspace;
        for (0..self.active) |r| {
            const slot = self.row_base + r;
            var at = ((@as(usize, layer) * 2 + @intFromBool(stage == .moe)) * self.rows + slot) * coarse_stride;
            inline for (.{ ws.view(.stream), ws.view(.normalized), ws.view(.branch) }, [_]usize{ 20480 * 2, 5120 * 2, 5120 * 2 }, [_]usize{ 20480, 5120, 5120 }) |source, bytes, elements| {
                enc.setPipeline(self.pipeline);
                enc.setBuffer(source.buffer, source.offset + r * elements * 2, 0);
                enc.setBuffer(self.coarse_buffer, at, 1);
                enc.setValue(@as(u32, @intCast(bytes / 4)), 2);
                enc.dispatchThreads(.{ .width = bytes / 4 }, .{ .width = 256 });
                at += bytes;
            }
            inline for (.{ ws.view(.route_ids), ws.view(.route_weights) }, [_]usize{ 6, 6 }) |source, count| {
                enc.setPipeline(self.pipeline);
                enc.setBuffer(source.buffer, source.offset + r * count * 4, 0);
                enc.setBuffer(self.coarse_buffer, at, 1);
                enc.setValue(@as(u32, count), 2);
                enc.dispatchThreads(.{ .width = count }, .{ .width = 64 });
                at += count * 4;
            }
            enc.barrier();
        }
    }
};

pub const Metric = struct {
    unequal: usize = 0,
    elements: usize = 0,
    nrms: f64 = 0,
    max_abs: f32 = 0,
};

pub fn compareHalves(a: []const u16, b: []const u16) Metric {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var max_abs: f32 = 0;
    var unequal: usize = 0;
    for (a, b) |x, y| {
        if (x != y) unequal += 1;
        const xf: f32 = @bitCast(@as(u32, x) << 16);
        const yf: f32 = @bitCast(@as(u32, y) << 16);
        const d = @as(f64, xf) - @as(f64, yf);
        squared += d * d;
        denominator += @as(f64, yf) * yf;
        max_abs = @max(max_abs, @abs(xf - yf));
    }
    return .{ .unequal = unequal, .elements = a.len, .nrms = if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator), .max_abs = max_abs };
}

pub fn compareByteHalves(a: []const u8, b: []const u8) Metric {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var max_abs: f32 = 0;
    var unequal: usize = 0;
    var i: usize = 0;
    while (i + 1 < a.len) : (i += 2) {
        const x = std.mem.readInt(u16, a[i..][0..2], .little);
        const y = std.mem.readInt(u16, b[i..][0..2], .little);
        if (x != y) unequal += 1;
        const xf: f32 = @bitCast(@as(u32, x) << 16);
        const yf: f32 = @bitCast(@as(u32, y) << 16);
        const d = @as(f64, xf) - @as(f64, yf);
        squared += d * d;
        denominator += @as(f64, yf) * yf;
        max_abs = @max(max_abs, @abs(xf - yf));
    }
    return .{ .unequal = unequal, .elements = a.len / 2, .nrms = if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator), .max_abs = max_abs };
}

pub fn compareByteF32(a: []const u8, b: []const u8) Metric {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var max_abs: f32 = 0;
    var unequal: usize = 0;
    var i: usize = 0;
    while (i + 3 < a.len) : (i += 4) {
        const x: f32 = @bitCast(std.mem.readInt(u32, a[i..][0..4], .little));
        const y: f32 = @bitCast(std.mem.readInt(u32, b[i..][0..4], .little));
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) unequal += 1;
        const d = @as(f64, x) - @as(f64, y);
        squared += d * d;
        denominator += @as(f64, y) * y;
        max_abs = @max(max_abs, @abs(x - y));
    }
    return .{ .unequal = unequal, .elements = a.len / 4, .nrms = if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator), .max_abs = max_abs };
}

pub fn compareByteIds(a: []const u8, b: []const u8) Metric {
    var unequal: usize = 0;
    for (0..6) |i| {
        const x = std.mem.readInt(u32, a[i * 4 ..][0..4], .little);
        const y = std.mem.readInt(u32, b[i * 4 ..][0..4], .little);
        if (x != y) unequal += 1;
    }
    return .{ .unequal = unequal, .elements = 6 };
}

pub fn compareF32(a: []const f32, b: []const f32) Metric {
    var squared: f64 = 0;
    var denominator: f64 = 0;
    var max_abs: f32 = 0;
    var unequal: usize = 0;
    for (a, b) |x, y| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) unequal += 1;
        const d = @as(f64, x) - @as(f64, y);
        squared += d * d;
        denominator += @as(f64, y) * y;
        max_abs = @max(max_abs, @abs(x - y));
    }
    return .{ .unequal = unequal, .elements = a.len, .nrms = if (denominator == 0) @sqrt(squared) else @sqrt(squared / denominator), .max_abs = max_abs };
}

pub fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    var high = -std.math.inf(f32);
    for (row, 0..) |v, i| if (v > high) {
        high = v;
        best = @intCast(i);
    };
    return best;
}

pub const FirstDiff = struct {
    row: u32,
    layer: u32,
    item: []const u8,
    metric: Metric,
};

pub fn firstCaptureDifference(a_attn: []const u16, b_attn: []const u16, a_coarse: []const u8, b_coarse: []const u8, rows: u32) ?FirstDiff {
    for (0..rows) |row| {
        for (0..layer_count) |layer| {
            inline for (std.enums.values(attention.DiagnosticStage)) |stage| {
                if (stage != .kv_quantized) {
                    const base = (layer * rows + row) * attention_elements + attentionOffset(stage);
                    const width = attention_widths[@backingInt(stage)];
                    const a = a_attn[base..][0..width];
                    const b = b_attn[base..][0..width];
                    if (!std.mem.eql(u16, a, b)) {
                        return .{ .row = @intCast(row), .layer = @intCast(layer), .item = @tagName(stage), .metric = compareHalves(a, b) };
                    }
                }
            }
            inline for (0..2) |boundary| {
                const base = ((layer * 2 + boundary) * rows + row) * coarse_stride;
                const a = a_coarse[base..][0..coarse_stride];
                const b = b_coarse[base..][0..coarse_stride];
                // The `.attention` point is BEFORE this layer's MoE: its route fields are the previous forward's.
                const compared: usize = if (boundary == 0) coarse_field_offsets[2] + coarse_field_lengths[2] else coarse_stride;
                if (!std.mem.eql(u8, a[0..compared], b[0..compared])) {
                    inline for (0..5) |field| {
                        if (boundary != 0 or field < 3) {
                            const off = coarse_field_offsets[field];
                            const len = coarse_field_lengths[field];
                            if (!std.mem.eql(u8, a[off..][0..len], b[off..][0..len])) {
                                const metric: Metric = switch (field) {
                                    0, 1, 2 => compareByteHalves(a[off..][0..len], b[off..][0..len]),
                                    3 => compareByteIds(a[off..][0..len], b[off..][0..len]),
                                    else => compareByteF32(a[off..][0..len], b[off..][0..len]),
                                };
                                return .{ .row = @intCast(row), .layer = @intCast(layer), .item = coarse_names[boundary * 5 + field], .metric = metric };
                            }
                        }
                    }
                }
            }
        }
    }
    return null;
}

pub fn secondBest(row: []const f32, first: u32) u32 {
    var best: u32 = if (first == 0) 1 else 0;
    var high = -std.math.inf(f32);
    for (row, 0..) |v, i| {
        if (i == first) continue;
        if (v > high) {
            high = v;
            best = @intCast(i);
        }
    }
    return best;
}
