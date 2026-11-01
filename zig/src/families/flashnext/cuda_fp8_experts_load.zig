//! The MTP head's block-FP8 routed experts: E4M3 weights, BF16 128x128 weight_scale_inv, packed for Fp8Experts (W8A16).
const std = @import("std");
const cuda = @import("cuda");
const experts = @import("cuda_modelopt_experts.zig");
const wt = @import("cuda_prompt_wt.zig");
const fx = cuda.experts.Fp8Experts;

const blk = fx.block;

/// One expert's share of each buffer: gate|up words, down words, f32 scales (gate then up), down scales.
pub const Expert = struct { up: []u32, down: []u32, up_scale: []f32, down_scale: []f32 };

/// The words and scales one expert takes in each buffer, `width` intermediate channels by `dims` model columns.
pub const Shape = struct {
    up: usize,
    down: usize,
    up_scale: usize,
    down_scale: usize,

    pub fn of(dims: usize, width: usize) !Shape {
        if (dims == 0 or width == 0 or dims % blk != 0 or width % blk != 0) return error.UnexpectedTensor;
        const weights = std.math.mul(usize, dims, width) catch return error.UnexpectedTensor;
        const blocks = (dims / blk) * (width / blk);
        return .{ .up = weights / 2, .down = weights / 4, .up_scale = 2 * blocks, .down_scale = blocks };
    }
};

/// A layer of `count` experts on one GPU; a rank cut would split the 128-wide scale blocks, so world > 1 is refused.
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, world: usize, dims: usize, width: usize, count: usize) !experts.Owned {
    defer store.release(); // loadShape's format probe borrowed the first gate weight
    if (world != 1) return error.UnevenExpertSplit;
    const one = try Shape.of(dims, width);
    if (count == 0) return error.UnexpectedTensor;
    const up_total = std.math.mul(usize, count, one.up * 4) catch return error.UnexpectedTensor;
    const down_total = std.math.mul(usize, count, one.down * 4) catch return error.UnexpectedTensor;
    var up = try cuda.DeviceBuffer.alloc(d, up_total);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, down_total);
    errdefer down.free();
    const up_scale = try a.alloc(f32, count * one.up_scale);
    defer a.free(up_scale);
    const down_scale = try a.alloc(f32, count * one.down_scale);
    defer a.free(down_scale);
    const up_words = try a.alloc(u32, one.up);
    defer a.free(up_words);
    const down_words = try a.alloc(u32, one.down);
    defer a.free(down_words);
    for (0..count) |i| {
        const at: Expert = .{ .up = up_words, .down = down_words, .up_scale = up_scale[i * one.up_scale ..][0..one.up_scale], .down_scale = down_scale[i * one.down_scale ..][0..one.down_scale] };
        try packExpert(a, store, prefix, i, dims, width, at);
        // a failed wait may leave a copy writing `up` or `down`: drain before the errdefers free them (fromHost)
        up.upload(i * one.up * 4, std.mem.sliceAsBytes(up_words)) catch |e| return drained(d, e);
        down.upload(i * one.down * 4, std.mem.sliceAsBytes(down_words)) catch |e| return drained(d, e);
    }
    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_scale));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_scale));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count, .format = .fp8g };
}

fn drained(d: *const cuda.Driver, e: anyerror) anyerror {
    cuda.completion.drain(d, .{ .d = d, .handle = null }, "an FP8 expert upload's default stream and context did not drain");
    return e;
}

/// Expert `i` checked and packed as Fp8Experts.packGateUp / packOne lay it out, its scales widened to f32.
pub fn packExpert(a: std.mem.Allocator, store: anytype, prefix: []const u8, i: usize, dims: usize, width: usize, out: Expert) !void {
    defer store.release();
    const one = try Shape.of(dims, width);
    if (out.up.len != one.up or out.down.len != one.down or out.up_scale.len != one.up_scale or out.down_scale.len != one.down_scale) return error.UnexpectedTensor;
    const half = one.up_scale / 2;
    const gate = try part(store, prefix, i, "gate_proj", width, dims, out.up_scale[0..half]);
    const up = try part(store, prefix, i, "up_proj", width, dims, out.up_scale[half..]);
    try fx.packGateUp(a, out.up, gate, up, width, dims);
    fx.packOne(out.down, try part(store, prefix, i, "down_proj", dims, width, out.down_scale), dims, width);
}

/// One projection's E4M3 weight [n, k]; its BF16 weight_scale_inv [n/128, k/128] widened (exactly) into `scale`.
fn part(store: anytype, prefix: []const u8, i: usize, name: []const u8, n: usize, k: usize, scale: []f32) ![]const u8 {
    var buf: [512]u8 = undefined;
    const w = try store.get(try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}.weight", .{ prefix, i, name }));
    if (!w.is(.f8_e4m3, &.{ n, k }) or w.bytes.len != n * k) return error.UnexpectedTensor;
    const s = try store.get(try std.fmt.bufPrint(&buf, "{s}.experts.{d}.{s}.weight_scale_inv", .{ prefix, i, name }));
    if (!s.is(.bf16, &.{ n / blk, k / blk }) or s.bytes.len != 2 * scale.len) return error.UnexpectedTensor;
    for (scale, 0..) |*v, j| {
        v.* = wt.promote(std.mem.readInt(u16, s.bytes[2 * j ..][0..2], .little));
        if (!std.math.isFinite(v.*)) return error.UnsupportedQuantization;
    }
    return w.bytes;
}

test {
    _ = @import("cuda_fp8_experts_load_test.zig");
}
