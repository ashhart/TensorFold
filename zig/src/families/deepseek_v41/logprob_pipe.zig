//! Raw head columns gathered in vocabulary order; reply logprobs use the full, untempered target distribution.
const std = @import("std");
const tp = @import("tp");
const lanes = @import("lanes");
const partition = @import("partition.zig");

pub const Device = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    pub const VTable = struct {
        scratch: *const fn (*anyopaque, u32, usize) anyerror!u64,
        pad: *const fn (*anyopaque, u64, usize, usize, u64) anyerror!void,
        download: *const fn (*anyopaque, u64, []u8) anyerror!void,
        stream: *const fn (*anyopaque) tp.collective.Stream,
    };
};

pub const Capture = struct {
    values: std.ArrayList(f32) = .empty,
    padded: std.ArrayList(f32) = .empty,
    vocab: usize = 0,

    pub fn deinit(c: *Capture, a: std.mem.Allocator) void {
        c.values.deinit(a);
        c.padded.deinit(a);
    }

    /// All ranks call alike before a sampler or grammar can alter the logits; only rank 0 retains host rows.
    pub fn gather(c: *Capture, a: std.mem.Allocator, comm: tp.collective.Collective, d: Device, lg: u64, n: usize, vocab: u32) !void {
        const world = comm.world();
        const own = try partition.share(vocab, comm.rank(), world);
        const stride = (try partition.share(vocab, 0, world)).len;
        const vt = d.vtable;
        c.vocab = vocab;
        c.values.clearRetainingCapacity();
        if (comm.rank() == 0) {
            try c.values.resize(a, n * vocab);
            try c.padded.resize(a, world * stride);
        }
        const recv = try vt.scratch(d.ptr, 0, 4 * world * stride);
        const send = if (own.len < stride) try vt.scratch(d.ptr, 1, 4 * stride) else 0;
        for (0..n) |r| {
            const source = lg + 4 * r * own.len;
            if (own.len < stride) try vt.pad(d.ptr, source, own.len, stride, send);
            try comm.allGather(if (own.len < stride) send else source, recv, stride, .f32, vt.stream(d.ptr));
            if (comm.rank() != 0) continue;
            try vt.download(d.ptr, recv, std.mem.sliceAsBytes(c.padded.items));
            try comm.check();
            for (0..world) |rank| {
                const part = try partition.share(vocab, rank, world);
                @memcpy(c.values.items[r * vocab + part.lo ..][0..part.len], c.padded.items[rank * stride ..][0..part.len]);
            }
        }
    }

    pub fn row(c: *const Capture, r: usize, token: u32, k: u8) !lanes.logprob.Row {
        if (c.vocab == 0 or r >= c.values.items.len / c.vocab) return error.LogprobRow;
        return score(c.values.items[r * c.vocab ..][0..c.vocab], token, k);
    }
};

/// Stable float64 normalization of the actual FP32 head values, rounded only at the public Row boundary.
pub fn score(values: []const f32, token: u32, k: u8) !lanes.logprob.Row {
    if (values.len == 0 or token >= values.len or k > lanes.logprob.max_top) return error.LogprobRow;
    var maximum: f64 = -std.math.inf(f64);
    for (values) |v| {
        if (std.math.isNan(v) or v == std.math.inf(f32)) return error.NonFiniteLogit;
        maximum = @max(maximum, v);
    }
    if (!std.math.isFinite(maximum)) return error.NoFiniteLogit;
    var sum: f64 = 0;
    for (values) |v| sum += @exp(@as(f64, v) - maximum);
    const mass = @log(sum);
    var out: lanes.logprob.Row = .{ .token = token, .logprob = @floatCast((@as(f64, values[token]) - maximum) - mass) };
    const want = @min(@as(usize, k), values.len);
    for (values, 0..) |v, id| {
        out.rank += @intFromBool(v >= values[token]);
        var at: usize = 0;
        while (at < out.count and values[out.ids[at]] >= v) : (at += 1) {}
        if (at >= want) continue;
        const end = @min(@as(usize, out.count), want - 1);
        var j = end;
        while (j > at) : (j -= 1) out.ids[j] = out.ids[j - 1];
        out.ids[at] = @intCast(id);
        out.count = @intCast(@min(@as(usize, out.count) + 1, want));
    }
    for (out.ids[0..out.count], out.logprobs[0..out.count]) |id, *lp|
        lp.* = @floatCast((@as(f64, values[id]) - maximum) - mass);
    return out;
}
