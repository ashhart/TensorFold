//! Tokens from logits rows on the device: argmax per greedy row, top_k + MARGIN candidates per sampled row.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const sample = @import("sample.zig");

/// Most candidates a row downloads; wider requests (top_k 0, or past this) download the row and draw on the host.
pub const max_candidates = 512;

/// One row's draw: its sampling (null is greedy) and the position that keys it.
pub const Request = struct { sampling: ?lanes.Sampling, position: u64 };

fn greedy(s: ?lanes.Sampling) bool {
    return if (s) |x| x.temperature <= 0.0 else true;
}

/// Candidates a sampled row asks for, or 0 when it takes the whole row.
fn candidates(s: lanes.Sampling, vocab: usize) usize {
    if (s.top_k == 0) return 0;
    const n = @min(vocab, @as(usize, s.top_k) + sample.MARGIN);
    return if (n > max_candidates) 0 else n;
}

fn view(comptime T: type, bytes: []u8) []T {
    return std.mem.bytesAsSlice(T, @as([]align(@alignOf(T)) u8, @alignCast(bytes)));
}

pub const Drawer = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    dtype: sample.Dtype,
    vocab: usize,
    rows: usize,
    /// Per row: the candidate count, the argmax, then (rows, stride) candidate ids and value patterns.
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    row: hip.HostBuffer,

    /// Device bytes a drawer of `rows` rows holds: each row's argmax and its sampling candidates.
    pub fn deviceBytes(rows: usize) usize {
        return rows * (8 + max_candidates * 6);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, dtype: sample.Dtype, vocab: usize, rows: usize) !Drawer {
        const bytes = deviceBytes(rows);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        var dev = try hip.DeviceBuffer.alloc(d, bytes);
        errdefer dev.free();
        return .{ .gpa = gpa, .d = d, .dtype = dtype, .vocab = vocab, .rows = rows, .host = host, .dev = dev, .row = try hip.HostBuffer.alloc(d, vocab * 2) };
    }

    pub fn deinit(w: *Drawer) void {
        w.row.free();
        w.dev.free();
        w.host.free();
    }

    /// Row r's offsets into the scratch of `stride` candidates a row, as byte counts: ks, argmax, ids, values.
    fn layout(w: *const Drawer, stride: usize) struct { arg: usize, ids: usize, vals: usize, total: usize } {
        const arg = w.rows * 4;
        const ids = arg + w.rows * 4;
        const vals = ids + w.rows * stride * 4;
        return .{ .arg = arg, .ids = ids, .vals = vals, .total = vals + w.rows * stride * 2 };
    }

    /// Where the argmax of a logits block's first rows goes on the device (a graph can hold the launch).
    pub fn argmaxAt(w: *const Drawer) u64 {
        return w.dev.base() + w.layout(0).arg;
    }

    /// `out[r]` the token of row r of `logits` per `reqs[r]`, stream synchronized; `argmaxed`: argmax is at `argmaxAt`.
    pub fn draw(w: *Drawer, o: hip.ops.Ops, stream: hip.Stream, logits: hip.ops.Tensor, reqs: []const Request, out: []u32, argmaxed: bool) !void {
        const rows = reqs.len;
        if (rows > w.rows) return error.WindowTooWide;
        const ks = w.host.slice(i32)[0..rows];
        var stride: usize = 0;
        var any_greedy = false;
        for (reqs, ks) |r, *k| {
            k.* = 0;
            if (greedy(r.sampling)) {
                any_greedy = true;
            } else {
                const n = candidates(r.sampling.?, w.vocab);
                k.* = @intCast(n);
                stride = @max(stride, n);
            }
        }
        const at = w.layout(stride);
        const base = w.dev.base();
        if (any_greedy and !argmaxed) try o.argmaxRows(logits, rows, w.vocab, base + at.arg);
        if (stride > 0) {
            try hip.raw.upload(w.dev, 0, std.mem.sliceAsBytes(ks), stream.handle);
            try o.topkRows(logits, rows, w.vocab, base, stride, base + at.ids, base + at.vals);
        }
        const host = w.host.bytes;
        if (any_greedy) try hip.raw.download(w.dev, at.arg, host[at.arg..][0 .. rows * 4], stream.handle);
        if (stride > 0) try hip.raw.download(w.dev, at.ids, host[at.ids..at.total], stream.handle);
        try stream.synchronize();
        const arg = view(i32, host[at.arg..][0 .. rows * 4]);
        for (reqs, 0..) |r, i| {
            if (greedy(r.sampling)) {
                out[i] = @intCast(arg[i]);
                continue;
            }
            const s = r.sampling.?;
            const n: usize = @intCast(ks[i]);
            if (n == 0) {
                const bytes = w.vocab * 2;
                try hip.runtime.check(w.d.api.hipMemcpyDtoHAsync(w.row.bytes.ptr, @ptrFromInt(logits.ptr + i * bytes), bytes, stream.handle));
                try stream.synchronize();
                out[i] = try sample.draw(w.gpa, w.row.slice(u16)[0..w.vocab], w.dtype, s, r.position);
                continue;
            }
            const ids = view(i32, host[at.ids + i * stride * 4 ..][0 .. n * 4]);
            const vals = view(u16, host[at.vals + i * stride * 2 ..][0 .. n * 2]);
            var values: [max_candidates]f64 = undefined;
            var wide: [max_candidates]u64 = undefined;
            for (0..n) |j| {
                values[j] = sample.widen(vals, w.dtype, j);
                wide[j] = @intCast(ids[j]);
            }
            out[i] = @intCast(try lanes.sampling.choose(w.gpa, values[0..n], wide[0..n], r.position, s));
        }
    }
};
