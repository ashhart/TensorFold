//! EXLR8 matrices packed for the GPU: weight bytes, fp16 scales, signs, tile-run offsets and jobs, a node's slice each.

const std = @import("std");
const dense = @import("exlr8_dense.zig");
const codec = @import("exlr8_codec.zig");
const ex = @import("exlr8_experts.zig");
const job = @import("exlr8_job.zig");
const Job = job.Job;
const DenseJob = job.DenseJob;

/// Rows [lo, hi) of a list the pack holds.
pub const Range = struct { lo: u32, hi: u32 };

/// A trellis matrix as kernels run it: rotate the slice's inputs, then `jobs` (128 outputs each) on the rotated rows.
pub const Trellis = struct {
    jobs: Range,
    /// The rotation's sign table (shared rotation) or per-channel scales (own rotation), as offsets.
    sign: u32,
    scale: u32,
    in: u32,
    out: u32,
};

/// A row-stored matrix as one dense job.
pub const Rows = struct { job: u32, in: u32, out: u32 };

pub const Linear = union(enum) { rows: Rows, trellis: Trellis };

/// Where an expert half's bytes and side data went: its tiles, then its 128 channel scales and 48 block scales.
pub const Half = struct { at: u64, k: u8, chan: u32, block: u32 };

pub const Pack = struct {
    gpa: std.mem.Allocator,
    bytes: std.ArrayList(u8) = .empty,
    halves: std.ArrayList(f16) = .empty,
    signs: std.ArrayList(i8) = .empty,
    bases: std.ArrayList(u64) = .empty,
    dense_jobs: std.ArrayList(DenseJob) = .empty,
    jobs: std.ArrayList(Job) = .empty,

    pub fn init(gpa: std.mem.Allocator) Pack {
        return .{ .gpa = gpa };
    }

    pub fn deinit(p: *Pack) void {
        p.bytes.deinit(p.gpa);
        p.halves.deinit(p.gpa);
        p.signs.deinit(p.gpa);
        p.bases.deinit(p.gpa);
        p.dense_jobs.deinit(p.gpa);
        p.jobs.deinit(p.gpa);
    }

    fn put(p: *Pack, data: []const u8) !u64 {
        const at = std.mem.alignForward(usize, p.bytes.items.len, 64);
        try p.bytes.resize(p.gpa, at + data.len);
        @memcpy(p.bytes.items[at..][0..data.len], data);
        return at;
    }

    fn f16s(p: *Pack, t: anytype, lo: usize, n: usize) !u32 {
        const at: u32 = @intCast(p.halves.items.len);
        for (lo..lo + n) |i| try p.halves.append(p.gpa, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
        return at;
    }

    /// Matrix `m` restricted to outputs `o` and inputs `i` (a 128-aligned slice of the HF weight (out, in)).
    pub fn linear(p: *Pack, m: dense.Matrix, o: [2]usize, i: [2]usize) !Linear {
        const r = m.record;
        if (o[1] > r.out or i[1] > r.in or o[0] >= o[1] or i[0] >= i[1] or i[0] % 128 != 0 or i[1] % 128 != 0) return error.BadExlr8Slice;
        if (r.format == .trellis) return .{ .trellis = try p.trellis(m, o, i) };
        return .{ .rows = try p.rowStored(m, o, i) };
    }

    fn rowStored(p: *Pack, m: dense.Matrix, o: [2]usize, i: [2]usize) !Rows {
        const r = m.record;
        const outs = o[1] - o[0];
        var j: DenseJob = .{ .format = try r.kernelFormat(), .out0 = 0, .outs = @intCast(outs), .in0 = 0, .ins = @intCast(i[1] - i[0]), .out_total = @intCast(outs), .in_total = @intCast(r.in), .w = 0 };
        const d0 = m.data[0];
        switch (r.format) {
            .int4 => if (r.side == .input) {
                // whole 128-input blocks of the [in/128, out, 64] layout: the job's matrix is the slice
                if (o[0] != 0 or o[1] != r.out) return error.BadExlr8Slice;
                j.in_total = j.ins;
                j.out_total = @intCast(r.out);
                j.w = try p.put(d0.bytes[i[0] / 128 * r.out * 64 ..][0 .. (i[1] - i[0]) / 128 * r.out * 64]);
                j.s = try p.put(m.data[1].bytes[i[0] / 128 * r.out * 8 ..][0 .. (i[1] - i[0]) / 128 * r.out * 8]);
            } else {
                j.in0 = @intCast(i[0]);
                j.w = try p.put(d0.bytes[o[0] * r.in / 2 ..][0 .. outs * r.in / 2]);
                j.s = try p.put(m.data[1].bytes[o[0] * r.in / 16 ..][0 .. outs * r.in / 16]);
            },
            .mxfp8 => {
                j.in0 = @intCast(i[0]);
                j.w = try p.put(d0.bytes[o[0] * r.in ..][0 .. outs * r.in]);
                j.s = try p.put(m.data[1].bytes[o[0] * r.in / 32 ..][0 .. outs * r.in / 32]);
            },
            .fp8_block => {
                if (o[0] % 128 != 0) return error.BadExlr8Slice;
                const cols = (r.in + 127) / 128;
                j.in0 = @intCast(i[0]);
                j.w = try p.put(d0.bytes[o[0] * r.in ..][0 .. outs * r.in]);
                j.s = try p.put(m.data[1].bytes[o[0] / 128 * cols * 4 ..][0 .. (outs + 127) / 128 * cols * 4]);
            },
            .fp8_row => {
                j.in0 = @intCast(i[0]);
                j.w = try p.put(d0.bytes[o[0] * r.in ..][0 .. outs * r.in]);
                j.s = try p.put(m.data[1].bytes[o[0] * 4 ..][0 .. outs * 4]);
            },
            .bf16, .fp32 => {
                const e: usize = if (r.format == .bf16) 2 else 4;
                j.in0 = @intCast(i[0]);
                j.w = try p.put(d0.bytes[o[0] * r.in * e ..][0 .. outs * r.in * e]);
            },
            .trellis => unreachable,
        }
        const at: u32 = @intCast(p.dense_jobs.items.len);
        try p.dense_jobs.append(p.gpa, j);
        return .{ .job = at, .in = j.ins, .out = j.outs };
    }

    fn trellis(p: *Pack, m: dense.Matrix, o: [2]usize, i: [2]usize) !Trellis {
        const r = m.record;
        if (o[0] % 128 != 0 or o[1] % 128 != 0) return error.BadExlr8Slice;
        const tb = 64 * codec.tileBytes(r.k);
        const entry = m.data[0].dim(1);
        const shared = r.rotation != 0;
        const t: Trellis = .{ .jobs = .{ .lo = @intCast(p.jobs.items.len), .hi = 0 }, .sign = @intCast(p.signs.items.len), .scale = @intCast(p.halves.items.len), .in = @intCast(i[1] - i[0]), .out = @intCast(o[1] - o[0]) };
        // the input side: a shared rotation's signs (|suh| is constant over each 128-block), else suh by channel
        var block_at: u32 = 0;
        if (shared) {
            for (i[0]..i[1]) |c| try p.signs.append(p.gpa, if (f16At(m.data[1], c) < 0) -1 else 1);
            block_at = @intCast(p.halves.items.len);
            for (0..(i[1] - i[0]) / 128) |b| {
                const mag = @abs(f16At(m.data[1], i[0] + 128 * b));
                for (0..128) |c| if (@abs(f16At(m.data[1], i[0] + 128 * b + c)) != mag) return error.SuhNotBlockConstant;
                try p.halves.append(p.gpa, @floatCast(mag));
            }
        } else _ = try p.f16s(m.data[1], i[0], i[1] - i[0]);
        const svh = try p.f16s(m.data[2], o[0], o[1] - o[0]);
        const in_blocks = (i[1] - i[0]) / 128;
        if (r.side == .input) {
            const first = try p.put(m.data[0].bytes[i[0] / 128 * entry ..][0 .. in_blocks * entry]);
            for (0..(o[1] - o[0]) / 128) |ob| {
                const base: u32 = @intCast(p.bases.items.len);
                for (0..in_blocks) |g| try p.bases.append(p.gpa, first + g * entry + (o[0] / 128 + ob) * tb);
                try p.jobs.append(p.gpa, .{ .k = r.k, .blocks = @intCast(in_blocks), .base = base, .in_scale = if (shared) block_at else job.none, .out_scale = svh + @as(u32, @intCast(ob * 128)), .y_col = @intCast(ob * 128) });
            }
        } else {
            const first = try p.put(m.data[0].bytes[o[0] / 128 * entry ..][0 .. (o[1] - o[0]) / 128 * entry]);
            for (0..(o[1] - o[0]) / 128) |g| {
                const base: u32 = @intCast(p.bases.items.len);
                for (0..in_blocks) |b| try p.bases.append(p.gpa, first + g * entry + (i[0] / 128 + b) * tb);
                try p.jobs.append(p.gpa, .{ .k = r.k, .blocks = @intCast(in_blocks), .base = base, .in_scale = if (shared) block_at else job.none, .out_scale = svh + @as(u32, @intCast(g * 128)), .y_col = @intCast(g * 128) });
            }
        }
        var out = t;
        out.jobs.hi = @intCast(p.jobs.items.len);
        if (shared) out.scale = job.none else out.sign = job.none;
        return out;
    }

    /// One expert half's tiles and side data; the layer's signs go in once through `layerSigns`.
    pub fn half(p: *Pack, h: ex.Half) !Half {
        const at = try p.put(h.trellis);
        const chan: u32 = @intCast(p.halves.items.len);
        for (h.chan) |c| try p.halves.append(p.gpa, c);
        const block: u32 = @intCast(p.halves.items.len);
        for (h.block) |b| try p.halves.append(p.gpa, b);
        return .{ .at = at, .k = h.k, .chan = chan, .block = block };
    }

    /// The layer's signs.in then signs.out (6144 each); returns where signs.in starts.
    pub fn layerSigns(p: *Pack, l: *const ex.Layer) !u32 {
        const at: u32 = @intCast(p.signs.items.len);
        try p.signs.appendSlice(p.gpa, l.signs_in);
        try p.signs.appendSlice(p.gpa, l.signs_out);
        return at;
    }

    /// A plain bf16/f32 vector's bytes (norms, biases), 64-byte aligned.
    pub fn vector(p: *Pack, bytes: []const u8) !u64 {
        return p.put(bytes);
    }
};

fn f16At(t: anytype, i: usize) f32 {
    return @as(f16, @bitCast(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little)));
}

test "a trellis slice's jobs cover its outputs and a row-stored slice keeps its rows" {
    const st = @import("safetensors.zig");
    const gpa = std.testing.allocator;
    var p = Pack.init(gpa);
    defer p.deinit();
    var rec: [64]u8 = undefined;
    for ([8]i64{ 3, 256, 256, 0, 1, 128, 0, 0 }, 0..) |v, k| std.mem.writeInt(i64, rec[8 * k ..][0..8], v, .little);
    var fp8: [256 * 256]u8 = undefined;
    for (&fp8, 0..) |*b, k| b.* = @truncate(k * 7);
    var e8: [256 * 8]u8 = @splat(127);
    const m: dense.Matrix = .{
        .record = try dense.Record.parse(.{ .dtype = .i64, .rank = 1, .shape = .{ 8, 1, 1, 1, 1 }, .bytes = &rec }),
        .data = .{ st.Tensor{ .dtype = .u8, .rank = 2, .shape = .{ 256, 256, 1, 1, 1 }, .bytes = &fp8 }, st.Tensor{ .dtype = .u8, .rank = 2, .shape = .{ 256, 8, 1, 1, 1 }, .bytes = &e8 }, undefined },
        .checksum = null,
    };
    const l = try p.linear(m, .{ 128, 256 }, .{ 0, 256 });
    const j = p.dense_jobs.items[l.rows.job];
    try std.testing.expectEqual(@as(u32, 128), j.outs);
    try std.testing.expectEqual(fp8[128 * 256], p.bytes.items[j.w]);
    try std.testing.expectError(error.BadExlr8Slice, p.linear(m, .{ 0, 256 }, .{ 64, 256 }));
}
