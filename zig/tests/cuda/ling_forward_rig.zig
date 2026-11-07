//! What the Ling forward checks share: the graph's kit on one stream, runs that return logits, distances and flips.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");

pub const F = ling.forward;
pub const W = ling.weights;

pub fn bfOf(u: u16) f64 {
    return @as(f32, @bitCast(@as(u32, u) << 16));
}

pub const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    m: *const F.Modules,
    g: *const F.Graph,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    kda_wide: bool = true, // decode windows on KDA's three-kernel chain (false: the fused one)

    pub fn deinit(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
    }

    pub fn zeros(r: *Rig, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(len, 16));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0, null);
        return b.ptr;
    }

    pub fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
        const out = try r.a.alloc(u8, len);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, out);
        return out;
    }

    pub fn fwd(r: *const Rig, b: *const F.Buffers, picks: ?u64) F.Forward {
        return .{ .m = r.m, .g = r.g, .s = r.s, .b = b, .picks = picks, .kda_wide = r.kda_wide };
    }

    /// One forward of `segs` on `b`; its logits rows (bf16 bits) and, with `picks`, every MoE layer's picks.
    pub fn run(r: *Rig, b: *const F.Buffers, segs: []const F.Seg, ids: []const i32, picks: ?u64) ![]u16 {
        try r.fwd(b, picks).run(segs, ids);
        const v: usize = r.g.w.config.vocab;
        const raw = try r.back(b.logits, ids.len * v * 2);
        const out = try r.a.alloc(u16, ids.len * v);
        @memcpy(std.mem.sliceAsBytes(out), raw);
        return out;
    }

    /// A fresh sequence of `capacity` positions.
    pub fn state(r: *Rig, capacity: usize) !*F.State {
        const st = try r.a.create(F.State);
        st.* = try F.State.init(r.gpu.gpa, r.gpu.d, r.g, capacity);
        try st.reset(r.s);
        return st;
    }
};

/// Picks [layers][rows][top_k + 1] (i32) as each row's top-k sorted, [layers][rows][top_k].
pub fn sortedPicks(a: std.mem.Allocator, raw: []const u8, layers: usize, rows: usize, top: usize) ![]i32 {
    const all = std.mem.bytesAsSlice(i32, raw);
    const out = try a.alloc(i32, layers * rows * top);
    for (0..layers * rows) |i| {
        const dst = out[i * top ..][0..top];
        for (dst, 0..) |*d, k| d.* = all[i * (top + 1) + k];
        std.mem.sort(i32, dst, {}, std.sort.asc(i32));
    }
    return out;
}

/// Rows whose sorted picks differ from the reference's in any layer (`want` rows from `w0` of its `wrows`).
pub fn flips(got: []const i32, want: []const i32, layers: usize, rows: usize, w0: usize, wrows: usize, top: usize, out: []bool) usize {
    var n: usize = 0;
    for (0..layers) |l| for (0..rows) |t| {
        const g = got[(l * rows + t) * top ..][0..top];
        const e = want[(l * wrows + w0 + t) * top ..][0..top];
        if (!std.mem.eql(i32, g, e)) {
            n += 1;
            out[t] = true;
        }
    };
    return n;
}

/// bf16 rows against float64 rows: relative L2, the same excluding `skip` rows, the worst row's, argmax agreement.
pub const Dist = struct { rel: f64, rel_kept: f64, worst_row: f64, argmax: usize };

pub fn distance(got: []const u16, want: []const f64, rows: usize, width: usize, skip: []const bool) Dist {
    var dd: f64 = 0;
    var ww: f64 = 0;
    var kd: f64 = 0;
    var kw: f64 = 0;
    var worst: f64 = 0;
    var agree: usize = 0;
    for (0..rows) |t| {
        var rd: f64 = 0;
        var rw: f64 = 0;
        var ga: usize = 0;
        var wa: usize = 0;
        for (0..width) |j| {
            const g = bfOf(got[t * width + j]);
            const e = want[t * width + j];
            rd += (g - e) * (g - e);
            rw += e * e;
            if (g > bfOf(got[t * width + ga])) ga = j;
            if (e > want[t * width + wa]) wa = j;
        }
        dd += rd;
        ww += rw;
        if (!skip[t]) {
            kd += rd;
            kw += rw;
        }
        worst = @max(worst, @sqrt(rd / rw));
        agree += @intFromBool(ga == wa);
    }
    return .{ .rel = @sqrt(dd / ww), .rel_kept = if (kw > 0) @sqrt(kd / kw) else 0, .worst_row = worst, .argmax = agree };
}

/// Two runs' logits rows equal, bit for bit.
pub fn same(what: []const u8, got: []const u16, want: []const u16) !void {
    try check.sameBytes(what, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
}
