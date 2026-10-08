//! Nemotron-H prefill (windows above 16 rows) on the matrix engine, weights decoded inside the GEMM (nem_pf.cl).
const std = @import("std");
const rt = @import("xpu").rt;
const model = @import("xpu_model.zig");

const Buf = rt.Buffer;
const spv = @import("xpu").kernels.nem_pf;
const top_k: u32 = 6;

/// Per-kernel time (nem_win prof_on): every launch synced and timed.
pub var prof_on: bool = false;
var prof: [24]struct { h: ?*const anyopaque = null, ns: u64 = 0, calls: u64 = 0 } = @splat(.{});

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn printProf(mp: *MoePf) void {
    inline for (@typeInfo(MoePf).@"struct".field_names) |fname| {
        if (@TypeOf(@field(mp.*, fname)) == rt.Kernel) {
            const h: ?*const anyopaque = @ptrCast(@field(mp.*, fname).handle);
            for (prof) |e| if (e.h != null and e.h == h) std.debug.print("  kernel {s:<10} {d:>6} calls {d:>8.1} us avg {d:>8.2} ms a window\n", .{ fname, e.calls, @as(f64, @floatFromInt(e.ns)) / 1e3 / @as(f64, @floatFromInt(e.calls)), @as(f64, @floatFromInt(e.ns)) / 1e6 });
        }
    }
    prof = @splat(.{});
}

fn run(k: *rt.Kernel, groups: [3]u32, a: anytype) !void {
    const t0: u64 = if (prof_on) blk: {
        k.rt.sync() catch {};
        break :blk nowNs();
    } else 0;
    defer if (prof_on) {
        k.rt.sync() catch {};
        const dt = nowNs() - t0;
        for (&prof) |*e| {
            if (e.h == null) e.h = @ptrCast(k.handle);
            if (e.h == @as(?*const anyopaque, @ptrCast(k.handle))) {
                e.ns += dt;
                e.calls += 1;
                break;
            }
        }
    };
    inline for (a, 0..) |v, i| {
        const T = @TypeOf(v);
        if (T == Buf) try k.setBuffer(i, v) else if (T == f32) try k.setF32(i, v) else try k.setU32(i, @as(u32, v));
    }
    try k.launch(groups);
}

pub const MoePf = struct {
    sort: rt.Kernel,
    prep: rt.Kernel, // dense activation prep
    prepg: rt.Kernel, // gathered rows
    dgemm: rt.Kernel,
    gemm: rt.Kernel, // grouped
    dgs: rt.Kernel, // dense split-K
    fin: rt.Kernel,
    mgs: rt.Kernel, // grouped split-K
    mfin: rt.Kernel,
    relu2: rt.Kernel,
    cnt: Buf,
    segoff: Buf,
    row_slot: Buf,
    meta: Buf,
    xt: Buf, // gathered rows of the experts
    dxt: ?Buf = null, // dense activations
    elist: Buf,
    z: Buf, // split-K partials of the dense GEMMs
    zm: Buf, // of the grouped GEMM
    xs: Buf, // activation tile of a small window

    pub fn init(r: *rt.Runtime, rcap: u32, n_experts: u32, max_k: u32) !MoePf {
        var m = try r.moduleWith(spv, "-cl-intel-256-GRF-per-thread");
        const padded: usize = @as(usize, rcap) * top_k + @as(usize, n_experts) * 64;
        return .{
            .sort = try m.kernel("moe_sort", .{ n_experts * @max(1, @min(8, 1024 / n_experts)), 1, 1 }), // slices of the slots a work-group of up to 1024 items
            .prep = try m.kernel("pf_prep_b", .{ 64, 1, 1 }),
            .prepg = try m.kernel("pf_prep_gb", .{ 64, 1, 1 }),
            .dgemm = try m.kernel("pfgemm_b", .{ 16, 1, 1 }),
            .gemm = try m.kernel("pfgemm_moe_b", .{ 16, 1, 1 }),
            .dgs = try m.kernel("pfgemm_bs", .{ 16, 1, 1 }),
            .fin = try m.kernel("fin_bs", .{ 64, 1, 1 }),
            .mgs = try m.kernel("pfgemm_moe_bs", .{ 16, 1, 1 }),
            .mfin = try m.kernel("fin_moe", .{ 64, 1, 1 }),
            .relu2 = try m.kernel("relu2_bf16", .{ 64, 1, 1 }),
            .cnt = try r.alloc(n_experts * 4),
            .segoff = try r.alloc(n_experts * 4),
            .row_slot = try r.alloc(padded * 4),
            .meta = try r.alloc(16),
            .elist = try r.alloc(260 * 4),
            .z = try r.alloc(32 << 20),
            .zm = try r.alloc(32 << 20),
            .xs = try r.alloc(16 * @as(usize, @max(max_k, 17408)) * 2),
            .xt = try r.alloc(padded * max_k * 2),
        };
    }

    /// A dense projection of n rows: y [n][rows] (bf16, or fp32 when f32out) at y_off; rows % 64 == 0, in % 64 == 0.
    pub fn dense(self: *MoePf, r: *rt.Runtime, w: Buf, s: Buf, b: Buf, x: Buf, n: u32, y: Buf, in: u32, y_off: u32, rows: u32, f32out: bool) !void {
        const rgs = (n + 63) / 64 * 8;
        const need = @as(usize, rgs) * 8 * in * 2;
        if (self.dxt == null or self.dxt.?.len < need) {
            if (self.dxt) |*old| old.free();
            self.dxt = try r.alloc(need);
        }
        const xt = self.dxt.?;
        try run(&self.prep, .{ (in / 16 + 63) / 64, rgs, 1 }, .{ x, xt, n, in });
        try run(&self.dgemm, .{ (n + 15) / 16, rows / 64, 1 }, .{ xt, w, s, b, y, n, in, rows, y_off, @as(u32, @intFromBool(f32out)) });
    }

    /// Routed experts of n rows: xn [n][h] bf16, eids [n][6]; fills ey [n * 6][h] fp32 (slot = row * 6 + k).
    pub fn runExperts(self: *MoePf, m: *model.Model, mw: model.Moe, xn: Buf, eids: Buf, act: Buf, ey: Buf, n: u32) !void {
        const c = m.cfg;
        const h = c.hidden_size;
        const wd = c.moe_intermediate_size;
        const E = c.n_routed_experts;
        try run(&self.sort, .{ 1, 1, 1 }, .{ eids, self.cnt, self.segoff, self.row_slot, self.meta, self.elist, n * top_k, E, @as(u32, 16) });
        var meta: [2]u32 = undefined;
        try m.r.download(std.mem.sliceAsBytes(&meta), self.meta);
        try m.r.sync();
        const ptotal = meta[0];
        const panels = (meta[1] + 15) / 16;
        try run(&self.prepg, .{ (h / 16 + 63) / 64, ptotal / 8, 1 }, .{ xn, self.xt, self.row_slot, h, top_k });
        try run(&self.gemm, .{ panels, wd / 64, E }, .{ self.xt, mw.fc1.w, mw.fc1.s, mw.fc1.b, act, ey, self.cnt, self.segoff, self.row_slot, h, wd, @as(u32, 0) });
        try run(&self.prepg, .{ (wd / 16 + 63) / 64, ptotal / 8, 1 }, .{ act, self.xt, self.row_slot, wd, @as(u32, 1) });
        try run(&self.gemm, .{ panels, h / 64, E }, .{ self.xt, mw.fc2.w, mw.fc2.s, mw.fc2.b, act, ey, self.cnt, self.segoff, self.row_slot, wd, h, @as(u32, 1) });
    }

    /// K ranges of a split-K GEMM over `groups` 64-input groups: divisor (max 32) nearest ~1200 sub-groups; shape-only.
    pub fn splitFor(groups: u32, ncb: u32, active: u32) u32 {
        var best: u32 = 1;
        var score: f64 = 1e9;
        var s: u32 = 1;
        while (s <= 32 and s <= groups) : (s += 1) {
            if (groups % s != 0) continue;
            const d = @abs(@log(@as(f64, @floatFromInt(ncb * s * active)) / 1200.0));
            if (d < score) {
                score = d;
                best = s;
            }
        }
        return best;
    }

    /// Dense projection of n <= 16 rows (decode n = 1): split-K matrix-engine GEMM + finish; y [n][rows] bf16 or fp32.
    pub fn denseS(self: *MoePf, w: Buf, s: Buf, b: Buf, x: Buf, n: u32, y: Buf, in: u32, y_off: u32, rows: u32, f32out: bool) !void {
        return self.denseSM(w, s, b, x, n, y, in, y_off, rows, @intFromBool(f32out), true);
    }

    /// denseS with finish mode (0 bf16, 1 fp32, 2 bf16 of relu2); `prep` false reuses the last activation tile.
    pub fn denseSM(self: *MoePf, w: Buf, s: Buf, b: Buf, x: Buf, n: u32, y: Buf, in: u32, y_off: u32, rows: u32, mode: u32, prep: bool) !void {
        const S = splitFor(in / 64, rows / 64, 1);
        if (@as(usize, S) * 16 * rows * 4 > self.z.len) return error.Invalid;
        if (prep) try run(&self.prep, .{ (in / 16 + 63) / 64, 2, 1 }, .{ x, self.xs, n, in });
        try run(&self.dgs, .{ rows / 64, S, 1 }, .{ self.xs, w, s, b, self.z, in, rows, S });
        try run(&self.fin, .{ (rows + 63) / 64, n, 1 }, .{ self.z, y, n, rows, S, y_off, mode });
    }

    /// Routed experts of n <= 16 rows, no host round trip: sort, gather, split-K grouped GEMM, finish.
    pub fn runExpertsS(self: *MoePf, m: *model.Model, mw: model.Moe, xn: Buf, eids: Buf, act: Buf, ey: Buf, n: u32) !void {
        const c = m.cfg;
        const h = c.hidden_size;
        const wd = c.moe_intermediate_size;
        const E = c.n_routed_experts;
        const slots = n * top_k;
        const ng = @min(E, slots);
        try run(&self.sort, .{ 1, 1, 1 }, .{ eids, self.cnt, self.segoff, self.row_slot, self.meta, self.elist, slots, E, @as(u32, 16) });
        const rgs = ng * 16 / 8; // at most this many padded row groups (every chosen expert holds at least one row, segments of 16)
        const s1 = splitFor(h / 64, wd / 64, top_k);
        try run(&self.prepg, .{ (h / 16 + 63) / 64, rgs, 1 }, .{ xn, self.xt, self.row_slot, h, top_k });
        try run(&self.mgs, .{ 1, wd / 64, ng * s1 }, .{ self.xt, mw.fc1.w, mw.fc1.s, mw.fc1.b, self.zm, self.cnt, self.segoff, self.row_slot, self.elist, h, wd, s1, slots });
        try run(&self.mfin, .{ (wd + 63) / 64, slots, 1 }, .{ self.zm, act, ey, wd, s1, slots, @as(u32, 0) });
        const s2 = splitFor(wd / 64, h / 64, top_k);
        try run(&self.prepg, .{ (wd / 16 + 63) / 64, rgs, 1 }, .{ act, self.xt, self.row_slot, wd, @as(u32, 1) });
        try run(&self.mgs, .{ 1, h / 64, ng * s2 }, .{ self.xt, mw.fc2.w, mw.fc2.s, mw.fc2.b, self.zm, self.cnt, self.segoff, self.row_slot, self.elist, wd, h, s2, slots });
        try run(&self.mfin, .{ (h + 63) / 64, slots, 1 }, .{ self.zm, act, ey, h, s2, slots, @as(u32, 1) });
    }
};
