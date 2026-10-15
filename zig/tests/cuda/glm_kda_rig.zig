//! glm-kda-ling's rig: a call's inputs on the device, one path's outputs, downloads to f64.
const std = @import("std");
const cuda = @import("cuda");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const f64r = @import("glm_kda_f64.zig");
pub const Rig = qr.RigT;

pub const kda = glm.kda;
pub const dk = kda.dk;

/// glm_kda.cu's module (chain, replay, wide).
pub const Mods = struct {
    kda_mod: cuda.Module,
    ks: kda.Kernels,

    pub fn load(d: *const cuda.Driver) !Mods {
        var m: Mods = undefined;
        m.kda_mod = try cuda.Module.load(d, cuda.kernels.glm_kda);
        m.ks = try kda.Kernels.load(m.kda_mod);
        return m;
    }

    pub fn unload(m: *Mods) void {
        m.kda_mod.unload();
    }
};

/// How a call runs: the fused chain or the three-kernel (serial wide) chain.
pub const Path = enum { fused, wide };

/// One path's buffers for `rows` rows: out, state, saves (k, v, decay, beta), wide scratch (q, y).
pub const Out = struct {
    out: u64,
    state: u64,
    k: u64,
    v: u64,
    g: u64,
    b: u64,
    q: u64,
    y: u64,
    rows: usize,

    pub fn alloc(r: *Rig, heads: usize, rows: usize) !Out {
        const n = heads * dk;
        return .{ .out = try r.zeros(rows * n * 2), .state = try r.zeros(n * dk * 4), .k = try r.zeros(rows * n * 4), .v = try r.zeros(rows * n * 2), .g = try r.zeros(rows * n * 4), .b = try r.zeros(rows * heads * 4), .q = try r.zeros(rows * n * 4), .y = try r.zeros(rows * n * 2), .rows = rows };
    }

    pub fn saves(o: Out) kda.Saves {
        return .{ .out = o.out, .state_out = o.state, .k = o.k, .v = o.v, .g = o.g, .b = o.b };
    }

    /// The same buffers from row `lo` on (state unchanged).
    pub fn from(o: Out, heads: usize, lo: usize) Out {
        const n = heads * dk;
        return .{ .out = o.out + lo * n * 2, .state = o.state, .k = o.k + lo * n * 4, .v = o.v + lo * n * 2, .g = o.g + lo * n * 4, .b = o.b + lo * heads * 4, .q = o.q + lo * n * 4, .y = o.y + lo * n * 2, .rows = o.rows - lo };
    }
};

/// `rows` rows of c through `path` into o.
pub fn run(m: *const Mods, s: cuda.Stream, c: kda.Chain, rows: usize, o: Out, path: Path) !void {
    const w: kda.Wide = .{ .q = o.q, .y = o.y };
    switch (path) {
        .fused => try kda.chain(&m.ks, s, c, rows, o.saves(), null),
        .wide => try kda.chain(&m.ks, s, c, rows, o.saves(), w),
    }
}

pub fn floats(r: *Rig, ptr: u64, count: usize) ![]f32 {
    const raw = try r.back(ptr, count * 4);
    const out = try r.a.alloc(f32, count);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

pub fn halves(r: *Rig, ptr: u64, count: usize) ![]u16 {
    const raw = try r.back(ptr, count * 2);
    const out = try r.a.alloc(u16, count);
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

/// fp32 or bf16 device values as f64.
pub fn wide64(r: *Rig, ptr: u64, count: usize, comptime bf16: bool) ![]f64 {
    const out = try r.a.alloc(f64, count);
    if (bf16) {
        for (out, try halves(r, ptr, count)) |*d, x| d.* = f64r.bf(x);
    } else for (out, try floats(r, ptr, count)) |*d, x| {
        d.* = x;
    }
    return out;
}

/// prep's saves of o, downloaded.
pub fn saved(r: *Rig, o: Out, heads: usize) !f64r.Saved {
    const n = o.rows * heads * dk;
    return .{ .q = try floats(r, o.q, n), .k = try floats(r, o.k, n), .v = try halves(r, o.v, n), .g = try floats(r, o.g, n), .b = try floats(r, o.b, o.rows * heads) };
}

/// Largest |a - b| over the largest |b|, and how many elements differ at all.
pub fn diff(a: []const f64, b: []const f64) struct { rel: f64, count: usize } {
    var most: f64 = 0;
    var big: f64 = 0;
    var count: usize = 0;
    for (a, b) |x, y| {
        most = @max(most, @abs(x - y));
        big = @max(big, @abs(y));
        if (x != y) count += 1;
    }
    return .{ .rel = most / big, .count = count };
}
