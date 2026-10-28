//! The glue tests' device buffers, host inputs and byte comparisons, on one stream.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const Gpu = check.Gpu;
const gm = nemotron.glue_math;
const glue = nemotron.glue;

pub const Rig = struct {
    gpu: Gpu,
    a: std.mem.Allocator, // host data, freed together at the end
    s: cuda.Stream,
    f: glue.Fns,
    emb: cuda.embed_rows.EmbedRows,
    norms: cuda.norms.Norms,
    moe: cuda.moe_glue.MoeGlue,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bad: usize = 0,
    cases: usize = 0,
    sparsest: f64 = 1, // the smallest share of nonzero elements in any reference: none may be all zeros

    pub fn g(r: *Rig) glue.Glue {
        return .{ .set = null, .f = &r.f, .s = r.s, .emb = &r.emb, .norms = &r.norms, .moe = &r.moe };
    }

    /// A device copy of host values, landed before any kernel on the rig's stream reads it.
    pub fn dev(r: *Rig, comptime T: type, host: []const T) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
        try r.bufs.append(r.gpu.gpa, b);
        try r.gpu.ctx.synchronize();
        return b.ptr;
    }

    pub fn zeros(r: *Rig, comptime T: type, n: usize) !u64 {
        const host = try r.a.alloc(T, n);
        @memset(std.mem.sliceAsBytes(host), 0);
        return r.dev(T, host);
    }

    pub fn bfs(r: *Rig, n: usize, scale: f32) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| v.* = gm.f32ToBf16((r.rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    /// bf16 values of magnitude 2^-8 to 2^8, log-uniform: rows whose sums' order shows in their last bits.
    pub fn wides(r: *Rig, n: usize) ![]u16 {
        const out = try r.a.alloc(u16, n);
        for (out) |*v| {
            const sign: f32 = if (r.rng.boolean()) 1 else -1;
            v.* = gm.f32ToBf16(sign * std.math.exp2(r.rng.float(f32) * 16 - 8));
        }
        return out;
    }

    pub fn f32s(r: *Rig, n: usize, lo: f32, hi: f32) ![]f32 {
        const out = try r.a.alloc(f32, n);
        for (out) |*v| v.* = lo + r.rng.float(f32) * (hi - lo);
        return out;
    }

    /// The device bytes at `ptr` against `want`, element for element; a difference names its first element.
    pub fn same(r: *Rig, what: []const u8, comptime T: type, ptr: u64, want: []const T) !void {
        try r.s.synchronize();
        const got = try r.a.alloc(T, want.len);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(got.ptr, ptr, want.len * @sizeOf(T)), "cuMemcpyDtoH");
        r.cases += 1;
        var live: usize = 0;
        for (want) |w| live += @intFromBool(!std.mem.allEqual(u8, std.mem.asBytes(&w), 0));
        r.sparsest = @min(r.sparsest, @as(f64, @floatFromInt(live)) / @as(f64, @floatFromInt(want.len)));
        if (std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) return;
        r.bad += 1;
        var n: usize = 0;
        var first: ?usize = null;
        for (got, want, 0..) |x, y, i| if (!std.mem.eql(u8, std.mem.asBytes(&x), std.mem.asBytes(&y))) {
            n += 1;
            if (first == null) first = i;
        };
        std.debug.print("DIFFER {s}: {d} of {d} elements, first at {d}: got {any}, want {any}\n", .{ what, n, want.len, first.?, got[first.?], want[first.?] });
    }

    /// n elements of the device bytes at `ptr`, once the rig's stream is done.
    pub fn get(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        try r.s.synchronize();
        const out = try r.a.alloc(T, n);
        try r.gpu.d.check(r.gpu.d.api.cuMemcpyDtoH_v2(out.ptr, ptr, n * @sizeOf(T)), "cuMemcpyDtoH");
        return out;
    }
};
