//! Diagnostic CPU weights are independently decoded to F32 once, then multiplied with binary64 BLAS.
const std = @import("std");
const core = @import("core");
const adapter = @import("gguf.zig");
const cblas_dgemm = struct {
    extern "c" fn cblas_dgemm(layout: c_int, ta: c_int, tb: c_int, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void;
}.cblas_dgemm;
pub fn rounded(v: f64) f64 {
    const value: f32 = @floatCast(v);
    const bits: u32 = @bitCast(value);
    const raw = if ((bits & 0x7fffffff) > 0x7f800000) (bits >> 16) | 0x40 else (bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16;
    return @as(f32, @bitCast(raw << 16));
}
pub fn round(values: []f64) void {
    for (values) |*v| v.* = rounded(v.*);
}
pub const Matrix = struct { values: []f32, n: usize, k: usize };
pub const Weights = struct {
    file: core.gguf.File,
    arena: std.heap.ArenaAllocator,
    config: @import("config.zig").Config,
    names: []adapter.names.Name,
    matrices: std.StringHashMapUnmanaged(Matrix) = .empty,
    bytes: usize = 0,
    pub fn init(a: std.mem.Allocator, io: std.Io, path: []const u8, cache_limit: usize) !Weights {
        var file = try core.gguf.File.open(a, io, path, .{});
        errdefer file.close(io);
        var p: adapter.Problem = .{};
        try @import("gguf_projection.zig").admit(&file.document, &p);
        const c = try adapter.metadata.parse(&file.document, &p);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const mem = arena.allocator();
        const plan = try adapter.names.plan(mem, c);
        var total: usize = 0;
        for (plan) |name| if (name.role == .matrix) {
            total = try std.math.add(usize, total, try std.math.mul(usize, name.dims[0] * name.dims[1], 4));
        };
        if (total > cache_limit) return error.CpuReferenceCacheLimit;
        var result: Weights = .{ .file = file, .arena = arena, .config = c, .names = plan, .bytes = total };
        for (plan) |name| {
            if (name.role != .matrix) continue;
            const t = file.document.getTensor(name.gguf) orelse return error.MissingGgufWeight;
            const count = name.dims[0] * name.dims[1];
            const values = try mem.alloc(f32, count);
            const geometry = core.gguf_repack_ref.geometry(t.kind);
            for (values, 0..) |*v, i| {
                const block = t.bytes[i / geometry.values * geometry.bytes ..][0..geometry.bytes];
                v.* = @floatCast(try core.gguf_repack_ref.value(t.kind, block, i % geometry.values));
            }
            try result.matrices.put(mem, name.native, .{ .values = values, .n = name.dims[0], .k = name.dims[1] });
            std.debug.print("CPU_CACHE {s} elements={d}\n", .{ name.gguf, count });
        }
        result.arena = arena;
        return result;
    }
    pub fn deinit(w: *Weights, io: std.Io) void {
        w.arena.deinit();
        w.file.close(io);
    }
    pub fn vector(w: *const Weights, name: []const u8) ![]const u8 {
        const named = for (w.names) |n| {
            if (std.mem.eql(u8, n.native, name)) break n;
        } else return error.MissingGgufWeight;
        const tensor = w.file.document.getTensor(named.gguf).?;
        if (tensor.kind != .f32) return error.InvalidGgufScalar;
        return tensor.bytes;
    }
    pub fn matrix(w: *const Weights, name: []const u8) !Matrix {
        return w.matrices.get(name) orelse error.MissingGgufWeight;
    }
    /// Row-major binary64 X times independently decoded W transpose, tiled only to bound temporary memory.
    pub fn project(w: *const Weights, a: std.mem.Allocator, name: []const u8, input: []const f64, rows: usize) ![]f64 {
        const m = try w.matrix(name);
        if (input.len != rows * m.k or rows == 0) return error.InvalidCpuProjection;
        const out = try a.alloc(f64, rows * m.n);
        const tile = @min(m.n, 256);
        const decoded = try a.alloc(f64, tile * m.k);
        defer a.free(decoded);
        var first: usize = 0;
        while (first < m.n) : (first += tile) {
            const count = @min(tile, m.n - first);
            for (decoded[0 .. count * m.k], m.values[first * m.k ..][0 .. count * m.k]) |*dst, src| dst.* = src;
            cblas_dgemm(101, 111, 112, @intCast(rows), @intCast(count), @intCast(m.k), 1, input.ptr, @intCast(m.k), decoded.ptr, @intCast(m.k), 0, out.ptr + first, @intCast(m.n));
        }
        round(out);
        return out;
    }
};
test "Qwen27 GGUF CPU: BF16 checkpoints round ties to even and preserve non-finite classes" {
    const t = std.testing;
    try t.expectEqual(@as(f64, 1), rounded(1.00390625));
    try t.expectEqual(@as(f64, 1.015625), rounded(1.01171875));
    try t.expect(std.math.isNan(rounded(std.math.nan(f64))));
    try t.expect(std.math.isInf(rounded(std.math.inf(f64))));
}

pub fn scalar(bytes: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little)));
}
