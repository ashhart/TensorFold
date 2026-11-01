//! Idol's products on cuBLASLt: fp16 or bf16 linears into fp32 sums, true fp32 products (no TF32), a plan a shape.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;

pub const workspace_bytes = 32 << 20;

/// The operands' type: fp16 block linears, fp32 RePo maps and condition/output projections, bf16 DINOv3 linears.
pub const In = enum { f16, f32, bf16 };

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([4]u64, lt_.Linear) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y fp32 [rows, n] (rows `n` apart) = x [rows, k] . w [n, k]^T with fp32 compute and sums, on `stream`.
    pub fn run(g: *Gemms, s: cuda.Stream, in: In, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0) return;
        const p = try g.plan(in, rows, n, k);
        try p.run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
    }

    fn plan(g: *Gemms, in: In, rows: usize, n: usize, k: usize) !*const lt_.Linear {
        const got = try g.plans.getOrPut(g.gpa, .{ @backingInt(in), rows, n, k });
        if (got.found_existing) return got.value_ptr;
        const t: lt_.DataType = switch (in) {
            .f16 => .f16,
            .f32 => .f32,
            .bf16 => .bf16,
        };
        got.value_ptr.* = lt_.Linear.initTyped(&g.lt, rows, n, k, t, .f32, workspace_bytes, .none) catch |e| {
            g.plans.removeByPtr(got.key_ptr);
            return e;
        };
        return got.value_ptr;
    }
};

test "cuBLASLt's fp32 compute type is CUBLAS_COMPUTE_32F, not a TF32 or fast mode" {
    // cublasComputeType_t: 68 is CUBLAS_COMPUTE_32F; 74 and 77 are its FAST_TF32 and FAST_16F forms
    try std.testing.expectEqual(@as(c_int, 68), lt_.compute_32f);
    try std.testing.expectEqual(@as(c_int, 2), @backingInt(lt_.DataType.f16));
    try std.testing.expectEqual(@as(c_int, 0), @backingInt(lt_.DataType.f32));
    try std.testing.expectEqual(@as(c_int, 14), @backingInt(lt_.DataType.bf16));
}
