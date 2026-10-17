//! The H3 CUDA family's products on the runtime's cuBLASLt `Linear`: bf16 in and out, fp32 sums, one algorithm a shape.
const std = @import("std");
const cuda = @import("cuda");
const lt_ = cuda.cublaslt;

pub const workspace_bytes = 32 << 20;
/// The rows each (n, k)'s algorithm is chosen at: smaller batches retain that algorithm.
const canonical_rows = 16384;

pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: lt_.Library,
    workspace: cuda.DeviceBuffer,
    algos: std.AutoHashMapUnmanaged([2]u64, lt_.Algo) = .empty,
    plans: std.AutoHashMapUnmanaged([3]u64, lt_.Linear) = .empty,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |l| l.deinit();
        g.plans.deinit(g.gpa);
        g.algos.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y (rows, n) = x (rows, k) . w (n, k)^T in bf16 with fp32 sums, on `stream`.
    pub fn run(g: *Gemms, stream: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        const p = try g.plan(rows, n, k);
        try p.run(x, w, y, g.workspace.ptr, workspace_bytes, stream.handle);
    }

    /// The shape's plan, running the algorithm cuBLASLt's heuristic picks for (canonical_rows, n, k).
    fn plan(g: *Gemms, rows: usize, n: usize, k: usize) !*const lt_.Linear {
        const got = try g.plans.getOrPut(g.gpa, .{ rows, n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        const chosen = try g.algoFor(n, k);
        var l = try lt_.Linear.init(&g.lt, rows, n, k, .bf16, workspace_bytes);
        errdefer l.deinit();
        var checked: lt_.Heuristic = undefined;
        const s = g.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, &chosen, &checked);
        if (s != 0 or checked.state != 0 or checked.workspace_size > workspace_bytes) return error.NoAlgorithm;
        l.algo = chosen;
        l.workspace_need = checked.workspace_size;
        got.value_ptr.* = l;
        return got.value_ptr;
    }

    fn algoFor(g: *Gemms, n: usize, k: usize) !lt_.Algo {
        if (g.algos.get(.{ n, k })) |a| return a;
        var l = try lt_.Linear.init(&g.lt, canonical_rows, n, k, .bf16, workspace_bytes);
        defer l.deinit();
        try g.algos.put(g.gpa, .{ n, k }, l.algo);
        return l.algo;
    }
};
