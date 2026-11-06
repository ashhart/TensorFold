//! The decoder's prompt products on cuBLASLt: one algorithm per (n, k) at every row count, so a row's bits are its own.
const std = @import("std");
const cuda = @import("cuda");

const cublaslt = cuda.cublaslt;

/// Rows each (n, k)'s algorithm is picked at, the most one product takes: the decoder's prompt pass (`prompt_rows`).
pub const reference_rows = 2048;

/// Workspace bytes an algorithm may use, as `vit.Gemms` gives its own.
pub const workspace_bytes = 32 << 20;

/// An algorithm's split-K factor and reduction scheme as cuBLASLt reads them back.
pub const Config = struct { splitk: c_int, scheme: c_int };

/// Whether an algorithm sums each output's K in one order: no split K and no reduction of partials.
pub fn oneOrder(c: Config) bool {
    return c.splitk <= 1 and c.scheme == 0;
}

/// y [rows, n] = x [rows, k] . w [n, k]^T, bf16 in and out with fp32 sums; a plan's row count is set per call.
pub const Gemms = struct {
    gpa: std.mem.Allocator,
    lt: cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    plans: std.AutoHashMapUnmanaged([2]u64, Plan) = .empty,

    /// An (n, k)'s plan, and the row counts its algorithm was checked at.
    const Plan = struct {
        linear: cublaslt.Linear,
        checked: std.AutoHashMapUnmanaged(u64, void) = .empty,
    };

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try cublaslt.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |p| {
            p.checked.deinit(g.gpa);
            p.linear.deinit();
        }
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    pub fn run(g: *Gemms, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0) return;
        const p = try g.plan(n, k);
        const cols: u64 = rows;
        for ([_]cublaslt.Layout{ p.linear.b, p.linear.c }) |l| try g.lt.check(g.lt.api.cublasLtMatrixLayoutSetAttribute(l, cublaslt.layout_cols, &cols, @sizeOf(u64)), "set layout cols");
        if (!p.checked.contains(cols)) try g.check(p, rows);
        try p.linear.run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
    }

    /// The (n, k)'s plan: the heuristic's pick at `reference_rows`, refused if it splits K.
    fn plan(g: *Gemms, n: usize, k: usize) !*Plan {
        const got = try g.plans.getOrPut(g.gpa, .{ n, k });
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        var linear = try cublaslt.Linear.init(&g.lt, reference_rows, n, k, .bf16, workspace_bytes);
        errdefer linear.deinit();
        const c = config(&g.lt, &linear.algo) orelse return error.UocrSplitKUnreadable;
        if (!oneOrder(c)) {
            std.log.err("unlimited-ocr: the (n {d}, k {d}) algorithm splits K {d} ways (reduction {d})", .{ n, k, c.splitk, c.scheme });
            return error.UocrSplitKAlgorithm;
        }
        got.value_ptr.* = .{ .linear = linear };
        return got.value_ptr;
    }

    /// The plan's algorithm checked at `rows` rows once: it must run them within the workspace.
    fn check(g: *Gemms, p: *Plan, rows: usize) !void {
        const l = &p.linear;
        var h: cublaslt.Heuristic = undefined;
        if (g.lt.api.cublasLtMatmulAlgoCheck(l.handle, l.desc, l.a, l.b, l.c, l.c, &l.algo, &h) != 0 or h.state != 0) return error.UocrAlgorithmRows;
        if (h.workspace_size > workspace_bytes) return error.UocrAlgorithmWorkspace;
        l.workspace_need = @max(l.workspace_need, h.workspace_size);
        try p.checked.put(g.gpa, rows, {});
    }
};

fn config(lt: *const cublaslt.Library, algo: *const cublaslt.Algo) ?Config {
    var c: Config = .{ .splitk = 1, .scheme = 0 };
    for ([_]struct { c_int, *c_int }{ .{ cublaslt.algo_config_splitk, &c.splitk }, .{ cublaslt.algo_config_reduction, &c.scheme } }) |a| {
        var written: usize = 0;
        if (lt.api.cublasLtMatmulAlgoConfigGetAttribute(algo, a[0], a[1], @sizeOf(c_int), &written) != 0 or written != @sizeOf(c_int)) return null;
    }
    return c;
}

test "uocr: an algorithm keeps one K order unless it splits K or reduces partials" {
    try std.testing.expect(oneOrder(.{ .splitk = 1, .scheme = 0 }));
    try std.testing.expect(!oneOrder(.{ .splitk = 4, .scheme = 0 }));
    try std.testing.expect(!oneOrder(.{ .splitk = 1, .scheme = 3 }));
}
