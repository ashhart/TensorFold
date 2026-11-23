//! cuBLASLt projections, one plan and algorithm per (n, k) at any row count; 1 row may run a bit-equal second one.
const std = @import("std");
const cuda = @import("cuda");

const one_row = cuda.one_row;

const cublaslt = cuda.cublaslt;
const Allocator = std.mem.Allocator;

/// Rows the algorithm is asked at: a prompt pass's rows, past the point where any Qwen3 (n, k) needs to split K.
pub const reference_rows = 4096;

/// Workspace bytes a plan may ask for, the budget `vit.Gemms` gives its own algorithms.
pub const workspace_bytes = 32 << 20;

/// An algorithm's split-K factor and its reduction scheme.
pub const Config = struct { splitk: c_int, scheme: c_int };

/// The plan cache's key: an (n, k) alone, the row count set per call.
pub fn key(n: usize, k: usize) [2]u64 {
    return .{ n, k };
}

/// A layout's column count as cuBLASLt reads it: CUBLASLT_MATRIX_LAYOUT_COLS is a uint64_t.
pub fn columns(rows: usize) u64 {
    return @intCast(rows);
}

/// Whether an algorithm keeps a row's K sums in one order: split-K 1, no reduction scheme; null claims nothing.
pub fn oneOrder(x: Config) bool {
    return x.splitk <= 1 and x.scheme == 0;
}

/// An algorithm's config as cuBLASLt reads it back; null when it will not say.
pub fn config(lt: *const cublaslt.Library, algo: cublaslt.Algo) ?Config {
    var c: Config = .{ .splitk = 1, .scheme = 0 };
    if (!attribute(lt, &algo, cublaslt.algo_config_splitk, &c.splitk)) return null;
    if (!attribute(lt, &algo, cublaslt.algo_config_reduction, &c.scheme)) return null;
    return c;
}

fn attribute(lt: *const cublaslt.Library, algo: *const cublaslt.Algo, id: c_int, out: *c_int) bool {
    var written: usize = 0; // cuBLASLt writes a size_t here
    if (lt.api.cublasLtMatmulAlgoConfigGetAttribute(algo, id, out, @sizeOf(c_int), &written) != 0) return false;
    return written == @sizeOf(c_int);
}

pub const Gemms = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    lt: cublaslt.Library,
    workspace: cuda.DeviceBuffer,
    /// The one plan each (n, k) runs, on the one algorithm its rows keep at every count.
    plans: std.AutoHashMapUnmanaged([2]u64, Plan) = .empty,
    /// A check mode: every call of a plan with a 1-row algorithm also runs the other algorithm and compares each row.
    tally: ?*one_row.Tally = null,

    /// A (n, k)'s plan and the row counts its algorithm has been checked at.
    pub const Plan = struct {
        linear: cublaslt.Linear,
        /// The row counts already checked: one in here needs no `cublasLtMatmulAlgoCheck` again.
        checked: std.AutoHashMapUnmanaged(u64, void) = .empty,
        /// A 1-row call's algorithm when one gives the plan's bits (cuda/one_row.zig); null: the plan's own at 1 row.
        one: ?cublaslt.Linear = null,
    };

    pub fn init(gpa: Allocator, d: *const cuda.Driver) !Gemms {
        var lt = try cublaslt.Library.open();
        errdefer lt.close();
        return .{ .gpa = gpa, .d = d, .lt = lt, .workspace = try cuda.DeviceBuffer.alloc(d, workspace_bytes) };
    }

    pub fn deinit(g: *Gemms) void {
        var it = g.plans.valueIterator();
        while (it.next()) |p| {
            if (p.one) |*o| o.deinit();
            p.checked.deinit(g.gpa);
            p.linear.deinit();
        }
        g.plans.deinit(g.gpa);
        g.workspace.free();
        g.lt.close();
    }

    /// y [rows, n] = x [rows, k] . w [n, k]^T, bf16 in and out with fp32 sums, the (n, k)'s plan at `rows` rows.
    pub fn run(g: *Gemms, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        if (rows == 0) return;
        const p = try g.plan(n, k);
        if (rows == 1) if (p.one) |*o| {
            try o.run(x, w, y, g.workspace.ptr, workspace_bytes, s.handle);
            if (g.tally) |t| try g.compare(s, p, x, w, y, rows, n, k, t);
            return;
        };
        try g.runPlan(s.handle, p, x, w, y, rows, n, k);
        if (g.tally) |t| if (p.one != null) try g.compare(s, p, x, w, y, rows, n, k, t);
    }

    fn runPlan(g: *Gemms, s: cuda.abi.Stream, p: *Plan, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize) !void {
        try setCols(&g.lt, &p.linear, rows);
        if (!p.checked.contains(columns(rows))) try g.check(p, n, k, rows);
        try p.linear.run(x, w, y, g.workspace.ptr, workspace_bytes, s);
    }

    /// The check mode's comparison of `y` [rows, n]: the plan's algorithm at 1 row, or each row alone on the 1-row one.
    fn compare(g: *Gemms, s: cuda.Stream, p: *Plan, x: u64, w: u64, y: u64, rows: usize, n: usize, k: usize, t: *one_row.Tally) !void {
        var other = try cuda.DeviceBuffer.alloc(g.d, 2 * rows * n);
        defer other.free();
        if (rows == 1) try g.runPlan(s.handle, p, x, w, other.ptr, 1, n, k) else for (0..rows) |r| {
            try p.one.?.run(x + r * k * 2, w, other.ptr + r * n * 2, g.workspace.ptr, workspace_bytes, s.handle);
        }
        try s.synchronize();
        const a = try g.gpa.alloc(u8, 2 * rows * n);
        defer g.gpa.free(a);
        const b = try g.gpa.alloc(u8, 2 * rows * n);
        defer g.gpa.free(b);
        const view: cuda.DeviceBuffer = .{ .d = g.d, .ptr = y, .len = a.len, .borrowed = true };
        try view.download(0, a);
        try other.download(0, b);
        for (0..rows) |r| {
            if (!std.mem.eql(u8, a[r * n * 2 ..][0 .. n * 2], b[r * n * 2 ..][0 .. n * 2])) t.differ += 1;
        }
        t.rows += rows;
    }

    /// Builds the plans of the (n, k) pairs a pass multiplies by, so a split-K algorithm is refused before any request.
    pub fn prepare(g: *Gemms, pairs: []const [2]usize) !void {
        for (pairs) |p| _ = try g.plan(p[0], p[1]);
    }

    /// Checks every plan's algorithm at `rows` rows now, so a row count the window allows is not refused mid-request.
    pub fn checkRows(g: *Gemms, rows: usize) !void {
        var it = g.plans.iterator();
        while (it.next()) |kv| {
            const p = kv.value_ptr;
            try setCols(&g.lt, &p.linear, rows);
            if (!p.checked.contains(columns(rows))) try g.check(p, @intCast(kv.key_ptr[0]), @intCast(kv.key_ptr[1]), rows);
        }
    }

    /// The (n, k)'s plan, built once at `reference_rows` on its one algorithm, with its 1-row algorithm if any.
    pub fn plan(g: *Gemms, n: usize, k: usize) !*Plan {
        const got = try g.plans.getOrPut(g.gpa, key(n, k));
        if (got.found_existing) return got.value_ptr;
        errdefer g.plans.removeByPtr(got.key_ptr);
        var linear = try cublaslt.Linear.initEpilogue(&g.lt, reference_rows, n, k, .bf16, workspace_bytes, .none);
        errdefer linear.deinit();
        try accept(&g.lt, n, k, linear.algo);
        got.value_ptr.* = .{ .linear = linear };
        errdefer got.value_ptr.checked.deinit(g.gpa);
        try g.check(got.value_ptr, n, k, reference_rows);
        got.value_ptr.one = try g.oneRow(got.value_ptr, n, k);
        return got.value_ptr;
    }

    /// The algorithm checked at `rows` rows: its workspace counted and the row count kept, so it is checked once.
    fn check(g: *Gemms, p: *Plan, n: usize, k: usize, rows: usize) !void {
        var checked: cublaslt.Heuristic = undefined;
        if (g.lt.api.cublasLtMatmulAlgoCheck(p.linear.handle, p.linear.desc, p.linear.a, p.linear.b, p.linear.c, p.linear.c, &p.linear.algo, &checked) != 0 or checked.state != 0) {
            std.log.err("qwen3: the (n {d}, k {d}) algorithm does not run {d} rows' product", .{ n, k, rows });
            return error.AlgorithmMismatch;
        }
        if (checked.workspace_size > workspace_bytes) {
            std.log.err("qwen3: {d} rows of (n {d}, k {d}) want {d} workspace bytes, the budget is {d}", .{ rows, n, k, checked.workspace_size, workspace_bytes });
            return error.WorkspaceTooSmall;
        }
        p.linear.workspace_need = @max(p.linear.workspace_need, checked.workspace_size);
        try p.checked.put(g.gpa, columns(rows), {});
    }

    /// The 1-row heuristic's first candidate one_row.pick takes, or null: then 1-row calls run the plan's algorithm.
    fn oneRow(g: *Gemms, p: *Plan, n: usize, k: usize) !?cublaslt.Linear {
        var one = try cublaslt.Linear.initEpilogue(&g.lt, 1, n, k, .bf16, workspace_bytes, .none);
        errdefer one.deinit();
        var found: [one_row.candidates]cublaslt.Heuristic = undefined;
        var count: c_int = 0;
        try g.lt.check(g.lt.api.cublasLtMatmulAlgoGetHeuristic(one.handle, one.desc, one.a, one.b, one.c, one.c, one.pref, one_row.candidates, &found, &count), "cublasLtMatmulAlgoGetHeuristic");
        if (try g.oneRowFrom(p, &one, n, k, found[0..@intCast(@max(count, 0))])) return one;
        one.deinit();
        return null;
    }

    /// Sets `one` to the candidate one_row.pick takes from `cands`, against the plan's algorithm on the probe rows.
    pub fn oneRowFrom(g: *Gemms, p: *Plan, one: *cublaslt.Linear, n: usize, k: usize, cands: []const cublaslt.Heuristic) !bool {
        var pr = try one_row.Probe.init(g.gpa, g.d, n, k);
        defer pr.deinit(g.gpa);
        try g.probeRef(p, &pr, n, k);
        const ctx: Same = .{ .g = g, .one = one, .pr = &pr, .n = n, .k = k };
        const i = try one_row.pick(cands, workspace_bytes, ctx) orelse {
            std.log.info("qwen3: (n {d}, k {d}): 1 row on the {d}-row pick, no 1-row pick gives its bits", .{ n, k, reference_rows });
            return false;
        };
        one.algo = cands[i].algo;
        one.workspace_need = cands[i].workspace_size;
        std.log.info("qwen3: (n {d}, k {d}): 1 row on the 1-row heuristic's pick {d}: {d} probe rows alone give the {d}-row pick's bits", .{ n, k, i, one_row.rows, reference_rows });
        return true;
    }

    /// Whether `h` gives the plan's bits on every probe row, run alone; the K-order check is the caller's (a control).
    pub fn probeSame(g: *Gemms, p: *Plan, n: usize, k: usize, h: cublaslt.Heuristic) !bool {
        var one = try cublaslt.Linear.initEpilogue(&g.lt, 1, n, k, .bf16, workspace_bytes, .none);
        defer one.deinit();
        var pr = try one_row.Probe.init(g.gpa, g.d, n, k);
        defer pr.deinit(g.gpa);
        try g.probeRef(p, &pr, n, k);
        return (Same{ .g = g, .one = &one, .pr = &pr, .n = n, .k = k }).same(h);
    }

    /// The plan's algorithm over all the probe rows at once, downloaded into `pr.want`.
    fn probeRef(g: *Gemms, p: *Plan, pr: *one_row.Probe, n: usize, k: usize) !void {
        try g.runPlan(null, p, pr.x.ptr, pr.w.ptr, pr.ref.ptr, one_row.rows, n, k);
        try g.d.check(g.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        try pr.ref.download(0, pr.want);
    }

    /// one_row.pick's view of the device: a candidate's K order, and its probe rows run alone against the plan's.
    const Same = struct {
        g: *Gemms,
        one: *cublaslt.Linear,
        pr: *one_row.Probe,
        n: usize,
        k: usize,

        pub fn keepsOrder(c: Same, algo: cublaslt.Algo) bool {
            const x = config(&c.g.lt, algo) orelse return false;
            return oneOrder(x);
        }

        pub fn same(c: Same, h: cublaslt.Heuristic) !bool {
            const g = c.g;
            c.one.algo = h.algo;
            c.one.workspace_need = h.workspace_size;
            for (0..one_row.rows) |r| {
                try c.one.run(c.pr.x.ptr + r * c.k * 2, c.pr.w.ptr, c.pr.y.ptr + r * c.n * 2, g.workspace.ptr, workspace_bytes, null);
            }
            try g.d.check(g.d.api.cuCtxSynchronize(), "cuCtxSynchronize");
            try c.pr.y.download(0, c.pr.got);
            return std.mem.eql(u8, c.pr.got, c.pr.want);
        }
    };
};

/// Refuses an algorithm whose K sums are split or reduced, and says what it is: one algorithm an (n, k) keeps.
fn accept(lt: *const cublaslt.Library, n: usize, k: usize, algo: cublaslt.Algo) !void {
    const x = config(lt, algo) orelse {
        std.log.err("qwen3: (n {d}, k {d}): the {d}-row pick's split K cannot be read, so one K order is not proven", .{ n, k, reference_rows });
        return error.SplitKUnreadable;
    };
    if (!oneOrder(x)) {
        std.log.err("qwen3: the (n {d}, k {d}) algorithm splits K {d} ways with reduction {d}: its rows would not keep a batch's bits", .{ n, k, x.splitk, x.scheme });
        return error.SplitKAlgorithm;
    }
    std.log.info("qwen3: (n {d}, k {d}): the {d}-row pick, split K {d}", .{ n, k, reference_rows, x.splitk });
}

/// Sets a plan's X and D layouts to `rows` rows: cuBLASLt's column-major X is k x rows, D is n x rows.
fn setCols(lt: *const cublaslt.Library, p: *const cublaslt.Linear, rows: usize) !void {
    const cols = columns(rows);
    for ([_]cublaslt.Layout{ p.b, p.c }) |l| try lt.check(lt.api.cublasLtMatrixLayoutSetAttribute(l, cublaslt.layout_cols, &cols, @sizeOf(u64)), "set layout cols");
}

test "an algorithm keeps one K order unless it splits K or reduces partials" {
    try std.testing.expect(oneOrder(.{ .splitk = 1, .scheme = 0 }));
    try std.testing.expect(!oneOrder(.{ .splitk = 4, .scheme = 0 }));
    try std.testing.expect(!oneOrder(.{ .splitk = 1, .scheme = 3 }));
    try std.testing.expect(!oneOrder(.{ .splitk = 2, .scheme = 2 }));
}
