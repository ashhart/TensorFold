//! Prompt rows' dense EXL3 projections (exl3.md, Prompts): W decoded once into fp16, a cuBLASLt GEMM, the rotation.
const std = @import("std");
const cuda = @import("cuda");

const lt_ = cuda.cublaslt;
const Buf = cuda.DeviceBuffer;

/// Calls of fewer rows keep exl3.Dense (decode rows, short prompts: the same bits as decode).
pub const min_rows = 256;
const workspace_bytes = 32 << 20;

const Plan = struct { rows: usize, k: usize, n: usize, g: lt_.Gemm };

pub const Prompt = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lt: lt_.Library,
    handle: lt_.Handle,
    mod: cuda.Module,
    rotate: cuda.Function,
    unpack: cuda.Function,
    tail: cuda.Function,
    xh: Buf, // fp16 [rows][k]: (x * suh) H 2^-4
    w: Buf, // fp16 [k][n]: W_q in the rotated domain
    z: Buf, // fp32 [rows][n]: the GEMM's sums
    ws: Buf,
    plans: std.ArrayList(Plan) = .empty,
    rows_max: usize,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, tail: cuda.Function, rows_max: usize, k_max: usize, n_max: usize) !Prompt {
        var lt = try lt_.Library.open();
        errdefer lt.close();
        const handle = try lt_.createHandle(&lt);
        errdefer _ = lt.api.cublasLtDestroy(handle);
        var mod = try cuda.Module.load(d, cuda.kernels.exl3_candidate);
        errdefer mod.unload();
        var xh = try Buf.alloc(d, rows_max * k_max * 2);
        errdefer xh.free();
        var w = try Buf.alloc(d, k_max * n_max * 2);
        errdefer w.free();
        var z = try Buf.alloc(d, rows_max * n_max * 4);
        errdefer z.free();
        const ws = try Buf.alloc(d, workspace_bytes);
        return .{ .gpa = gpa, .d = d, .lt = lt, .handle = handle, .mod = mod, .rotate = try mod.function("tf_exl3_rotate_h"), .unpack = try mod.function("tf_exl3_unpack"), .tail = tail, .xh = xh, .w = w, .z = z, .ws = ws, .rows_max = rows_max };
    }

    /// Scratch for the widest attention projection of `layers` at `rows` rows.
    pub fn of(gpa: std.mem.Allocator, d: *const cuda.Driver, tail: cuda.Function, rows: usize, layers: anytype) !Prompt {
        var k: usize = 0;
        var n: usize = 0;
        for (layers) |l| for ([_]cuda.exl3.Weight{ l.attn.q, l.attn.k, l.attn.v, l.attn.r, l.attn.gate, l.attn.o }) |x| {
            k = @max(k, x.k);
            n = @max(n, x.n);
        };
        return init(gpa, d, tail, rows, k, n);
    }

    pub fn deinit(p: *Prompt) void {
        for (p.plans.items) |*x| x.g.deinit();
        p.plans.deinit(p.gpa);
        for ([_]*Buf{ &p.xh, &p.w, &p.z, &p.ws }) |b| b.free();
        p.mod.unload();
        _ = p.lt.api.cublasLtDestroy(p.handle);
        p.lt.close();
    }

    /// Whether `w` takes this path: whole bits (the unpack's widths), no bias, the stored layout, within the scratch.
    pub fn takes(p: *const Prompt, w: cuda.exl3.Weight, rows: usize) bool {
        const st = cuda.exl3.strides(w.k, w.n, w.halves, false);
        return rows >= min_rows and rows <= p.rows_max and w.halves % 2 == 0 and w.halves <= 16 and w.bias == 0 and
            w.stride_k == st.k and w.stride_nb == st.nb and @as(usize, w.k) * 2 * rows <= p.xh.len and
            @as(usize, w.k) * w.n * 2 <= p.w.len and rows * w.n * 4 <= p.z.len;
    }

    fn plan(p: *Prompt, rows: usize, k: usize, n: usize) !*const lt_.Gemm {
        for (p.plans.items) |*x| if (x.rows == rows and x.k == k and x.n == n) return &x.g;
        // column-major: Z^T [n, rows] = W^T [n, k] . Xh^T [k, rows]
        const g = try lt_.Gemm.init(&p.lt, p.handle, .{ .m = n, .n = rows, .k = k, .a = .f16, .b = .f16, .c = .f32, .lda = n, .ldb = k, .ldc = n }, workspace_bytes);
        try p.plans.append(p.gpa, .{ .rows = rows, .k = k, .n = n, .g = g });
        return &p.plans.items[p.plans.items.len - 1].g;
    }

    /// y [rows][n] (bf16 or fp32) = x [rows][k] (bf16) through `w`, on `s`.
    pub fn run(p: *Prompt, s: cuda.Stream, w: cuda.exl3.Weight, x: u64, rows: usize, y: u64, y_f32: bool) !void {
        const g = try p.plan(rows, w.k, w.n);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(u32, w.k));
        a.add(w.suh);
        a.add(p.xh.ptr);
        a.add(@as(u32, w.k));
        try cuda.launch.launch(p.rotate, .{ .grid = .{ .x = w.k / 128, .y = @intCast(rows) }, .block = .{ .x = 32 } }, s, &a);
        var b: cuda.Args = .{};
        b.add(w.trellis);
        b.add(MatrixArgs{ .k = w.k, .n = w.n, .bits = w.halves / 2, .tiles_n = w.n / 16, .split = 1 });
        b.add(@as(u32, @backingInt(w.codebook)));
        b.add(p.w.ptr);
        try cuda.launch.launch(p.unpack, .{ .grid = .{ .x = w.n / 16, .y = w.k / 16 }, .block = .{ .x = 32 } }, s, &b);
        try g.run(1.0, p.w.ptr, p.xh.ptr, 0.0, p.z.ptr, p.ws.ptr, p.ws.len, s.handle);
        var c: cuda.Args = .{};
        for ([_]u64{ p.z.ptr, w.svh }) |v| c.add(v);
        c.add(@as(u32, w.n));
        c.add(y);
        c.add(@as(u32, @intFromBool(y_f32)));
        try cuda.launch.launch(p.tail, .{ .grid = .{ .x = w.n / 128, .y = @intCast(rows) }, .block = .{ .x = 32 } }, s, &c);
    }
};

/// exl3_candidate.cu's TfExl3Matrix, by value.
const MatrixArgs = extern struct { k: u32, n: u32, bits: u32, tiles_n: u32, split: u32 };
