//! Prompt rows' dense EXL3 projections (exl3.md, Prompts): W decoded once into fp16, a cuBLASLt GEMM, the rotation.
const std = @import("std");
const cuda = @import("cuda");

const lt_ = cuda.cublaslt;
const Buf = cuda.DeviceBuffer;

/// Calls of fewer rows keep exl3.Dense (decode rows, short prompts: the same bits as decode).
pub const min_rows = 256;

/// Whether prompt rows take the GEMM: Small's do; Mini's keep the decode kernels (its loop grows the GEMM's sum order).
pub fn usedBy(c: anytype) bool {
    return c.loop == null;
}
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
        return make(gpa, d, lt, cuda.kernels.exl3_candidate, tail, rows_max, k_max, n_max);
    }

    /// Everything but the library, which the caller closes on failure; a failed step leaves nothing else behind.
    fn make(gpa: std.mem.Allocator, d: *const cuda.Driver, lt: lt_.Library, image: []const u8, tail: cuda.Function, rows_max: usize, k_max: usize, n_max: usize) !Prompt {
        const handle = try lt_.createHandle(&lt);
        errdefer _ = lt.api.cublasLtDestroy(handle);
        var mod = try cuda.Module.load(d, image);
        errdefer mod.unload();
        const rotate = try mod.function("tf_exl3_rotate_h");
        const unpack = try mod.function("tf_exl3_unpack");
        var xh = try Buf.alloc(d, rows_max * k_max * 2);
        errdefer xh.free();
        var w = try Buf.alloc(d, k_max * n_max * 2);
        errdefer w.free();
        var z = try Buf.alloc(d, rows_max * n_max * 4);
        errdefer z.free();
        const ws = try Buf.alloc(d, workspace_bytes);
        return .{ .gpa = gpa, .d = d, .lt = lt, .handle = handle, .mod = mod, .rotate = rotate, .unpack = unpack, .tail = tail, .xh = xh, .w = w, .z = z, .ws = ws, .rows_max = rows_max };
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
        var g = try lt_.Gemm.init(&p.lt, p.handle, .{ .m = n, .n = rows, .k = k, .a = .f16, .b = .f16, .c = .f32, .lda = n, .ldb = k, .ldc = n }, workspace_bytes);
        errdefer g.deinit();
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

/// A driver and cuBLASLt that make handles and count them, failing the `fail_at`th device allocation (0: none).
const Fake = struct {
    var made: usize = 0;
    var gone: usize = 0;
    var allocs: usize = 0;
    var fail_at: usize = 0;
    var fail_load = false;
    var token: u8 = 0;

    fn reset(at: usize, refuse_load: bool) void {
        made = 0;
        gone = 0;
        allocs = 0;
        fail_at = at;
        fail_load = refuse_load;
    }
    fn make(p: anytype) cuda.abi.Result {
        made += 1;
        p.* = @ptrCast(&token);
        return 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        gone += 1;
        return 0;
    }
    fn destroy(_: ?*anyopaque) callconv(.c) c_int {
        gone += 1;
        return 0;
    }
    fn load(m: *cuda.abi.Module, _: ?*const anyopaque) callconv(.c) cuda.abi.Result {
        return if (fail_load) cuda.abi.error_not_ready else make(m);
    }
    fn function(f: *cuda.abi.Function, _: cuda.abi.Module, _: [*:0]const u8) callconv(.c) cuda.abi.Result {
        f.* = @ptrCast(&token);
        return 0;
    }
    fn alloc(p: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
        allocs += 1;
        if (allocs == fail_at) return cuda.abi.error_not_ready;
        made += 1;
        p.* = allocs << 20;
        return 0;
    }
    fn free(_: cuda.abi.DevicePtr) callconv(.c) cuda.abi.Result {
        gone += 1;
        return 0;
    }
    fn create(h: *lt_.Handle) callconv(.c) c_int {
        return make(h);
    }
    fn desc(x: *lt_.MatmulDesc, _: c_int, _: lt_.DataType) callconv(.c) c_int {
        return make(x);
    }
    fn layout(x: *lt_.Layout, _: lt_.DataType, _: u64, _: u64, _: i64) callconv(.c) c_int {
        return make(x);
    }
    fn pref(x: *lt_.Preference) callconv(.c) c_int {
        return make(x);
    }
    fn set(_: ?*anyopaque, _: c_int, _: ?*const anyopaque, _: usize) callconv(.c) c_int {
        return 0;
    }
    fn heuristic(_: lt_.Handle, _: lt_.MatmulDesc, _: lt_.Layout, _: lt_.Layout, _: lt_.Layout, _: lt_.Layout, _: lt_.Preference, _: c_int, out: [*]lt_.Heuristic, n: *c_int) callconv(.c) c_int {
        out[0] = std.mem.zeroes(lt_.Heuristic);
        n.* = 1;
        return 0;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuModuleLoadData = load;
        d.api.cuModuleUnload = unload;
        d.api.cuModuleGetFunction = function;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        return d;
    }
    fn library() lt_.Library {
        var lt: lt_.Library = undefined;
        lt.api.cublasLtCreate = create;
        lt.api.cublasLtDestroy = @ptrCast(&destroy);
        lt.api.cublasLtMatmulDescCreate = desc;
        lt.api.cublasLtMatmulDescDestroy = @ptrCast(&destroy);
        lt.api.cublasLtMatmulDescSetAttribute = @ptrCast(&set);
        lt.api.cublasLtMatrixLayoutCreate = layout;
        lt.api.cublasLtMatrixLayoutDestroy = @ptrCast(&destroy);
        lt.api.cublasLtMatmulPreferenceCreate = pref;
        lt.api.cublasLtMatmulPreferenceDestroy = @ptrCast(&destroy);
        lt.api.cublasLtMatmulPreferenceSetAttribute = @ptrCast(&set);
        lt.api.cublasLtMatmulAlgoGetHeuristic = heuristic;
        return lt;
    }
};

test "prompt dense: a failed module load or allocation frees everything made before it" {
    var d = Fake.driver();
    const image: [8]u8 align(8) = @splat(1);
    const tail: cuda.Function = .{ .d = &d, .handle = null };
    Fake.reset(0, true);
    try std.testing.expectError(error.NotReady, Prompt.make(std.testing.allocator, &d, Fake.library(), &image, tail, 256, 128, 128));
    try std.testing.expectEqual(Fake.made, Fake.gone); // the handle
    for (1..5) |at| {
        Fake.reset(at, false);
        try std.testing.expectError(error.NotReady, Prompt.make(std.testing.allocator, &d, Fake.library(), &image, tail, 256, 128, 128));
        // the handle, the module and the allocations before the failed one
        try std.testing.expectEqual(at + 1, Fake.made);
        try std.testing.expectEqual(Fake.made, Fake.gone);
    }
}

test "prompt dense: a plan whose insertion fails destroys its descriptor and layouts" {
    var d = Fake.driver();
    const image: [8]u8 align(8) = @splat(1);
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    Fake.reset(0, false);
    var p = try Prompt.make(failing.allocator(), &d, Fake.library(), &image, .{ .d = &d, .handle = null }, 256, 128, 128);
    const made = Fake.made;
    const gone = Fake.gone;
    try std.testing.expectError(error.OutOfMemory, p.plan(256, 128, 128));
    try std.testing.expectEqual(made + 5, Fake.made); // a descriptor, three layouts and a preference
    try std.testing.expectEqual(gone + 5, Fake.gone);
    try std.testing.expectEqual(@as(usize, 0), p.plans.items.len);
}
