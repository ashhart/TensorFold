//! The DiT's fp16 projections on the runtime's cuBLASLt: D = X W^T (+ bias) (+ D), fp32 compute, a tuned plan a shape.
const std = @import("std");
const cuda = @import("cuda");

const lt_ = cuda.cublaslt;
const Ptr = cuda.abi.DevicePtr;

pub const Plan = struct {
    lib: *const lt_.Library,
    handle: lt_.Handle,
    desc: lt_.MatmulDesc,
    a: lt_.Layout,
    b: lt_.Layout,
    c: lt_.Layout,
    algo: lt_.Algo,
    ws_need: usize,
    bias: bool,
    beta: f32,
    m: u64,
    n: u64,
    k: u64,
    found: [max_algos]lt_.Heuristic = undefined,
    n_found: usize = 0,
    tuned: bool = false,

    /// m rows of X, n outputs, k inputs; `bias` adds a fp16 bias over the outputs; `accumulate` adds D's old values.
    pub fn init(lib: *const lt_.Library, handle: lt_.Handle, m: u64, n: u64, k: u64, bias: bool, accumulate: bool, ws_limit: usize) !Plan {
        var p: Plan = undefined;
        p.lib = lib;
        p.handle = handle;
        p.bias = bias;
        p.beta = if (accumulate) 1 else 0;
        p.m = m;
        p.n = n;
        p.k = k;
        const api = lib.api;
        try lib.check(api.cublasLtMatmulDescCreate(&p.desc, lt_.compute_32f, .f32), "cublasLtMatmulDescCreate");
        errdefer _ = api.cublasLtMatmulDescDestroy(p.desc);
        const ta: c_int = @backingInt(lt_.Op.t);
        const tb: c_int = @backingInt(lt_.Op.n);
        try lib.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt_.desc_transa, &ta, @sizeOf(c_int)), "transa");
        try lib.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt_.desc_transb, &tb, @sizeOf(c_int)), "transb");
        if (bias) {
            const e: u32 = @backingInt(lt_.Epilogue.bias);
            try lib.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt_.desc_epilogue, &e, @sizeOf(u32)), "epilogue");
            const bt: c_int = @backingInt(lt_.DataType.f16);
            try lib.check(api.cublasLtMatmulDescSetAttribute(p.desc, desc_bias_data_type, &bt, @sizeOf(c_int)), "bias type");
        }
        try lib.check(api.cublasLtMatrixLayoutCreate(&p.a, .f16, k, n, @intCast(k)), "layout W");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.a);
        try lib.check(api.cublasLtMatrixLayoutCreate(&p.b, .f16, k, m, @intCast(k)), "layout X");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.b);
        try lib.check(api.cublasLtMatrixLayoutCreate(&p.c, .f16, n, m, @intCast(n)), "layout D");
        errdefer _ = api.cublasLtMatrixLayoutDestroy(p.c);
        var pref: lt_.Preference = null;
        try lib.check(api.cublasLtMatmulPreferenceCreate(&pref), "preference");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = ws_limit;
        try lib.check(api.cublasLtMatmulPreferenceSetAttribute(pref, lt_.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "workspace");
        var count: c_int = 0;
        try lib.check(api.cublasLtMatmulAlgoGetHeuristic(handle, p.desc, p.a, p.b, p.c, p.c, pref, max_algos, &p.found, &count), "heuristic");
        if (count < 1 or p.found[0].state != 0) return error.NoAlgorithm;
        p.n_found = @intCast(count);
        p.algo = p.found[0].algo;
        p.ws_need = p.found[0].workspace_size;
        p.tuned = count == 1;
        return p;
    }

    /// Times the heuristic's candidates on this shape (D into `scratch`, the real output untouched), keeps the best.
    pub fn tune(p: *Plan, d: *const cuda.Driver, s: cuda.Stream, x: Ptr, w: Ptr, bias: Ptr, scratch: Ptr, ws: Ptr, ws_len: usize) !void {
        p.tuned = true;
        var e0 = try cuda.Event.init(d, true);
        defer e0.deinit();
        var e1 = try cuda.Event.init(d, true);
        defer e1.deinit();
        var best: f32 = std.math.inf(f32);
        var pick: usize = 0;
        for (p.found[0..p.n_found], 0..) |hcand, idx| {
            if (hcand.state != 0 or hcand.workspace_size > ws_len) continue;
            p.algo = hcand.algo;
            p.run(s, x, w, bias, scratch, ws, ws_len) catch continue;
            try e0.record(s);
            for (0..3) |_| try p.run(s, x, w, bias, scratch, ws, ws_len);
            try e1.record(s);
            try e1.synchronize();
            const t = try cuda.Event.elapsedMs(e0, e1);
            if (t < best) {
                best = t;
                pick = idx;
            }
        }
        p.algo = p.found[pick].algo;
        p.ws_need = p.found[pick].workspace_size;
        std.log.debug("music3: lt {d}x{d}x{d} bias {} acc {}: {d} candidates, best #{d} {d:.3} ms", .{ p.m, p.n, p.k, p.bias, p.beta != 0, p.n_found, pick, best / 3 });
    }

    pub fn deinit(p: *Plan) void {
        const api = p.lib.api;
        _ = api.cublasLtMatrixLayoutDestroy(p.c);
        _ = api.cublasLtMatrixLayoutDestroy(p.b);
        _ = api.cublasLtMatrixLayoutDestroy(p.a);
        _ = api.cublasLtMatmulDescDestroy(p.desc);
    }

    /// D = X W^T (+ bias) (+ D) on stream s.
    pub fn run(p: *const Plan, s: cuda.Stream, x: Ptr, w: Ptr, bias: Ptr, d: Ptr, ws: Ptr, ws_len: usize) !void {
        if (ws_len < p.ws_need) return error.WorkspaceTooSmall;
        const api = p.lib.api;
        if (p.bias) try p.lib.check(api.cublasLtMatmulDescSetAttribute(p.desc, lt_.desc_bias_pointer, &bias, @sizeOf(Ptr)), "bias");
        const one: f32 = 1;
        try p.lib.check(api.cublasLtMatmul(p.handle, p.desc, &one, w, p.a, x, p.b, &p.beta, d, p.c, d, p.c, &p.algo, ws, ws_len, s.handle), "cublasLtMatmul");
    }
};

const max_algos = 32;

/// CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
const desc_bias_data_type: c_int = 26;

/// Plans by shape, made on first use.
pub const Plans = struct {
    gpa: std.mem.Allocator,
    lib: lt_.Library,
    handle: lt_.Handle,
    list: std.ArrayList(Plan) = .empty,
    ws: cuda.DeviceBuffer,

    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, ws_bytes: usize) !Plans {
        var lib = try lt_.Library.open();
        errdefer lib.close();
        var h: lt_.Handle = null;
        try lib.check(lib.api.cublasLtCreate(&h), "cublasLtCreate");
        return .{ .gpa = gpa, .lib = lib, .handle = h, .ws = try cuda.DeviceBuffer.alloc(d, ws_bytes) };
    }

    pub fn close(ps: *Plans) void {
        for (ps.list.items) |*p| p.deinit();
        ps.list.deinit(ps.gpa);
        _ = ps.lib.api.cublasLtDestroy(ps.handle);
        ps.lib.close();
        ps.ws.free();
    }

    pub fn get(ps: *Plans, m: u64, n: u64, k: u64, bias: bool, accumulate: bool) !*Plan {
        for (ps.list.items) |*p| if (p.m == m and p.n == n and p.k == k and p.bias == bias and (p.beta != 0) == accumulate) return p;
        try ps.list.append(ps.gpa, try Plan.init(&ps.lib, ps.handle, m, n, k, bias, accumulate, ps.ws.len));
        return &ps.list.items[ps.list.items.len - 1];
    }

    /// D[m, n] = X W^T (+ bias when nonzero) (+ D when accumulate); a new shape is tuned first.
    pub fn linear(ps: *Plans, s: cuda.Stream, x: Ptr, w: Ptr, bias: Ptr, d: Ptr, m: usize, n: usize, k: usize, accumulate: bool) !void {
        const p = try ps.get(m, n, k, bias != 0, accumulate);
        if (!p.tuned) {
            var scratch = try cuda.DeviceBuffer.alloc(ps.ws.d, m * n * 2);
            defer scratch.free();
            try s.synchronize();
            try p.tune(ps.ws.d, s, x, w, bias, scratch.ptr, ps.ws.ptr, ps.ws.len);
            try s.synchronize();
        }
        try p.run(s, x, w, bias, d, ps.ws.ptr, ps.ws.len);
    }
};
