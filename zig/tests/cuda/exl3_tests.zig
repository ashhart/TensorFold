//! EXL3 projections on CUDA: codebooks and tiles bit for bit against core/exl3_format.zig, FP64 bounds, row windows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const Gpu = check.Gpu;
const fmt = core.exl3_format;
const E = cuda.exl3_linear;
const st = core.safetensors;

const k_max = 16384; // GLM-5.3-Flash's widest input (o_proj: 64 heads of 256)
const u = std.math.pow(f64, 2, -24);

/// Device buffers for one test, freed together.
pub const Rig = struct {
    gpu: Gpu,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn put(r: *Rig, items: anytype) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(items));
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }
    pub fn zeros(r: *Rig, len: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(len, 16));
        try b.fill8(0, null);
        try r.gpu.d.check(r.gpu.d.api.cuStreamSynchronize(null), "cuStreamSynchronize");
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }
    pub fn get(r: *Rig, ptr: u64, out: []u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = out.len };
        try b.download(0, out);
    }
    pub fn deinit(r: *Rig) void {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(r.gpu.gpa);
    }
};

fn refBook(c: E.Codebook) fmt.Codebook {
    return switch (c) {
        .inst3 => .inst3,
        .mcg => .mcg,
        .mul1 => .mul1,
    };
}

pub fn bf16(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn ofBf16(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

fn gamma(n: usize) f64 {
    return gammaU(n, u);
}

fn gammaU(n: usize, unit: f64) f64 {
    const nu = @as(f64, @floatFromInt(n)) * unit;
    return nu / (1 - nu);
}

/// `tf-cuda-test exl3 [DIR]`: codebooks, tiles, FP64 bounds and row windows; DIR adds a checkpoint's real matrices.
pub fn run(gpu: Gpu, dir: ?[]const u8) !void {
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.exl3);
    defer mod.unload();
    var k = try E.Kernels.resolve(gpu.d, mod, k_max);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();
    try codebooks(gpu, &k, s);
    try tiles(gpu, &k, s);
    var b: Bound = .{};
    const cases = [_]Case{
        .{ .k = 256, .n = 256, .bits = 1, .book = .mul1 },   .{ .k = 256, .n = 384, .bits = 2, .book = .mcg },
        .{ .k = 384, .n = 256, .bits = 3, .book = .inst3 },  .{ .k = 256, .n = 256, .bits = 4, .book = .mul1 },
        .{ .k = 512, .n = 256, .bits = 5, .book = .mul1 },   .{ .k = 256, .n = 512, .bits = 6, .book = .mul1 },
        .{ .k = 256, .n = 256, .bits = 7, .book = .mcg },    .{ .k = 256, .n = 256, .bits = 8, .book = .mul1 },
        .{ .k = 1024, .n = 256, .bits = 4, .book = .inst3 }, .{ .k = 2048, .n = 128, .bits = 4, .book = .mul1 },
        .{ .k = 1024, .n = 256, .bits = 6, .book = .mcg },
    };
    for (cases, 0..) |c, i| try synthetic(gpu, &k, s, c, 0xE3 + i, &b);
    check.pass("{d} synthetic matrices, widths 1-8, three codebooks: {d} FP64 values, max error {e}, max {d:.4} of the bound, worst column {e} of the largest output", .{ cases.len, b.values, b.max_error, b.max_fraction, b.worst_column });
    try windows(gpu, &k, s);
    if (dir) |d| try real(gpu, &k, s, d);
}

fn codebooks(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream) !void {
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const out = try rig.zeros(65536 * 2);
    const got = try gpu.gpa.alloc(u16, 65536);
    defer gpu.gpa.free(got);
    for ([_]E.Codebook{ .inst3, .mcg, .mul1 }) |c| {
        try k.codebook(s, c, out);
        try s.synchronize();
        try rig.get(out, std.mem.sliceAsBytes(got));
        for (got, 0..) |g, i| try check.expect(g == @as(u16, @bitCast(fmt.codebookValue(refBook(c), @intCast(i)))), "exl3 {t} state {d}", .{ c, i });
    }
    check.pass("3 codebooks x 65536 states: the GPU's fp16 weights equal exl3_format.codebookValue bit for bit", .{});
}

fn tiles(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream) !void {
    var prng = std.Random.DefaultPrng.init(0x7E11);
    const r = prng.random();
    const tk = 2;
    const tn = 3;
    var count: usize = 0;
    for (1..9) |bits| for ([_]E.Codebook{ .inst3, .mcg, .mul1 }) |c| {
        var rig: Rig = .{ .gpu = gpu };
        defer rig.deinit();
        const words = try gpu.gpa.alloc(i16, tk * tn * 16 * bits);
        defer gpu.gpa.free(words);
        for (words) |*w| w.* = r.int(i16);
        const m: E.Matrix = .{ .trellis = try rig.put(words), .suh = 0, .svh = 0, .k = tk * 16, .n = tn * 16, .bits = @intCast(bits), .codebook = c };
        const out = try rig.zeros(m.k * m.n * 2);
        try k.unpack(s, m, out);
        try s.synchronize();
        var got: [tk * tn * 256]f16 = undefined;
        var want: [tk * tn * 256]f16 = undefined;
        try rig.get(out, std.mem.sliceAsBytes(&got));
        fmt.unpack(.{ .halves = @intCast(2 * bits) }, refBook(c), words, tk, tn, &want);
        try check.sameBytes("exl3 unpack", std.mem.sliceAsBytes(&got), std.mem.sliceAsBytes(&want));
        count += 1;
    };
    check.pass("{d} width/codebook pairs: random tiles unpack to exl3_format.unpack's W_q bit for bit", .{count});
}

pub const Case = struct { k: usize, n: usize, bits: u32, book: E.Codebook };

/// Worst observed error, as a value and as a fraction of the fp32 bound, over every value checked.
pub const Bound = struct {
    values: usize = 0,
    max_error: f64 = 0,
    max_fraction: f64 = 0,
    worst_column: f64 = 0,
};

/// One matrix's host copy (a column block of a real one, or a whole synthetic one) with its rows of input.
pub const Host = struct {
    k: usize,
    n: usize,
    bits: u32,
    book: E.Codebook,
    trellis: []const i16,
    suh: []const f16,
    svh: []const f16,
};

/// FP64 y = x W from exl3_format.dequantize, and per value the bound of exl3.cu's order (rotate, chain, transform).
pub fn reference(a: std.mem.Allocator, h: Host, x: []const u16, rows: usize, y: []f64, bound: []f64, prompt: bool) !void {
    const w = try a.alloc(f64, h.k * h.n);
    defer a.free(w);
    const scratch = try a.alloc(f64, 2 * h.k * h.n);
    defer a.free(scratch);
    fmt.dequantize(.{ .halves = @intCast(2 * h.bits) }, refBook(h.book), h.trellis, h.k / 16, h.n / 16, h.suh, h.svh, w, scratch);
    const wq = try a.alloc(f16, h.k * h.n);
    defer a.free(wq);
    fmt.unpack(.{ .halves = @intCast(2 * h.bits) }, refBook(h.book), h.trellis, h.k / 16, h.n / 16, wq);
    const rot = try a.alloc(f64, 2 * h.k);
    defer a.free(rot);
    const z = try a.alloc(f64, 2 * h.n);
    defer a.free(z);
    for (0..rows) |i| {
        const xr = x[i * h.k ..][0..h.k];
        for (0..h.n) |j| {
            var acc: f64 = 0;
            for (0..h.k) |l| acc += ofBf16(xr[l]) * w[l * h.n + j];
            y[i * h.n + j] = acc;
        }
        // r = (x suh) H unnormalized, |r| and its error from one product and seven adds
        const r = rot[0..h.k];
        const re = rot[h.k..];
        for (0..h.k) |l| r[l] = ofBf16(xr[l]) * @as(f64, @floatCast(h.suh[l]));
        var blk: usize = 0;
        while (blk < h.k) : (blk += 128) {
            var sum: f64 = 0;
            for (r[blk..][0..128]) |v| sum += @abs(v);
            fwht(r[blk..][0..128]);
            for (re[blk..][0..128], r[blk..][0..128]) |*e, v| {
                e.* = gamma(8) * sum;
                // prompt rows: r / 16 rounded to fp16 (relative 2^-11, 2^-25 absolute among subnormals)
                if (prompt) e.* += std.math.pow(f64, 2, -11) * (@abs(v) + e.*) + std.math.pow(f64, 2, -21);
            }
        }
        const za = z[0..h.n];
        const ze = z[h.n..];
        for (0..h.n) |j| {
            var mag: f64 = 0;
            var carried: f64 = 0;
            for (0..h.k) |l| {
                const q = @abs(@as(f64, @floatCast(wq[l * h.n + j])));
                mag += (@abs(r[l]) + re[l]) * q;
                carried += re[l] * q;
            }
            ze[j] = (if (prompt) gammaU(h.k + 3, std.math.pow(f64, 2, -22)) else gamma(h.k + 3)) * mag + carried;
            za[j] = mag;
        }
        var ob: usize = 0;
        while (ob < h.n) : (ob += 128) {
            var zsum: f64 = 0;
            var esum: f64 = 0;
            for (ob..ob + 128) |j| {
                zsum += za[j] + ze[j];
                esum += ze[j];
            }
            for (ob..ob + 128) |j| {
                const e = std.math.pow(f64, 2, -7) * @abs(@as(f64, @floatCast(h.svh[j]))) * (gamma(7) * zsum + esum);
                bound[i * h.n + j] = e * (1 + u) + (if (prompt) std.math.pow(f64, 2, -8) else u) * (@abs(y[i * h.n + j]) + e);
            }
        }
    }
}

fn fwht(v: []f64) void {
    var d: usize = 1;
    while (d < v.len) : (d <<= 1) {
        for (0..v.len) |i| {
            if (i & d != 0) continue;
            const a = v[i];
            const b = v[i + d];
            v[i] = a + b;
            v[i + d] = a - b;
        }
    }
}

/// Every value within its bound; the largest error by column against that column's largest output.
pub fn compare(what: []const u8, got: []const f32, y: []const f64, bound: []const f64, rows: usize, n: usize, b: *Bound) !void {
    var scale: f64 = 0;
    for (y[0 .. rows * n]) |v| scale = @max(scale, @abs(v));
    for (0..n) |j| {
        var col_err: f64 = 0;
        for (0..rows) |i| {
            const at = i * n + j;
            const err = @abs(@as(f64, got[at]) - y[at]);
            try check.expect(err <= bound[at], "{s}: row {d} column {d} error {e} over its bound {e}", .{ what, i, j, err, bound[at] });
            b.max_error = @max(b.max_error, err);
            if (bound[at] > 0) b.max_fraction = @max(b.max_fraction, err / bound[at]);
            col_err = @max(col_err, err);
        }
        if (scale > 0) b.worst_column = @max(b.worst_column, col_err / scale);
    }
    b.values += rows * n;
}

pub fn randomMatrix(a: std.mem.Allocator, r: std.Random, c: Case) !Host {
    const trellis = try a.alloc(i16, E.Matrix.trellisBytes(c.k, c.n, c.bits) / 2);
    for (trellis) |*w| w.* = r.int(i16);
    const suh = try a.alloc(f16, c.k);
    const svh = try a.alloc(f16, c.n);
    for (suh) |*v| v.* = @floatCast((0.25 + 1.75 * r.float(f32)) * @as(f32, if (r.boolean()) -1 else 1));
    for (svh) |*v| v.* = @floatCast((0.002 + 0.02 * r.float(f32)) * @as(f32, if (r.boolean()) -1 else 1));
    return .{ .k = c.k, .n = c.n, .bits = c.bits, .book = c.book, .trellis = trellis, .suh = suh, .svh = svh };
}

pub fn freeHost(a: std.mem.Allocator, h: Host) void {
    a.free(h.trellis);
    a.free(h.suh);
    a.free(h.svh);
}

pub fn randomRows(a: std.mem.Allocator, r: std.Random, rows: usize, k: usize) ![]u16 {
    const x = try a.alloc(u16, rows * k);
    for (x) |*v| v.* = bf16(r.floatNorm(f32) * 2);
    return x;
}

fn upload(rig: *Rig, h: Host) !E.Matrix {
    return .{ .trellis = try rig.put(h.trellis), .suh = try rig.put(h.suh), .svh = try rig.put(h.svh), .k = h.k, .n = h.n, .bits = h.bits, .codebook = h.book };
}

fn synthetic(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream, c: Case, seed: u64, b: *Bound) !void {
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(seed);
    const h = try randomMatrix(a, prng.random(), c);
    defer freeHost(a, h);
    const rows = 16;
    const x = try randomRows(a, prng.random(), rows, c.k);
    defer a.free(x);
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const m = try upload(&rig, h);
    const out = try rig.zeros(rows * c.n * 4);
    try k.rows(s, try rig.put(x), c.k, m, out, rows, true);
    try s.synchronize();
    const got = try a.alloc(f32, rows * c.n);
    defer a.free(got);
    try rig.get(out, std.mem.sliceAsBytes(got));
    const y = try a.alloc(f64, rows * c.n);
    defer a.free(y);
    const bound = try a.alloc(f64, rows * c.n);
    defer a.free(bound);
    try reference(a, h, x, rows, y, bound, true);
    try compare("exl3 synthetic", got, y, bound, rows, c.n, b);
    // the same rows as a prompt: the tensor units' arithmetic, bf16 out
    try k.prompt(s, try rig.put(x), c.k, m, out, rows);
    try s.synchronize();
    try getBf16(&rig, out, got);
    try reference(a, h, x, rows, y, bound, true);
    try compare("exl3 synthetic prompt", got, y, bound, rows, c.n, b);
}

fn getBf16(rig: *Rig, ptr: u64, out: []f32) !void {
    const raw = try rig.gpu.gpa.alloc(u16, out.len);
    defer rig.gpu.gpa.free(raw);
    try rig.get(ptr, std.mem.sliceAsBytes(raw));
    for (out, raw) |*o, v| o.* = @floatCast(ofBf16(v));
}

/// Each row alone, then every window 1-16 and a 40-row prompt through qlinear: each row's bf16 bytes as when alone.
fn windows(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream) !void {
    const a = gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x3D16);
    const h = try randomMatrix(a, prng.random(), .{ .k = 2048, .n = 256, .bits = 4, .book = .mul1 });
    defer freeHost(a, h);
    const rows = 40;
    const prompt_rows = 300; // past a 256-row chunk and a 64-row block
    const x = try randomRows(a, prng.random(), prompt_rows, h.k);
    defer a.free(x);
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const w: cuda.qlinear.Weight = .{ .exl3 = try upload(&rig, h) };
    const l: cuda.qlinear.Linear = .{ .exl3 = k };
    const xd = try rig.put(x);
    const out = try rig.zeros(prompt_rows * h.n * 2);
    const row_bytes = h.n * 2;
    const alone = try a.alloc(u8, rows * row_bytes);
    defer a.free(alone);
    const got = try a.alloc(u8, rows * row_bytes);
    defer a.free(got);
    for (0..rows) |i| {
        try l.decode(s, w, .{ .x = xd + i * h.k * 2 }, out, 1);
        try s.synchronize();
        try rig.get(out, alone[i * row_bytes ..][0..row_bytes]);
    }
    for (1..E.window + 1) |m| {
        try l.decode(s, w, .{ .x = xd }, out, m);
        try s.synchronize();
        try rig.get(out, got[0 .. m * row_bytes]);
        try check.sameBytes("exl3 decode window", got[0 .. m * row_bytes], alone[0 .. m * row_bytes]);
    }
    const p_alone = try a.alloc(u8, prompt_rows * row_bytes);
    defer a.free(p_alone);
    const p_got = try a.alloc(u8, prompt_rows * row_bytes);
    defer a.free(p_got);
    for (0..prompt_rows) |i| {
        try l.prompt(s, w, .{ .x = xd + i * h.k * 2 }, out, 1);
        try s.synchronize();
        try rig.get(out, p_alone[i * row_bytes ..][0..row_bytes]);
    }
    for ([_]usize{ 2, 63, 64, 65, 255, 256, 257, prompt_rows }) |m| {
        try l.prompt(s, w, .{ .x = xd }, out, m);
        try s.synchronize();
        try rig.get(out, p_got[0 .. m * row_bytes]);
        try check.sameBytes("exl3 prompt rows", p_got[0 .. m * row_bytes], p_alone[0 .. m * row_bytes]);
    }
    check.pass("decode windows of 1-16 rows ({d} input parts) and prompts of 2-{d} rows through qlinear: every row's bf16 bytes equal the row alone on its path", .{ E.split(h.k, h.n), prompt_rows });
}

/// The checkpoint's real matrices: FP64 on three column blocks, then GB/s at 1 and 16 rows.
fn real(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream, dir: []const u8) !void {
    const a = gpu.gpa;
    const index_path = try std.fmt.allocPrint(a, "{s}/model.safetensors.index.json", .{dir});
    defer a.free(index_path);
    const index_bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, index_path, a, .limited(64 << 20));
    defer a.free(index_bytes);
    const index = try std.json.parseFromSlice(std.json.Value, a, index_bytes, .{});
    defer index.deinit();
    const map = index.value.object.get("weight_map").?.object;
    const P = "model.language_model.layers.";
    const names = [_][]const u8{ P ++ "3.mlp.experts.0.gate_proj", P ++ "3.mlp.experts.0.down_proj", P ++ "0.mlp.down_proj", P ++ "3.mlp.shared_experts.up_proj", P ++ "3.self_attn.q_b_proj", P ++ "3.self_attn.o_proj", "lm_head" };
    var b: Bound = .{};
    for (names) |name| {
        const tname = try std.fmt.allocPrint(a, "{s}.trellis", .{name});
        defer a.free(tname);
        const shard = map.get(tname) orelse return error.TensorNotInIndex;
        const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, shard.string });
        defer a.free(path);
        var f = try st.File.open(a, gpu.io, path);
        defer f.close(gpu.io);
        try realOne(gpu, k, s, &f, name, &b);
    }
    check.pass("{d} real matrices: {d} FP64 values, max error {e}, max {d:.4} of the bound, worst column {e} of the largest output", .{ names.len, b.values, b.max_error, b.max_fraction, b.worst_column });
}

fn tensor(f: *const st.File, name: []const u8, part: []const u8) ?st.Tensor {
    var buf: [256]u8 = undefined;
    return f.get(std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, part }) catch return null);
}

fn realOne(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream, f: *const st.File, name: []const u8, b: *Bound) !void {
    const a = gpu.gpa;
    const t = tensor(f, name, "trellis") orelse return error.MissingTrellis;
    const suh_t = tensor(f, name, "suh") orelse return error.MissingSuh;
    const svh_t = tensor(f, name, "svh") orelse return error.MissingSvh;
    const book: E.Codebook = if (tensor(f, name, "mul1") != null) .mul1 else if (tensor(f, name, "mcg") != null) .mcg else .inst3;
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 16 != 0) return error.UnsupportedTrellis;
    const kk = t.dim(0) * 16;
    const n = t.dim(1) * 16;
    const bits: u32 = @intCast(t.dim(2) / 16);
    const words: []align(1) const i16 = std.mem.bytesAsSlice(i16, t.bytes);
    const suh: []align(1) const f16 = std.mem.bytesAsSlice(f16, suh_t.bytes);
    const svh: []align(1) const f16 = std.mem.bytesAsSlice(f16, svh_t.bytes);
    var rig: Rig = .{ .gpu = gpu };
    defer rig.deinit();
    const m: E.Matrix = .{ .trellis = try rig.put(t.bytes), .suh = try rig.put(suh_t.bytes), .svh = try rig.put(svh_t.bytes), .k = kk, .n = n, .bits = bits, .codebook = book };
    var prng = std.Random.DefaultPrng.init(0xF1A5);
    const rows = 4;
    const prompt_rows = 512;
    const x = try randomRows(a, prng.random(), prompt_rows, kk);
    defer a.free(x);
    const xd = try rig.put(x);
    const out = try rig.zeros(prompt_rows * n * 4);
    try k.rows(s, xd, kk, m, out, rows, true);
    try s.synchronize();
    const got = try a.alloc(f32, rows * n);
    defer a.free(got);
    try rig.get(out, std.mem.sliceAsBytes(got));
    // three 128-column blocks (first, middle, last), each a trellis of its own 8 tile columns
    const tile_words = 16 * @as(usize, bits);
    const sub = try a.alloc(i16, kk / 16 * 8 * tile_words);
    defer a.free(sub);
    const suh_c = try a.alloc(f16, kk);
    defer a.free(suh_c);
    for (suh_c, suh) |*d, v| d.* = v;
    const svh_c = try a.alloc(f16, 128);
    defer a.free(svh_c);
    const y = try a.alloc(f64, rows * 128);
    defer a.free(y);
    const bound = try a.alloc(f64, rows * 128);
    defer a.free(bound);
    const g = try a.alloc(f32, rows * 128);
    defer a.free(g);
    const prompt_got = try a.alloc(f32, rows * n);
    defer a.free(prompt_got);

    try k.prompt(s, xd, kk, m, out, rows);
    try s.synchronize();
    try getBf16(&rig, out, prompt_got);
    const blocks = n / 128;
    for ([_]usize{ 0, blocks / 2, blocks - 1 }) |ob| {
        for (0..kk / 16) |tk| for (0..8) |c| for (0..tile_words) |wi| {
            sub[(tk * 8 + c) * tile_words + wi] = words[(tk * (n / 16) + 8 * ob + c) * tile_words + wi];
        };
        for (svh_c, 0..) |*v, j| v.* = svh[ob * 128 + j];
        const sub_h: Host = .{ .k = kk, .n = 128, .bits = bits, .book = book, .trellis = sub, .suh = suh_c, .svh = svh_c };
        try reference(a, sub_h, x, rows, y, bound, true);
        for (0..rows) |i| @memcpy(g[i * 128 ..][0..128], got[i * n + ob * 128 ..][0..128]);
        try compare(name, g, y, bound, rows, 128, b);
        for (0..rows) |i| @memcpy(g[i * 128 ..][0..128], prompt_got[i * n + ob * 128 ..][0..128]);
        try compare(name, g, y, bound, rows, 128, b);
    }
    const one = try time(gpu, k, s, xd, m, out, 1);
    const sixteen = try time(gpu, k, s, xd, m, out, E.window);
    const many = try timePrompt(gpu, k, s, xd, m, out, prompt_rows);
    const gb = @as(f64, @floatFromInt(t.bytes.len)) / 1e9;
    check.pass("{s}: K={d} N={d} {d} bits {t}, 3 column blocks within FP64 bounds (decode and prompt); 1 row {d:.1} us ({d:.0} GB/s), 16 rows {d:.1} us, a {d}-row prompt {d:.0} rows/s", .{ name, kk, n, bits, book, one * 1e6, gb / one, sixteen * 1e6, prompt_rows, prompt_rows / many });
}

/// Seconds a call, the median of 7 runs of 20 after a warm-up, bf16 out.
fn time(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream, x: u64, m: E.Matrix, out: u64, rows: usize) !f64 {
    try k.rows(s, x, m.k, m, out, rows, false);
    try s.synchronize();
    var runs: [7]f64 = undefined;
    for (&runs) |*r| {
        const t0 = check.now(gpu.io);
        for (0..20) |_| try k.rows(s, x, m.k, m, out, rows, false);
        try s.synchronize();
        r.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 20e9;
    }
    return check.median(&runs);
}

/// Seconds a prompt call, the median of 7 runs of 5 after a warm-up.
fn timePrompt(gpu: Gpu, k: *const E.Kernels, s: cuda.Stream, x: u64, m: E.Matrix, out: u64, rows: usize) !f64 {
    try k.prompt(s, x, m.k, m, out, rows);
    try s.synchronize();
    var runs: [7]f64 = undefined;
    for (&runs) |*r| {
        const t0 = check.now(gpu.io);
        for (0..5) |_| try k.prompt(s, x, m.k, m, out, rows);
        try s.synchronize();
        r.* = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 5e9;
    }
    return check.median(&runs);
}
