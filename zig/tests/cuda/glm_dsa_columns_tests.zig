//! TF_GLM_DSA_COLUMNS: full-K half projections and their gather layout, byte-equal to replicated glm_mm output.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
pub const Gpu = check.Gpu;

const row_counts = [_]usize{ 1, 2, 4, 8, 16, 32 };
const shapes = [_][2]usize{ .{ 2048, 4096 }, .{ 4096, 1536 } };

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn copy(gpu: Gpu, s: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
    try gpu.d.check(gpu.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, s.handle), "cuMemcpyDtoDAsync");
}

fn runShape(gpu: Gpu, s: cuda.Stream, mk: *const glm.mm.Kernels, set: glm.glue.Set, n: usize, k: usize) !void {
    const a = gpu.gpa;
    const half = n / 2;
    const rows_max = row_counts[row_counts.len - 1];
    const hw = try a.alloc(u16, n * k);
    defer a.free(hw);
    const hx = try a.alloc(u16, rows_max * k);
    defer a.free(hx);
    var prng = std.Random.DefaultPrng.init(n * 131 + k);
    const rng = prng.random();
    for (hw) |*v| v.* = bf((rng.float(f32) - 0.5) / 16);
    for (hx) |*v| v.* = bf((rng.float(f32) - 0.5) / 16);

    var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hw));
    defer w.free();
    var wt = try cuda.DeviceBuffer.alloc(gpu.d, hw.len * 2);
    defer wt.free();
    try glm.mm.tile(mk, s, w.ptr, wt.ptr, n, k);
    const full: glm.mm.Tiled = .{ .w = wt.ptr, .n = n, .k = k };
    var locals: [2]glm.mm.Tiled = undefined;
    var local_w: [2]cuda.DeviceBuffer = undefined;
    var local_t: [2]cuda.DeviceBuffer = undefined;
    var local_count: usize = 0;
    defer for (0..local_count) |rank| {
        local_w[rank].free();
        local_t[rank].free();
    };
    for (0..2) |rank| {
        const bytes = std.mem.sliceAsBytes(hw[rank * half * k ..][0 .. half * k]);
        local_w[rank] = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
        local_t[rank] = cuda.DeviceBuffer.alloc(gpu.d, bytes.len) catch |e| {
            local_w[rank].free();
            return e;
        };
        local_count += 1;
        try glm.mm.tile(mk, s, local_w[rank].ptr, local_t[rank].ptr, half, k);
        locals[rank] = .{ .w = local_t[rank].ptr, .n = half, .k = k };
    }

    var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, rows_max * n * 2);
    defer reference.free();
    var gathered = try cuda.DeviceBuffer.alloc(gpu.d, rows_max * n * 2);
    defer gathered.free();
    var stage = try cuda.DeviceBuffer.alloc(gpu.d, rows_max * n * 2);
    defer stage.free();
    var shards: [2]cuda.DeviceBuffer = undefined;
    var shard_count: usize = 0;
    defer for (0..shard_count) |rank| shards[rank].free();
    for (0..2) |rank| {
        shards[rank] = try cuda.DeviceBuffer.alloc(gpu.d, rows_max * half * 2);
        shard_count += 1;
    }
    var part = try cuda.DeviceBuffer.alloc(gpu.d, @max(16, glm.mm.partBytes(rows_max, full)));
    defer part.free();
    const want = try a.alloc(u8, rows_max * n * 2);
    defer a.free(want);
    const got = try a.alloc(u8, rows_max * n * 2);
    defer a.free(got);
    defer s.synchronize() catch @panic("GLM DSA column contract retains device work until completion");

    for (row_counts) |rows| {
        try glm.mm.matmul(mk, s, x.ptr, k, full, reference.ptr, part.ptr, rows, false, false);
        if (rows == 1) {
            for (0..2) |rank| try glm.mm.matmul(mk, s, x.ptr, k, locals[rank], gathered.ptr + rank * half * 2, part.ptr, rows, false, false);
        } else {
            for (0..2) |rank| {
                try glm.mm.matmul(mk, s, x.ptr, k, locals[rank], shards[rank].ptr, part.ptr, rows, false, false);
                try copy(gpu, s, stage.ptr + rank * rows * half * 2, shards[rank].ptr, rows * half * 2);
            }
            try glm.dsa_block.unpackColumns(set, s, stage.ptr, gathered.ptr, half, n, rows, 2);
        }
        try s.synchronize();
        try reference.download(0, want[0 .. rows * n * 2]);
        try gathered.download(0, got[0 .. rows * n * 2]);
        try check.sameBytes("glm-dsa-columns-contract: full projection versus two whole-K shards and gather", want[0 .. rows * n * 2], got[0 .. rows * n * 2]);
    }
}

pub fn run(gpu: Gpu) !void {
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var mm_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer mm_mod.unload();
    var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
    defer ops.unload();
    const mk = try glm.mm.Kernels.load(mm_mod);
    const glue: glm.glue.Kernels = undefined;
    const set: glm.glue.Set = .{ .glue = &glue, .copy_rows = try ops.function("tf_glm_copy_rows") };
    for (shapes) |shape| try runShape(gpu, s, &mk, set, shape[0], shape[1]);
    check.pass("glm-dsa-columns-contract: proj and qb, rows 1/2/4/8/16/32, every BF16 byte identical", .{});
}
