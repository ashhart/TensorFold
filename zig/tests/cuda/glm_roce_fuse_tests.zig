//! TP2 GLM fusion against a real all-gather and standalone post kernel: `tf-cuda-test glm-roce-fuse <rank>`.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

const d = 4096;
const rows_cases = [_]usize{ 1, 2, 4, 8, 16 };
const max_rows = 16;
const max_partials = 2 * max_rows * d * 4;
const max_x = max_rows * 4 * d * 2;
const special = [_]u32{
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // positive subnormal
    0x8000_0001, // negative subnormal
    0x0080_0000, // minimum positive normal
    0x8080_0000, // minimum magnitude normal
    0x7f80_0000, // +infinity
    0xff80_0000, // -infinity
    0x7f7f_ffff, // maximum finite magnitude
    0xff7f_ffff, // maximum negative finite magnitude
    0x7fc0_0001, // quiet NaN with a payload
    0x7f81_0000, // another NaN payload
    0x0000_3f80, // paired BF16 finite values in the word tests
    0x8000_bf80,
};

/// Raw words, not host floating-point arithmetic: both BF16 streams and FP32 coefficients see edge encodings.
fn fill(host: []u8, seed: u64, rank: usize) void {
    var rng = std.Random.DefaultPrng.init(seed *% 1_000_003 +% rank *% 7_919 +% host.len);
    rng.random().bytes(host);
    const words = std.mem.bytesAsSlice(u32, @as([]align(1) u8, host));
    for (special, 0..) |word, i| {
        if (i < words.len) words[i] = word;
    }
}

fn standalone(kind: usize, f: cuda.Function, s: cuda.Stream, x: u64, out: u64, gathered: u64, post: u64, comb: u64, rows: usize) !void {
    var a: cuda.Args = .{};
    a.add(x);
    a.add(out);
    a.add(gathered);
    if (kind == 1) {
        a.add(post);
        a.add(comb);
    }
    a.add(@as(i64, @intCast(rows * d)));
    a.add(@as(i64, @intCast(d)));
    a.add(@as(c_int, 2));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d + 1023) / 1024) }, .block = .{ .x = 128 } }, s, &a);
}

pub fn run(gpu: check.Gpu, rank: usize) !void {
    const a = gpu.gpa;
    try check.expect(std.c.getenv("NCCL_COMM_ID") != null, "glm-roce-fuse: NCCL_COMM_ID names rank 0's address", .{});
    try check.expect(rank < 2, "glm-roce-fuse: rank {d} of world 2", .{rank});
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.tp_group.Group.open(gpu.d, rank, 2);
    defer g.close();
    try g.openRoce(a, gpu.io, gpu.d, s, max_partials / 2);
    try check.expect(g.fast != null, "glm-roce-fuse: the TP2 RoCE path did not come up", .{});
    var module = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer module.unload();
    const reference = [_]cuda.Function{
        try module.function("tf_glm_residual_add"),
        try module.function("tf_glm_hc_post"),
    };

    var partials = try cuda.DeviceBuffer.alloc(gpu.d, max_partials);
    defer partials.free();
    var x = try cuda.DeviceBuffer.alloc(gpu.d, max_x);
    defer x.free();
    var post = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * 4 * 4);
    defer post.free();
    var comb = try cuda.DeviceBuffer.alloc(gpu.d, max_rows * 16 * 4);
    defer comb.free();
    const partial_host = try a.alloc(u8, max_partials / 2);
    defer a.free(partial_host);
    const x_host = try a.alloc(u8, max_x);
    defer a.free(x_host);
    const post_host = try a.alloc(u8, max_rows * 4 * 4);
    defer a.free(post_host);
    const comb_host = try a.alloc(u8, max_rows * 16 * 4);
    defer a.free(comb_host);
    const fused_host = try a.alloc(u8, max_x);
    defer a.free(fused_host);
    const ordinary_host = try a.alloc(u8, max_x);
    defer a.free(ordinary_host);

    var cells: usize = 0;
    for (rows_cases) |rows| for (1..4) |seed| {
        const bytes = rows * d * 4;
        const x_bytes = rows * 4 * d * 2;
        fill(partial_host[0..bytes], seed, rank);
        fill(x_host[0..x_bytes], seed + 10, 0);
        fill(post_host[0 .. rows * 4 * 4], seed + 20, 0);
        fill(comb_host[0 .. rows * 16 * 4], seed + 30, 0);
        try partials.upload(rank * bytes, partial_host[0..bytes]);
        try x.upload(0, x_host[0..x_bytes]);
        try post.upload(0, post_host[0 .. rows * 4 * 4]);
        try comb.upload(0, comb_host[0 .. rows * 16 * 4]);
        for (0..2) |kind| {
            try x.upload(0, x_host[0..x_bytes]); // both epilogues write x in place; each starts from identical bytes
            const out_bytes = if (kind == 0) rows * d * 2 else x_bytes;
            const epilogue: cuda.roce.Post = if (kind == 0)
                .{ .residual_add = .{ .x = x.ptr, .xout = x.ptr } }
            else
                .{ .hc_post = .{ .x = x.ptr, .xout = x.ptr, .post = post.ptr, .comb = comb.ptr } };
            try check.expect(try g.gatherPost(s, partials.ptr + rank * bytes, partials.ptr, bytes, d, epilogue), "glm-roce-fuse: an eligible TP2 cell did not fuse", .{});
            try s.synchronize();
            try g.check();
            try x.download(0, fused_host[0..out_bytes]);

            try x.upload(0, x_host[0..x_bytes]); // the reference must see the source before the fused in-place stores
            try g.allGather(s, partials.ptr + rank * bytes, partials.ptr, bytes);
            try standalone(kind, reference[kind], s, x.ptr, x.ptr, partials.ptr, post.ptr, comb.ptr, rows);
            try s.synchronize();
            try g.check();
            try x.download(0, ordinary_host[0..out_bytes]);
            try check.sameBytes("GLM fused RoCE post against all-gather plus the standalone kernel", fused_host[0..out_bytes], ordinary_host[0..out_bytes]);
            cells += 1;
            check.pass("glm-roce-fuse rank {d}: {d} rows, epilogue {d}, seed {d}, {d} output bytes equal", .{ rank, rows, kind, seed, out_bytes });
        }
    };
    std.debug.print("RESULT glm-roce-fuse rank {d}: {d} fused/unfused cells, 0 byte mismatches\n", .{ rank, cells });
}
