//! TF_Q35_MLP_ACT_FUSED on the GPU: the fused NVFP4 up write-out against up then tf_q35_mlp_act, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const qmmf = cuda.qmmf;
const act_ = cuda.qmmf_act;

fn tobf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

/// The value sets: random spreads, or every edge a bf16 row and an NVFP4 weight can carry.
const Fill = enum { normal, adversarial };

/// bf16 rows: a spread, or signed zeros, subnormals, max-magnitude and infinities among them.
fn rows(a: std.mem.Allocator, rng: std.Random, n: usize, fill: Fill) ![]u16 {
    const out = try a.alloc(u16, n);
    const edges = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x3F80, 0xBF80 };
    for (out, 0..) |*v, i| v.* = if (fill == .adversarial and i % 7 == 0) edges[rng.uintLessThan(usize, edges.len)] else tobf((rng.float(f32) - 0.5) * 8);
    return out;
}

/// An NVFP4 lane weight [n, k]: random e2m1 codes, e4m3 block scales (adversarial: zero, subnormal and max among them).
const Packed = struct {
    words: cuda.DeviceBuffer,
    tiles: cuda.DeviceBuffer,
    w: qmmf.Weight,

    fn init(gpu: Gpu, rng: std.Random, n: usize, k: usize, scale: f32, fill: Fill) !Packed {
        const a = gpu.gpa;
        const np = cuda.modelopt.padded(n);
        const codes = try a.alloc(u8, n * k / 2);
        defer a.free(codes);
        rng.bytes(codes);
        const scales = try a.alloc(u8, n * k / 16);
        defer a.free(scales);
        const edges = [_]u8{ 0x00, 0x01, 0x07, 0x7E, 0x38 };
        for (scales, 0..) |*b, i| b.* = if (fill == .adversarial and i % 5 == 0) edges[rng.uintLessThan(usize, edges.len)] else 0x18 + rng.uintLessThan(u8, 0x30);
        const words = try a.alloc(u32, np / 64 * (k / 64) * 512);
        defer a.free(words);
        @memset(words, 0);
        cuda.nvfp4.packWords(words, codes, n, k);
        const tiles = try a.alloc(u8, np * (k / 16));
        defer a.free(tiles);
        @memset(tiles, 0);
        cuda.nvfp4.packScales(tiles, scales, n, k);
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
        errdefer dw.free();
        const dt = try cuda.DeviceBuffer.fromHost(gpu.d, tiles);
        return .{ .words = dw, .tiles = dt, .w = .{ .mode = .fp4, .codes = dw.ptr, .scales = dt.ptr, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(np) } };
    }

    fn free(p: *Packed) void {
        p.words.free();
        p.tiles.free();
    }
};

pub fn run(gpu: Gpu) !void {
    const capability = try gpu.ctx.capability();
    if (capability / 10 < 8) {
        std.debug.print("SKIP mlp-act-fused: the lane matmul needs sm_80\n", .{});
        return;
    }
    var lane = try qmmf.Lane.load(gpu.d, capability / 10);
    defer lane.unload();
    const fused = lane.up_act orelse return error.NoFusedUpKernel;
    var q35 = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
    defer q35.unload();
    const mlp_act = try q35.function("tf_q35_mlp_act");
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0xac7_f05e);
    const rng = prng.random();
    // the 27B's up [17408, 5120], then narrower and odd tile counts; global scales plain, tiny, subnormal and infinite
    const shapes = [_][2]usize{ .{ 17408, 5120 }, .{ 192, 320 }, .{ 448, 64 }, .{ 64, 128 } };
    const scales = [_]f32{ 1.0, 0.000244140625, 1.0e-40, std.math.inf(f32) };
    var compared: usize = 0;
    for (shapes, 0..) |nk, si| for ([_]Fill{ .normal, .adversarial }) |fill| {
        const n = nk[0];
        const k = nk[1];
        var gate = try Packed.init(gpu, rng, n, k, scales[si % scales.len], fill);
        defer gate.free();
        var up = try Packed.init(gpu, rng, n, k, scales[(si + 1) % scales.len], fill);
        defer up.free();
        for (1..17) |m| {
            if (!act_.eligible(lane, m, up.w)) continue; // today's plan for this shape is not the copied instance
            const x = try rows(gpu.gpa, rng, m * k, fill);
            defer gpu.gpa.free(x);
            for ([_]bool{ false, true }) |params| {
                try compare(gpu, lane, stream, mlp_act, fused, gate.w, up.w, x, m, params);
                compared += 1;
            }
        }
    };
    if (compared == 0) return error.NothingEligible;
    check.pass("mlp act fused: {d} NVFP4 up write-outs (rows 1-16, 4 shapes, adversarial values, P on and off) equal up then tf_q35_mlp_act, every byte and guard", .{compared});
}

/// One case: the gate, then up and tf_q35_mlp_act against the fused up act kernel; both acts compared whole.
fn compare(gpu: Gpu, lane: qmmf.Lane, s: cuda.Stream, mlp_act: cuda.Function, fused: cuda.Function, gw: qmmf.Weight, uw: qmmf.Weight, x: []const u16, m: usize, with_params: bool) !void {
    const n = uw.n;
    const bytes = m * n * 2 + 256; // a guard past the rows
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    const guard = try gpu.gpa.alloc(u8, bytes);
    defer gpu.gpa.free(guard);
    @memset(guard, 0xA5);
    var gate_out = try cuda.DeviceBuffer.fromHost(gpu.d, guard);
    defer gate_out.free();
    var up_out = try cuda.DeviceBuffer.fromHost(gpu.d, guard);
    defer up_out.free();
    var want = try cuda.DeviceBuffer.fromHost(gpu.d, guard);
    defer want.free();
    var got = try cuda.DeviceBuffer.fromHost(gpu.d, guard);
    defer got.free();
    // P[2]: the pass's rows; one fewer than the launch's, so both leave the last row's guard untouched
    const p = [3]i32{ 0, 0, @intCast(m - 1) };
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&p));
    defer dp.free();
    const params: u64 = if (with_params) dp.ptr else 0;
    try lane.matmulOut(s, dx.ptr, gw.k, m, gw, gate_out.ptr, .bf16, 0);
    try lane.matmulOut(s, dx.ptr, uw.k, m, uw, up_out.ptr, .bf16, 0);
    var args: cuda.Args = .{};
    args.add(gate_out.ptr);
    args.add(up_out.ptr);
    args.add(want.ptr);
    args.add(@as(i64, @intCast(m * n)));
    args.add(params);
    args.add(@as(i32, @intCast(n)));
    args.add(@as(u64, 0));
    try cuda.launch.launch(mlp_act, .{ .grid = .{ .x = @intCast((m * n + 1023) / 1024), .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &args);
    try act_.launch(fused, lane, s, dx.ptr, uw.k, m, uw, gate_out.ptr, got.ptr, params);
    try s.synchronize();
    const a = try check.download(gpu, want);
    defer gpu.gpa.free(a);
    const b = try check.download(gpu, got);
    defer gpu.gpa.free(b);
    try check.sameBytes("mlp act fused against up then tf_q35_mlp_act", b, a);
}
