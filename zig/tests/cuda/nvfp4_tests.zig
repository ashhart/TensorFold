//! NVFP4 against recorded fixture bytes: the repack, then lane and prompt rows, direct and via qlinear.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const nvfp4 = cuda.nvfp4;
const qmmf = cuda.qmmf;

pub fn run(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const n: usize = @intCast(try fx.int("n"));
    const k: usize = @intCast(try fx.int("k"));
    const scale: f32 = @floatCast(try fx.float("global_scale"));
    const npad = nvfp4.padded(n);

    const codes = try fx.bytes("codes");
    defer a.free(codes);
    const scales = try fx.bytes("scales");
    defer a.free(scales);
    const words = try a.alloc(u32, npad / 64 * (k / nvfp4.group) * 512);
    defer a.free(words);
    nvfp4.packWords(words, codes, n, k);
    const want_words = try fx.bytes("words");
    defer a.free(want_words);
    try check.sameBytes("packed words against qmm.pack", std.mem.sliceAsBytes(words), want_words);
    const bs = try a.alloc(u8, npad * (k / 16));
    defer a.free(bs);
    nvfp4.packScales(bs, scales, n, k);
    const want_bs = try fx.bytes("bs");
    defer a.free(want_bs);
    try check.sameBytes("scale tiles against Fp4Linear.from_checkpoint", bs, want_bs);
    check.pass("nvfp4 repack: [{d}, {d}] words and scale tiles equal the Python layout", .{ n, k });

    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, bs);
    defer ds.free();
    const w = nvfp4.weight(dw.ptr, ds.ptr, scale, n, k);
    const cap = try gpu.ctx.capability();
    var lane = try qmmf.Lane.load(gpu.d, cap / 10);
    defer lane.unload();
    var prompt = try nvfp4.Prompt.load(gpu.d);
    defer prompt.unload();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    const lin: cuda.qlinear.Linear = .{ .lane = &lane, .nvfp4_prompt = &prompt };

    for ([_][]const u8{ "lane", "prompt" }) |path| {
        var it = std.mem.tokenizeScalar(u8, try fx.string(path), ',');
        while (it.next()) |tok| {
            const m = try std.fmt.parseInt(usize, tok, 10);
            var name: [24]u8 = undefined;
            const x = try fx.bytes(try std.fmt.bufPrint(&name, "x{d}", .{m}));
            defer a.free(x);
            const want = try fx.bytes(try std.fmt.bufPrint(&name, "{s}{d}", .{ path, m }));
            defer a.free(want);
            var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
            defer dx.free();
            var dy = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
            defer dy.free();
            var dp = try cuda.DeviceBuffer.alloc(gpu.d, @max(lane.partBytes(m, w), 4));
            defer dp.free();
            const decode = std.mem.eql(u8, path, "lane");
            for ([_]bool{ false, true }) |via| {
                try dy.fill8(0xff, stream.handle); // a call that writes nothing fails rather than pass on old bytes
                const in: cuda.qlinear.Rows = .{ .x = dx.ptr, .ldx = k, .part = dp.ptr };
                if (via) {
                    if (decode) try lin.decode(stream, .{ .nvfp4 = w }, in, dy.ptr, m) else try lin.prompt(stream, .{ .nvfp4 = w }, in, dy.ptr, m);
                } else if (decode) try lane.matmul(stream, dx.ptr, k, m, w, dy.ptr, dp.ptr) else try prompt.matmul(stream, dx.ptr, k, m, w, dy.ptr);
                try stream.synchronize();
                const got = try check.download(gpu, dy);
                defer a.free(got);
                try check.sameBytes(try std.fmt.bufPrint(&name, "{s}{d}", .{ path, m }), got, want);
            }
            check.pass("nvfp4 {s}: {d} rows x [{d}, {d}] equal Python's bytes, direct and via qlinear", .{ path, m, n, k });
        }
    }
}
