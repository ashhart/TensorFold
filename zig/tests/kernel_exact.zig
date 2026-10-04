//! Bit-exact check of one kernel against an MLX oracle case, dispatched the way mx.fast.metal_kernel dispatches it.
const std = @import("std");
const mtl = @import("metal");
const common = @import("common.zig");

const Binding = struct { index: usize, file: []const u8, stride: usize, bytes: usize = 0 };

const Case = struct {
    function: []const u8 = "",
    grid: [3]usize = .{ 1, 1, 1 },
    group: [3]usize = .{ 1, 1, 1 },
    trials: usize = 1,
    inputs: [16]Binding = undefined,
    input_count: usize = 0,
    output: Binding = undefined,
};

fn parse(text: []const u8) !Case {
    var case = Case{};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var words = std.mem.tokenizeScalar(u8, line, ' ');
        const key = words.next() orelse continue;
        if (std.mem.eql(u8, key, "function")) {
            case.function = words.next() orelse return error.BadManifest;
        } else if (std.mem.eql(u8, key, "grid") or std.mem.eql(u8, key, "group")) {
            const dims = if (std.mem.eql(u8, key, "grid")) &case.grid else &case.group;
            for (dims) |*d| d.* = try std.fmt.parseInt(usize, words.next() orelse return error.BadManifest, 10);
        } else if (std.mem.eql(u8, key, "trials")) {
            case.trials = try std.fmt.parseInt(usize, words.next() orelse return error.BadManifest, 10);
        } else if (std.mem.eql(u8, key, "buffer") or std.mem.eql(u8, key, "output")) {
            var b = Binding{
                .index = try std.fmt.parseInt(usize, words.next() orelse return error.BadManifest, 10),
                .file = words.next() orelse return error.BadManifest,
                .stride = try std.fmt.parseInt(usize, words.next() orelse return error.BadManifest, 10),
            };
            if (words.next()) |n| b.bytes = try std.fmt.parseInt(usize, n, 10);
            if (key[0] == 'o') case.output = b else {
                if (case.input_count == case.inputs.len) return error.BadManifest;
                case.inputs[case.input_count] = b;
                case.input_count += 1;
            }
        }
    }
    if (case.function.len == 0 or case.input_count == 0) return error.BadManifest;
    return case;
}

fn library(gpu: common.Gpu, args: []const [:0]const u8) !?mtl.Library {
    var math: mtl.types.MathMode = .safe;
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--math")) math = std.meta.stringToEnum(mtl.types.MathMode, args[i + 1]) orelse return error.BadMathMode;
    }
    i = 2;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--metallib")) return try mtl.Library.fromFile(gpu.device, args[i + 1]);
        if (std.mem.eql(u8, args[i], "--source")) {
            const src = try mtl.MappedFile.open(args[i + 1]);
            defer src.deinit();
            var options = mtl.CompileOptions.mlx();
            options.math_mode = math;
            return try mtl.Library.fromSource(gpu.device, src.bytes[0..src.size], options);
        }
    }
    return null;
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        std.debug.print("usage: tf-kernel-exact CASE_DIR [--metallib FILE | --source FILE [--math safe|relaxed|fast]]\n", .{});
        std.process.exit(2);
    }
    const dir = args[1];
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gpu = try common.Gpu.init();
    defer gpu.deinit();

    const manifest = try mtl.MappedFile.open(try arena.printSentinel("{s}/manifest.txt", .{dir}, 0));
    defer manifest.deinit();
    const case = try parse(manifest.bytes[0..manifest.size]);

    const own = try library(gpu, args);
    defer if (own) |l| l.deinit();
    const pipeline = try mtl.Pipeline.init(gpu.device, own orelse gpu.lib, case.function, false);
    defer pipeline.deinit();
    const group = mtl.Size.of(@min(case.group[0], case.grid[0]), @min(case.group[1], case.grid[1]), @min(case.group[2], case.grid[2]));
    if (group.width * group.height * group.depth > pipeline.maxThreads()) return error.ThreadgroupTooLarge;

    // every input file mapped once and wrapped with no copy (weights included)
    var maps: [16]mtl.MappedFile = undefined;
    var bufs: [16]mtl.Buffer = undefined;
    for (case.inputs[0..case.input_count], 0..) |b, k| {
        maps[k] = try mtl.MappedFile.open(try arena.printSentinel("{s}/{s}", .{ dir, b.file }, 0));
        bufs[k] = try gpu.device.bufferNoCopy(maps[k].bytes.ptr, maps[k].bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    }
    defer for (0..case.input_count) |k| {
        bufs[k].deinit();
        maps[k].deinit();
    };
    const expected = try mtl.MappedFile.open(try arena.printSentinel("{s}/{s}", .{ dir, case.output.file }, 0));
    defer expected.deinit();
    const out = try gpu.device.buffer(case.trials * case.output.stride, mtl.ResourceOptions.shared);
    defer out.deinit();
    @memset(out.slice(u8, case.trials * case.output.stride), 0xff);

    // the command buffer lives in its own pool, released before the mapped files are unmapped
    const gpu_ms = blk: {
        const inner = mtl.objc.Pool.push();
        defer inner.pop();
        const cb = gpu.queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pipeline);
        for (0..case.trials) |t| {
            for (case.inputs[0..case.input_count], 0..) |b, k| enc.setBuffer(bufs[k], t * b.stride, b.index);
            enc.setBuffer(out, t * case.output.stride, case.output.index);
            enc.dispatchThreads(mtl.Size.of(case.grid[0], case.grid[1], case.grid[2]), group);
        }
        enc.end();
        try common.run(cb);
        break :blk cb.gpuSeconds() * 1e3;
    };

    // compare as bf16 words over each trial's written bytes (the strides' padding is not output)
    const bytes = if (case.output.bytes != 0) case.output.bytes else case.output.stride;
    if (expected.size < case.trials * case.output.stride) return error.ShortExpected;
    var words: usize = 0;
    var differ: usize = 0;
    var first: ?[2]u16 = null;
    var first_at: usize = 0;
    for (0..case.trials) |t| {
        const at = t * case.output.stride / 2;
        const got = out.slice(u16, at + bytes / 2)[at..];
        const want = @as([*]const u16, @ptrCast(expected.bytes.ptr))[at .. at + bytes / 2];
        for (got, want, 0..) |g, w, i| if (g != w) {
            differ += 1;
            if (first == null) {
                first = .{ g, w };
                first_at = words + i;
            }
        };
        words += bytes / 2;
    }
    std.debug.print("{s}: {d} trials, {d} bf16 words, {d} differ from MLX", .{ dir, case.trials, words, differ });
    if (first) |f| std.debug.print(" (first at word {d}: 0x{x:0>4} vs MLX 0x{x:0>4})", .{ first_at, f[0], f[1] });
    std.debug.print("; GPU {d:.3} ms\n", .{gpu_ms});
    if (differ != 0) std.process.exit(1);
}
