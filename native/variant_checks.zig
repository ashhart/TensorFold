//! Replay original kernel launches using embedded sources and compare raw output bits.
const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Parameter = struct {
    name: [:0]const u8,
    kind: enum { integer, boolean, dtype },
    integer: i32 = 0,
    boolean: bool = false,
    dtype: []const u8 = "",
};
const Case = struct {
    name: []const u8,
    kernel: []const u8,
    source_sha256: []const u8,
    @"test": []const u8,
    templates: []const Parameter,
    grid: [3]i32,
    group: [3]i32,
    input_count: usize,
    output_count: usize,
};
fn find(name: []const u8) !src.Spec {
    inline for (comptime std.meta.declarations(src)) |decl| {
        const value = @field(src, decl);
        if (@TypeOf(value) == src.Spec) {
            if (std.mem.eql(u8, value.name, name)) return value;
        }
    }
    return error.UnknownKernelFixture;
}
fn dtype(name: []const u8) !mx.c.mlx_dtype {
    const names = .{ "bfloat16", "float16", "float32", "int32", "uint32", "uint16", "uint8", "int64", "uint64", "bool_" };
    const types = .{ mx.bf16, mx.c.MLX_FLOAT16, mx.f32t, mx.i32t, mx.c.MLX_UINT32, mx.c.MLX_UINT16, mx.c.MLX_UINT8, mx.c.MLX_INT64, mx.c.MLX_UINT64, mx.c.MLX_BOOL };
    inline for (names, types) |label, value| if (std.mem.eql(u8, name, label)) return value;
    return error.UnsupportedFixtureDType;
}
fn raw(s: *mx.Scope, value: mx.Array) !mx.Array {
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_view(&out, try s.contiguous(try s.reshape(value, &.{-1})), mx.c.MLX_UINT8, mx.stream);
    return s.result(rc, out);
}
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    var covered = std.StringHashMap(usize).init(mx.allocator);
    defer covered.deinit();
    for (cases.value) |case| {
        errdefer std.debug.print("Variant failure: {s}: {s} ({s})\n", .{ case.name, case.kernel, case.@"test" });
        const spec = try find(case.kernel);
        if (case.input_count != spec.inputs.len or case.output_count != spec.outputs.len) return error.InvalidKernelArity;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(spec.header);
        hash.update(&.{0});
        hash.update(spec.source);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), case.source_sha256)) return error.KernelSourceMismatch;
        var weights = @import("checkpoint.zig").Store.init(64);
        defer weights.deinit();
        try weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var scope = mx.Scope{};
        defer scope.deinit();
        const inputs = try mx.allocator.alloc(mx.Array, case.input_count);
        defer mx.allocator.free(inputs);
        const outputs = try mx.allocator.alloc(mx.Output, case.output_count);
        defer mx.allocator.free(outputs);
        const expected = try mx.allocator.alloc(mx.Array, case.output_count);
        defer mx.allocator.free(expected);
        const results = try mx.allocator.alloc(mx.Array, case.output_count);
        defer mx.allocator.free(results);
        const params = try mx.allocator.alloc(mx.Template, case.templates.len);
        defer mx.allocator.free(params);
        for (inputs, 0..) |*input, i| input.* = try weights.get(try std.fmt.bufPrint(&path, "input{d}", .{i}));
        for (expected, outputs, 0..) |*value, *output, i| {
            value.* = try weights.get(try std.fmt.bufPrint(&path, "output{d}", .{i}));
            output.* = .{ .shape = mx.shape(value.*), .dtype = mx.dtype(value.*) };
        }
        for (params, case.templates) |*param, value| param.* = switch (value.kind) {
            .integer => mx.ti(value.name, value.integer),
            .boolean => mx.tb(value.name, value.boolean),
            .dtype => mx.td(value.name, try dtype(value.dtype)),
        };
        try kernels.runInto(&scope, spec, inputs, params, case.grid, case.group, outputs, results, 0);
        if (std.mem.eql(u8, case.kernel, "lane_qmm_lowbit") or std.mem.eql(u8, case.kernel, "lane_qmm_bytes")) {
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const bits = parameter(case, "BITS");
            const groups = @divExact(width, 64);
            const words = @divExact(64 * bits, 32);
            const w = if (parameter(case, "TILED") == 1) try scope.contiguous(try scope.reshape(try scope.transpose(try scope.reshape(inputs[2], &.{ @divExact(n, 32), groups, 32, words }), &.{ 0, 2, 1, 3 }), &.{ n, groups * words })) else inputs[2];
            const sb = try scope.transpose(inputs[3], &.{ 1, 0, 2 });
            const sc = try scope.reshape(try scope.slice(sb, 2, 0, 1), &.{ n, groups });
            const bs = try scope.reshape(try scope.slice(sb, 2, 1, 2), &.{ n, groups });
            var linear = try @import("lanes.zig").Linear.initFormat(&scope, w, sc, bs, .{ .bits = bits });
            defer linear.deinit();
            // Only default launch reductions are the production dispatch contract.
            var split: i32 = 1;
            while (split < 8 and @divTrunc(n + 31, 32) * split < 1024 and @divTrunc(groups, split * 2) >= 8) split *= 2;
            if (split == parameter(case, "SK")) try equalBits(&scope, try linear.apply(&kernels, &scope, .{ .x = inputs[0], .sums = inputs[1] }), expected[0]);
        }
        if (std.mem.eql(u8, case.kernel, "simd_qmm_mma")) {
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const group = parameter(case, "GS");
            const split: i32 = if (n <= 64) 32 else if (n <= 6144) 16 else 8;
            if (parameter(case, "S") == split and mx.dim(inputs[0], 0) <= 128) {
                const was_tensor = mx.tensor_units;
                mx.tensor_units = false;
                defer mx.tensor_units = was_tensor;
                var linear = try @import("lanes.zig").Linear.initFormat(&scope, try matrix(&scope, inputs[1], n, @divExact(width, 8)), try matrix(&scope, inputs[2], n, @divExact(width, group)), try matrix(&scope, inputs[3], n, @divExact(width, group)), .{ .bits = 4, .group_size = group });
                defer linear.deinit();
                try equalBits(&scope, try linear.apply(&kernels, &scope, .{ .x = inputs[0] }), expected[0]);
            }
        }
        if (std.mem.eql(u8, case.kernel, "affine_rows")) {
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const format = @import("quantization.zig").Spec{ .bits = parameter(case, "BITS"), .group_size = parameter(case, "GS") };
            const w = try matrix(&scope, inputs[1], n, @divExact(width * format.bits, 32));
            const sc = try matrix(&scope, inputs[2], n, @divExact(width, format.group_size));
            const bs = try matrix(&scope, inputs[3], n, @divExact(width, format.group_size));
            var linear = try @import("lanes.zig").Linear.initFormat(&scope, w, sc, bs, format);
            defer linear.deinit();
            // The pre-existing 4/64 path has its own lane/SIMD arithmetic.
            if (linear.generic) {
                const projected = try linear.apply(&kernels, &scope, .{ .x = inputs[0] });
                try equalBits(&scope, projected, expected[0]);
                var selected = try linear.selectRanges(&scope, &.{.{ 0, @min(n, 3) }});
                defer selected.deinit();
                try equalBits(&scope, try selected.apply(&kernels, &scope, .{ .x = inputs[0] }), try scope.slice(expected[0], 1, 0, @min(n, 3)));
            }
        }
        for (results, expected, 0..) |result, want, i| {
            const a = try raw(&scope, result);
            const b = try raw(&scope, want);
            try mx.evalMany(&.{ a, b }, false);
            const n = mx.c.mlx_array_size(a);
            if (n != mx.c.mlx_array_size(b) or !std.mem.eql(u8, mx.c.mlx_array_data_uint8(a)[0..n], mx.c.mlx_array_data_uint8(b)[0..n])) {
                std.debug.print("Output {d} differs, shape {any}\n", .{ i, mx.shape(result) });
                return error.KernelVariantMismatch;
            }
        }
        const entry = try covered.getOrPut(case.kernel);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var it = covered.iterator();
    while (it.next()) |entry| std.debug.print("PASS: {s}: {d} launches, every output bit exact\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    std.debug.print("PASS: {d} native launches across {d} embedded Metal variants\n", .{ cases.value.len, covered.count() });
}

fn parameter(case: Case, name: []const u8) i32 {
    for (case.templates) |p| if (std.mem.eql(u8, p.name, name)) return p.integer;
    unreachable;
}
fn matrix(s: *mx.Scope, a: mx.Array, n: i32, width: i32) !mx.Array {
    return s.reshape(try s.slice(try s.reshape(a, &.{-1}), 0, 0, n * width), &.{ n, width });
}
fn equalBits(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    const x = try raw(s, a);
    const y = try raw(s, b);
    try mx.evalMany(&.{ x, y }, false);
    const count = mx.c.mlx_array_size(x);
    if (count != mx.c.mlx_array_size(y) or !std.mem.eql(u8, mx.c.mlx_array_data_uint8(x)[0..count], mx.c.mlx_array_data_uint8(y)[0..count])) return error.NativeAffineMismatch;
}
