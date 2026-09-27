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
