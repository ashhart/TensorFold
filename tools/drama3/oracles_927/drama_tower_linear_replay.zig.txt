//! Diagnostic: the production tower's layer-0 eh_proj cuBLASLt plans replayed on frozen inputs.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const lt_api = cuda.cublaslt;

const n_out = 2048; // tower.zig projections[0] = { hidden, 2 * hidden }: out 2048
const k_in = 4096; // in 4096
const workspace_bytes = 8 << 20; // tower.zig `workspace_bytes`
const batch_rows = 188;
const streaming_rows = 7;
const plan_rows = [_]usize{ 1, 7, 53, 128 };
const repeats = 3;

const Cell = struct { label: []const u8, streaming: bool, row0: usize, m: usize };
const cells = [_]Cell{
    .{ .label = "batch-call0", .streaming = false, .row0 = 0, .m = 128 },
    .{ .label = "batch-call1", .streaming = false, .row0 = 128, .m = 53 },
    .{ .label = "batch-call2", .streaming = false, .row0 = 181, .m = 7 },
    .{ .label = "streaming-row-181", .streaming = true, .row0 = 0, .m = 1 },
    .{ .label = "streaming-row-182", .streaming = true, .row0 = 1, .m = 1 },
    .{ .label = "streaming-row-183", .streaming = true, .row0 = 2, .m = 1 },
    .{ .label = "streaming-row-184", .streaming = true, .row0 = 3, .m = 1 },
    .{ .label = "streaming-row-185", .streaming = true, .row0 = 4, .m = 1 },
    .{ .label = "streaming-row-186", .streaming = true, .row0 = 5, .m = 1 },
    .{ .label = "streaming-row-187", .streaming = true, .row0 = 6, .m = 1 },
};
// cublasLtMatmulAlgoConfigAttributes_t (cublasLt.h): id, name, exact storage size in bytes.
const AlgoAttribute = struct { id: c_int, name: []const u8, size: usize };
const algo_attributes = [_]AlgoAttribute{
    .{ .id = 0, .name = "algo_id", .size = 4 },          .{ .id = 1, .name = "tile_id", .size = 4 },
    .{ .id = 2, .name = "splitk_num", .size = 4 },       .{ .id = 3, .name = "reduction_scheme", .size = 4 },
    .{ .id = 4, .name = "cta_swizzling", .size = 4 },    .{ .id = 5, .name = "custom_option", .size = 4 },
    .{ .id = 6, .name = "stages_id", .size = 4 },        .{ .id = 7, .name = "inner_shape_id", .size = 2 },
    .{ .id = 8, .name = "cluster_shape_id", .size = 2 },
};
const AlgoConfigGetAttribute = *const fn (*const lt_api.Algo, c_int, *anyopaque, usize, *usize) callconv(.c) lt_api.Status;

comptime {
    for (cells) |c| std.debug.assert(c.row0 + c.m <= (if (c.streaming) streaming_rows else batch_rows));
}

const Errors = struct {
    gpa: Allocator,
    json: std.ArrayList(u8) = .empty,
    count: usize = 0,

    fn add(e: *Errors, comptime f: []const u8, args: anytype) !void {
        std.debug.print("error: " ++ f ++ "\n", args);
        try e.json.appendSlice(e.gpa, if (e.count == 0) "\n    \"" else ",\n    \"");
        try e.json.print(e.gpa, f, args);
        try e.json.append(e.gpa, '"');
        e.count += 1;
    }
};

fn differences(a: []const u16, b: []const u16) usize {
    var count: usize = 0;
    for (a, b) |x, y| {
        if (x != y) count += 1;
    }
    return count;
}

fn nonfiniteBf16(values: []const u16) usize {
    var count: usize = 0;
    for (values) |bits| {
        if ((bits & 0x7f80) == 0x7f80) count += 1;
    }
    return count;
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var out: [32]u8 = undefined;
    Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

fn fileSha256(gpa: Allocator, io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    var hash = Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

fn readExact(io: std.Io, path: []const u8, bytes: []u8) !void {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    if (try file.length(io) != bytes.len or try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.BadInputLength;
}

fn jsonString(list: *std.ArrayList(u8), gpa: Allocator, s: []const u8) !void {
    try list.append(gpa, '"');
    for (s) |c| switch (c) {
        '"', '\\' => {
            try list.append(gpa, '\\');
            try list.append(gpa, c);
        },
        0...0x1f => try list.print(gpa, "\\u{x:0>4}", .{c}),
        else => try list.append(gpa, c),
    };
    try list.append(gpa, '"');
}

/// The path of the mapped libcublasLt, from /proc/self/maps (after the library was opened).
fn mappedCublasLt(gpa: Allocator, io: std.Io, out: *std.ArrayList(u8)) !void {
    const file = try std.Io.Dir.cwd().openFile(io, "/proc/self/maps", .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &buffer);
    const maps = try reader.interface.allocRemaining(gpa, .limited(1 << 24));
    defer gpa.free(maps);
    var lines = std.mem.splitScalar(u8, maps, '\n');
    while (lines.next()) |line| {
        if (std.mem.indexOf(u8, line, "libcublasLt.so") == null) continue;
        const slash = std.mem.indexOfScalar(u8, line, '/') orelse continue;
        try out.appendSlice(gpa, line[slash..]);
        return;
    }
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-tower-linear-replay WEIGHT.bf16 BATCH_JOINED.bf16 STREAMING_JOINED.bf16 OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);

    // Every input is read before any CUDA call.
    const weight = try arena.alloc(u16, n_out * k_in);
    const batch = try arena.alloc(u16, batch_rows * k_in);
    const streaming = try arena.alloc(u16, streaming_rows * k_in);
    try readExact(io, args[1], std.mem.sliceAsBytes(weight));
    try readExact(io, args[2], std.mem.sliceAsBytes(batch));
    try readExact(io, args[3], std.mem.sliceAsBytes(streaming));
    const weight_sha = sha256Hex(std.mem.sliceAsBytes(weight));
    const batch_sha = sha256Hex(std.mem.sliceAsBytes(batch));
    const streaming_sha = sha256Hex(std.mem.sliceAsBytes(streaming));
    for ([_][]const u16{ weight, batch, streaming }) |values| if (nonfiniteBf16(values) != 0) return error.NonfiniteInput;

    const max_m = 128;
    const result = try arena.alloc(u16, (max_m + 1) * n_out);
    const again = try arena.alloc(u16, (max_m + 1) * n_out);
    const input_check = try arena.alloc(u16, max_m * k_in);
    const weight_check = try arena.alloc(u16, n_out * k_in);

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var lt = try lt_api.Library.open();
    defer lt.close();
    const get_attribute = lt.lib.lookup(AlgoConfigGetAttribute, "cublasLtMatmulAlgoConfigGetAttribute") orelse return error.MissingSymbol;

    var plans: [plan_rows.len]lt_api.Linear = undefined;
    var plans_made: usize = 0;
    defer for (plans[0..plans_made]) |*plan| plan.deinit();
    for (plan_rows, 0..) |m, i| {
        plans[i] = try lt_api.Linear.init(&lt, m, n_out, k_in, .bf16, workspace_bytes);
        plans_made += 1;
    }

    var device_weight = try cuda.DeviceBuffer.alloc(&driver, n_out * k_in * 2);
    defer device_weight.free();
    try device_weight.upload(0, std.mem.sliceAsBytes(weight));
    var device_x = try cuda.DeviceBuffer.alloc(&driver, max_m * k_in * 2);
    defer device_x.free();
    var device_d = try cuda.DeviceBuffer.alloc(&driver, (max_m + 1) * n_out * 2);
    defer device_d.free();
    var workspace = try cuda.DeviceBuffer.alloc(&driver, workspace_bytes);
    defer workspace.free();

    // Plans: the chosen algorithm, decoded.
    var plans_json: std.ArrayList(u8) = .empty;
    defer plans_json.deinit(gpa);
    for (plan_rows, 0..) |m, i| {
        const plan = &plans[i];
        try plans_json.print(gpa, "{s}\n    {{\"m\": {d}, \"n\": {d}, \"k\": {d}, \"workspace_need\": {d}, \"workspace_limit\": {d}, \"algo_bytes\": \"", .{
            if (i == 0) "" else ",", m, n_out, k_in, plan.workspace_need, workspace_bytes,
        });
        for (std.mem.asBytes(&plan.algo)) |byte| try plans_json.print(gpa, "{x:0>2}", .{byte});
        try plans_json.appendSlice(gpa, "\"");
        for (algo_attributes) |attribute| {
            var buffer: [8]u8 = @splat(0);
            var written: usize = 0;
            const status = get_attribute(&plan.algo, attribute.id, &buffer, attribute.size, &written);
            if (status != 0 or written != attribute.size) {
                try errors.add("m {d}: cublasLtMatmulAlgoConfigGetAttribute({s}) status {d} written {d}", .{ m, attribute.name, status, written });
                try plans_json.print(gpa, ", \"{s}\": null", .{attribute.name});
            } else {
                const value: i64 = if (attribute.size == 4) std.mem.readInt(i32, buffer[0..4], .little) else std.mem.readInt(u16, buffer[0..2], .little);
                try plans_json.print(gpa, ", \"{s}\": {d}", .{ attribute.name, value });
            }
        }
        try plans_json.appendSlice(gpa, "}");
    }

    var cells_json: std.ArrayList(u8) = .empty;
    defer cells_json.deinit(gpa);
    var name_buffer: [96]u8 = undefined;
    for (cells, 0..) |cell, ci| {
        const plan_index = std.mem.indexOfScalar(usize, &plan_rows, cell.m) orelse unreachable;
        const plan = &plans[plan_index];
        const source = if (cell.streaming) streaming else batch;
        const x = source[cell.row0 * k_in .. (cell.row0 + cell.m) * k_in];
        const out_len = cell.m * n_out;
        var repeat_differences: usize = 0;
        var guard_broken: usize = 0;
        var input_changed: usize = 0;
        var nonfinite: usize = 0;
        for (0..repeats) |r| {
            try device_x.upload(0, std.mem.sliceAsBytes(x));
            try device_d.fill8(0xff, stream.handle);
            try workspace.fill8(0xff, stream.handle);
            try plan.run(device_x.ptr, device_weight.ptr, device_d.ptr, workspace.ptr, workspace.len, stream.handle);
            try stream.synchronize();
            const target = if (r == 0) result else again;
            try device_d.download(0, std.mem.sliceAsBytes(target[0 .. out_len + n_out]));
            for (target[out_len .. out_len + n_out]) |bits| {
                if (bits != 0xffff) guard_broken += 1;
            }
            if (r == 0) nonfinite = nonfiniteBf16(result[0..out_len]) else repeat_differences += differences(result[0..out_len], again[0..out_len]);
            try device_x.download(0, std.mem.sliceAsBytes(input_check[0 .. cell.m * k_in]));
            input_changed += differences(input_check[0 .. cell.m * k_in], x);
        }
        if (repeat_differences != 0 or guard_broken != 0 or input_changed != 0 or nonfinite != 0)
            try errors.add("{s}: repeat differences {d}, guard values changed {d}, device input changed {d}, nonfinite {d}", .{
                cell.label, repeat_differences, guard_broken, input_changed, nonfinite,
            });
        const name = try std.fmt.bufPrint(&name_buffer, "native-replay-{s}.bf16", .{cell.label});
        const bytes = std.mem.sliceAsBytes(result[0..out_len]);
        try out.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
        const sha = sha256Hex(bytes);
        try cells_json.print(gpa, "{s}\n    {{\"label\": \"{s}\", \"source\": \"{s}\", \"source_row0\": {d}, \"m\": {d}, \"plan_m\": {d}, \"runs\": {d}," ++
            " \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}, \"repeat_differences\": {d}," ++
            " \"guard_values_changed\": {d}, \"device_input_changed\": {d}}}", .{
            if (ci == 0) "" else ",", cell.label, if (cell.streaming) "streaming" else "batch", cell.row0, cell.m,    plan_rows[plan_index], repeats,
            name,                     sha[0..],   cell.m,                                       n_out,     nonfinite, repeat_differences,    guard_broken,
            input_changed,
        });
    }

    try device_weight.download(0, std.mem.sliceAsBytes(weight_check));
    const device_weight_differences = differences(weight_check, weight);
    if (device_weight_differences != 0) try errors.add("device weight changed in {d} values", .{device_weight_differences});
    const host_unchanged = std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(weight)), &weight_sha) and
        std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(batch)), &batch_sha) and
        std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(streaming)), &streaming_sha);
    if (!host_unchanged) try errors.add("host inputs changed", .{});
    const used = cuda.usage(false);

    var library_path: std.ArrayList(u8) = .empty;
    defer library_path.deinit(gpa);
    try mappedCublasLt(gpa, io, &library_path);
    var library_sha: [64]u8 = @splat('0');
    if (library_path.items.len == 0) {
        try errors.add("libcublasLt mapping not found", .{});
    } else {
        library_sha = fileSha256(gpa, io, library_path.items) catch |err| blk: {
            try errors.add("libcublasLt hash failed: {s}", .{@errorName(err)});
            break :blk library_sha;
        };
    }
    var binary_json: std.ArrayList(u8) = .empty;
    defer binary_json.deinit(gpa);
    if (fileSha256(gpa, io, "/proc/self/exe")) |sha| {
        try binary_json.print(gpa, "\"{s}\"", .{sha[0..]});
    } else |err| {
        std.debug.print("binary hash unavailable: {s}\n", .{@errorName(err)});
        try binary_json.appendSlice(gpa, "null");
    }
    const driver_source_sha = sha256Hex(@embedFile("drama_tower_linear_replay.zig"));
    const evidence_valid = errors.count == 0;

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-tower-linear-replay\",\n  \"weight_path\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"batch_joined_path\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.appendSlice(gpa, ",\n  \"streaming_joined_path\": ");
    try jsonString(&manifest, gpa, args[3]);
    try manifest.appendSlice(gpa, ",\n  \"cublaslt_path\": ");
    try jsonString(&manifest, gpa, library_path.items);
    try manifest.print(gpa, ",\n  \"weight_sha256\": \"{s}\",\n  \"batch_joined_sha256\": \"{s}\",\n  \"streaming_joined_sha256\": \"{s}\"," ++
        "\n  \"operation\": \"cuda.cublaslt.Linear: D[m,{d}] = X[m,{d}] . W[{d},{d}]^T, BF16 A/B/D, compute 32F, scale 32F, transa T, transb N, first heuristic, workspace limit {d}\"," ++
        "\n  \"cublaslt_version\": {d},\n  \"cublaslt_sha256\": \"{s}\",\n  \"repeats\": {d}," ++
        "\n  \"plans\": [{s}\n  ],\n  \"cells\": [{s}\n  ]," ++
        "\n  \"device_weight_differences\": {d},\n  \"host_inputs_unchanged\": {}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}}}," ++
        "\n  \"binary_sha256\": {s},\n  \"driver_source_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        weight_sha[0..],             batch_sha[0..],            streaming_sha[0..],     n_out,
        k_in,                        n_out,                     k_in,                   workspace_bytes,
        lt.api.cublasLtGetVersion(), library_sha[0..],          repeats,                plans_json.items,
        cells_json.items,            device_weight_differences, host_unchanged,         used.device,
        used.peak,                   binary_json.items,         driver_source_sha[0..], evidence_valid,
        errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    std.debug.print("cells={d} evidence_valid={} errors={d}\n", .{ cells.len, evidence_valid, errors.count });
    return if (evidence_valid) 0 else 3;
}
