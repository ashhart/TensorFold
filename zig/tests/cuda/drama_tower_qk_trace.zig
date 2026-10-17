//! Diagnostic: the production tower's Q/K norm and RoPE operands and outputs on the full-context fixture.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const tower = drama.tower;
const Stage = tower.Stage;
const Frame = tower.Frame;
const Impl = @typeInfo(@FieldType(tower.Model, "impl")).pointer.child;

const width = 2048;
const qkv_width = 9216;
const query_width = 4096;
const kv_width = 512;
const vocab = 4096;
const max_rows = 128;
const context = 512;
const prefix_rows = 181;
const suffix_rows = 128;
const hidden_rows = prefix_rows + suffix_rows;
const frame_rows = hidden_rows + 1;
const prefix_chunks = [_]usize{ 128, 53 };
const row_counts = [_]usize{ 1, 7, suffix_rows };
const projection_count = 6;
const head_kind = 5;
const head_index = 8;
const head_plan = (1 - 1) * projection_count + head_kind;
const layer_count = @typeInfo(@FieldType(Impl, "layers")).array.len;
const captured = [_]Stage{ .qkv, .query, .key, .value };

comptime {
    std.debug.assert(layer_count == 13);
    std.debug.assert(tower.stageWidth(.qkv) == qkv_width and tower.stageWidth(.query) == query_width);
    std.debug.assert(tower.stageWidth(.key) == kv_width and tower.stageWidth(.value) == kv_width);
    var sum: usize = 0;
    for (prefix_chunks) |c| sum += c;
    std.debug.assert(sum == prefix_rows and hidden_rows <= context);
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

fn stageIndex(stage: Stage) ?usize {
    for (captured, 0..) |s, i| if (s == stage) return i;
    return null;
}

/// Per layer and captured stage, [max_rows][stageWidth] BF16.
const Captures = struct {
    data: [layer_count][captured.len][]u16,

    fn alloc(arena: Allocator) !Captures {
        var c: Captures = undefined;
        for (&c.data) |*layer| for (layer, captured) |*slot, stage| {
            slot.* = try arena.alloc(u16, max_rows * tower.stageWidth(stage));
        };
        return c;
    }

    fn rows(c: *const Captures, layer: usize, which: usize, n: usize) []u16 {
        return c.data[layer][which][0 .. n * tower.stageWidth(captured[which])];
    }
};

const Tracer = struct {
    n: usize,
    staging: []align(16) u8,
    out: *Captures,
    seen: [layer_count][captured.len]usize = std.mem.zeroes([layer_count][captured.len]usize),
    expected: usize = 0, // next capture in (layer, stage) order
    out_of_order: usize = 0,
    bad_row0: usize = 0,

    pub fn capture(t: *Tracer, stream: cuda.Stream, layer: usize, stage: Stage, src: cuda.DeviceBuffer, row0: usize) !void {
        const which = stageIndex(stage) orelse return;
        if (layer * captured.len + which != t.expected) t.out_of_order += 1;
        t.expected = layer * captured.len + which + 1;
        if (row0 != prefix_rows) t.bad_row0 += 1;
        t.seen[layer][which] += 1;
        const dst = t.out.rows(layer, which, t.n);
        const bytes = std.mem.sliceAsBytes(dst);
        if (src.len != bytes.len or bytes.len > t.staging.len) return error.TraceCaptureShape;
        try src.downloadAsync(0, t.staging[0..bytes.len], stream.handle);
        try stream.synchronize();
        @memcpy(bytes, t.staging[0..bytes.len]);
    }
};

const Finals = struct {
    hidden: []u16,
    head: []u16,
    kv: []u16,

    fn alloc(arena: Allocator) !Finals {
        return .{ .hidden = try arena.alloc(u16, max_rows * width), .head = try arena.alloc(u16, vocab), .kv = try arena.alloc(u16, layer_count * 2 * max_rows * kv_width) };
    }

    fn kvRows(f: Finals, n: usize, layer: usize, which: usize) []u16 {
        const at = (layer * 2 + which) * n * kv_width;
        return f.kv[at .. at + n * kv_width];
    }
};

fn view(buffer: cuda.DeviceBuffer, offset: usize, len: usize) cuda.DeviceBuffer {
    std.debug.assert(offset <= buffer.len and len <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = len, .borrowed = true };
}

fn cacheOffset(layer: usize, which: usize, row: usize) usize {
    return ((layer * 2 + which) * context + row) * kv_width * 2;
}

fn runBatch(model: *tower.Model, stream: cuda.Stream, dev: cuda.DeviceBuffer, frames: []const Frame, n: usize, trace: anytype, f: Finals) !void {
    const p = model.impl;
    try model.reset(stream, &.{}, 0);
    var row: usize = 0;
    for (prefix_chunks) |c| {
        _ = try model.setInput(stream, frames[row .. row + c]);
        try model.prefill(stream, view(dev, row * width * 2, c * width * 2), frames[row + 1 .. row + c + 1]);
        row += c;
    }
    const a = prefix_rows;
    _ = try model.setInput(stream, frames[a .. a + n]);
    if (@TypeOf(trace) == void)
        try model.prefill(stream, view(dev, a * width * 2, n * width * 2), frames[a + 1 .. a + n + 1])
    else
        try model.prefillTraced(stream, view(dev, a * width * 2, n * width * 2), frames[a + 1 .. a + n + 1], trace);
    try p.output_hidden.download(0, std.mem.sliceAsBytes(f.hidden[0 .. n * width]));
    try p.head_logits.fill8(0xff, stream.handle);
    try p.plans.items[head_plan].run(p.output_hidden.ptr + (n - 1) * width * 2, p.heads[head_index].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    try stream.synchronize();
    try p.head_logits.download(0, std.mem.sliceAsBytes(f.head));
    for (0..layer_count) |layer| for (0..2) |which| {
        try p.cache.download(cacheOffset(layer, which, prefix_rows), std.mem.sliceAsBytes(f.kvRows(n, layer, which)));
    };
}

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
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.bytesToHex(out, .lower);
}

fn fileSha256(gpa: Allocator, io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    const buffer = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(buffer);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: u64 = 0;
    while (true) {
        const got = try file.readPositionalAll(io, buffer, offset);
        hash.update(buffer[0..got]);
        offset += got;
        if (got < buffer.len) break;
    }
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

fn readExact(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []u8) !void {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    if (try file.length(io) != bytes.len or try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.BadFixtureLength;
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

fn emit(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), key: []const u8, name: []const u8, data: []const u16, rows: usize, cols: usize, errors: *Errors) !usize {
    const bytes = std.mem.sliceAsBytes(data);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
    const bad = nonfiniteBf16(data);
    if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
    const sha = sha256Hex(bytes);
    try json.print(gpa, "\"{s}\": {{\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}}}", .{ key, name, sha[0..], rows, cols, bad });
    return bytes.len;
}

fn emitFinals(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), n: usize, label: []const u8, f: Finals, errors: *Errors) !usize {
    var name_buffer: [96]u8 = undefined;
    var written: usize = 0;
    try json.print(gpa, "\"{s}\": {{", .{label});
    var name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-hidden.bf16", .{ n, label });
    written += try emit(gpa, io, dir, json, "hidden", name, f.hidden[0 .. n * width], n, width, errors);
    try json.appendSlice(gpa, ", ");
    name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-head9.bf16", .{ n, label });
    written += try emit(gpa, io, dir, json, "head9", name, f.head, 1, vocab, errors);
    try json.appendSlice(gpa, ", \"kv\": {");
    for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
        var key_buffer: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buffer, "layer-{d:0>2}-{s}", .{ layer, kind });
        name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-layer-{d:0>2}-{s}.bf16", .{ n, label, layer, kind });
        if (layer != 0 or which != 0) try json.appendSlice(gpa, ", ");
        written += try emit(gpa, io, dir, json, key, name, f.kvRows(n, layer, which), n, kv_width, errors);
    };
    try json.appendSlice(gpa, "}}");
    return written;
}

fn compareFinals(n: usize, a: Finals, b: Finals) [3]usize {
    return .{
        differences(a.hidden[0 .. n * width], b.hidden[0 .. n * width]),
        differences(a.head, b.head),
        differences(a.kv[0 .. layer_count * 2 * n * kv_width], b.kv[0 .. layer_count * 2 * n * kv_width]),
    };
}

fn checkTracer(t: *const Tracer, n: usize, label: []const u8, errors: *Errors) !void {
    if (t.out_of_order != 0 or t.bad_row0 != 0) try errors.add("rows {d} {s}: {d} captures out of order, {d} with a bad row0", .{ n, label, t.out_of_order, t.bad_row0 });
    for (0..layer_count) |layer| for (0..captured.len) |which| {
        if (t.seen[layer][which] != 1) try errors.add("rows {d} {s}: layer {d} stage {s} captured {d} times", .{ n, label, layer, @tagName(captured[which]), t.seen[layer][which] });
    };
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-tower-qk-trace CHECKPOINT FIXTURE_DIRECTORY OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const fixture = try std.Io.Dir.cwd().openDir(io, args[2], .{});
    defer fixture.close(io);
    const out = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer out.close(io);

    const hidden = try arena.alloc(u16, hidden_rows * width);
    var frames: [frame_rows]Frame = undefined;
    try readExact(io, fixture, "hidden.bf16", std.mem.sliceAsBytes(hidden));
    try readExact(io, fixture, "frames.u32le", std.mem.sliceAsBytes(&frames));
    const hidden_sha = sha256Hex(std.mem.sliceAsBytes(hidden));
    const frames_sha = sha256Hex(std.mem.sliceAsBytes(&frames));
    for (hidden) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteFixtureHidden;

    var first = try Captures.alloc(arena);
    var again = try Captures.alloc(arena);
    const traced = try Finals.alloc(arena);
    const untraced = try Finals.alloc(arena);
    const repeat = try Finals.alloc(arena);
    const device_check = try arena.alloc(u16, hidden_rows * width);

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try tower.Model.init(gpa, io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer model.deinit();
    const p = model.impl;
    if (p.plans.items.len != max_rows * projection_count or p.head_logits.len != vocab * 2 or p.output_hidden.len != max_rows * width * 2 or
        p.qkv.len != max_rows * qkv_width * 2 or p.queries.len != max_rows * query_width * 2 or
        p.cache.len != layer_count * 2 * context * kv_width * 2) return error.UnexpectedTowerLayout;
    const model_usage = cuda.usage(false);
    var device_hidden = try cuda.DeviceBuffer.alloc(&driver, hidden_rows * width * 2);
    defer device_hidden.free();
    try device_hidden.upload(0, std.mem.sliceAsBytes(hidden));
    var staging = try cuda.HostBuffer.alloc(&driver, max_rows * qkv_width * 2);
    defer staging.free();

    var cases: std.ArrayList(u8) = .empty;
    defer cases.deinit(gpa);
    var written: usize = 0;
    var name_buffer: [96]u8 = undefined;
    for (row_counts, 0..) |n, case_index| {
        var t1: Tracer = .{ .n = n, .staging = staging.bytes, .out = &first };
        try runBatch(&model, stream, device_hidden, &frames, n, &t1, traced);
        try runBatch(&model, stream, device_hidden, &frames, n, {}, untraced);
        var t2: Tracer = .{ .n = n, .staging = staging.bytes, .out = &again };
        try runBatch(&model, stream, device_hidden, &frames, n, &t2, repeat);
        try checkTracer(&t1, n, "traced", &errors);
        try checkTracer(&t2, n, "repeat", &errors);

        var capture_repeat_differences: usize = 0;
        var cache_vs_capture: usize = 0;
        for (0..layer_count) |layer| {
            for (0..captured.len) |which| capture_repeat_differences += differences(first.rows(layer, which, n), again.rows(layer, which, n));
            cache_vs_capture += differences(first.rows(layer, 2, n), traced.kvRows(n, layer, 0));
            cache_vs_capture += differences(first.rows(layer, 3, n), traced.kvRows(n, layer, 1));
        }
        if (capture_repeat_differences != 0) try errors.add("rows {d}: {d} captured values differ between traced runs", .{ n, capture_repeat_differences });
        if (cache_vs_capture != 0) try errors.add("rows {d}: key/value captures differ from the exported cache rows in {d} values", .{ n, cache_vs_capture });
        const du = compareFinals(n, traced, untraced);
        const dr = compareFinals(n, traced, repeat);
        if (du[0] != 0 or du[1] != 0 or du[2] != 0) try errors.add("rows {d}: traced vs untraced bit differences hidden {d} head9 {d} kv {d}", .{ n, du[0], du[1], du[2] });
        if (dr[0] != 0 or dr[1] != 0 or dr[2] != 0) try errors.add("rows {d}: traced vs repeat bit differences hidden {d} head9 {d} kv {d}", .{ n, dr[0], dr[1], dr[2] });

        try cases.print(gpa, "{s}\n    {{\"rows\": {d}, \"suffix_positions\": [{d}, {d}]," ++
            " \"traced_vs_untraced_bit_differences\": {{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}}}," ++
            " \"traced_vs_repeat_bit_differences\": {{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}}}," ++
            " \"capture_repeat_differences\": {d}, \"key_value_capture_vs_cache_differences\": {d}, \"captures\": [", .{
            if (case_index == 0) "" else ",", n,                prefix_rows, prefix_rows + n - 1, du[0], du[1], du[2], dr[0], dr[1], dr[2],
            capture_repeat_differences,       cache_vs_capture,
        });
        for (0..layer_count) |layer| {
            try cases.print(gpa, "{s}{{\"layer\": {d}", .{ if (layer == 0) "" else ", ", layer });
            for (captured, 0..) |stage, which| {
                try cases.appendSlice(gpa, ", ");
                const name = try std.fmt.bufPrint(&name_buffer, "native-{d}-layer-{d:0>2}-{s}.bf16", .{ n, layer, @tagName(stage) });
                written += try emit(gpa, io, out, &cases, @tagName(stage), name, first.rows(layer, which, n), n, tower.stageWidth(stage), &errors);
            }
            try cases.appendSlice(gpa, "}");
        }
        try cases.appendSlice(gpa, "], \"finals\": {");
        written += try emitFinals(gpa, io, out, &cases, n, "traced", traced, &errors);
        try cases.appendSlice(gpa, ", ");
        written += try emitFinals(gpa, io, out, &cases, n, "untraced", untraced, &errors);
        try cases.appendSlice(gpa, ", ");
        written += try emitFinals(gpa, io, out, &cases, n, "repeat", repeat, &errors);
        try cases.appendSlice(gpa, "}}");
        std.debug.print("suffix_rows={d} untraced_diff={d}/{d}/{d} repeat_diff={d}/{d}/{d} capture_repeat_diff={d}\n", .{
            n, du[0], du[1], du[2], dr[0], dr[1], dr[2], capture_repeat_differences,
        });
    }

    try device_hidden.download(0, std.mem.sliceAsBytes(device_check));
    const device_input_differences = differences(device_check, hidden);
    if (device_input_differences != 0) try errors.add("device input hidden changed in {d} values", .{device_input_differences});
    const host_hidden_unchanged = std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(hidden)), &hidden_sha);
    const host_frames_unchanged = std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(&frames)), &frames_sha);
    if (!host_hidden_unchanged or !host_frames_unchanged) try errors.add("host fixture changed: hidden unchanged {} frames unchanged {}", .{ host_hidden_unchanged, host_frames_unchanged });
    const used = cuda.usage(false);

    var binary_json: std.ArrayList(u8) = .empty;
    defer binary_json.deinit(gpa);
    if (fileSha256(gpa, io, "/proc/self/exe")) |sha| {
        try binary_json.print(gpa, "\"{s}\"", .{sha[0..]});
    } else |err| {
        std.debug.print("binary hash unavailable: {s}\n", .{@errorName(err)});
        try binary_json.appendSlice(gpa, "null");
    }
    const tower_source_sha = sha256Hex(tower.trace_source);
    const driver_source_sha = sha256Hex(@embedFile("drama_tower_qk_trace.zig"));
    const kernel_image_sha = sha256Hex(cuda.kernels.drama_tower);
    if (cuda.kernels.drama_tower.len == 0) try errors.add("embedded drama_tower kernel image is empty", .{});
    const evidence_valid = errors.count == 0;

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-tower-qk-trace\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"fixture_directory\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.print(gpa, ",\n  \"input_hidden_sha256\": \"{s}\",\n  \"input_frames_sha256\": \"{s}\",\n  \"layer_count\": {d}," ++
        "\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}},\n  \"prefix_rows\": {d},\n  \"prefix_chunks\": [{d}, {d}],\n  \"suffix_rows\": [{d}, {d}, {d}]," ++
        "\n  \"captured_stages\": [\"qkv\", \"query\", \"key\", \"value\"]," ++
        "\n  \"rope_launch\": \"drama_rope_cache(qkv, qnorm, knorm, queries, keys, values, start = position, 16, 2, 256, 64, rope_base, eps), grid rows * 18, block 256\"," ++
        "\n  \"row_mapping\": \"capture row j is fixture row 181 + j at position 181 + j; qkv in checkpoint order [query 16x256 | gate 16x256 | key 2x256 | value 2x256]\"," ++
        "\n  \"cases\": [{s}\n  ]," ++
        "\n  \"input_immutable\": {{\"device_hidden_differences\": {d}, \"host_hidden_unchanged\": {}, \"host_frames_unchanged\": {}}}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}}}," ++
        "\n  \"bytes_written\": {d}," ++
        "\n  \"binary_sha256\": {s},\n  \"tower_source_sha256\": \"{s}\",\n  \"driver_source_sha256\": \"{s}\",\n  \"tower_kernel_image_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        hidden_sha[0..],          frames_sha[0..],       layer_count,            context,
        max_rows,                 prefix_rows,           prefix_chunks[0],       prefix_chunks[1],
        row_counts[0],            row_counts[1],         row_counts[2],          cases.items,
        device_input_differences, host_hidden_unchanged, host_frames_unchanged,  model_usage.device,
        used.device,              used.peak,             used.host,              written,
        binary_json.items,        tower_source_sha[0..], driver_source_sha[0..], kernel_image_sha[0..],
        evidence_valid,           errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    std.debug.print("cases={d} bytes_written={d} evidence_valid={} errors={d} cuda_bytes={d} cuda_peak={d}\n", .{
        row_counts.len, written + manifest.items.len, evidence_valid, errors.count, used.device, used.peak,
    });
    return if (evidence_valid) 0 else 3;
}
