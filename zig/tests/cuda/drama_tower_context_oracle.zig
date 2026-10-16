//! Diagnostic: the production tower teacher-forced over a prompt prefix, then a suffix, at absolute positions.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const tower = drama.tower;
const Frame = tower.Frame;
const Impl = @typeInfo(@FieldType(tower.Model, "impl")).pointer.child;

const width = 2048;
const kv_width = 512;
const vocab = 4096;
const max_rows = 128;
const context = 512;
const prefix_rows = 181;
const suffix_rows = 128;
const hidden_rows = prefix_rows + suffix_rows;
const frame_rows = hidden_rows + 1;
const prefix_chunks = [_]usize{ 128, 53 }; // the production host's teacher-forced prompt chunks
const row_counts = [_]usize{ 1, 7, suffix_rows };
const projection_count = 6; // tower.zig `projections.len`
const head_kind = 5; // tower.zig `projections[5]` = { 4096, hidden }
const head_index = 8; // codebook 9: audio_decoder_mtp.heads.8
const head_plan = (1 - 1) * projection_count + head_kind; // rows = 1, as generate() and tf-drama-tower-oracle
const layer_count = @typeInfo(@FieldType(Impl, "layers")).array.len;

const Mode = enum { batch, repeat, streaming, rebased };
const modes = std.meta.tags(Mode);

comptime {
    std.debug.assert(layer_count == 13);
    var sum: usize = 0;
    for (prefix_chunks) |c| sum += c;
    std.debug.assert(sum == prefix_rows);
    std.debug.assert(hidden_rows <= context);
    for (prefix_chunks) |c| std.debug.assert(c <= max_rows);
}

/// Recorded failures that leave the files written but the evidence invalid. Messages hold no quote or backslash.
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

/// One schedule's outputs, sized for the largest suffix; a case uses the first n rows.
const Run = struct {
    hidden: []u16, // [max_rows][width] suffix final hidden
    head: []u16, // [vocab]
    kv: []u16, // [layer][keys, values][n][kv_width] suffix cache rows
    prefix_hidden: []u16, // [prefix_rows][width]
    prefix_kv: []u16, // [layer][keys, values][prefix_rows][kv_width]

    fn alloc(arena: Allocator) !Run {
        return .{
            .hidden = try arena.alloc(u16, max_rows * width),
            .head = try arena.alloc(u16, vocab),
            .kv = try arena.alloc(u16, layer_count * 2 * max_rows * kv_width),
            .prefix_hidden = try arena.alloc(u16, prefix_rows * width),
            .prefix_kv = try arena.alloc(u16, layer_count * 2 * prefix_rows * kv_width),
        };
    }

    fn kvRows(r: Run, n: usize, layer: usize, which: usize) []u16 {
        const at = (layer * 2 + which) * n * kv_width;
        return r.kv[at .. at + n * kv_width];
    }

    fn prefixKvRows(r: Run, layer: usize, which: usize) []u16 {
        const at = (layer * 2 + which) * prefix_rows * kv_width;
        return r.prefix_kv[at .. at + prefix_rows * kv_width];
    }
};

fn view(buffer: cuda.DeviceBuffer, offset: usize, len: usize) cuda.DeviceBuffer {
    std.debug.assert(offset <= buffer.len and len <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = len, .borrowed = true };
}

/// tower.zig cache layout: per layer a keys block then a values block of `context` rows of kv_width BF16.
fn cacheOffset(layer: usize, which: usize, row: usize) usize {
    return ((layer * 2 + which) * context + row) * kv_width * 2;
}

/// The oracle's head: rows = 1 plan, heads[8], one output_hidden row, after a NaN fill so a no-op cannot pass.
fn head(p: *Impl, stream: cuda.Stream, row: usize, out: []u16) !void {
    try p.head_logits.fill8(0xff, stream.handle);
    try p.plans.items[head_plan].run(p.output_hidden.ptr + row * width * 2, p.heads[head_index].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    try stream.synchronize();
    try p.head_logits.download(0, std.mem.sliceAsBytes(out));
}

/// One teacher-forced schedule (see the file comment); row r always uses frames[r] and labels frames[r + 1].
fn runSchedule(model: *tower.Model, stream: cuda.Stream, dev: cuda.DeviceBuffer, frames: []const Frame, n: usize, mode: Mode, r: Run) !void {
    const p = model.impl;
    try model.reset(stream, &.{}, 0);
    if (mode != .rebased) {
        var row: usize = 0;
        for (prefix_chunks) |c| {
            _ = try model.setInput(stream, frames[row .. row + c]);
            try model.prefill(stream, view(dev, row * width * 2, c * width * 2), frames[row + 1 .. row + c + 1]);
            try p.output_hidden.download(0, std.mem.sliceAsBytes(r.prefix_hidden[row * width .. (row + c) * width]));
            row += c;
        }
        for (0..layer_count) |layer| for (0..2) |which| {
            try p.cache.download(cacheOffset(layer, which, 0), std.mem.sliceAsBytes(r.prefixKvRows(layer, which)));
        };
    }
    const a = prefix_rows; // absolute fixture row of the first suffix row
    switch (mode) {
        .batch, .repeat, .rebased => {
            _ = try model.setInput(stream, frames[a .. a + n]);
            try model.prefill(stream, view(dev, a * width * 2, n * width * 2), frames[a + 1 .. a + n + 1]);
            try p.output_hidden.download(0, std.mem.sliceAsBytes(r.hidden[0 .. n * width]));
            try head(p, stream, n - 1, r.head);
        },
        .streaming => {
            for (0..n) |i| {
                _ = try model.setInput(stream, frames[a + i .. a + i + 1]);
                try model.prefill(stream, view(dev, (a + i) * width * 2, width * 2), frames[a + i + 1 .. a + i + 2]);
                try p.output_hidden.download(0, std.mem.sliceAsBytes(r.hidden[i * width .. (i + 1) * width]));
            }
            try head(p, stream, 0, r.head);
        },
    }
    const position0: usize = if (mode == .rebased) 0 else prefix_rows;
    for (0..layer_count) |layer| for (0..2) |which| {
        try p.cache.download(cacheOffset(layer, which, position0), std.mem.sliceAsBytes(r.kvRows(n, layer, which)));
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

/// Writes one BF16 array as a new file and appends `"key": {file, sha256, shape, nonfinite}` to json.
fn emit(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), key: []const u8, name: []const u8, data: []const u16, rows: usize, cols: usize, errors: *Errors) !usize {
    const bytes = std.mem.sliceAsBytes(data);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
    const bad = nonfiniteBf16(data);
    if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
    const sha = sha256Hex(bytes);
    try json.print(gpa, "\"{s}\": {{\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}}}", .{ key, name, sha[0..], rows, cols, bad });
    return bytes.len;
}

/// One schedule's suffix outputs as new files: native-N-MODE-{hidden,head9,layer-LL-keys,layer-LL-values}.bf16.
fn emitRun(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), n: usize, mode: Mode, r: Run, errors: *Errors) !usize {
    var name_buffer: [96]u8 = undefined;
    var written: usize = 0;
    try json.print(gpa, "\"{s}\": {{", .{@tagName(mode)});
    var name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-hidden.bf16", .{ n, @tagName(mode) });
    written += try emit(gpa, io, dir, json, "hidden", name, r.hidden[0 .. n * width], n, width, errors);
    try json.appendSlice(gpa, ", ");
    name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-head9.bf16", .{ n, @tagName(mode) });
    written += try emit(gpa, io, dir, json, "head9", name, r.head, 1, vocab, errors);
    try json.appendSlice(gpa, ", \"kv\": {");
    for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
        var key_buffer: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buffer, "layer-{d:0>2}-{s}", .{ layer, kind });
        name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-layer-{d:0>2}-{s}.bf16", .{ n, @tagName(mode), layer, kind });
        if (layer != 0 or which != 0) try json.appendSlice(gpa, ", ");
        written += try emit(gpa, io, dir, json, key, name, r.kvRows(n, layer, which), n, kv_width, errors);
    };
    try json.appendSlice(gpa, "}}");
    return written;
}

const Diff = struct { hidden: usize, head9: usize, kv: usize, prefix_hidden: usize, prefix_kv: usize };

fn compareRuns(n: usize, a: Run, b: Run, prefixes: bool) Diff {
    return .{
        .hidden = differences(a.hidden[0 .. n * width], b.hidden[0 .. n * width]),
        .head9 = differences(a.head, b.head),
        .kv = differences(a.kv[0 .. layer_count * 2 * n * kv_width], b.kv[0 .. layer_count * 2 * n * kv_width]),
        .prefix_hidden = if (prefixes) differences(a.prefix_hidden, b.prefix_hidden) else 0,
        .prefix_kv = if (prefixes) differences(a.prefix_kv, b.prefix_kv) else 0,
    };
}

fn printDiff(json: *std.ArrayList(u8), gpa: Allocator, d: Diff) !void {
    try json.print(gpa, "{{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}, \"prefix_hidden\": {d}, \"prefix_kv\": {d}}}", .{ d.hidden, d.head9, d.kv, d.prefix_hidden, d.prefix_kv });
}

fn seconds(ns: anytype) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-tower-context-oracle CHECKPOINT FIXTURE_DIRECTORY OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const fixture = try std.Io.Dir.cwd().openDir(io, args[2], .{});
    defer fixture.close(io);
    const out = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer out.close(io);

    // Every input is read before any CUDA call, so a missing or short fixture never touches the device.
    const hidden = try arena.alloc(u16, hidden_rows * width);
    var frames: [frame_rows]Frame = undefined;
    try readExact(io, fixture, "hidden.bf16", std.mem.sliceAsBytes(hidden));
    try readExact(io, fixture, "frames.u32le", std.mem.sliceAsBytes(&frames));
    const hidden_sha = sha256Hex(std.mem.sliceAsBytes(hidden));
    const frames_sha = sha256Hex(std.mem.sliceAsBytes(&frames));
    for (hidden) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteFixtureHidden;

    var runs: [modes.len]Run = undefined;
    for (&runs) |*r| r.* = try Run.alloc(arena);
    const base_prefix_hidden = try arena.alloc(u16, prefix_rows * width);
    const base_prefix_kv = try arena.alloc(u16, layer_count * 2 * prefix_rows * kv_width);
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
        p.cache.len != layer_count * 2 * context * kv_width * 2) return error.UnexpectedTowerLayout;
    const model_usage = cuda.usage(false);
    var device_hidden = try cuda.DeviceBuffer.alloc(&driver, hidden_rows * width * 2);
    defer device_hidden.free();
    try device_hidden.upload(0, std.mem.sliceAsBytes(hidden));

    var cases: std.ArrayList(u8) = .empty;
    defer cases.deinit(gpa);
    var prefix_json: std.ArrayList(u8) = .empty;
    defer prefix_json.deinit(gpa);
    var written: usize = 0;
    var have_base_prefix = false;
    for (row_counts, 0..) |n, case_index| {
        var timing: [modes.len]f64 = undefined;
        for (modes, 0..) |mode, i| {
            // The repeat runs on a poisoned cache: any read of a row this request did not write turns into NaN.
            if (mode == .repeat) {
                try p.cache.fill8(0xff, stream.handle);
                try stream.synchronize();
            }
            const start = std.Io.Clock.awake.now(io).toNanoseconds();
            try runSchedule(&model, stream, device_hidden, &frames, n, mode, runs[i]);
            timing[i] = seconds(std.Io.Clock.awake.now(io).toNanoseconds() - start);
        }
        const batch = runs[@backingInt(Mode.batch)];
        const repeat_diff = compareRuns(n, batch, runs[@backingInt(Mode.repeat)], true);
        if (repeat_diff.hidden != 0 or repeat_diff.head9 != 0 or repeat_diff.kv != 0 or repeat_diff.prefix_hidden != 0 or repeat_diff.prefix_kv != 0)
            try errors.add("rows {d}: batch vs poisoned-cache repeat bit differences hidden {d} head9 {d} kv {d} prefix_hidden {d} prefix_kv {d}", .{
                n, repeat_diff.hidden, repeat_diff.head9, repeat_diff.kv, repeat_diff.prefix_hidden, repeat_diff.prefix_kv,
            });
        const streaming = runs[@backingInt(Mode.streaming)];
        const streaming_prefix = Diff{ .hidden = 0, .head9 = 0, .kv = 0, .prefix_hidden = differences(batch.prefix_hidden, streaming.prefix_hidden), .prefix_kv = differences(batch.prefix_kv, streaming.prefix_kv) };
        if (streaming_prefix.prefix_hidden != 0 or streaming_prefix.prefix_kv != 0)
            try errors.add("rows {d}: streaming prefix differs from batch prefix hidden {d} kv {d}", .{ n, streaming_prefix.prefix_hidden, streaming_prefix.prefix_kv });
        if (!have_base_prefix) {
            @memcpy(base_prefix_hidden, batch.prefix_hidden);
            @memcpy(base_prefix_kv, batch.prefix_kv);
            have_base_prefix = true;
            var name_buffer: [96]u8 = undefined;
            try prefix_json.appendSlice(gpa, "{");
            written += try emit(gpa, io, out, &prefix_json, "hidden", "native-prefix-hidden.bf16", base_prefix_hidden, prefix_rows, width, &errors);
            try prefix_json.appendSlice(gpa, ", \"kv\": {");
            for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
                var key_buffer: [32]u8 = undefined;
                const key = try std.fmt.bufPrint(&key_buffer, "layer-{d:0>2}-{s}", .{ layer, kind });
                const name = try std.fmt.bufPrint(&name_buffer, "native-prefix-layer-{d:0>2}-{s}.bf16", .{ layer, kind });
                if (layer != 0 or which != 0) try prefix_json.appendSlice(gpa, ", ");
                written += try emit(gpa, io, out, &prefix_json, key, name, batch.prefixKvRows(layer, which), prefix_rows, kv_width, &errors);
            };
            try prefix_json.appendSlice(gpa, "}}");
        }
        const across_cases = .{ differences(base_prefix_hidden, batch.prefix_hidden), differences(base_prefix_kv, batch.prefix_kv) };
        if (across_cases[0] != 0 or across_cases[1] != 0)
            try errors.add("rows {d}: batch prefix differs from the first case prefix hidden {d} kv {d}", .{ n, across_cases[0], across_cases[1] });

        try cases.print(gpa, "{s}\n    {{\"rows\": {d}, \"suffix_positions\": [{d}, {d}], \"head_row\": \"last suffix row\"," ++
            " \"timing_s\": {{\"batch\": {d:.6}, \"repeat\": {d:.6}, \"streaming\": {d:.6}, \"rebased\": {d:.6}}}," ++
            " \"prefix_differences_vs_first_case\": {{\"hidden\": {d}, \"kv\": {d}}}, \"batch_vs_repeat_bit_differences\": ", .{
            if (case_index == 0) "" else ",", n, prefix_rows, prefix_rows + n - 1, timing[0], timing[1], timing[2], timing[3], across_cases[0], across_cases[1],
        });
        try printDiff(&cases, gpa, repeat_diff);
        try cases.appendSlice(gpa, ", \"streaming_vs_batch_prefix_bit_differences\": ");
        try printDiff(&cases, gpa, streaming_prefix);
        try cases.appendSlice(gpa, ", \"files\": {");
        for (modes, 0..) |mode, i| {
            if (i != 0) try cases.appendSlice(gpa, ", ");
            written += try emitRun(gpa, io, out, &cases, n, mode, runs[i], &errors);
        }
        try cases.appendSlice(gpa, "}}");
        std.debug.print("suffix_rows={d} repeat_hidden_diff={d} repeat_kv_diff={d} repeat_prefix_kv_diff={d} streaming_prefix_kv_diff={d}\n", .{
            n, repeat_diff.hidden, repeat_diff.kv, repeat_diff.prefix_kv, streaming_prefix.prefix_kv,
        });
    }

    // Input immutability: the device rows every schedule read, and the host fixture bytes, are what was loaded.
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
    const driver_source_sha = sha256Hex(@embedFile("drama_tower_context_oracle.zig"));
    const kernel_image_sha = sha256Hex(cuda.kernels.drama_tower);
    if (cuda.kernels.drama_tower.len == 0) try errors.add("embedded drama_tower kernel image is empty", .{});
    const evidence_valid = errors.count == 0;

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-tower-context-oracle\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"fixture_directory\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.print(gpa, ",\n  \"input_hidden_sha256\": \"{s}\",\n  \"input_frames_sha256\": \"{s}\",\n  \"width\": {d},\n  \"dtype\": \"bf16le\",\n  \"layer_count\": {d}," ++
        "\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}},\n  \"prefix_rows\": {d},\n  \"prefix_chunks\": [{d}, {d}],\n  \"suffix_rows\": [{d}, {d}, {d}]," ++
        "\n  \"schedules\": [\"batch\", \"repeat\", \"streaming\", \"rebased\"]," ++
        "\n  \"row_mapping\": \"fixture row r is request position r (rebased: suffix row j at position j); input frame frames[r], next-frame label frames[r+1]; suffix KV rows are cache positions 181.. (rebased 0..)\"," ++
        "\n  \"repeat_control\": \"KV cache filled with 0xFF before the repeat schedule; reset does not clear it\"," ++
        "\n  \"head\": {{\"weight\": \"audio_decoder_mtp.heads.{d}.weight\", \"plan\": \"plans[{d}] (rows 1, kind {d})\", \"input\": \"last suffix row of output_hidden\"}}," ++
        "\n  \"prefix\": {s},\n  \"cases\": [{s}\n  ]," ++
        "\n  \"input_immutable\": {{\"device_hidden_differences\": {d}, \"host_hidden_unchanged\": {}, \"host_frames_unchanged\": {}}}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}}}," ++
        "\n  \"bytes_written\": {d}," ++
        "\n  \"binary_sha256\": {s},\n  \"tower_source_sha256\": \"{s}\",\n  \"driver_source_sha256\": \"{s}\",\n  \"tower_kernel_image_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        hidden_sha[0..],       frames_sha[0..],          width,                 layer_count,
        context,               max_rows,                 prefix_rows,           prefix_chunks[0],
        prefix_chunks[1],      row_counts[0],            row_counts[1],         row_counts[2],
        head_index,            head_plan,                head_kind,             prefix_json.items,
        cases.items,           device_input_differences, host_hidden_unchanged, host_frames_unchanged,
        model_usage.device,    used.device,              used.peak,             used.host,
        written,               binary_json.items,        tower_source_sha[0..], driver_source_sha[0..],
        kernel_image_sha[0..], evidence_valid,           errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    std.debug.print("cases={d} bytes_written={d} evidence_valid={} errors={d} cuda_bytes={d} cuda_peak={d} pinned_bytes={d}\n", .{
        row_counts.len, written + manifest.items.len, evidence_valid, errors.count, used.device, used.peak, used.host,
    });
    return if (evidence_valid) 0 else 3;
}
