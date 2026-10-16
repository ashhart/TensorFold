//! Diagnostic: every tower stage along the full-context trajectory of a seven-row suffix, batch and streaming.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const tower = drama.tower;
const Stage = tower.Stage;
const Frame = tower.Frame;
const context_capture = @import("drama_tower_context_capture.zig");
const Store = context_capture.Store;
const Tracer = context_capture.Tracer;
const checkTracer = context_capture.checkTracer;
const uncovered = context_capture.uncovered;
const Impl = @typeInfo(@FieldType(tower.Model, "impl")).pointer.child;
const LayerWeights = @typeInfo(@FieldType(Impl, "layers")).array.child;

const width = 2048;
const kv_width = 512;
const vocab = 4096;
const max_rows = 128;
const context = 512;
const prefix_rows = 181;
const suffix = 7;
pub const total_rows = prefix_rows + suffix;
const fixture_hidden_rows = 309;
const fixture_frame_rows = fixture_hidden_rows + 1;
const prefix_chunks = [_]usize{ 128, 53 };
const projection_count = 6; // tower.zig `projections.len`
const head_kind = 5; // tower.zig `projections[5]` = { 4096, hidden }
const head_index = 8; // codebook 9: audio_decoder_mtp.heads.8
const head_plan = (1 - 1) * projection_count + head_kind;
pub const layer_count = @typeInfo(@FieldType(Impl, "layers")).array.len;
pub const stages = std.meta.tags(Stage);
pub const stage_count = stages.len;
pub const per_layer = stage_count - 1; // every stage except shared_input, once per layer
pub const captures_per_call = 1 + layer_count * per_layer;
const max_stage_width = 12288; // gate_up

const Schedule = enum { batch, streaming };
const schedules = std.meta.tags(Schedule);

comptime {
    std.debug.assert(layer_count == 13);
    std.debug.assert(stages[0] == .shared_input and stage_count == 16);
    for (stages) |s| std.debug.assert(tower.stageWidth(s) <= max_stage_width);
    var sum: usize = 0;
    for (prefix_chunks) |c| sum += c;
    std.debug.assert(sum == prefix_rows and total_rows <= fixture_hidden_rows and total_rows <= context);
}

pub const Errors = struct {
    gpa: Allocator,
    json: std.ArrayList(u8) = .empty,
    count: usize = 0,

    pub fn add(e: *Errors, comptime f: []const u8, args: anytype) !void {
        std.debug.print("error: " ++ f ++ "\n", args);
        try e.json.appendSlice(e.gpa, if (e.count == 0) "\n    \"" else ",\n    \"");
        try e.json.print(e.gpa, f, args);
        try e.json.append(e.gpa, '"');
        e.count += 1;
    }
};

const Finals = struct {
    hidden: []u16, // [suffix][width]
    head: []u16, // [vocab]
    kv: []u16, // [layer][keys, values][suffix][kv_width]
    prefix_hidden: []u16, // [prefix_rows][width]
    prefix_kv: []u16, // [layer][keys, values][prefix_rows][kv_width]

    fn alloc(arena: Allocator) !Finals {
        return .{ .hidden = try arena.alloc(u16, suffix * width), .head = try arena.alloc(u16, vocab), .kv = try arena.alloc(u16, layer_count * 2 * suffix * kv_width), .prefix_hidden = try arena.alloc(u16, prefix_rows * width), .prefix_kv = try arena.alloc(u16, layer_count * 2 * prefix_rows * kv_width) };
    }

    fn kvRows(f: Finals, layer: usize, which: usize) []u16 {
        const at = (layer * 2 + which) * suffix * kv_width;
        return f.kv[at .. at + suffix * kv_width];
    }

    fn prefixKvRows(f: Finals, layer: usize, which: usize) []u16 {
        const at = (layer * 2 + which) * prefix_rows * kv_width;
        return f.prefix_kv[at .. at + prefix_rows * kv_width];
    }
};

const Diff = struct { hidden: usize, head9: usize, kv: usize, prefix_hidden: usize, prefix_kv: usize };

fn compareFinals(a: Finals, b: Finals) Diff {
    return .{ .hidden = differences(a.hidden, b.hidden), .head9 = differences(a.head, b.head), .kv = differences(a.kv, b.kv), .prefix_hidden = differences(a.prefix_hidden, b.prefix_hidden), .prefix_kv = differences(a.prefix_kv, b.prefix_kv) };
}

fn zero(d: Diff) bool {
    return d.hidden == 0 and d.head9 == 0 and d.kv == 0 and d.prefix_hidden == 0 and d.prefix_kv == 0;
}

fn printDiff(json: *std.ArrayList(u8), gpa: Allocator, d: Diff) !void {
    try json.print(gpa, "{{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}, \"prefix_hidden\": {d}, \"prefix_kv\": {d}}}", .{ d.hidden, d.head9, d.kv, d.prefix_hidden, d.prefix_kv });
}

fn view(buffer: cuda.DeviceBuffer, offset: usize, len: usize) cuda.DeviceBuffer {
    std.debug.assert(offset <= buffer.len and len <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = len, .borrowed = true };
}

/// tower.zig cache layout: per layer a keys block then a values block of `context` rows of kv_width BF16.
fn cacheOffset(layer: usize, which: usize, row: usize) usize {
    return ((layer * 2 + which) * context + row) * kv_width * 2;
}

/// tf-drama-tower-context-oracle's head: rows = 1 plan, heads[8], one output_hidden row, after a NaN fill.
fn head(p: *Impl, stream: cuda.Stream, row: usize, out: []u16) !void {
    try p.head_logits.fill8(0xff, stream.handle);
    try p.plans.items[head_plan].run(p.output_hidden.ptr + row * width * 2, p.heads[head_index].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    try stream.synchronize();
    try p.head_logits.download(0, std.mem.sliceAsBytes(out));
}

/// One prefill of `rows` fixture rows from `row0` (positions row0..); traced when `trace` is a tracer pointer.
fn call(model: *tower.Model, stream: cuda.Stream, dev: cuda.DeviceBuffer, frames: []const Frame, row0: usize, rows: usize, trace: anytype) !void {
    _ = try model.setInput(stream, frames[row0 .. row0 + rows]);
    const hidden_rows = view(dev, row0 * width * 2, rows * width * 2);
    const labels = frames[row0 + 1 .. row0 + rows + 1];
    if (@TypeOf(trace) == void) {
        try model.prefill(stream, hidden_rows, labels);
    } else {
        trace.begin(row0, rows);
        try model.prefillTraced(stream, hidden_rows, labels, trace);
        trace.end();
    }
}

fn runSchedule(model: *tower.Model, stream: cuda.Stream, dev: cuda.DeviceBuffer, frames: []const Frame, schedule: Schedule, trace: anytype, f: Finals) !void {
    const p = model.impl;
    try model.reset(stream, &.{}, 0);
    var row: usize = 0;
    for (prefix_chunks) |c| {
        try call(model, stream, dev, frames, row, c, trace);
        try p.output_hidden.download(0, std.mem.sliceAsBytes(f.prefix_hidden[row * width .. (row + c) * width]));
        row += c;
    }
    for (0..layer_count) |layer| for (0..2) |which| {
        try p.cache.download(cacheOffset(layer, which, 0), std.mem.sliceAsBytes(f.prefixKvRows(layer, which)));
    };
    switch (schedule) {
        .batch => {
            try call(model, stream, dev, frames, prefix_rows, suffix, trace);
            try p.output_hidden.download(0, std.mem.sliceAsBytes(f.hidden));
            try head(p, stream, suffix - 1, f.head);
        },
        .streaming => {
            for (0..suffix) |i| {
                try call(model, stream, dev, frames, prefix_rows + i, 1, trace);
                try p.output_hidden.download(0, std.mem.sliceAsBytes(f.hidden[i * width .. (i + 1) * width]));
            }
            try head(p, stream, 0, f.head);
        },
    }
    for (0..layer_count) |layer| for (0..2) |which| {
        try p.cache.download(cacheOffset(layer, which, prefix_rows), std.mem.sliceAsBytes(f.kvRows(layer, which)));
    };
}

fn hashBuffer(hash: *Sha256, buffer: cuda.DeviceBuffer, host: []u8) !void {
    var offset: usize = 0;
    while (offset < buffer.len) {
        const n = @min(host.len, buffer.len - offset);
        try buffer.download(offset, host[0..n]);
        hash.update(host[0..n]);
        offset += n;
    }
}

/// SHA-256 over every device weight the tower reads: layers in field order, the nine heads, the embedding table.
fn weightDigest(p: *Impl, host: []u8) ![64]u8 {
    var hash = Sha256.init(.{});
    for (p.layers) |layer_weights| {
        inline for (@typeInfo(LayerWeights).@"struct".field_names) |name| try hashBuffer(&hash, @field(layer_weights, name), host);
    }
    for (p.heads) |h| try hashBuffer(&hash, h, host);
    try hashBuffer(&hash, p.embeddings, host);
    return std.fmt.bytesToHex(hash.finalResult(), .lower);
}

pub fn differences(a: []const u16, b: []const u16) usize {
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

/// Writes one BF16 array as a new file and appends `"file": .., "sha256": .., "shape": [r, c], "nonfinite": n`.
fn emitFields(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), name: []const u8, data: []const u16, rows: usize, cols: usize, errors: *Errors) !usize {
    const bytes = std.mem.sliceAsBytes(data);
    try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
    const bad = nonfiniteBf16(data);
    if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
    const sha = sha256Hex(bytes);
    try json.print(gpa, "\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}", .{ name, sha[0..], rows, cols, bad });
    return bytes.len;
}

fn emitKeyed(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), key: []const u8, name: []const u8, data: []const u16, rows: usize, cols: usize, errors: *Errors) !usize {
    try json.print(gpa, "\"{s}\": {{", .{key});
    const written = try emitFields(gpa, io, dir, json, name, data, rows, cols, errors);
    try json.appendSlice(gpa, "}");
    return written;
}

fn emitFinals(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), schedule: Schedule, f: Finals, errors: *Errors) !usize {
    const s = @tagName(schedule);
    var name_buffer: [96]u8 = undefined;
    var written: usize = 0;
    try json.appendSlice(gpa, "{");
    written += try emitKeyed(gpa, io, dir, json, "hidden", try std.fmt.bufPrint(&name_buffer, "native-{s}-hidden.bf16", .{s}), f.hidden, suffix, width, errors);
    try json.appendSlice(gpa, ", ");
    written += try emitKeyed(gpa, io, dir, json, "head9", try std.fmt.bufPrint(&name_buffer, "native-{s}-head9.bf16", .{s}), f.head, 1, vocab, errors);
    try json.appendSlice(gpa, ", ");
    written += try emitKeyed(gpa, io, dir, json, "prefix_hidden", try std.fmt.bufPrint(&name_buffer, "native-{s}-prefix-hidden.bf16", .{s}), f.prefix_hidden, prefix_rows, width, errors);
    try json.appendSlice(gpa, ", \"kv\": {");
    for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
        var key_buffer: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buffer, "layer-{d:0>2}-{s}", .{ layer, kind });
        if (layer != 0 or which != 0) try json.appendSlice(gpa, ", ");
        written += try emitKeyed(gpa, io, dir, json, key, try std.fmt.bufPrint(&name_buffer, "native-{s}-layer-{d:0>2}-{s}.bf16", .{ s, layer, kind }), f.kvRows(layer, which), suffix, kv_width, errors);
    };
    try json.appendSlice(gpa, "}, \"prefix_kv\": {");
    for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
        var key_buffer: [32]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buffer, "layer-{d:0>2}-{s}", .{ layer, kind });
        if (layer != 0 or which != 0) try json.appendSlice(gpa, ", ");
        written += try emitKeyed(gpa, io, dir, json, key, try std.fmt.bufPrint(&name_buffer, "native-{s}-prefix-layer-{d:0>2}-{s}.bf16", .{ s, layer, kind }), f.prefixKvRows(layer, which), prefix_rows, kv_width, errors);
    };
    try json.appendSlice(gpa, "}}");
    return written;
}

/// The stage files of one schedule: batch all 188 rows, streaming its 7 suffix rows.
fn emitStages(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), schedule: Schedule, store: *const Store, errors: *Errors) !usize {
    const row0: usize = if (schedule == .batch) 0 else prefix_rows;
    const n: usize = total_rows - row0;
    var name_buffer: [96]u8 = undefined;
    var written: usize = 0;
    var first = true;
    try json.appendSlice(gpa, "[");
    for (0..layer_count) |layer| for (stages, 0..) |stage, k| {
        if (!Store.captured(layer, k)) continue;
        try json.print(gpa, "{s}\n      {{\"layer\": {d}, \"stage\": \"{s}\", \"row0\": {d}, ", .{ if (first) "" else ",", layer, @tagName(stage), row0 });
        first = false;
        const name = try std.fmt.bufPrint(&name_buffer, "native-{s}-layer-{d:0>2}-{s}.bf16", .{ @tagName(schedule), layer, @tagName(stage) });
        written += try emitFields(gpa, io, dir, json, name, store.rows(layer, k, row0, n), n, tower.stageWidth(stage), errors);
        try json.appendSlice(gpa, "}");
    };
    try json.appendSlice(gpa, "]");
    return written;
}

/// Layer-12 output and every layer's key / value captures equal the exported hidden and KV rows.
fn consistency(store: *const Store, f: Finals) usize {
    const output = @backingInt(Stage.output);
    const key = @backingInt(Stage.key);
    const value = @backingInt(Stage.value);
    var count = differences(store.rows(layer_count - 1, output, 0, prefix_rows), f.prefix_hidden) +
        differences(store.rows(layer_count - 1, output, prefix_rows, suffix), f.hidden);
    for (0..layer_count) |layer| {
        count += differences(store.rows(layer, key, 0, prefix_rows), f.prefixKvRows(layer, 0));
        count += differences(store.rows(layer, value, 0, prefix_rows), f.prefixKvRows(layer, 1));
        count += differences(store.rows(layer, key, prefix_rows, suffix), f.kvRows(layer, 0));
        count += differences(store.rows(layer, value, prefix_rows, suffix), f.kvRows(layer, 1));
    }
    return count;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-tower-context-trace CHECKPOINT HIDDEN.bf16 FRAMES.u32le OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);

    // Every input is read before any CUDA call.
    const hidden = try arena.alloc(u16, fixture_hidden_rows * width);
    var frames: [fixture_frame_rows]Frame = undefined;
    try readExact(io, args[2], std.mem.sliceAsBytes(hidden));
    try readExact(io, args[3], std.mem.sliceAsBytes(&frames));
    const hidden_sha = sha256Hex(std.mem.sliceAsBytes(hidden));
    const frames_sha = sha256Hex(std.mem.sliceAsBytes(&frames));
    for (hidden) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteFixtureHidden;

    var stores: [schedules.len]Store = undefined;
    for (&stores) |*store| store.* = try Store.alloc(arena);
    var finals: [schedules.len][3]Finals = undefined; // traced, untraced, repeat
    for (&finals) |*per| for (per) |*f| {
        f.* = try Finals.alloc(arena);
    };
    const device_check = try arena.alloc(u16, fixture_hidden_rows * width);
    const host_chunk = try gpa.alloc(u8, 64 << 20);
    defer gpa.free(host_chunk);

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
    const weights_before = try weightDigest(p, host_chunk);
    var device_hidden = try cuda.DeviceBuffer.alloc(&driver, fixture_hidden_rows * width * 2);
    defer device_hidden.free();
    try device_hidden.upload(0, std.mem.sliceAsBytes(hidden));
    var staging = try cuda.HostBuffer.alloc(&driver, max_rows * max_stage_width * 2);
    defer staging.free();

    var schedule_json: std.ArrayList(u8) = .empty;
    defer schedule_json.deinit(gpa);
    var written: usize = 0;
    for (schedules, 0..) |schedule, si| {
        const label = @tagName(schedule);
        const expected_calls = prefix_chunks.len + @as(usize, if (schedule == .batch) 1 else suffix);
        var traced: Tracer = .{ .store = &stores[si], .compare = false, .staging = staging.bytes };
        try runSchedule(&model, stream, device_hidden, &frames, schedule, &traced, finals[si][0]);
        try runSchedule(&model, stream, device_hidden, &frames, schedule, {}, finals[si][1]);
        try p.cache.fill8(0xff, stream.handle);
        try stream.synchronize();
        var again: Tracer = .{ .store = &stores[si], .compare = true, .staging = staging.bytes };
        try runSchedule(&model, stream, device_hidden, &frames, schedule, &again, finals[si][2]);
        try checkTracer(&traced, label, expected_calls, &errors);
        try checkTracer(&again, label, expected_calls, &errors);
        const missing = uncovered(&stores[si]);
        if (missing != 0) try errors.add("{s}: {d} stage rows not written exactly once", .{ label, missing });
        const untraced_diff = compareFinals(finals[si][0], finals[si][1]);
        const repeat_diff = compareFinals(finals[si][0], finals[si][2]);
        if (!zero(untraced_diff)) try errors.add("{s}: traced vs untraced finals differ hidden {d} head9 {d} kv {d} prefix_hidden {d} prefix_kv {d}", .{ label, untraced_diff.hidden, untraced_diff.head9, untraced_diff.kv, untraced_diff.prefix_hidden, untraced_diff.prefix_kv });
        if (!zero(repeat_diff)) try errors.add("{s}: traced vs poisoned repeat finals differ hidden {d} head9 {d} kv {d} prefix_hidden {d} prefix_kv {d}", .{ label, repeat_diff.hidden, repeat_diff.head9, repeat_diff.kv, repeat_diff.prefix_hidden, repeat_diff.prefix_kv });
        const inconsistent = consistency(&stores[si], finals[si][0]);
        if (inconsistent != 0) try errors.add("{s}: {d} layer-12 output or key/value capture values differ from the exported hidden/KV", .{ label, inconsistent });

        try schedule_json.print(gpa, "{s}\n    \"{s}\": {{\"calls\": [", .{ if (si == 0) "" else ",", label });
        var row: usize = 0;
        for (0..expected_calls) |c| {
            const rows: usize = if (c < prefix_chunks.len) prefix_chunks[c] else if (schedule == .batch) suffix else 1;
            try schedule_json.print(gpa, "{s}{{\"row0\": {d}, \"rows\": {d}}}", .{ if (c == 0) "" else ", ", row, rows });
            row += rows;
        }
        try schedule_json.print(gpa, "], \"trace\": {{\"calls\": {d}, \"captures_per_call\": {d}, \"out_of_order\": {d}, \"incomplete_calls\": {d}," ++
            " \"bad_rows\": {d}, \"duplicate_rows\": {d}, \"uncovered_rows\": {d}, \"repeat_capture_differences\": {d}, \"repeat_calls\": {d}," ++
            " \"repeat_out_of_order\": {d}}}, \"output_kv_capture_inconsistencies\": {d}, \"traced_vs_untraced\": ", .{
            traced.calls,              captures_per_call, traced.out_of_order, traced.incomplete_calls, traced.bad_rows, traced.duplicate_rows, missing,
            again.compare_differences, again.calls,       again.out_of_order,  inconsistent,
        });
        try printDiff(&schedule_json, gpa, untraced_diff);
        try schedule_json.appendSlice(gpa, ", \"traced_vs_repeat\": ");
        try printDiff(&schedule_json, gpa, repeat_diff);
        try schedule_json.appendSlice(gpa, ", \"finals\": ");
        written += try emitFinals(gpa, io, out, &schedule_json, schedule, finals[si][0], &errors);
        try schedule_json.appendSlice(gpa, ", \"stages\": ");
        written += try emitStages(gpa, io, out, &schedule_json, schedule, &stores[si], &errors);
        try schedule_json.appendSlice(gpa, "}");
        std.debug.print("{s}: calls={d} untraced_diff={d}/{d}/{d} repeat_capture_diff={d} uncovered={d}\n", .{
            label, traced.calls, untraced_diff.hidden, untraced_diff.head9, untraced_diff.kv, again.compare_differences, missing,
        });
    }

    // The streaming prefix calls are the batch prefix calls: finals and every stage row must be identical.
    const bi = @backingInt(Schedule.batch);
    const st = @backingInt(Schedule.streaming);
    const prefix_finals = .{ differences(finals[bi][0].prefix_hidden, finals[st][0].prefix_hidden), differences(finals[bi][0].prefix_kv, finals[st][0].prefix_kv) };
    var prefix_stage_differences: usize = 0;
    for (0..layer_count) |layer| for (0..stage_count) |k| {
        if (Store.captured(layer, k)) prefix_stage_differences += differences(stores[bi].rows(layer, k, 0, prefix_rows), stores[st].rows(layer, k, 0, prefix_rows));
    };
    if (prefix_finals[0] != 0 or prefix_finals[1] != 0 or prefix_stage_differences != 0)
        try errors.add("streaming prefix differs from batch: hidden {d} kv {d} stage values {d}", .{ prefix_finals[0], prefix_finals[1], prefix_stage_differences });

    try device_hidden.download(0, std.mem.sliceAsBytes(device_check));
    const device_input_differences = differences(device_check, hidden);
    if (device_input_differences != 0) try errors.add("device input hidden changed in {d} values", .{device_input_differences});
    const host_hidden_unchanged = std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(hidden)), &hidden_sha);
    const host_frames_unchanged = std.mem.eql(u8, &sha256Hex(std.mem.sliceAsBytes(&frames)), &frames_sha);
    if (!host_hidden_unchanged or !host_frames_unchanged) try errors.add("host fixture changed: hidden unchanged {} frames unchanged {}", .{ host_hidden_unchanged, host_frames_unchanged });
    const weights_after = try weightDigest(p, host_chunk);
    const weights_unchanged = std.mem.eql(u8, &weights_before, &weights_after);
    if (!weights_unchanged) try errors.add("device weights changed during the run", .{});
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
    const driver_source_sha = sha256Hex(@embedFile("drama_tower_context_trace.zig"));
    const kernel_image_sha = sha256Hex(cuda.kernels.drama_tower);
    if (cuda.kernels.drama_tower.len == 0) try errors.add("embedded drama_tower kernel image is empty", .{});
    const evidence_valid = errors.count == 0;

    var stage_order: std.ArrayList(u8) = .empty;
    defer stage_order.deinit(gpa);
    for (stages, 0..) |stage, k| try stage_order.print(gpa, "{s}{{\"stage\": \"{s}\", \"width\": {d}}}", .{ if (k == 0) "" else ", ", @tagName(stage), tower.stageWidth(stage) });

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-tower-context-trace\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"hidden_path\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.appendSlice(gpa, ",\n  \"frames_path\": ");
    try jsonString(&manifest, gpa, args[3]);
    try manifest.print(gpa, ",\n  \"input_hidden_sha256\": \"{s}\",\n  \"input_frames_sha256\": \"{s}\",\n  \"width\": {d},\n  \"dtype\": \"bf16le\",\n  \"layer_count\": {d}," ++
        "\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}},\n  \"prefix_rows\": {d},\n  \"prefix_chunks\": [{d}, {d}],\n  \"suffix_rows\": {d},\n  \"total_rows\": {d}," ++
        "\n  \"stage_order\": [{s}],\n  \"captures_per_call\": {d}," ++
        "\n  \"row_mapping\": \"stage row r is request position r = fixture row r; input frame frames[r], next-frame label frames[r+1]; batch stage files hold rows 0..187, streaming stage files rows 181..187; shared_input is layer 0 only (the layer-0 mixer extra input)\"," ++
        "\n  \"column_mapping\": \"qkv in checkpoint order [query 16x256 | gate 16x256 | key 2x256 | value 2x256]; key / value are the layer's cache rows of the call\"," ++
        "\n  \"repeat_control\": \"KV cache filled with 0xFF before each traced repeat; its captures are compared with the first traced run\"," ++
        "\n  \"head\": {{\"weight\": \"audio_decoder_mtp.heads.{d}.weight\", \"plan\": \"plans[{d}] (rows 1, kind {d})\", \"input\": \"last suffix row of output_hidden\"}}," ++
        "\n  \"schedules\": {{{s}\n  }},", .{
        hidden_sha[0..],     frames_sha[0..], width,       layer_count,
        context,             max_rows,        prefix_rows, prefix_chunks[0],
        prefix_chunks[1],    suffix,          total_rows,  stage_order.items,
        captures_per_call,   head_index,      head_plan,   head_kind,
        schedule_json.items,
    });
    try manifest.print(gpa, "\n  \"streaming_prefix_vs_batch\": {{\"prefix_hidden\": {d}, \"prefix_kv\": {d}, \"stage_values\": {d}}}," ++
        "\n  \"input_immutable\": {{\"device_hidden_differences\": {d}, \"host_hidden_unchanged\": {}, \"host_frames_unchanged\": {}}}," ++
        "\n  \"weights\": {{\"sha256_before\": \"{s}\", \"sha256_after\": \"{s}\", \"unchanged\": {}}}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}}}," ++
        "\n  \"bytes_written\": {d}," ++
        "\n  \"binary_sha256\": {s},\n  \"tower_source_sha256\": \"{s}\",\n  \"driver_source_sha256\": \"{s}\",\n  \"tower_kernel_image_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        prefix_finals[0],         prefix_finals[1],       prefix_stage_differences,
        device_input_differences, host_hidden_unchanged,  host_frames_unchanged,
        weights_before[0..],      weights_after[0..],     weights_unchanged,
        model_usage.device,       used.device,            used.peak,
        used.host,                written,                binary_json.items,
        tower_source_sha[0..],    driver_source_sha[0..], kernel_image_sha[0..],
        evidence_valid,           errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    std.debug.print("bytes_written={d} evidence_valid={} errors={d} cuda_bytes={d} cuda_peak={d}\n", .{
        written + manifest.items.len, evidence_valid, errors.count, used.device, used.peak,
    });
    return if (evidence_valid) 0 else 3;
}
