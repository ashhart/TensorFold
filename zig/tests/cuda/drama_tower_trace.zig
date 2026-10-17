//! Diagnostic: captures every stage of every production tower layer on the real-input fixture.
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
const kv_width = 512;
const vocab = 4096;
const max_rows = 128;
/// tf-drama-tower-oracle's options: the same cuBLASLt plan list and KV cache layout.
const context = 128;
const fixture_frames = max_rows + 1;
const row_counts = [_]usize{ 1, 7, max_rows };
const variants = [_][]const u8{ "traced", "untraced", "repeat" };
const projection_count = 6; // tower.zig `projections.len`
const head_kind = 5; // tower.zig `projections[5]` = { 4096, hidden }, the codebook head shape
const head_index = 8; // codebook 9: audio_decoder_mtp.heads.8, the oracle's `p.heads[8]`
const head_plan = (1 - 1) * projection_count + head_kind; // rows = 1, as generate() and the oracle
/// The prefill's own layer array: every layer it iterates is captured.
const layer_count = @typeInfo(@FieldType(Impl, "layers")).array.len;
const stages = std.meta.tags(Stage);
const stage_count = stages.len;
const slots = layer_count * stage_count;
/// shared_input once, then every other stage once per layer.
const capture_count = 1 + layer_count * (stage_count - 1);
const max_stage_width = blk: {
    var m: usize = 0;
    for (stages) |s| m = @max(m, tower.stageWidth(s));
    break :blk m;
};

comptime {
    std.debug.assert(layer_count == 13);
    std.debug.assert(stages[0] == .shared_input);
    std.debug.assert(tower.stageWidth(.output) == width and tower.stageWidth(.key) == kv_width);
}

/// What each stage file holds, in `tower.zig` terms.
const stage_sources =
    \\{"shared_input": "Impl.shared after setInput: drama_frame_embeddings sum of codes 2..9 (layer 0 mix extra input), BF16 [N, 2048]",
    \\    "joined": "Impl.joined after drama_mix: [enorm(code embedding (+ shared_input)), hnorm(layer input)], BF16 [N, 4096]",
    \\    "mixed": "Impl.mixed after eh_proj (projection 0), BF16 [N, 2048]",
    \\    "attention_input": "Impl.normalized after drama_norm(mixed, input_norm), BF16 [N, 2048]",
    \\    "qkv": "Impl.qkv after wqkv (projection 1), checkpoint column order [query 16x256 | gate 16x256 | key 2x256 | value 2x256], BF16 [N, 9216]",
    \\    "query": "Impl.queries after drama_rope_cache: q_norm then interleaved rope on the first 64 dims, BF16 [N, 4096]",
    \\    "key": "layer key cache rows position..position+N after drama_rope_cache: k_norm then rope, BF16 [N, 512]",
    \\    "value": "layer value cache rows position..position+N after drama_rope_cache: raw qkv value, BF16 [N, 512]",
    \\    "attended": "Impl.attended after drama_attention: causal attention times sigmoid(gate), the wo input, BF16 [N, 4096]",
    \\    "attention_output": "Impl.branch after wo (projection 2), BF16 [N, 2048]",
    \\    "mlp_input": "Impl.normalized after drama_norm(attention_output + mixed, post_norm), BF16 [N, 2048]",
    \\    "mlp_residual": "Impl.residual after the same drama_norm: BF16 attention_output + mixed, BF16 [N, 2048]",
    \\    "gate_up": "Impl.gate_up after projection 3: [w1 gate | w3 up], BF16 [N, 12288]",
    \\    "activated": "Impl.activated after drama_silu_mul: silu(gate) * up, BF16 [N, 6144]",
    \\    "mlp_output": "Impl.branch after w2 (projection 4), BF16 [N, 2048]",
    \\    "output": "Impl.output_hidden after drama_norm(mlp_output + mlp_residual, final_norm): layer output, BF16 [N, 2048]"}
;

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

const Slot = struct { layer: usize, stage: Stage };

/// The k-th capture of one prefill, in execution order.
fn expectedSlot(k: usize) Slot {
    if (k == 0) return .{ .layer = 0, .stage = .shared_input };
    return .{ .layer = (k - 1) / (stage_count - 1), .stage = stages[1 + (k - 1) % (stage_count - 1)] };
}

fn slotOf(layer: usize, stage: Stage) usize {
    return layer * stage_count + @backingInt(stage);
}

fn stageFile(buffer: []u8, n: usize, layer: usize, stage: Stage) ![]u8 {
    return std.fmt.bufPrint(buffer, "native-{d}-layer-{d:0>2}-{s}.bf16", .{ n, layer, @tagName(stage) });
}

/// The `drama.tower.Model.prefillTraced` tracer for one run of N rows.
const Tracer = struct {
    io: std.Io,
    dir: std.Io.Dir,
    n: usize,
    /// Pinned, max_rows * max_stage_width * 2 bytes: the target of the stream-ordered copy.
    staging: []align(16) u8,
    /// null: the first traced run, which writes every capture. Otherwise that run's hashes, compared instead.
    first: ?*const [slots][32]u8,
    hashes: [slots][32]u8 = undefined,
    nonfinite: [slots]usize = @splat(0),
    captures: usize = 0,
    repeat_mismatches: usize = 0,
    written: usize = 0,

    pub fn capture(t: *Tracer, stream: cuda.Stream, layer: usize, stage: Stage, src: cuda.DeviceBuffer, row0: usize) !void {
        if (t.captures >= capture_count) return error.TraceTooManyCaptures;
        const want = expectedSlot(t.captures);
        if (layer != want.layer or stage != want.stage) return error.TraceCaptureOutOfOrder;
        if (row0 != 0) return error.TraceRowsOutOfRange;
        const len = t.n * tower.stageWidth(stage) * 2;
        if (src.len != len or len > t.staging.len) return error.TraceCaptureShape;
        // Queued behind the stage's kernel and ahead of the next one, then drained before the prefill continues.
        const bytes = t.staging[0..len];
        try src.downloadAsync(0, bytes, stream.handle);
        try stream.synchronize();
        const at = slotOf(layer, stage);
        std.crypto.hash.sha2.Sha256.hash(bytes, &t.hashes[at], .{});
        t.nonfinite[at] = nonfiniteBf16(std.mem.bytesAsSlice(u16, bytes));
        if (t.first) |first| {
            if (!std.mem.eql(u8, &first[at], &t.hashes[at])) t.repeat_mismatches += 1;
        } else {
            var name_buffer: [80]u8 = undefined;
            const name = try stageFile(&name_buffer, t.n, layer, stage);
            try t.dir.writeFile(t.io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
            t.written += bytes.len;
        }
        t.captures += 1;
    }
};

/// One run's outputs, sized for max_rows; a case uses the first n rows.
const Finals = struct {
    hidden: []u16,
    head: []u16,
    /// [layer][keys, values][n][kv_width]
    kv: []u16,

    fn alloc(gpa: Allocator) !Finals {
        const hidden = try gpa.alloc(u16, max_rows * width);
        errdefer gpa.free(hidden);
        const head = try gpa.alloc(u16, vocab);
        errdefer gpa.free(head);
        return .{ .hidden = hidden, .head = head, .kv = try gpa.alloc(u16, layer_count * 2 * max_rows * kv_width) };
    }

    fn free(f: Finals, gpa: Allocator) void {
        gpa.free(f.hidden);
        gpa.free(f.head);
        gpa.free(f.kv);
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

/// The oracle's teacher-forced call, then its final hidden, head and KV exports. `trace` void: production prefill.
fn runCase(model: *tower.Model, stream: cuda.Stream, n: usize, hidden: cuda.DeviceBuffer, frames: []const Frame, trace: anytype, f: Finals) !void {
    try model.reset(stream, &.{}, 0);
    _ = try model.setInput(stream, frames[0..n]);
    if (@TypeOf(trace) == void)
        try model.prefill(stream, view(hidden, 0, n * width * 2), frames[1 .. n + 1])
    else
        try model.prefillTraced(stream, view(hidden, 0, n * width * 2), frames[1 .. n + 1], trace);
    const p = model.impl;
    try p.output_hidden.download(0, std.mem.sliceAsBytes(f.hidden[0 .. n * width]));
    // 0xffff is a BF16 NaN: a head that writes nothing cannot pass. Same plan, weights and row as the oracle.
    try p.head_logits.fill8(0xff, stream.handle);
    try p.plans.items[head_plan].run(p.output_hidden.ptr + (n - 1) * width * 2, p.heads[head_index].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
    try stream.synchronize();
    try p.head_logits.download(0, std.mem.sliceAsBytes(f.head));
    for (0..layer_count) |layer| for (0..2) |which| {
        try p.cache.download((layer * 2 + which) * context * kv_width * 2, std.mem.sliceAsBytes(f.kvRows(n, layer, which)));
    };
}

fn differences(a: []const u16, b: []const u16) usize {
    var count: usize = 0;
    for (a, b) |x, y| {
        if (x != y) count += 1;
    }
    return count;
}

const Diff = struct { hidden: usize, head9: usize, kv: usize };

fn compareFinals(n: usize, a: Finals, b: Finals) Diff {
    return .{
        .hidden = differences(a.hidden[0 .. n * width], b.hidden[0 .. n * width]),
        .head9 = differences(a.head, b.head),
        .kv = differences(a.kv[0 .. layer_count * 2 * n * kv_width], b.kv[0 .. layer_count * 2 * n * kv_width]),
    };
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

/// Writes one run's final outputs as new files and appends their manifest entries.
fn emitFinals(gpa: Allocator, io: std.Io, dir: std.Io.Dir, json: *std.ArrayList(u8), n: usize, variant: []const u8, f: Finals, errors: *Errors) !usize {
    var name_buffer: [96]u8 = undefined;
    var written: usize = 0;
    try json.print(gpa, "\"{s}\": {{", .{variant});
    const singles = [_]struct { kind: []const u8, data: []const u16, cols: usize, rows: usize }{
        .{ .kind = "hidden", .data = f.hidden[0 .. n * width], .cols = width, .rows = n },
        .{ .kind = "head9", .data = f.head, .cols = vocab, .rows = 1 },
    };
    for (singles) |s| {
        const name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-{s}.bf16", .{ n, variant, s.kind });
        const bytes = std.mem.sliceAsBytes(s.data);
        try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
        written += bytes.len;
        const bad = nonfiniteBf16(s.data);
        if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
        const sha = sha256Hex(bytes);
        try json.print(gpa, "\"{s}\": {{\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}}}, ", .{ s.kind, name, sha[0..], s.rows, s.cols, bad });
    }
    try json.appendSlice(gpa, "\"kv\": [");
    for (0..layer_count) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
        const rows = f.kvRows(n, layer, which);
        const name = try std.fmt.bufPrint(&name_buffer, "native-{d}-{s}-layer-{d:0>2}-{s}.bf16", .{ n, variant, layer, kind });
        const bytes = std.mem.sliceAsBytes(rows);
        try dir.writeFile(io, .{ .sub_path = name, .data = bytes, .flags = .{ .exclusive = true } });
        written += bytes.len;
        const bad = nonfiniteBf16(rows);
        if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
        const sha = sha256Hex(bytes);
        try json.print(gpa, "{s}{{\"layer\": {d}, \"kind\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"nonfinite\": {d}}}", .{
            if (layer == 0 and which == 0) "" else ", ", layer, kind, name, sha[0..], n, kv_width, bad,
        });
    };
    try json.appendSlice(gpa, "]}");
    return written;
}

fn seconds(ns: anytype) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-tower-trace CHECKPOINT INPUT_DIRECTORY OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const input = try std.Io.Dir.cwd().openDir(io, args[2], .{});
    defer input.close(io);
    const out = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer out.close(io);

    // Every input is read before any CUDA call, so a missing or short fixture never touches the device.
    const hidden = try gpa.alloc(u16, max_rows * width);
    defer gpa.free(hidden);
    var frames: [fixture_frames]Frame = undefined;
    try readExact(io, input, "hidden.bf16", std.mem.sliceAsBytes(hidden));
    try readExact(io, input, "frames.u32le", std.mem.sliceAsBytes(&frames));
    const hidden_sha = sha256Hex(std.mem.sliceAsBytes(hidden));
    const frames_sha = sha256Hex(std.mem.sliceAsBytes(&frames));
    for (hidden) |bits| if ((bits & 0x7f80) == 0x7f80) return error.NonfiniteFixtureHidden;

    var finals: [variants.len]Finals = undefined;
    var allocated: usize = 0;
    defer {
        for (finals[0..allocated]) |f| f.free(gpa);
    }
    for (&finals) |*f| {
        f.* = try Finals.alloc(gpa);
        allocated += 1;
    }
    const device_check = try gpa.alloc(u16, max_rows * width);
    defer gpa.free(device_check);

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    // Same options as drama_tower_oracle.zig, so the plan list, weights, cache layout and buffers are the ones it ran.
    var model = try tower.Model.init(gpa, io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer model.deinit();
    const p = model.impl;
    if (p.plans.items.len != max_rows * projection_count or p.head_logits.len != vocab * 2 or p.output_hidden.len != max_rows * width * 2 or
        p.cache.len != layer_count * 2 * context * kv_width * 2) return error.UnexpectedTowerLayout;
    const model_usage = cuda.usage(false);
    var device_hidden = try cuda.DeviceBuffer.alloc(&driver, max_rows * width * 2);
    defer device_hidden.free();
    try device_hidden.upload(0, std.mem.sliceAsBytes(hidden));
    var staging = try cuda.HostBuffer.alloc(&driver, max_rows * max_stage_width * 2);
    defer staging.free();

    var captures: std.ArrayList(u8) = .empty;
    defer captures.deinit(gpa);
    var cases: std.ArrayList(u8) = .empty;
    defer cases.deinit(gpa);
    var written: usize = 0;
    var total_captures: usize = 0;
    var name_buffer: [80]u8 = undefined;
    for (row_counts, 0..) |n, case_index| {
        var traced: Tracer = .{ .io = io, .dir = out, .n = n, .staging = staging.bytes, .first = null };
        const traced_start = std.Io.Clock.awake.now(io).toNanoseconds();
        try runCase(&model, stream, n, device_hidden, &frames, &traced, finals[0]);
        const traced_ns = std.Io.Clock.awake.now(io).toNanoseconds() - traced_start;
        const untraced_start = std.Io.Clock.awake.now(io).toNanoseconds();
        try runCase(&model, stream, n, device_hidden, &frames, {}, finals[1]);
        const untraced_ns = std.Io.Clock.awake.now(io).toNanoseconds() - untraced_start;
        var repeat: Tracer = .{ .io = io, .dir = out, .n = n, .staging = staging.bytes, .first = &traced.hashes };
        try runCase(&model, stream, n, device_hidden, &frames, &repeat, finals[2]);
        total_captures += traced.captures + repeat.captures;
        written += traced.written;

        for ([_]*const Tracer{ &traced, &repeat }, [_][]const u8{ "traced", "repeat" }) |t, label| {
            if (t.captures != capture_count) try errors.add("rows {d} {s}: {d} captures, expected {d}", .{ n, label, t.captures, capture_count });
        }
        if (repeat.repeat_mismatches != 0) try errors.add("rows {d}: {d} stage captures differ between traced runs", .{ n, repeat.repeat_mismatches });
        const untraced_diff = compareFinals(n, finals[0], finals[1]);
        const repeat_diff = compareFinals(n, finals[0], finals[2]);
        for ([_]Diff{ untraced_diff, repeat_diff }, [_][]const u8{ "untraced", "repeat" }) |d, label| {
            if (d.hidden != 0 or d.head9 != 0 or d.kv != 0)
                try errors.add("rows {d}: traced vs {s} bit differences hidden {d} head9 {d} kv {d}", .{ n, label, d.hidden, d.head9, d.kv });
        }

        for (0..capture_count) |k| {
            const slot = expectedSlot(k);
            const at = slotOf(slot.layer, slot.stage);
            const name = try stageFile(&name_buffer, n, slot.layer, slot.stage);
            const bad = traced.nonfinite[at];
            if (bad != 0) try errors.add("{s}: {d} nonfinite values", .{ name, bad });
            const repeat_equal = std.mem.eql(u8, &traced.hashes[at], &repeat.hashes[at]);
            const sha = std.fmt.bytesToHex(traced.hashes[at], .lower);
            try captures.print(gpa, "{s}\n    {{\"rows\": {d}, \"order\": {d}, \"layer\": {d}, \"stage\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"dtype\": \"bf16le\", \"row0\": 0, \"nonfinite\": {d}, \"repeat_bit_equal\": {}}}", .{
                if (captures.items.len == 0) "" else ",", n, k, slot.layer, @tagName(slot.stage), name, sha[0..], n, tower.stageWidth(slot.stage), bad, repeat_equal,
            });
        }

        try cases.print(gpa, "{s}\n    {{\"rows\": {d}, \"head_row\": {d}, \"captures_per_run\": {d}, \"stage_repeat_mismatches\": {d}," ++
            " \"traced_vs_untraced_bit_differences\": {{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}}}," ++
            " \"traced_vs_repeat_bit_differences\": {{\"hidden\": {d}, \"head9\": {d}, \"kv\": {d}}}," ++
            " \"timing_s\": {{\"traced\": {d:.6}, \"untraced\": {d:.6}}}, \"finals\": {{", .{
            if (case_index == 0) "" else ",", n,                   n - 1,                capture_count,      repeat.repeat_mismatches,
            untraced_diff.hidden,             untraced_diff.head9, untraced_diff.kv,     repeat_diff.hidden, repeat_diff.head9,
            repeat_diff.kv,                   seconds(traced_ns),  seconds(untraced_ns),
        });
        for (variants, finals, 0..) |variant, f, i| {
            if (i != 0) try cases.appendSlice(gpa, ", ");
            written += try emitFinals(gpa, io, out, &cases, n, variant, f, &errors);
        }
        try cases.appendSlice(gpa, "}}");
        std.debug.print("tower_rows={d} captures={d} traced_untraced_hidden_diff={d} head9_diff={d} kv_diff={d} repeat_hidden_diff={d} stage_repeat_mismatches={d}\n", .{
            n, traced.captures, untraced_diff.hidden, untraced_diff.head9, untraced_diff.kv, repeat_diff.hidden, repeat.repeat_mismatches,
        });
    }

    // Input immutability: the device rows every case read, and the host fixture bytes, are what was loaded.
    try device_hidden.download(0, std.mem.sliceAsBytes(device_check));
    const device_input_differences = differences(device_check, hidden);
    if (device_input_differences != 0) try errors.add("device input hidden changed in {d} values", .{device_input_differences});
    const hidden_sha_after = sha256Hex(std.mem.sliceAsBytes(hidden));
    const frames_sha_after = sha256Hex(std.mem.sliceAsBytes(&frames));
    const host_hidden_unchanged = std.mem.eql(u8, &hidden_sha_after, &hidden_sha);
    const host_frames_unchanged = std.mem.eql(u8, &frames_sha_after, &frames_sha);
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
    const trace_source_sha = sha256Hex(@embedFile("drama_tower_trace.zig"));
    const kernel_image_sha = sha256Hex(cuda.kernels.drama_tower);
    if (cuda.kernels.drama_tower.len == 0) try errors.add("embedded drama_tower kernel image is empty", .{});
    const evidence_valid = errors.count == 0;

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-tower-trace\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"input_directory\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.print(gpa, ",\n  \"input_hidden_sha256\": \"{s}\",\n  \"input_frames_sha256\": \"{s}\",\n  \"width\": {d},\n  \"dtype\": \"bf16le\",\n  \"layer_count\": {d}," ++
        "\n  \"rows\": [{d}, {d}, {d}],\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}},\n  \"stage_order\": [", .{
        hidden_sha[0..], frames_sha[0..], width, layer_count, row_counts[0], row_counts[1], row_counts[2], context, max_rows,
    });
    for (stages, 0..) |stage, i| try manifest.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", @tagName(stage) });
    try manifest.appendSlice(gpa, "],\n  \"stage_widths\": {");
    for (stages, 0..) |stage, i| try manifest.print(gpa, "{s}\"{s}\": {d}", .{ if (i == 0) "" else ", ", @tagName(stage), tower.stageWidth(stage) });
    try manifest.print(gpa, "}},\n  \"stage_sources\": {s},\n  \"capture_order\": \"shared_input (layer 0) once, then the remaining stages in stage_order for layers 0..12; enforced\"," ++
        "\n  \"row_mapping\": \"row r of every capture and final is request position r (reset position 0; prefill rows 0..N-1, labels frames[r+1]); KV rows are cache positions 0..N-1\"," ++
        "\n  \"head\": {{\"weight\": \"audio_decoder_mtp.heads.{d}.weight\", \"plan\": \"plans[{d}] (rows 1, kind {d})\", \"input\": \"layer 12 output row N-1\"}}," ++
        "\n  \"captures\": [{s}\n  ],\n  \"cases\": [{s}\n  ]," ++
        "\n  \"input_immutable\": {{\"device_hidden_differences\": {d}, \"host_hidden_unchanged\": {}, \"host_frames_unchanged\": {}}}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}, \"trace_staging_pinned_bytes\": {d}}}," ++
        "\n  \"bytes_written\": {d},\n  \"total_captures\": {d}," ++
        "\n  \"binary_sha256\": {s},\n  \"tower_source_sha256\": \"{s}\",\n  \"trace_source_sha256\": \"{s}\",\n  \"tower_kernel_image_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        stage_sources,         head_index,            head_plan,                head_kind,
        captures.items,        cases.items,           device_input_differences, host_hidden_unchanged,
        host_frames_unchanged, model_usage.device,    used.device,              used.peak,
        used.host,             staging.bytes.len,     written,                  total_captures,
        binary_json.items,     tower_source_sha[0..], trace_source_sha[0..],    kernel_image_sha[0..],
        evidence_valid,        errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    std.debug.print("cases={d} captures={d} bytes_written={d} evidence_valid={} errors={d} cuda_bytes={d} cuda_peak={d} pinned_bytes={d}\n", .{
        row_counts.len, total_captures, written + manifest.items.len, evidence_valid, errors.count, used.device, used.peak, used.host,
    });
    return if (evidence_valid) 0 else 3;
}
