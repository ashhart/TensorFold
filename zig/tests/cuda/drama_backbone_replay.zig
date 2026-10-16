//! Diagnostic: replays backbone attention and MoE operators on frozen native and reference stage inputs.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Model = drama.backbone.Model;
const Stage = drama.backbone.Stage;
const Operator = drama.backbone.ReplayOperator;
const linear = drama.backbone.config.linear;
const replay_fixtures = @import("drama_backbone_replay_fixtures.zig");
const Source = replay_fixtures.Source;
const loadSource = replay_fixtures.loadSource;
const loadTokens = replay_fixtures.loadTokens;

pub const width = 2048;
const candidate_count = 4097;
pub const max_rows = 128;
pub const context = 512;
pub const rows = 182;
pub const tensor_bytes = rows * width * 2;
/// The forward's own layer array.
pub const layer_count = @typeInfo(@FieldType(@FieldType(Model, "w"), "layers")).array.len;
pub const contract_tokens_sha256 = "fe39111abb37fc10cb2e5c846313464c38fad13d92d44f082ef465d5b3baba86";
const contract_name = "drama-backbone-replay-contract/1";
pub const replay_layers = [_]usize{ 0, 3, 13, 39 };
/// The native capture's chunks: Model.run over 182 rows at max_rows 128.
pub const chunks = [_]usize{ 128, 54 };
const operators = std.meta.tags(Operator);
pub const Origin = enum(usize) { native, reference };
const origins = std.meta.tags(Origin);
/// Stages read from each source directory per replayed layer: the operators' inputs and original outputs.
pub const used_stages = [_]Stage{ .attention_input, .attention_output, .mlp_input, .mlp_output };

comptime {
    std.debug.assert(drama.backbone.config.hidden == width);
    std.debug.assert(drama.backbone.weights.head_cols == candidate_count);
    var total: usize = 0;
    for (chunks) |c| {
        std.debug.assert(c > 0 and c <= max_rows);
        total += c;
    }
    std.debug.assert(total == rows and rows <= context);
    for (replay_layers) |layer| std.debug.assert(layer < layer_count);
}

fn inputStage(op: Operator) Stage {
    return switch (op) {
        .attention => .attention_input,
        .moe => .mlp_input,
    };
}

fn outputStage(op: Operator) Stage {
    return switch (op) {
        .attention => .attention_output,
        .moe => .mlp_output,
    };
}

/// The production method `Model.replayOperator` runs for this layer and operator.
fn methodName(layer: usize, op: Operator) []const u8 {
    return switch (op) {
        .attention => if (linear(layer)) "deltaNet" else "fullAttention",
        .moe => "moe",
    };
}

pub fn usedIndex(stage: Stage) usize {
    for (used_stages, 0..) |s, i| {
        if (s == stage) return i;
    }
    unreachable;
}

pub fn stageFileName(buffer: []u8, layer: usize, stage: Stage) ![]u8 {
    return std.fmt.bufPrint(buffer, "layer-{d:0>2}-{s}.bf16", .{ layer, @tagName(stage) });
}

/// Recorded failures that leave the outputs written but the evidence invalid.
pub const Errors = struct {
    gpa: Allocator,
    json: std.ArrayList(u8) = .empty,
    count: usize = 0,

    pub fn add(e: *Errors, comptime f: []const u8, args: anytype) !void {
        const message = try std.fmt.allocPrint(e.gpa, f, args);
        defer e.gpa.free(message);
        std.debug.print("error: {s}\n", .{message});
        try e.json.appendSlice(e.gpa, if (e.count == 0) "\n    " else ",\n    ");
        try jsonString(&e.json, e.gpa, message);
        e.count += 1;
    }
};

pub fn nonfiniteBf16(values: []const u16) usize {
    var count: usize = 0;
    for (values) |bits| {
        if ((bits & 0x7f80) == 0x7f80) count += 1;
    }
    return count;
}

fn nonfiniteF32(values: []const f32) usize {
    var count: usize = 0;
    for (values) |v| {
        if (!std.math.isFinite(v)) count += 1;
    }
    return count;
}

fn bf16(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

/// `got` against an original capture `want`, both BF16 [rows, width].
const Comparison = struct {
    bit_differences: usize = 0,
    max_abs: f64 = 0,
    /// ||got - want|| / ||want||.
    relative_l2: f64 = 0,
    min_row_cosine: f64 = 1,
    min_row: usize = 0,

    fn of(got_bytes: []align(16) const u8, want_bytes: []align(16) const u8) Comparison {
        const got = std.mem.bytesAsSlice(u16, got_bytes);
        const want = std.mem.bytesAsSlice(u16, want_bytes);
        var c: Comparison = .{};
        var diff_sq: f64 = 0;
        var want_sq: f64 = 0;
        for (0..rows) |r| {
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got[r * width ..][0..width], want[r * width ..][0..width]) |gb, wb| {
                if (gb != wb) c.bit_differences += 1;
                const g = bf16(gb);
                const w = bf16(wb);
                const d = g - w;
                c.max_abs = @max(c.max_abs, @abs(d));
                diff_sq += d * d;
                dot += g * w;
                gg += g * g;
                ww += w * w;
            }
            want_sq += ww;
            const cosine: f64 = if (gg == 0 and ww == 0) 1 else if (gg == 0 or ww == 0) 0 else dot / (@sqrt(gg) * @sqrt(ww));
            if (r == 0 or cosine < c.min_row_cosine) {
                c.min_row_cosine = cosine;
                c.min_row = r;
            }
        }
        c.relative_l2 = if (want_sq == 0) (if (diff_sq == 0) 0 else std.math.inf(f64)) else @sqrt(diff_sq / want_sq);
        return c;
    }

    fn json(c: Comparison, list: *std.ArrayList(u8), gpa: Allocator, want_sha: []const u8) !void {
        try list.print(gpa, "{{\"original_sha256\": \"{s}\", \"bit_differences\": {d}, \"bit_exact\": {}, \"max_abs\": ", .{ want_sha, c.bit_differences, c.bit_differences == 0 });
        try jsonNumber(list, gpa, c.max_abs);
        try list.appendSlice(gpa, ", \"relative_l2\": ");
        try jsonNumber(list, gpa, c.relative_l2);
        try list.appendSlice(gpa, ", \"min_row_cosine\": ");
        try jsonNumber(list, gpa, c.min_row_cosine);
        try list.print(gpa, ", \"min_row\": {d}}}", .{c.min_row});
    }
};

/// A finite number, else null (JSON has no NaN or infinity).
fn jsonNumber(list: *std.ArrayList(u8), gpa: Allocator, v: f64) !void {
    if (std.math.isFinite(v)) try list.print(gpa, "{e}", .{v}) else try list.appendSlice(gpa, "null");
}

pub fn sha256Hex(bytes: []const u8) [64]u8 {
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

fn nowNs(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn seconds(ns: i128) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e9;
}

/// What the GPU phase measured besides the per-capture JSON.
const Gpu = struct {
    after_init: cuda.Usage,
    end: cuda.Usage,
    replay_pinned_bytes: usize,
    captures: usize,
    replay_ns: i128,
    run_after_replay_refused: bool,
    replay_after_replay_refused: bool,
    final_hidden_sha: [64]u8,
    semantic_logits_sha: [64]u8,
    production_bit_exact: bool,
};

fn replayAll(gpa: Allocator, io: std.Io, checkpoint: []const u8, sources: *const [origins.len]Source, tokens: []const u32, out: std.Io.Dir, errors: *Errors, captures: *std.ArrayList(u8), self_control: *std.ArrayList(u8)) !Gpu {
    const native = &sources[@backingInt(Origin.native)];
    const reference = &sources[@backingInt(Origin.reference)];
    const hidden = try gpa.alloc(u16, rows * width);
    defer gpa.free(hidden);
    var logits: [candidate_count]f32 = undefined;

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try Model.init(gpa, io, &ctx, checkpoint, "", .{ .context = context, .max_rows = max_rows });
    defer model.deinit();
    if (model.hidden != width) return error.UnexpectedBackboneWidth;
    const after_init = cuda.usage(false);
    var input = try cuda.HostBuffer.alloc(&driver, tensor_bytes);
    defer input.free();
    var output = try cuda.HostBuffer.alloc(&driver, tensor_bytes);
    defer output.free();
    const in_bytes: []align(16) u8 = input.bytes[0..tensor_bytes];
    const out_bytes: []align(16) u8 = output.bytes[0..tensor_bytes];
    for (origins) |origin| try out.createDir(io, @tagName(origin), .default_dir);

    // 2-3. replay every origin, layer and operator from a fresh reset; compare against both original captures
    var replay_ns: i128 = 0;
    var count: usize = 0;
    var name_buffer: [64]u8 = undefined;
    var path_buffer: [96]u8 = undefined;
    for (origins) |origin| {
        const src = &sources[@backingInt(origin)];
        for (replay_layers, 0..) |layer, li| {
            for (operators) |op| {
                const in_stage = inputStage(op);
                const out_stage = outputStage(op);
                @memcpy(in_bytes, src.tensor(li, in_stage));
                @memset(out_bytes, 0xff); // BF16 NaN: a row the replay never wrote would show as nonfinite
                try model.reset(stream);
                const start = nowNs(io);
                try model.replayOperator(stream, layer, op, &chunks, in_bytes, out_bytes);
                const ns = nowNs(io) - start;
                replay_ns += ns;
                count += 1;

                const name = try stageFileName(&name_buffer, layer, out_stage);
                const path = try std.fmt.bufPrint(&path_buffer, "{s}/{s}", .{ @tagName(origin), name });
                try out.writeFile(io, .{ .sub_path = path, .data = out_bytes, .flags = .{ .exclusive = true } });
                const sha = sha256Hex(out_bytes);
                const nonfinite = nonfiniteBf16(std.mem.bytesAsSlice(u16, out_bytes));
                if (nonfinite != 0) try errors.add("{s}: {d} nonfinite values", .{ path, nonfinite });
                const vs_native = Comparison.of(out_bytes, native.tensor(li, out_stage));
                const vs_reference = Comparison.of(out_bytes, reference.tensor(li, out_stage));

                try captures.print(gpa, "{s}\n    {{\"origin\": \"{s}\", \"layer\": {d}, \"stage\": \"{s}\", \"method\": \"{s}\", \"input_stage\": \"{s}\", \"input_sha256\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"chunks\": [{d}, {d}], \"nonfinite\": {d}, \"replay_s\": {d:.6}, \"vs_native_original\": ", .{
                    if (captures.items.len == 0) "" else ",", @tagName(origin), layer, @tagName(out_stage), methodName(layer, op), @tagName(in_stage), src.sha(li, in_stage), path, sha[0..], rows, width, chunks[0], chunks[1], nonfinite, seconds(ns),
                });
                try vs_native.json(captures, gpa, native.sha(li, out_stage));
                try captures.appendSlice(gpa, ", \"vs_reference_original\": ");
                try vs_reference.json(captures, gpa, reference.sha(li, out_stage));
                try captures.appendSlice(gpa, "}");

                if (origin == .native) {
                    if (vs_native.bit_differences != 0) try errors.add("native self-control {s}: {d} bit differences vs the original native capture", .{ path, vs_native.bit_differences });
                    try self_control.print(gpa, "{s}\n    {{\"layer\": {d}, \"stage\": \"{s}\", \"method\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"original_sha256\": \"{s}\", \"bit_differences\": {d}, \"bit_exact\": {}}}", .{
                        if (self_control.items.len == 0) "" else ",", layer, @tagName(out_stage), methodName(layer, op), path, sha[0..], native.sha(li, out_stage), vs_native.bit_differences, vs_native.bit_differences == 0,
                    });
                }
                std.debug.print("{s}: {s} sha256={s} vs_native_bits={d} vs_reference_bits={d} replay_s={d:.6}\n", .{ path, methodName(layer, op), sha[0..], vs_native.bit_differences, vs_reference.bit_differences, seconds(ns) });
            }
        }
    }
    if (count != origins.len * replay_layers.len * operators.len) try errors.add("{d} replays, expected {d}", .{ count, origins.len * replay_layers.len * operators.len });

    // 4. poison guard, then the production run over the contract prompt
    const run_refused = if (model.run(stream, .prefill, tokens[0..1], null)) |_| false else |err| err == error.DramaStateNeedsReset;
    if (!run_refused) try errors.add("Model.run after a replay was not refused with DramaStateNeedsReset", .{});
    const replay_refused = if (model.replayOperator(stream, replay_layers[0], .moe, &chunks, in_bytes, out_bytes)) |_| false else |err| err == error.DramaStateNeedsReset;
    if (!replay_refused) try errors.add("replayOperator after a replay was not refused with DramaStateNeedsReset", .{});
    try model.reset(stream);
    var at: usize = 0;
    for (chunks) |n| {
        const result = try model.run(stream, .prefill, tokens[at..][0..n], null);
        if (result.rows != n) return error.UnexpectedBackboneRows;
        try result.hidden.download(0, std.mem.sliceAsBytes(hidden[at * width ..][0 .. n * width]));
        at += n;
        if (at == rows) try result.semantic_logits.download(0, std.mem.sliceAsBytes(&logits));
    }
    const end = cuda.usage(false);
    const hidden_sha = sha256Hex(std.mem.sliceAsBytes(hidden));
    const logits_sha = sha256Hex(std.mem.sliceAsBytes(&logits));
    const hidden_nonfinite = nonfiniteBf16(hidden);
    const logits_nonfinite = nonfiniteF32(&logits);
    if (hidden_nonfinite != 0 or logits_nonfinite != 0) try errors.add("production control: {d} hidden and {d} logit nonfinite values", .{ hidden_nonfinite, logits_nonfinite });
    const hidden_equal = std.mem.eql(u8, &hidden_sha, native.final_hidden_sha.?);
    const logits_equal = std.mem.eql(u8, &logits_sha, native.semantic_logits_sha.?);
    if (!hidden_equal) try errors.add("production control: final hidden sha256 {s} differs from the native trace {s}", .{ hidden_sha[0..], native.final_hidden_sha.? });
    if (!logits_equal) try errors.add("production control: semantic logits sha256 {s} differ from the native trace {s}", .{ logits_sha[0..], native.semantic_logits_sha.? });
    return .{
        .after_init = after_init,
        .end = end,
        .replay_pinned_bytes = input.bytes.len + output.bytes.len,
        .captures = count,
        .replay_ns = replay_ns,
        .run_after_replay_refused = run_refused,
        .replay_after_replay_refused = replay_refused,
        .final_hidden_sha = hidden_sha,
        .semantic_logits_sha = logits_sha,
        .production_bit_exact = hidden_equal and logits_equal,
    };
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-backbone-replay CHECKPOINT NATIVE_STAGE_DIR REFERENCE_STAGE_DIR OUTPUT_DIR\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    const out = try std.Io.Dir.cwd().openDir(io, args[4], .{});
    defer out.close(io);

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);

    // 1. fixtures
    var sources: [origins.len]Source = undefined;
    for (origins) |origin| sources[@backingInt(origin)] = try loadSource(arena, io, origin, args[2 + @backingInt(origin)], &errors);
    const native = &sources[@backingInt(Origin.native)];
    const tokens = try loadTokens(arena, io, native, &errors);
    var fixture_host_bytes: usize = 0;
    for (sources) |src| {
        for (src.tensors) |layer_tensors| {
            for (layer_tensors) |t| fixture_host_bytes += if (t) |bytes| bytes.len else 0;
        }
    }

    var captures: std.ArrayList(u8) = .empty;
    defer captures.deinit(gpa);
    var self_control: std.ArrayList(u8) = .empty;
    defer self_control.deinit(gpa);
    const gpu: ?Gpu = if (errors.count == 0)
        try replayAll(gpa, io, args[1], &sources, tokens.?, out, &errors, &captures, &self_control)
    else blk: {
        std.debug.print("fixture validation failed: no model load, no replay\n", .{});
        break :blk null;
    };

    // 5. manifest
    const exe_path = "/proc/self/exe";
    var binary_json: std.ArrayList(u8) = .empty;
    defer binary_json.deinit(gpa);
    if (fileSha256(gpa, io, exe_path)) |sha| {
        try binary_json.print(gpa, "\"{s}\"", .{sha[0..]});
    } else |err| {
        std.debug.print("binary hash unavailable: {s}\n", .{@errorName(err)});
        try binary_json.appendSlice(gpa, "null");
    }
    const backbone_source_sha = sha256Hex(drama.backbone.trace_source);
    const replay_source_sha = sha256Hex(@embedFile("drama_backbone_replay.zig"));
    const evidence_valid = errors.count == 0;

    var m: std.ArrayList(u8) = .empty;
    defer m.deinit(gpa);
    try m.print(gpa, "{{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-backbone-replay\",\n  \"contract\": \"{s}\",\n  \"checkpoint\": ", .{contract_name});
    try jsonString(&m, gpa, args[1]);
    try m.appendSlice(gpa, ",\n  \"origins\": [");
    for (sources, 0..) |src, i| {
        try m.print(gpa, "{s}\n    {{\"origin\": \"{s}\", \"stage_dir\": ", .{ if (i == 0) "" else ",", @tagName(src.origin) });
        try jsonString(&m, gpa, src.path);
        if (src.manifest_sha) |sha| try m.print(gpa, ", \"manifest_sha256\": \"{s}\"", .{sha[0..]}) else try m.appendSlice(gpa, ", \"manifest_sha256\": null");
        try m.print(gpa, ", \"manifest_evidence_valid\": {}}}", .{src.manifest_evidence_valid});
    }
    try m.appendSlice(gpa, "\n  ],\n  \"tokens_file\": ");
    if (native.tokens_file) |path| try jsonString(&m, gpa, path) else try m.appendSlice(gpa, "null");
    try m.print(gpa, ",\n  \"tokens_sha256\": {s}{s}{s},\n  \"rows\": {d},\n  \"width\": {d},\n  \"dtype\": \"bf16le\",\n  \"layer_count\": {d},\n  \"layers\": [", .{
        if (tokens != null) "\"" else "", if (tokens != null) contract_tokens_sha256 else "null", if (tokens != null) "\"" else "", rows, width, layer_count,
    });
    for (replay_layers, 0..) |layer, i| try m.print(gpa, "{s}{d}", .{ if (i == 0) "" else ", ", layer });
    try m.print(gpa, "],\n  \"chunks\": [{d}, {d}],\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}}," ++
        "\n  \"replay\": \"per origin, layer and operator: Model.reset, then Model.replayOperator, which uploads each chunk of the frozen input into Scratch.normed and runs the production method at request position row0; MoE replays the frozen mlp_input\"," ++
        "\n  \"inputs\": [", .{ chunks[0], chunks[1], context, max_rows });
    var first = true;
    var name_buffer: [64]u8 = undefined;
    for (sources) |src| {
        for (replay_layers, 0..) |layer, li| {
            for (used_stages, 0..) |stage, si| {
                if (src.tensors[li][si] == null) continue;
                const name = try stageFileName(&name_buffer, layer, stage);
                try m.print(gpa, "{s}\n    {{\"origin\": \"{s}\", \"layer\": {d}, \"stage\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}]}}", .{
                    if (first) "" else ",", @tagName(src.origin), layer, @tagName(stage), name, src.shas[li][si][0..], rows, width,
                });
                first = false;
            }
        }
    }
    try m.print(gpa, "\n  ],\n  \"captures\": [{s}\n  ],\n  \"native_self_control\": [{s}\n  ],", .{ captures.items, self_control.items });
    if (gpu) |g| {
        try m.print(gpa, "\n  \"production_control\": {{\"kind\": \"after all replays: Model.run and replayOperator without reset must be refused; then reset and production Model.run over the tokens_file in chunks 128 + 54\", \"run_after_replay_refused\": {}, \"replay_after_replay_refused\": {}, \"final_hidden_sha256\": \"{s}\", \"expected_final_hidden_sha256\": \"{s}\", \"semantic_logits_sha256\": \"{s}\", \"expected_semantic_logits_sha256\": \"{s}\", \"bit_exact\": {}}}," ++
            "\n  \"timing_s\": {{\"replay_total\": {d:.6}}}," ++
            "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}, \"replay_pinned_bytes\": {d}, \"fixture_host_bytes\": {d}}},", .{
            g.run_after_replay_refused, g.replay_after_replay_refused, g.final_hidden_sha[0..], native.final_hidden_sha.?, g.semantic_logits_sha[0..], native.semantic_logits_sha.?, g.production_bit_exact,
            seconds(g.replay_ns),       g.after_init.device,           g.end.device,            g.end.peak,                g.end.host,                 g.replay_pinned_bytes,        fixture_host_bytes,
        });
    } else {
        try m.print(gpa, "\n  \"production_control\": null,\n  \"timing_s\": null,\n  \"allocations\": {{\"cuda_device_bytes_after_init\": null, \"fixture_host_bytes\": {d}}},", .{fixture_host_bytes});
    }
    try m.print(gpa, "\n  \"binary_sha256\": {s},\n  \"backbone_source_sha256\": \"{s}\",\n  \"replay_source_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{ binary_json.items, backbone_source_sha[0..], replay_source_sha[0..], evidence_valid, errors.json.items });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = m.items, .flags = .{ .exclusive = true } });

    std.debug.print("replays={d} evidence_valid={} errors={d}\n", .{ if (gpu) |g| g.captures else 0, evidence_valid, errors.count });
    return if (evidence_valid) 0 else 3;
}
