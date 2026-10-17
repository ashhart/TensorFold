//! Diagnostic: captures six stages of every backbone layer from the production forward, checked against run.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;
const Model = drama.backbone.Model;
const Stage = drama.backbone.Stage;

const width = 2048;
const candidate_count = 4097;
const max_rows = 128;
const context = 512;
const stages = std.meta.tags(Stage);
const stage_count = stages.len;
/// The forward's own layer array: every layer it iterates is captured.
const layer_count = @typeInfo(@FieldType(@FieldType(Model, "w"), "layers")).array.len;
const contract_tokens_sha256 = "fe39111abb37fc10cb2e5c846313464c38fad13d92d44f082ef465d5b3baba86";

comptime {
    std.debug.assert(drama.backbone.config.hidden == width);
    std.debug.assert(drama.backbone.weights.head_cols == candidate_count);
}

/// What each stage file holds, in `Model.forward`'s terms.
const stage_sources =
    \\{"attention_input": "Scratch.normed after addNorm(x, residual, ln1): input RMSNorm output",
    \\    "attention_residual": "Scratch.residual after addNorm(x, residual, ln1): updated residual (layer 0: the embedding)",
    \\    "attention_output": "Scratch.x after deltaNet or fullAttention: out projection, before the second add-norm",
    \\    "mlp_input": "Scratch.normed after addNorm(x, residual, ln2): post-attention RMSNorm output",
    \\    "mlp_residual": "Scratch.residual after addNorm(x, residual, ln2): updated residual",
    \\    "mlp_output": "Scratch.x after moe: routed experts combined with the sigmoid-gated shared expert"}
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

/// The runTraced tracer: each (layer, stage) file is written at its rows' offset and hashed as it goes.
const Tracer = struct {
    io: std.Io,
    dir: std.Io.Dir,
    rows: usize,
    /// Pinned, max_rows * width * 2 bytes: the target of the stream-ordered copy.
    staging: []align(16) u8,
    /// Rows captured so far per (layer, stage); a capture must start exactly there.
    next_row: [slots]usize = @splat(0),
    hashes: [slots]std.crypto.hash.sha2.Sha256 = @splat(std.crypto.hash.sha2.Sha256.init(.{})),
    nonfinite: [slots]usize = @splat(0),
    captures: usize = 0,
    written: usize = 0,

    const slots = layer_count * stage_count;

    fn slotOf(layer: usize, stage: Stage) usize {
        return layer * stage_count + @backingInt(stage);
    }

    fn fileName(buffer: []u8, layer: usize, stage: Stage) ![]u8 {
        return std.fmt.bufPrint(buffer, "layer-{d:0>2}-{s}.bf16", .{ layer, @tagName(stage) });
    }

    pub fn capture(t: *Tracer, stream: cuda.Stream, layer: usize, stage: Stage, src: cuda.DeviceBuffer, row0: usize) !void {
        if (layer >= layer_count) return error.TraceLayerOutOfRange;
        if (src.len == 0 or src.len % (width * 2) != 0 or src.len > t.staging.len) return error.TraceCaptureShape;
        const n = src.len / (width * 2);
        if (row0 > t.rows or n > t.rows - row0) return error.TraceRowsOutOfRange;
        const at = slotOf(layer, stage);
        if (row0 != t.next_row[at]) return error.TraceDuplicateOrMissingRows;
        // Queued behind the stage's kernel and ahead of the next one, then drained before the forward continues.
        const bytes = t.staging[0..src.len];
        try src.downloadAsync(0, bytes, stream.handle);
        try stream.synchronize();
        var name_buffer: [64]u8 = undefined;
        const name = try fileName(&name_buffer, layer, stage);
        const file = if (row0 == 0)
            try t.dir.createFile(t.io, name, .{ .exclusive = true })
        else
            try t.dir.openFile(t.io, name, .{ .mode = .write_only });
        defer file.close(t.io);
        try file.writePositionalAll(t.io, bytes, row0 * width * 2);
        t.hashes[at].update(bytes);
        t.nonfinite[at] += nonfiniteBf16(std.mem.bytesAsSlice(u16, bytes));
        t.next_row[at] = row0 + n;
        t.captures += 1;
        t.written += bytes.len;
    }
};

/// Reset, then the whole prompt in max_rows chunks: `Model.run` for a void trace, else `Model.runTraced`.
fn runPrompt(model: *Model, stream: cuda.Stream, tokens: []const u32, trace: anytype, hidden: []u16, logits: *[candidate_count]f32) !void {
    try model.reset(stream);
    var at: usize = 0;
    while (at < tokens.len) {
        const end = @min(at + max_rows, tokens.len);
        const output = if (@TypeOf(trace) == void)
            try model.run(stream, .prefill, tokens[at..end], null)
        else
            try model.runTraced(stream, .prefill, tokens[at..end], null, trace);
        if (output.rows != end - at) return error.UnexpectedBackboneRows;
        try output.hidden.download(0, std.mem.sliceAsBytes(hidden[at * width .. end * width]));
        if (end == tokens.len) try output.semantic_logits.download(0, std.mem.sliceAsBytes(logits));
        at = end;
    }
}

fn nonfiniteBf16(values: []const u16) usize {
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

/// Writes one new file and returns its SHA-256.
fn emit(dir: std.Io.Dir, io: std.Io, name: []const u8, data: []const u8) ![64]u8 {
    try dir.writeFile(io, .{ .sub_path = name, .data = data, .flags = .{ .exclusive = true } });
    return sha256Hex(data);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-backbone-trace CHECKPOINT TOKENS_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;

    var bytes: [context * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const input = try std.Io.Dir.cwd().openFile(io, args[2], .{});
    defer input.close(io);
    const count = try input.readPositionalAll(io, &bytes, 0);
    if (count < 8 or count > context * 4 or count % 4 != 0) return error.ExpectedTwoTo512Tokens;
    const tokens = std.mem.bytesAsSlice(u32, @as([]align(@alignOf(u32)) u8, bytes[0..count]));
    const tokens_sha = sha256Hex(bytes[0..count]);
    const rows = tokens.len;
    const chunks = (rows + max_rows - 1) / max_rows;
    const out = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer out.close(io);

    var errors: Errors = .{ .gpa = gpa };
    defer errors.json.deinit(gpa);
    if (!std.mem.eql(u8, &tokens_sha, contract_tokens_sha256)) try errors.add("tokens_sha256 {s} is not the contract prompt {s}", .{ tokens_sha[0..], contract_tokens_sha256 });

    const traced_hidden = try gpa.alloc(u16, rows * width);
    defer gpa.free(traced_hidden);
    const plain_hidden = try gpa.alloc(u16, rows * width);
    defer gpa.free(plain_hidden);
    const trace_host_bytes = (traced_hidden.len + plain_hidden.len) * 2 + @sizeOf(Tracer);
    var traced_logits: [candidate_count]f32 = undefined;
    var plain_logits: [candidate_count]f32 = undefined;

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try Model.init(gpa, io, &ctx, args[1], "", .{ .context = context, .max_rows = max_rows });
    defer model.deinit();
    if (model.hidden != width) return error.UnexpectedBackboneWidth;
    for (tokens) |token| if (token >= model.semantic_start and token <= model.semantic_end) return error.TextFixtureMustNotContainSemanticTokens;
    const model_usage = cuda.usage(false);
    var staging = try cuda.HostBuffer.alloc(&driver, max_rows * width * 2);
    defer staging.free();

    // 1. traced
    var tracer: Tracer = .{ .io = io, .dir = out, .rows = rows, .staging = staging.bytes };
    const traced_start = std.Io.Clock.awake.now(io).toNanoseconds();
    try runPrompt(&model, stream, tokens, &tracer, traced_hidden, &traced_logits);
    const traced_ns = std.Io.Clock.awake.now(io).toNanoseconds() - traced_start;

    // 2. control: production run, no tracer
    const plain_start = std.Io.Clock.awake.now(io).toNanoseconds();
    try runPrompt(&model, stream, tokens, {}, plain_hidden, &plain_logits);
    const plain_ns = std.Io.Clock.awake.now(io).toNanoseconds() - plain_start;
    const used = cuda.usage(false);

    var hidden_differences: usize = 0;
    for (traced_hidden, plain_hidden) |a, b| {
        if (a != b) hidden_differences += 1;
    }
    const logits_equal = std.mem.eql(u8, std.mem.sliceAsBytes(&traced_logits), std.mem.sliceAsBytes(&plain_logits));
    const control_exact = hidden_differences == 0 and logits_equal;
    if (!control_exact) try errors.add("traced vs untraced: {d} final hidden bit differences, semantic logits bit equal {}", .{ hidden_differences, logits_equal });
    if (tracer.captures != layer_count * stage_count * chunks) try errors.add("{d} captures, expected {d}", .{ tracer.captures, layer_count * stage_count * chunks });

    // 3. manifest entries for the stage files the tracer wrote; then the final outputs
    var captures: std.ArrayList(u8) = .empty;
    defer captures.deinit(gpa);
    var name_buffer: [64]u8 = undefined;
    for (0..layer_count) |layer| {
        for (stages) |stage| {
            const name = try Tracer.fileName(&name_buffer, layer, stage);
            const at = Tracer.slotOf(layer, stage);
            const got = tracer.next_row[at];
            if (got != rows) try errors.add("{s}: captured {d} of {d} rows", .{ name, got, rows });
            const nonfinite = tracer.nonfinite[at];
            if (nonfinite != 0) try errors.add("{s}: {d} nonfinite values", .{ name, nonfinite });
            const sha = std.fmt.bytesToHex(tracer.hashes[at].finalResult(), .lower);
            try captures.print(gpa, "{s}\n    {{\"layer\": {d}, \"stage\": \"{s}\", \"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"rows_captured\": {d}, \"nonfinite\": {d}}}", .{
                if (captures.items.len == 0) "" else ",", layer, @tagName(stage), name, sha[0..], rows, width, got, nonfinite,
            });
        }
    }
    var written = tracer.written;
    const finals = [_]struct { name: []const u8, bytes: []const u8, nonfinite: usize }{
        .{ .name = "final-hidden.bf16", .bytes = std.mem.sliceAsBytes(traced_hidden), .nonfinite = nonfiniteBf16(traced_hidden) },
        .{ .name = "semantic-logits.f32", .bytes = std.mem.sliceAsBytes(&traced_logits), .nonfinite = nonfiniteF32(&traced_logits) },
        .{ .name = "untraced-final-hidden.bf16", .bytes = std.mem.sliceAsBytes(plain_hidden), .nonfinite = nonfiniteBf16(plain_hidden) },
        .{ .name = "untraced-semantic-logits.f32", .bytes = std.mem.sliceAsBytes(&plain_logits), .nonfinite = nonfiniteF32(&plain_logits) },
    };
    var final_sha: [finals.len][64]u8 = undefined;
    for (finals, &final_sha) |f, *sha| {
        if (f.nonfinite != 0) try errors.add("{s}: {d} nonfinite values", .{ f.name, f.nonfinite });
        sha.* = try emit(out, io, f.name, f.bytes);
        written += f.bytes.len;
    }

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
    const trace_source_sha = sha256Hex(@embedFile("drama_backbone_trace.zig"));
    const evidence_valid = errors.count == 0;

    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(gpa);
    try manifest.appendSlice(gpa, "{\n  \"schema_version\": 1,\n  \"producer\": \"tf-drama-backbone-trace\",\n  \"checkpoint\": ");
    try jsonString(&manifest, gpa, args[1]);
    try manifest.appendSlice(gpa, ",\n  \"tokens_file\": ");
    try jsonString(&manifest, gpa, args[2]);
    try manifest.print(gpa, ",\n  \"tokens_sha256\": \"{s}\",\n  \"contract_tokens_sha256\": \"{s}\",\n  \"rows\": {d},\n  \"width\": {d},\n  \"dtype\": \"bf16le\",\n  \"layer_count\": {d}," ++
        "\n  \"stages\": [", .{ tokens_sha[0..], contract_tokens_sha256, rows, width, layer_count });
    for (stages, 0..) |stage, i| try manifest.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", @tagName(stage) });
    try manifest.print(gpa, "],\n  \"stage_sources\": {s},\n  \"row_order\": \"logical request position; each chunk's rows land at Model.pos\"," ++
        "\n  \"model_options\": {{\"context\": {d}, \"max_rows\": {d}}},\n  \"chunks\": [", .{ stage_sources, context, max_rows });
    for (0..chunks) |c| try manifest.print(gpa, "{s}{d}", .{ if (c == 0) "" else ", ", @min(max_rows, rows - c * max_rows) });
    try manifest.print(gpa, "],\n  \"eos\": {d},\n  \"semantic_start\": {d},\n  \"semantic_end\": {d},\n  \"captures\": [{s}\n  ]," ++
        "\n  \"final_hidden\": {{\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}, {d}], \"dtype\": \"bf16le\", \"source\": \"Model.run Rows.hidden, final RMSNorm, traced run\"}}," ++
        "\n  \"semantic_logits\": {{\"file\": \"{s}\", \"sha256\": \"{s}\", \"shape\": [{d}], \"dtype\": \"f32le\", \"row\": {d}, \"layout\": \"EOS then semantic_start..semantic_end\", \"source\": \"Model.run Rows.semantic_logits, last row, traced run\"}},", .{
        model.eos,      model.semantic_start, model.semantic_end, captures.items,
        finals[0].name, final_sha[0][0..],    rows,               width,
        finals[1].name, final_sha[1][0..],    candidate_count,    rows - 1,
    });
    try manifest.print(gpa, "\n  \"control\": {{\"kind\": \"reset, then production Model.run (void trace) over the same chunks\", \"untraced_final_hidden\": {{\"file\": \"{s}\", \"sha256\": \"{s}\"}}, \"untraced_semantic_logits\": {{\"file\": \"{s}\", \"sha256\": \"{s}\"}}, \"final_hidden_bit_differences\": {d}, \"semantic_logits_bit_equal\": {}, \"traced_untraced_bit_exact\": {}}}," ++
        "\n  \"timing_s\": {{\"traced\": {d:.6}, \"untraced\": {d:.6}}}," ++
        "\n  \"allocations\": {{\"cuda_device_bytes_after_init\": {d}, \"cuda_device_bytes\": {d}, \"cuda_device_peak_bytes\": {d}, \"pinned_host_bytes\": {d}, \"trace_staging_pinned_bytes\": {d}, \"trace_host_bytes\": {d}}}," ++
        "\n  \"binary_sha256\": {s},\n  \"backbone_source_sha256\": \"{s}\",\n  \"trace_source_sha256\": \"{s}\"," ++
        "\n  \"evidence_valid\": {},\n  \"errors\": [{s}\n  ]\n}}\n", .{
        finals[2].name,                          final_sha[2][0..],     finals[3].name,   final_sha[3][0..],
        hidden_differences,                      logits_equal,          control_exact,    @as(f64, @floatFromInt(traced_ns)) / 1e9,
        @as(f64, @floatFromInt(plain_ns)) / 1e9, model_usage.device,    used.device,      used.peak,
        used.host,                               staging.bytes.len,     trace_host_bytes, binary_json.items,
        backbone_source_sha[0..],                trace_source_sha[0..], evidence_valid,   errors.json.items,
    });
    try out.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });

    std.debug.print("rows={d} chunks={d} layers={d} captures={d} bytes_written={d} traced_s={d:.6} untraced_s={d:.6} final_hidden_bit_differences={d} semantic_logits_bit_equal={} evidence_valid={} errors={d} cuda_bytes={d} cuda_peak={d} pinned_bytes={d}\n", .{
        rows,                                     chunks,                                  layer_count,        tracer.captures, written + manifest.items.len,
        @as(f64, @floatFromInt(traced_ns)) / 1e9, @as(f64, @floatFromInt(plain_ns)) / 1e9, hidden_differences, logits_equal,    evidence_valid,
        errors.count,                             used.device,                             used.peak,          used.host,
    });
    return if (evidence_valid) 0 else 3;
}
