//! Runs the production backbone across chunk sizes and reset on 2..512 u32 text tokens; the caller owns the GPU.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const width = 2048;
const candidate_count = 4097;

fn runChunks(model: *drama.backbone.Model, stream: cuda.Stream, tokens: []const u32, chunk: usize, hidden: []u16, logits: []f32) !void {
    try model.reset(stream);
    var at: usize = 0;
    while (at < tokens.len) {
        const end = @min(at + chunk, tokens.len);
        const output = try model.run(stream, .prefill, tokens[at..end], null);
        try output.hidden.download(0, std.mem.sliceAsBytes(hidden[at * width .. end * width]));
        if (end == tokens.len) try output.semantic_logits.download(0, std.mem.sliceAsBytes(logits));
        at = end;
    }
    for (hidden) |bits| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonFiniteBackboneHidden;
    }
    for (logits) |value| if (!std.math.isFinite(value)) return error.NonFiniteBackboneLogits;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-backbone-oracle CHECKPOINT TOKENS_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var bytes: [512 * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const file = try std.Io.Dir.cwd().openFile(init.io, args[2], .{});
    defer file.close(init.io);
    const count = try file.readPositionalAll(init.io, &bytes, 0);
    if (count < 8 or count > 512 * 4 or count % 4 != 0) return error.ExpectedTwoTo512Tokens;
    const tokens = std.mem.bytesAsSlice(u32, @as([]align(@alignOf(u32)) u8, bytes[0..count]));
    var out = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer out.close(init.io);
    const baseline = try init.gpa.alloc(u16, tokens.len * width);
    defer init.gpa.free(baseline);
    const actual = try init.gpa.alloc(u16, tokens.len * width);
    defer init.gpa.free(actual);
    var baseline_logits: [candidate_count]f32 = undefined;
    var logits: [candidate_count]f32 = undefined;
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var model = try drama.backbone.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = 512, .max_rows = 128 });
    defer model.deinit();
    for (tokens) |token| if (token >= model.semantic_start and token <= model.semantic_end) return error.TextFixtureMustNotContainSemanticTokens;
    var all_equal = true;
    var name_buffer: [128]u8 = undefined;
    for ([_]usize{ 1, 7, 128 }) |chunk| {
        const started = std.Io.Clock.awake.now(init.io).toNanoseconds();
        try runChunks(&model, stream, tokens, chunk, actual, &logits);
        const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - started;
        var name = try std.fmt.bufPrint(&name_buffer, "native-chunk-{d}-hidden.bf16", .{chunk});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(actual) });
        name = try std.fmt.bufPrint(&name_buffer, "native-chunk-{d}-semantic.f32", .{chunk});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(&logits) });
        if (chunk == 1) {
            @memcpy(baseline, actual);
            baseline_logits = logits;
        }
        var hidden_differences: usize = 0;
        for (baseline, actual) |a, b| if (a != b) {
            hidden_differences += 1;
        };
        const head_equal = std.mem.eql(u8, std.mem.sliceAsBytes(&baseline_logits), std.mem.sliceAsBytes(&logits));
        all_equal = all_equal and hidden_differences == 0 and head_equal;
        std.debug.print("rows={d} chunk={d} elapsed_s={d:.6} hidden_bit_differences={d} semantic_logits_bit_equal={}\n", .{
            tokens.len, chunk, @as(f64, @floatFromInt(elapsed)) / 1e9, hidden_differences, head_equal,
        });
    }
    // Repeat the exact last execution after reset; chunk consistency and reset are separate invariants.
    @memcpy(baseline, actual);
    baseline_logits = logits;
    try runChunks(&model, stream, tokens, 128, actual, &logits);
    const reset_equal = std.mem.eql(u16, baseline, actual) and std.mem.eql(u8, std.mem.sliceAsBytes(&baseline_logits), std.mem.sliceAsBytes(&logits));
    var metadata: [512]u8 = undefined;
    const text = try std.fmt.bufPrint(&metadata, "{{\"rows\":{d},\"hidden_width\":{d},\"eos\":{d},\"semantic_start\":{d},\"semantic_end\":{d},\"chunk_bit_exact\":{},\"reset_bit_exact\":{}}}\n", .{ tokens.len, width, model.eos, model.semantic_start, model.semantic_end, all_equal, reset_equal });
    try out.writeFile(init.io, .{ .sub_path = "native-metadata.json", .data = text });
    const used = cuda.usage(false);
    std.debug.print("reset_bit_exact={} cuda_bytes={d} pinned_bytes={d}\n", .{ reset_equal, used.device, used.host });
    return if (all_equal and reset_equal) 0 else 1;
}
