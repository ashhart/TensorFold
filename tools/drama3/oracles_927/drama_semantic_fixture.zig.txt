//! Captures paired backbone hidden states and sampled frames in drama_host.zig's order, 128 frames, no codec.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const Frame = drama.tower.Frame;
const width = 2048;
const rows = 128;
const context = 512;

fn plainFrame(token: u32) Frame {
    var frame: Frame = @splat(0);
    frame[0] = token;
    return frame;
}

fn lastRow(buffer: cuda.DeviceBuffer, count: usize, row_bytes: usize) !cuda.DeviceBuffer {
    if (count == 0 or buffer.len != count * row_bytes) return error.DramaRowsMismatch;
    return .{ .d = buffer.d, .ptr = buffer.ptr + (count - 1) * row_bytes, .len = row_bytes, .borrowed = true };
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-semantic-fixture CHECKPOINT TOKENS_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var bytes: [(context - rows) * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const file = try std.Io.Dir.cwd().openFile(init.io, args[2], .{});
    defer file.close(init.io);
    const size = try file.readPositionalAll(init.io, &bytes, 0);
    if (size < 8 or size > (context - rows) * 4 or size % 4 != 0) return error.ExpectedTwoTo384Tokens;
    const tokens = std.mem.bytesAsSlice(u32, @as([]align(@alignOf(u32)) u8, bytes[0..size]));
    for (tokens) |token| if (token >= drama.backbone.config.vocab) return error.InvalidPromptToken;
    var out = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer out.close(init.io);
    const hidden_capture = try init.gpa.alloc(u16, rows * width);
    defer init.gpa.free(hidden_capture);
    var captured_frames: [rows + 1]Frame = undefined;
    var staged: [rows + 1]Frame = undefined;

    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var backbone = try drama.backbone.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = rows });
    defer backbone.deinit();
    var tower = try drama.tower.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = context, .max_rows = rows });
    defer tower.deinit();
    if (backbone.hidden != width) return error.DramaHiddenMismatch;
    for (tokens) |token| if (token >= backbone.semantic_start and token <= backbone.semantic_end) return error.TextFixtureMustNotContainSemanticTokens;
    try backbone.reset(stream);
    try tower.reset(stream, tokens, 0);

    var last: drama.backbone.Rows = undefined;
    var start: usize = 0;
    while (start < tokens.len) {
        const end = @min(start + rows, tokens.len);
        const n = end - start;
        const look = @min(end + 1, tokens.len);
        for (tokens[start..look], staged[0 .. look - start]) |token, *frame| frame.* = plainFrame(token);
        const extra = try tower.setInput(stream, staged[0..n]);
        last = try backbone.run(stream, .prefill, tokens[start..end], extra);
        if (last.rows != n) return error.DramaRowsMismatch;
        const taught = look - start - 1;
        if (taught > 0) try tower.prefill(stream, last.hidden, staged[1 .. 1 + taught]);
        if (end == tokens.len and taught > 0) _ = try tower.setInput(stream, staged[n - 1 .. n]);
        start = end;
    }

    // Row 0 is the final prompt row; frame i + 1 is the label of hidden row i.
    captured_frames[0] = plainFrame(tokens[tokens.len - 1]);
    var hidden = try lastRow(last.hidden, last.rows, width * 2);
    var logits = try lastRow(last.semantic_logits, 1, drama.backbone.weights.head_cols * 4);
    const sampling: drama.tower.Sampling = .{ .temperature = 1, .top_p = 0.9, .top_k = 30 };
    for (0..rows) |i| {
        try stream.synchronize();
        try hidden.download(0, std.mem.sliceAsBytes(hidden_capture[i * width .. (i + 1) * width]));
        const frame = try tower.generate(stream, logits, hidden, sampling);
        if (frame[0] == backbone.eos) {
            std.debug.print("early_eos generated_frames={d}; no padded fixture is written\n", .{i});
            return error.InsufficientGeneratedFrames;
        }
        if (frame[0] < backbone.semantic_start or frame[0] > backbone.semantic_end) return error.BadSpeechFrame;
        if (frame[0] - backbone.semantic_start != frame[1]) return error.SemanticCodeMismatch;
        for (frame[1..]) |code| if (code >= 4096) return error.InvalidCodebookIndex;
        captured_frames[i + 1] = frame;
        if (i + 1 < rows) {
            staged[0] = frame;
            const extra = try tower.setInput(stream, staged[0..1]);
            const decoded = try backbone.run(stream, .decode, staged[0][0..1], extra);
            if (decoded.rows != 1) return error.DramaRowsMismatch;
            hidden = try lastRow(decoded.hidden, 1, width * 2);
            logits = try lastRow(decoded.semantic_logits, 1, drama.backbone.weights.head_cols * 4);
        }
    }
    for (hidden_capture) |bits| {
        const value: f32 = @bitCast(@as(u32, bits) << 16);
        if (!std.math.isFinite(value)) return error.NonFiniteBackboneHidden;
    }
    try out.writeFile(init.io, .{ .sub_path = "hidden.bf16", .data = std.mem.sliceAsBytes(hidden_capture) });
    try out.writeFile(init.io, .{ .sub_path = "frames.u32le", .data = std.mem.sliceAsBytes(&captured_frames) });
    const used = cuda.usage(false);
    var metadata: [1024]u8 = undefined;
    const text = try std.fmt.bufPrint(&metadata, "{{\"prompt_rows\":{d},\"hidden_rows\":{d},\"frame_rows\":{d},\"hidden_width\":{d},\"context\":{d},\"first_position\":{d},\"semantic_start\":{d},\"semantic_end\":{d},\"seed\":0,\"temperature\":1,\"top_p\":0.9,\"top_k\":30,\"stopped_at_fixture_bound\":true,\"codec_loaded\":false,\"cuda_bytes\":{d},\"pinned_bytes\":{d},\"scope\":\"Native-generated fixed-input suffix; independent tower replay rebases positions and omits earlier prompt KV. Not reference generation or complete speech.\"}}\n", .{ tokens.len, rows, rows + 1, width, context, tokens.len - 1, backbone.semantic_start, backbone.semantic_end, used.device, used.host });
    try out.writeFile(init.io, .{ .sub_path = "capture.json", .data = text });
    std.debug.print("{s}", .{text});
    return 0;
}
