//! Native 1B HIP stage capture; local checkpoints and reference receipts only, with no qualification claim.
const std = @import("std");
const hip = @import("hip");
const family = @import("lighton_ocr");
const fixture = @import("fixture.zig");
const Allocator = std.mem.Allocator;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: tf-lighton-hip-capture MODEL REFERENCE_DIR OUTPUT_DIR [DEVICE]\n", .{});
        return 2;
    }
    var files = try fixture.Files.init(a, init.io, args[2], args[3]);
    defer files.output.close(init.io);
    try files.verifyCheckpoint(args[1]);
    const c = try family.config.Config.read(a, init.io, args[1]);
    const request = try fixture.field(files.manifest, "request");
    if ((try fixture.list(try fixture.field(request, "images_sha256"))).len != 1) return error.OneImageRequired;
    const prompt = try ids(a, try fixture.field(request, "prompt_ids"), c.vocab);
    const continuation = try ids(a, try fixture.field(request, "continuation_ids"), c.vocab);
    if (prompt.len == 0 or prompt.len + continuation.len > c.context) return error.PastContext;
    const tokens = try std.mem.concat(a, u32, &.{ prompt, continuation });
    const pixels = try files.floats("processor.pixel_values");
    const patches = try fixture.patchify(a, pixels.array.shape[0..pixels.array.rank], pixels.data);
    const shape = try family.tower.Shape.init(c, patches.gh, patches.gw, patches.data.len);
    const image_first = try imageRange(prompt, c.image_token, shape.tokens);
    const device_index = if (args.len == 5) try std.fmt.parseInt(c_int, args[4], 10) else 0;
    if (device_index < 0) return error.BadDevice;
    const caps = hip.Device.capsOf(device_index) orelse return error.UnsupportedGpu;
    const dev: hip.Device = .{ .index = device_index, .caps = caps, .policy = .{ .slots = 0, .graphs = .off } };
    var problem: []const u8 = "";
    const loaded = family.native.open(a, init.gpa, init.io, dev, args[1], family.native.Options{
        .window = tokens.len,
        .streams = 1,
        .fixed = true,
        .cache_gib = 0,
        .vision = .{ .image_tokens = @intCast(shape.tokens), .max_images = 1 },
    }, &problem) catch |err| {
        std.debug.print("native HIP load failed: {s}\n", .{problem});
        return err;
    };
    defer loaded.deinit(loaded.ctx);
    const graph = try loaded.capture();
    std.debug.print("{s}\n", .{loaded.startup});
    var projected = try hip.DeviceBuffer.alloc(graph.r, shape.tokens * c.hidden * 2);
    defer projected.free();
    defer graph.s.synchronize() catch @panic("HIP capture work did not drain before release");
    try graph.tower.encode(patches.data, patches.gh, patches.gw, projected.base());
    const plan = try family.tower.Scratch.init(c, shape.rows);
    const encoder = try download(a, graph, graph.tower.scratch.base() + plan.x, shape.rows * c.vision.hidden);
    const projector = try download(a, graph, projected.base(), shape.tokens * c.hidden);
    const encoder_ref = try files.read("vision.encoder");
    const projector_ref = try files.read("vision.projector");
    try files.write("vision.encoder", encoder_ref.shape[0..encoder_ref.rank], encoder);
    try files.write("vision.projector", projector_ref.shape[0..projector_ref.rank], projector);
    const encoder_stats = try files.compare("vision.encoder", encoder);
    const projector_stats = try files.compare("vision.projector", projector);

    const embeds = try a.alloc(f32, tokens.len * c.hidden);
    const logits = try a.alloc(f32, (continuation.len + 1) * c.vocab);
    const first_logit = prompt.len - 1;
    var first: usize = 0;
    while (first < tokens.len) {
        const count = @min(family.text.max_rows, tokens.len - first);
        const batch = tokens[first..][0..count];
        const spans = [_]family.text.Span{.{ .slot = 0, .first = 0, .rows = @intCast(count), .committed = @intCast(first) }};
        var images: [1]family.text.ImageRows = undefined;
        const begin = @max(first, image_first);
        const end = @min(first + count, image_first + shape.tokens);
        const image_rows: []const family.text.ImageRows = if (begin < end) blk: {
            images[0] = .{ .first = @intCast(begin - first), .rows = @intCast(end - begin), .features = projected.base() + (begin - image_first) * c.hidden * 2 };
            break :blk &images;
        } else &.{};
        try captureEmbeds(a, graph, batch, image_rows, embeds[first * c.hidden ..][0 .. count * c.hidden]);
        try graph.text.forward(batch, &spans, image_rows);
        try graph.s.synchronize();
        const keep = @max(first, first_logit);
        if (keep < first + count) {
            const got = try download(a, graph, graph.text.logits + (keep - first) * c.vocab * 2, (first + count - keep) * c.vocab);
            @memcpy(logits[(keep - first_logit) * c.vocab ..][0..got.len], got);
        }
        first += count;
    }
    const embeds_ref = try files.read("decoder.inputs_embeds");
    const logits_ref = try files.read("logits");
    try files.write("decoder.inputs_embeds", embeds_ref.shape[0..embeds_ref.rank], embeds);
    try files.write("logits", logits_ref.shape[0..logits_ref.rank], logits);
    const embeds_stats = try files.compare("decoder.inputs_embeds", embeds);
    const logits_stats = try files.compare("logits", logits);
    const want_logits = (try files.floats("logits")).data;
    const top = try a.alloc(struct { position: usize, reference: usize, native: usize }, continuation.len + 1);
    var top_differences: usize = 0;
    for (top, 0..) |*row, i| {
        row.* = .{ .position = first_logit + i, .reference = argmax(want_logits[i * c.vocab ..][0..c.vocab]), .native = argmax(logits[i * c.vocab ..][0..c.vocab]) };
        top_differences += @intFromBool(row.reference != row.native);
    }
    std.debug.print("logits: top token differences {d}/{d}\n", .{ top_differences, top.len });
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .schema = "tensorfold.lighton_hip.capture.v1",
        .qualification = "native HIP diagnostic capture; numerical differences reported without acceptance thresholds",
        .checkpoint = try fixture.field(files.manifest, "checkpoint"),
        .request = request,
        .reference_directory = args[2],
        .model_directory = args[1],
        .device = device_index,
        .architecture = hip.kernels.arch,
        .grid = .{ patches.gh, patches.gw },
        .stages = .{ .vision_encoder = encoder_stats, .vision_projector = projector_stats, .decoder_inputs_embeds = embeds_stats, .logits = logits_stats },
        .logit_top = top,
    }, .{ .whitespace = .indent_2 });
    try files.output.writeFile(init.io, .{ .sub_path = "capture.json", .data = receipt });
    return 0;
}

fn ids(a: Allocator, value: std.json.Value, vocab: u32) ![]u32 {
    const values = try fixture.list(value);
    const out = try a.alloc(u32, values.len);
    for (out, values) |*dst, v| {
        const id = try fixture.integer(v);
        if (id >= vocab) return error.TextToken;
        dst.* = @intCast(id);
    }
    return out;
}

fn imageRange(prompt: []const u32, token: u32, count: usize) !usize {
    const first = std.mem.indexOfScalar(u32, prompt, token) orelse return error.ImageRows;
    if (count == 0 or count > prompt.len - first) return error.ImageRows;
    for (prompt[first..][0..count]) |id| if (id != token) return error.ImageRows;
    if (std.mem.indexOfScalar(u32, prompt[first + count ..], token) != null) return error.ImageRows;
    return first;
}

fn download(a: Allocator, graph: anytype, address: u64, count: usize) ![]f32 {
    try graph.s.synchronize();
    const words = try a.alloc(u16, count);
    try hip.runtime.check(graph.r.api.hipMemcpyDtoH(@ptrCast(words.ptr), @ptrFromInt(address), count * 2));
    const out = try a.alloc(f32, count);
    for (out, words) |*dst, word| dst.* = @bitCast(@as(u32, word) << 16);
    return out;
}

fn captureEmbeds(a: Allocator, graph: anytype, tokens: []const u32, images: []const family.text.ImageRows, out: []f32) !void {
    try graph.s.synchronize();
    var token_buffer = try hip.DeviceBuffer.fromHost(graph.r, std.mem.sliceAsBytes(tokens));
    defer token_buffer.free();
    var output = try hip.DeviceBuffer.alloc(graph.r, out.len * 2);
    defer output.free();
    defer graph.s.synchronize() catch @panic("HIP embedding capture did not drain");
    try graph.text.shared.tf_embed_dense(@ptrFromInt(graph.text.embedding), @backingInt(hip.ops.Kind.f32), @ptrFromInt(token_buffer.base()), @intCast(tokens.len), @intCast(graph.c.hidden), @ptrFromInt(output.base()), @backingInt(hip.ops.Kind.bf16), graph.s.handle);
    for (images) |image| try output.copyFromAsync(@as(usize, image.first) * graph.c.hidden * 2, image.features, @as(usize, image.rows) * graph.c.hidden * 2, graph.s);
    @memcpy(out, try download(a, graph, output.base(), out.len));
}

fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return best;
}

test {
    _ = fixture;
}
