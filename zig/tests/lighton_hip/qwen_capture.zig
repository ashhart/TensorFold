//! Native Qwen HIP receipt capture: the production loader, tower, image rows and prompt formulas.
const std = @import("std");
const hip = @import("hip");
const family = @import("qwen3_5");
const shared = @import("qwen_hip_vision");
const lanes = @import("lanes");
const fixture = @import("fixture.zig");
const qf = @import("qwen_fixture.zig");
const Allocator = std.mem.Allocator;
const Engine = family.engine.Engine;
const Request = @typeInfo(@FieldType(family.hip_lanes.Lane, "vision")).optional.child;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 4 or args.len > 5) {
        std.debug.print("usage: tf-qwen-hip-capture MODEL REFERENCE_DIR OUTPUT_DIR [DEVICE]\n", .{});
        return 2;
    }
    var files = try fixture.Files.init(a, init.io, args[2], args[3]);
    defer files.output.close(init.io);
    try qf.verify(files, args[1]);
    const request = try fixture.field(files.manifest, "request");
    if ((try fixture.list(try fixture.field(request, "images_sha256"))).len != 1) return error.OneImageRequired;
    var cfg = try family.Config.read(a, init.io, args[1]);
    defer cfg.deinit();
    const c = cfg.spec;
    const prompt = try qf.ids(a, try fixture.field(request, "prompt_ids"), c.vocab);
    const continuation = try qf.ids(a, try fixture.field(request, "continuation_ids"), c.vocab);
    if (prompt.len == 0) return error.EmptyPrompt;
    const tokens = try std.mem.concat(a, u32, &.{ prompt, continuation });
    const teacher_ids = try qf.integers(a, files, "teacher.input_ids");
    if (teacher_ids.len != tokens.len) return error.BadFixture;
    for (teacher_ids, tokens) |want, got| if (want != got) return error.BadFixture;
    const pixels = try files.floats("processor.pixel_values");
    const grid = try qf.integers(a, files, "processor.image_grid_thw");
    if (grid.len != 3 or grid[0] != 1 or grid[1] <= 0 or grid[2] <= 0 or grid[1] > 4096 or grid[2] > 4096) return error.BadFixture;
    const gh: u32 = @intCast(grid[1]);
    const gw: u32 = @intCast(grid[2]);
    const patches = try shared.rowsOf(&.{.{ .h = gh, .w = gw }}, pixels.data.len);
    const image_tokens = patches / 4;
    if (pixels.array.rank != 2 or pixels.array.shape[0] != patches or pixels.array.shape[1] != shared.patch_width) return error.BadFixture;
    const device = if (args.len == 5) try std.fmt.parseInt(c_int, args[4], 10) else 0;
    if (device < 0) return error.BadDevice;
    var preflight: family.engine.Preflight = .{};
    const e = try Engine.open(init.gpa, init.io, args[1], .{
        .capacity = tokens.len + 17,
        .pool_pages = family.pages.pagesFor(tokens.len + 17) + family.pages.pagesFor(1),
        .batch_rows = 1,
        .streams = 1,
        .prompt_rows = @min(tokens.len, family.prefix.chunk),
        .device = device,
        .policy = .{ .slots = 0, .graphs = .off },
        .graphs = false,
        .preflight = &preflight,
        .vision = .{ .image_tokens = @intCast(image_tokens), .max_images = 1 },
    });
    defer e.deinit();
    defer e.drain();
    const vision = e.vision orelse return error.VisionNotLoaded;
    const image_first = try qf.imageRange(prompt, vision.c.image_token, image_tokens);
    const images = [_]lanes.stream.Image{.{ .at = @intCast(image_first), .gh = gh, .gw = gw, .tokens = @intCast(image_tokens), .pixels = pixels.data }};
    var media = try Request.init(vision, &images, tokens, e.o.capacity);
    defer media.deinit(init.gpa);
    defer e.drain();
    const sc = try shared.Scratch.init(vision.c, patches);
    const encoder = try download(a, e, vision.tower.scratch + sc.x, .bf16, patches * vision.c.hidden);
    const projector = try download(a, e, media.features.base(), .bf16, image_tokens * c.hidden);
    const encoder_stats = try record(files, "vision.encoder", encoder);
    const projector_stats = try record(files, "vision.projector", projector);
    const positions = try a.alloc(i64, tokens.len * 3);
    for (0..tokens.len) |row| {
        const p = media.layout.position(@intCast(row));
        for (0..3) |axis| positions[axis * tokens.len + row] = p[axis];
    }
    const want_positions = try qf.integers(a, files, "decoder.position_ids");
    if (positions.len != want_positions.len) return error.BadCaptureShape;
    var position_differences: usize = 0;
    for (positions, want_positions) |got, want| position_differences += @intFromBool(got != want);
    try qf.writePositions(files, positions, tokens.len);

    var caches = try e.newCaches(e.o.capacity);
    defer caches.deinit(init.gpa);
    defer e.drain();
    e.bridge.model.frame = media.frame();
    defer e.bridge.model.frame = null;
    const embeds = try a.alloc(f32, tokens.len * c.hidden);
    const logits = try a.alloc(f32, (continuation.len + 1) * c.vocab);
    const first_logit = prompt.len - 1;
    var first: usize = 0;
    while (first < tokens.len) {
        const count = @min(e.o.prompt_rows, tokens.len - first);
        const batch = tokens[first..][0..count];
        e.prompts.reset();
        const o = e.ops(&e.prompts);
        @memcpy(e.ids.slice(u32)[0..count], batch);
        try e.ids_dev.uploadAsync(0, e.ids.view(0, 4 * count), e.stream);
        try captureEmbeds(a, e, first, count, embeds[first * c.hidden ..][0 .. count * c.hidden]);
        e.prompts.reset();
        const hidden = try family.forward.span(o, e.model(), &caches, e.ids_dev.base(), count, first, null);
        const mark = e.prompts.mark();
        const begin = @max(first, first_logit);
        if (begin < first + count) for (begin..first + count) |row| {
            e.prompts.release(mark);
            const last = family.forward.at(hidden, (row - first) * c.hidden);
            const y = try o.project(last, e.model().head, 1, false);
            const got = try download(a, e, y.ptr, y.kind, c.vocab);
            @memcpy(logits[(row - first_logit) * c.vocab ..][0..c.vocab], got);
        };
        try e.stream.synchronize();
        first += count;
    }
    const embeds_stats = try record(files, "decoder.inputs_embeds", embeds);
    const logits_stats = try record(files, "logits", logits);
    const want_logits = (try files.floats("logits")).data;
    const top = try a.alloc(struct { position: usize, reference: usize, native: usize }, continuation.len + 1);
    var top_differences: usize = 0;
    for (top, 0..) |*row, i| {
        row.* = .{ .position = first_logit + i, .reference = argmax(want_logits[i * c.vocab ..][0..c.vocab]), .native = argmax(logits[i * c.vocab ..][0..c.vocab]) };
        top_differences += @intFromBool(row.reference != row.native);
    }
    std.debug.print("positions: {d}/{d} differences; logits: {d}/{d} top token differences\n", .{ position_differences, positions.len, top_differences, top.len });
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .schema = "tensorfold.lighton_qwen_hip.capture.v1",
        .qualification = "native HIP diagnostic capture; no acceptance threshold applied",
        .arithmetic = "production BF16 activations, losslessly widened FP32 checkpoint weights; full prompt and teacher forcing",
        .checkpoint = try fixture.field(files.manifest, "checkpoint"),
        .request = request,
        .reference_directory = args[2],
        .model_directory = args[1],
        .device = device,
        .architecture = hip.kernels.arch,
        .grid = .{ gh, gw },
        .position_differences = position_differences,
        .stages = .{ .vision_encoder = encoder_stats, .vision_projector = projector_stats, .decoder_inputs_embeds = embeds_stats, .logits = logits_stats },
        .logit_top = top,
    }, .{ .whitespace = .indent_2 });
    try files.output.writeFile(init.io, .{ .sub_path = "capture.json", .data = receipt });
    return 0;
}

fn captureEmbeds(a: Allocator, e: *Engine, first: usize, rows: usize, out: []f32) !void {
    const m = e.model();
    const o = e.ops(&e.prompts);
    const x = try family.forward.take(o, m.act, rows * m.spec.hidden);
    try o.embedRows(m.embed, e.ids_dev.base(), rows, x);
    for (m.frame.?.runs) |run| if (run.clip(first, rows, m.spec.hidden)) |clip| {
        try o.copyCols(.{ .ptr = clip.features, .kind = .bf16 }, m.spec.hidden, 0, x.ptr + clip.first * m.spec.hidden * 2, clip.rows, m.spec.hidden);
    };
    @memcpy(out, try download(a, e, x.ptr, x.kind, out.len));
}

fn download(a: Allocator, e: *Engine, address: u64, kind: hip.ops.Kind, count: usize) ![]f32 {
    try e.stream.synchronize();
    const out = try a.alloc(f32, count);
    switch (kind) {
        .f32 => try hip.runtime.check(e.driver.api.hipMemcpyDtoH(@ptrCast(out.ptr), @ptrFromInt(address), count * 4)),
        .bf16 => {
            const words = try a.alloc(u16, count);
            defer a.free(words);
            try hip.runtime.check(e.driver.api.hipMemcpyDtoH(@ptrCast(words.ptr), @ptrFromInt(address), count * 2));
            for (out, words) |*dst, word| dst.* = @bitCast(@as(u32, word) << 16);
        },
        else => return error.UnsupportedCaptureKind,
    }
    return out;
}

fn record(files: fixture.Files, name: []const u8, got: []const f32) !fixture.Stats {
    const ref = try files.read(name);
    try files.write(name, ref.shape[0..ref.rank], got);
    return files.compare(name, got);
}

fn argmax(row: []const f32) usize {
    var best: usize = 0;
    for (row, 0..) |value, i| if (value > row[best]) {
        best = i;
    };
    return best;
}

test {
    _ = qf;
}
