//! Host refusals at LFM3-VL's image boundary: the processor file, image budget, image size, pixels and vision tensors.
const std = @import("std");
const image = @import("image.zig");
const layout = @import("vision_layout.zig");
const picture = @import("../../core/picture/picture.zig");

const processor_json =
    \\{
    \\  "image_processor": {
    \\    "do_convert_rgb": true,
    \\    "do_normalize": true,
    \\    "do_rescale": true,
    \\    "do_resize": true,
    \\    "image_mean": [0.5, 0.5, 0.5],
    \\    "image_processor_type": "Lfm3VlImageProcessor",
    \\    "image_std": [0.5, 0.5, 0.5],
    \\    "merge_size": 2,
    \\    "patch_size": 16,
    \\    "resample": 3,
    \\    "rescale_factor": 0.00392156862745098,
    \\    "size": {"shortest_edge": 65536, "longest_edge": 16777216},
    \\    "temporal_patch_size": 2
    \\  },
    \\  "processor_class": "Lfm3VlProcessor"
    \\}
;

/// A checkpoint folder holding `processor_config.json` with `text`, or none, and what the engine's open says of it.
fn openProcessor(text: ?[]const u8) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    if (text) |bytes| try tmp.dir.writeFile(io, .{ .sub_path = "processor_config.json", .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    return image.loadProcessor(io, a, dir);
}

/// `processor_json` with its first `from` replaced by `to`.
fn edited(a: std.mem.Allocator, from: []const u8, to: []const u8) ![]u8 {
    try std.testing.expect(std.mem.indexOf(u8, processor_json, from) != null);
    return std.mem.replaceOwned(u8, a, processor_json, from, to);
}

test "LFM3 processor file: the official settings open, a missing or unreadable file stops the open" {
    try openProcessor(processor_json);
    try std.testing.expectError(error.FileNotFound, openProcessor(null));
    try std.testing.expect(std.meta.isError(openProcessor("{")));
    try std.testing.expectError(error.BadLfm3Processor, openProcessor("[]"));
    try std.testing.expectError(error.BadLfm3Processor, openProcessor("{}"));
}

test "LFM3 processor file: any setting that is not the tower's own is refused" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "\"image_processor\"", "\"video_processor\"" },
        .{ "\"image_processor\": {", "\"image_processor\": 3, \"other\": {" },
        .{ "\"image_processor_type\": \"Lfm3VlImageProcessor\"", "\"image_processor_type\": \"Qwen2VLImageProcessor\"" },
        .{ "\"do_convert_rgb\": true", "\"do_convert_rgb\": false" },
        .{ "\"do_normalize\": true", "\"do_normalize\": false" },
        .{ "\"do_rescale\": true", "\"do_rescale\": false" },
        .{ "\"do_resize\": true", "\"do_resize\": false" },
        .{ "    \"do_resize\": true,\n", "" },
        .{ "\"patch_size\": 16", "\"patch_size\": 14" },
        .{ "\"merge_size\": 2", "\"merge_size\": 4" },
        .{ "\"temporal_patch_size\": 2", "\"temporal_patch_size\": 1" },
        .{ "\"resample\": 3", "\"resample\": 2" },
        .{ "\"rescale_factor\": 0.00392156862745098", "\"rescale_factor\": 0.5" },
        .{ "\"image_mean\": [0.5, 0.5, 0.5]", "\"image_mean\": [0.5, 0.5]" },
        .{ "\"image_mean\": [0.5, 0.5, 0.5]", "\"image_mean\": [0.48, 0.45, 0.4]" },
        .{ "\"image_std\": [0.5, 0.5, 0.5]", "\"image_std\": [1.0, 1.0, 1.0]" },
        .{ "\"shortest_edge\": 65536", "\"shortest_edge\": 4096" },
        .{ "\"longest_edge\": 16777216", "\"longest_edge\": 25165824" },
        .{ "\"size\": {\"shortest_edge\": 65536, \"longest_edge\": 16777216}", "\"size\": 65536" },
    };
    for (cases) |case| {
        const text = try edited(a, case[0], case[1]);
        defer a.free(text);
        openProcessor(text) catch |err| {
            try std.testing.expect(err == error.BadLfm3Processor);
            continue;
        };
        std.debug.print("processor setting was accepted: {s} -> {s}\n", .{ case[0], case[1] });
        return error.ProcessorAccepted;
    }
}

test "LFM3 image budget: 64 to 4096 tokens and at least one image, 256 when unset" {
    try std.testing.expectEqual(@as(u32, 256), try image.admitBudget(false, 4096, 4));
    try std.testing.expectEqual(@as(u32, 256), try image.admitBudget(false, 0, 1));
    try std.testing.expectEqual(@as(u32, 64), try image.admitBudget(true, 64, 4));
    try std.testing.expectEqual(@as(u32, 4096), try image.admitBudget(true, 4096, 4));
    try std.testing.expectEqual(@as(u32, 1000), try image.admitBudget(true, 1000, 1));
    for ([_]u32{ 0, 1, 63, 4097, 16384, std.math.maxInt(u32) }) |tokens|
        try std.testing.expectError(error.BadLfm3ImageBudget, image.admitBudget(true, tokens, 4));
    try std.testing.expectError(error.BadLfm3ImageBudget, image.admitBudget(true, 256, 0));
    try std.testing.expectError(error.BadLfm3ImageBudget, image.admitBudget(false, 0, 0));
}

test "LFM3 image size: a budget under 64 tokens, an empty image and an extreme shape are refused, a small one is enlarged" {
    const a = std.testing.allocator;
    var small: [8 * 8 * 3]u8 = @splat(100);
    const tiny: picture.Rgb = .{ .width = 8, .height = 8, .pixels = &small };
    for ([_]u32{ 0, 1, 63 }) |tokens|
        try std.testing.expectError(error.ImageTooSmall, image.prepareRgb(a, tiny, 0, tokens));
    var none: [0]u8 = .{};
    try std.testing.expectError(error.ImageTooSmall, image.prepareRgb(a, .{ .width = 0, .height = 8, .pixels = &none }, 0, 256));
    try std.testing.expectError(error.ImageTooSmall, image.prepareRgb(a, .{ .width = 8, .height = 0, .pixels = &none }, 0, 256));
    // the 65,536-pixel floor enlarges an 8 x 8 image to 256 x 256: 16 x 16 patches, 64 tokens, within a 64-token budget
    for ([_]u32{ 64, 256 }) |tokens| {
        const grown = try image.prepareRgb(a, tiny, 0, tokens);
        defer a.free(grown.pixels);
        try std.testing.expectEqual([3]u32{ 16, 16, 64 }, [3]u32{ grown.gh, grown.gw, grown.tokens });
        try std.testing.expectEqual(@as(usize, 16 * 16 * image.patch_width), grown.pixels.len);
    }
    var thin: [201 * 3]u8 = @splat(0);
    try std.testing.expectError(error.AspectRatio, image.prepareRgb(a, .{ .width = 201, .height = 1, .pixels = &thin }, 0, 256));
}

test "LFM3 prepared images below the processor minimum are refused, the minimum is admitted" {
    const Input = struct { pixels: []const f32, gh: u32, gw: u32, tokens: u32, at: u32 = 0 };
    const pixels = try std.testing.allocator.alloc(f32, 256 * image.patch_width);
    defer std.testing.allocator.free(pixels);
    var inputs = [_]Input{.{ .pixels = pixels[0 .. 252 * image.patch_width], .gh = 14, .gw = 18, .tokens = 63 }};
    try std.testing.expectError(error.ImageTooSmall, image.validate(&inputs, 64, 64, 1));
    inputs[0] = .{ .pixels = pixels, .gh = 16, .gw = 16, .tokens = 64 };
    try image.validate(&inputs, 64, 64, 1);
}

test "LFM3 pixels: NaN and infinity refuse the image wherever they sit, finite values round to BF16" {
    var words: [6]u16 = undefined;
    const tie_even: f32 = @bitCast(@as(u32, 0x3f808000)); // halfway between 0x3f80 and 0x3f81: the even one
    const tie_odd: f32 = @bitCast(@as(u32, 0x3f818000)); // halfway between 0x3f81 and 0x3f82: the even one
    try image.packPixels(&.{ 1, -1, 0, 0.5, tie_even, tie_odd }, &words);
    try std.testing.expectEqualSlices(u16, &.{ 0x3f80, 0xbf80, 0x0000, 0x3f00, 0x3f80, 0x3f82 }, &words);
    const bad = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (bad) |value| {
        for (0..words.len) |at| {
            var pixels: [6]f32 = @splat(0.25);
            pixels[at] = value;
            try std.testing.expectError(error.NonfiniteLfm3Image, image.packPixels(&pixels, &words));
        }
    }
    try std.testing.expectError(error.BadLfm3ImageShape, image.packPixels(&.{ 1, 2, 3 }, &words));
}

const Ref = u32;
const Found = layout.Found(Ref);
const Table = std.StringHashMapUnmanaged(Found);

/// The official exports' 138 vision tensors from the committed header catalogue; each ref is its catalogue line.
fn official(a: std.mem.Allocator) !Table {
    var table: Table = .empty;
    errdefer table.deinit(a);
    var lines = std.mem.splitScalar(u8, @embedFile("fixtures/vision_tensors.tsv"), '\n');
    var offset: usize = 0;
    var line_no: Ref = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        var fields = std.mem.splitScalar(u8, line, '\t');
        const name = fields.next().?;
        try std.testing.expectEqualStrings("BF16", fields.next().?);
        var found: Found = .{ .ref = line_no, .bf16 = true, .rank = 0, .shape = @splat(1), .offset = offset, .bytes = 2, .storage = 0 };
        var dims = std.mem.splitScalar(u8, fields.next().?, ',');
        while (dims.next()) |dim| {
            found.shape[found.rank] = try std.fmt.parseInt(usize, dim, 10);
            found.bytes *= found.shape[found.rank];
            found.rank += 1;
        }
        offset += found.bytes;
        try table.put(a, name, found);
        line_no += 1;
    }
    try std.testing.expectEqual(@as(usize, layout.tensor_count), table.count());
    var it = table.valueIterator();
    while (it.next()) |found| found.storage = offset;
    return table;
}

/// A checkpoint index over a table, as the engine's offers its tensors to the layout.
const Index = struct {
    table: *const Table,

    const Names = struct {
        inner: Table.KeyIterator,

        pub fn next(self: *Names) ?[]const u8 {
            const name = self.inner.next() orelse return null;
            return name.*;
        }
    };

    pub fn find(self: Index, name: []const u8) !?Found {
        return self.table.get(name);
    }

    pub fn keys(self: Index) Names {
        return .{ .inner = self.table.keyIterator() };
    }
};

fn loadTable(table: *const Table) !layout.Layout(Ref).Weights {
    return layout.load(Ref, std.testing.allocator, Index{ .table = table });
}

fn refOf(table: *const Table, name: []const u8) Ref {
    return table.get(name).?.ref;
}

test "LFM3 vision tensors: the official 138 load into the layout the tower kernels read" {
    var table = try official(std.testing.allocator);
    defer table.deinit(std.testing.allocator);
    const w = try loadTable(&table);
    try std.testing.expectEqual(refOf(&table, "vision_tower.patchifier.weight"), w.patch);
    try std.testing.expectEqual(refOf(&table, "vision_tower.embedding_norm.weight"), w.norm);
    try std.testing.expectEqual(refOf(&table, "multi_modal_projector.linear_2.bias"), w.bias2);
    try std.testing.expectEqual(refOf(&table, "multi_modal_projector.layer_norm.bias"), w.projector_norm_bias);
    try std.testing.expectEqual(refOf(&table, "vision_tower.layers.13.feed_forward.w2.weight"), w.blocks[13].down);
    try std.testing.expectEqual(refOf(&table, "vision_tower.layers.1.conv.spatial_conv.weight"), w.blocks[1].mixer.conv.spatial);
    var attention: usize = 0;
    for (w.blocks, 0..) |block, i| switch (block.mixer) {
        .conv => try std.testing.expect(i < 2 or i % 2 == 1),
        .attention => |m| {
            try std.testing.expect(i >= 2 and i % 2 == 0);
            attention += 1;
            var buf: [96]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "vision_tower.layers.{d}.self_attn.k_layernorm.weight", .{i});
            try std.testing.expectEqual(refOf(&table, name), m.k_norm);
        },
    };
    try std.testing.expectEqual(@as(usize, 6), attention);
    // text tensors beside the tower are not its to refuse
    try table.put(std.testing.allocator, "language_model.model.layers.0.operator_norm.weight", table.get("vision_tower.layers.0.ffn_norm.weight").?);
    _ = try loadTable(&table);
}

test "LFM3 vision tensors: every missing tensor stops the load by name" {
    var table = try official(std.testing.allocator);
    defer table.deinit(std.testing.allocator);
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(std.testing.allocator);
    var it = table.keyIterator();
    while (it.next()) |name| try names.append(std.testing.allocator, name.*);
    for (names.items) |name| {
        const kept = table.get(name).?;
        _ = table.remove(name);
        std.testing.expectError(error.MissingTensor, loadTable(&table)) catch |err| {
            std.debug.print("removing {s} gave a different outcome\n", .{name});
            return err;
        };
        try table.put(std.testing.allocator, name, kept);
    }
    _ = try loadTable(&table);
}

test "LFM3 vision tensors: an unexpected vision or projector tensor stops the load, one outside them does not" {
    var table = try official(std.testing.allocator);
    defer table.deinit(std.testing.allocator);
    const template = table.get("vision_tower.layers.0.ffn_norm.weight").?;
    for ([_][]const u8{
        "vision_tower.layers.14.operator_norm.weight",
        "vision_tower.layers.0.self_attn.q_proj.weight", // an attention tensor in a convolution layer
        "vision_tower.layers.2.conv.in_proj.weight", // a convolution tensor in an attention layer
        "vision_tower.patchifier.bias",
        "vision_tower.post_layernorm.weight",
        "multi_modal_projector.linear_3.weight",
        "vision_tower_extra",
    }) |name| {
        try table.put(std.testing.allocator, name, template);
        const outcome = loadTable(&table);
        _ = table.remove(name);
        if (std.mem.startsWith(u8, name, "vision_tower_extra")) {
            _ = try outcome; // a different namespace: not the tower's
        } else try std.testing.expectError(error.UnexpectedLfm3VisionTensor, outcome);
    }
    _ = try loadTable(&table);
}

test "LFM3 vision tensors: a wrong type, shape, size or place stops the load" {
    var table = try official(std.testing.allocator);
    defer table.deinit(std.testing.allocator);
    const probes = [_][]const u8{
        "vision_tower.patchifier.weight",
        "vision_tower.layers.0.conv.spatial_conv.weight",
        "vision_tower.layers.2.self_attn.k_proj.weight",
        "vision_tower.layers.13.feed_forward.w3.weight",
        "multi_modal_projector.linear_1.weight",
        "multi_modal_projector.linear_2.bias",
    };
    for (probes) |name| {
        const good = table.get(name).?;
        var bad = good;
        bad.bf16 = false; // F16 or a quantized word, not BF16
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.UnexpectedLfm3VisionTensor, loadTable(&table));
        bad = good;
        bad.shape[0] += 1;
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.UnexpectedLfm3VisionTensor, loadTable(&table));
        bad = good;
        bad.shape[bad.rank] = 1; // a trailing dimension of one is another rank
        bad.rank += 1;
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.UnexpectedLfm3VisionTensor, loadTable(&table));
        bad = good;
        bad.rank -= 1;
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.UnexpectedLfm3VisionTensor, loadTable(&table));
        bad = good;
        bad.bytes -= 2; // a header whose byte range is short of its shape
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.TensorBytes, loadTable(&table));
        bad = good;
        bad.offset += 2; // not word aligned
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.TensorBytes, loadTable(&table));
        bad = good;
        bad.storage = good.offset + good.bytes - 2; // its buffer ends inside it
        try table.put(std.testing.allocator, name, bad);
        try std.testing.expectError(error.TensorBytes, loadTable(&table));
        try table.put(std.testing.allocator, name, good);
    }
    _ = try loadTable(&table);
}
