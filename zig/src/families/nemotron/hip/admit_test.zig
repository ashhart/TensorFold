//! Admission tests for Nemotron-H on HIP: widths, BF16 embeddings, refusals, and MTP head metadata.
const std = @import("std");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");
const plan = @import("plan.zig");
const admit_mod = @import("admit.zig");

const Facts = admit_mod.Facts;
const Ask = admit_mod.Ask;
const admit = admit_mod.admit;
const tooBig = admit_mod.tooBig;

const testing = std.testing;
const stock_config = @embedFile("../fixtures/config.json");
const stock_index = @embedFile("../fixtures/index.json");
const gib: u64 = 1 << 30;

/// The stock MTP file's names, from the fixture's lines.
fn stockMtp(a: std.mem.Allocator) ![]const []const u8 {
    var list: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, @embedFile("../fixtures/mtp_names.txt"), "\n\r ");
    while (it.next()) |n| try list.append(a, n);
    return list.items;
}

/// `text` with every `from` replaced by `to`.
fn edit(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]const u8 {
    try testing.expect(std.mem.indexOf(u8, text, from) != null);
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// The stock config with module `name` at `bits` in groups of `group`.
fn module(a: std.mem.Allocator, name: []const u8, bits: u8, group: u16) ![]const u8 {
    const entry = try std.fmt.allocPrint(a, "\"group_size\": 64, \"{s}\": {{\"bits\": {d}, \"group_size\": {d}}},", .{
        name,
        bits,
        group,
    });
    return edit(a, stock_config, "\"group_size\": 64,", entry);
}

/// admit() must refuse with a reason naming each of `words`.
fn expectRefused(a: std.mem.Allocator, f: Facts, ask: Ask, words: []const []const u8) !void {
    var problem: []const u8 = "";
    try testing.expectError(error.Refused, admit(a, f, ask, &problem));
    for (words) |w| if (std.mem.indexOf(u8, problem, w) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ problem, w });
        return error.TestUnexpectedResult;
    };
}

test "the stock checkpoint and its MTP head are admitted; on one V620 a stream takes the model's window" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const f: Facts = .{ .config = stock_config, .index = stock_index, .mtp = try stockMtp(a) };
    const one = try admit(a, f, .{ .window = 262144, .room = 30 * gib }, &problem);
    try testing.expect(one.fits and one.mtp);
    try testing.expectEqual(@as(usize, 52), one.config.layers);
    const two = try admit(a, f, .{ .world = 2, .rank = 1, .window = 262144, .room = 30 * gib }, &problem);
    try testing.expect(two.fits and two.weights < one.weights / 2);
    const lone = try admit(
        a,
        .{ .config = stock_config, .index = stock_index },
        .{ .window = 4096, .room = 30 * gib },
        &problem,
    );
    try testing.expect(!lone.mtp and lone.weights < one.weights);
    const small = try admit(a, f, .{ .window = 4096, .room = 16 * gib }, &problem);
    try testing.expect(!small.fits);
    try testing.expectEqual(error.Refused, tooBig(a, small, .{ .window = 4096, .room = 16 * gib }, &problem));
    try testing.expect(std.mem.indexOf(u8, problem, "16.00 GiB this card grants") != null);
}

test "uniform 8-bit and mixed 4/5/6/8-bit checkpoints in groups of 64 are admitted, their bytes at their widths" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ask: Ask = .{ .window = 4096, .room = 60 * gib };
    const four = try admit(a, .{ .config = stock_config, .index = stock_index }, ask, &problem);
    const eight_text = try edit(a, stock_config, "\"bits\": 4", "\"bits\": 8");
    const eight = try admit(a, .{ .config = eight_text, .index = stock_index }, ask, &problem);
    try testing.expect(eight.fits and eight.weights > four.weights + four.weights / 2);
    var text: []const u8 = stock_config;
    for ([_][]const u8{ "backbone.embeddings", "lm_head", "backbone.layers.1.mixer.shared_experts.up_proj" }) |n| {
        const entry = try std.fmt.allocPrint(a, "\"group_size\": 64, \"{s}\": {{\"bits\": 8}},", .{n});
        text = try edit(a, text, "\"group_size\": 64,", entry);
    }
    text = try edit(a, text, "\"group_size\": 64,", "\"group_size\": 64, " ++
        "\"backbone.layers.0.mixer.in_proj\": {\"bits\": 5}, " ++
        "\"backbone.layers.0.mixer.out_proj\": {\"bits\": 6},");
    const mixed = try admit(a, .{ .config = text, .index = stock_index }, ask, &problem);
    try testing.expect(mixed.fits and mixed.weights > four.weights and mixed.weights < eight.weights);
    try testing.expectEqual(@as(u8, 5), mixed.config.widths[0][0].bits);
}

test "the MTP head file is admitted beside a model not stored 4-bit in groups of 64" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ask: Ask = .{ .window = 4096, .room = 60 * gib };
    const mtp = try stockMtp(a);
    for ([_][]const u8{ "\"bits\": 6", "\"bits\": 8" }) |bits| {
        const text = try edit(a, stock_config, "\"bits\": 4", bits);
        const v = try admit(a, .{ .config = text, .index = stock_index, .mtp = mtp }, ask, &problem);
        try testing.expect(v.mtp and v.fits);
        try testing.expectEqual(config.Width{ .bits = 4, .group = 64 }, v.head);
    }
    const mixed = try module(a, "lm_head", 8, 64);
    const v = try admit(a, .{ .config = mixed, .index = stock_index, .mtp = mtp }, ask, &problem);
    try testing.expect(v.mtp and v.fits);
}

test "altered checkpoints are refused with a reason" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ask: Ask = .{ .window = 4096, .room = 30 * gib };
    const mtp = try stockMtp(a);
    const gone = "backbone.layers.7.mixer.A_log";
    try expectRefused(
        a,
        .{
            .config = stock_config,
            .index = try edit(a, stock_index, "\"" ++ gone ++ "\"", "\"backbone.layers.7.mixer.B_log\""),
        },
        ask,
        &.{gone},
    );
    const scales = "backbone.layers.12.mixer.o_proj.scales";
    try expectRefused(
        a,
        .{ .config = stock_config, .index = try edit(a, stock_index, "\"" ++ scales ++ "\"", "\"x\"") },
        ask,
        &.{ "backbone.layers.12.mixer.o_proj", "Living Weights" },
    );
    try expectRefused(
        a,
        .{
            .config = stock_config,
            .index = try edit(a, stock_index, "\"lm_head.weight\"", "\"lm_head.weight\": \"a\", \"vision.w\""),
        },
        ask,
        &.{"vision.w"},
    );
    try expectRefused(
        a,
        .{ .config = try edit(a, stock_config, "\"group_size\": 64", "\"group_size\": 32"), .index = stock_index },
        ask,
        &.{ "backbone.embeddings is 4-bit in groups of 32", "groups of 64" },
    );
    const three = try module(a, "backbone.layers.4.mixer.in_proj", 3, 64);
    try expectRefused(a, .{ .config = three, .index = stock_index }, ask, &.{
        "backbone.layers.4.mixer.in_proj is 3-bit in groups of 64",
    });
    try expectRefused(a, .{ .config = try module(a, "lm_head", 8, 128), .index = stock_index }, ask, &.{
        "lm_head is 8-bit in groups of 128",
    });
    try expectRefused(
        a,
        .{ .config = try module(a, "backbone.layers.1.mixer.switch_mlp.fc1", 2, 64), .index = stock_index },
        ask,
        &.{"backbone.layers.1.mixer.switch_mlp.fc1 is 2-bit"},
    );
    try expectRefused(
        a,
        .{
            .config = try edit(a, stock_config, "\"group_size\": 64,", "\"group_size\": 64, " ++
                "\"backbone.layers.5.mixer.q_proj\": false,"),
            .index = stock_index,
        },
        ask,
        &.{ "backbone.layers.5.mixer.q_proj", "unquantized" },
    );
    try expectRefused(
        a,
        .{
            .config = stock_config,
            .index = try edit(a, stock_index, "\"lm_head.weight\"", "\"lm_head.weight\": \"a\", " ++
                "\"mtp.layers.0.eh_proj.weight\""),
        },
        ask,
        &.{ "mtp.layers.0.eh_proj.weight", "inside the model's shards", tensors.mtp_file },
    );
    try expectRefused(
        a,
        .{
            .config = try edit(
                a,
                stock_config,
                "\"quantization\": {",
                "\"quantization\": {\"quant_method\": \"modelopt\",",
            ),
            .index = stock_index,
        },
        ask,
        &.{ "ModelOpt", "CUDA" },
    );
    try expectRefused(
        a,
        .{ .config = try edit(a, stock_config, "\"mode\": \"affine\"", "\"mode\": \"nvfp4\""), .index = stock_index },
        ask,
        &.{"nvfp4"},
    );
    try expectRefused(
        a,
        .{
            .config = try edit(
                a,
                stock_config,
                "\"model_type\": \"nemotron_h\"",
                "\"model_type\": \"nemotron_h\", \"tensorfold_learned_neurons\": 64",
            ),
            .index = stock_index,
        },
        ask,
        &.{"pocket"},
    );
    try expectRefused(a, .{ .config = stock_config, .index = stock_index, .living = true }, ask, &.{"Living Weights"});
    try expectRefused(a, .{ .config = stock_config, .index = stock_index, .mtp = mtp[1..] }, ask, &.{tensors.mtp_file});
    try expectRefused(
        a,
        .{ .config = stock_config, .index = stock_index },
        .{ .world = 3, .window = 4096, .room = 30 * gib },
        &.{"--tp 3"},
    );
}

/// The stock config with the raw quantization entries `raw` added to its block.
fn entries(a: std.mem.Allocator, raw: []const u8) ![]const u8 {
    return edit(a, stock_config, "\"group_size\": 64,", try std.fmt.allocPrint(a, "\"group_size\": 64, {s},", .{raw}));
}

/// The stock index without the embedding's scales and biases: a bf16 token table's tensors.
fn bf16Index(a: std.mem.Allocator) ![]const u8 {
    const biases = "\"backbone.embeddings.biases\": \"model-00001-of-00004.safetensors\",";
    const scales = "\"backbone.embeddings.scales\": \"model-00001-of-00004.safetensors\",";
    return edit(a, try edit(a, stock_index, biases, ""), scales, "");
}

test "a bf16 token table beside quantized projections is admitted, its bytes counted as stored" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ask: Ask = .{ .window = 4096, .room = 60 * gib };
    const text = try entries(a, "\"backbone.embeddings\": false");
    const mtp = try stockMtp(a);
    const stock = try admit(a, .{ .config = stock_config, .index = stock_index, .mtp = mtp }, ask, &problem);
    const v = try admit(a, .{ .config = text, .index = try bf16Index(a), .mtp = mtp }, ask, &problem);
    try testing.expect(v.fits and v.mtp and v.config.embed_bf16);
    const quantized = plan.affine(.{}, v.config.vocab, v.config.hidden);
    try testing.expectEqual(stock.weights - quantized + v.config.vocab * v.config.hidden * 2, v.weights);
    // as MLX loads it, no scales in the files is enough: config.json need not list the table
    const unlisted = try admit(a, .{ .config = stock_config, .index = try bf16Index(a), .mtp = mtp }, ask, &problem);
    try testing.expect(unlisted.config.embed_bf16);
    try testing.expectEqual(v.weights, unlisted.weights);
    // the same text still refuses where the token table is not the engine's to read bare
    var why: config.Why = .{};
    try testing.expectError(error.UnsupportedQuantization, config.parse(a, text, &why));
    try testing.expect(std.mem.indexOf(u8, why.text(), "backbone.embeddings") != null);
}

test "with a bf16 token table, every other unquantized or unread tensor is still refused by name" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ask: Ask = .{ .window = 4096, .room = 60 * gib };
    const bare = "\"backbone.embeddings\": false";
    const index = try bf16Index(a);
    // the table's own mismatches: scales left in the index, or none behind a quantized config
    try expectRefused(a, .{ .config = try entries(a, bare), .index = stock_index }, ask, &.{
        "backbone.embeddings.scales",
        "does not read",
    });
    try expectRefused(a, .{ .config = stock_config, .index = try edit(a, index, "\"backbone.embeddings.weight\"", "\"x\"") }, ask, &.{
        "backbone.embeddings.weight",
    });
    try expectRefused(a, .{ .config = try entries(a, "\"backbone.embeddings\": {}"), .index = index }, ask, &.{
        "backbone.embeddings",
        "unquantized",
    });
    // a bf16 head, projection or attention matrix is not the table
    for ([_][]const u8{ "lm_head", "backbone.layers.1.mixer.shared_experts.up_proj", "backbone.layers.5.mixer.q_proj" }) |n| {
        const text = try entries(a, try std.fmt.allocPrint(a, "{s}, \"{s}\": false", .{ bare, n }));
        try expectRefused(a, .{ .config = text, .index = index }, ask, &.{ n, "unquantized" });
    }
    const o_proj = "backbone.layers.12.mixer.o_proj.scales";
    try expectRefused(a, .{
        .config = try entries(a, bare),
        .index = try edit(a, index, "\"" ++ o_proj ++ "\"", "\"x\""),
    }, ask, &.{ "backbone.layers.12.mixer.o_proj", "Living Weights" });
    // other widths and group sizes
    const three = try entries(a, bare ++ ", \"backbone.layers.4.mixer.in_proj\": {\"bits\": 3}");
    try expectRefused(a, .{ .config = three, .index = index }, ask, &.{"backbone.layers.4.mixer.in_proj is 3-bit in groups of 64"});
    const wide = try entries(a, bare ++ ", \"lm_head\": {\"bits\": 8, \"group_size\": 128}");
    try expectRefused(a, .{ .config = wide, .index = index }, ask, &.{"lm_head is 8-bit in groups of 128"});
    const grouped = try edit(a, try entries(a, bare), "\"group_size\": 64,", "\"group_size\": 32,");
    try expectRefused(a, .{ .config = grouped, .index = index }, ask, &.{"groups of 32"});
    // the MTP head inside the shards stays refused; its file drafts beside 8-bit unless the head itself is unserved
    const inside = try edit(a, index, "\"lm_head.weight\"", "\"lm_head.weight\": \"a\", \"mtp.layers.0.eh_proj.weight\"");
    try expectRefused(a, .{ .config = try entries(a, bare), .index = inside }, ask, &.{
        "mtp.layers.0.eh_proj.weight",
        "inside the model's shards",
    });
    const eight = try edit(a, try entries(a, bare), "\"bits\": 4", "\"bits\": 8");
    const stock_head = try stockMtp(a);
    const v8 = try admit(a, .{ .config = eight, .index = index, .mtp = stock_head }, ask, &problem);
    try testing.expect(v8.fits and v8.mtp and v8.config.embed_bf16 and v8.head.bits == 4);
    try expectRefused(
        a,
        .{ .config = eight, .index = index, .mtp = stock_head, .mtp_quantization = "{\"group_size\": 32, \"bits\": 4}" },
        ask,
        &.{ tensors.mtp_file, "4-bit in groups of 32" },
    );
}

test "the MTP head is read at the width its own file states, beside a model of any width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const ask: Ask = .{ .window = 4096, .room = 60 * gib };
    const mtp = try stockMtp(a);
    const eight = try edit(a, stock_config, "\"bits\": 4", "\"bits\": 8");
    const stock_head = "{\"group_size\": 64, \"bits\": 4}";
    const f: Facts = .{ .config = eight, .index = stock_index, .mtp = mtp, .mtp_quantization = stock_head };
    const v = try admit(a, f, ask, &problem);
    try testing.expectEqual(config.Width{ .bits = 4, .group = 64 }, v.head);
    const lone = try admit(a, .{ .config = eight, .index = stock_index }, ask, &problem);
    // the head at 4/64 and its draft rows at the 8-bit vocabulary head's width
    const want = (try plan.weights(&v.config, 1, .device, .{ .mtp = true, .drafts = admit_mod.draft_rows })).head;
    try testing.expectEqual(lone.weights + want, v.weights);
    const unstated = try admit(a, .{ .config = eight, .index = stock_index, .mtp = mtp }, ask, &problem);
    try testing.expectEqual(v.weights, unstated.weights);
    const six = "{\"group_size\": 64, \"bits\": 6}";
    var g = f;
    g.mtp_quantization = six;
    const wide = try admit(a, g, ask, &problem);
    try testing.expectEqual(config.Width{ .bits = 6, .group = 64 }, wide.head);
    try testing.expect(wide.weights > v.weights);
    g.mtp_quantization = "{\"group_size\": 32, \"bits\": 4}";
    try expectRefused(a, g, ask, &.{ tensors.mtp_file, "4-bit in groups of 32" });
    g.mtp_quantization = "nvfp4";
    try expectRefused(a, g, ask, &.{ tensors.mtp_file, "nvfp4" });
}
