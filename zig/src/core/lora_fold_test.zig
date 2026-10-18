//! `tensorfold merge`'s default, the fold onto the shipped scales: its arithmetic, folders, gates, forms and groups.
const std = @import("std");
const st = @import("safetensors.zig");
const se = @import("shard_edit.zig");
const host4 = @import("affine4_host.zig");
const fx = @import("lora_fixture.zig");
const dir = @import("lora_merge_dir.zig");
const lm = dir.math;
const lp = dir.plan;
const Io = std.Io;

const io = fx.io;
const module = fx.module;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

fn codeAt(words: []const u32, i: usize) u32 {
    return (words[i / 8] >> @intCast(4 * (i % 8))) & 0xf;
}

fn absent(s: *fx.Scratch, name: []const u8) !void {
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try s.path(&.{name}), .{}));
}

test "the fold moves each changed value to its group's nearest code, on a negative scale, clamped at the ends" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 32, 64, 128 }) |group| {
        const a = try gpa.alloc(f32, group);
        defer gpa.free(a);
        @memset(a, 0);
        // on the grid 2 - 0.5 q: unchanged, 1.48 and 1.52 steps in, past each end, and exactly the last step
        for ([_]f32{ 0, -0.74, -0.76, 1, -10, -7.5 }, 0..) |d, c| a[c] = d;
        const words = try gpa.alloc(u32, group / 8);
        defer gpa.free(words);
        @memset(words, 0);
        const scales = [_]u16{host4.bf16of(-0.5)};
        const biases = [_]u16{host4.bf16of(2)};
        const shipped: lm.View = .{ .words = words, .scales = &scales, .biases = &biases };
        const l: lm.Lora = .{ .a = a, .b = &[_]f32{1}, .r = 1, .scale = 1 };
        const out = try lm.allocQuant(gpa, 1, group, group);
        defer lm.freeQuant(gpa, out);
        var bad: usize = 0;
        try lm.foldShipped(gpa, group, 1, group, shipped, l, out, &bad);
        for ([_]u32{ 0, 1, 2, 0, 15, 15 }, 0..) |want, c| try std.testing.expectEqual(want, codeAt(out.words, c));
        for (6..group) |c| try std.testing.expectEqual(@as(u32, 0), codeAt(out.words, c));
        try std.testing.expectEqualSlices(u16, &scales, out.scales);
        try std.testing.expectEqualSlices(u16, &biases, out.biases);
        try std.testing.expectEqual(@as(usize, 0), try lm.offGrid(gpa, group, 1, group, shipped, l, lm.viewOf(out)));
        out.words[0] ^= 0x30; // value 1 now decodes a step past its nearest
        try std.testing.expectEqual(@as(usize, 1), try lm.offGrid(gpa, group, 1, group, shipped, l, lm.viewOf(out)));
        out.words[0] ^= 0x30;
        out.scales[0] = host4.bf16of(-0.25); // a moved table counts its whole group
        try std.testing.expect(try lm.offGrid(gpa, group, 1, group, shipped, l, lm.viewOf(out)) >= group);
    }
}

test "the fold refuses a non-finite shipped scale in a group it changes, naming the group, and leaves others alone" {
    const gpa = std.testing.allocator;
    var words: [8]u32 = @splat(0);
    const scales = [_]u16{ host4.bf16of(0.5), 0x7f80 };
    const biases = [_]u16{ 0, 0 };
    const shipped: lm.View = .{ .words = &words, .scales = &scales, .biases = &biases };
    var a: [64]f32 = @splat(0);
    a[40] = 1;
    const out = try lm.allocQuant(gpa, 1, 64, 32);
    defer lm.freeQuant(gpa, out);
    var bad: usize = 0;
    const l: lm.Lora = .{ .a = &a, .b = &[_]f32{1}, .r = 1, .scale = 1 };
    try std.testing.expectError(error.NonFinite, lm.foldShipped(gpa, 32, 1, 64, shipped, l, out, &bad));
    try std.testing.expectEqual(@as(usize, 1), bad);
    a[40] = 0;
    a[3] = 1;
    try lm.foldShipped(gpa, 32, 1, 64, shipped, l, out, &bad);
    try std.testing.expectEqual(@as(u32, 2), codeAt(out.words, 3));
}

test "the default fold writes codes only: tables and every other tensor and file as shipped, and its gates pass" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .value = 0.1 }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.fold(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expect(contains(r.out, "folded onto the shipped scales"));
    for ([_][]const u8{ fx.files[0], fx.files[1], fx.files[3] }) |f| try std.testing.expectEqualStrings(try s.read("base", f), try s.read("merged", f));
    var a = try st.File.open(std.testing.allocator, io, try s.path(&.{ "base", fx.shard_names[0] }));
    defer a.close(io);
    var b = try st.File.open(std.testing.allocator, io, try s.path(&.{ "merged", fx.shard_names[0] }));
    defer b.close(io);
    for ([_][]const u8{ module ++ ".scales", module ++ ".biases", "backbone.embeddings.weight" }) |name| try std.testing.expectEqualSlices(u8, a.get(name).?.bytes, b.get(name).?.bytes);
    try std.testing.expect(!std.mem.eql(u8, a.get(module ++ ".weight").?.bytes, b.get(module ++ ".weight").?.bytes));
    var big: [16384]u8 = undefined;
    const g = try fx.foldGates(&s, base, adapter, &big);
    try std.testing.expect(g.ok);
    try std.testing.expect(contains(g.out, "differ from OUT; codes off the nearest step of their group's shipped grid: 0"));
}

test "a zero adapter folds to every shipped file byte for byte, and the fold's gates pass" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .zero = true }} });
    var buf: [4096]u8 = undefined;
    try std.testing.expectEqual(@as(u8, 0), (try fx.fold(&s, base, adapter, &buf)).code);
    for (fx.files) |f| try std.testing.expectEqualStrings(try s.read("base", f), try s.read("merged", f));
    var big: [16384]u8 = undefined;
    const g = try fx.foldGates(&s, base, adapter, &big);
    try std.testing.expect(g.ok and contains(g.out, "zero effect: true"));
}

test "a folded code off its nearest step, or a moved scale, fails the fold's gates" {
    for ([_]bool{ false, true }) |table| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, fx.base_config);
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .value = 0.1 }} });
        var buf: [4096]u8 = undefined;
        try std.testing.expectEqual(@as(u8, 0), (try fx.fold(&s, base, adapter, &buf)).code);
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var err_buf: [256]u8 = undefined;
        var err = Io.Writer.fixed(&err_buf);
        var shards = try lp.Shards.init(std.testing.allocator, arena.allocator(), io, &err, try s.path(&.{"merged"}));
        defer shards.deinit();
        const j: lp.Job = .{ .module = module, .n = 8, .k = 128, .group = 64, .dtype = .u32 };
        const q = try lp.quantOf(std.testing.allocator, &shards, j);
        defer lp.free(std.testing.allocator, q);
        const name = if (table) module ++ ".scales" else module ++ ".weight";
        if (table) q.scales[3] ^= 1 else q.words[5] ^= 0x7 << 8;
        const bytes = if (table) std.mem.sliceAsBytes(q.scales) else std.mem.sliceAsBytes(q.words);
        const shape: [2]usize = if (table) .{ 8, 2 } else .{ 8, 16 };
        try se.overwrite(std.testing.allocator, io, try s.path(&.{"merged"}), &.{.{ .name = name, .dtype = if (table) .bf16 else .u32, .shape = &shape, .bytes = bytes }});
        var big: [16384]u8 = undefined;
        const g = try fx.foldGates(&s, base, adapter, &big);
        try std.testing.expect(!g.ok);
        try std.testing.expect(contains(g.out, if (table) "bytes outside the named tensors differ" else "1 differ from OUT"));
    }
}

test "merge's forms: MODEL ADAPTER OUT folds, DENSE ADAPTER OUT --quantize SHIPPED quantizes fresh, others refused" {
    const fold = dir.parseArgs(&.{ "m", "a", "o" }).?;
    try std.testing.expect(fold.dense == null);
    try std.testing.expectEqualStrings("m", fold.shipped);
    try std.testing.expectEqualStrings("a", fold.adapter);
    try std.testing.expectEqualStrings("o", fold.dest);
    const fresh = dir.parseArgs(&.{ "d", "a", "o", "--quantize", "s" }).?;
    try std.testing.expectEqualStrings("d", fresh.dense.?);
    try std.testing.expectEqualStrings("s", fresh.shipped);
    try std.testing.expectEqualStrings("a", fresh.adapter);
    try std.testing.expectEqualStrings("o", fresh.dest);
    for ([_][]const []const u8{ &.{}, &.{ "m", "a" }, &.{ "m", "a", "o", "x" }, &.{ "d", "a", "o", "--fold", "s" }, &.{ "d", "a", "o", "--quantize" }, &.{ "d", "a", "o", "--quantize", "s", "x" } }) |args|
        try std.testing.expect(dir.parseArgs(args) == null);
}

test "a dense folder given as the model to fold is refused before anything is written: it holds no quantization" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    _ = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.fold(&s, try s.path(&.{"dense"}), adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "config.json"));
    try absent(&s, "merged");
}

test "groups other than 32, 64 and 128 are refused before anything is written, folded or quantized fresh" {
    const configs = [_][]const u8{
        "{\"quantization\":{\"group_size\":256,\"bits\":4}}",
        "{\"quantization\":{\"group_size\":16,\"bits\":4}}",
        "{\"quantization\":{\"group_size\":64,\"bits\":4,\"" ++ module ++ "\":{\"group_size\":256,\"bits\":4}}}",
    };
    for (configs, 0..) |cfg, i| for ([_]bool{ false, true }) |fresh| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, cfg);
        const adapter = try fx.makeAdapter(&s, .{});
        var buf: [4096]u8 = undefined;
        const r = if (fresh) try fx.merge(&s, base, adapter, &buf) else try fx.fold(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, if (i < 2) "not MLX affine" else module));
        try absent(&s, "merged");
    };
}

const hf = "model.language_model.layers.0.linear_attn.out_proj"; // a dense source's spelling
const mlx = "language_model.model.layers.0.linear_attn.out_proj"; // an MLX conversion's

test "the fold finds a module named in either namespace under the checkpoint's own spelling" {
    for ([_][2][]const u8{ .{ hf, mlx }, .{ mlx, hf } }) |c| { // the adapter's name, then the checkpoint's
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeNamed(&s, &.{.{ .dense = c[1], .shipped = c[1] }});
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = c[0], .value = 0.1 }} });
        var buf: [4096]u8 = undefined;
        const r = try fx.fold(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 0), r.code);
        var big: [16384]u8 = undefined;
        try std.testing.expect((try fx.foldGates(&s, base, adapter, &big)).ok);
        var a = try st.File.open(std.testing.allocator, io, try s.path(&.{ "base", "model.safetensors" }));
        defer a.close(io);
        var b = try st.File.open(std.testing.allocator, io, try s.path(&.{ "merged", "model.safetensors" }));
        defer b.close(io);
        const weight = try std.fmt.allocPrint(s.arena.allocator(), "{s}.weight", .{c[1]});
        try std.testing.expect(!std.mem.eql(u8, a.get(weight).?.bytes, b.get(weight).?.bytes));
    }
}

test "the fold refuses a module whose two spellings the checkpoint both stores, naming both" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{ .{ .dense = hf, .shipped = hf }, .{ .dense = mlx, .shipped = mlx } });
    for ([_][]const u8{ hf, mlx }) |name| {
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = name }} });
        var buf: [4096]u8 = undefined;
        const r = try fx.fold(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, "ambiguous") and contains(r.err, hf ++ ".weight") and contains(r.err, mlx ++ ".weight"));
        try absent(&s, "merged");
    }
}

test "two adapter modules that resolve to one projection are refused before anything is written, in both modes" {
    for ([_]bool{ false, true }) |fresh| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeNamed(&s, &.{.{ .dense = hf, .shipped = mlx }});
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{ .{ .module = hf }, .{ .module = mlx } } });
        var buf: [4096]u8 = undefined;
        const r = if (fresh) try fx.merge(&s, base, adapter, &buf) else try fx.fold(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, "one adapter module a projection") and contains(r.err, hf) and contains(r.err, mlx));
        try absent(&s, "merged");
    }
}
