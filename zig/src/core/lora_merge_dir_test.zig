//! `tensorfold merge --quantize` on synthetic folders: the merge, its refusals and its all-or-nothing writes.
const std = @import("std");
const st = @import("safetensors.zig");
const se = @import("shard_edit.zig");
const fx = @import("lora_fixture.zig");
const dir = @import("lora_merge_dir.zig");
const lm = dir.math;
const lp = dir.plan;
const Io = std.Io;

const io = fx.io;
const files = fx.files;
const module = fx.module;

fn contains(hay: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, hay, needle) != null;
}

fn absent(s: *fx.Scratch, name: []const u8) !void {
    try std.testing.expectError(error.FileNotFound, Io.Dir.cwd().statFile(io, try s.path(&.{name}), .{}));
}

test "a zero adapter yields every shipped file byte for byte: the fresh quantization is the shipped one" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .zero = true }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expect(contains(r.out, "0 of 16 groups differ"));
    for (files) |f| try std.testing.expectEqualStrings(try s.read("base", f), try s.read("merged", f));
    // nothing but the finished folder is left beside it
    var staged: usize = 0;
    for (try s.list(".")) |name| staged += @intFromBool(std.mem.startsWith(u8, name, "merged"));
    try std.testing.expectEqual(@as(usize, 1), staged);
}

test "a random adapter changes only the named projection and passes every gate" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const before = [_][]const u8{ try s.read("base", files[0]), try s.read("base", files[2]), try s.read("base", files[3]) };
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expectEqualStrings(try s.read("base", files[1]), try s.read("merged", files[1]));
    try std.testing.expectEqualStrings(before[0], try s.read("merged", files[0]));
    try std.testing.expectEqualStrings(before[2], try s.read("merged", files[3]));
    try std.testing.expectEqualStrings(before[1], try s.read("base", files[2]));
    var big: [16384]u8 = undefined;
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(g.ok);
    try std.testing.expect(contains(g.out, "merge check: all gates pass"));
}

test "a failure on the second projection leaves no OUT and no staging folder" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{ .{}, .{ .module = "backbone.layers.1.mixer.out_proj", .k = 64, .nan = true } } });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "backbone.layers.1.mixer.out_proj"));
    try std.testing.expect(contains(r.out, "layers.0.mixer.in_proj")); // the first projection had been merged by then
    try absent(&s, "merged");
    for (try s.list(".")) |name| try std.testing.expect(!std.mem.startsWith(u8, name, "merged"));
}

test "an empty existing OUT is replaced, a non-empty one is refused untouched" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{});
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"merged"}));
    var buf: [4096]u8 = undefined;
    var r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expectEqual(files.len, (try s.list("merged")).len);
    try s.write("merged", "keep.txt", "mine");
    try Io.Dir.cwd().deleteFile(io, try s.path(&.{ "merged", "config.json" }));
    r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "new or empty"));
    try std.testing.expectEqualStrings("mine", try s.read("merged", "keep.txt"));
}

/// The index of the shipped folder with every leaf of `module` mapped to `shard`.
fn remap(s: *fx.Scratch, shard: []const u8) !void {
    const a = s.arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{\"weight_map\":{");
    for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |leaf, i| try out.print(a, "{s}\"{s}.{s}\":\"{s}\"", .{ if (i == 0) "" else ",", module, leaf, shard });
    try out.appendSlice(a, "}}");
    try s.write("base", "model.safetensors.index.json", out.items);
}

test "index shard paths that leave the folder are refused before anything is written" {
    for ([_][]const u8{ "../shared.safetensors", "/etc/shared.safetensors", "sub/model.safetensors", "..", ".", "", "a\\\\b" }) |shard| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
        try s.write(".", "shared.safetensors", "original");
        try remap(&s, shard);
        const adapter = try fx.makeAdapter(&s, .{});
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, "plain file name"));
        try std.testing.expectEqualStrings("original", try s.read(".", "shared.safetensors"));
        try absent(&s, "merged");
    }
}

test "a source subdirectory is refused with its name, not skipped" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{ "base", "extras" }));
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "extras"));
    try absent(&s, "merged");
}

test "a shipped shard that is a symlink is copied as a file; the file it points to is never written" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"elsewhere"}));
    const real = try s.path(&.{ "elsewhere", "shard" });
    try Io.Dir.cwd().rename(try s.path(&.{ "base", files[2] }), Io.Dir.cwd(), real, io);
    try Io.Dir.cwd().symLink(io, "../elsewhere/shard", try s.path(&.{ "base", files[2] }), .{});
    const before = try s.read("elsewhere", "shard");
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expectEqualStrings(before, try s.read("elsewhere", "shard"));
    try std.testing.expect(!std.mem.eql(u8, before, try s.read("merged", files[2])));
    const kind = (try Io.Dir.cwd().statFile(io, try s.path(&.{ "merged", files[2] }), .{ .follow_symlinks = false })).kind;
    try std.testing.expectEqual(Io.File.Kind.file, kind);
}

/// A shipped folder whose leaves of `module` have the given scales and biases shapes, and a default adapter.
fn oddLayout(s: *fx.Scratch, rows: usize, words_cols: usize, sc: []const usize, bi: []const usize) !struct { base: []const u8, adapter: []const u8 } {
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    var quantized: std.ArrayList(se.New) = .empty;
    try fx.leaves(a, module, 8, 128, try fx.randomValues(a, 8 * 128, 1), &dense, &quantized);
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"dense"}));
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"base"}));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    const odd = [_]se.New{
        .{ .name = module ++ ".weight", .dtype = .u32, .shape = &.{ rows, words_cols }, .bytes = &.{} },
        .{ .name = module ++ ".scales", .dtype = .bf16, .shape = sc, .bytes = &.{} },
        .{ .name = module ++ ".biases", .dtype = .bf16, .shape = bi, .bytes = &.{} },
    };
    try se.create(std.testing.allocator, try s.path(&.{ "base", "model.safetensors" }), &.{}, &odd);
    try s.write("base", "config.json", fx.base_config);
    return .{ .base = try s.path(&.{"base"}), .adapter = try fx.makeAdapter(s, .{}) };
}

test "malformed layouts are refused naming the tensor: biases rank, empty and overflowing dimensions" {
    const cases = [_]struct { rows: usize, cols: usize, sc: []const usize, bi: []const usize, name: []const u8 }{
        .{ .rows = 0, .cols = 0, .sc = &.{ 0, 2 }, .bi = &.{ 0, 2, 1 }, .name = ".scales" },
        .{ .rows = 0, .cols = 16, .sc = &.{ 0, 2 }, .bi = &.{ 0, 2 }, .name = "empty" },
        .{ .rows = 0, .cols = 16, .sc = &.{ 0, 1 << 62 }, .bi = &.{ 0, 1 << 62 }, .name = "overflow" },
    };
    for (cases) |c| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const o = try oddLayout(&s, c.rows, c.cols, c.sc, c.bi);
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, o.base, o.adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, c.name) or contains(r.err, module));
        try absent(&s, "merged");
    }
}

test "adapter r and alpha: out-of-range values are refused, alpha 0 is a valid zero adapter" {
    for ([_][]const u8{ "1e30", "1e300", "0", "-2", "2.5" }) |rank| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, fx.base_config);
        const adapter = try fx.makeAdapter(&s, .{ .rank = rank });
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try absent(&s, "merged");
    }
    for ([_][]const u8{ "-1", "1e999" }) |alpha| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, fx.base_config);
        const adapter = try fx.makeAdapter(&s, .{ .alpha = alpha });
        var buf: [4096]u8 = undefined;
        try std.testing.expectEqual(@as(u8, 1), (try fx.merge(&s, base, adapter, &buf)).code);
        try absent(&s, "merged");
    }
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{ .alpha = "0" });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    for (files) |f| try std.testing.expectEqualStrings(try s.read("base", f), try s.read("merged", f));
}

test "both PEFT spellings of one factor are refused, naming both keys" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const names = [_][]const u8{
        "base_model.model." ++ module ++ ".lora_A.weight",         "base_model.model." ++ module ++ ".lora_B.weight",
        "base_model.model." ++ module ++ ".lora_A.default.weight", "base_model.model." ++ module ++ ".lora_B.default.weight",
    };
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{ .{}, .{} }, .names = &names });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, ".lora_A.default.weight") and contains(r.err, "duplicates"));
    try std.testing.expect(contains(r.err, ".lora_A.weight"));
    try absent(&s, "merged");
}

test "non-finite dense values, factors and merged values are refused naming the module" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
    const a = s.arena.allocator();
    var dense: std.ArrayList(se.New) = .empty;
    const vals = try a.alloc(u16, 8 * 128);
    @memset(vals, 0x3f80);
    vals[9] = 0x7f80; // an infinity
    try dense.append(a, .{ .name = module ++ ".weight", .dtype = .bf16, .shape = &.{ 8, 128 }, .bytes = std.mem.sliceAsBytes(vals) });
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, dense.items);
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, module) and contains(r.err, "non-finite"));
    try absent(&s, "merged");
}

test "a merged value beyond f32 is refused with the group" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .value = 3e38 }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, module) and contains(r.err, "group"));
    try absent(&s, "merged");
}

test "refusals name the tensor: unserved module, fused or absent projection, shape mismatch, 8-bit, variants" {
    const Case = struct { cfg: []const u8, o: fx.Adapter, name: []const u8 };
    const cases = [_]Case{
        .{ .cfg = fx.base_config, .o = .{ .specs = &.{.{ .module = "backbone.layers.0.mixer.experts.0.up_proj" }} }, .name = "backbone.layers.0.mixer.experts.0.up_proj" },
        .{ .cfg = fx.base_config, .o = .{ .specs = &.{.{ .module = "model.layers.0.self_attn.q_proj" }} }, .name = "model.layers.0.self_attn.q_proj.weight" },
        .{ .cfg = fx.base_config, .o = .{ .specs = &.{.{ .k = 64 }} }, .name = module },
        .{ .cfg = fx.base_config, .o = .{ .specs = &.{.{ .n = 4 }} }, .name = module },
        .{ .cfg = "{\"quantization\":{\"group_size\":64,\"bits\":4,\"" ++ module ++ "\":{\"group_size\":64,\"bits\":8}}}", .o = .{}, .name = "4-bit" },
        .{ .cfg = fx.base_config, .o = .{ .extra = ",\"use_dora\":true" }, .name = "use_dora" },
    };
    for (cases) |c| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, c.cfg);
        const adapter = try fx.makeAdapter(&s, c.o);
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, c.name));
        try absent(&s, "merged");
    }
}

test "a dense base that lacks the tensor, or holds it at another shape, is refused naming it" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{});
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, &.{.{ .name = "backbone.layers.0.norm.weight", .dtype = .bf16, .shape = &.{1}, .bytes = "ab" }});
    var buf: [4096]u8 = undefined;
    var r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, module ++ ".weight"));
    try se.create(std.testing.allocator, try s.path(&.{ "dense", "model.safetensors" }), &.{}, &.{.{ .name = module ++ ".weight", .dtype = .bf16, .shape = &.{ 8, 64 }, .bytes = &@as([1024]u8, @splat(0)) }});
    r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "dense shape"));
    try absent(&s, "merged");
}

test "a dangling symlink in the source is refused naming it, never skipped" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    try Io.Dir.cwd().symLink(io, "no-such-file", try s.path(&.{ "base", "tokenizer.json" }), .{});
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "tokenizer.json") and contains(r.err, "dangling symlink"));
    try absent(&s, "merged");
}

test "publishing replaces an empty OUT, creates a missing one and never overwrites a non-empty one" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    var buf: [512]u8 = undefined;
    var err = Io.Writer.fixed(&buf);
    try s.write("stage", "a.txt", "staged");
    try dir.publish(io, &err, try s.path(&.{"stage"}), try s.path(&.{"out"}));
    try std.testing.expectEqualStrings("staged", try s.read("out", "a.txt"));
    try absent(&s, "stage");
    // an empty OUT is replaced in the one rename
    try s.write("stage", "b.txt", "second");
    try Io.Dir.cwd().createDirPath(io, try s.path(&.{"empty"}));
    try dir.publish(io, &err, try s.path(&.{"stage"}), try s.path(&.{"empty"}));
    try std.testing.expectEqualStrings("second", try s.read("empty", "b.txt"));
    // a non-empty OUT that appeared after any earlier check is refused and keeps its files
    try s.write("stage", "c.txt", "third");
    try s.write("late", "mine.txt", "mine");
    try std.testing.expectError(error.Refused, dir.publish(io, &err, try s.path(&.{"stage"}), try s.path(&.{"late"})));
    try std.testing.expectEqualStrings("mine", try s.read("late", "mine.txt"));
    try std.testing.expectEqual(@as(usize, 1), (try s.list("late")).len);
    try std.testing.expectEqualStrings("third", try s.read("stage", "c.txt"));
    try std.testing.expect(contains(err.buffered(), "new or empty"));
}

test "a shard whose tensor payload ranges overlap is refused before anything is written" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeSingle(&s, 8, 128, try fx.randomValues(s.arena.allocator(), 8 * 128, 4));
    const json =
        \\{"backbone.layers.0.mixer.in_proj.weight":{"dtype":"U32","shape":[8,16],"data_offsets":[0,512]},
        \\"backbone.layers.0.mixer.in_proj.scales":{"dtype":"BF16","shape":[8,2],"data_offsets":[512,544]},
        \\"backbone.layers.0.mixer.in_proj.biases":{"dtype":"BF16","shape":[8,2],"data_offsets":[544,576]},
        \\"alias":{"dtype":"U8","shape":[32],"data_offsets":[0,32]}}
    ;
    const a = s.arena.allocator();
    const file = try a.alloc(u8, 8 + json.len + 576);
    std.mem.writeInt(u64, file[0..8], json.len, .little);
    @memcpy(file[8..][0..json.len], json);
    @memset(file[8 + json.len ..], 0);
    try s.write("base", "model.safetensors", file);
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "model.safetensors") and contains(r.err, "overlap"));
    try absent(&s, "merged");
}

test "a directory standing in for the adapter file, an adapter or shipped config, or the shipped index is refused, not read" {
    const Stand = enum { adapter_file, adapter_config, shipped_config, shipped_index };
    for ([_]Stand{ .adapter_file, .adapter_config, .shipped_config, .shipped_index }) |t| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const base = try fx.makeBase(&s, fx.base_config);
        const adapter = try fx.makeAdapter(&s, .{});
        const target = switch (t) {
            .adapter_file => .{ "adapter", "adapter_model.safetensors" },
            .adapter_config => .{ "adapter", "adapter_config.json" },
            .shipped_config => .{ "base", "config.json" },
            .shipped_index => .{ "base", "model.safetensors.index.json" },
        };
        try Io.Dir.cwd().deleteFile(io, try s.path(&.{ target[0], target[1] }));
        try Io.Dir.cwd().createDirPath(io, try s.path(&.{ target[0], target[1] }));
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, target[1]) and contains(r.err, "not a regular file"));
        try absent(&s, "merged");
    }
}

test "an allocation that fails at any step of the merge leaves no OUT and no staging folder, and no leak" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeBase(&s, fx.base_config);
    const adapter = try fx.makeAdapter(&s, .{});
    var buf: [4096]u8 = undefined;
    var at: usize = 0;
    while (at < 2000) : (at += 1) {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = at });
        var out = Io.Writer.fixed(buf[0 .. buf.len / 2]);
        var err = Io.Writer.fixed(buf[buf.len / 2 ..]);
        const code = try dir.run(failing.allocator(), io, &out, &err, try s.path(&.{"dense"}), adapter, try s.path(&.{"merged"}), base);
        if (!failing.has_induced_failure) {
            try std.testing.expectEqual(@as(u8, 0), code);
            break;
        }
        try std.testing.expectEqual(@as(u8, 1), code);
        try absent(&s, "merged");
        for (try s.list(".")) |name| try std.testing.expect(!std.mem.startsWith(u8, name, "merged"));
    }
    try std.testing.expect(at > 10 and at < 2000);
}

const out_proj = "model.layers.1.linear_attn.out_proj";

test "MLP and linear-attention modules the shipped checkpoint quantizes are merged, whatever their names" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{
        .{ .dense = "model.layers.0.mlp.gate_proj", .shipped = "model.layers.0.mlp.gate_proj" },
        .{ .dense = "model.layers.0.linear_attn.in_proj_qkv", .shipped = "model.layers.0.linear_attn.in_proj_qkv" },
        .{ .dense = out_proj, .shipped = out_proj, .k = 64 },
    });
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{
        .{ .module = "model.layers.0.mlp.gate_proj" },
        .{ .module = "model.layers.0.linear_attn.in_proj_qkv" },
        .{ .module = out_proj, .k = 64 },
    } });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    try std.testing.expect(contains(r.out, "merged 3 projections"));
    try std.testing.expect(!std.mem.eql(u8, try s.read("base", "model.safetensors"), try s.read("merged", "model.safetensors")));
    var big: [16384]u8 = undefined;
    try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok);
}

const dense_name = "model.language_model.layers.0.linear_attn.out_proj";
const shipped_name = "language_model.model.layers.0.linear_attn.out_proj";

test "a dense name that the shipped checkpoint holds under a rewritten prefix is resolved" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{.{ .dense = dense_name, .shipped = shipped_name }});
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = dense_name }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    var big: [16384]u8 = undefined;
    const g = try fx.gates(&s, base, adapter, &big);
    try std.testing.expect(g.ok and contains(g.out, "0 of 16 groups differ from OUT"));
    // the leaves written are the shipped names, and the shard is the only thing that changed
    var f = try st.File.open(std.testing.allocator, io, try s.path(&.{ "merged", "model.safetensors" }));
    defer f.close(io);
    try std.testing.expect(f.get(shipped_name ++ ".scales") != null and f.get(dense_name ++ ".scales") == null);
}

test "a stacked expert tensor and an unquantized norm are refused with their reason" {
    const Case = struct { kind: fx.Kind, why: []const u8 };
    for ([_]Case{ .{ .kind = .stacked, .why = "rank 3" }, .{ .kind = .plain, .why = "not quantized" } }) |c| {
        var s = try fx.Scratch.init();
        defer s.deinit();
        const name = "model.layers.0.mlp.experts.fc1";
        const base = try fx.makeNamed(&s, &.{.{ .dense = name, .shipped = name, .kind = c.kind }});
        const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = name }} });
        var buf: [4096]u8 = undefined;
        const r = try fx.merge(&s, base, adapter, &buf);
        try std.testing.expectEqual(@as(u8, 1), r.code);
        try std.testing.expect(contains(r.err, name) and contains(r.err, c.why));
        try absent(&s, "merged");
    }
}

test "a module quantized in the shipped checkpoint but absent from the dense source is refused" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{.{ .dense = out_proj, .shipped = out_proj, .kind = .dense_absent }});
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = out_proj }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, out_proj ++ ".weight") and contains(r.err, "dense base"));
    try absent(&s, "merged");
}

test "two shipped names that both resolve are refused as ambiguous, naming every candidate" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{
        .{ .dense = dense_name, .shipped = dense_name },
        .{ .dense = "unused.weight_holder", .shipped = shipped_name, .kind = .dense_absent },
    });
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = dense_name }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, "ambiguous") and contains(r.err, dense_name ++ ".weight") and contains(r.err, shipped_name ++ ".weight"));
    try absent(&s, "merged");
}

test "a module found under no candidate name is refused listing each one tried" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{.{ .dense = out_proj, .shipped = out_proj }});
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = dense_name }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 1), r.code);
    try std.testing.expect(contains(r.err, dense_name ++ ".weight") and contains(r.err, shipped_name ++ ".weight"));
    try absent(&s, "merged");
}

test "an adapter that names the module in the shipped namespace is resolved to the dense name" {
    var s = try fx.Scratch.init();
    defer s.deinit();
    const base = try fx.makeNamed(&s, &.{.{ .dense = dense_name, .shipped = shipped_name }});
    const adapter = try fx.makeAdapter(&s, .{ .specs = &.{.{ .module = shipped_name }} });
    var buf: [4096]u8 = undefined;
    const r = try fx.merge(&s, base, adapter, &buf);
    try std.testing.expectEqual(@as(u8, 0), r.code);
    var big: [16384]u8 = undefined;
    try std.testing.expect((try fx.gates(&s, base, adapter, &big)).ok);
}
