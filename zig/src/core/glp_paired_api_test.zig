//! The paired GLP vector's writer, its model checks, its strength rules and the one loader the consumers share.

const std = @import("std");
const glp = @import("glp.zig");
const gguf = @import("glp_container.zig");
const t = @import("glp_paired_test.zig");

const io = std.testing.io;

/// One direction per id, every component `id / 8` so a layer's numbers name it.
fn layersOf(a: std.mem.Allocator, ids: []const u32, width: usize) ![]glp.Layer {
    const layers = try a.alloc(glp.Layer, ids.len);
    for (ids, layers) |id, *l| {
        const dir = try a.alloc(f32, width);
        @memset(dir, @as(f32, @floatFromInt(id)) / 8);
        l.* = .{ .id = id, .dir = dir };
    }
    return layers;
}

/// What a producer writes for the paired site; the caller bends one field at a time.
fn pairedSpec(layers: []const glp.Layer) glp.WriteSpec {
    return .{
        .spec_version = 3,
        .hook_point = glp.paired_site,
        .alpha = 0.25,
        .base_model = t.repo,
        .base_revision = t.commit,
        .method = "synthetic_fixture",
        .contrast = "none",
        .publication = .{ .name = "a paired name", .author = "tests", .license = "Apache-2.0", .created = "2026-11-20" },
        .layers = layers,
    };
}

test "a written paired vector is read back as paired, with every field and the keys a consumer sees" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = try layersOf(a, &.{ 5, 2, 9 }, 8);
    const bytes = try glp.write(a, pairedSpec(layers));
    const v = try t.readOk(a, bytes);
    try std.testing.expectEqual(@as(u32, 3), v.spec_version);
    try std.testing.expect(v.isPaired());
    try std.testing.expectEqual(@as(f32, 0.25), v.alpha);
    try std.testing.expectEqualStrings(t.repo, v.base_model);
    try std.testing.expectEqualStrings(t.commit, v.base_revision);
    try std.testing.expectEqualStrings("a paired name", v.publication.name.?);
    try std.testing.expectEqualStrings("Apache-2.0", v.publication.license.?);
    try std.testing.expect(v.digest_claimed);
    try std.testing.expectEqual(@as(u32, 2), v.layers[0].id);
    try std.testing.expectEqual(@as(f32, 9.0 / 8.0), v.layer(9).?[7]);
    // the container itself says what it is: version 3, the paired site, add, rank one
    var problem: gguf.Problem = .{};
    const file = try gguf.parse(a, bytes, &problem);
    try std.testing.expectEqual(@as(u64, 3), file.num("glp.spec_version").?);
    try std.testing.expectEqualStrings("hc_fold_pre_engram", file.str("glp.hook_point").?);
    try std.testing.expectEqualStrings("add", file.str("glp.mode").?);
    try std.testing.expectEqual(@as(u64, 1), file.num("glp.rank").?);
    try std.testing.expectEqualStrings("2,5,9", file.str("glp.layer_ids_zero_based").?);
}

test "a paired vector written again keeps its version, its site and every byte" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try glp.write(a, pairedSpec(try layersOf(a, &.{ 1, 2, 3 }, 8)));
    const again = try glp.writeVector(a, try t.readOk(a, bytes));
    try std.testing.expectEqualSlices(u8, bytes, again);
    // a built (producer-shaped) file keeps its version and site through a rewrite too
    const built = try t.readOk(a, try t.build(a, .{}));
    const rebuilt = try t.readOk(a, try glp.writeVector(a, built));
    try std.testing.expectEqual(@as(u32, 3), rebuilt.spec_version);
    try std.testing.expectEqualStrings(glp.paired_site, rebuilt.hook_point);
    try std.testing.expectEqualSlices(u8, &built.sha256, &rebuilt.sha256);
    try std.testing.expectEqualStrings(t.repo, rebuilt.base_model);
}

test "a zero strength survives a write: a loaded file with no effect, not an absent one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var spec = pairedSpec(try layersOf(a, &.{1}, 8));
    spec.alpha = 0;
    const v = try t.readOk(a, try glp.write(a, spec));
    try std.testing.expectEqual(@as(f32, 0), v.alpha);
    const g: glp.Glp = .{ .vector = v, .alpha = v.alpha };
    try std.testing.expect(g.off());
}

test "a base model written as an id, the hub's URL, another URL or a bare name is read back as that repo" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = try layersOf(a, &.{1}, 8);
    const forms = [_][2][]const u8{
        .{ t.repo, t.repo },
        .{ "https://huggingface.co/" ++ t.repo, t.repo },
        .{ "https://models.example/synthetic", "https://models.example/synthetic" },
        .{ "Synthetic-HC-Model", "Synthetic-HC-Model" },
        .{ "", "" },
    };
    for (forms) |form| {
        var spec = pairedSpec(layers);
        spec.base_model = form[0];
        const v = try t.readOk(a, try glp.write(a, spec));
        try std.testing.expectEqualStrings(form[1], v.base_model);
        try std.testing.expectEqualStrings(t.commit, v.base_revision);
    }
}

test "the residual and the paired vector never turn into each other on a write" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = try layersOf(a, &.{ 1, 2 }, 8);
    // version 1, the default, is the residual vector
    const residual = try t.readOk(a, try glp.write(a, .{ .layers = layers }));
    try std.testing.expectEqual(@as(u32, 1), residual.spec_version);
    try std.testing.expectEqualStrings(glp.residual_site, residual.hook_point);
    // a paired hook under version 1, the residual hook under version 3, and any other version are refused
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .hook_point = glp.paired_site, .layers = layers }));
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .spec_version = 3, .layers = layers }));
    for ([_]u32{ 0, 2, 4 }) |version| {
        try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .spec_version = version, .hook_point = glp.paired_site, .layers = layers }));
        try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .spec_version = version, .layers = layers }));
    }
    // a vector read as one cannot be written as the other
    var relabelled = try t.readOk(a, try glp.write(a, pairedSpec(layers)));
    relabelled.spec_version = 1;
    try std.testing.expectError(error.BadGlpVector, glp.writeVector(a, relabelled));
    var forced = residual;
    forced.spec_version = 3;
    try std.testing.expectError(error.BadGlpVector, glp.writeVector(a, forced));
    // and a residual vector rewritten stays the residual vector
    const again = try t.readOk(a, try glp.writeVector(a, residual));
    try std.testing.expectEqual(@as(u32, 1), again.spec_version);
}

test "a paired vector the reader would refuse is refused by the writer too" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const good = try layersOf(a, &.{ 1, 2 }, 8);
    for ([_]f32{ -1, std.math.nan(f32), std.math.inf(f32), 1e-40 }) |alpha| {
        var spec = pairedSpec(good);
        spec.alpha = alpha;
        try std.testing.expectError(error.BadGlpVector, glp.write(a, spec));
    }
    const zero = try layersOf(a, &.{ 0, 2 }, 8);
    try std.testing.expectError(error.BadGlpVector, glp.write(a, pairedSpec(zero)));
    const twice = try layersOf(a, &.{ 2, 2 }, 8);
    try std.testing.expectError(error.BadGlpVector, glp.write(a, pairedSpec(twice)));
    const uneven = [_]glp.Layer{ good[0], .{ .id = 2, .dir = good[1].dir[0..4] } };
    try std.testing.expectError(error.BadGlpVector, glp.write(a, pairedSpec(&uneven)));
    const bad = try layersOf(a, &.{1}, 8);
    @constCast(bad[0].dir)[3] = std.math.nan(f32);
    try std.testing.expectError(error.BadGlpVector, glp.write(a, pairedSpec(bad)));
    try std.testing.expectError(error.BadGlpVector, glp.write(a, pairedSpec(&.{})));
}

test "the paired check holds a residual vector out, and the residual check holds a paired one out" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = try layersOf(a, &.{ 1, 2 }, 8);
    const residual = try t.readOk(a, try glp.write(a, .{ .layers = layers, .base_model = t.repo }));
    const paired = try t.readOk(a, try glp.write(a, pairedSpec(layers)));
    var why: glp.Why = .{};
    try glp.checkModel(residual, 8, t.repo, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(paired, 8, t.repo, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "residual vectors (version 1) only") != null);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(paired, 8, "", &why));
    try glp.checkPairedModel(paired, 8, 3, t.repo, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(residual, 8, 3, t.repo, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "takes version 3 at hc_fold_pre_engram") != null);
}

test "the paired check holds the width, the backbone's layers and the very repo" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try t.readOk(a, try t.build(a, .{}));
    var why: glp.Why = .{};
    try glp.checkPairedModel(v, 8, 3, t.repo, &why);
    try glp.checkPairedModel(v, 8, 3, "", &why);
    // the width
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(v, 16, 3, t.repo, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "8 wide; the model is 16") != null);
    // the deepest layer is the last backbone block: layer 2 steers a model of 3 layers and not one of 2
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(v, 8, 2, t.repo, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "layer 2 of a model with 2 backbone layers: its deepest steerable layer is 1") != null);
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(v, 8, 0, t.repo, &why));
    // layer 0, unordered ids and a short direction cannot come from the reader, and are refused if a caller builds them
    const dir: [8]f32 = @splat(0);
    var hand = v;
    hand.layers = &.{.{ .id = 0, .dir = &dir }};
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(hand, 8, 3, t.repo, &why));
    hand.layers = &.{ .{ .id = 2, .dir = &dir }, .{ .id = 1, .dir = &dir } };
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(hand, 8, 3, t.repo, &why));
    hand.layers = &.{.{ .id = 1, .dir = dir[0..4] }};
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(hand, 8, 3, t.repo, &why));
    hand.layers = &.{};
    try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(hand, 8, 3, t.repo, &why));
    // the repo: the same one in any case or as the hub's URL, and nothing merely like it
    for ([_][]const u8{ "EXAMPLE-ORG/synthetic-hc-model", "https://huggingface.co/example-org/Synthetic-HC-Model" }) |same| {
        try glp.checkPairedModel(v, 8, 3, same, &why);
    }
    for ([_][]const u8{ "example-org/Synthetic-HC-Model-EXL3", "other-org/Synthetic-HC-Model", "example-org/Synthetic-HC", "Synthetic-HC-Model", "x/y" }) |other| {
        try std.testing.expectError(error.BadGlpVector, glp.checkPairedModel(v, 8, 3, other, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "the vector is for example-org/Synthetic-HC-Model; the model is") != null);
    }
    // a file that names no checkpoint takes any
    const unnamed = try t.readOk(a, try t.build(a, .{ .edits = &.{
        .{ .key = "general.base_model.count", .value = null },
        .{ .key = "general.base_model.0.name", .value = null },
        .{ .key = "general.base_model.0.organization", .value = null },
        .{ .key = "general.base_model.0.version", .value = null },
        .{ .key = "general.base_model.0.repo_url", .value = null },
    } }));
    try glp.checkPairedModel(unnamed, 8, 3, "other-org/Anything", &why);
}

test "a pinned revision matches the served commit, in full or cut to seven characters, and nothing else" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try t.readOk(a, try t.build(a, .{}));
    var why: glp.Why = .{};
    try glp.checkPairedRevision(v, t.commit, &why);
    try glp.checkPairedRevision(v, "0123456", &why);
    try glp.checkPairedRevision(v, "0123456789ABCDEF", &why);
    try glp.checkPairedRevision(v, "", &why);
    for ([_][]const u8{ "012345", "0123457", "fedcba9876543210", "0123456789abcdef0123456789abcdef01234568" }) |other| {
        try std.testing.expectError(error.BadGlpVector, glp.checkPairedRevision(v, other, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "the vector is for revision") != null);
    }
    var unpinned = v;
    unpinned.base_revision = "";
    try glp.checkPairedRevision(unpinned, "fedcba9876543210", &why);
}

test "a strength is zero or a normal float from 0, and an edit past what BF16 holds is refused" {
    for ([_]f64{ 0, -0.0, 1, 0.5, 4096, 1e30, std.math.floatMin(f32), 3.0e38 }) |x| try std.testing.expect(glp.strength(x) != null);
    for ([_]f64{ -1, -1e-300, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 1e-40, 1e-300, 1e39, 1e300 }) |x| try std.testing.expect(glp.strength(x) == null);
    try std.testing.expect(!std.math.signbit(glp.strength(-0.0).?));

    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const big = try t.readOk(a, try t.build(a, .{ .value = 1e38 }));
    var why: glp.Why = .{};
    try glp.checkStrength(big, 0, &why);
    try glp.checkStrength(big, 1, &why);
    try glp.checkStrength(big, 3, &why);
    for ([_]f32{ 4, 10, 1e30 }) |alpha| {
        try std.testing.expectError(error.BadGlpVector, glp.checkStrength(big, alpha, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "past what BF16 holds") != null);
    }
    for ([_]f32{ -1, std.math.nan(f32), std.math.inf(f32) }) |alpha| {
        try std.testing.expectError(error.BadGlpVector, glp.checkStrength(big, alpha, &why));
    }
}

test "only a strength of exactly zero turns every edit off" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try t.readOk(a, try t.build(a, .{}));
    try std.testing.expect((glp.Glp{ .vector = v, .alpha = 0 }).off());
    try std.testing.expect((glp.Glp{ .vector = v, .alpha = -0.0 }).off());
    try std.testing.expect(!(glp.Glp{ .vector = v, .alpha = std.math.floatMin(f32) }).off());
    try std.testing.expect(!(glp.Glp{ .vector = v, .alpha = 1 }).off());
}

/// A file in a fresh folder, by the absolute path the loader opens.
fn put(a: std.mem.Allocator, tmp: *std.testing.TmpDir, name: []const u8, bytes: []const u8) ![]const u8 {
    try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    return std.fs.path.join(a, &.{ dir, name });
}

test "the loader takes the file's strength or the operator's, zero included, and checks it" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try t.build(a, .{});
    const path = try put(a, &tmp, "paired.gguf", bytes);
    var why: glp.Why = .{};
    const own = try glp.load(a, io, path, null, &why);
    try std.testing.expectEqual(@as(f32, 0.5), own.alpha);
    try std.testing.expect(own.vector.isPaired() and !own.off());
    var file_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &file_sha, .{});
    try std.testing.expectEqualSlices(u8, &file_sha, &own.vector.file_sha256);
    const given = try glp.load(a, io, path, 0.125, &why);
    try std.testing.expectEqual(@as(f32, 0.125), given.alpha);
    // zero is a strength: the file loads, every edit off, its hash intact
    const off = try glp.load(a, io, path, 0, &why);
    try std.testing.expect(off.off());
    try std.testing.expectEqualSlices(u8, &file_sha, &off.vector.file_sha256);
    const negative = try glp.load(a, io, path, -0.0, &why);
    try std.testing.expect(negative.off() and !std.math.signbit(negative.alpha));
    // a strength FP32 cannot hold as itself is refused, not rounded to something else
    for ([_]f32{ -1, std.math.nan(f32), std.math.inf(f32), 1e-40 }) |alpha| {
        try std.testing.expectError(error.BadGlpVector, glp.load(a, io, path, alpha, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "not finite and from 0") != null);
    }
}

test "the loader refuses an edit BF16 cannot hold, and a file the reader refuses, by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var why: glp.Why = .{};
    const big = try put(a, &tmp, "big.gguf", try t.build(a, .{ .value = 1e38 }));
    try std.testing.expectError(error.BadGlpVector, glp.load(a, io, big, 10, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "past what BF16 holds") != null);
    const fits = try glp.load(a, io, big, 0, &why);
    try std.testing.expect(fits.off());
    const broken = try put(a, &tmp, "broken.gguf", try t.build(a, .{ .edits = &.{.{ .key = "glp.mode", .value = .{ .string = "project" } }} }));
    try std.testing.expectError(error.BadGlpVector, glp.load(a, io, broken, null, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "add vectors only") != null);
    const missing = try std.fs.path.join(a, &.{ std.fs.path.dirname(big).?, "no-such-vector.gguf" });
    try std.testing.expectError(error.BadGlpVector, glp.readFile(a, io, missing, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "cannot read") != null);
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "no such file") != null);
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "no-such-vector.gguf") != null);
}

test "a residual file loads as it did: its own strength must be valid unless the operator gives one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const residual = [_]t.Edit{
        .{ .key = "glp.spec_version", .value = .{ .u32v = 1 } },
        .{ .key = "glp.hook_point", .value = .{ .string = glp.residual_site } },
        .{ .key = "glp.derived_at", .value = .{ .string = glp.residual_site } },
    };
    var why: glp.Why = .{};
    const fine = try put(a, &tmp, "residual.gguf", try t.build(a, .{ .edits = &residual }));
    const g = try glp.load(a, io, fine, null, &why);
    try std.testing.expectEqual(@as(u32, 1), g.vector.spec_version);
    try std.testing.expectEqual(@as(f32, 0.5), g.alpha);
    // a residual file's own negative strength is refused, unless the operator names another
    const bad_edits = residual ++ [_]t.Edit{.{ .key = "glp.alpha_default", .value = .{ .f32v = -1 } }};
    const bad = try put(a, &tmp, "negative.gguf", try t.build(a, .{ .edits = &bad_edits }));
    try std.testing.expectError(error.BadGlpVector, glp.load(a, io, bad, null, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "alpha_default is -1") != null);
    try std.testing.expectEqual(@as(f32, 0.75), (try glp.load(a, io, bad, 0.75, &why)).alpha);
    // zero loads, and a family that needs a positive strength refuses it where it knows the family
    const zero_edits = residual ++ [_]t.Edit{.{ .key = "glp.alpha_default", .value = .{ .f32v = 0 } }};
    const zero = try put(a, &tmp, "zero.gguf", try t.build(a, .{ .edits = &zero_edits }));
    try std.testing.expect((try glp.load(a, io, zero, null, &why)).off());
}
