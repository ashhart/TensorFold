//! The DeepSeek paired GLP file's acceptance rules, over synthetic containers written the way a producer writes them.

const std = @import("std");
const glp = @import("glp.zig");
const gguf = @import("glp_container.zig");

/// One metadata change to a valid file: the key's new value, or null to drop the key.
pub const Edit = struct { key: []const u8, value: ?gguf.Value };

/// What a fixture holds, so a test can bend one rule at a time.
pub const Opts = struct {
    edits: []const Edit = &.{},
    /// Metadata appended as it stands, so a test can give a key twice.
    extra: []const gguf.Entry = &.{},
    ids: []const u32 = &.{ 1, 2 },
    /// Tensor names to write instead of "direction.<id>".
    names: ?[]const []const u8 = null,
    width: u64 = 8,
    /// Every component's value, when a test needs one the reader refuses.
    value: ?f32 = null,
    kind: u32 = gguf.f32_type,
    /// A second dimension, making every tensor two-dimensional.
    rows: u64 = 0,
    /// The last tensor's width, when it differs from the others.
    last_width: ?u64 = null,
};

/// The base repo a synthetic file names, and the commit it pins.
pub const repo = "example-org/Synthetic-HC-Model";
pub const commit = "0123456789abcdef0123456789abcdef01234567";

fn text(comptime key: []const u8, comptime value: []const u8) Edit {
    return .{ .key = key, .value = .{ .string = value } };
}

fn number(comptime key: []const u8, comptime value: u32) Edit {
    return .{ .key = key, .value = .{ .u32v = value } };
}

fn drop(comptime key: []const u8) Edit {
    return .{ .key = key, .value = null };
}

fn real(comptime key: []const u8, comptime value: f32) Edit {
    return .{ .key = key, .value = .{ .f32v = value } };
}

/// A valid version 3 container, bent by `o`; its bytes live in `a`.
pub fn build(a: std.mem.Allocator, o: Opts) ![]u8 {
    const count = o.ids.len;
    var payload: std.ArrayList(u8) = .empty;
    const starts = try a.alloc(usize, count + 1);
    const widths = try a.alloc(u64, count);
    for (0..count) |i| {
        widths[i] = if (i == count - 1) o.last_width orelse o.width else o.width;
        starts[i] = payload.items.len;
        for (0..widths[i] * @max(o.rows, 1)) |k| {
            const v: f32 = o.value orelse @as(f32, @floatFromInt(i + 1)) * 0.25 + @as(f32, @floatFromInt(k)) * 0.0625;
            try payload.appendSlice(a, &std.mem.toBytes(@as(u32, @bitCast(v))));
        }
    }
    starts[count] = payload.items.len;

    const tensors = try a.alloc(gguf.OutTensor, count);
    var ids: std.ArrayList(u8) = .empty;
    for (tensors, 0..) |*t, i| {
        const dims = try a.alloc(u64, if (o.rows > 0) 2 else 1);
        dims[0] = widths[i];
        if (o.rows > 0) dims[1] = o.rows;
        const name = if (o.names) |names| names[i] else try std.fmt.allocPrint(a, "direction.{d}", .{o.ids[i]});
        t.* = .{ .name = name, .dims = dims, .kind = o.kind, .data = payload.items[starts[i]..starts[i + 1]] };
        if (i > 0) try ids.append(a, ',');
        try ids.print(a, "{d}", .{o.ids[i]});
    }
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload.items, &sha, .{});
    const hex = try a.dupe(u8, &std.fmt.bytesToHex(sha, .lower));

    var entries: std.ArrayList(gguf.Entry) = .empty;
    const base = [_]gguf.Entry{
        .{ .key = "general.architecture", .value = .{ .string = "controlvector" } },
        .{ .key = "general.alignment", .value = .{ .u32v = 32 } },
        .{ .key = "controlvector.layer_count", .value = .{ .u32v = @intCast(count) } },
        .{ .key = "glp.spec_version", .value = .{ .u32v = 3 } },
        .{ .key = "glp.mode", .value = .{ .string = "add" } },
        .{ .key = "glp.rank", .value = .{ .u32v = 1 } },
        .{ .key = "glp.alpha_default", .value = .{ .f32v = 0.5 } },
        .{ .key = "glp.hook_point", .value = .{ .string = glp.paired_site } },
        .{ .key = "glp.derived_at", .value = .{ .string = glp.paired_site } },
        .{ .key = "glp.method", .value = .{ .string = "synthetic_fixture" } },
        .{ .key = "glp.contrast", .value = .{ .string = "none" } },
        .{ .key = "glp.layer_ids_zero_based", .value = .{ .string = ids.items } },
        .{ .key = "glp.content_sha256", .value = .{ .string = hex } },
        .{ .key = "general.base_model.count", .value = .{ .u32v = 1 } },
        .{ .key = "general.base_model.0.name", .value = .{ .string = "Synthetic-HC-Model" } },
        .{ .key = "general.base_model.0.organization", .value = .{ .string = "example-org" } },
        .{ .key = "general.base_model.0.version", .value = .{ .string = commit } },
        .{ .key = "general.base_model.0.repo_url", .value = .{ .string = "https://huggingface.co/" ++ repo } },
        .{ .key = "general.name", .value = .{ .string = "synthetic paired fixture" } },
        .{ .key = "general.author", .value = .{ .string = "tests" } },
    };
    try entries.appendSlice(a, &base);
    for (o.edits) |e| {
        const at = for (entries.items, 0..) |x, i| {
            if (std.mem.eql(u8, x.key, e.key)) break i;
        } else null;
        if (e.value) |v| {
            if (at) |i| entries.items[i].value = v else try entries.append(a, .{ .key = e.key, .value = v });
        } else if (at) |i| _ = entries.orderedRemove(i);
    }
    try entries.appendSlice(a, o.extra);
    return gguf.write(a, entries.items, tensors);
}

/// The vector a file holds; a refusal prints its reason and fails.
pub fn readOk(a: std.mem.Allocator, bytes: []const u8) !glp.Vector {
    var why: glp.Why = .{};
    return glp.read(a, bytes, &why) catch |err| {
        std.debug.print("refused: {s}\n", .{why.text()});
        return err;
    };
}

/// The reason a file was refused, copied into `a`; accepting it fails.
pub fn reason(a: std.mem.Allocator, bytes: []const u8) ![]const u8 {
    var why: glp.Why = .{};
    if (glp.read(a, bytes, &why)) |_| return error.NotRefused else |err| {
        if (err != error.BadGlpVector) return err;
        return a.dupe(u8, why.text());
    }
}

const Case = struct { opts: Opts, says: []const u8 };

const zero_digest = "0000000000000000000000000000000000000000000000000000000000000000";

const refusals = [_]Case{
    .{ .opts = .{ .edits = &.{drop("general.architecture")} }, .says = "no general.architecture" },
    .{ .opts = .{ .edits = &.{text("general.architecture", "llama")} }, .says = "not controlvector" },
    .{ .opts = .{ .edits = &.{number("general.architecture", 1)} }, .says = "general.architecture is not text" },
    // the operation: add only, never inferred, never another word
    .{ .opts = .{ .edits = &.{text("glp.mode", "project")} }, .says = "mode is project; this reader applies add vectors only" },
    .{ .opts = .{ .edits = &.{text("glp.mode", "scale")} }, .says = "not one this reader applies" },
    .{ .opts = .{ .edits = &.{text("glp.mode", "ADD")} }, .says = "not one this reader applies" },
    .{ .opts = .{ .edits = &.{drop("glp.mode")} }, .says = "no glp.mode" },
    .{ .opts = .{ .edits = &.{number("glp.mode", 1)} }, .says = "glp.mode is not text" },
    // rank and scale metadata
    .{ .opts = .{ .edits = &.{number("glp.rank", 2)} }, .says = "glp.rank 2" },
    .{ .opts = .{ .edits = &.{text("glp.rank", "1")} }, .says = "glp.rank is not an unsigned integer" },
    .{ .opts = .{ .edits = &.{.{ .key = "glp.rank", .value = .{ .i32v = -1 } }} }, .says = "glp.rank is not an unsigned integer" },
    .{ .opts = .{ .edits = &.{text("glp.orthonormal", "true")} }, .says = "glp.orthonormal is not a boolean" },
    .{ .opts = .{ .edits = &.{text("glp.dir_scales", "1.0")} }, .says = "glp.dir_scales" },
    .{ .opts = .{ .edits = &.{text("glp.layer_scales", "1:0.5")} }, .says = "glp.layer_scales" },
    .{ .opts = .{ .edits = &.{text("dspark.mode", "add")} }, .says = "dspark.* keys" },
    // the hook: only the paired site, whatever else the file calls it
    .{ .opts = .{ .edits = &.{drop("glp.hook_point")} }, .says = "names no glp.hook_point" },
    .{ .opts = .{ .edits = &.{number("glp.hook_point", 1)} }, .says = "glp.hook_point is not text" },
    .{ .opts = .{ .edits = &.{text("glp.hook_point", "residual_stream_post_layer")} }, .says = "is not applied" },
    .{ .opts = .{ .edits = &.{text("glp.hook_point", "ffn_out_pre_residual")} }, .says = "is not applied" },
    .{ .opts = .{ .edits = &.{text("glp.hook_point", "attn_out_pre_residual")} }, .says = "is not applied" },
    .{ .opts = .{ .edits = &.{text("glp.hook_point", "hc_fold_pre_engram ")} }, .says = "is not applied" },
    .{ .opts = .{ .edits = &.{text("glp.hook_point", "")} }, .says = "is not applied" },
    // the version, which decides what the other keys mean
    .{ .opts = .{ .edits = &.{number("glp.spec_version", 2)} }, .says = "glp.spec_version 2" },
    .{ .opts = .{ .edits = &.{number("glp.spec_version", 4)} }, .says = "glp.spec_version 4" },
    .{ .opts = .{ .edits = &.{number("glp.spec_version", 0)} }, .says = "glp.spec_version 0" },
    .{ .opts = .{ .edits = &.{text("glp.spec_version", "3")} }, .says = "glp.spec_version is not an unsigned integer" },
    .{ .opts = .{ .edits = &.{.{ .key = "glp.spec_version", .value = .{ .i32v = -3 } }} }, .says = "glp.spec_version is not an unsigned integer" },
    .{ .opts = .{ .edits = &.{number("glp.spec_version", 1)} }, .says = "needs glp.spec_version 3" },
    .{ .opts = .{ .edits = &.{drop("glp.spec_version")} }, .says = "glp keys but no glp.spec_version" },
    // the strength
    .{ .opts = .{ .edits = &.{text("glp.alpha_default", "1")} }, .says = "glp.alpha_default is not a floating-point strength" },
    .{ .opts = .{ .edits = &.{number("glp.alpha_default", 1)} }, .says = "glp.alpha_default is not a floating-point strength" },
    .{ .opts = .{ .edits = &.{real("glp.alpha_default", -1)} }, .says = "not a finite strength from 0" },
    .{ .opts = .{ .edits = &.{real("glp.alpha_default", std.math.nan(f32))} }, .says = "not a finite strength from 0" },
    .{ .opts = .{ .edits = &.{real("glp.alpha_default", std.math.inf(f32))} }, .says = "not a finite strength from 0" },
    .{ .opts = .{ .edits = &.{real("glp.alpha_default", 1e-40)} }, .says = "not a finite strength from 0" },
    .{ .opts = .{ .edits = &.{.{ .key = "glp.alpha_default", .value = .{ .f64v = 1e300 } }} }, .says = "not a finite strength from 0" },
    .{ .opts = .{ .edits = &.{.{ .key = "glp.alpha_default", .value = .{ .f64v = 1e-300 } }} }, .says = "not a finite strength from 0" },
    // the tensors
    .{ .opts = .{ .ids = &.{ 0, 2 } }, .says = "direction.0" },
    .{ .opts = .{ .ids = &.{ 5, 5 }, .names = &.{ "direction.5", "direction.05" } }, .says = "does not name a layer as direction.N" },
    .{ .opts = .{ .ids = &.{ 1, 1 }, .names = &.{ "direction.1", "direction.1" } }, .says = "two directions for layer 1" },
    .{ .opts = .{ .names = &.{ "direction.1", "weights.2" } }, .says = "which is not a direction" },
    .{ .opts = .{ .names = &.{ "direction.1", "direction.2.1" } }, .says = "a rank above one" },
    .{ .opts = .{ .names = &.{ "direction.1", "direction.x" } }, .says = "does not name a layer" },
    .{ .opts = .{ .names = &.{ "direction.1", "direction." } }, .says = "does not name a layer as direction.N" },
    .{ .opts = .{ .names = &.{ "direction.1", "direction.99999999999" } }, .says = "past 32 bits" },
    .{ .opts = .{ .kind = 1 }, .says = "is not fp32" },
    .{ .opts = .{ .rows = 2 }, .says = "is not one dimensional" },
    .{ .opts = .{ .last_width = 4 }, .says = "wide; the other directions are" },
    .{ .opts = .{ .value = std.math.nan(f32) }, .says = "not finite" },
    .{ .opts = .{ .value = std.math.inf(f32) }, .says = "not finite" },
    .{ .opts = .{ .value = -std.math.inf(f32) }, .says = "not finite" },
    // the digest, and the layer list the file also gives
    .{ .opts = .{ .edits = &.{text("glp.content_sha256", zero_digest)} }, .says = "content digest does not match" },
    .{ .opts = .{ .edits = &.{number("glp.content_sha256", 7)} }, .says = "glp.content_sha256 is not text" },
    .{ .opts = .{ .edits = &.{text("glp.layer_ids_zero_based", "1,3")} }, .says = "layer id 3 does not match" },
    .{ .opts = .{ .edits = &.{text("glp.layer_ids_zero_based", "1")} }, .says = "names 1 layers and holds 2" },
    .{ .opts = .{ .edits = &.{text("glp.layer_ids_zero_based", "1,2,3")} }, .says = "more layers than it holds" },
    .{ .opts = .{ .edits = &.{text("glp.layer_ids_zero_based", "a,b")} }, .says = "not a number list" },
    .{ .opts = .{ .edits = &.{number("glp.layer_ids_zero_based", 1)} }, .says = "glp.layer_ids_zero_based is not text" },
    // a key given twice has no single meaning
    .{ .opts = .{ .extra = &.{.{ .key = "glp.mode", .value = .{ .string = "add" } }} }, .says = "\"glp.mode\" twice" },
    .{ .opts = .{ .extra = &.{.{ .key = "general.name", .value = .{ .string = "again" } }} }, .says = "\"general.name\" twice" },
    // the checkpoint it names
    .{ .opts = .{ .edits = &.{number("general.base_model.count", 2)} }, .says = "2 base models" },
    .{ .opts = .{ .edits = &.{number("general.base_model.count", 0)} }, .says = "count is 0 but the file names a base model" },
    .{ .opts = .{ .edits = &.{text("general.base_model.count", "1")} }, .says = "general.base_model.count is not an unsigned integer" },
    .{ .opts = .{ .edits = &.{number("general.base_model.0.name", 5)} }, .says = "general.base_model.0.name is not text" },
    .{ .opts = .{ .edits = &.{text("general.base_model.0.organization", "other-org")} }, .says = "by its URL" },
    .{ .opts = .{ .edits = &.{text("glp.base_model", "example-org/Other-Model")} }, .says = "as glp.base_model" },
    .{ .opts = .{ .edits = &.{text("glp.base_revision", "fedcba9876")} }, .says = "as glp.base_revision" },
    // what the file says about itself is text or it is refused
    .{ .opts = .{ .edits = &.{number("general.name", 1)} }, .says = "general.name is not text" },
    .{ .opts = .{ .edits = &.{number("glp.method", 1)} }, .says = "glp.method is not text" },
    .{ .opts = .{ .edits = &.{number("glp.derived_at", 1)} }, .says = "glp.derived_at is not text" },
    .{ .opts = .{ .edits = &.{number("glp.measured_effect", 1)} }, .says = "glp.measured_effect is not text" },
};

test "a paired file is refused for every metadata, tensor and digest fault, each with its reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (refusals, 0..) |c, i| {
        const why = try reason(a, try build(a, c.opts));
        std.testing.expect(std.mem.indexOf(u8, why, c.says) != null) catch |err| {
            std.debug.print("case {d}: wanted \"{s}\", the reader said \"{s}\"\n", .{ i, c.says, why });
            return err;
        };
    }
}

test "a truncated or tiny paired file is refused, not read past" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try build(a, .{});
    _ = try reason(a, bytes[0 .. bytes.len - 64]);
    _ = try reason(a, bytes[0..4]);
    _ = try reason(a, bytes[0 .. bytes.len / 2]);
}

test "a paired file is accepted with everything it says, and nothing relabelled" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try build(a, .{ .edits = &.{text("glp.created", "2026-11-20")} });
    const v = try readOk(a, bytes);
    try std.testing.expectEqual(@as(u32, 3), v.spec_version);
    try std.testing.expect(v.isPaired());
    try std.testing.expectEqualStrings(glp.paired_site, v.hook_point);
    try std.testing.expectEqualStrings(glp.paired_site, v.derived_at);
    try std.testing.expect(!v.transferred);
    try std.testing.expectEqual(@as(f32, 0.5), v.alpha);
    try std.testing.expectEqual(@as(usize, 8), v.width);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
    try std.testing.expectEqual(@as(u32, 1), v.layers[0].id);
    try std.testing.expectEqual(@as(u32, 2), v.layers[1].id);
    try std.testing.expectEqualStrings(repo, v.base_model);
    try std.testing.expectEqualStrings(commit, v.base_revision);
    try std.testing.expectEqualStrings("synthetic_fixture", v.method);
    try std.testing.expectEqualStrings("synthetic paired fixture", v.publication.name.?);
    try std.testing.expectEqualStrings("2026-11-20", v.publication.created.?);
    try std.testing.expect(v.digest_claimed);
    var file_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &file_sha, .{});
    try std.testing.expectEqualSlices(u8, &file_sha, &v.file_sha256);
    // the file's numbers, exactly: layer 1's first components are 0.25 and 0.3125
    try std.testing.expectEqual(@as(f32, 0.25), v.layers[0].dir[0]);
    try std.testing.expectEqual(@as(f32, 0.3125), v.layers[0].dir[1]);
    try std.testing.expectEqual(@as(f32, 0.5), v.layers[1].dir[0]);
}

test "a layer is looked up by its id, and a layer the file lacks is absent" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try readOk(a, try build(a, .{ .ids = &.{ 3, 7, 12 } }));
    try std.testing.expectEqual(@as(f32, 0.25), v.layer(3).?[0]);
    try std.testing.expectEqual(@as(f32, 0.5), v.layer(7).?[0]);
    try std.testing.expectEqual(@as(f32, 0.75), v.layer(12).?[0]);
    for ([_]u32{ 0, 1, 4, 8, 13, std.math.maxInt(u32) }) |id| try std.testing.expect(v.layer(id) == null);
}

test "a file whose tensors descend is read ascending, the layer list in any order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try build(a, .{ .ids = &.{ 2, 1 }, .edits = &.{text("glp.layer_ids_zero_based", "2,1")} });
    const v = try readOk(a, bytes);
    try std.testing.expectEqual(@as(u32, 1), v.layers[0].id);
    try std.testing.expectEqual(@as(u32, 2), v.layers[1].id);
    // layer 2 came first in the file, so its components are the first tensor's
    try std.testing.expectEqual(@as(f32, 0.25), v.layer(2).?[0]);
}

test "a zero strength and a zero direction are accepted: loaded, not absent, and no-ops" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const off = try readOk(a, try build(a, .{ .edits = &.{real("glp.alpha_default", 0)} }));
    try std.testing.expectEqual(@as(f32, 0), off.alpha);
    // negative zero is the same strength as zero
    const negative = try readOk(a, try build(a, .{ .edits = &.{real("glp.alpha_default", -0.0)} }));
    try std.testing.expect(negative.alpha == 0 and !std.math.signbit(negative.alpha));
    const zero_dir = try readOk(a, try build(a, .{ .value = 0 }));
    for (zero_dir.layers) |l| for (l.dir) |x| try std.testing.expectEqual(@as(f32, 0), x);
    // a loaded file still has its hash: it is a policy of its own, not the absence of one
    try std.testing.expect(!std.mem.allEqual(u8, &off.file_sha256, 0));
}

test "a paired file's strength defaults to 1 and its digest may be absent or any case" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bare = try readOk(a, try build(a, .{ .edits = &.{ drop("glp.alpha_default"), drop("glp.content_sha256"), drop("glp.layer_ids_zero_based") } }));
    try std.testing.expectEqual(@as(f32, 1), bare.alpha);
    try std.testing.expect(!bare.digest_claimed);
    const plain = try readOk(a, try build(a, .{}));
    const claimed = std.fmt.bytesToHex(plain.sha256, .upper);
    const shouted = try readOk(a, try build(a, .{ .edits = &.{.{ .key = "glp.content_sha256", .value = .{ .string = &claimed } }} }));
    try std.testing.expectEqualSlices(u8, &plain.sha256, &shouted.sha256);
}

test "a vector derived elsewhere is read, transferred, and still applied at the paired site" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const v = try readOk(a, try build(a, .{ .edits = &.{text("glp.derived_at", "ffn_out_pre_residual")} }));
    try std.testing.expect(v.transferred);
    try std.testing.expectEqualStrings("ffn_out_pre_residual", v.derived_at);
    try std.testing.expectEqualStrings(glp.paired_site, v.hook_point);
}

test "the base checkpoint comes from GGUF's keys, from glp.base_model alone, or from both when they agree" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const own = try readOk(a, try build(a, .{ .edits = &.{
        drop("general.base_model.count"),
        drop("general.base_model.0.name"),
        drop("general.base_model.0.organization"),
        drop("general.base_model.0.version"),
        drop("general.base_model.0.repo_url"),
        text("glp.base_model", repo),
        text("glp.base_revision", commit),
    } }));
    try std.testing.expectEqualStrings(repo, own.base_model);
    try std.testing.expectEqualStrings(commit, own.base_revision);
    // the hub's URL names the same repo as its id, in any case
    const both = try readOk(a, try build(a, .{ .edits = &.{ text("glp.base_model", "https://huggingface.co/EXAMPLE-ORG/synthetic-hc-model"), text("glp.base_revision", commit) } }));
    try std.testing.expectEqualStrings(repo, both.base_model);
    const none = try readOk(a, try build(a, .{ .edits = &.{
        drop("general.base_model.count"),
        drop("general.base_model.0.name"),
        drop("general.base_model.0.organization"),
        drop("general.base_model.0.version"),
        drop("general.base_model.0.repo_url"),
    } }));
    try std.testing.expectEqualStrings("", none.base_model);
    try std.testing.expectEqualStrings("", none.base_revision);
}

test "a file with no glp keys and a version 1 file are still the residual vector, never the paired one" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try readOk(a, try build(a, .{ .edits = &.{
        drop("glp.spec_version"),
        drop("glp.mode"),
        drop("glp.rank"),
        drop("glp.alpha_default"),
        drop("glp.hook_point"),
        drop("glp.derived_at"),
        drop("glp.method"),
        drop("glp.contrast"),
        drop("glp.layer_ids_zero_based"),
        drop("glp.content_sha256"),
    } }));
    try std.testing.expectEqual(@as(u32, 1), plain.spec_version);
    try std.testing.expect(!plain.isPaired());
    try std.testing.expectEqualStrings(glp.residual_site, plain.hook_point);
    const residual = try readOk(a, try build(a, .{ .edits = &.{ number("glp.spec_version", 1), text("glp.hook_point", glp.residual_site), text("glp.derived_at", glp.residual_site) } }));
    try std.testing.expectEqual(@as(u32, 1), residual.spec_version);
    try std.testing.expect(!residual.isPaired());
}
