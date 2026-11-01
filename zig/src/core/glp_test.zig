//! The GLP container's acceptance rules, over synthetic containers written the way a producer writes them.

const std = @import("std");
const glp = @import("glp.zig");
const gguf = @import("gguf.zig");

/// What a container holds, so a test can bend one rule at a time.
const Shape = struct {
    mode: ?[]const u8 = "project",
    spec_version: ?u32 = 1,
    rank: ?u32 = 1,
    hook_point: ?[]const u8 = "residual_stream_post_layer",
    derived_at: ?[]const u8 = null,
    alpha: f32 = 1.0,
    layers: []const u32 = &.{ 10, 11 },
    width: usize = 8,
    unit: bool = true,
    digest: enum { correct, wrong, absent } = .correct,
    base_model: ?[]const u8 = "Qwen/Qwen3.8-27B",
    layer_ids_key: bool = true,
    layer_ids: ?[]const u8 = null,
};

/// The metadata a fixture writes: the key's bytes and how many keys there are.
const Kv = struct {
    bytes: std.ArrayList(u8) = .empty,
    count: u64 = 0,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator) Kv {
        return .{ .a = a };
    }

    fn deinit(self: *Kv) void {
        self.bytes.deinit(self.a);
    }

    fn key(self: *Kv, name: []const u8) !void {
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, name.len)));
        try self.bytes.appendSlice(self.a, name);
        self.count += 1;
    }

    fn text(self: *Kv, name: []const u8, value: []const u8) !void {
        try self.key(name);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, 8)));
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u64, value.len)));
        try self.bytes.appendSlice(self.a, value);
    }

    fn number(self: *Kv, name: []const u8, value: u32) !void {
        try self.key(name);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, 4)));
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(value));
    }

    fn real(self: *Kv, name: []const u8, value: f32) !void {
        try self.key(name);
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, 6)));
        try self.bytes.appendSlice(self.a, &std.mem.toBytes(@as(u32, @bitCast(value))));
    }
};

/// Write a container holding one direction per listed layer.
fn writeFixture(shape: Shape) ![]u8 {
    const a = std.testing.allocator;
    var payload: std.ArrayList(u8) = .empty;
    defer payload.deinit(a);
    var offsets: std.ArrayList(u64) = .empty;
    defer offsets.deinit(a);
    for (shape.layers) |_| {
        try offsets.append(a, payload.items.len);
        for (0..shape.width) |_| {
            const v: f32 = if (shape.unit) 1.0 / @sqrt(@as(f32, @floatFromInt(shape.width))) else 0.25;
            try payload.appendSlice(a, &std.mem.toBytes(@as(u32, @bitCast(v))));
        }
    }

    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    h.update(payload.items);
    var sha: [32]u8 = undefined;
    h.final(&sha);
    const hex = std.fmt.bytesToHex(sha, .lower);

    var kv: Kv = Kv.init(a);
    defer kv.deinit();
    try kv.text("general.architecture", "controlvector");
    try kv.text("general.alignment", "32");
    if (shape.mode) |m| try kv.text("glp.mode", m);
    if (shape.spec_version) |v| try kv.number("glp.spec_version", v);
    if (shape.rank) |v| try kv.number("glp.rank", v);
    try kv.real("glp.alpha_default", shape.alpha);
    if (shape.hook_point) |h2| try kv.text("glp.hook_point", h2);
    if (shape.derived_at) |d| try kv.text("glp.derived_at", d);
    if (shape.base_model) |b| try kv.text("glp.base_model", b);
    if (shape.layer_ids_key) {
        if (shape.layer_ids) |ids| {
            try kv.text("glp.layer_ids_zero_based", ids);
        } else {
            var list: std.ArrayList(u8) = .empty;
            defer list.deinit(a);
            for (shape.layers, 0..) |id, i| {
                if (i > 0) try list.appendSlice(a, ",");
                try list.print(a, "{d}", .{id});
            }
            try kv.text("glp.layer_ids_zero_based", list.items);
        }
    }
    switch (shape.digest) {
        .correct => try kv.text("glp.content_sha256", &hex),
        .wrong => try kv.text("glp.content_sha256", "0000000000000000000000000000000000000000000000000000000000000000"),
        .absent => {},
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "GGUF");
    try out.appendSlice(a, &std.mem.toBytes(@as(u32, 3)));
    try out.appendSlice(a, &std.mem.toBytes(@as(u64, shape.layers.len)));
    try out.appendSlice(a, &std.mem.toBytes(kv.count));
    try out.appendSlice(a, kv.bytes.items);
    for (shape.layers, offsets.items) |id, off| {
        var name_buf: [32]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buf, "direction.{d}", .{id});
        try out.appendSlice(a, &std.mem.toBytes(@as(u64, name.len)));
        try out.appendSlice(a, name);
        try out.appendSlice(a, &std.mem.toBytes(@as(u32, 1)));
        try out.appendSlice(a, &std.mem.toBytes(@as(u64, shape.width)));
        try out.appendSlice(a, &std.mem.toBytes(@as(u32, 0)));
        try out.appendSlice(a, &std.mem.toBytes(off));
    }
    while (out.items.len % 32 != 0) try out.append(a, 0);
    try out.appendSlice(a, payload.items);
    return out.toOwnedSlice(a);
}

/// A vector and the arena that holds its strings and layers.
pub const Read = struct {
    arena: std.heap.ArenaAllocator,
    vector: glp.Vector,

    pub fn deinit(self: *Read) void {
        self.arena.deinit();
    }
};

pub fn readOk(bytes: []const u8) !Read {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    errdefer arena.deinit();
    var why: glp.Why = .{};
    const v = try glp.read(arena.allocator(), bytes, &why);
    return .{ .arena = arena, .vector = v };
}

/// Fail unless the file was refused.
fn refused(bytes: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: glp.Why = .{};
    if (glp.read(arena.allocator(), bytes, &why)) |_| {
        return error.NotRefused;
    } else |err| {
        if (err != error.BadGlpVector) return err;
    }
}

/// The reason a file was refused, copied out so it outlives the arena.
fn reason(bytes: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var why: glp.Why = .{};
    if (glp.read(arena.allocator(), bytes, &why)) |_| {
        return error.NotRefused;
    } else |err| {
        if (err != error.BadGlpVector) return err;
        return std.testing.allocator.dupe(u8, why.text());
    }
}

/// A unit vector of a given width, pointing along one axis.
pub fn unitDir(a: std.mem.Allocator, width: usize, axis: usize) ![]f32 {
    const dir = try a.alloc(f32, width);
    for (dir) |*v| v.* = 0.0;
    dir[axis] = 1.0;
    return dir;
}

/// A container and the arena that holds its metadata.
pub const Parsed = struct {
    arena: std.heap.ArenaAllocator,
    file: gguf.File,

    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }
};

/// The container a file holds, read with the generic reader.
pub fn parseFile(a: std.mem.Allocator, bytes: []const u8) !Parsed {
    var arena = std.heap.ArenaAllocator.init(a);
    errdefer arena.deinit();
    var problem: gguf.Problem = .{};
    const file = try gguf.parse(arena.allocator(), bytes, &problem);
    return .{ .arena = arena, .file = file };
}

/// The vector a producer writes for two layers, handed over in the order a caller would hold them.
fn writeSample(a: std.mem.Allocator) ![]u8 {
    const dir10 = try unitDir(a, 8, 0);
    defer a.free(dir10);
    const dir11 = try unitDir(a, 8, 1);
    defer a.free(dir11);
    const layers = [_]glp.Layer{ .{ .id = 11, .dir = dir11 }, .{ .id = 10, .dir = dir10 } };
    return glp.write(a, .{
        .mode = .project,
        .alpha = 0.75,
        .hook_point = glp.residual_site,
        .derived_at = null,
        .base_model = "Qwen/Qwen3.8-27B",
        .base_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        .method = "difference_of_means_per_layer",
        .contrast = "diverse_harmful_vs_form_matched_harmless_unpaired",
        .layers = &layers,
    });
}

test "a projected vector with the metadata a producer writes is accepted" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    try std.testing.expectEqual(glp.Mode.project, v.mode);
    try std.testing.expectEqual(@as(f32, 1.0), v.alpha);
    try std.testing.expectEqual(@as(usize, 8), v.width);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
    try std.testing.expectEqual(@as(u32, 10), v.layers[0].id);
    try std.testing.expectEqual(@as(u32, 11), v.layers[1].id);
    try std.testing.expectEqualStrings("residual_stream_post_layer", v.hook_point);
    try std.testing.expectEqualStrings("Qwen/Qwen3.8-27B", v.base_model);
    try std.testing.expect(!v.transferred);
}

test "a file with no glp keys is llama.cpp's additive control vector" {
    const bytes = try writeFixture(.{ .mode = null, .spec_version = null, .rank = null, .hook_point = null, .layer_ids_key = false, .digest = .absent });
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    try std.testing.expectEqual(glp.Mode.add, v.mode);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
}

test "an unknown mode is fatal" {
    const bytes = try writeFixture(.{ .mode = "scale" });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a spec version this reader does not implement is refused" {
    const bytes = try writeFixture(.{ .spec_version = 2 });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a rank above one is refused" {
    const bytes = try writeFixture(.{ .rank = 2 });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a hook point this reader does not apply at is refused" {
    const bytes = try writeFixture(.{ .hook_point = "ffn_out_pre_residual" });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "layer zero is not steerable" {
    const bytes = try writeFixture(.{ .layers = &.{ 0, 11 } });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a direction that is not a unit vector is refused" {
    const bytes = try writeFixture(.{ .unit = false });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a content digest that does not cover the vectors is refused" {
    const bytes = try writeFixture(.{ .digest = .wrong });
    defer std.testing.allocator.free(bytes);
    try refused(bytes);
}

test "a layer id list that disagrees with the tensors is refused" {
    const bytes = try writeFixture(.{ .layers = &.{ 10, 11 }, .layer_ids = "9,10" });
    defer std.testing.allocator.free(bytes);
    const why = try reason(bytes);
    defer std.testing.allocator.free(why);
    try std.testing.expect(std.mem.indexOf(u8, why, "layer id") != null);
}

test "a vector derived at another site is a warning, not a refusal" {
    const bytes = try writeFixture(.{ .derived_at = "ffn_out_pre_residual" });
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    try std.testing.expect(v.transferred);
}

test "a vector whose width is not the model's is refused" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    var why: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, 4096, "Qwen/Qwen3.8-27B", &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "wide") != null);
}

test "a vector for another checkpoint is refused" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    var why: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, 8, "Qwen/Qwen3.5-2B", &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "the model is") != null);
}

test "a truncated file is refused, not read past" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    const cut = try std.testing.allocator.dupe(u8, bytes[0 .. bytes.len - 64]);
    defer std.testing.allocator.free(cut);
    try refused(cut);
}

test "a written vector reads back with every field intact" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    try std.testing.expectEqual(glp.Mode.project, v.mode);
    try std.testing.expectEqual(@as(f32, 0.75), v.alpha);
    try std.testing.expectEqualStrings("residual_stream_post_layer", v.hook_point);
    try std.testing.expectEqualStrings("residual_stream_post_layer", v.derived_at);
    try std.testing.expectEqualStrings("Qwen/Qwen3.8-27B", v.base_model);
    try std.testing.expectEqualStrings("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", v.base_revision);
    try std.testing.expectEqualStrings("difference_of_means_per_layer", v.method);
    try std.testing.expectEqualStrings("diverse_harmful_vs_form_matched_harmless_unpaired", v.contrast);
    try std.testing.expectEqual(@as(usize, 8), v.width);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
    try std.testing.expectEqual(@as(u32, 10), v.layers[0].id);
    try std.testing.expectEqual(@as(u32, 11), v.layers[1].id);
    try std.testing.expect(!v.transferred);
    try std.testing.expect(!std.mem.allEqual(u8, &v.sha256, 0));
}

test "a written vector's digest covers its tensor bytes" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var parsed = try parseFile(a, bytes);
    defer parsed.deinit();
    const claimed = parsed.file.str("glp.content_sha256").?;
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (parsed.file.tensors) |t| h.update(parsed.file.bytesOf(t).?);
    var sha: [32]u8 = undefined;
    h.final(&sha);
    const hex = std.fmt.bytesToHex(sha, .lower);
    try std.testing.expectEqualStrings(hex[0..], claimed);
}

test "a written vector is accepted by the reader's own rules" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    _ = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(@as(usize, 0), why.len);
}

test "a written vector holds one direction tensor per layer" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var parsed = try parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.file.tensors.len);
    try std.testing.expectEqualStrings("direction.10", parsed.file.tensors[0].name);
    try std.testing.expectEqualStrings("direction.11", parsed.file.tensors[1].name);
    try std.testing.expectEqual(@as(u64, 8), parsed.file.tensors[0].dims[0]);
    try std.testing.expectEqual(@as(u32, gguf.f32_type), parsed.file.tensors[0].kind);
    try std.testing.expectEqualStrings("10,11", parsed.file.str("glp.layer_ids_zero_based").?);
    const first = parsed.file.bytesOf(parsed.file.tensors[0]).?;
    const second = parsed.file.bytesOf(parsed.file.tensors[1]).?;
    try std.testing.expectEqual(@as(usize, 32), first.len);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.0))), std.mem.readInt(u32, first[0..4], .little));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.0))), std.mem.readInt(u32, first[4..8], .little));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 0.0))), std.mem.readInt(u32, second[0..4], .little));
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.0))), std.mem.readInt(u32, second[4..8], .little));
}

test "a written vector derived at another site is transferred" {
    const a = std.testing.allocator;
    const dir = try unitDir(a, 4, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .hook_point = glp.residual_site, .derived_at = "ffn_out_pre_residual", .layers = &layers });
    defer a.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    try std.testing.expect(v.transferred);
    try std.testing.expectEqualStrings("ffn_out_pre_residual", v.derived_at);
    try std.testing.expectEqualStrings("residual_stream_post_layer", v.hook_point);
}

test "a vector written again keeps its digest and its layers" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const again = try glp.writeVector(a, read.vector);
    defer a.free(again);
    var reread = try readOk(again);
    defer reread.deinit();
    try std.testing.expectEqualSlices(u8, &read.vector.sha256, &reread.vector.sha256);
    try std.testing.expectEqual(read.vector.layers.len, reread.vector.layers.len);
    try std.testing.expectEqual(@as(u32, 10), reread.vector.layers[0].id);
}

test "a vector re-stamped for another checkpoint keeps its directions" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    // The same weights under another name: the directions stay, the checkpoint they name changes.
    const again = try glp.write(a, .{
        .mode = read.vector.mode,
        .alpha = read.vector.alpha,
        .hook_point = read.vector.hook_point,
        .derived_at = read.vector.derived_at,
        .base_model = "mlx-community/Qwen3.8-27B-4bit",
        .base_revision = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
        .method = read.vector.method,
        .contrast = read.vector.contrast,
        .publication = read.vector.publication,
        .layers = read.vector.layers,
    });
    defer a.free(again);
    var reread = try readOk(again);
    defer reread.deinit();
    try std.testing.expectEqualStrings("mlx-community/Qwen3.8-27B-4bit", reread.vector.base_model);
    try std.testing.expectEqualStrings("10c35caafbb80f7dc6a7a432cdd11af10a6d4818", reread.vector.base_revision);
    // The numbers and the digest over them are what they were: only the name moved.
    try std.testing.expectEqualSlices(u8, &read.vector.sha256, &reread.vector.sha256);
    try std.testing.expectEqual(read.vector.layers.len, reread.vector.layers.len);
    for (read.vector.layers, reread.vector.layers) |before, after| {
        try std.testing.expectEqual(before.id, after.id);
        try std.testing.expectEqualSlices(f32, before.dir, after.dir);
    }
    // The binding moved with it: the new checkpoint takes the vector, the old one does not.
    var why: glp.Why = .{};
    try glp.checkModel(reread.vector, 8, "mlx-community/Qwen3.8-27B-4bit", &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(reread.vector, 8, "Qwen/Qwen3.8-27B", &why));
}

test "a producer refuses a direction that is not a unit vector" {
    const a = std.testing.allocator;
    const dir = try a.alloc(f32, 4);
    defer a.free(dir);
    @memset(dir, 0.25);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .layers = &layers }));
}

test "a producer refuses a layer the reader cannot steer" {
    const a = std.testing.allocator;
    const dir = try unitDir(a, 4, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 0, .dir = dir }};
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .layers = &layers }));
}

test "a written container is a version 3 file" {
    const a = std.testing.allocator;
    const bytes = try writeSample(a);
    defer a.free(bytes);
    try std.testing.expectEqualStrings("GGUF", bytes[0..4]);
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, bytes[4..8], .little));
    try std.testing.expectEqual(@as(u64, 2), std.mem.readInt(u64, bytes[8..16], .little));
    try std.testing.expectEqual(@as(u64, 21), std.mem.readInt(u64, bytes[16..24], .little));
    try std.testing.expectEqual(@as(usize, 0), bytes.len % 32);
}

test "the container keeps every value type it is given" {
    const a = std.testing.allocator;
    const entries = [_]gguf.Entry{
        .{ .key = "a.u8", .value = .{ .u8v = 7 } },
        .{ .key = "a.i8", .value = .{ .i8v = -7 } },
        .{ .key = "a.u16", .value = .{ .u16v = 1000 } },
        .{ .key = "a.i16", .value = .{ .i16v = -1000 } },
        .{ .key = "a.u32", .value = .{ .u32v = 100000 } },
        .{ .key = "a.i32", .value = .{ .i32v = -100000 } },
        .{ .key = "a.f32", .value = .{ .f32v = 0.5 } },
        .{ .key = "a.bool", .value = .{ .boolv = true } },
        .{ .key = "a.string", .value = .{ .string = "hello" } },
        .{ .key = "a.array", .value = .{ .array = .{ .elem = .u32v, .items = &.{ .{ .u32v = 1 }, .{ .u32v = 2 } } } } },
        .{ .key = "a.u64", .value = .{ .u64v = 1 << 40 } },
        .{ .key = "a.i64", .value = .{ .i64v = -(1 << 40) } },
        .{ .key = "a.f64", .value = .{ .f64v = 0.25 } },
    };
    const bytes = try gguf.write(a, &entries, &.{});
    defer a.free(bytes);
    var parsed = try parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 13), parsed.file.entries.len);
    try std.testing.expectEqual(@as(u8, 7), parsed.file.find("a.u8").?.u8v);
    try std.testing.expectEqual(@as(i8, -7), parsed.file.find("a.i8").?.i8v);
    try std.testing.expectEqual(@as(u16, 1000), parsed.file.find("a.u16").?.u16v);
    try std.testing.expectEqual(@as(i16, -1000), parsed.file.find("a.i16").?.i16v);
    try std.testing.expectEqual(@as(u32, 100000), parsed.file.find("a.u32").?.u32v);
    try std.testing.expectEqual(@as(i32, -100000), parsed.file.find("a.i32").?.i32v);
    try std.testing.expectEqual(@as(f32, 0.5), parsed.file.find("a.f32").?.f32v);
    try std.testing.expectEqual(true, parsed.file.find("a.bool").?.boolv);
    try std.testing.expectEqualStrings("hello", parsed.file.find("a.string").?.string);
    try std.testing.expectEqual(@as(usize, 2), parsed.file.find("a.array").?.array.items.len);
    try std.testing.expectEqual(@as(u32, 2), parsed.file.find("a.array").?.array.items[1].u32v);
    try std.testing.expectEqual(@as(u64, 1 << 40), parsed.file.find("a.u64").?.u64v);
    try std.testing.expectEqual(@as(i64, -(1 << 40)), parsed.file.find("a.i64").?.i64v);
    try std.testing.expectEqual(@as(f64, 0.25), parsed.file.find("a.f64").?.f64v);
}

test "a conversion of the vector's model takes the vector" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    var why: glp.Why = .{};
    try glp.checkModel(v, 8, "TensorFold/Qwen3.8-27B-MLX-4bit", &why);
    try glp.checkModel(v, 8, "Qwen/Qwen3.8-27B", &why);
}

test "a different model of the same size is refused" {
    const bytes = try writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    var read = try readOk(bytes);
    defer read.deinit();
    const v = read.vector;
    var why: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, 8, "Qwen/Qwen3.5-2B", &why));
}
