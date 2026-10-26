//! The GLP container's acceptance rules, over synthetic containers written the way a producer writes them.

const std = @import("std");
const glp = @import("glp.zig");

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
const Read = struct {
    arena: std.heap.ArenaAllocator,
    vector: glp.Vector,

    fn deinit(self: *Read) void {
        self.arena.deinit();
    }
};

fn readOk(bytes: []const u8) !Read {
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
    const bytes = try writeFixture(.{ .layers = &.{0, 11} });
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
