//! Checkpoint mutations through pack admission and the real persisted-state namespace, without a GPU.
const std = @import("std");
const t = std.testing;
const pio = @import("src/families/flashnext/pack_io.zig");
const ckpt = @import("src/core/checkpoint.zig");
const fixture = @import("src/core/checkpoint_fixture.zig");
const disk = @import("src/native/disk_identity.zig");
const Imprint = @import("engine_api").prompt_imprint.Imprint;
const tfq3 = @import("src/families/glm/tfq3_check.zig");

const original_index = "{\"weight_map\":{\"a\":\"one.safetensors\",\"b\":\"one.safetensors\",\"c\":\"two.safetensors\",\"d\":\"two.safetensors\"}}";
const changed_index = "{\"weight_map\":{\"a\":\"two.safetensors\",\"b\":\"one.safetensors\",\"c\":\"two.safetensors\",\"d\":\"two.safetensors\"}}";

const Case = struct {
    tmp: t.TmpDir,
    dir: []const u8,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator) !Case {
        var tmp = t.tmpDir(.{});
        errdefer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
        try fixture.write(a, t.io, tmp.dir, "one.safetensors", &.{
            .{ .name = "a", .shape = &.{4}, .bytes = "abcd" },
            .{ .name = "b", .shape = &.{4}, .bytes = "efgh" },
        });
        try fixture.write(a, t.io, tmp.dir, "two.safetensors", &.{
            .{ .name = "c", .shape = &.{4}, .bytes = "ijkl" },
            .{ .name = "d", .shape = &.{4}, .bytes = "mnop" },
        });
        var c: Case = .{ .tmp = tmp, .dir = dir, .a = a };
        try c.write("config.json", "{\"hidden_size\":4,\"torch_dtype\":\"uint8\"}");
        try c.write("tokenizer.json", "{\"model\":\"first\"}");
        try c.write("model.safetensors.index.json", original_index);
        return c;
    }

    fn path(c: Case, name: []const u8) ![]const u8 {
        return std.fs.path.join(c.a, &.{ c.dir, name });
    }
    fn write(c: Case, name: []const u8, bytes: []const u8) !void {
        try c.tmp.dir.writeFile(t.io, .{ .sub_path = name, .data = bytes });
    }
    fn read(c: Case, name: []const u8) ![]u8 {
        return c.tmp.dir.readFileAlloc(t.io, name, c.a, .limited(1 << 20));
    }
    fn mutate(c: Case) !void {
        const bytes = try c.read("one.safetensors");
        bytes[bytes.len - 1] ^= 1;
        try c.write("one.safetensors", bytes);
    }
    fn swap(c: Case) !void {
        const one = try c.read("one.safetensors");
        const two = try c.read("two.safetensors");
        try c.write("one.safetensors", two);
        try c.write("two.safetensors", one);
    }
    fn pack(c: Case) !void {
        const source = try pio.sourceIdentity(c.a, t.io, c.dir);
        var out = pio.Out.init(c.a);
        defer out.deinit();
        try out.put("packed", "U8", &.{4}, try c.a.dupe(u8, "pack"));
        try out.write(c.a, t.io, try c.path("pack.safetensors"), source);
        try pio.checkSource(c.a, t.io, try c.path("pack.safetensors"), c.dir);
    }
    fn refusePack(c: Case) !void {
        const path_ = try c.path("pack.safetensors");
        try t.expectError(error.PackSourceMismatch, pio.checkSource(c.a, t.io, path_, c.dir));
        const source = try pio.sourceIdentity(c.a, t.io, c.dir);
        try t.expectError(error.PackSourceMismatch, pio.checkIdentityPath(c.a, t.io, path_, source));
        try t.expectError(error.PackSourceMismatch, pio.checkSourceMapped(c.a, try c.read("pack.safetensors"), source, "pack"));
    }
    fn namespace(c: Case) !u64 {
        var h = std.hash.Wyhash.init(0);
        try disk.checkpoint(&h, c.a, t.io, c.dir);
        return h.final();
    }
    fn tfq3Files(c: Case) !void {
        const bytes = try c.a.alloc(u8, 32768 + 2048 * 4);
        @memset(bytes, 0);
        var h = std.mem.zeroes(tfq3.NonMoeHeader);
        h.magic = tfq3.magic_nonmoe;
        h.version = 3;
        h.num_tensors = 1;
        h.off_meta = 128;
        h.off_scales = 16384;
        h.off_weights = 32768;
        h.total_scale_halfs = 1;
        h.total_weight_words = 2048;
        var m = std.mem.zeroes(tfq3.NonMoeTensorMeta);
        @memcpy(m.name[0..7], "lm_head");
        m.R = 128;
        m.C = 128;
        m.b_bits = 4;
        m.word_count = 2048;
        @memcpy(bytes[0..@sizeOf(@TypeOf(h))], std.mem.asBytes(&h));
        @memcpy(bytes[128..][0..@sizeOf(@TypeOf(m))], std.mem.asBytes(&m));
        _ = try tfq3.nonMoe(bytes);
        try c.write("nonmoe.tffpw", bytes);
        const tables = try c.a.alloc(u8, 16384);
        @memset(tables, 0);
        std.mem.writeInt(u32, tables[0..4], tfq3.magic_v3_tables, .little);
        std.mem.writeInt(u32, tables[4..8], 3, .little);
        try tfq3.tables(tables);
        try c.write("pw_v3_tables.bin", tables);
    }
};

test "artifact identity: swapped shards refuse the old pack" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    try c.pack();
    try c.swap();
    try c.refusePack();
}

test "artifact identity: admission: index retarget refuses the old pack and inconsistent raw checkpoint" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    try c.pack();
    try c.write("model.safetensors.index.json", changed_index);
    try c.refusePack();
    if (ckpt.Checkpoint.openModel(c.a, t.io, c.dir)) |opened| {
        var ck = opened;
        defer ck.close();
        // Demonstrate that the old loader finds a's bytes in a shard the index no longer assigns to it.
        try t.expectEqualStrings("abcd", (try ck.get("a")).bytes);
        return error.TestExpectedIndexRefusal;
    } else |err| try t.expectEqual(error.IndexTensorShardMismatch, err);
}

test "artifact identity: shape and dtype config changes refuse the old pack" {
    for ([_][]const u8{ "{\"hidden_size\":8,\"torch_dtype\":\"uint8\"}", "{\"hidden_size\":4,\"torch_dtype\":\"float16\"}" }) |config| {
        var arena = std.heap.ArenaAllocator.init(t.allocator);
        defer arena.deinit();
        var c = try Case.init(arena.allocator());
        defer c.tmp.cleanup();
        try c.pack();
        try c.write("config.json", config);
        try c.refusePack();
    }
}

test "artifact identity: admission: an unmarked supplied pack is refused by every admission entry" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    const bytes = try fixture.image(c.a, &.{.{ .name = "packed", .shape = &.{4}, .bytes = "pack" }});
    try c.write("pack.safetensors", bytes);
    const source = try pio.sourceIdentity(c.a, t.io, c.dir);
    const path_ = try c.path("pack.safetensors");
    try t.expectError(error.PackHasNoSource, pio.checkSource(c.a, t.io, path_, c.dir));
    try t.expectError(error.PackHasNoSource, pio.checkIdentityPath(c.a, t.io, path_, source));
    try t.expectError(error.PackHasNoSource, pio.checkSourceMapped(c.a, bytes, source, "pack"));
}

const Mutation = enum { payload, swap, index, shape, dtype, tokenizer, tfq3, tables };

fn restart(comptime mutation: Mutation) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Case.init(a);
    defer c.tmp.cleanup();
    try c.tfq3Files();
    const before = try c.namespace();
    const root = try c.path("states");
    const tokens = &.{ 1, 2, 3 };
    const key = Imprint.keyOf(tokens);
    {
        var store = try Imprint.open(a, root, before, 1 << 20);
        defer store.deinit();
        try store.add(key, 2, tokens, &.{}, 8);
    }
    {
        var same = try Imprint.open(a, root, try c.namespace(), 1 << 20);
        defer same.deinit();
        try t.expect(same.has(key));
    }
    switch (mutation) {
        .payload => try c.mutate(),
        .swap => try c.swap(),
        .index => try c.write("model.safetensors.index.json", changed_index),
        .shape => try c.write("config.json", "{\"hidden_size\":8,\"torch_dtype\":\"uint8\"}"),
        .dtype => try c.write("config.json", "{\"hidden_size\":4,\"torch_dtype\":\"float16\"}"),
        .tokenizer => try c.write("tokenizer.json", "{\"model\":\"other\"}"),
        .tfq3 => {
            const bytes = try c.read("nonmoe.tffpw");
            bytes[bytes.len - 1] ^= 1;
            _ = try tfq3.nonMoe(bytes);
            try c.write("nonmoe.tffpw", bytes);
        },
        .tables => {
            const bytes = try c.read("pw_v3_tables.bin");
            bytes[8] ^= 1; // A finite coefficient changes without damaging the table's format.
            try tfq3.tables(bytes);
            try c.write("pw_v3_tables.bin", bytes);
        },
    }
    var stale = try Imprint.open(a, root, try c.namespace(), 1 << 20);
    defer stale.deinit();
    if (stale.has(key)) std.debug.print("stale state accepted after {s}\n", .{@tagName(mutation)});
    try t.expect(!stale.has(key));
    try t.expect(stale.best(tokens, &.{}, false, 0) == null);
}

test "artifact identity: restart refuses state from changed checkpoint bytes or tokenizer" {
    inline for (.{ Mutation.payload, Mutation.swap, Mutation.index, Mutation.shape, Mutation.dtype, Mutation.tokenizer }) |mutation| try restart(mutation);
}
test "artifact identity: restart refuses state from changed TFQ3 tables" {
    try t.expectEqual(tfq3.tables_bytes, disk.tfq3_table_bytes);
    try restart(.tables);
}

test "artifact identity: orphan tables do not affect a checkpoint with no TFQ3 records" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    const before = try c.namespace();
    try c.write("pw_v3_tables.bin", "an orphan table is not loaded");
    try t.expectEqual(before, try c.namespace());
}

test "artifact identity: admission: unchanged tensors and selected namespaces stay byte-identical" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    {
        var ck = try ckpt.Checkpoint.openModel(c.a, t.io, c.dir);
        defer ck.close();
        for ([_][]const u8{ "a", "b", "c", "d" }, [_][]const u8{ "abcd", "efgh", "ijkl", "mnop" }) |name, bytes|
            try t.expectEqualStrings(bytes, (try ck.get(name)).bytes);
    }
    try c.write("model.safetensors.index.json", changed_index);
    var selected = try ckpt.Checkpoint.openModelPrefix(c.a, t.io, c.dir, "c");
    defer selected.close();
    try t.expectEqualStrings("ijkl", (try selected.get("c")).bytes);
}

test "artifact identity: restart against another checkpoint never sees the old state's token key" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var first = try Case.init(a);
    defer first.tmp.cleanup();
    var second = try Case.init(a);
    defer second.tmp.cleanup();
    try second.mutate();
    const root = try first.path("states");
    const tokens = &.{ 1, 2, 3 };
    const key = Imprint.keyOf(tokens);
    {
        var saved = try Imprint.open(a, root, try first.namespace(), 1 << 20);
        defer saved.deinit();
        try saved.add(key, 2, tokens, &.{}, 8);
    }
    var stale = try Imprint.open(a, root, try second.namespace(), 1 << 20);
    defer stale.deinit();
    try t.expect(!stale.has(key));
    try t.expect(stale.best(tokens, &.{}, false, 0) == null);
}
