//! Checkpoint mutations through pack admission and the real persisted-state namespace, without a GPU.
const std = @import("std");
const t = std.testing;
const pio = @import("src/families/flashnext/pack_io.zig");
const ckpt = @import("src/core/checkpoint.zig");
const fixture = @import("src/core/checkpoint_fixture.zig");
const disk = @import("src/native/disk_identity.zig");
pub const digest_cache = @import("src/core/digest_cache.zig");
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
        const source = try pio.identityWithContent(c.a, t.io, c.dir, try c.path("digests"), true);
        var out = pio.Out.init(c.a);
        defer out.deinit();
        try out.put("packed", "U8", &.{4}, try c.a.dupe(u8, "pack"));
        try out.write(c.a, t.io, try c.path("pack.safetensors"), source);
        try pio.checkSourceWithCache(c.a, t.io, try c.path("pack.safetensors"), c.dir, try c.path("digests"));
    }
    fn refusePack(c: Case) !void {
        const path_ = try c.path("pack.safetensors");
        const root = try c.path("digests");
        const current = pio.identityWithContent(c.a, t.io, c.dir, root, false) catch |err| blk: {
            try t.expectEqual(error.DigestFileChanged, err);
            // A separate verification of the changed bytes still must not admit the old converted pack.
            break :blk try pio.identityWithContent(c.a, t.io, c.dir, root, true);
        };
        try t.expectError(error.PackSourceMismatch, pio.checkSourceWithCache(c.a, t.io, path_, c.dir, root));
        try t.expectError(error.PackSourceMismatch, pio.checkIdentityPath(c.a, t.io, path_, current, c.dir));
        try t.expectError(error.PackSourceMismatch, pio.checkSourceMapped(c.a, t.io, try c.read("pack.safetensors"), current, "pack", c.dir));
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

test "artifact identity: same-size payload edit refuses the old converted pack" {
    if (!@import("src/core/digest_key.zig").persistent) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    try c.pack();
    try c.mutate();
    try c.refusePack();
}

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
    try t.expectError(error.PackHasNoSource, pio.checkIdentityPath(c.a, t.io, path_, source, c.dir));
    try t.expectError(error.PackHasNoSource, pio.checkSourceMapped(c.a, t.io, bytes, source, "pack", c.dir));
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
test "artifact identity: restart refuses state from a changed TFQ3 shard" {
    try restart(.tfq3);
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

test "artifact identity: directory spellings preserve full and selected index bindings" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Case.init(a);
    defer c.tmp.cleanup();
    const cwd = try std.Io.Dir.cwd().realPathFileAlloc(t.io, ".", a);
    const relative = try std.fs.path.relativeAlloc(a, cwd, null, cwd, c.dir);
    try c.tmp.dir.symLink(t.io, c.dir, "linked", .{ .is_directory = true });
    const linked = try c.path("linked");
    const paths = [_][]const u8{
        c.dir,
        relative,
        try std.fmt.allocPrint(a, "./{s}", .{relative}),
        try std.fmt.allocPrint(a, "{s}/", .{c.dir}),
        try std.fmt.allocPrint(a, "{s}//", .{c.dir}),
        try std.fmt.allocPrint(a, "{s}/", .{relative}),
        linked,
        try std.fmt.allocPrint(a, "{s}/", .{linked}),
    };
    for (paths) |path_| try directoryBindings(c, path_);
    if (@import("builtin").os.tag == .windows)
        try directoryBindings(c, try std.fmt.allocPrint(a, "{s}\\", .{c.dir}));
}

fn directoryBindings(c: Case, dir: []const u8) !void {
    try c.write("model.safetensors.index.json", original_index);
    {
        var ck = try ckpt.Checkpoint.openModel(c.a, t.io, dir);
        defer ck.close();
        for ([_][]const u8{ "a", "b", "c", "d" }, [_][]const u8{ "abcd", "efgh", "ijkl", "mnop" }) |name, bytes|
            try t.expectEqualStrings(bytes, (try ck.get(name)).bytes);
    }
    {
        var ck = try ckpt.Checkpoint.openModelPrefix(c.a, t.io, dir, "c");
        defer ck.close();
        try t.expectEqualStrings("ijkl", (try ck.get("c")).bytes);
    }
    try c.write("model.safetensors.index.json", changed_index);
    try t.expectError(error.IndexTensorShardMismatch, ckpt.Checkpoint.openModel(c.a, t.io, dir));
    try t.expectError(error.IndexTensorShardMismatch, ckpt.Checkpoint.openModelPrefix(c.a, t.io, dir, "a"));
    var selected = try ckpt.Checkpoint.openModelPrefix(c.a, t.io, dir, "c");
    defer selected.close();
    try t.expectEqualStrings("ijkl", (try selected.get("c")).bytes);
}

test "artifact identity: nested shard names preserve both directions of selected index bindings" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var c = try Case.init(arena.allocator());
    defer c.tmp.cleanup();
    try c.tmp.dir.createDir(t.io, "nested", .default_dir);
    try c.write("nested/one.safetensors", try c.read("one.safetensors"));
    const index = "{\"weight_map\":{\"a\":\"nested/one.safetensors\",\"b\":\"nested/one.safetensors\",\"c\":\"two.safetensors\",\"d\":\"two.safetensors\"}}";
    try c.write("model.safetensors.index.json", index);
    const dir = try std.fmt.allocPrint(c.a, "{s}/", .{c.dir});
    {
        var ck = try ckpt.Checkpoint.openModel(c.a, t.io, dir);
        defer ck.close();
        try t.expectEqualStrings("abcd", (try ck.get("a")).bytes);
    }
    {
        var ck = try ckpt.Checkpoint.openModelPrefix(c.a, t.io, dir, "a");
        defer ck.close();
        try t.expectEqualStrings("abcd", (try ck.get("a")).bytes);
    }
    try c.write("nested/one.safetensors", try fixture.image(c.a, &.{
        .{ .name = "a", .shape = &.{4}, .bytes = "abcd" },
        .{ .name = "b", .shape = &.{4}, .bytes = "efgh" },
        .{ .name = "extra", .shape = &.{4}, .bytes = "qrst" },
    }));
    try t.expectError(error.IndexTensorShardMismatch, ckpt.Checkpoint.openModel(c.a, t.io, dir));
    var selected = try ckpt.Checkpoint.openModelPrefix(c.a, t.io, dir, "a");
    defer selected.close();
    try t.expectEqualStrings("abcd", (try selected.get("a")).bytes);
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

test "artifact identity: legacy pack advice names the file and an executable quoted rebuild command" {
    var message = std.Io.Writer.Allocating.init(t.allocator);
    defer message.deinit();
    try pio.sourceRefusalMessage(&message.writer, "legacy dir/pack.safetensors", "model's dir");
    try t.expectEqualStrings("pack legacy dir/pack.safetensors has no recorded source; rebuild it with: tf-flashnext-pack 'model'\\''s dir' 'legacy dir' --draft-vocab 'zig/src/families/flashnext/draft_vocab_default.txt'", message.written());
}

test "artifact identity: pack writes record the actual emitted digest and opened-pack keys bind it" {
    if (!@import("src/core/digest_key.zig").persistent) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Case.init(a);
    defer c.tmp.cleanup();
    const root = try c.path("digests");
    const generated = try digest_cache.generatedRoot(a, root);
    const path_ = try c.path("pack.safetensors");
    var out = pio.Out.init(a);
    defer out.deinit();
    try out.put("w", "U8", &.{4}, try a.dupe(u8, "pack"));
    try out.writeWithCache(a, t.io, path_, "source", generated);
    const got = (try (digest_cache.Cache{ .root = generated }).lookup(a, t.io, path_)).?;
    const bytes = try c.read("pack.safetensors");
    try t.expectEqualSlices(u8, &pio.hashBytes(bytes), &got.value.sha256);
    const file = try c.tmp.dir.openFile(t.io, "pack.safetensors", .{});
    defer file.close(t.io);
    const policy = @import("src/native/persistence_policy.zig");
    try t.expectError(error.UnverifiedPersistence, policy.filesWithCache(7, a, t.io, &.{file.handle}, null));
    const verified = try policy.filesWithCache(7, a, t.io, &.{file.handle}, root);
    try t.expectEqual(verified, try policy.filesWithCache(7, a, t.io, &.{file.handle}, root));
}

test "artifact identity: existing source digests are bound and missing digests stay a metadata-only miss" {
    if (!@import("src/core/digest_key.zig").persistent) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Case.init(a);
    defer c.tmp.cleanup();
    const root = try c.path("digests");
    const before = try pio.sourceIdentityWithCache(a, t.io, c.dir, root);
    try t.expectError(error.FileNotFound, c.tmp.dir.openDir(t.io, "digests", .{}));
    const path_ = try c.path("one.safetensors");
    const file = try c.tmp.dir.openFile(t.io, "one.safetensors", .{});
    defer file.close(t.io);
    const key = try digest_cache.Key.of(file, t.io);
    const sha = pio.hashBytes(try c.read("one.safetensors"));
    try (digest_cache.Cache{ .root = root }).remember(a, t.io, path_, key, sha);
    const verified = try pio.sourceIdentityWithCache(a, t.io, c.dir, root);
    try t.expectEqualStrings(before, verified);
    const field = try std.fmt.allocPrint(a, "payload {s}\n", .{&std.fmt.bytesToHex(sha, .lower)});
    try t.expect(std.mem.indexOf(u8, verified, field) == null);
    const content = try pio.identityWithContent(a, t.io, c.dir, root, false);
    try t.expect(std.mem.indexOf(u8, content, field) != null);
    try t.expectEqualStrings(verified, try pio.sourceIdentityWithCache(a, t.io, c.dir, root));
    const generated = try digest_cache.generatedRoot(a, root);
    var out = pio.Out.init(a);
    defer out.deinit();
    try out.put("w", "U8", &.{4}, try a.dupe(u8, "pack"));
    try out.writeWithCache(a, t.io, try c.path("pack.safetensors"), verified, generated);
    try t.expectEqualStrings(verified, try pio.sourceIdentityWithCache(a, t.io, c.dir, root));
}

test "artifact identity: source identity cannot read a protected payload page" {
    if (@import("builtin").os.tag != .linux and @import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const page = std.heap.pageSize();
    const payload = try a.alloc(u8, 2 * page);
    @memset(payload, 0);
    const image_ = try fixture.image(a, &.{.{ .name = "w", .shape = &.{payload.len}, .bytes = payload }});
    const memory = try std.posix.mmap(null, 4 * page, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
    defer std.posix.munmap(memory);
    @memcpy(memory[0..image_.len], image_);
    try t.expectEqual(@as(c_int, 0), std.c.mprotect(@ptrCast(@alignCast(memory.ptr + page)), page, .{}));
    const stamp: digest_cache.Key = .{ .device = 1, .inode = 1, .size = image_.len, .mtime = 1, .ctime = 1 };
    const identity = try pio.identityFromMapped(a, null, "{}", &.{.{ .name = "source", .size = image_.len, .bytes = memory[0..image_.len], .stamp = stamp }});
    try t.expect(std.mem.indexOf(u8, identity, "stamp ") == null);
}

test "artifact identity: known limit preserves identity for a payload edit with identical supplied stamps" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const bytes = try fixture.image(a, &.{.{ .name = "w", .shape = &.{4}, .bytes = "abcd" }});
    const stamp: digest_cache.Key = .{ .device = 1, .inode = 1, .size = bytes.len, .mtime = 1, .ctime = 1 };
    const mapped: pio.MappedShard = .{ .name = "source", .size = bytes.len, .bytes = bytes, .stamp = stamp };
    const before = try pio.identityFromMapped(a, null, "{}", &.{mapped});
    bytes[bytes.len - 1] ^= 1;
    try t.expectEqualStrings(before, try pio.identityFromMapped(a, null, "{}", &.{mapped}));
}

test "artifact identity: source copy and digest cache warmth preserve identity cache key and emitted bytes" {
    if (!@import("src/core/digest_key.zig").persistent) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var c = try Case.init(a);
    defer c.tmp.cleanup();
    var copy = try Case.init(a);
    defer copy.tmp.cleanup();
    const root = try c.path("digests");
    const cold = try pio.sourceIdentityWithCache(a, t.io, c.dir, root);
    const copied = try pio.sourceIdentityWithCache(a, t.io, copy.dir, root);
    try t.expectEqualStrings(cold, copied);
    const path_ = try c.path("one.safetensors");
    const file = try c.tmp.dir.openFile(t.io, "one.safetensors", .{});
    defer file.close(t.io);
    const sha = pio.hashBytes(try c.read("one.safetensors"));
    try (digest_cache.Cache{ .root = root }).remember(a, t.io, path_, try digest_cache.Key.of(file, t.io), sha);
    const warm = try pio.sourceIdentityWithCache(a, t.io, c.dir, root);
    try t.expectEqualStrings(cold, warm);
    const cache = @import("src/families/flashnext/pack_cache.zig");
    try t.expectEqualStrings(try cache.directory(a, "packs", cold), try cache.directory(a, "packs", warm));
    var out = pio.Out.init(a);
    defer out.deinit();
    try out.put("w", "U8", &.{4}, try a.dupe(u8, "pack"));
    try out.writeWithCache(a, t.io, try c.path("cold.safetensors"), cold, null);
    try out.writeWithCache(a, t.io, try c.path("warm.safetensors"), warm, null);
    try t.expectEqualSlices(u8, try c.read("cold.safetensors"), try c.read("warm.safetensors"));
}
