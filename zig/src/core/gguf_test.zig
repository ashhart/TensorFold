//! Locally serialized containers, including hostile lengths. No downloaded fixtures or GPU dependencies.
const std = @import("std");
const g = @import("gguf.zig");
const t = std.testing;
const Fixture = @import("gguf_test_fixture.zig").Fixture;

fn parse(b: *const Fixture) !g.Document {
    return g.Document.parse(t.allocator, b.slice(), .{});
}
fn refused(b: *const Fixture, err: anyerror) !void {
    if (parse(b)) |value| {
        var doc = value;
        defer doc.deinit();
        return error.TestExpectedError;
    } else |actual| try t.expectEqual(err, actual);
}
test "GGUF: every metadata type and borrowed arrays, including embedded zero tokens" {
    var b = Fixture.init(0, 13);
    const values = [_]g.Value{
        .{ .uint8 = 255 },          .{ .int8 = -128 },                    .{ .uint16 = 65535 },               .{ .int16 = -32768 },
        .{ .uint32 = 0xffff_ffff }, .{ .int32 = -2147483648 },            .{ .float32 = -1.25 },              .{ .boolean = true },
        .{ .string = "a\x00b" },    .{ .uint64 = 0xffff_ffff_ffff_ffff }, .{ .int64 = -9223372036854775808 }, .{ .float64 = 1.0 / 3.0 },
    };
    const keys = [_][]const u8{ "u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "u64", "i64", "f64" };
    for (keys, values) |key, value| b.kv(key, value);
    b.array("nested", .array, 1);
    b.int(u32, @backingInt(g.MetadataKind.string));
    b.int(u64, 2);
    b.string("first");
    b.string("second");
    b.finish(32, 0);
    var doc = try parse(&b);
    defer doc.deinit();
    for (keys, values) |key, value| {
        const actual = doc.get(key).?;
        try t.expectEqual(std.meta.activeTag(value), std.meta.activeTag(actual));
        switch (value) {
            .string => |x| try t.expectEqualStrings(x, actual.string),
            inline else => |x, tag| try t.expectEqual(x, @field(actual, @tagName(tag))),
        }
    }
    const array = (try doc.get("nested").?.array.at(0)).array;
    var it = array.iterator();
    try t.expectEqualStrings("first", (try it.next()).?.string);
    try t.expectEqualStrings("second", (try it.next()).?.string);
    try t.expect((try it.next()) == null);
    try t.expectError(error.ArrayIndexOutOfBounds, array.at(2));
    const str = doc.get("string").?.string;
    try t.expect(@intFromPtr(str.ptr) >= @intFromPtr(b.bytes[0..].ptr));
    try t.expect(@intFromPtr(str.ptr) + str.len <= @intFromPtr(b.bytes[0..].ptr) + b.len);
    try t.expect(doc.get("absent") == null);
    try t.expectError(error.MissingArchitecture, doc.config());
}

test "GGUF: architecture scoped config, dimensions, rope, norm and complete tokenizer views" {
    var b = Fixture.init(0, 28);
    b.kv("general.architecture", .{ .string = "test" });
    const keys = [_][]const u8{ "test.block_count", "test.embedding_length", "test.feed_forward_length", "test.attention.head_count", "test.attention.head_count_kv", "test.attention.key_length", "test.attention.value_length", "test.context_length", "test.rope.dimension_count", "test.rope.scaling.original_context_length" };
    const dims = [_]u32{ 4, 256, 512, 8, 2, 32, 64, 4096, 32, 1024 };
    for (keys, dims) |k, v| b.kv(k, .{ .uint32 = v });
    b.kv("test.rope.freq_base", .{ .float32 = 10000 });
    b.kv("test.rope.scaling.type", .{ .string = "linear" });
    b.kv("test.rope.scaling.factor", .{ .float64 = 4 });
    b.kv("test.rope.scaling.finetuned", .{ .boolean = true });
    b.kv("test.attention.layer_norm_rms_epsilon", .{ .float32 = 0.00001 });
    b.kv("test.attention.layer_norm_epsilon", .{ .float64 = 0.00002 });
    b.kv("other.block_count", .{ .uint32 = 999 });
    b.kv("tokenizer.ggml.model", .{ .string = "gpt2" });
    b.kv("tokenizer.ggml.pre", .{ .string = "test" });
    b.array("tokenizer.ggml.tokens", .string, 2);
    b.string("\x00");
    b.string("two");
    b.array("tokenizer.ggml.scores", .float32, 2);
    b.int(u32, @bitCast(@as(f32, -1)));
    b.int(u32, @bitCast(@as(f32, 0.5)));
    b.array("tokenizer.ggml.token_type", .int32, 2);
    b.int(i32, 1);
    b.int(i32, 3);
    b.array("tokenizer.ggml.merges", .string, 1);
    b.string("a b");
    b.kv("tokenizer.ggml.bos_token_id", .{ .uint32 = 0 });
    b.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = 1 });
    b.kv("tokenizer.ggml.add_bos_token", .{ .boolean = true });
    b.kv("tokenizer.ggml.add_eos_token", .{ .boolean = false });
    b.finish(32, 0);
    var doc = try parse(&b);
    defer doc.deinit();
    const c = try doc.config();
    try t.expectEqualStrings("test", c.architecture);
    const actual = [_]?u64{ c.layers, c.embedding, c.feed_forward, c.heads, c.kv_heads, c.key_length, c.value_length, c.context, c.rope_dimensions, c.rope_original_context };
    for (actual, dims) |got, want| try t.expectEqual(@as(u64, want), got.?);
    try t.expectEqual(@as(f64, 10000), c.rope_base.?);
    try t.expectEqualStrings("linear", c.rope_scaling.?);
    try t.expectEqual(@as(f64, 4), c.rope_factor.?);
    try t.expect(c.rope_finetuned.?);
    try t.expectEqual(@as(f64, @as(f32, 0.00001)), c.rms_epsilon.?);
    try t.expectEqual(@as(f64, 0.00002), c.layer_norm_epsilon.?);
    const tok = try doc.tokenizer();
    try t.expectEqualStrings("gpt2", tok.model.?);
    try t.expectEqualStrings("test", tok.pre.?);
    try t.expectEqualStrings("\x00", (try tok.tokens.?.at(0)).string);
    try t.expectEqual(@as(f32, 0.5), (try tok.scores.?.at(1)).float32);
    try t.expectEqual(@as(i32, 3), (try tok.types.?.at(1)).int32);
    try t.expectEqualStrings("a b", (try tok.merges.?.at(0)).string);
    try t.expectEqual(@as(u64, 0), tok.bos.?);
    try t.expectEqual(@as(u64, 1), tok.eos.?);
    try t.expect(tok.add_bos.? and !tok.add_eos.?);
}

test "GGUF: tensor lengths follow row geometry and sorted ranges, with custom alignment" {
    var b = Fixture.init(2, 1);
    b.kv("general.alignment", .{ .uint32 = 64 });
    b.tensor("later", &.{ 32, 2 }, .q8_0, 64);
    b.tensor("first", &.{ 4, 2, 1, 1 }, .f32, 0);
    b.finish(64, 132);
    var doc = try parse(&b);
    defer doc.deinit();
    try t.expectEqual(@as(usize, 64), doc.alignment);
    try t.expectEqual(@as(usize, 0), doc.data_offset % 64);
    try t.expectEqualStrings("first", doc.tensors[0].name);
    try t.expectEqual(@as(usize, 32), doc.getTensor("first").?.bytes.len);
    try t.expectEqual(@as(usize, 68), doc.getTensor("later").?.bytes.len);
    try t.expect(doc.getTensor("missing") == null);
    try t.expectEqualSlices(u8, b.slice()[doc.data_offset + 64 ..], doc.getTensor("later").?.bytes);
}

test "GGUF: every supported block size and raw dtype has an exact bounded byte view" {
    inline for (std.meta.tags(g.TensorKind)) |kind| {
        const layout = kind.block();
        var b = Fixture.init(1, 0);
        b.tensor("w", &.{ @intCast(layout.values), 2 }, kind, 0);
        b.finish(32, 2 * layout.bytes);
        var doc = try parse(&b);
        defer doc.deinit();
        try t.expectEqual(2 * layout.bytes, doc.tensors[0].bytes.len);
        b.len -= 1;
        try refused(&b, error.TensorOutsideFile);
    }
}

test "GGUF: malformed headers, length words, booleans and alignment fail by name" {
    var b = Fixture.init(0, 0);
    b.finish(32, 0);
    b.bytes[0] = 'X';
    try refused(&b, error.BadGgufMagic);
    b.bytes[0] = 'G';
    b.set(u32, 4, 2);
    try refused(&b, error.UnsupportedGgufVersion);
    b = Fixture.init(0, 0);
    try refused(&b, error.TruncatedAlignmentPadding);
    b = Fixture.init(0, 1);
    b.int(u64, std.math.maxInt(u64));
    b.len = 64;
    try refused(&b, error.TruncatedInput);
    b = Fixture.init(0, 1);
    b.string("x");
    b.int(u32, 123);
    b.finish(32, 0);
    try refused(&b, error.UnknownMetadataType);
    b = Fixture.init(0, 1);
    b.string("x");
    b.int(u32, 7);
    b.int(u8, 2);
    b.finish(32, 0);
    try refused(&b, error.InvalidBoolean);
    inline for (.{ @as(u32, 0), @as(u32, 3) }) |a| {
        b = Fixture.init(0, 1);
        b.kv("general.alignment", .{ .uint32 = a });
        b.finish(32, 0);
        try refused(&b, error.InvalidAlignment);
    }
    b = Fixture.init(0, 1);
    b.kv("general.alignment", .{ .uint64 = 32 });
    b.finish(32, 0);
    try refused(&b, error.InvalidAlignmentType);
}

test "GGUF: counts, strings, arrays and tensor geometry are refused before allocator use" {
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const bad = struct {
        fn check(b: *const Fixture, a: std.mem.Allocator, err: anyerror) !void {
            try t.expectError(err, g.Document.parse(a, b.slice(), .{}));
        }
    }.check;
    var b = Fixture.init(std.math.maxInt(u64), 0);
    try bad(&b, failing.allocator(), error.TensorCountExceedsFile);
    b = Fixture.init(0, std.math.maxInt(u64));
    try bad(&b, failing.allocator(), error.MetadataCountExceedsFile);
    b = Fixture.init(0, 1);
    b.array("a", .uint64, std.math.maxInt(u64));
    b.finish(32, 0);
    try bad(&b, failing.allocator(), error.ArrayCountExceedsFile);
    b = Fixture.init(0, 1);
    b.string("a");
    b.int(u32, 8);
    b.int(u64, std.math.maxInt(u64));
    b.finish(32, 0);
    try bad(&b, failing.allocator(), error.TruncatedInput);
    b = Fixture.init(1, 0);
    b.tensor("w", &.{ std.math.maxInt(u64), 2 }, .f32, 0);
    b.finish(32, 0);
    try bad(&b, failing.allocator(), error.TensorShapeOverflow);
    b = Fixture.init(1, 0);
    b.tensor("w", &.{32}, .q8_0, std.math.maxInt(u64) - 31);
    b.finish(32, 34);
    try bad(&b, failing.allocator(), error.TensorOutsideFile);
    try t.expect(!failing.has_induced_failure);
}

test "GGUF: policy caps apply to metadata bytes, entries, tensors, arrays and depth" {
    var b = Fixture.init(0, 1);
    b.kv("a", .{ .string = "payload" });
    b.finish(32, 0);
    try t.expectError(error.MetadataTooLarge, g.Document.parse(t.allocator, b.slice(), .{ .metadata_bytes = 16 }));
    try t.expectError(error.TooManyMetadataEntries, g.Document.parse(t.allocator, b.slice(), .{ .metadata_entries = 0 }));
    b = Fixture.init(1, 0);
    b.tensor("w", &.{1}, .f32, 0);
    b.finish(32, 4);
    try t.expectError(error.TooManyTensors, g.Document.parse(t.allocator, b.slice(), .{ .tensors = 0 }));
    b = Fixture.init(0, 1);
    b.array("a", .uint8, 2);
    b.int(u16, 42);
    b.finish(32, 0);
    try t.expectError(error.TooManyArrayElements, g.Document.parse(t.allocator, b.slice(), .{ .array_elements = 1 }));
    try t.expectError(error.ArrayNestingTooDeep, g.Document.parse(t.allocator, b.slice(), .{ .depth = 0 }));
}

test "GGUF: invalid names, rank, row blocks, dtypes, offsets, duplicates and overlap" {
    var b = Fixture.init(1, 0);
    b.tensor("", &.{1}, .f32, 0);
    b.finish(32, 4);
    try refused(&b, error.InvalidTensorName);
    b = Fixture.init(1, 0);
    b.tensor("x", &.{}, .f32, 0);
    b.finish(32, 64);
    try refused(&b, error.InvalidTensorRank);
    b = Fixture.init(1, 0);
    b.tensor("x", &.{0}, .f32, 0);
    b.finish(32, 4);
    try refused(&b, error.ZeroTensorDimension);
    b = Fixture.init(1, 0);
    b.tensor("x", &.{ 16, 2 }, .q8_0, 0);
    b.finish(32, 34);
    try refused(&b, error.InvalidBlockShape);
    b = Fixture.init(1, 0);
    b.tensor("x", &.{1}, .f32, 1);
    b.finish(32, 8);
    try refused(&b, error.UnalignedTensorOffset);
    b = Fixture.init(1, 0);
    b.tensor("x", &.{1}, .f32, 0);
    b.set(u32, b.len - 12, 0xffff_ffff);
    b.finish(32, 4);
    try refused(&b, error.UnsupportedTensorType);
    b = Fixture.init(2, 0);
    b.tensor("x", &.{9}, .f32, 0);
    b.tensor("y", &.{1}, .f32, 32);
    b.finish(32, 36);
    try refused(&b, error.OverlappingTensors);
    b = Fixture.init(2, 0);
    b.tensor("x", &.{1}, .f32, 0);
    b.tensor("x", &.{1}, .f32, 32);
    b.finish(32, 36);
    try refused(&b, error.DuplicateTensorName);
    b = Fixture.init(0, 2);
    b.kv("x", .{ .uint8 = 1 });
    b.kv("x", .{ .uint8 = 2 });
    b.finish(32, 0);
    try refused(&b, error.DuplicateMetadataKey);
    b = Fixture.init(0, 1);
    b.kv("x\x00y", .{ .uint8 = 1 });
    b.finish(32, 0);
    try refused(&b, error.InvalidMetadataKey);
}

test "GGUF: all truncated prefixes and deterministic mutations return safely" {
    var b = Fixture.rich();
    for (0..b.len) |n| {
        if (g.Document.parse(t.allocator, b.bytes[0..n], .{})) |valid| {
            var doc = valid;
            doc.deinit();
            return error.TruncationAccepted;
        } else |_| {}
    }
    const original = b;
    for (0..original.len) |i| {
        for ([_]u8{ 0, 1, 127, 255 }) |v| {
            b = original;
            b.bytes[i] = v;
            if (parse(&b)) |valid| {
                var doc = valid;
                defer doc.deinit();
                for (doc.tensors) |tensor| try t.expect(tensor.bytes.len == tensor.len);
            } else |_| {}
        }
    }
}
fn allocationTrial(a: std.mem.Allocator) !void {
    var b = Fixture.rich();
    var doc = try g.Document.parse(a, b.slice(), .{});
    defer doc.deinit();
    try t.expectEqual(@as(usize, 34), doc.getTensor("w").?.bytes.len);
}
test "GGUF: every index allocation failure releases all prior allocations" {
    try t.checkAllAllocationFailures(t.allocator, allocationTrial, .{});
}
test "GGUF: typed views refuse wrong kinds, mismatched vocabulary and out of range ids" {
    var b = Fixture.init(0, 1);
    b.kv("general.architecture", .{ .uint32 = 4 });
    b.finish(32, 0);
    var doc = try parse(&b);
    try t.expectError(error.MetadataTypeMismatch, doc.config());
    doc.deinit();
    b = Fixture.init(0, 2);
    b.kv("general.architecture", .{ .string = "test" });
    b.kv("test.block_count", .{ .int64 = -1 });
    b.finish(32, 0);
    doc = try parse(&b);
    try t.expectError(error.MetadataTypeMismatch, doc.config());
    doc.deinit();
    b = Fixture.init(0, 1);
    b.array("tokenizer.ggml.tokens", .uint8, 1);
    b.int(u8, 0);
    b.finish(32, 0);
    doc = try parse(&b);
    try t.expectError(error.MetadataTypeMismatch, doc.tokenizer());
    doc.deinit();
    b = Fixture.init(0, 2);
    b.array("tokenizer.ggml.tokens", .string, 0);
    b.array("tokenizer.ggml.scores", .float32, 1);
    b.int(u32, 0);
    b.finish(32, 0);
    doc = try parse(&b);
    try t.expectError(error.TokenizerLengthMismatch, doc.tokenizer());
    doc.deinit();
    b = Fixture.init(0, 2);
    b.array("tokenizer.ggml.tokens", .string, 1);
    b.string("only");
    b.kv("tokenizer.ggml.eos_token_id", .{ .uint32 = 1 });
    b.finish(32, 0);
    doc = try parse(&b);
    try t.expectError(error.TokenIdOutsideVocabulary, doc.tokenizer());
    doc.deinit();
    b = Fixture.init(0, 1);
    b.kv("general.architecture", .{ .string = "test" });
    b.finish(32, 0);
    doc = try parse(&b);
    defer doc.deinit();
    try t.expect((try doc.config()).kv_heads == null);
    try t.expect((try doc.tokenizer()).tokens == null);
}
test "GGUF: bounded nested arrays and invalid element kinds or string lengths" {
    var b = Fixture.init(0, 1);
    b.array("a", .array, 1);
    for (0..16) |_| {
        b.int(u32, 9);
        b.int(u64, 1);
    }
    b.finish(32, 0);
    try refused(&b, error.ArrayNestingTooDeep);
    b = Fixture.init(0, 1);
    b.array("a", .string, 1);
    b.int(u64, std.math.maxInt(u64));
    b.finish(32, 0);
    try refused(&b, error.TruncatedInput);
    b = Fixture.init(0, 1);
    b.array("a", .string, 1);
    b.set(u32, b.len - 12, 99);
    b.finish(32, 0);
    try refused(&b, error.UnknownMetadataType);
    b = Fixture.init(1, 0);
    b.tensor("w", &.{std.math.maxInt(u64)}, .f32, 0);
    b.finish(32, 0);
    try refused(&b, error.TensorShapeOverflow);
}
test "GGUF: mapped file owns bytes until close and rejects short files" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var b = Fixture.rich();
    try tmp.dir.writeFile(t.io, .{ .sub_path = "small.gguf", .data = b.slice() });
    const path = try tmp.dir.realPathFileAlloc(t.io, "small.gguf", t.allocator);
    defer t.allocator.free(path);
    var file = try g.File.open(t.allocator, t.io, path, .{});
    try t.expectEqual(@as(usize, 34), file.document.getTensor("w").?.bytes.len);
    file.close(t.io);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "small.gguf", .data = "GGUF" });
    try t.expectError(error.TruncatedInput, g.File.open(t.allocator, t.io, path, .{}));
}

const MutatingAllocator = struct {
    bytes: []u8,
    at: usize,
    value: u64,
    changed: bool = false,
    fn allocator(m: *MutatingAllocator) std.mem.Allocator {
        return .{ .ptr = m, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
        const ptr = t.allocator.rawAlloc(len, alignment, ra) orelse return null;
        const m: *MutatingAllocator = @ptrCast(@alignCast(ctx));
        if (!m.changed) {
            // Validation allocates nothing: this mutation occurs after preflight and before indexing.
            m.changed = true;
            std.mem.writeInt(u64, m.bytes[m.at..][0..8], m.value, .little);
        }
        return ptr;
    }
    fn resize(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) bool {
        return t.allocator.rawResize(memory, alignment, len, ra);
    }
    fn remap(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        return t.allocator.rawRemap(memory, alignment, len, ra);
    }
    fn free(_: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
        t.allocator.rawFree(memory, alignment, ra);
    }
};
test "GGUF: mutation between validation and indexing rechecks tensor range before slicing" {
    const cases = [_]struct { shape: bool = false, value: u64, err: anyerror }{
        .{ .value = 1, .err = error.UnalignedTensorOffset },
        .{ .value = 64, .err = error.TensorOutsideFile },
        .{ .value = std.math.maxInt(u64) - 31, .err = error.TensorOutsideFile },
        .{ .shape = true, .value = 128, .err = error.TensorOutsideFile },
    };
    for (cases) |case| {
        var b = Fixture.init(1, 1);
        b.kv("x", .{ .uint8 = 1 });
        const info = b.len;
        b.tensor("w", &.{32}, .q8_0, 0);
        const offset = b.len - 8;
        b.finish(32, 96);
        var m: MutatingAllocator = .{ .bytes = b.bytes[0..b.len], .at = if (case.shape) info + 13 else offset, .value = case.value };
        if (g.Document.parse(m.allocator(), b.slice(), .{})) |accepted| {
            var doc = accepted;
            defer doc.deinit();
            return error.MutatedTensorAccepted;
        } else |err| try t.expectEqual(case.err, err);
        try t.expect(m.changed);
    }
}
