//! A saved bank is checked before use: its geometry, its data's CRC and its files' owner-only modes.
const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const store = @import("store.zig");
const unit_store = @import("unit_store.zig");
const bridges = @import("bridges.zig");
const gate = @import("gate.zig");
const ArchSpec = @import("types.zig").ArchSpec;
const t = std.testing;

fn saved(tmp: *std.testing.TmpDir, mock: *safety.Mock) ![:0]u8 {
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    errdefer t.allocator.free(dir);
    const b = try safety.taught(t.allocator, dir, mock, true);
    b.deinit();
    return dir;
}

test "bank validate: a bank written for another geometry of the same sizes is refused by name" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try saved(&tmp, &mock);
    defer t.allocator.free(dir);
    var other: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    other.arch.rope_theta = 1_000_000.0; // every tensor the same size, every stored K rotated for another model
    const b = try Bank.init(t.allocator, t.io, dir, other.backend());
    defer b.deinit();
    try t.expectError(error.LivingWeightsArchMismatch, b.loadIfPresent());
    try t.expect(!b.hasUnits());
}

test "bank validate: a flipped byte in a saved tensor is refused by name before any value is used" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try saved(&tmp, &mock);
    defer t.allocator.free(dir);
    {
        const f = try tmp.dir.openFile(t.io, store.file_name, .{ .mode = .read_write });
        defer f.close(t.io);
        var len: [8]u8 = undefined;
        _ = try f.readPositionalAll(t.io, &len, 0);
        const data = 8 + std.mem.readInt(u64, &len, .little);
        var byte: [1]u8 = undefined;
        _ = try f.readPositionalAll(t.io, &byte, data + 64);
        byte[0] ^= 0x10;
        try f.writePositionalAll(t.io, &byte, data + 64);
    }
    const b = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer b.deinit();
    try t.expectError(error.LivingWeightsCorrupt, b.loadIfPresent());
}

test "bank validate: the base file and the journal are created readable by their owner alone" {
    if (@import("builtin").os.tag == .windows) return error.SkipZigTest;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(dir);
    const b = try safety.taught(t.allocator, dir, &mock, true);
    defer b.deinit();
    const more = [_]u32{ 40, 41, 42, 43 };
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &more, .gate_ids = &more, .is_small_fact = true });
    _ = try b.save(); // into the journal
    for ([_][]const u8{ store.file_name, store.delta_file_name }) |name| {
        const st = try tmp.dir.statFile(t.io, name, .{});
        try t.expectEqual(@as(std.posix.mode_t, 0o600), st.permissions.toMode() & 0o777);
    }
}

test "bank validate: base replacement never follows a pre-positioned temporary link" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const before = "target must stay byte-for-byte unchanged";
    try tmp.dir.writeFile(io, .{ .sub_path = "target.safetensors", .data = before });
    tmp.dir.symLink(io, "target.safetensors", store.file_name ++ ".tmp", .{}) catch |e| return switch (e) {
        error.AccessDenied, error.PermissionDenied => if (@import("builtin").os.tag == .windows) error.SkipZigTest else e,
        else => e,
    };
    const dir = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.file_name });
    defer t.allocator.free(path);
    _ = try store.writeFileAtomic(t.allocator, io, path, &.{}, &.{});
    var after: [64]u8 = undefined;
    try t.expectEqualStrings(before, try tmp.dir.readFile(io, "target.safetensors", &after));
}

test "bank validate: an oversized sparse base is refused before allocation" {
    const io = t.io;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", t.allocator);
    defer t.allocator.free(dir);
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.file_name });
    defer t.allocator.free(path);
    var file = try tmp.dir.createFile(io, store.file_name, .{});
    try file.setLength(io, 8 * 1024 * 1024 * 1024 + 16);
    file.close(io);
    var small: [1024]u8 = undefined;
    var fba: std.heap.FixedBufferAllocator = .init(&small);
    try t.expectError(error.BadSafetensors, store.readFile(fba.allocator(), io, path));
}

fn rawSafetensors(header: []const u8, data: []const u8) ![]u8 {
    const raw = try t.allocator.alloc(u8, 8 + header.len + data.len);
    std.mem.writeInt(u64, raw[0..8], @intCast(header.len), .little);
    @memcpy(raw[8 .. 8 + header.len], header);
    @memcpy(raw[8 + header.len ..], data);
    return raw;
}

fn expectBadSafetensors(header: []const u8, data: []const u8) !void {
    const raw = try rawSafetensors(header, data);
    defer t.allocator.free(raw);
    try t.expectError(error.BadSafetensors, store.parseOwnedBytes(t.allocator, raw));
}

fn emptyLoadedFile() !store.LoadedFile {
    const raw = try rawSafetensors("{\"__metadata__\":{}}", &.{});
    errdefer t.allocator.free(raw);
    return store.parseOwnedBytes(t.allocator, raw);
}

fn tinyArch() ArchSpec {
    return .{
        .family = .qwen3_5,
        .hidden = 1,
        .num_layers = 1,
        .layer_lex = 0,
        .layer_sem = 0,
        .num_attn = 0,
        .kv_heads = 0,
        .head_dim = 1,
        .rotary_dim = 0,
        .rope_theta = 10_000,
        .num_rec = 0,
        .rec_heads = 1,
        .rec_rows = 1,
        .rec_cols = 1,
        .conv_elems = 0,
    };
}

fn expectBadUnit(lf: *const store.LoadedFile, arch: ArchSpec, json: []const u8, ui: usize) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, t.allocator, json, .{});
    defer parsed.deinit();
    try t.expectError(error.BadLivingWeights, unit_store.loadUnitFromObject(t.allocator, lf, arch, parsed.value, ui));
}

fn emptyWhitenerFile() !store.LoadedFile {
    const mean = [_]f32{0};
    const tensors = [_]store.TensorSpec{
        .{ .name = "w.lex.mean", .dtype = .f32, .shape = &.{1}, .bytes = std.mem.sliceAsBytes(&mean) },
        .{ .name = "w.lex.basis", .dtype = .f32, .shape = &.{ 0, 1 }, .bytes = &.{} },
        .{ .name = "w.lex.cal", .dtype = .f32, .shape = &.{ 0, 1 }, .bytes = &.{} },
    };
    const raw = try store.serializeToBytes(t.allocator, &.{}, &tensors);
    errdefer t.allocator.free(raw);
    return store.parseOwnedBytes(t.allocator, raw);
}

test "safetensors validation: dtype byte products and copy lengths reject overflow" {
    if (@sizeOf(usize) != 8) return error.SkipZigTest;
    try expectBadSafetensors("{\"x\":{\"dtype\":\"F32\",\"shape\":[4611686018427387904],\"data_offsets\":[0,0]}}", &.{});

    const raw = try rawSafetensors("{\"f\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]},\"b\":{\"dtype\":\"BF16\",\"shape\":[0],\"data_offsets\":[0,0]}}", &.{});
    errdefer t.allocator.free(raw);
    var lf = try store.parseOwnedBytes(t.allocator, raw);
    defer lf.deinit();
    try t.expectError(error.BadTensorShape, lf.copyF32(t.allocator, "f", std.math.maxInt(usize) / 4 + 1));
    try t.expectError(error.BadTensorShape, lf.copyBf16(t.allocator, "b", std.math.maxInt(usize) / 2 + 1));
}

test "safetensors validation: tensor ranges reject aliases gaps and uncovered bytes" {
    try expectBadSafetensors("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 0 });
    try expectBadSafetensors("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
    try expectBadSafetensors("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,12]}}", &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try expectBadSafetensors("{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 0, 0, 0, 0, 0 });
}

test "safetensors validation: unsorted adjacent and zero-length ranges remain valid" {
    const raw = try rawSafetensors("{\"a\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[2,4]},\"z\":{\"dtype\":\"U8\",\"shape\":[0],\"data_offsets\":[4,4]},\"b\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", &.{ 10, 11, 20, 21 });
    errdefer t.allocator.free(raw);
    var lf = try store.parseOwnedBytes(t.allocator, raw);
    defer lf.deinit();
    try t.expectEqualSlices(u8, &.{ 20, 21 }, lf.getTensor("a").?.bytes);
    try t.expectEqualSlices(u8, &.{ 10, 11 }, lf.getTensor("b").?.bytes);
    try t.expectEqual(@as(usize, 0), lf.getTensor("z").?.bytes.len);
}

test "unit metadata validation: writer prototype bounds are enforced" {
    var lf = try emptyLoadedFile();
    defer lf.deinit();
    try expectBadUnit(&lf, tinyArch(), "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":211,\"nbl\":0,\"ns\":0,\"rk\":0,\"nsemb\":0}", 0);
    try expectBadUnit(&lf, tinyArch(), "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":181,\"nbl\":181,\"ns\":0,\"rk\":0,\"nsemb\":0}", 0);
    try expectBadUnit(&lf, tinyArch(), "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":0,\"nbl\":0,\"ns\":118,\"rk\":0,\"nsemb\":0}", 0);
}

test "unit metadata validation: prototype KV and recurrent products reject overflow" {
    var lf = try emptyLoadedFile();
    defer lf.deinit();
    var arch = tinyArch();
    arch.hidden = std.math.maxInt(usize) / 2 + 1;
    try expectBadUnit(&lf, arch, "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":2,\"nbl\":1,\"ns\":0,\"rk\":0,\"nsemb\":0}", 0);

    arch = tinyArch();
    arch.kv_heads = std.math.maxInt(usize) / 2;
    try expectBadUnit(&lf, arch, "{\"uid\":1,\"pid\":1,\"tok\":4,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":0,\"nsemb\":0}", 0);

    arch = tinyArch();
    arch.num_rec = 1;
    arch.rec_heads = std.math.maxInt(usize) / 2;
    arch.rec_rows = 4;
    try expectBadUnit(&lf, arch, "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":1,\"nsemb\":0}", 0);
}

test "whitener metadata validation: products and writer bounds reject impossible counts" {
    var lf = try emptyWhitenerFile();
    defer lf.deinit();
    const too_wide = std.math.maxInt(usize) / 2 + 1;
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.lex", 0, 2, 0.1, too_wide));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.lex", 2, 0, 0.1, too_wide));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.lex", 17, 0, 0.1, 64));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.lex", 0, 561, 0.1, 1));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.lex", 5, 0, 0.1, 4));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.sem", 33, 0, 0.1, 64));
    try t.expectError(error.BadLivingWeights, unit_store.loadWhitener(&lf, t.allocator, "w.sem", 0, 481, 0.1, 1));
}

test "unit metadata validation: tensor payload bounds the supported unit index" {
    const raw = try rawSafetensors("{\"x\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}", &.{ 0, 0, 0, 0 });
    errdefer t.allocator.free(raw);
    var lf = try store.parseOwnedBytes(t.allocator, raw);
    defer lf.deinit();
    var arch = tinyArch();
    arch.num_attn = 1;
    arch.kv_heads = 1;
    try expectBadUnit(&lf, arch, "{\"uid\":1,\"pid\":1,\"tok\":1,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":0,\"nsemb\":0}", 1);
}

test "bridge validation: pair-matrix sizing rejects overflow" {
    var empty: [0]f32 = .{};
    var dummy_gate: gate.UnitGate = .{
        .dim = 0,
        .lex_protos = empty[0..],
        .lex_weights = empty[0..],
        .n_lex = 0,
        .sem_protos = empty[0..],
        .sem_weights = empty[0..],
        .n_sem = 0,
    };
    const view: gate.UnitView = .{ .index = 0, .parent_id = 0, .n_tokens = 1, .gate = &dummy_gate };
    const one = (&view)[0..1];
    const impossible = one.ptr[0..std.math.maxInt(usize)];
    try t.expectError(error.OutOfMemory, bridges.computeBridgeEdges(t.allocator, impossible));
}
