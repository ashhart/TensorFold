const std = @import("std");
const safety = @import("safety_test.zig");
const Bank = @import("bank.zig").Bank;
const store = @import("store.zig");
const t = std.testing;

const Replacement = struct { key: []const u8, value: []const u8 };

const Specs = struct {
    meta: std.ArrayList(store.MetaKV) = .empty,
    tensors: std.ArrayList(store.TensorSpec) = .empty,

    fn deinit(self: *Specs, gpa: std.mem.Allocator) void {
        self.meta.deinit(gpa);
        self.tensors.deinit(gpa);
    }
};

fn specs(gpa: std.mem.Allocator, lf: *store.LoadedFile, replacement: ?Replacement) !Specs {
    var out: Specs = .{};
    errdefer out.deinit(gpa);
    var mi = lf.meta.iterator();
    while (mi.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.eql(u8, key, "data_crc32")) continue;
        const value = if (replacement) |r| if (std.mem.eql(u8, key, r.key)) r.value else entry.value_ptr.* else entry.value_ptr.*;
        try out.meta.append(gpa, .{ .key = key, .value = value });
    }
    var ti = lf.tensors.iterator();
    while (ti.next()) |entry| {
        try out.tensors.append(gpa, .{
            .name = entry.key_ptr.*,
            .dtype = entry.value_ptr.dtype,
            .shape = entry.value_ptr.shape[0..entry.value_ptr.rank],
            .bytes = entry.value_ptr.bytes,
        });
    }
    return out;
}

fn bankDir(tmp: *std.testing.TmpDir) ![:0]u8 {
    return tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
}

fn rewriteBase(dir: []const u8, replacement: Replacement) !void {
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.file_name });
    defer t.allocator.free(path);
    var lf = try store.readFile(t.allocator, t.io, path);
    defer lf.deinit();
    var all = try specs(t.allocator, &lf, replacement);
    defer all.deinit(t.allocator);
    _ = try store.writeFileAtomic(t.allocator, t.io, path, all.meta.items, all.tensors.items);
}

fn rewriteUnit(gpa: std.mem.Allocator, payload: []const u8, replacement: Replacement) ![]u8 {
    const owned = try gpa.dupe(u8, payload);
    var lf = store.parseOwnedBytes(gpa, owned) catch |e| {
        gpa.free(owned);
        return e;
    };
    defer lf.deinit();
    var all = try specs(gpa, &lf, replacement);
    defer all.deinit(gpa);
    return store.serializeToBytes(gpa, all.meta.items, all.tensors.items);
}

fn expectLoadError(dir: []const u8, mock: *safety.Mock, expected: anyerror) !void {
    const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
    defer back.deinit();
    try t.expectError(expected, back.loadIfPresent());
}

fn savedBase(dir: []const u8, mock: *safety.Mock) !*Bank {
    return safety.taught(t.allocator, dir, mock, true);
}

fn isolatedAdd(dir: []const u8, b: *Bank) ![]u8 {
    _ = try b.teachChunk(.{ .parent_id = 8, .body_ids = &safety.fact_ids, .gate_ids = &safety.fact_ids, .is_small_fact = true });
    _ = try b.save();
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.delta_file_name });
    defer t.allocator.free(path);
    const frames = try store.readDeltaFrames(t.allocator, t.io, path, b.journal_gen);
    defer {
        for (frames) |*frame| frame.deinit(t.allocator);
        t.allocator.free(frames);
    }
    try t.expectEqual(@as(usize, 1), frames.len);
    const payload = try t.allocator.dupe(u8, frames[0].payload);
    errdefer t.allocator.free(payload);
    try std.Io.Dir.cwd().deleteFile(t.io, path);
    return payload;
}

fn appendAddThenValid(dir: []const u8, generation: u64, malformed: []const u8, valid: []const u8) !void {
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.delta_file_name });
    defer t.allocator.free(path);
    _ = try store.appendDeltaFrame(t.allocator, t.io, path, generation, .add_unit, malformed);
    _ = try store.appendDeltaFrame(t.allocator, t.io, path, generation, .add_unit, valid);
}

fn malformedAddCase(unit_meta: ?[]const u8, expected: anyerror) !void {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try savedBase(dir, &mock);
    defer b.deinit();
    const valid = try isolatedAdd(dir, b);
    defer t.allocator.free(valid);
    var meta_buf: [1]store.MetaKV = undefined;
    const meta: []const store.MetaKV = if (unit_meta) |value| blk: {
        meta_buf[0] = .{ .key = "unit", .value = value };
        break :blk meta_buf[0..1];
    } else meta_buf[0..0];
    const malformed = try store.serializeToBytes(t.allocator, meta, &.{});
    defer t.allocator.free(malformed);
    try appendAddThenValid(dir, b.journal_gen, malformed, valid);
    try expectLoadError(dir, &mock, expected);
}

test "bank replay: malformed add records before a valid add are not skipped" {
    const malformed_json = [_]store.MetaKV{.{ .key = "unit", .value = "{" }};
    const semantic = try store.serializeToBytes(t.allocator, &malformed_json, &.{});
    defer t.allocator.free(semantic);
    for ([_][]const u8{ "not a safetensors add", semantic }) |malformed| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try bankDir(&tmp);
        defer t.allocator.free(dir);
        var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
        const b = try savedBase(dir, &mock);
        defer b.deinit();
        const valid = try isolatedAdd(dir, b);
        defer t.allocator.free(valid);
        try appendAddThenValid(dir, b.journal_gen, malformed, valid);
        const back = try Bank.init(t.allocator, t.io, dir, mock.backend());
        defer back.deinit();
        if (back.loadIfPresent()) |_| return error.ExpectedReplayFailure else |_| {}
    }
}

test "bank replay: add metadata and unit failures propagate ahead of a valid frame" {
    try malformedAddCase(null, error.MissingMetadata);
    try malformedAddCase("{}", error.BadLivingWeights);
}

test "bank replay: every tombstone has exactly its serialized payload size" {
    const cases = [_]struct { op: store.DeltaOp, len: usize }{
        .{ .op = .forget_parent, .len = 3 },
        .{ .op = .forget_parent, .len = 5 },
        .{ .op = .reinforce_parent, .len = 0 },
        .{ .op = .reinforce_parent, .len = 8 },
        .{ .op = .supersede_parent, .len = 7 },
        .{ .op = .supersede_parent, .len = 9 },
    };
    for (cases) |case| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try bankDir(&tmp);
        defer t.allocator.free(dir);
        var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
        const b = try savedBase(dir, &mock);
        defer b.deinit();
        const path = try std.fs.path.join(t.allocator, &.{ dir, store.delta_file_name });
        defer t.allocator.free(path);
        const payload = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        _ = try store.appendDeltaFrame(t.allocator, t.io, path, b.journal_gen, case.op, payload[0..case.len]);
        try expectLoadError(dir, &mock, error.BadDeltaFrame);
    }
}

test "bank load: counters are present, range checked, and never advanced past their domain" {
    const replacements = [_]Replacement{
        .{ .key = "next_uid", .value = "4294967296" },
        .{ .key = "next_epoch", .value = "not-an-epoch" },
        .{ .key = "units", .value = "[{\"uid\":4294967295,\"pid\":7,\"tok\":5,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":0,\"cv\":false,\"sm\":true,\"st\":false,\"dg\":false,\"nsemb\":0,\"ep\":1,\"sup\":0,\"rf\":1}]" },
        .{ .key = "units", .value = "[{\"uid\":1,\"pid\":7,\"tok\":5,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":0,\"cv\":false,\"sm\":true,\"st\":false,\"dg\":false,\"nsemb\":0,\"ep\":18446744073709551615,\"sup\":0,\"rf\":1}]" },
    };
    for (replacements) |replacement| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try bankDir(&tmp);
        defer t.allocator.free(dir);
        var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
        const b = try savedBase(dir, &mock);
        defer b.deinit();
        try rewriteBase(dir, replacement);
        try expectLoadError(dir, &mock, error.BadLivingWeights);
    }
}

test "bank replay: maximum unit and epoch values cannot wrap the next counters" {
    const cases = [_]struct { uid: u32, epoch: u64 }{
        .{ .uid = std.math.maxInt(u32), .epoch = 2 },
        .{ .uid = 2, .epoch = std.math.maxInt(u64) },
    };
    for (cases) |case| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try bankDir(&tmp);
        defer t.allocator.free(dir);
        var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
        const b = try savedBase(dir, &mock);
        defer b.deinit();
        const valid = try isolatedAdd(dir, b);
        defer t.allocator.free(valid);
        const unit_json = try std.fmt.allocPrint(t.allocator, "{{\"uid\":{d},\"pid\":8,\"tok\":5,\"nl\":0,\"nbl\":0,\"ns\":0,\"rk\":0,\"cv\":false,\"sm\":true,\"st\":false,\"dg\":false,\"nsemb\":0,\"ep\":{d},\"sup\":0,\"rf\":1}}", .{ case.uid, case.epoch });
        defer t.allocator.free(unit_json);
        const malformed = try rewriteUnit(t.allocator, valid, .{ .key = "unit", .value = unit_json });
        defer t.allocator.free(malformed);
        try appendAddThenValid(dir, b.journal_gen, malformed, valid);
        try expectLoadError(dir, &mock, error.BadLivingWeights);
    }
}

test "bank teach: exhausted unit and epoch counters fail before compilation" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try safety.taught(t.allocator, dir, &mock, false);
    defer b.deinit();
    const input: @import("bank.zig").TeachChunkInput = .{ .parent_id = 8, .body_ids = &safety.fact_ids, .gate_ids = &safety.fact_ids, .is_small_fact = true };
    const count = b.units.items.len;
    b.next_unit_id = std.math.maxInt(u32);
    try t.expectError(error.UnitIdExhausted, b.teachChunk(input));
    b.next_unit_id = 2;
    b.next_epoch = std.math.maxInt(u64);
    try t.expectError(error.EpochExhausted, b.teachChunk(input));
    try t.expectEqual(count, b.units.items.len);
}

test "bank load: whitener sizes obey the writer limits" {
    const replacements = [_]Replacement{
        .{ .key = "lex_k", .value = "17" },
        .{ .key = "lex_cal", .value = "561" },
        .{ .key = "sem_k", .value = "33" },
        .{ .key = "sem_cal", .value = "481" },
    };
    for (replacements) |replacement| {
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try bankDir(&tmp);
        defer t.allocator.free(dir);
        var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
        const b = try savedBase(dir, &mock);
        defer b.deinit();
        try rewriteBase(dir, replacement);
        try expectLoadError(dir, &mock, error.BadLivingWeights);
    }
}

test "bank load: unit count cannot exceed the minimum persisted KV payload" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try bankDir(&tmp);
    defer t.allocator.free(dir);
    var mock: safety.Mock = .{ .gpa = t.allocator, .arch = safety.Mock.spec };
    const b = try savedBase(dir, &mock);
    defer b.deinit();
    const path = try std.fs.path.join(t.allocator, &.{ dir, store.file_name });
    defer t.allocator.free(path);
    var lf = try store.readFile(t.allocator, t.io, path);
    defer lf.deinit();
    const original = lf.getMeta("units").?;
    const one = original[1 .. original.len - 1];
    var many: std.ArrayList(u8) = .empty;
    defer many.deinit(t.allocator);
    try many.append(t.allocator, '[');
    for (0..2048) |i| {
        if (i != 0) try many.append(t.allocator, ',');
        try many.appendSlice(t.allocator, one);
    }
    try many.append(t.allocator, ']');
    try rewriteBase(dir, .{ .key = "units", .value = many.items });
    try expectLoadError(dir, &mock, error.BadLivingWeights);
}
