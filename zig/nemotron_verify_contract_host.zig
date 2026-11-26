//! Record the actual diagnostic launchers and verify test-only geometry selection without Metal.
const std = @import("std");
const decode = @import("decode");
const legacy = @import("tests/contract/nemotron_decode_legacy.zig");
const expert = @import("tests/contract/nemotron_verify_experts.zig");
const plain = @import("tests/contract/nemotron_experts_plain.zig");
const cases = @import("tests/contract/nemotron_experts_cases.zig");
const Ref = struct { buf: usize, off: usize = 0 };
const Size = struct { width: usize, height: usize, depth: usize };
const Shot = struct { pipe: usize = 0, buffers: [12]Ref = @splat(.{ .buf = 0 }), dims: [5]i32 = @splat(0), route: [3][8]i32 = @splat(@splat(0)), pairs: i32 = 0, grid: [3]usize = @splat(0), group: [3]usize = @splat(0), page_bytes: u64 = 0 };
const Encoder = struct {
    current: Shot = .{},
    shots: [256]Shot = undefined,
    count: usize = 0,
    pub fn pipe(e: *Encoder, pipeline: usize) void {
        e.current.pipe = pipeline;
    }
    pub fn buf(e: *Encoder, b: usize, off: usize, index: usize) void {
        e.current.buffers[index] = .{ .buf = b, .off = off };
    }
    pub fn bytes(e: *Encoder, value: anytype, index: usize) void {
        if (@TypeOf(value) == [5]i32) e.current.dims = value;
        if (@TypeOf(value) == u64 and index == 11) e.current.page_bytes = value;
    }
    pub fn run(e: *Encoder, grid: [3]usize, group: [3]usize) void {
        e.current.grid = grid;
        e.current.group = group;
        e.shots[e.count] = e.current;
        e.count += 1;
    }
    pub const setPipeline = pipe;
    pub const setBuffer = buf;
    pub fn setValue(e: *Encoder, value: anytype, index: usize) void {
        if (@TypeOf(value) == [8]i32) e.current.route[index - 2] = value;
        if (@TypeOf(value) == i32) e.current.pairs = value;
    }
    pub fn dispatchThreads(e: *Encoder, grid: Size, group: Size) void {
        e.run(.{ grid.width, grid.height, grid.depth }, .{ group.width, group.height, group.depth });
    }
};
const t = std.testing;
test "verify diagnostics: legacy window and serial one-row launches are different recorded arms" {
    for (1..65) |rows| for ([_]bool{ false, true }) |paged| {
        const call: decode.Call = .{ .head_dim = 128, .heads = 32, .kv_heads = 2, .qkv_n = 4608, .capacity = 4096, .len = 511, .first = 3, .rows = rows, .qkv = .{ .buf = 1, .off = 64 }, .qp = .{ .buf = 2 }, .k = .{ .buf = 3 }, .v = .{ .buf = 4, .off = 128 }, .po = .{ .buf = 5 }, .pm = .{ .buf = 6 }, .pl = .{ .buf = 7 }, .att = .{ .buf = 8 }, .out = .{ .buf = 9 }, .table = if (paged) .{ .buf = 10, .off = 64 } else null, .page_bytes = if (paged) 262144 else 0 };
        const pipelines: decode.Pipelines = .{ .q = 1, .partial = 2, .merge = 3, .out = 4 };
        var old: Encoder = .{};
        legacy.encode(&old, pipelines, call);
        var serial: Encoder = .{};
        decode.encode(&serial, pipelines, call);
        try t.expectEqual(@as(usize, 4), old.count);
        try t.expectEqual(4 * rows, serial.count);
        const partial = old.shots[1];
        const tiles: usize = @min(rows, 16);
        try t.expectEqual([5]i32{ @intCast(511 + rows), @intCast((511 + rows + 511) / 512), @intCast(rows), 1, @intCast(rows) }, partial.dims);
        try t.expectEqual([3]usize{ 64 * tiles, (511 + rows + 511) / 512, (rows + tiles - 1) / tiles }, partial.grid);
        try t.expectEqual([3]usize{ 32 * tiles, 1, 1 }, partial.group);
        try t.expectEqual(@as(usize, 64 + 3 * 4608 * 2), old.shots[0].buffers[0].off);
        for (0..rows) |r| try t.expectEqual([5]i32{ @intCast(512 + r), @intCast((512 + r + 511) / 512), 1, 1, 1 }, serial.shots[4 * r + 1].dims);
        if (paged) {
            try t.expectEqual(@as(usize, 10), partial.buffers[10].buf);
            try t.expectEqual(@as(u64, 262144), partial.page_bytes);
        }
    };
}
test "verify diagnostics: plain experts use COUNT=1 and row-local input IDs and output slices" {
    const F = struct {
        shape: @import("contract").spec.Shape,
        proj: cases.Proj,
        pub fn xRow(f: @This(), pair: usize) usize {
            return if (f.proj == .up) pair / 6 else pair;
        }
    };
    for ([_]cases.Proj{ .up, .down }) |proj| for ([_]usize{ 1, 2, 15, 16, 17, 64, 128 }) |rows| {
        const nk = cases.dims(proj);
        const P = struct { f: ?F, bufs: [9]Ref, plain_ids: Ref };
        var p: P = .{ .f = .{ .shape = .{ .rows = rows, .n = nk[0], .k = nk[1] }, .proj = proj }, .bufs = undefined, .plain_ids = .{ .buf = 10, .off = 64 } };
        for (&p.bufs, 1..) |*b, i| b.* = .{ .buf = i, .off = 64 };
        var e: Encoder = .{};
        plain.encode(&p, &e, @as(usize, 17), Ref{ .buf = 11, .off = 96 });
        try t.expectEqual(rows, e.count);
        for (e.shots[0..e.count], 0..) |shot, row| {
            try t.expectEqual(@as(usize, 17), shot.pipe);
            try t.expectEqual(64 + row * (if (proj == .up) @as(usize, 1) else 6) * nk[1] * 2, shot.buffers[0].off);
            try t.expectEqual(64 + row * 24, shot.buffers[1].off);
            try t.expectEqual(96 + row * 6 * nk[0] * 2, shot.buffers[9].off);
            try t.expectEqual([8]i32{ 1, 1, 1, 1, 1, 1, 0, 0 }, shot.route[1]);
            try t.expectEqual([8]i32{ 0, 1, 2, 3, 4, 5, 0, 0 }, shot.route[0]);
            try t.expectEqual(shot.route[0], shot.route[2]);
            try t.expectEqual(@as(i32, 6), shot.pairs);
            try t.expectEqual([3]usize{ 64, nk[0] / 8, 6 }, shot.grid);
        }
    };
}
test "verify diagnostics: every width and M1 forced geometry are selected independently of host generation" {
    const Visitor = struct {
        a: std.mem.Allocator,
        count: usize = 0,
        pub fn addPlain(v: *@This(), entry: cases.Entry, _: []const u8) !void {
            try t.expectEqualStrings(if (entry.proj == .up) "tf_xup_rows2r8" else "tf_xdown_rows2r8", entry.symbol);
            try t.expectEqual(@as(usize, 8), entry.rps);
            try t.expectEqual(@as(usize, 2), entry.sg);
            try t.expectEqual(@as(usize, 2), entry.mb);
            v.count += 1;
        }
    };
    var visitor: Visitor = .{ .a = t.allocator };
    try expert.visit(&visitor, "forced_m1_rows2r8");
    try t.expectEqual(@as(usize, 3), visitor.count);
    for ([_]cases.Proj{ .up, .down }) |proj| {
        var seen: [128]bool = @splat(false);
        for (expert.served(proj)) |s| {
            seen[s.rows - 1] = true;
            const nk = cases.dims(proj);
            try t.expectEqual(nk, [2]usize{ s.n, s.k });
        }
        for (seen[0..64]) |yes| try t.expect(yes);
        try t.expect(seen[126] and seen[127]);
        const w3 = expert.entry(proj, .w3, .none);
        try t.expectEqual(@as(usize, 4), w3.rps);
        try t.expectEqual(@as(usize, 4), w3.sg);
        try t.expectEqual(@as(usize, 3), w3.mb);
    }
}

test "verify diagnostics: legacy arm is explicitly selected and is not a planted control" {
    const attention = @import("tests/contract/nemotron_decode_cases.zig");
    try t.expect(!attention.legacyEnabled(null));
    try t.expect(!attention.legacyEnabled("nemotron verify attention"));
    try t.expect(attention.legacyEnabled("nemotron verify attention legacy-window bits"));
}

test "verify diagnostics: shared fixture groups keep arms comparable and preserve default seeds" {
    const gates = @import("contract").gates;
    for ([_]u64{ 0, 1, 0x6b63, 0x9152 }) |seed| {
        try t.expectEqual(std.hash.Wyhash.hash(seed, "existing kernel"), gates.inputSeed(seed, "existing kernel", null));
        try t.expectEqual(gates.inputSeed(seed, "legacy window", "attention"), gates.inputSeed(seed, "serial candidate", "attention"));
        try t.expect(gates.inputSeed(seed, "up w3", "up") != gates.inputSeed(seed, "up rows2r8", "down"));
    }
}

const selection = @import("tests/contract/metal_selection.zig");
const Inventory = struct {
    a: std.mem.Allocator = t.allocator,
    items: std.ArrayList(struct { control: ?bool }) = .empty,
    pub fn addDecode(v: *@This(), _: bool, _: bool, control: bool, _: []const u8) !void {
        try v.items.append(v.a, .{ .control = if (control) true else null });
    }
    pub fn addPlain(v: *@This(), entry: cases.Entry, _: []const u8) !void {
        try v.items.append(v.a, .{ .control = if (entry.fault == .none) null else true });
    }
};
test "verify diagnostics: explicit selection refuses zero registrations" {
    var inventory: Inventory = .{};
    defer inventory.items.deinit(inventory.a);
    try selection.validate(null, inventory.items.items);
    try t.expectError(error.EmptyExplicitSelection, selection.validate("typo", inventory.items.items));
    try t.expectError(error.EmptyExplicitSelection, selection.validate("forced_m1_rows2r8", inventory.items.items));
    try inventory.items.append(inventory.a, .{ .control = null });
    try selection.validate("unrelated selected kernel", inventory.items.items);
}
test "verify diagnostics: actual registration visitors require complete real and control inventories" {
    for (selection.diagnostics) |diagnostic| {
        var inventory: Inventory = .{};
        defer inventory.items.deinit(inventory.a);
        try @import("tests/contract/nemotron_decode_cases.zig").visit(&inventory, diagnostic.selector);
        try expert.visit(&inventory, diagnostic.selector);
        try selection.validate(diagnostic.selector, inventory.items.items);
        try t.expectEqual(diagnostic.real + diagnostic.controls, inventory.items.items.len);
        try t.expectError(error.IncompleteDiagnosticSelection, selection.validate(diagnostic.selector, inventory.items.items[0 .. inventory.items.items.len - 1]));
        const before = inventory.items.items[0].control;
        inventory.items.items[0].control = if (before == null) true else null;
        try t.expectError(error.IncompleteDiagnosticSelection, selection.validate(diagnostic.selector, inventory.items.items));
        inventory.items.items[0].control = before;
        try inventory.items.append(inventory.a, .{ .control = true });
        try t.expectError(error.IncompleteDiagnosticSelection, selection.validate(diagnostic.selector, inventory.items.items));
    }
}
