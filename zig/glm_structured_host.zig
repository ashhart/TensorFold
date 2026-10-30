//! Prompt redraw workflow with distinct owned rows and overwritten admission scratch.
const std = @import("std");
const glm = @import("glm");
const E = struct {
    bp: struct { logits: u64 },
    bd: struct { logits: u64 },
    share: struct { rank: usize = 0 } = .{},
    row: u16,
    projected: u16 = 0,
    projections: usize = 0,
    position: usize = 0,
    pub fn promptHead(e: *E) !void {
        e.projected = e.row;
        e.projections += 1;
        e.bd.logits = @intFromPtr(&e.projected);
    }
    pub fn drawWhole(e: *E, logits: u64, rows: usize, first: usize, bits: u64, out: []u32) !void {
        try std.testing.expectEqual(@as(usize, 1), rows);
        try std.testing.expectEqual(@as(u64, 1234), bits);
        e.position = first;
        if (e.share.rank != 0) return;
        const p: *const u16 = @ptrFromInt(logits);
        out[0] = p.*;
    }
};
test "GLM structured: redraw uses the bound prompt after later admissions overwrite scratch" {
    var scratch: u16 = 999;
    var e: E = .{ .bp = .{ .logits = @intFromPtr(&scratch) }, .bd = .{ .logits = 0 }, .row = 41 };
    try std.testing.expectEqual(@as(u32, 41), try glm.structured.redrawFirst(&e, 1025, 1234));
    scratch = 777;
    e.row = 93;
    try std.testing.expectEqual(@as(u32, 93), try glm.structured.redrawFirst(&e, 2051, 1234));
    try std.testing.expectEqual(@as(usize, 2051), e.position);
    try std.testing.expectEqual(@as(usize, 2), e.projections);
}

test "GLM structured: a follower redraw returns a defined unused token" {
    var scratch: u16 = 999;
    var e: E = .{ .bp = .{ .logits = @intFromPtr(&scratch) }, .bd = .{ .logits = 0 }, .row = 41, .share = .{ .rank = 1 } };
    try std.testing.expectEqual(@as(u32, 0), try glm.structured.redrawFirst(&e, 19, 1234));
    try std.testing.expectEqual(@as(usize, 1), e.projections);
}

test "GLM structured: sampling launches retain independent compiler receipts" {
    const registry = @import("core").contract.registry;
    const cases = .{ .{ "tf_draw", 1024 }, .{ "tf_logprob_rows", 1024 }, .{ "tf_grammar_mask", 256 }, .{ "tf_glm_top1", 1024 } };
    inline for (cases) |c| {
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, r.launch.symbol, c[0])) {
            found = true;
            try std.testing.expectEqual(@as(u32, c[1]), r.launch.threads);
            for (r.usage) |u| try std.testing.expect(u.regs > 0);
            try std.testing.expectEqualDeep(r.usage, registry.usageFor(r, .{ .major = 13, .minor = 0 }));
            var without_base = r;
            without_base.usage = registry.none;
            try std.testing.expectEqualDeep(registry.usageFor(without_base, .{ .major = 13, .minor = 3 }), registry.usageFor(r, .{ .major = 13, .minor = 3 }));
            without_base.launch.name = "sample no compiler receipt";
            for (registry.usageFor(without_base, .{ .major = 13, .minor = 3 })) |u| try std.testing.expectEqual(@as(u32, 0), u.regs);
        };
        try std.testing.expect(found);
    }
}

const cuda = @import("cuda");
var mapped: [1024]u8 align(16) = undefined;
var refuse_mapping = true;
var freed: usize = 0;
fn hostAlloc(p: *?*anyopaque, n: usize, _: c_uint) callconv(.c) cuda.abi.Result {
    if (n > mapped.len) return 2;
    p.* = &mapped;
    return 0;
}
fn hostFree(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
    freed += 1;
    return 0;
}
fn hostMap(p: *u64, _: ?*anyopaque, _: c_uint) callconv(.c) cuda.abi.Result {
    if (refuse_mapping) return cuda.abi.error_not_ready;
    p.* = @intFromPtr(&mapped);
    return 0;
}
test "GLM structured: failed mask mapping publishes no owner and can retry" {
    var d: cuda.Driver = undefined;
    d.api.cuMemHostAlloc = hostAlloc;
    d.api.cuMemFreeHost = hostFree;
    d.api.cuMemHostGetDevicePointer_v2 = hostMap;
    var e: glm.engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.d = &d;
    e.c.vocab = 64;
    var l: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer l.deinit();
    refuse_mapping = true;
    freed = 0;
    const before = cuda.memory.usage(false).host;
    try std.testing.expectError(error.NotReady, l.putBits(0, 1, &.{ 7, 9 }));
    try std.testing.expect(l.bits == null);
    try std.testing.expectEqual(@as(usize, 1), freed);
    try std.testing.expectEqual(before, cuda.memory.usage(false).host);
    refuse_mapping = false;
    try std.testing.expect(try l.putBits(0, 1, &.{ 7, 9 }) != 0);
    try std.testing.expectEqualSlices(u32, &.{ 7, 9 }, l.bits.?.slice(u32)[0..2]);
}
