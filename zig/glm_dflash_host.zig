//! DFlash allocation boundaries and the production loader's partial owner, without a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const t = std.testing;

test "glm dflash: gather and read fit for every supported rank count" {
    for (1..17) |world| for ([_]usize{ 1, 8, glm.dflash.MAX_M }) |rows| {
        const n = glm.dflash_candidates.Layout.init(world, rows, glm.dflash.CW);
        const gathered = world * rows * glm.dflash.CW;
        try t.expect(gathered <= n.device);
        try t.expect(n.read <= n.host);
        try t.expectEqual(gathered, n.read);
    };
}

const Resources = struct {
    pred: []u16 = &.{},
    succ: []u16 = &.{},
    scratch: []u8 = &.{},
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{},
    gh: []f32 = &.{},
};
const Ops = struct {
    g: std.mem.Allocator,
    fail_read: ?usize = null,
    pub fn readBook(o: *Ops, i: usize, dst: []u16) !void {
        if (o.fail_read == i) return error.BadBook;
        @memset(dst, 7);
    }
    pub fn scratch(o: *Ops, r: *Resources) !void {
        r.scratch = try o.g.alloc(u8, 64);
    }
    pub fn freeScratch(o: *Ops, r: *Resources) void {
        o.g.free(r.scratch);
    }
};
fn release(g: std.mem.Allocator, r: Resources) void {
    g.free(r.pred);
    g.free(r.succ);
    g.free(r.scratch);
    g.free(r.h_cpos);
    g.free(r.h_crings);
    g.free(r.cand);
    g.free(r.gh);
}
fn attempt(fail_index: usize, fail_read: ?usize) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index });
    const g = f.allocator();
    var r: Resources = .{};
    var o: Ops = .{ .g = g, .fail_read = fail_read };
    if (glm.dflash_load_owner.acquire(g, &r, &o, Ops, .{ .book = 16, .ctx = 8, .cand = 32, .hidden = 16 })) |_| {
        try t.expect(fail_index >= 7 and fail_read == null);
        try t.expect(f.allocated_bytes > f.freed_bytes);
        release(g, r);
    } else |err| {
        try t.expect(err == error.OutOfMemory or err == error.BadBook);
    }
    try t.expectEqual(f.allocated_bytes, f.freed_bytes);
}

test "glm dflash: every allocation failure releases earlier tail resources" {
    for (0..8) |i| try attempt(i, null);
}

test "glm dflash: either book read failure releases its allocated storage" {
    try attempt(99, 0);
    try attempt(99, 1);
}

test "glm dflash: every loaded symbol has a measured launch record" {
    inline for (@typeInfo(glm.dflash.Kernels).@"struct".field_names) |name| {
        const symbol = "df_" ++ name;
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            try t.expectEqualStrings("glm_dflash", r.launch.image);
            for (r.usage) |u| try t.expect(u.regs > 0);
        };
        try t.expect(found);
    }
}
