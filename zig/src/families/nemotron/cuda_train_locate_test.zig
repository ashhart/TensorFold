//! Exercise the actual causal row launcher on host pointers, with the independent CPU norm reference and poisoned rows.
const std = @import("std");
const row = @import("cuda_train_locate.zig");
const ref = @import("glue_ref.zig");
const t = std.testing;
const d = 64;
const rows = 4;
const Kind = enum { moe, mamba, attention };
const Block = struct { norm: u64, kind: Kind, mamba: struct { in_proj: u64 = 0 } = .{}, attn: struct { qkv: u64 = 0 } = .{} };
const Ops = struct {
    projected: *usize,
    pub fn copy(_: @This(), dst: u64, src: u64, n: usize) !void {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
    }
    pub fn prefillDense(o: @This(), _: u64, _: u64, _: u64, n: usize) !void {
        try t.expectEqual(@as(usize, rows), n);
        o.projected.* += 1;
    }
};
const Engine = struct {
    w: struct { blocks: [2]Block },
    projected: usize = 0,
    pub fn ops(e: *@This()) Ops {
        return .{ .projected = &e.projected };
    }
};
const Toy = struct {
    c: struct { hidden: usize = d, eps: f32 = 1e-5 } = .{},
    locate: struct { layer: ?usize = 0, row: usize = 1 } = .{},
    e: *Engine,
    before: *[rows * d]u16,
    pub fn savedAt(m: *@This(), _: usize) u64 {
        return @intFromPtr(m.before);
    }
};
const Glue = struct {
    pub fn addRmsnorm(_: @This(), x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, n: usize, hidden: usize, eps: f32) !void {
        try t.expect(r == null);
        try t.expectEqual(@as(usize, 1), n);
        try ref.addRmsnorm(t.allocator, @as([*]const u16, @ptrFromInt(x))[0..hidden], null, @as([*]const u16, @ptrFromInt(w))[0..hidden], @as([*]u16, @ptrFromInt(h))[0..hidden], @as([*]u16, @ptrFromInt(y))[0..hidden], @as([*]f32, @ptrFromInt(xs))[0 .. hidden / 64], n, hidden, eps);
    }
};
test "causal CUDA: only the subject residual and norm row are replaced, with the same projection batch" {
    for ([_]Kind{ .moe, .mamba, .attention }) |kind| {
        var before: [rows * d]u16 = @splat(0x3f80);
        var h: [rows * d]u16 = @splat(0x4000);
        var y: [rows * d]u16 = @splat(0xdead);
        var xs: [rows]f32 = @splat(-99);
        const norm: [d]u16 = @splat(0x3f80);
        var engine: Engine = .{ .w = .{ .blocks = @splat(.{ .norm = @intFromPtr(&norm), .kind = kind }) } };
        var model: Toy = .{ .e = &engine, .before = &before };
        var walk: struct { x: u64, rows: usize } = .{ .x = @intFromPtr(&h), .rows = rows };
        const buffers = .{ .y = @intFromPtr(&y), .xs = @intFromPtr(&xs), .proj = @as(u64, 0), .qkv = @as(u64, 0) };
        const forward = .{ .b = &buffers, .glue = Glue{} };
        try row.patch(&model, &forward, &walk, 1);
        for (0..rows) |r| {
            try t.expect(std.mem.allEqual(u16, h[r * d ..][0..d], if (r == 1) 0x3f80 else 0x4000));
            if (r != 1) {
                try t.expect(std.mem.allEqual(u16, y[r * d ..][0..d], 0xdead));
                try t.expectEqual(@as(f32, -99), xs[r]);
            }
        }
        try t.expect(xs[1] != -99);
        try t.expect(std.mem.allEqual(u16, y[d..][0..d], 0x3f80));
        try t.expectEqual(@as(usize, if (kind == .moe) 0 else 1), engine.projected);
        const hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(&h));
        model.locate.layer = null;
        try row.patch(&model, &forward, &walk, 1);
        try t.expectEqual(hash, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(&h)));
    }
}
