//! The flashnext prompt contract's host-visible declarations: the served row range and the state alias.
const std = @import("std");
const core = @import("core");
const flash = @import("flashnext");
const glue = @import("contract_flashnext_glue.zig");

const contract = core.contract;
const c = contract.spec;
const parts = flash.batch_state;

test "the flashnext prompt specs serve the admitted row range with the dispatch and tile seams" {
    for ([_]glue.Kind{ .chain, .fused }) |kind| {
        const s = glue.spec(kind);
        var served: [3]bool = @splat(false);
        for (s.served) |x| for ([_]usize{ 64, parts.default_chunk, parts.max_chunk }, 0..) |want, i| {
            if (x.rows == want) served[i] = true;
        };
        for (served) |yes| try std.testing.expect(yes);
        const list = try c.shapes(std.testing.allocator, s);
        defer std.testing.allocator.free(list);
        var seams: [4]bool = @splat(false);
        for (list) |x| for ([_]usize{ 1, 15, 16, 17 }, 0..) |want, i| {
            if (x.rows == want) seams[i] = true;
        };
        for (seams) |yes| try std.testing.expect(yes);
    }
}

test "the fused block declares the state alias and binds one buffer for it" {
    try std.testing.expect(glue.spec(.fused).in_place);
    try std.testing.expect(!glue.spec(.chain).in_place); // the chain's kernel marks its state input __restrict__
    if (@hasDecl(glue, "fusedArgs")) {
        var ins: [7]u64 = .{ 1, 2, 3, 4, 5, 6, 7 };
        var o: [4]u64 = .{ 8, 9, 10, 11 };
        var a = glue.fusedArgs(&ins, &o, 64, 12);
        try std.testing.expectEqual(a.integer(5), a.integer(6)); // the state in and the state out are one buffer
        try std.testing.expectEqual(@as(?u64, 10), a.integer(5));
        try std.testing.expectEqual(@as(?u64, 64), a.integer(7));
    } else return error.MissingFusedArgs;
}

const lay = @import("flashnext_gdn_heads.zig");
const heads = @import("contract_flashnext_heads.zig");

test "a rank's DeltaNet heads keep the design's rows of the full layout, and the two ranks cover it once" {
    const want = [_][2]usize{ .{ 0, 1024 }, .{ 2048, 1024 }, .{ 4096, 3072 }, .{ 10240, 3072 }, .{ 16384, 24 }, .{ 16432, 24 } };
    const step = [_]usize{ 1024, 1024, 3072, 3072, 24, 24 };
    for (0..2) |r| {
        const cs = lay.cuts(.proj, .half, r);
        try std.testing.expectEqual(want.len, cs.slice().len);
        for (cs.slice(), want, step) |got, w, s| try std.testing.expectEqual(lay.Cut{ .from = w[0] + s * r, .len = w[1] }, got);
    }
    try std.testing.expectEqual(@as(usize, 8240), lay.width(.proj, .half));
    for (std.enums.values(lay.Part)) |part| {
        const n = lay.width(part, .full);
        const seen = try std.testing.allocator.alloc(u8, n);
        defer std.testing.allocator.free(seen);
        @memset(seen, 0);
        var local: usize = 0;
        for (0..2) |r| {
            const cs = lay.cuts(part, .half, r);
            for (cs.slice()) |x| for (seen[x.from..][0..x.len]) |*v| {
                v.* += 1;
            };
            for (cs.slice()) |x| local += x.len;
        }
        var at: usize = 0; // the full heads' one rank keeps every element in place
        for (lay.cuts(part, .full, 0).slice()) |x| {
            try std.testing.expectEqual(at, x.from);
            at += x.len;
        }
        try std.testing.expectEqual(n, at);
        try std.testing.expectEqual(2 * lay.width(part, .half), local);
        for (seen) |v| try std.testing.expectEqual(@as(u8, 1), v);
    }
}

test "the gather copies each kept element from where fullIndex says it lies" {
    const a = std.testing.allocator;
    const rows = 3;
    const n = rows * lay.width(.proj, .full);
    const src = try a.alloc(u32, n);
    defer a.free(src);
    for (src, 0..) |*v, i| v.* = @intCast(i);
    for (0..2) |r| {
        const got = try lay.gather(a, std.mem.sliceAsBytes(src), 4, .proj, .half, r);
        defer a.free(got);
        const vals = std.mem.bytesAsSlice(u32, got);
        const lw = lay.width(.proj, .half);
        try std.testing.expectEqual(rows * lw, vals.len);
        for (0..rows) |row| for (0..lw) |i| {
            try std.testing.expectEqual(row * lay.width(.proj, .full) + lay.fullIndex(.proj, .half, r, i), vals[row * lw + i]);
        };
    }
    const same = try lay.gather(a, std.mem.sliceAsBytes(src), 4, .proj, .full, 0);
    defer a.free(same);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(src), same);
}

test "each DeltaNet twin is a candidate on both ranks, replacing its full original, at the served rows and seams" {
    for (std.enums.values(heads.Kind)) |kind| for (0..2) |r| {
        const s = heads.spec(kind, r);
        try std.testing.expect(std.mem.indexOf(u8, s.name, lay.symbol(kind, true)) != null);
        try std.testing.expect(std.mem.startsWith(u8, s.replaces.?, lay.symbol(kind, false)));
        try std.testing.expectEqual(kind == .rows or kind == .fused, s.in_place);
        var served: [3]bool = @splat(false);
        for (s.served) |x| for ([_]usize{ 64, parts.default_chunk, parts.max_chunk }, 0..) |want, i| {
            if (x.rows == want and x.n == lay.Heads.half.valueDim()) served[i] = true;
        };
        for (served) |yes| try std.testing.expect(yes);
        const list = try c.shapes(std.testing.allocator, s);
        defer std.testing.allocator.free(list);
        var seams: [4]bool = @splat(false);
        for (list) |x| for ([_]usize{ 1, 15, 16, 17 }, 0..) |want, i| {
            if (x.rows == want) seams[i] = true;
        };
        for (seams) |yes| try std.testing.expect(yes);
    };
    try std.testing.expect(!std.mem.eql(u8, heads.spec(.front, 0).name, heads.spec(.front, 1).name));
}
