//! Actual packed-lane output and cache identity compared with the preceding gather and padding recipes.
const std = @import("std");
const pack = @import("src/families/flashnext/pack.zig");
const pio = @import("src/families/flashnext/pack_io.zig");
const pc = @import("src/families/flashnext/pack_cache.zig");
const pv = @import("src/core/proposal_vocab.zig");
const old = @import("flashnext_proposal_oracle.zig");
const table = @import("src/families/flashnext/proposal_table.zig").text;
const t = std.testing;
const gpa = t.allocator;
test "Flash Next proposal: original padded support pack bytes and cache fingerprints stay equal" {
    const before = try old.paddedIds(gpa, table, 248320);
    defer gpa.free(before);
    const after = try pv.legacy.ids(gpa, table, .{ .vocab = 248320, .tile = 64 });
    defer gpa.free(after);
    try t.expectEqual(@as(usize, 79616), after.len);
    try t.expectEqualSlices(u32, before, after);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]u8{ 4, 6, 7, 8 }) |bits| {
        const kw = 64 * @as(usize, bits) / 32;
        const codes = try a.alloc(u32, 248320 * kw);
        for (codes, 0..) |*v, i| v.* = @truncate(i *% 113);
        const scales = try a.alloc(u16, 248320 * 2);
        const biases = try a.alloc(u16, scales.len);
        for (scales, biases, 0..) |*s, *b, i| {
            s.* = @truncate(i *% 71);
            b.* = @truncate(i *% 97);
        }
        const head: pack.Q = .{ .w = .{ .dtype = .u32, .rank = 2, .shape = .{ 248320, kw, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(codes) }, .s = .{ .dtype = .bf16, .rank = 2, .shape = .{ 248320, 2, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(scales) }, .b = .{ .dtype = .bf16, .rank = 2, .shape = .{ 248320, 2, 0, 0, 0 }, .bytes = std.mem.sliceAsBytes(biases) }, .rows = 248320, .k = 64, .kw = kw, .spec = .{ .bits = bits, .group = 32 } };
        var prior = pack.Out.init(gpa);
        defer prior.deinit();
        var next = pack.Out.init(gpa);
        defer next.deinit();
        try old.putDraftHead(pack, &prior, gpa, a, "mtp.draft", head, before);
        try pack.putDraftHead(&next, gpa, a, "mtp.draft", head, after);
        try t.expectEqual(prior.blobs.items.len, next.blobs.items.len);
        for (prior.names.items, next.names.items, prior.dtypes.items, next.dtypes.items, prior.shapes.items, next.shapes.items, prior.blobs.items, next.blobs.items) |pn, nn, pd, nd, ps, ns, pb, nb| {
            try t.expectEqualStrings(pn, nn);
            try t.expectEqualStrings(pd, nd);
            try t.expectEqualSlices(usize, ps, ns);
            try t.expectEqualSlices(u8, pb, nb);
        }
        var tmp = t.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
        defer gpa.free(dir);
        const pa = try std.fs.path.join(gpa, &.{ dir, "old.safetensors" });
        defer gpa.free(pa);
        const pb = try std.fs.path.join(gpa, &.{ dir, "new.safetensors" });
        defer gpa.free(pb);
        const identity = try pio.identityString(gpa, @as([32]u8, @splat(1)), @as([32]u8, @splat(2)), &.{});
        defer gpa.free(identity);
        try prior.write(gpa, t.io, pa, identity);
        try next.write(gpa, t.io, pb, identity);
        const old_file = try tmp.dir.readFileAlloc(t.io, "old.safetensors", gpa, .limited(1 << 28));
        defer gpa.free(old_file);
        const new_file = try tmp.dir.readFileAlloc(t.io, "new.safetensors", gpa, .limited(1 << 28));
        defer gpa.free(new_file);
        try t.expectEqualSlices(u8, old_file, new_file);
        const key = try pc.directory(gpa, "/example/packs", identity);
        defer gpa.free(key);
        const expected = try std.fs.path.join(gpa, &.{ "/example/packs", &std.fmt.bytesToHex(pio.hashBytes(identity), .lower) });
        defer gpa.free(expected);
        try t.expectEqualStrings(expected, key);
    }
}
