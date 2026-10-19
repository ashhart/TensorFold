//! Several streams' drafted windows in one shared verify pass against each window alone, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const mtp = flash.mtp;

/// Window widths in one shared pass: none past 16 rows in all, past 16 (the router's few rows), 32, and 64.
const shapes = [_][]const usize{ &.{ 1, 1, 1, 1 }, &.{ 5, 5, 5, 5 }, &.{ 1, 16, 3, 7 }, &.{ 16, 16 }, &.{ 16, 16, 16, 16 } };

const Side = struct {
    seqs: [4]prompt.Seq = undefined,
    drafts: [4]mtp.Draft = undefined,
    made: usize = 0,

    fn deinit(s: *Side, gpa: std.mem.Allocator) void {
        for (s.seqs[0..s.made], s.drafts[0..s.made]) |*q, *d| {
            d.deinit();
            q.deinit(gpa);
        }
    }
};

pub fn run(gpu: check.Gpu, dir: []const u8, ids_path: []const u8) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var base: std.ArrayList(u32) = .empty;
    defer base.deinit(gpu.gpa);
    var words = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (words.next()) |word| try base.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (base.items.len < 16) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    const e = try prompt.Engine.init(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir);
    defer e.deinit();
    const ops: mtp.Ops(flash.Tri) = .{ .e = e, .tri = tri, .h = undefined };
    const row_bytes = prompt.vocabulary * 2;
    var shared = try cuda.DeviceBuffer.alloc(gpu.d, 64 * row_bytes);
    defer shared.free();
    // a short prompt (dense attention) and one past 2,048 keys (the block indexer's sparse selection)
    for ([_]usize{ @min(base.items.len, 100), 2600 }) |length| {
        const ids = try gpu.gpa.alloc(u32, length);
        defer gpu.gpa.free(ids);
        for (ids, 0..) |*t, i| t.* = base.items[i % base.items.len];
        for (shapes) |widths| try shape(gpu, e, ops, &stream, shared.ptr, ids, widths);
    }
}

fn shape(gpu: check.Gpu, e: *prompt.Engine, ops: mtp.Ops(flash.Tri), stream: *cuda.Stream, shared: u64, ids: []const u32, widths: []const usize) !void {
    const gpa = gpu.gpa;
    const row_bytes = prompt.vocabulary * 2;
    const cap = ids.len + 64;
    var alone: Side = .{};
    defer alone.deinit(gpa);
    var together: Side = .{};
    defer together.deinit(gpa);
    var windows: [4][16]u32 = undefined;
    var total: usize = 0;
    for (widths, 0..) |w, i| {
        // each stream its own prompt length (so its own positions) and its own forced window
        const prompt_len = ids.len - 3 * i;
        for (windows[i][0..w], 0..) |*t, r| t.* = @intCast((37 + 101 * i + 17 * r) % 50_000);
        inline for (.{ &alone, &together }) |side| {
            side.seqs[i] = try e.newSeq(cap);
            side.drafts[i] = mtp.Draft.init(gpa, e.driver, stream.*, cap, 1, e.chunk) catch |err| {
                side.seqs[i].deinit(gpa);
                return err;
            };
            side.made += 1;
            try e.forward(flash.Tri, ops.tri, &side.seqs[i], ids[0..prompt_len], .{});
        }
        total += w;
    }
    const hidden = alone.drafts[0].streams.len / alone.drafts[0].rows_cap;
    // each window alone, its rows' scores and hidden rows read back
    const want_logits = try gpa.alloc(u8, total * row_bytes);
    defer gpa.free(want_logits);
    const want_hidden = try gpa.alloc(u8, total * hidden);
    defer gpa.free(want_hidden);
    var row0: usize = 0;
    for (widths, 0..) |w, i| {
        try ops.verify(&alone.seqs[i], &alone.drafts[i], windows[i][0..w]);
        try stream.synchronize();
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(want_logits[row0 * row_bytes ..].ptr, alone.drafts[i].logits.ptr, w * row_bytes), "cuMemcpyDtoH");
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(want_hidden[row0 * hidden ..].ptr, alone.drafts[i].streams.ptr, w * hidden), "cuMemcpyDtoH");
        row0 += w;
    }
    // the same windows in one shared pass
    var seqs: [4]*prompt.Seq = undefined;
    var ds: [4]*mtp.Draft = undefined;
    var idss: [4][]const u32 = undefined;
    for (widths, 0..) |w, i| {
        seqs[i] = &together.seqs[i];
        ds[i] = &together.drafts[i];
        idss[i] = windows[i][0..w];
    }
    const n = widths.len;
    try ops.verifyMany(seqs[0..n], ds[0..n], idss[0..n], shared);
    try stream.synchronize();
    const got = try gpa.alloc(u8, @max(row_bytes, hidden) * 16);
    defer gpa.free(got);
    row0 = 0;
    for (widths, 0..) |w, i| {
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(got.ptr, together.drafts[i].logits.ptr, w * row_bytes), "cuMemcpyDtoH");
        try check.sameBytes("a shared window's scores equal its lone pass's", got[0 .. w * row_bytes], want_logits[row0 * row_bytes ..][0 .. w * row_bytes]);
        try e.driver.check(e.driver.api.cuMemcpyDtoH_v2(got.ptr, together.drafts[i].streams.ptr, w * hidden), "cuMemcpyDtoH");
        try check.sameBytes("a shared window's hidden rows equal its lone pass's", got[0 .. w * hidden], want_hidden[row0 * hidden ..][0 .. w * hidden]);
        try check.sameBytes("a shared window's last host row equals its lone pass's", together.seqs[i].logits, alone.seqs[i].logits);
        row0 += w;
    }
    // every stream keeps part of its window (its rollback replays the kept rows), then one more token
    for (widths, 0..) |w, i| {
        const keep = 1 + (3 * i) % w;
        try ops.rollback(&alone.seqs[i], &alone.drafts[i], keep);
        try ops.rollback(&together.seqs[i], &together.drafts[i], keep);
        try check.expect(alone.seqs[i].fed == together.seqs[i].fed, "kept positions match", .{});
        try check.expect(std.mem.eql(i64, &alone.seqs[i].history, &together.seqs[i].history), "kept n-gram histories match", .{});
        try e.forward(flash.Tri, ops.tri, &alone.seqs[i], &.{13}, .{});
        try e.forward(flash.Tri, ops.tri, &together.seqs[i], &.{13}, .{});
        try check.sameBytes("the row after a shared window's rollback equals the lone one's", together.seqs[i].logits, alone.seqs[i].logits);
    }
    check.pass("shared verify, {d} windows of {any} rows ({d} in all) at {d} prompt tokens: scores, hidden rows, rollback and next row byte-exact", .{ n, widths, total, ids.len });
}
