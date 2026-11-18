//! `tf-qwen35-cuda`'s readout rows: a prompt's final-norm rows piece by piece (keep = .rows), and their checks.
const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen35");
const E = q.engine;

/// `<model> rows <ids file> <out> [piece]`, `<model> rows-last <ids file>`, `<model> rows-cuts <ids file>`.
pub fn run(init: std.process.Init, a: std.mem.Allocator, args: []const [:0]const u8) !u8 {
    const mode = args[2];
    const rows = std.mem.eql(u8, mode, "rows");
    if (!rows and !std.mem.eql(u8, mode, "rows-last") and !std.mem.eql(u8, mode, "rows-cuts")) return 2;
    if (if (rows) args.len != 5 and args.len != 6 else args.len != 4) return 2;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(64 << 20));
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try ids.append(a, try std.fmt.parseInt(u32, t, 10));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const e = try E.Engine.init(init.gpa, init.io, &ctx, args[1], false, null);
    defer e.deinit();
    if (std.mem.eql(u8, mode, "rows-last")) return last(init.gpa, e, ids.items);
    if (std.mem.eql(u8, mode, "rows-cuts")) {
        try cuts(init.gpa, e, ids.items);
        return 0;
    }
    const piece = if (args.len == 6) try std.fmt.parseInt(usize, args[5], 10) else 1024;
    try write(init.gpa, init.io, e, ids.items, args[4], piece);
    return 0;
}

/// The final-norm rows of `ids` in pieces of `piece` rows, as BF16 bits, row after row.
pub fn collect(gpa: std.mem.Allocator, e: *E.Engine, ids: []const u32, piece: usize) ![]u16 {
    if (piece == 0 or piece > E.max_rows) return error.InvalidQwenPiece;
    var cache = try E.Cache.init(gpa, e, ids.len);
    defer cache.deinit(gpa);
    const w = e.g.hidden;
    const out = try gpa.alloc(u16, ids.len * w);
    errdefer gpa.free(out);
    errdefer cuda.completion.drain(e.d, e.stream, "readout rows did not drain"); // before the cache and rows go
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(piece, ids.len - at);
        try e.forward(&.{.{ .cache = &cache, .rows = n }}, ids[at..][0..n], .rows);
        try copy(e, out[at * w ..][0 .. n * w], 0);
        at += n;
    }
    return out;
}

/// `dst.len` BF16 values of the engine's `x` from row `row` on, once the stream's work is done.
fn copy(e: *E.Engine, dst: []u16, row: usize) !void {
    const src = e.x + row * e.g.hidden * 2;
    try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(dst.ptr), src, dst.len * 2, e.stream.handle), "cuMemcpyDtoHAsync");
    try e.stream.synchronize();
}

/// `rows <ids> <out> [piece]`: the rows to `out` (raw BF16, `hidden` a row) for the backbone gate.
pub fn write(gpa: std.mem.Allocator, io: std.Io, e: *E.Engine, ids: []const u32, path: []const u8, piece: usize) !void {
    const rows = try collect(gpa, e, ids, piece);
    defer gpa.free(rows);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(rows) });
    std.debug.print("rows: {d} of {d} values in pieces of {d} to {s}\n", .{ ids.len, e.g.hidden, piece, path });
}

/// `rows-last <ids>`: one prompt chunk kept as `.last` and as `.rows` leaves the same final-norm last row; 1 if not.
pub fn last(gpa: std.mem.Allocator, e: *E.Engine, ids: []const u32) !u8 {
    if (ids.len == 0 or ids.len > E.max_rows) return error.InvalidQwenPiece;
    const w = e.g.hidden;
    const kept = try gpa.alloc(u16, w);
    defer gpa.free(kept);
    {
        var cache = try E.Cache.init(gpa, e, ids.len);
        defer cache.deinit(gpa);
        try e.forward(&.{.{ .cache = &cache, .rows = ids.len }}, ids, .last);
        try copy(e, kept, ids.len - 1);
    }
    const rows = try collect(gpa, e, ids, ids.len);
    defer gpa.free(rows);
    const same = std.mem.eql(u16, kept, rows[(ids.len - 1) * w ..][0..w]);
    std.debug.print("{s} rows-last: the last row of {d} kept as .rows {s} the .last chunk's final-norm row\n", .{ if (same) "PASS" else "FAIL", ids.len, if (same) "equals" else "differs from" });
    return if (same) 0 else 1;
}

/// `rows-cuts <ids>`: each piece size against one whole piece, row by row; a report (qmmf's row buckets may differ).
pub fn cuts(gpa: std.mem.Allocator, e: *E.Engine, ids: []const u32) !void {
    const n = @min(ids.len, 1024);
    const w = e.g.hidden;
    const whole = try collect(gpa, e, ids[0..n], n);
    defer gpa.free(whole);
    for ([_]usize{ 16, 17, 32, 33, 64, 65, 255, 256, 512 }) |piece| {
        const got = try collect(gpa, e, ids[0..n], piece);
        defer gpa.free(got);
        var differ: usize = 0;
        for (0..n) |r| differ += @intFromBool(!std.mem.eql(u16, got[r * w ..][0..w], whole[r * w ..][0..w]));
        std.debug.print("rows-cuts: pieces of {d}: {d} of {d} rows differ from one piece of {d}\n", .{ piece, differ, n, n });
    }
}
