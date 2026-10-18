//! fn_qsa_select (flashnext_batch.cu) against the lists Triton's _select and _select_tiles made (capture_sparse.sh).

const std = @import("std");
const cuda = @import("cuda");

const Case = struct { name: []const u8, rows: usize, pos0: i32, nb: usize, idw: usize };

fn readAll(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, ext: []const u8, len: usize) ![]u8 {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.{s}", .{ dir, name, ext });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    if (bytes.len != len) return error.BadFixture;
    return bytes;
}

fn int(bytes: []const u8, i: usize) i32 {
    return std.mem.readInt(i32, bytes[4 * i ..][0..4], .little);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: flashnext-select <fixtures-dir>\n", .{});
        return 2;
    }
    const dir = args[1];
    const index = try std.fmt.allocPrint(gpa, "{s}/cases.json", .{dir});
    defer gpa.free(index);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice([]Case, gpa, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var module = try cuda.Module.load(&driver, cuda.kernels.flashnext_batch);
    defer module.unload();
    const select = try module.function("fn_qsa_select");

    var failed = false;
    for (parsed.value) |c| {
        const scores = try readAll(gpa, io, dir, c.name, "scores", c.rows * c.nb * 4);
        defer gpa.free(scores);
        const want_ids = try readAll(gpa, io, dir, c.name, "ids", c.rows * c.idw * 4);
        defer gpa.free(want_ids);
        const want_nk = try readAll(gpa, io, dir, c.name, "nk", c.rows * 4);
        defer gpa.free(want_nk);
        const want_sp = try readAll(gpa, io, dir, c.name, "sparse", c.rows * 4);
        defer gpa.free(want_sp);

        var sc_b = try cuda.DeviceBuffer.fromHost(&driver, scores);
        defer sc_b.free();
        var pos_b = try cuda.DeviceBuffer.fromHost(&driver, std.mem.asBytes(&c.pos0));
        defer pos_b.free();
        var ids_b = try cuda.DeviceBuffer.alloc(&driver, c.rows * c.idw * 4);
        defer ids_b.free();
        try ids_b.fill8(0xFF, null);
        var nk_b = try cuda.DeviceBuffer.alloc(&driver, c.rows * 4);
        defer nk_b.free();
        var sp_b = try cuda.DeviceBuffer.alloc(&driver, c.rows * 4);
        defer sp_b.free();
        var a: cuda.Args = .{};
        a.add(sc_b.ptr);
        a.add(pos_b.ptr);
        a.add(ids_b.ptr);
        a.add(nk_b.ptr);
        a.add(sp_b.ptr);
        a.add(@as(c_int, @intCast(c.nb)));
        a.add(@as(c_int, @intCast(c.idw)));
        a.add(@as(c_int, 512));
        a.add(@as(c_int, 4));
        const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
        try cuda.launch.launch(select, .{ .grid = .{ .x = @intCast(c.rows) }, .block = .{ .x = 1024 } }, stream, &a);
        try stream.synchronize();
        const us = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e3;
        const got_ids = try gpa.alloc(u8, c.rows * c.idw * 4);
        defer gpa.free(got_ids);
        const got_nk = try gpa.alloc(u8, c.rows * 4);
        defer gpa.free(got_nk);
        const got_sp = try gpa.alloc(u8, c.rows * 4);
        defer gpa.free(got_sp);
        try ids_b.download(0, got_ids);
        try nk_b.download(0, got_nk);
        try sp_b.download(0, got_sp);
        var bad: usize = 0;
        var sparse_rows: usize = 0;
        for (0..c.rows) |r| {
            const nk = int(want_nk, r);
            if (int(got_nk, r) != nk or int(got_sp, r) != int(want_sp, r)) {
                bad += 1;
                continue;
            }
            if (int(want_sp, r) == 0) continue;
            sparse_rows += 1;
            for (0..@intCast(nk)) |i| {
                if (int(got_ids, r * c.idw + i) != int(want_ids, r * c.idw + i)) {
                    bad += 1;
                    break;
                }
            }
        }
        std.debug.print("{s}: {d} rows at {d}, {d} blocks a row, {d} sparse: {d} differ ({d:.0} us)\n", .{ c.name, c.rows, c.pos0, c.nb, sparse_rows, bad, us });
        if (bad != 0) failed = true;
    }
    std.debug.print("{s}\n", .{if (failed) "differ" else "match"});
    return if (failed) 1 else 0;
}
