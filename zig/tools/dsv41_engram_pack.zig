//! `tf-dsv41-engram-pack`: the release's FP8 Engram tables as packed row shards, and a read-back check of them.
const std = @import("std");
const host = @import("dsv41");

const ep = host.engram_pack;

const usage =
    \\usage: tf-dsv41-engram-pack pack   RELEASE_DIR OUT_DIR [--layers 1,14] [--chunk-rows N]
    \\       tf-dsv41-engram-pack verify RELEASE_DIR OUT_DIR [--layers 1,14] [--samples N] [--seed S]
    \\RELEASE_DIR holds the release's safetensors shards and index; OUT_DIR gets engram-l<L>-r0of1.bin, the engram_dir
    \\of the server. pack streams each table (never holding it); verify decodes N random rows (default 100000, plus the
    \\first and last) of each shard through the server's reader and compares them with the release's own bytes.
    \\
;

fn progress(layer: u32, done: u64, total: u64) void {
    if (done == total or (done / ep.default_chunk) % 256 == 0) std.debug.print("layer {d}: {d} / {d} rows\n", .{ layer, done, total });
}

fn layerList(a: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |x| try out.append(a, try std.fmt.parseInt(u32, x, 10));
    return out.toOwnedSlice(a);
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4 or !(std.mem.eql(u8, args[1], "pack") or std.mem.eql(u8, args[1], "verify"))) {
        std.debug.print("{s}", .{usage});
        return error.Usage;
    }
    var layers: ?[]u32 = null;
    var chunk: u64 = ep.default_chunk;
    var samples: u64 = 100_000;
    var seed: u64 = 1;
    var i: usize = 4;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len) return error.Usage;
        const v = args[i + 1];
        if (std.mem.eql(u8, args[i], "--layers")) layers = try layerList(init.arena.allocator(), v) else if (std.mem.eql(u8, args[i], "--chunk-rows")) chunk = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, args[i], "--samples")) samples = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, args[i], "--seed")) seed = try std.fmt.parseInt(u64, v, 10) else return error.Usage;
    }
    var pack = try host.pack.Pack.open(a, io, args[2]);
    defer pack.deinit();
    const found = try ep.layersOf(a, &pack);
    defer a.free(found);
    const ls = layers orelse found;
    if (ls.len == 0) return error.NoEngramTables;
    if (std.mem.eql(u8, args[1], "pack")) {
        var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[3], .{});
        defer dir.close(io);
        for (ls) |l| {
            const rows = try ep.pack(a, io, &pack, dir, l, chunk, progress);
            std.debug.print("layer {d}: {d} rows packed\n", .{ l, rows });
        }
    }
    const v = try ep.verify(a, io, &pack, args[3], ls, samples, seed);
    std.debug.print("{s}: {d} rows read back through engram_rows, {d} differ\n", .{ if (v.bad == 0) "PASS" else "FAIL", v.rows, v.bad });
    if (v.bad != 0) return error.VerifyFailed;
}
