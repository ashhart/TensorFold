//! The release's FP8 Engram tables as the packed row shards `engram_rows` reads, streamed; and a read-back check.
const std = @import("std");
const er = @import("engram_rows.zig");
const Pack = @import("pack.zig").Pack;
const Info = @import("pack.zig").Info;
pub const Plan = @import("engram_tables.zig").Plan;

const Io = std.Io;

pub const default_chunk: u64 = 1 << 16;
const weight_suffix = ".engram.embed.weight";

/// The shard's file name for one rank of one world: this packer writes the whole table as rank 0 of 1.
pub fn shardName(buf: []u8, layer: u32) ![]u8 {
    return std.fmt.bufPrint(buf, "engram-l{d}-r0of1.bin", .{layer});
}

/// The shard's header block: magic, layer, first row, end row, total rows, row bytes, then zeros to 4096.
pub fn header(layer: u32, rows: u64) [er.header_bytes]u8 {
    var b: [er.header_bytes]u8 = @splat(0);
    for ([_]u64{ er.magic, layer, 0, rows, rows, er.row_bytes }, 0..) |v, i| std.mem.writeInt(u64, b[8 * i ..][0..8], v, .little);
    return b;
}

/// The release's Engram layers, ascending: every `layers.N.engram.embed.weight` the pack holds.
pub fn layersOf(a: std.mem.Allocator, p: *const Pack) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var it = p.names.iterator();
    while (it.next()) |e| {
        const n = e.key_ptr.*;
        if (!std.mem.startsWith(u8, n, "layers.") or !std.mem.endsWith(u8, n, weight_suffix)) continue;
        const id = n["layers.".len .. n.len - weight_suffix.len];
        try out.append(a, std.fmt.parseInt(u32, id, 10) catch continue);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    return out.toOwnedSlice(a);
}

/// The table of `layer`: its E4M3 codes [rows][256] and E8M0 scales [rows][8], shapes checked.
pub fn tableOf(p: *const Pack, layer: u32) !Plan {
    var buf: [96]u8 = undefined;
    const w = try p.need(try std.fmt.bufPrint(&buf, "layers.{d}{s}", .{ layer, weight_suffix }));
    if (w.rank != 2 or w.shape[0] == 0) return error.InvalidEngramTable;
    return Plan.fromPack(p, layer, w.shape[0]);
}

/// The manifest's row indices: up to `n` seeded random rows, plus the first and last, ascending.
pub fn samples(a: std.mem.Allocator, rows: u64, n: u64, seed: u64) ![]u64 {
    if (rows == 0) return error.NoEngramTableRows;
    const want: u64 = @min(n, rows);
    var out: std.ArrayList(u64) = .empty;
    errdefer out.deinit(a);
    var rng = std.Random.DefaultPrng.init(seed);
    var drawn: u64 = 0;
    while (drawn < want) {
        const x = rng.random().uintLessThan(u64, rows);
        if (std.mem.indexOfScalar(u64, out.items, x) == null) {
            drawn += 1;
            try out.append(a, x);
        }
    }
    for ([_]u64{ 0, rows - 1 }) |x| if (std.mem.indexOfScalar(u64, out.items, x) == null) try out.append(a, x);
    std.mem.sort(u64, out.items, {}, std.sort.asc(u64));
    return out.toOwnedSlice(a);
}

pub const Progress = *const fn (layer: u32, done: u64, total: u64) void;

/// Writes layer's table to `<dir>/engram-l<layer>-r0of1.bin` through a synced `.part` file, `chunk` rows at a time.
pub fn pack(a: std.mem.Allocator, io: Io, p: *const Pack, dir: Io.Dir, layer: u32, chunk: u64, progress: ?Progress) !u64 {
    if (chunk == 0) return error.InvalidChunk;
    const t = try tableOf(p, layer);
    var name_buf: [64]u8 = undefined;
    const name = try shardName(&name_buf, layer);
    var part_buf: [72]u8 = undefined;
    const part = try std.fmt.bufPrint(&part_buf, "{s}.part", .{name});
    const file = try dir.createFile(io, part, .{});
    var closed = false;
    errdefer {
        if (!closed) file.close(io);
        dir.deleteFile(io, part) catch {};
    }
    const head = header(layer, t.rows);
    try file.writePositionalAll(io, &head, 0);
    const n: usize = @intCast(@min(chunk, t.rows));
    const codes = try a.alloc(u8, n * er.values);
    defer a.free(codes);
    const scales = try a.alloc(u8, n * 8);
    defer a.free(scales);
    const out = try a.alloc(u8, n * er.row_bytes);
    defer a.free(out);
    var row: u64 = 0;
    while (row < t.rows) {
        const k: usize = @intCast(@min(n, t.rows - row));
        try p.read(io, t.weight, .{ .offset = row * er.values, .len = k * er.values }, codes[0 .. k * er.values]);
        try p.read(io, t.scale, .{ .offset = row * 8, .len = k * 8 }, scales[0 .. k * 8]);
        for (0..k) |i| {
            @memcpy(out[i * er.row_bytes ..][0..er.values], codes[i * er.values ..][0..er.values]);
            @memcpy(out[i * er.row_bytes + er.values ..][0..8], scales[i * 8 ..][0..8]);
        }
        try file.writePositionalAll(io, out[0 .. k * er.row_bytes], er.header_bytes + row * er.row_bytes);
        row += k;
        if (progress) |f| f(layer, row, t.rows);
    }
    try file.sync(io);
    file.close(io);
    closed = true;
    try dir.rename(part, dir, name, io);
    return t.rows;
}

pub const Verified = struct { rows: u64, bad: u64 };

/// Reads rows through `engram_rows` and compares them with the release bytes decoded by `Table`.
pub fn verify(a: std.mem.Allocator, io: Io, p: *const Pack, dir_path: []const u8, layers: []const u32, n: u64, seed: u64) !Verified {
    const rows = try er.Rows.open(a, dir_path, 0, 1, layers, 4096);
    defer rows.close();
    var rng = std.Random.DefaultPrng.init(seed);
    var v: Verified = .{ .rows = 0, .bad = 0 };
    const batch = 64;
    var idx: [batch]i64 = undefined;
    var got: [batch * er.values]u16 = undefined;
    for (layers, 0..) |layer, li| {
        const t = try tableOf(p, layer);
        const every = n >= t.rows;
        const want = if (every) t.rows else @min(t.rows, n + 2);
        var done: u64 = 0;
        while (done < want) {
            const k: usize = @intCast(@min(batch, want - done));
            for (idx[0..k], 0..) |*x, i| {
                const at = done + i;
                x.* = @intCast(if (every) at else if (at == 0) 0 else if (at == 1) t.rows - 1 else rng.random().uintLessThan(u64, t.rows));
            }
            try rows.rows(li, idx[0..k], er.values, got[0 .. k * er.values]);
            for (idx[0..k], 0..) |x, i| {
                var codes: [er.values]u8 = undefined;
                var sc: [8]u8 = undefined;
                const r: u64 = @intCast(x);
                try p.read(io, t.weight, .{ .offset = r * er.values, .len = er.values }, &codes);
                try p.read(io, t.scale, .{ .offset = r * 8, .len = 8 }, &sc);
                var exp: [er.values]u16 = undefined;
                try (er.Table{ .weights = &codes, .scales = &sc, .rows = 1 }).row(0, &exp);
                v.rows += 1;
                if (!std.mem.eql(u16, &exp, got[i * er.values ..][0..er.values])) v.bad += 1;
            }
            done += k;
        }
    }
    return v;
}
