//! Portable synchronous NVMe reader and bounded LRU. Linux retains its async physical reader.
const std = @import("std");
const f = @import("engram_rows.zig");
const eh = @import("engram_host.zig");
const Entry = struct { id: u64 = std.math.maxInt(u64), age: u64 = 0, bytes: [264]u8 = undefined };
const Table = struct { file: std.Io.File, h: f.Header, entries: []Entry };
pub const Stats = struct { hits: u64 = 0, misses: u64 = 0, issued: u64 = 0, reads: u64 = 0, bytes: u64 = 0 };
pub const Rows = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    tables: []Table,
    clock: u64 = 0,
    stats: Stats = .{},
    pub fn open(a: std.mem.Allocator, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        return openWithIo(a, std.Io.Threaded.global_single_threaded.io(), dir, rank, world, layers, cap);
    }
    pub fn openWithIo(a: std.mem.Allocator, io: std.Io, dir: []const u8, rank: u32, world: u32, layers: []const u32, cap: u32) !*Rows {
        if (world == 0 or rank >= world or cap == 0 or layers.len == 0) return error.BadEngramShard;
        const r = try a.create(Rows);
        errdefer a.destroy(r);
        const tables = try a.alloc(Table, layers.len);
        errdefer a.free(tables);
        var made: usize = 0;
        errdefer for (tables[0..made]) |t| {
            t.file.close(io);
            a.free(t.entries);
        };
        for (layers, tables) |layer, *t| {
            const path = try std.fmt.allocPrint(a, "{s}/engram-l{d}-r{d}of{d}.bin", .{ dir, layer, rank, world });
            defer a.free(path);
            const file = try std.Io.Dir.cwd().openFile(io, path, .{});
            errdefer file.close(io);
            var bytes: [48]u8 = undefined;
            if (try file.readPositionalAll(io, &bytes, 0) != 48) return error.BadEngramShard;
            const h = try f.Header.parse(&bytes, try file.length(io));
            if (h.layer != layer) return error.BadEngramShard;
            const entries = try a.alloc(Entry, cap);
            @memset(entries, .{});
            t.* = .{ .file = file, .h = h, .entries = entries };
            made += 1;
        }
        r.* = .{ .gpa = a, .io = io, .tables = tables };
        return r;
    }
    pub fn close(r: *Rows) void {
        for (r.tables) |t| {
            t.file.close(r.io);
            r.gpa.free(t.entries);
        }
        r.gpa.free(r.tables);
        r.gpa.destroy(r);
    }
    fn validate(r: *Rows, li: usize, ids: []const i64) !void {
        if (li >= r.tables.len) return error.BadEngramShard;
        const h = r.tables[li].h;
        for (ids) |id| if (id < 0 or @as(u64, @intCast(id)) < h.lo or @as(u64, @intCast(id)) >= h.hi)
            return error.EngramRowOutsideShard;
    }
    fn record(r: *Rows, li: usize, id: u64) !*[264]u8 {
        const table = &r.tables[li];
        r.clock +%= 1;
        var slot = &table.entries[0];
        for (table.entries) |*entry| {
            if (entry.id == id) {
                entry.age = r.clock;
                r.stats.hits += 1;
                return &entry.bytes;
            }
            if (entry.age < slot.age) slot = entry;
        }
        // Publish the cache identity only after the full positional read succeeds.
        var bytes: [264]u8 = undefined;
        const offset = f.header_bytes + (id - table.h.lo) * f.row_bytes;
        if (try table.file.readPositionalAll(r.io, &bytes, offset) != bytes.len) return error.ShortRead;
        slot.* = .{ .id = id, .age = r.clock, .bytes = bytes };
        r.stats.misses += 1;
        r.stats.reads += 1;
        r.stats.bytes += 264;
        return &slot.bytes;
    }
    /// Portable prefetch is synchronous. The Source contract permits completion before return.
    pub fn issue(r: *Rows, li: usize, ids: []const i64) !void {
        try r.validate(li, ids);
        for (ids) |id| {
            _ = try r.record(li, @intCast(id));
            r.stats.issued += 1;
        }
    }
    pub fn rows(r: *Rows, li: usize, ids: []const i64, dim: usize, out: []u16) !void {
        try r.validate(li, ids);
        if (dim != 256 or out.len != try std.math.mul(usize, ids.len, 256)) return error.BadEngramShard;
        for (ids, 0..) |id, i| {
            const b = try r.record(li, @intCast(id));
            for (0..256) |j| out[i * 256 + j] = f.bf16(f.e4m3(b[j]) * f.scale(b[256 + j / 32]));
        }
    }
    pub fn source(r: *Rows) eh.Source {
        return .{ .ptr = r, .vtable = &.{ .issue = issueFn, .rows = rowsFn } };
    }
    fn issueFn(p: *anyopaque, li: usize, ids: []const i64) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.issue(li, ids);
    }
    fn rowsFn(p: *anyopaque, li: usize, ids: []const i64, dim: usize, out: []u16) anyerror!void {
        const r: *Rows = @ptrCast(@alignCast(p));
        return r.rows(li, ids, dim, out);
    }
};
