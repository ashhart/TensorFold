//! The release's safetensors shards mapped read-only, each wrapped whole as one Metal buffer with no copy.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");

const Pack = host.pack.Pack;
const Info = host.pack.Info;

/// A tensor's place in a shard's buffer: its file offset (the buffer starts at the shard's first byte).
pub const Placed = struct { buf: mtl.Buffer, off: usize };

const Shard = struct { file: mtl.MappedFile, buf: mtl.Buffer };

pub const Stats = struct {
    shards: u32 = 0,
    /// the bytes of the shard buffers (whole pages of every shard a tensor was placed in)
    wired: u64 = 0,
    tensors: u64 = 0,
    bytes: u64 = 0,
    /// raw tensors the map left to the caller's copy (offset not a multiple of the width the kernel loads)
    refused: u64 = 0,
    refused_bytes: u64 = 0,
};

/// Whether a tensor is part of an Engram embedding table (never resident: the row shards serve them).
pub fn isTable(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".engram.embed.") != null;
}

/// Shards are mapped on first use, so a shard that only holds Engram tables is never mapped.
pub const Map = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    arena: std.heap.ArenaAllocator,
    paths: []const [:0]const u8,
    /// shards where most bytes are Engram table: their other tensors are copied, the shard never wrapped
    table_heavy: []bool,
    shards: []?Shard,
    max_buffer: u64,
    stats: Stats = .{},

    pub fn open(a: std.mem.Allocator, device: mtl.Device, pack: *const Pack) !Map {
        if (pack.dir.len == 0) return error.HeaderOnlyPack;
        var m: Map = .{ .a = a, .device = device, .arena = std.heap.ArenaAllocator.init(a), .paths = &.{}, .table_heavy = &.{}, .shards = &.{}, .max_buffer = device.maxBufferLength() };
        errdefer m.arena.deinit();
        const n = pack.files.items.len;
        const paths = try m.arena.allocator().alloc([:0]const u8, n);
        for (paths, pack.files.items) |*p, f| p.* = try std.fmt.allocPrintSentinel(m.arena.allocator(), "{s}/{s}", .{ pack.dir, f }, 0);
        const table = try a.alloc(u64, n);
        defer a.free(table);
        const total = try a.alloc(u64, n);
        defer a.free(total);
        @memset(table, 0);
        @memset(total, 0);
        var it = pack.names.iterator();
        while (it.next()) |e| {
            total[e.value_ptr.file] += e.value_ptr.nbytes;
            if (isTable(e.key_ptr.*)) table[e.value_ptr.file] += e.value_ptr.nbytes;
        }
        const heavy = try a.alloc(bool, n);
        errdefer a.free(heavy);
        for (heavy, table, total) |*h, t, all| h.* = t > all / 2;
        const shards = try a.alloc(?Shard, n);
        @memset(shards, null);
        m.paths = paths;
        m.table_heavy = heavy;
        m.shards = shards;
        return m;
    }

    /// Unwraps and unmaps every shard; no Ref into one may be used after.
    pub fn deinit(m: *Map) void {
        for (m.shards) |s| if (s) |x| {
            x.buf.deinit();
            x.file.deinit();
        };
        m.a.free(m.shards);
        m.a.free(m.table_heavy);
        m.arena.deinit();
        m.* = undefined;
    }

    fn shard(m: *Map, file: u32) !*Shard {
        if (m.shards[file]) |*s| return s;
        const f = try mtl.MappedFile.open(m.paths[file].ptr);
        errdefer f.deinit();
        if (f.bytes.len > m.max_buffer) return error.ShardExceedsBufferLimit;
        // the buffer stays untracked: weights are written once, before any command reads them
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const buf = try m.device.bufferNoCopy(f.bytes.ptr, f.bytes.len, opts);
        m.shards[file] = .{ .file = f, .buf = buf };
        m.stats.shards += 1;
        m.stats.wired += f.bytes.len;
        return &m.shards[file].?;
    }

    /// The tensor's bytes in place when `al` divides its file offset; null: the caller copies it.
    pub fn place(m: *Map, info: Info, al: u32) !?Placed {
        if (m.table_heavy[info.file]) return null;
        if (info.start % al != 0) {
            m.stats.refused += 1;
            m.stats.refused_bytes += info.nbytes;
            return null;
        }
        const s = try m.shard(info.file);
        if (info.start + info.nbytes > s.file.size) return error.BadSafetensors;
        m.stats.tensors += 1;
        m.stats.bytes += info.nbytes;
        return .{ .buf = s.buf, .off = @intCast(info.start) };
    }

    /// Every shard buffer made so far, appended to `out`.
    pub fn buffers(m: *const Map, a: std.mem.Allocator, out: *std.ArrayList(mtl.Buffer)) !void {
        for (m.shards) |s| if (s) |x| try out.append(a, x.buf);
    }
};
