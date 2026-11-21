//! Versioned mmap cache for the Ling 8-row CPU layout.
const std = @import("std");
const g = @import("core").gguf;
const repack8 = @import("cpu_gguf_repack8.zig");
const BlockQ4Kx8 = repack8.BlockQ4Kx8;
const BlockQ6Kx8 = repack8.BlockQ6Kx8;
const Repacked = repack8.Repacked;
const releaseSourcePages = repack8.releaseSourcePages;
const repackQ4KInto = repack8.repackQ4KInto;
const repackQ6KInto = repack8.repackQ6KInto;

pub const Cache = struct {
    const magic = "TFRP8V05";
    const Header = extern struct {
        magic: [8]u8,
        source_size: u64,
        source_inode: u64,
        source_padding: u64 = 0,
        source_mtime: i128,
        source_ctime: i128,
        source_hash: [32]u8,
        entry_count: u64,
        data_offset: u64,
        total_bytes: u64,
        reserved: [24]u8 = @splat(0),
    };
    pub const Entry = extern struct {
        src_offset: u64,
        src_len: u64,
        dst_offset: u64,
        dst_len: u64,
        kind: u32,
        groups: u32,
        blocks_per_row: u32,
        pad: u32 = 0,
    };

    file: std.Io.File,
    map: std.Io.File.MemoryMap,
    entries: []const Entry,

    const SourceIdentity = struct {
        size: u64,
        inode: u64,
        mtime: i128,
        ctime: i128,

        fn read(io: std.Io, source: *const g.File) !SourceIdentity {
            const stat = try source.file.stat(io);
            if (stat.size != source.map.memory.len) return error.GgufFileChanged;
            return .{ .size = stat.size, .inode = @intCast(stat.inode), .mtime = stat.mtime.nanoseconds, .ctime = stat.ctime.nanoseconds };
        }
        fn matches(self: SourceIdentity, hdr: *align(1) const Header) bool {
            return self.size == hdr.source_size and self.inode == hdr.source_inode and self.mtime == hdr.source_mtime and self.ctime == hdr.source_ctime;
        }
    };

    fn contentHash(bytes: []const u8) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var offset: usize = 0;
        while (offset < bytes.len) {
            const end = @min(bytes.len, offset + 16 * 1024 * 1024);
            hash.update(bytes[offset..end]);
            releaseSourcePages(bytes[offset..end]);
            offset = end;
        }
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        return digest;
    }

    fn alignUp16(v: u64) u64 {
        return (v + 15) & ~@as(u64, 15);
    }

    fn tryOpenExisting(io: std.Io, cache_path: []const u8, identity: SourceIdentity) ?Cache {
        const file = std.Io.Dir.cwd().openFile(io, cache_path, .{}) catch return null;
        var keep = false;
        defer if (!keep) file.close(io);
        const len = std.math.cast(usize, file.length(io) catch return null) orelse return null;
        if (len < @sizeOf(Header)) return null;
        var map = std.Io.File.MemoryMap.create(io, file, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false }) catch return null;
        defer if (!keep) map.destroy(io);
        const hdr: *align(1) const Header = @ptrCast(map.memory.ptr);
        if (!std.mem.eql(u8, &hdr.magic, magic)) return null;
        if (!identity.matches(hdr) or hdr.total_bytes != len) return null;
        const table_bytes = std.math.mul(u64, hdr.entry_count, @sizeOf(Entry)) catch return null;
        const table_end = std.math.add(u64, @sizeOf(Header), table_bytes) catch return null;
        if (hdr.data_offset < table_end or hdr.data_offset > len or (hdr.data_offset & 15) != 0) return null;
        const entries: []const Entry = @as([*]const Entry, @ptrCast(@alignCast(map.memory.ptr + @sizeOf(Header))))[0..@intCast(hdr.entry_count)];
        var prev_src: u64 = 0;
        for (entries, 0..) |e, i| {
            if (i > 0 and e.src_offset <= prev_src) return null;
            prev_src = e.src_offset;
            if ((e.dst_offset & 15) != 0 or e.dst_offset < hdr.data_offset or e.dst_offset > len or e.dst_len > len - e.dst_offset) return null;
            const elem_size: u64 = if (e.kind == @backingInt(g.TensorKind.q4_k)) @sizeOf(BlockQ4Kx8) else if (e.kind == @backingInt(g.TensorKind.q6_k)) @sizeOf(BlockQ6Kx8) else return null;
            const expected_len = std.math.mul(u64, std.math.mul(u64, e.groups, e.blocks_per_row) catch return null, elem_size) catch return null;
            if (e.dst_len != expected_len) return null;
        }
        keep = true;
        return .{ .file = file, .map = map, .entries = entries };
    }

    pub fn openOrBuild(a: std.mem.Allocator, io: std.Io, model_path: []const u8, source: *const g.File) !Cache {
        const src_mem = source.map.memory;
        const identity = try SourceIdentity.read(io, source);
        const cache_path = try std.fmt.allocPrint(a, "{s}.tf_repack8.v5", .{model_path});
        defer a.free(cache_path);

        if (tryOpenExisting(io, cache_path, identity)) |existing_value| {
            var existing = existing_value;
            errdefer existing.close(io);
            if (!std.meta.eql(identity, try SourceIdentity.read(io, source))) {
                return error.GgufFileChanged;
            }
            return existing;
        }

        const source_hash = contentHash(src_mem);
        if (!std.meta.eql(identity, try SourceIdentity.read(io, source))) return error.GgufFileChanged;
        const src_base = @intFromPtr(src_mem.ptr);
        var list: std.ArrayList(Entry) = .empty;
        defer list.deinit(a);

        for (source.document.tensors) |t| {
            if (t.kind != .q4_k and t.kind != .q6_k) continue;
            if (t.rank != 2 and t.rank != 3) continue;
            const width = std.math.cast(usize, t.shape[0]) orelse continue;
            const n = std.math.cast(usize, t.shape[1]) orelse continue;
            if (width == 0 or width % 256 != 0 or n < 8) continue;
            const bpr = width / 256;
            const blk_bytes: usize = if (t.kind == .q4_k) 144 else 210;
            const slice_len = n * bpr * blk_bytes;
            const slices: usize = if (t.rank == 3) (std.math.cast(usize, t.shape[2]) orelse continue) else 1;
            if (t.bytes.len != slices * slice_len) continue;
            const t_off = @intFromPtr(t.bytes.ptr) - src_base;
            const rep_elem: u64 = if (t.kind == .q4_k) @sizeOf(BlockQ4Kx8) else @sizeOf(BlockQ6Kx8);
            const dst_len: u64 = @as(u64, n / 8) * bpr * rep_elem;
            for (0..slices) |s_idx| {
                try list.append(a, .{
                    .src_offset = t_off + s_idx * slice_len,
                    .src_len = slice_len,
                    .dst_offset = 0,
                    .dst_len = dst_len,
                    .kind = @backingInt(t.kind),
                    .groups = @intCast(n / 8),
                    .blocks_per_row = @intCast(bpr),
                });
            }
        }

        std.mem.sort(Entry, list.items, {}, struct {
            fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
                return lhs.src_offset < rhs.src_offset;
            }
        }.lessThan);

        const data_offset = alignUp16(@sizeOf(Header) + @as(u64, list.items.len) * @sizeOf(Entry));
        var cursor: u64 = data_offset;
        var max_q4_blocks: usize = 0;
        var max_q6_blocks: usize = 0;
        for (list.items) |*e| {
            e.dst_offset = cursor;
            cursor = alignUp16(cursor + e.dst_len);
            const blocks: usize = @as(usize, e.groups) * @as(usize, e.blocks_per_row);
            if (e.kind == @backingInt(g.TensorKind.q4_k)) max_q4_blocks = @max(max_q4_blocks, blocks) else max_q6_blocks = @max(max_q6_blocks, blocks);
        }

        const hdr = Header{
            .magic = magic.*,
            .source_size = identity.size,
            .source_inode = identity.inode,
            .source_mtime = identity.mtime,
            .source_ctime = identity.ctime,
            .source_hash = source_hash,
            .entry_count = list.items.len,
            .data_offset = data_offset,
            .total_bytes = cursor,
        };

        const tmp_path = try std.fmt.allocPrint(a, "{s}.tf_repack8.v5.tmp", .{model_path});
        defer a.free(tmp_path);
        const out_file = try std.Io.Dir.cwd().createFile(io, tmp_path, .{ .truncate = true });
        var output_open = true;
        errdefer {
            if (output_open) out_file.close(io);
            std.Io.Dir.cwd().deleteFile(io, tmp_path) catch {};
        }

        try out_file.writeStreamingAll(io, std.mem.asBytes(&hdr));
        try out_file.writeStreamingAll(io, std.mem.sliceAsBytes(list.items));
        const pad_len = data_offset - (@sizeOf(Header) + @as(u64, list.items.len) * @sizeOf(Entry));
        if (pad_len > 0) {
            const zeros: [16]u8 = @splat(0);
            try out_file.writeStreamingAll(io, zeros[0..@intCast(pad_len)]);
        }

        const q4_scratch = try a.alloc(BlockQ4Kx8, max_q4_blocks);
        defer a.free(q4_scratch);
        const q6_scratch = try a.alloc(BlockQ6Kx8, max_q6_blocks);
        defer a.free(q6_scratch);
        const zeros: [16]u8 = @splat(0);

        for (list.items) |e| {
            const slice = src_mem[@intCast(e.src_offset)..][0..@intCast(e.src_len)];
            const kind: g.TensorKind = @fromBackingInt(@intCast(e.kind));
            try g.affine.validate(kind, slice);
            const blocks: usize = @as(usize, e.groups) * @as(usize, e.blocks_per_row);
            const width: usize = @as(usize, e.blocks_per_row) * 256;
            const n: usize = @as(usize, e.groups) * 8;
            if (kind == .q4_k) {
                repackQ4KInto(slice, width, n, q4_scratch[0..blocks]);
                try out_file.writeStreamingAll(io, std.mem.sliceAsBytes(q4_scratch[0..blocks]));
            } else {
                repackQ6KInto(slice, width, n, q6_scratch[0..blocks]);
                try out_file.writeStreamingAll(io, std.mem.sliceAsBytes(q6_scratch[0..blocks]));
            }
            const next_pad = alignUp16(e.dst_offset + e.dst_len) - (e.dst_offset + e.dst_len);
            if (next_pad > 0) try out_file.writeStreamingAll(io, zeros[0..@intCast(next_pad)]);
            // Release original source pages immediately so cold build RSS stays small.
            releaseSourcePages(slice);
        }

        out_file.close(io);
        output_open = false;
        if (!std.meta.eql(identity, try SourceIdentity.read(io, source))) return error.GgufFileChanged;
        try std.Io.Dir.cwd().rename(tmp_path, std.Io.Dir.cwd(), cache_path, io);
        return tryOpenExisting(io, cache_path, identity) orelse error.RepackCacheCorrupt;
    }

    pub fn lookup(self: *const Cache, src_offset: u64, src_len: u64, kind: g.TensorKind) ?Repacked {
        const want_kind = @backingInt(kind);
        var lo: usize = 0;
        var hi: usize = self.entries.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const e = self.entries[mid];
            if (e.src_offset < src_offset) {
                lo = mid + 1;
            } else if (e.src_offset > src_offset) {
                hi = mid;
            } else {
                if (e.src_len != src_len or e.kind != want_kind) return null;
                const raw = self.map.memory[@intCast(e.dst_offset)..][0..@intCast(e.dst_len)];
                const blocks: usize = @as(usize, e.groups) * @as(usize, e.blocks_per_row);
                return switch (kind) {
                    .q4_k => .{ .q4_k = @as([*]const BlockQ4Kx8, @ptrCast(@alignCast(raw.ptr)))[0..blocks] },
                    .q6_k => .{ .q6_k = @as([*]const BlockQ6Kx8, @ptrCast(@alignCast(raw.ptr)))[0..blocks] },
                    else => null,
                };
            }
        }
        return null;
    }

    pub fn close(self: *Cache, io: std.Io) void {
        self.map.destroy(io);
        self.file.close(io);
        self.* = undefined;
    }
};
