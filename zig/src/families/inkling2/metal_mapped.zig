//! A safetensors file in place: a private mapping as one no-copy Metal buffer; the 19.5 GB checkpoint is never copied.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const ck_host = @import("../../core/checkpoint.zig");
const st = @import("../../core/safetensors.zig");

pub const Mapped = struct {
    /// The whole file, rounded up to whole pages (the last page reads zeros past the end of the file).
    span: []align(std.heap.page_size_min) u8,
    file_len: usize,
    /// Offset of the first tensor byte: 8 + the JSON header's length.
    data_start: usize,

    pub fn open(path: [:0]const u8) !Mapped {
        const fd = platform.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) {
            std.log.err("cannot open {s}", .{path});
            return error.OpenFailed;
        }
        defer _ = platform.c.close(fd);
        const end = platform.c.lseek(fd, 0, platform.c.SEEK.END);
        if (end < 8) return error.BadSafetensors;
        const file_len = std.math.cast(usize, end) orelse return error.BadSafetensors;
        const page = std.heap.pageSize();
        // Metal requires writable virtual memory for a shared allocation. The file is never written.
        const span = try std.posix.mmap(null, std.mem.alignForward(usize, file_len, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE }, fd, 0);
        errdefer std.posix.munmap(span);
        const header_len = try st.headerSize(std.mem.readInt(u64, span[0..8], .little), file_len);
        return .{ .span = span, .file_len = file_len, .data_start = 8 + header_len };
    }

    pub fn deinit(m: *Mapped) void {
        std.posix.munmap(m.span);
    }

    pub fn header(m: *const Mapped) []const u8 {
        return m.span[8..m.data_start];
    }

    pub fn dataLen(m: *const Mapped) usize {
        return m.file_len - m.data_start;
    }

    /// The file as one shared no-copy buffer (the caller owns it; the mapping must outlive it).
    pub fn buffer(m: *const Mapped, device: mtl.Device) !mtl.Buffer {
        if (m.span.len > device.maxBufferLength()) return error.CheckpointTooLarge;
        return device.bufferNoCopy(m.span.ptr, m.span.len, mtl.ResourceOptions.shared);
    }

    /// Consume `buf`, registering the file's tensors at their offsets; `c` must be empty and on error keeps the buffer.
    pub fn adopt(m: *const Mapped, c: *ck.Checkpoint, buf: mtl.Buffer) !void {
        if (c.tensors.count() != 0) {
            buf.deinit();
            return error.CheckpointNotEmpty;
        }
        try m.adoptShard(c, buf);
    }

    /// adopt for one shard of a multi-file checkpoint: on error `buf` is freed here or owned by `c`, never both.
    pub fn adoptShard(m: *const Mapped, c: *ck.Checkpoint, buf: mtl.Buffer) !void {
        {
            errdefer buf.deinit();
            try c.own(buf, m.span.len);
        }
        var arena = std.heap.ArenaAllocator.init(c.allocator);
        defer arena.deinit();
        const entries = try st.parseHeader(arena.allocator(), m.header(), m.dataLen());
        for (entries.keys(), entries.values()) |name, entry| {
            if (c.tensors.contains(name)) return error.DuplicateTensor;
            try c.set(name, .{
                .buffer = buf,
                .offset = m.data_start + entry.begin,
                .bytes = entry.end - entry.begin,
                .dtype = entry.dtype,
                .rank = entry.rank,
                .shape = entry.shape,
            });
        }
    }

    /// Fault every page in on `threads` threads: the GPU takes no page fault, so the first command buffer pages it in.
    pub fn warm(m: *const Mapped, threads: usize) void {
        const page = std.heap.pageSize();
        const pages = m.span.len / page;
        var sums: [16]u8 = @splat(0);
        var handles: [16]?std.Thread = @splat(null);
        const n = @max(1, @min(threads, sums.len));
        const chunk = (pages + n - 1) / n;
        for (0..n) |i| {
            const lo = @min(i * chunk, pages);
            const hi = @min(lo + chunk, pages);
            handles[i] = std.Thread.spawn(.{}, touch, .{ m.span, page, lo, hi, &sums[i] }) catch null;
            if (handles[i] == null) touch(m.span, page, lo, hi, &sums[i]);
        }
        for (handles) |h| if (h) |t| t.join();
        std.mem.doNotOptimizeAway(sums);
    }
};

/// addFileSplit's `later`: nothing stays in its file, so every tensor packs at the checkpoint's alignment.
fn packAll(name: []const u8, rank: usize) bool {
    _ = name;
    _ = rank;
    return false;
}

fn touch(span: []const u8, page: usize, lo: usize, hi: usize, sink: *u8) void {
    var acc: u8 = 0;
    for (lo..hi) |p| acc +%= @as(*const volatile u8, &span[p * page]).*;
    sink.* = acc;
}

/// The checkpoint's shards as one set: each file stays mapped when its offsets meet alignment, else repacked.
pub const Set = struct {
    gpa: std.mem.Allocator,
    files: std.ArrayList(File) = .empty,

    const File = union(enum) {
        /// A private mapping adopted no-copy into the checkpoint.
        mapped: Mapped,
        /// A shard repacked by Checkpoint.addFileSplit at adopt (its file offsets are not aligned).
        repacked: [:0]u8, // the path, owned
    };

    /// The shard paths of `dir`: model.safetensors.index.json's weight_map, else model*.safetensors.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Set {
        const paths = try ck_host.shardFiles(gpa, io, dir);
        defer ck_host.freeShardFiles(gpa, paths);
        var s: Set = .{ .gpa = gpa };
        errdefer s.deinit();
        for (paths) |path| {
            var m = try Mapped.open(path);
            if (try mappable(gpa, &m)) {
                s.files.append(gpa, .{ .mapped = m }) catch |err| {
                    m.deinit();
                    return err;
                };
            } else {
                m.deinit();
                const owned = try gpa.dupeSentinel(u8, path, 0);
                s.files.append(gpa, .{ .repacked = owned }) catch |err| {
                    gpa.free(owned);
                    return err;
                };
            }
        }
        return s;
    }

    /// Every tensor's absolute offset meets the strictest alignment its dtype's consumers ask for.
    fn mappable(gpa: std.mem.Allocator, m: *const Mapped) !bool {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const entries = try st.parseHeader(arena.allocator(), m.header(), m.dataLen());
        for (entries.values()) |e| {
            if ((m.data_start + e.begin) % alignFor(e.dtype) != 0) return false;
        }
        return true;
    }

    fn alignFor(dt: st.DType) usize {
        return switch (dt) {
            .bf16 => 16, // the dense kernels' vector loads
            .f32, .i32, .u32 => 4,
            .f16, .i16, .u16 => 2,
            .f64, .i64, .u64 => 8,
            else => 1,
        };
    }

    /// Adopt every shard into `c`: mapped files no-copy, packed files repacked and aligned.
    pub fn adopt(s: *Set, c: *ck.Checkpoint, device: mtl.Device) !void {
        for (s.files.items) |*f| {
            switch (f.*) {
                .mapped => |*m| {
                    const buf = try m.buffer(device);
                    try m.adoptShard(c, buf);
                },
                .repacked => |path| try c.addFileSplit(device, path, "", null, packAll),
            }
        }
    }

    /// Fault every mapped page in on `threads` threads; repacked shards were read at adopt.
    pub fn warm(s: *Set, threads: usize) void {
        for (s.files.items) |*f| switch (f.*) {
            .mapped => |*m| m.warm(threads),
            .repacked => {},
        };
    }

    pub fn deinit(s: *Set) void {
        for (s.files.items) |*f| switch (f.*) {
            .mapped => |*m| m.deinit(),
            .repacked => |path| s.gpa.free(path),
        };
        s.files.deinit(s.gpa);
    }
};
