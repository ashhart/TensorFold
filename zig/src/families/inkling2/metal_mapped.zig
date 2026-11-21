//! A safetensors file in place: a private mapping as one no-copy Metal buffer; the 19.5 GB checkpoint is never copied.
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
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
        {
            errdefer buf.deinit();
            if (c.tensors.count() != 0) return error.CheckpointNotEmpty;
            try c.own(buf, m.span.len);
        }
        var arena = std.heap.ArenaAllocator.init(c.allocator);
        defer arena.deinit();
        const entries = try st.parseHeader(arena.allocator(), m.header(), m.dataLen());
        for (entries.keys(), entries.values()) |name, entry| {
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

fn touch(span: []const u8, page: usize, lo: usize, hi: usize, sink: *u8) void {
    var acc: u8 = 0;
    for (lo..hi) |p| acc +%= @as(*const volatile u8, &span[p * page]).*;
    sink.* = acc;
}
