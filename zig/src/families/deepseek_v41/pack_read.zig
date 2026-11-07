//! Positional pack reads through bounded aligned scratch, or the original buffered descriptor.
const std = @import("std");
const core = @import("core");
const direct = core.direct_io;
const Io = std.Io;

pub const limit = direct.chunk;

pub const Reader = struct {
    gpa: std.mem.Allocator,
    io: Io,
    file: union(enum) { direct: direct.File, buffered: Io.File },
    scratch: direct.Buffer = &.{},

    pub fn open(gpa: std.mem.Allocator, io: Io, path: [:0]const u8, enabled: bool) !Reader {
        return .{ .gpa = gpa, .io = io, .file = if (enabled)
            .{ .direct = try direct.File.open(path) }
        else
            .{ .buffered = try Io.Dir.cwd().openFile(io, path, .{}) } };
    }

    pub fn close(r: *Reader) void {
        switch (r.file) {
            .direct => |*f| f.close(),
            .buffered => |f| f.close(r.io),
        }
        if (r.scratch.len > 0) r.gpa.free(r.scratch);
        r.* = undefined;
    }

    pub fn read(r: *Reader, out: []u8, offset: u64) !void {
        if (out.len == 0) return;
        switch (r.file) {
            .buffered => |f| {
                if (try f.readPositionalAll(r.io, out, offset) != out.len) return error.ShortRead;
            },
            .direct => |f| {
                if (r.scratch.len == 0) {
                    const size = direct.span(offset, @min(out.len, direct.fits(limit)));
                    r.scratch = try r.gpa.alignedAlloc(u8, .fromByteUnits(direct.alignment), size);
                }
                var done: usize = 0;
                while (done < out.len) {
                    const at = offset + done;
                    const n = @min(out.len - done, r.scratch.len - @as(usize, @intCast(at % direct.alignment)));
                    const bytes = f.read(r.scratch, at, n) catch |err| return switch (err) {
                        error.EndOfFile => error.ShortRead,
                        else => err,
                    };
                    @memcpy(out[done..][0..n], bytes);
                    done += n;
                }
            },
        }
    }

    /// The existing pack policy, restricted to this completed range; never a release of all mapped shards.
    pub fn dropCache(r: *const Reader, offset: u64, len: usize) void {
        if (@import("builtin").os.tag != .linux) return;
        const file: Io.File = switch (r.file) {
            .direct => |f| .{ .handle = f.fd, .flags = .{ .nonblocking = false } },
            .buffered => |f| f,
        };
        core.checkpoint.cache.range(file, offset, len);
    }
};
