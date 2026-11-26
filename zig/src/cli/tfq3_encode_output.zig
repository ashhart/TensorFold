//! Exact TFPN extent arithmetic and transactional output-directory publication.
const std = @import("std");
const Allocator = std.mem.Allocator;
const output_guard = @import("tfq3_cli_output.zig");

pub const MatrixCounts = struct { halfs: u64, words: u64 };
pub const ShardLayout = struct { off_meta: u64 = 64, off_scales: u64, off_weights: u64, halfs: u64, words: u64, size: u64 };

fn add(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch error.OutputTooLarge;
}

fn mul(a: u64, b: u64) !u64 {
    return std.math.mul(u64, a, b) catch error.OutputTooLarge;
}

fn align16k(x: u64) !u64 {
    const rounded = try add(x, 16383);
    return try mul(rounded / 16384, 16384);
}

pub fn matrixCounts(r: u32, c: u32, bits: u32) !MatrixCounts {
    const cells = try mul(r, c);
    return .{ .halfs = try mul(r / 128, c / 128), .words = try mul(cells, bits) / 32 };
}

pub fn shardLayout(count: usize, halfs: u64, words: u64) !ShardLayout {
    const meta_end = try add(64, try mul(count, 128));
    const off_scales = try align16k(meta_end);
    const off_weights = try align16k(try add(off_scales, try mul(halfs, 2)));
    return .{
        .off_scales = off_scales,
        .off_weights = off_weights,
        .halfs = halfs,
        .words = words,
        .size = try align16k(try add(off_weights, try mul(words, 4))),
    };
}

pub const Stage = struct {
    path: []const u8,
    out: []const u8,
    existing_empty: bool,
    active: bool = true,

    /// Output must be absent or empty and may never equal or contain the source, or be contained by it.
    pub fn init(a: Allocator, io: std.Io, src: []const u8, out: []const u8) !Stage {
        if (out.len == 0) return error.InvalidOutput;
        const base = std.fs.path.basename(out);
        if (base.len == 0 or std.mem.eql(u8, base, ".") or std.mem.eql(u8, base, "..")) return error.InvalidOutput;
        if (try output_guard.overlaps(a, io, src, out)) return error.SourceOutputAlias;
        const cwd = std.Io.Dir.cwd();
        var exists = true;
        const st = cwd.statFile(io, out, .{ .follow_symlinks = false }) catch |e| switch (e) {
            error.FileNotFound, error.NotDir => blk: {
                exists = false;
                break :blk null;
            },
            else => return e,
        };
        if (st) |s| {
            if (s.kind != .directory) return error.OutputNotEmpty;
            var d = try cwd.openDir(io, out, .{ .iterate = true });
            defer d.close(io);
            var it = d.iterate();
            if (try it.next(io) != null) return error.OutputNotEmpty;
        }
        const parent = std.fs.path.dirname(out) orelse ".";
        try cwd.createDirPath(io, parent);
        var candidate: []const u8 = undefined;
        for (0..100) |_| {
            const nonce: u64 = @truncate(@as(u96, @bitCast(std.Io.Clock.awake.now(io).toNanoseconds())));
            candidate = try std.fmt.allocPrint(a, "{s}/.{s}.tfq3-tmp-{x}", .{ parent, base, nonce });
            cwd.createDir(io, candidate, .default_dir) catch |e| switch (e) {
                error.PathAlreadyExists => continue,
                else => return e,
            };
            errdefer cwd.deleteTree(io, candidate) catch {};
            return .{ .path = candidate, .out = try a.dupe(u8, out), .existing_empty = exists };
        }
        return error.TempDirectoryUnavailable;
    }

    pub fn cleanup(self: *Stage, io: std.Io) void {
        if (self.active) std.Io.Dir.cwd().deleteTree(io, self.path) catch {};
        self.active = false;
    }

    /// Rename replaces an empty directory atomically; fallback publication is safe elsewhere.
    pub fn publish(self: *Stage, io: std.Io) !void {
        const cwd = std.Io.Dir.cwd();
        cwd.rename(self.path, cwd, self.out, io) catch |e| {
            if (!self.existing_empty) return e;
            switch (e) {
                error.DirNotEmpty => {},
                else => return e,
            }
            var d = try cwd.openDir(io, self.out, .{ .iterate = true });
            var empty = true;
            {
                defer d.close(io);
                var it = d.iterate();
                empty = (try it.next(io) == null);
            }
            if (!empty) return error.OutputNotEmpty;
            try cwd.deleteDir(io, self.out);
            cwd.rename(self.path, cwd, self.out, io) catch |rename_error| {
                cwd.createDirPath(io, self.out) catch {};
                return rename_error;
            };
        };
        self.active = false;
    }
};
