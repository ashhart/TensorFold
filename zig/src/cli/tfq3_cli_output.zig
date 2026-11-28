//! Source/output alias and existing-output preflight for TFQ3 conversion.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// Returns zero when conversion may proceed, one after printing a refusal.
pub fn check(a: Allocator, io: std.Io, err: *std.Io.Writer, src_dir: []const u8, out_dir: []const u8, allow_nonempty: bool) !u8 {
    const cwd = std.Io.Dir.cwd();
    const src_real = canonical(cwd, a, io, src_dir) catch |e| {
        if (e == error.OutOfMemory) return e;
        try err.print("tensorfold tfq3: cannot resolve source {s}: {t}\n", .{ src_dir, e });
        return 1;
    };
    const out_real = canonical(cwd, a, io, out_dir) catch |e| {
        if (e == error.OutOfMemory) return e;
        try err.print("tensorfold tfq3: cannot resolve output {s}: {t}\n", .{ out_dir, e });
        return 1;
    };
    if (std.mem.eql(u8, src_real, out_real) or contains(src_real, out_real) or contains(out_real, src_real)) {
        try err.print("tensorfold tfq3: source and output overlap ({s}, {s}); refusing conversion that could destroy the source\n", .{ src_real, out_real });
        return 1;
    }

    var dir = cwd.openDir(io, out_dir, .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound => return 0,
        else => {
            try err.print("tensorfold tfq3: output {s} is not a usable directory: {t}\n", .{ out_dir, e });
            return 1;
        },
    };
    defer dir.close(io);
    var it = dir.iterate();
    const first = it.next(io) catch |e| {
        try err.print("tensorfold tfq3: cannot inspect output {s}: {t}\n", .{ out_dir, e });
        return 1;
    };
    if (first != null and !allow_nonempty) {
        try err.print("tensorfold tfq3: output directory {s} is not empty; use --skip-existing to allow existing external-converter output\n", .{out_dir});
        return 1;
    }
    return 0;
}

/// Direct encoder callers receive the same canonical containment check as CLI callers.
pub fn overlaps(a: Allocator, io: std.Io, src_dir: []const u8, out_dir: []const u8) !bool {
    const cwd = std.Io.Dir.cwd();
    const src_real = try canonical(cwd, a, io, src_dir);
    const out_real = try canonical(cwd, a, io, out_dir);
    return std.mem.eql(u8, src_real, out_real) or contains(src_real, out_real) or contains(out_real, src_real);
}

/// Existing paths resolve symlinks directly; only missing basenames recurse through a real parent.
fn canonical(cwd: std.Io.Dir, a: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    return cwd.realPathFileAlloc(io, path, a) catch |e| switch (e) {
        error.FileNotFound, error.NotDir => {
            const parent = std.fs.path.dirname(path) orelse ".";
            if (std.mem.eql(u8, parent, path)) return e;
            const real_parent = try canonical(cwd, a, io, parent);
            const base = std.fs.path.basename(path);
            if (base.len == 0) return e;
            return std.fs.path.resolveAlloc(a, &.{ real_parent, base });
        },
        else => return e,
    };
}

fn contains(parent: []const u8, child: []const u8) bool {
    const prefix = std.mem.trimEnd(u8, parent, std.fs.path.sep_str);
    if (prefix.len == 0) return parent.len != 0;
    return child.len > prefix.len and std.mem.startsWith(u8, child, prefix) and std.fs.path.isSep(child[prefix.len]);
}
