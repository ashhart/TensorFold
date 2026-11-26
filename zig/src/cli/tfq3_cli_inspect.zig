//! Native TFQ3 inspection backed by the shared full table and shard validators.
const std = @import("std");
const Allocator = std.mem.Allocator;
const validate = @import("tfq3_validate.zig");

pub fn run(a: Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, dir: []const u8) !u8 {
    const cwd = std.Io.Dir.cwd();
    const tbl_path = try std.fs.path.join(a, &.{ dir, "pw_v3_tables.bin" });
    const tbl = cwd.readFileAlloc(io, tbl_path, a, .limited(64 << 10)) catch |e| {
        if (e == error.OutOfMemory) return e;
        try err.print("FAIL {s}: {t}\n", .{ tbl_path, e });
        return 1;
    };
    defer a.free(tbl);
    validate.tables(tbl) catch |e| {
        try err.print("FAIL {s}: {t}\n", .{ tbl_path, e });
        return 1;
    };
    try out.print("OK   {s} PWV3 v3 | 16384 bytes\n", .{tbl_path});

    var d = cwd.openDir(io, dir, .{ .iterate = true }) catch |e| {
        try err.print("FAIL {s}: {t}\n", .{ dir, e });
        return 1;
    };
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |entry| {
        if ((entry.kind == .file or entry.kind == .sym_link) and std.mem.endsWith(u8, entry.name, ".tffpw")) {
            try names.append(a, try a.dupe(u8, entry.name));
        }
    }
    if (names.items.len == 0) {
        try err.print("FAIL {s}: no .tffpw shards found\n", .{dir});
        return 1;
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.less);

    var ok: usize = 0;
    for (names.items) |name| {
        const path = try std.fs.path.join(a, &.{ dir, name });
        const st = cwd.statFile(io, path, .{}) catch |e| {
            try err.print("FAIL {s}: {t}\n", .{ path, e });
            return 1;
        };
        if ((st.size & 16383) != 0) {
            try err.print("FAIL {s}: size {d} is not 16 KiB-aligned\n", .{ path, st.size });
            return 1;
        }
        const info = validate.shard(a, io, path, st.size) catch |e| {
            if (e == error.OutOfMemory) return e;
            try err.print("FAIL {s}: {t}\n", .{ path, e });
            return 1;
        };
        const gib = @as(f64, @floatFromInt(st.size)) / @as(f64, @floatFromInt(@as(u64, 1) << 30));
        switch (info.kind) {
            .dense => try out.print("OK   {s} TFPN v3 | {d} tensors | {d:.3} GiB\n", .{ path, info.tensors, gib }),
            .layer => try out.print("OK   {s} TFPW v3 | L{d:0>2} | {d} experts | {d:.3} GiB\n", .{ path, info.layer, info.experts, gib }),
        }
        ok += 1;
    }
    try out.print("validated {d} TFQ3 shard(s) in {s}\n", .{ ok, dir });
    return 0;
}
