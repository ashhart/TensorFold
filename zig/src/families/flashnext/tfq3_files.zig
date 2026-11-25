//! Host-only TFQ3 checkpoint sidecar discovery, checking and byte sizing for Flash Next admission and budgeting.
const std = @import("std");
pub const check = @import("tfq3_check.zig");

pub const max_layers = 48;

pub fn statSize(io: std.Io, path: []const u8) ?u64 {
    const s = std.Io.Dir.cwd().statFile(io, path, .{}) catch return null;
    return s.size;
}

pub fn layerSize(io: std.Io, dir: []const u8, layer: usize) ?u64 {
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/layer_{d:0>2}.tffpw", .{ dir, layer }) catch return null;
    return statSize(io, path);
}

pub fn mtpSize(io: std.Io, dir: []const u8) ?u64 {
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/mtp_00.tffpw", .{dir}) catch return null;
    return statSize(io, path);
}

pub fn tablesSize(io: std.Io, dir: []const u8) ?u64 {
    var buf: [512]u8 = undefined;
    const path = std.fmt.bufPrint(&buf, "{s}/pw_v3_tables.bin", .{dir}) catch return null;
    return statSize(io, path);
}

pub fn hasLayer(io: std.Io, dir: []const u8, layer: usize) bool {
    return layerSize(io, dir, layer) != null;
}

pub fn hasMtp(io: std.Io, dir: []const u8) bool {
    return mtpSize(io, dir) != null;
}

pub const Presence = struct {
    layers: [max_layers]bool = @splat(false),
    mtp: bool = false,
    any: bool = false,
    total_bytes: u64 = 0,
    routed_bytes: u64 = 0,
};

/// A shard's checked layout from its header and metadata, read by position (null: no such file); `layer` its index.
pub fn checkShard(io: std.Io, path: []const u8, layer: ?u32) !?check.Layout {
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| return if (err == error.FileNotFound) null else err;
    defer f.close(io);
    const len = std.math.cast(usize, try f.length(io)) orelse return error.Tfq3Overflow;
    var head: [@sizeOf(check.LayerHeader)]u8 = undefined;
    if (len < head.len or try f.readPositionalAll(io, &head, 0) != head.len) return error.Tfq3Truncated;
    var l = try check.header(&head, len, layer);
    var meta: [check.num_experts * @sizeOf(check.ExpertMeta)]u8 = undefined;
    if (try f.readPositionalAll(io, &meta, l.meta.off) != meta.len) return error.Tfq3Truncated;
    try check.experts(&meta, &l);
    return l;
}

/// The table file's length once checked (null: absent).
pub fn checkTables(io: std.Io, dir: []const u8) !?u64 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/pw_v3_tables.bin", .{dir});
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| return if (err == error.FileNotFound) null else err;
    defer f.close(io);
    const len = try f.length(io);
    var head: [8]u8 = undefined;
    const got = try f.readPositionalAll(io, &head, 0);
    try check.tables(head[0..got], std.math.cast(usize, len) orelse return error.Tfq3Overflow);
    return len;
}

/// Each present shard checked before it counts: a malformed one is refused by name, never admitted by presence.
pub fn scan(io: std.Io, dir: []const u8, num_layers: usize) !Presence {
    var p: Presence = .{};
    const lim = @min(num_layers, max_layers);
    var buf: [512]u8 = undefined;
    for (0..lim) |i| {
        const path = try std.fmt.bufPrint(&buf, "{s}/layer_{d:0>2}.tffpw", .{ dir, i });
        _ = (try refuse(path, checkShard(io, path, @intCast(i)))) orelse continue;
        const sz = layerSize(io, dir, i) orelse continue;
        p.layers[i] = true;
        p.any = true;
        p.routed_bytes +|= sz;
        p.total_bytes +|= sz;
    }
    const mtp = try std.fmt.bufPrint(&buf, "{s}/mtp_00.tffpw", .{dir});
    if (try refuse(mtp, checkShard(io, mtp, null))) |_| if (mtpSize(io, dir)) |sz| {
        p.mtp = true;
        p.any = true;
        p.routed_bytes +|= sz;
        p.total_bytes +|= sz;
    };
    if (p.any) {
        const sz = (try refuse("pw_v3_tables.bin", checkTables(io, dir))) orelse {
            if (!@import("builtin").is_test) std.log.err("flash next: TFQ3 shards in {s} need pw_v3_tables.bin beside them", .{dir});
            return error.Tfq3Table;
        };
        p.total_bytes +|= sz;
    }
    return p;
}

/// A check's result, its refusal logged with the file it names.
fn refuse(path: []const u8, got: anytype) @TypeOf(got) {
    return got catch |err| {
        if (!@import("builtin").is_test) std.log.err("flash next: TFQ3 {s} refused: {t}", .{ path, err });
        return err;
    };
}

test {
    _ = check;
}
