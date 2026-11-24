//! Host-only TFQ3 checkpoint sidecar discovery and byte sizing for Flash Next admission and budgeting.
const std = @import("std");

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

pub fn scan(io: std.Io, dir: []const u8, num_layers: usize) Presence {
    var p: Presence = .{};
    const lim = @min(num_layers, max_layers);
    for (0..lim) |i| {
        if (layerSize(io, dir, i)) |sz| {
            p.layers[i] = true;
            p.any = true;
            p.routed_bytes +|= sz;
            p.total_bytes +|= sz;
        }
    }
    if (mtpSize(io, dir)) |sz| {
        p.mtp = true;
        p.any = true;
        p.routed_bytes +|= sz;
        p.total_bytes +|= sz;
    }
    if (p.any) {
        if (tablesSize(io, dir)) |sz| p.total_bytes +|= sz;
    }
    return p;
}
