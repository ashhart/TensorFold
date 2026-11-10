//! flashnext-exl3: the Flash Next CUDA family on a real EXL3 pack against our float64 oracle; this hub parses the mode.
const std = @import("std");
const check = @import("../../check.zig");
const shared = @import("flashnext_exl3_rig.zig");
const names_tests = @import("flashnext_exl3_names_tests.zig");
const proj_tests = @import("flashnext_exl3_proj_tests.zig");
const layer_tests = @import("flashnext_exl3_layer_tests.zig");

const Gpu = check.Gpu;

pub const default_ids = shared.default_ids;
pub const RawPack = shared.RawPack;
pub const bfValue = shared.bfValue;
pub const Err = shared.Err;
pub const errOf = shared.errOf;
pub const download = shared.download;
pub const uploadTo = shared.uploadTo;
pub const Pass = shared.Pass;
pub const Rig = shared.Rig;
pub const Taps = layer_tests.Taps;
pub const deviceLayer = layer_tests.deviceLayer;
pub const layerCheck = layer_tests.layerCheck;

fn parseList(gpa: std.mem.Allocator, text: []const u8) ![]usize {
    var list: std.ArrayList(usize) = .empty;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| if (tok.len > 0) try list.append(gpa, try std.fmt.parseInt(usize, tok, 10));
    return list.toOwnedSlice(gpa);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 2) return error.MissingArgument;
    const dir = rest[0];
    const mode = rest[1];
    if (std.mem.eql(u8, mode, "names")) return names_tests.names(gpu, dir);
    if (std.mem.eql(u8, mode, "norms")) return names_tests.norms(gpu, dir);
    if (std.mem.eql(u8, mode, "ple")) return proj_tests.pleMode(gpu, dir);
    if (std.mem.eql(u8, mode, "router")) {
        const which = try parseList(gpu.gpa, if (rest.len > 2) rest[2] else "0,3,17");
        defer gpu.gpa.free(which);
        const rows = if (rest.len > 3) try std.fmt.parseInt(usize, rest[3], 10) else 4096;
        return names_tests.router(gpu, dir, which, rows);
    }
    if (std.mem.eql(u8, mode, "layer")) {
        const which = try parseList(gpu.gpa, if (rest.len > 2) rest[2] else "0,3");
        defer gpu.gpa.free(which);
        return layer_tests.layerMode(gpu, dir, which);
    }
    return error.UnknownMode;
}
