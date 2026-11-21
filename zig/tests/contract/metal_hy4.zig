//! Hy4-flash's Metal kernels under the contract, a file a group; flint's helpers (ranges, edits, lists) are shared.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
pub const fl = @import("metal_flint.zig");
pub const hy = @import("tensorfold").hy4_contract;
const g = contract.gates;

/// Hy4-flash's served widths: `text()` parses the checkpoint's config.json (hy.config_json); hidden 3,072.
pub const hidden = 3072;
pub fn text(a: std.mem.Allocator) !std.json.Parsed(hy.config.Model) {
    return hy.config.parse(a, hy.config_json);
}
/// Rows a call serves: one decode row, DFly2's window (the pending row and five drafts), a prompt chunk.
pub const decode_rows = 1;
pub const window_rows = 6;
pub const prompt_rows = 512;

const groups = .{ @import("metal_hy4_index.zig"), @import("metal_hy4_project.zig"), @import("metal_hy4_attend.zig"), @import("metal_hy4_nax.zig"), @import("metal_hy4_ihc.zig"), @import("metal_hy4_route.zig"), @import("metal_hy4_experts.zig"), @import("metal_hy4_rows.zig") };
const fields = .{ "index", "project", "attend", "nax", "ihc", "route", "experts", "rows" };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    index: groups[0].Suite = undefined,
    project: groups[1].Suite = undefined,
    attend: groups[2].Suite = undefined,
    nax: groups[3].Suite = undefined,
    ihc: groups[4].Suite = undefined,
    route: groups[5].Suite = undefined,
    experts: groups[6].Suite = undefined,
    rows: groups[7].Suite = undefined,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (fields) |f| {
            @field(s, f) = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
            try @field(s, f).collect(only);
        }
    }
    pub fn deinit(s: *Suite) void {
        inline for (fields) |f| @field(s, f).deinit();
    }
};

/// Gate 5's texts for an `hy4 ` registration: the group whose launches it names compiles it as its loader does.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !fl.Texts {
    if (std.mem.startsWith(u8, name, "hy4 units lane ")) return laneText(a, device, name);
    inline for (groups) |grp| if (try grp.limitsText(a, device, name)) |t| return t;
    return error.UnknownLaunch;
}

/// A "hy4 units lane qB nN kK skS [inline] fn" record's text: the layout metal_tensor_qkv.zig builds at that width.
fn laneText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !fl.Texts {
    const lane = @import("tensorfold").lane_projection;
    var words = std.mem.tokenizeScalar(u8, name["hy4 units lane ".len..], ' ');
    var v: [4]usize = undefined;
    for (&v, "qnks") |*x, letter| {
        const w = words.next() orelse return error.UnknownLaunch;
        if (w.len < 2 or w[0] != letter) return error.UnknownLaunch;
        x.* = try std.fmt.parseInt(usize, w[if (letter == 's') 2 else 1..], 10);
    }
    const sums = std.mem.indexOf(u8, name, " inline ") != null;
    const l: lane.Layout = .{ .n = v[1], .k = v[2], .sk = v[3], .format = .{ .bits = @intCast(v[0]), .group = 64 }, .reg = true, .precompute_sums = true, .rows = prompt_rows, .inline_sums = sums };
    try l.validate();
    const body = try lane.source(a, l);
    const here = try lane.frags.source(device, a, body);
    return .{ .here = here, .pre_m5 = body };
}

/// Launches named `hy4 units ` load only on tensor units (their loaders' `device.tensorUnits()` branches).
pub fn tensorOnly(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "hy4 units ");
}
