//! Recorded any-width kernels (0.6.6's qa_ texts) at another width: the same text with its constants swapped.
const std = @import("std");
const mtl = @import("metal");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Variant = replay.Variant;
const Hc = replay.Hc;
const experts4 = @import("experts4.zig");
const rows_w = @import("rows_w.zig");

/// A recorded `constexpr int` and the value it takes.
pub const Const = struct { name: []const u8, value: usize };

/// `text` with `constexpr int name = ...;` set to `value`; the declaration must appear exactly once.
pub fn constant(a: std.mem.Allocator, text: []const u8, name: []const u8, value: usize) ![]const u8 {
    const key = try std.fmt.allocPrint(a, "constexpr int {s} = ", .{name});
    if (std.mem.count(u8, text, key) != 1) return error.KernelPatch;
    const at = std.mem.indexOf(u8, text, key).? + key.len;
    const end = at + (std.mem.indexOfScalar(u8, text[at..], ';') orelse return error.KernelPatch);
    return std.fmt.allocPrint(a, "{s}{d}{s}", .{ text[0..at], value, text[end..] });
}

/// A pack table of u32s read into host memory: read on the host only, so never a device buffer.
pub fn table(r: *Run, name: []const u8) ![]const u32 {
    const e = try r.entry(name);
    if (e.len % 4 != 0) return error.BadPackTable;
    const out = try r.arena.alloc(u32, e.len / 4);
    try replay.readAll(e.fd, std.mem.sliceAsBytes(out), e.at);
    return out;
}

/// The recorded variant `v` at `consts`, compiled once and kept beside the recorded variants under its own key.
pub fn patched(r: *Run, v: *Variant, consts: []const Const) !*Variant {
    var key: std.ArrayList(u8) = .empty;
    try key.appendSlice(r.arena, v.name);
    for (consts) |c| try key.print(r.arena, "|{s}={d}", .{ c.name, c.value });
    if (r.variants.get(key.items)) |hit| return hit;
    var text = try Run.variantText(r.arena, v);
    for (consts) |c| text = try constant(r.arena, text, c.name, c.value);
    if (experts4.halfApplies(text, consts) and rows_w.nib(r.device.generation(), 2) == 1) text = try experts4.half(r.arena, text);
    if (experts4.applies(text, consts)) text = try experts4.balanced(r.arena, text);
    if (experts4.gateApplies(text, consts)) text = try experts4.gated(r.arena, text);
    const lib = try r.ownResource(try mtl.Library.fromSource(r.device, text, mtl.CompileOptions.mlx()));
    const nv = try r.arena.create(Variant);
    nv.* = v.*;
    nv.pipe = try r.ownResource(try mtl.Pipeline.init(r.device, lib, try std.fmt.allocPrintSentinel(r.arena, "{s}", .{v.name}, 0), false));
    nv.text = text;
    try r.variants.put(r.arena, key.items, nv);
    return nv;
}

/// `role` launched as recorded at `as_rows` rows on its kernel at `consts` (Run.callAs when there are none).
pub fn callAs(r: *Run, role: []const u8, as_rows: usize, ins: []const Buf, outs: []const Buf, consts: []const Const) !void {
    if (consts.len == 0) return r.callAs(role, as_rows, ins, outs);
    if (r.skip & Run.class(role) != 0) return;
    if (r.hcskip != 0) for ([_][]const u8{ "hc_norm", "hc_down", "hc_up" }, 0..) |n, i| {
        if (r.hcskip & (@as(u32, 1) << @intCast(i)) != 0 and std.mem.indexOf(u8, role, n) != null) return;
    };
    var name: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&name, "{s}|{d}", .{ role, as_rows })) orelse return error.NoSite;
    try r.bindV(try patched(r, s.v, consts), ins, outs);
    r.enc.dispatchThreads(s.grid, s.tg);
    if (!r.serial) r.enc.barrier();
}

/// The one-row variant of `role` at `consts`, as callRows takes an override; null for the recorded width.
pub fn oneRow(r: *Run, role: []const u8, consts: []const Const) !?*Variant {
    if (consts.len == 0) return null;
    var name: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&name, "{s}|1", .{role})) orelse return error.NoSite;
    return try patched(r, s.v, consts);
}

/// A width's two constants under the names a kernel declares them; none for the recorded 6/32 (null).
pub fn width(buf: *[2]Const, bits: []const u8, group: []const u8, w: ?[2]usize) []const Const {
    const v = w orelse return &.{};
    buf.* = .{ .{ .name = bits, .value = v[0] }, .{ .name = group, .value = v[1] } };
    return buf;
}

/// Expert widths as the expert kernels name them: routed (WB, WG) and shared (SWB, SWG), gate/up and down apart.
pub const Experts = struct { gu: [4]usize, down: [4]usize };

pub fn four(buf: *[4]Const, w: [4]usize) []const Const {
    buf.* = .{ .{ .name = "WB", .value = w[0] }, .{ .name = "WG", .value = w[1] }, .{ .name = "SWB", .value = w[2] }, .{ .name = "SWG", .value = w[3] } };
    return buf;
}

/// The routed and shared experts on the recorded any-width kernels at `w` (Run.experts' recorded path).
pub fn experts(r: *Run, gu_role: []const u8, down_role: []const u8, x: Buf, lg: Buf, e: [18]Buf, act: Buf, pick: Buf, wts: Buf, rows_buf: Buf, y: Buf, w: Experts) !void {
    if (r.skip & (1 << 2) != 0) return;
    var cg: [4]Const = undefined;
    var cd: [4]Const = undefined;
    r.enc.setBuffer(rows_buf.b, rows_buf.off, 17); // the 4/32 gate/up's row count (experts4.gated)
    try callAs(r, gu_role, r.rows, &.{ x, lg, e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8], e[9], e[10], e[11] }, &.{ act, pick, wts }, four(&cg, w.gu));
    try callAs(r, down_role, r.rows, &.{ act, pick, e[12], e[13], e[14], e[15], e[16], e[17], rows_buf }, &.{y}, four(&cd, w.down));
}

/// A pack hyper-connection: its seven tensors, and its `.widths` table (down, up) when not 6/32.
pub fn hcOf(r: *Run, comptime fmt: []const u8, args: anytype) !Hc {
    var name: [96]u8 = undefined;
    const stem = try std.fmt.bufPrint(&name, fmt, args);
    var full: [128]u8 = undefined;
    const parts = [_][]const u8{ "scale", "down.w", "down.s", "down.b", "up.w", "up.s", "up.b" };
    var out: [7]Buf = undefined;
    for (parts, 0..) |p, i| out[i] = try r.load(try std.fmt.bufPrint(&full, "{s}.{s}", .{ stem, p }));
    var hc: Hc = .{ .scale = out[0], .dw = out[1], .ds = out[2], .db = out[3], .uw = out[4], .us = out[5], .ub = out[6] };
    const widths = try std.fmt.bufPrint(&full, "{s}.widths", .{stem}); // any width but 6/32 (pack.putHc)
    if (r.index.contains(widths)) {
        const v = try table(r, widths);
        if (v.len != 4) return error.BadPackTable;
        hc.w = .{ .{ v[0], v[1] }, .{ v[2], v[3] } };
    }
    return hc;
}
