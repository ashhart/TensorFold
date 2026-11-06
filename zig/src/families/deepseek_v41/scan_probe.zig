// ! The sparse scan's own input/output hash. The model owns the ! staging buffer (one per forward, reported after
// its wait); this module records a whole layer's rows.
const std = @import("std");
const mtl = @import("metal");
const ops = @import("ops.zig");
const View = @import("view.zig").View;

pub const Tag = enum(u8) { q_pre, q_post, chunk, main, dense, sink, q_pre2, q_post2, pos, pos2, sco, sco1, posl1, posl2, scof1, scof2 };

pub var layer: i32 = -1;
pub var stage: ?mtl.Buffer = null;
pub var cap: usize = 0;
pub var base: usize = 0;
const Ev = struct { tag: Tag, layer: u32, off: usize, len: usize };
pub var ev: [32]Ev = undefined;
pub var nev: usize = 0;

pub fn on() bool {
    return std.c.getenv("TF_DSV41_SCAN_PROBE") != null;
}
pub fn active(layer_index: u32) bool {
    return stage != null and layer >= 0 and layer_index == @as(u32, @intCast(layer));
}
pub fn init(rows: u32, q_width: u32) void {
    var lay: i32 = -1;
    if (std.c.getenv("TF_DSV41_SCAN_PROBE")) |v| lay = std.fmt.parseInt(i32, std.mem.span(v), 10) catch -1;
    layer = lay;
    // two query dumps, the chunk, the compressed cache and the dense table's live rows
    cap = @as(usize, rows) * q_width * 2 * 4 + @as(usize, rows) * 528 +
        @as(usize, rows) * 288 + (@as(usize, rows) + 4096) * 1024 + 4096;
}
pub fn reset() void {
    base = 0;
    nev = 0;
}
pub fn record(enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, width_u16: u32, tag: Tag, lay: u32) void {
    const stg = stage orelse return;
    if (rows == 0 or width_u16 == 0) return;
    if (nev >= ev.len) return;
    const len: usize = @as(usize, rows) * width_u16 * 2;
    const off = base;
    base += std.mem.alignForward(usize, len, 256);
    if (base > cap) return;
    ev[nev] = .{ .tag = tag, .layer = lay, .off = off, .len = len };
    nev += 1;
    ops.encodeCopy(enc, p, view, .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width_u16, .x_stride = width_u16, .y_stride = width_u16 });
}
pub fn recordStrided(enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, x_stride_u16: u32, width_u16: u32, tag: Tag, lay: u32) void {
    const stg = stage orelse return;
    if (rows == 0 or width_u16 == 0) return;
    if (nev >= ev.len) return;
    const len: usize = @as(usize, rows) * width_u16 * 2;
    const off = base;
    base += std.mem.alignForward(usize, len, 256);
    if (base > cap) return;
    ev[nev] = .{ .tag = tag, .layer = lay, .off = off, .len = len };
    nev += 1;
    ops.encodeCopy(enc, p, view, .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width_u16, .x_stride = x_stride_u16, .y_stride = width_u16 });
}
pub fn reportStrided(enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, x_stride_u16: u32, width_u16: u32, tag: Tag, lay: u32) void {
    _ = enc;
    _ = p;
    _ = view;
    _ = rows;
    _ = x_stride_u16;
    _ = width_u16;
    _ = tag;
    _ = lay;
}
pub fn recordAt(enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, width_u16: u32, tag: Tag, lay: u32, src_row: u32, dst_row: u32) void {
    _ = dst_row;
    const stg = stage orelse return;
    if (rows == 0 or width_u16 == 0) return;
    if (nev >= ev.len) return;
    const len: usize = @as(usize, rows) * width_u16 * 2;
    const off = base;
    base += std.mem.alignForward(usize, len, 256);
    if (base > cap) return;
    ev[nev] = .{ .tag = tag, .layer = lay, .off = off, .len = len };
    nev += 1;
    ops.encodeCopy(enc, p, view.shifted(@as(usize, src_row) * width_u16 * 2), .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width_u16, .x_stride = width_u16, .y_stride = width_u16 });
}
fn fnvMore(h: u64, bytes: []const u8) u64 {
    var x = h;
    for (bytes) |b| {
        x ^= b;
        x *%= 0x100000001b3;
    }
    return x;
}
pub fn report() void {
    const stg = stage orelse return;
    const raw: [*]const u8 = stg.contents();
    for (ev[0..nev]) |e| {
        const tag: []const u8 = switch (e.tag) {
            .q_pre => "qpre",
            .q_post => "qpost",
            .chunk => "chunk",
            .main => "main",
            .dense => "dense",
            .sink => "sink",
            .q_pre2 => "qpre2",
            .q_post2 => "qpost2",
            .pos => "pos",
            .pos2 => "pos2",
            .sco => "sco",
            .sco1 => "sco1",
            .posl1 => "posl1",
            .posl2 => "posl2",
            .scof1 => "scof1",
            .scof2 => "scof2",
        };
        std.log.info("scanprobe {s} l{d} {x:0>16}", .{ tag, e.layer, fnvMore(0xcbf29ce484222325, raw[e.off..][0..e.len]) });
        if (e.tag == .posl1 or e.tag == .posl2) {
            const pi: [*]const i32 = @ptrCast(@alignCast(raw + e.off));
            var b: [4096]u8 = undefined;
            var off2: usize = 0;
            for (0..512) |q| {
                if (off2 >= b.len) break;
                const piece = std.fmt.bufPrint(b[off2..], "{d} ", .{pi[q]}) catch break;
                off2 += piece.len;
            }
            const nm: []const u8 = if (e.tag == .posl1) "posl1" else "posl2";
            std.log.info("scanposl {s} l{d} {s}", .{ nm, e.layer, b[0..off2] });
        }
        if (e.tag == .sco or e.tag == .sco1) {
            const f: [*]const f32 = @ptrCast(@alignCast(raw + e.off));
            const n = e.len / 4;
            var bad: usize = 0;
            var first: f32 = 0;
            for (0..n) |q| {
                const bits: u32 = @bitCast(f[q]);
                if ((bits & 0xFFFF) != 0) {
                    if (bad == 0) first = f[q];
                    bad += 1;
                }
            }
            std.log.info("scanprobe sco-nonbf16 l{d} {d} of {d} first={d}", .{ e.layer, bad, n, first });
        }
    }
}
