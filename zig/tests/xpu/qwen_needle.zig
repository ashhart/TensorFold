//! Needle in a haystack: windowed prefill of a long prompt, then greedy decode. usage: ...-test CKPT CTX IDS N_GEN

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qw = @import("qwen_xpu").win;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) return error.Usage;
    const dir = args[1];
    const ctx = try std.fmt.parseInt(u32, args[2], 10);
    const slash = std.mem.lastIndexOfScalar(u8, args[3], '/').?;
    const text = try ld.readFile(gpa, args[3][0..slash], args[3][slash + 1 ..]);
    const n_gen = try std.fmt.parseInt(usize, args[4], 10);
    if (n_gen < 1) return error.BadArgument; // before the device is opened
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    const is_mlx = !is_gguf and !std.mem.containsAtLeast(u8, dir, 1, "exl3");
    const rows: u32 = if (args.len > 5) try std.fmt.parseInt(u32, args[5], 10) else if (is_mlx) 16 else 512;
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    qw.default_rows = rows;
    var r = try rt.open();
    defer r.deinit();
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx);
    m.bf16_logits = true;
    std.debug.print("loaded; device memory {d:.2} GB; prompt {d} tokens in windows of {d}\n", .{ @as(f64, @floatFromInt(l.total)) / 1e9, ids.items.len, rows });
    const t0 = nowNs();
    var t: usize = 0;
    var first: [1]i32 = undefined;
    while (t < ids.items.len) {
        const n = @min(rows, ids.items.len - t);
        const last = t + n == ids.items.len;
        try m.forwardRows(ids.items[t .. t + n], if (last) 1 else 0, true);
        if (last) try m.rowArgmax(&first) else try m.r.sync();
        t += n;
    }
    const pf = @as(f64, @floatFromInt(nowNs() - t0)) / 1e9;
    std.debug.print("prefill {d:.1} s ({d:.0} tokens/s)\n", .{ pf, @as(f64, @floatFromInt(ids.items.len)) / pf });
    var tok: u32 = @intCast(first[0]);
    std.debug.print("generated ids: {d}", .{tok});
    const t1 = nowNs();
    for (1..n_gen) |_| {
        try m.forward(tok, true);
        tok = try m.argmax();
        std.debug.print(" {d}", .{tok});
    }
    std.debug.print("\ndecode {d:.2} ms/token\n", .{@as(f64, @floatFromInt(nowNs() - t1)) / 1e6 / @as(f64, @floatFromInt(@max(n_gen - 1, 1)))});
}
