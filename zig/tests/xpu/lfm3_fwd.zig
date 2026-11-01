//! LFM3 forward on the checkpoint: every layer against ref64.py's float64 dumps, then 64 greedy tokens, twice.

const std = @import("std");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const stop = @import("xpu").stop;

const H = 2048;
const prompts = [_][]const u8{ "capitals", "chat", "items" };
const reply_len = 64;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn bf(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

/// The hidden rows of every layer (index 24: the final norm) as the windows pass, bf16.
const Rows = struct {
    layers: [25][]u16,
    row0: usize = 0,

    fn keep(ctx: *anyopaque, layer: i32, rows: []const u16) void {
        const s: *Rows = @ptrCast(@alignCast(ctx));
        const li: usize = if (layer < 0) 24 else @intCast(layer);
        @memcpy(s.layers[li][s.row0 * H ..][0..rows.len], rows);
    }
};

/// Per-row ||xpu - ref|| / ||ref||: mean and max over the rows.
fn relRows(got: []const u16, want: []const f32) [2]f64 {
    const n = want.len / H;
    var sum: f64 = 0;
    var mx: f64 = 0;
    for (0..n) |r| {
        var d: f64 = 0;
        var w: f64 = 0;
        for (0..H) |i| {
            const a = bf(got[r * H + i]);
            const b: f64 = want[r * H + i];
            d += (a - b) * (a - b);
            w += b * b;
        }
        const e = @sqrt(d / w);
        sum += e;
        mx = @max(mx, e);
    }
    return .{ sum / @as(f64, @floatFromInt(n)), mx };
}

fn readF32(gpa: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    const out = try gpa.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    gpa.free(bytes);
    return out;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("skipped: usage tf-xpu-lfm3_fwd-test MODEL_DIR REF_DIR [WINDOW_ROWS] [OUT_JSON]\n", .{});
        return;
    }
    const dir = args[1];
    const ref = args[2];
    const rows: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 64;
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, io, dir);
    defer c.deinit();
    var l = try @import("xpu").loader.Loader.init(gpa, &r, dir);
    var t0 = nowNs();
    var m = try lfm3.model.Model.load(gpa, &r, &l, &c, 4096, rows);
    std.debug.print("loaded in {d:.1} s, {d:.0} MiB on the device, windows of {d} rows\n", .{ @as(f64, @floatFromInt(nowNs() - t0)) / 1e9, @as(f64, @floatFromInt(m.device_bytes)) / (1 << 20), rows });

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try out.writer.writeAll("{");
    var bad = false;
    for (prompts, 0..) |name, pi| {
        const meta_path = try std.fmt.allocPrint(gpa, "{s}/{s}/meta.json", .{ ref, name });
        const meta_bytes = try std.Io.Dir.cwd().readFileAlloc(io, meta_path, gpa, .unlimited);
        const meta = try std.json.parseFromSlice(std.json.Value, gpa, meta_bytes, .{});
        const id_vals = meta.value.object.get("ids").?.array.items;
        const ids = try gpa.alloc(u32, id_vals.len);
        for (id_vals, ids) |v, *o| o.* = @intCast(v.integer);
        const T = ids.len;
        var keep: Rows = .{ .layers = undefined };
        for (&keep.layers) |*ly| ly.* = try gpa.alloc(u16, T * H);
        const hk: lfm3.model.Hook = .{ .ctx = &keep, .f = Rows.keep };
        try m.reset();
        var i: usize = 0;
        while (i < T) {
            const n = @min(T - i, rows);
            keep.row0 = i;
            try m.forwardPrompt(ids[i .. i + n], i + n == T, hk);
            i += n;
        }
        const next = try m.argmax();
        const gaps = meta.value.object.get("bf16_gap").?.object;
        std.debug.print("{s}: {d} prompt tokens; per-row relative L2 error against float64 (mean / max), and the bf16-points pass's own gap\n", .{ name, T });
        var worst: f64 = 0;
        for (0..25) |li| {
            const layer: i32 = if (li == 24) -1 else @intCast(li);
            const path = if (li == 24) try std.fmt.allocPrint(gpa, "{s}/{s}/final.f32", .{ ref, name }) else try std.fmt.allocPrint(gpa, "{s}/{s}/h{d:0>2}.f32", .{ ref, name, li });
            const want = try readF32(gpa, io, path);
            defer gpa.free(want);
            const e = relRows(keep.layers[li], want);
            var key: [8]u8 = undefined;
            const g = gaps.get(try std.fmt.bufPrint(&key, "{d}", .{layer})).?.array.items;
            const gm: f64 = switch (g[0]) {
                .float => |f| f,
                else => 0,
            };
            const gx: f64 = switch (g[1]) {
                .float => |f| f,
                else => 0,
            };
            const ratio = e[0] / gm;
            worst = @max(worst, ratio);
            std.debug.print("  {s} {d:>2}: xpu {e:.2} / {e:.2}   bf16 gap {e:.2} / {e:.2}   mean ratio {d:.2}\n", .{ if (li == 24) "final" else "layer", if (li == 24) 0 else li, e[0], e[1], gm, gx, ratio });
        }
        const top = meta.value.object.get("top1").?.array.items;
        const want_next: u32 = @intCast(top[T - 1].integer);
        std.debug.print("  next token after the prompt: xpu {d}, float64 {d}; worst layer mean ratio {d:.2}\n", .{ next, want_next, worst });
        if (worst > 4.0) bad = true;
        // greedy reply, twice from a fresh prefill: the same tokens
        var reply: [2][reply_len]u32 = undefined;
        var dec_ns: u64 = 0;
        for (0..2) |run_i| {
            try m.reset();
            try m.prefill(ids, null);
            reply[run_i][0] = try m.argmax();
            t0 = nowNs();
            for (1..reply_len) |j| {
                try m.forward(reply[run_i][j - 1 .. j], true, null);
                reply[run_i][j] = try m.argmax();
            }
            dec_ns = nowNs() - t0;
        }
        const same = std.mem.eql(u32, &reply[0], &reply[1]);
        if (!same) bad = true;
        std.debug.print("  {d} greedy tokens, second run {s}; decode {d:.1} tok/s (synchronous argmax per token)\n", .{ reply_len, if (same) "identical" else "DIFFERENT", @as(f64, reply_len - 1) / (@as(f64, @floatFromInt(dec_ns)) / 1e9) });
        try out.writer.print("{s}\"{s}\": {{\"prompt\": [", .{ if (pi == 0) "" else ", ", name });
        for (ids, 0..) |v, k| try out.writer.print("{s}{d}", .{ if (k == 0) "" else ", ", v });
        try out.writer.writeAll("], \"reply\": [");
        for (reply[0], 0..) |v, k| try out.writer.print("{s}{d}", .{ if (k == 0) "" else ", ", v });
        try out.writer.writeAll("]}");
    }
    try out.writer.writeAll("}\n");
    if (args.len > 4) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[4], .data = out.written() });
    std.debug.print("{s}\n", .{if (bad) "lfm3_fwd: FAILED" else "lfm3_fwd: ok"});
    if (bad) return error.TestFailed;
}
