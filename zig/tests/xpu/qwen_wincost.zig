//! Window cost: wall time of R-row windows vs a plain step. usage: ...-test CKPT [R,R,...] [PROFILE_R]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;
const windows = 16;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn step(m: *model.Model, r: *rt.Runtime, tokens: []const u32, i: usize, rows: u32) !void {
    if (rows == 1) try m.forward(tokens[i], true) else try m.forwardRows(tokens[i .. i + rows], rows, true);
    try r.sync();
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    var r = try rt.open();
    defer r.deinit();
    const want_prof = @import("xpu").rt.prof_on;
    @import("xpu").rt.prof_on = false;
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 512);
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const tokens = try gpa.alloc(u32, 16 * windows + 16);
    const prompt = [_]u32{ 760, 6511, 314, 9338, 369 };
    for (tokens, 0..) |*t, i| t.* = prompt[i % prompt.len] + @as(u32, @intCast(i));
    const rs: []const u8 = if (args.len > 2) args[2] else "1,2,3,4,5,6,8";
    const prof_r: u32 = if (args.len > 3) try std.fmt.parseInt(u32, args[3], 10) else 0;
    var it = std.mem.tokenizeScalar(u8, rs, ',');
    var base_ms: f64 = 0;
    while (it.next()) |s| {
        const rows = try std.fmt.parseInt(u32, s, 10);
        try m.reset(zeros);
        try step(&m, &r, tokens, 0, 1);
        var best: f64 = 1e18;
        var sum: f64 = 0;
        var i: usize = 1;
        for (0..windows) |_| {
            const t0 = nowNs();
            try step(&m, &r, tokens, i, rows);
            const dt = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
            best = @min(best, dt);
            sum += dt;
            i += rows;
        }
        if (rows == 1) base_ms = best;
        std.debug.print("R={d}: min {d:.2} ms, mean {d:.2} ms, x{d:.2} of plain min ({d:.2} ms/row)\n", .{ rows, best, sum / windows, best / base_ms, best / @as(f64, @floatFromInt(rows)) });
        if (want_prof and (prof_r == rows or prof_r == 0 and (rows == 1 or rows == 4))) {
            @import("xpu").rt.prof_on = true;
            for (0..4) |_| {
                try step(&m, &r, tokens, i, rows);
                i += rows;
            }
            @import("xpu").rt.prof_on = false;
            var lab: [16]u8 = undefined;
            rt.profDump(std.fmt.bufPrint(&lab, "R={d} x4", .{rows}) catch "");
        }
    }
}
