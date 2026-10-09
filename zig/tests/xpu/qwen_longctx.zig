//! Decode speed vs context length, synthetic random KV cache (speed only). usage: xpu-qwen_longctx-test CKPT CTX L,L

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;
const al = @import("qwen_xpu").attn_long;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn parseList(s: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, s, ',');
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, std.mem.trim(u8, t, " "), 10));
    return out.toOwnedSlice(gpa);
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) return error.Usage;
    const dir = args[1];
    const ctx = try std.fmt.parseInt(u32, args[2], 10);
    const lens = try parseList(args[3]);
    var steps: usize = 8;
    var rows: u32 = 0;
    var i: usize = 4;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--rows")) {
            i += 1;
            rows = try std.fmt.parseInt(u32, args[i], 10);
        } else steps = try std.fmt.parseInt(usize, args[i], 10);
    }
    var r = try rt.open();
    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    const t_load = nowNs();
    var m = try model.Model.load(gpa, &r, &l, cfg.value, ctx);
    std.debug.print("loaded in {d:.1} s, device memory {d:.2} GB, context {d}\n", .{ @as(f64, @floatFromInt(nowNs() - t_load)) / 1e9, @as(f64, @floatFromInt(l.total)) / 1e9, ctx });
    // random finite bf16 K/V in every attention layer
    const block: usize = 64 << 20;
    const host = try gpa.alloc(u16, block / 2);
    var prng = std.Random.DefaultPrng.init(11);
    for (host) |*v| v.* = @truncate(@as(u32, @bitCast((prng.random().float(f32) - 0.5) * 4.0)) >> 16);
    var n_attn: usize = 0;
    var qsrc: ?rt.Buffer = null;
    for (m.layers) |ly| switch (ly.mixer) {
        .attn => |a| {
            n_attn += 1;
            if (al.kvMode() != .bf16) { // quantize random rows into the records
                const qrows: u32 = 16384;
                if (qsrc == null) {
                    qsrc = try r.alloc(@as(usize, qrows) * 1024 * 2);
                    try r.upload(qsrc.?, std.mem.sliceAsBytes(host)[0 .. @as(usize, qrows) * 1024 * 2]);
                    try r.sync();
                }
                var p0: u32 = 0;
                while (p0 < ctx) : (p0 += qrows) try m.ops.kvAppend(a.kc, a.vc, qsrc.?, qsrc.?, p0, @min(qrows, ctx - p0));
                try r.sync();
                continue;
            }
            const total = @as(usize, ctx) * 1024 * 2;
            var off: usize = 0;
            while (off < total) {
                const n = @min(block, total - off);
                const dst_k: rt.Buffer = .{ .rt = a.kc.rt, .ptr = @ptrFromInt(@intFromPtr(a.kc.ptr.?) + off), .len = n };
                const dst_v: rt.Buffer = .{ .rt = a.vc.rt, .ptr = @ptrFromInt(@intFromPtr(a.vc.ptr.?) + off), .len = n };
                try r.upload(dst_k, std.mem.sliceAsBytes(host)[0..n]);
                try r.upload(dst_v, std.mem.sliceAsBytes(host)[0..n]);
                try r.sync();
                off += n;
            }
        },
        .gdn => {},
    };
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    std.debug.print("filled the KV cache of {d} attention layers\n", .{n_attn});
    for (lens) |len| {
        try m.reset(zeros);
        m.pos = len;
        var tok: u32 = 760;
        try m.forward(tok, true); // warm-up
        tok = try m.argmax();
        const t0 = nowNs();
        for (0..steps) |_| {
            try m.forward(tok, true);
            tok = try m.argmax();
        }
        const ms = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6 / @as(f64, @floatFromInt(steps));
        std.debug.print("context {d:>6}: decode {d:.2} ms/token = {d:.2} tokens/s", .{ len, ms, 1e3 / ms });
        if (rows > 0) {
            var toks: [16]u32 = undefined;
            for (0..rows) |j| toks[j] = 760 + @as(u32, @intCast(j));
            m.pos = len;
            try m.forwardRows(toks[0..rows], rows, false);
            try m.r.sync();
            var best: u64 = std.math.maxInt(u64);
            for (0..3) |_| {
                m.pos = len;
                const t1 = nowNs();
                try m.forwardRows(toks[0..rows], rows, false);
                try m.r.sync();
                best = @min(best, nowNs() - t1);
            }
            std.debug.print("; {d}-row window {d:.2} ms", .{ rows, @as(f64, @floatFromInt(best)) / 1e6 });
        }
        std.debug.print("\n", .{});
    }
}
