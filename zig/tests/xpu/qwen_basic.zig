//! 4-bit matvec, rmsnorm, embedding and lm_head + argmax on real weights vs numpy. usage: xpu-qwen-basic-test [DIR]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cq = @import("qwen_xpu").config;
const qb = @import("qwen_xpu").blocks;
const ql = @import("qwen_xpu").load;
const c = @import("qwen_common.zig");

const spv = @import("xpu").kernels.qwen_basic;
const Fix = struct { name: []const u8, prefix: []const u8, rows: u32, in: u32, x: []const u8, y: []const u8 };
fn mats() ![]const Fix {
    const S = struct {
        var v: [3]Fix = undefined;
        var done = false;
    };
    if (!S.done) {
        S.v = [_]Fix{
    .{ .name = "in_proj_qkv", .prefix = "linear_attn.in_proj_qkv", .rows = 10240, .in = 5120, .x = (try tfix.load("qwen_basic_qkv_x")), .y = (try tfix.load("qwen_basic_qkv_y")) },
    .{ .name = "out_proj", .prefix = "linear_attn.out_proj", .rows = 5120, .in = 6144, .x = (try tfix.load("qwen_basic_out_x")), .y = (try tfix.load("qwen_basic_out_y")) },
    .{ .name = "down_proj", .prefix = "mlp.down_proj", .rows = 5120, .in = 17408, .x = (try tfix.load("qwen_basic_down_x")), .y = (try tfix.load("qwen_basic_down_y")) },
};
        S.done = true;
    }
    return &S.v;
}
const emb_ids = [_]u32{ 5, 1000, 248000 };

/// Largest bf16 bit distance and the count of mismatching values.
fn compareBits(name: []const u8, got: []const u16, want: []const u16) !void {
    var worst: u32 = 0;
    var off: usize = 0;
    for (got, want) |g, w| {
        const d: u32 = @abs(@as(i32, g) - @as(i32, w));
        if (d > 0) off += 1;
        worst = @max(worst, d);
    }
    std.debug.print("{s}: {d} values, {d} differ, worst {d} ulp\n", .{ name, got.len, off, worst });
    if (worst > 1) return error.TooInaccurate;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    const dir = try c.checkpoint(init);
    var r = try rt.open();
    defer r.deinit();
    const cfg = try cq.parse(c.gpa, try ld.readFile(c.gpa, dir, "config.json"));
    var l = try ld.Loader.init(c.gpa, &r, dir);
    var m = try r.module(spv);
    var qmv = try m.kernel("qmv4_f32", .{ 64, 1, 1 });

    for (try mats()) |f| {
        var buf: [160]u8 = undefined;
        const t = try ql.table(c.gpa, &l, try std.fmt.bufPrint(&buf, "language_model.model.layers.0.{s}", .{f.prefix}), f.rows, f.in);
        const x = try c.up(&r, f.x);
        const y = try r.alloc(f.rows * 4);
        try qmv.setBuffer(0, t.w);
        try qmv.setBuffer(1, t.s);
        try qmv.setBuffer(2, t.b);
        try qmv.setBuffer(3, x);
        try qmv.setBuffer(4, y);
        try qmv.setU32(5, f.in);
        try qmv.setU32(6, 0);
        try qmv.setU32(7, 0);
        try qmv.setU32(8, f.rows);
        try qmv.launch(.{ (f.rows + 3) / 4, 1, 1 });
        const got = try c.fetchF32(&r, y, f.rows);
        const want = try c.alignedF32(f.y);
        var mx: f32 = 0;
        for (want) |v| mx = @max(mx, @abs(v));
        var worst: f32 = 0;
        for (got, want) |g, w| worst = @max(worst, @abs(g - w) / mx);
        std.debug.print("qmv4_f32 {s} {d}x{d}: worst {e:.2} of max|y|\n", .{ f.name, f.rows, f.in, worst });
        if (worst > 1e-5) return error.TooInaccurate;
    }

    var ops = try qb.Ops.init(&r, &l, cfg.value, 16);
    try ql.attach(c.gpa, &ops, &l, cfg.value);
    // rmsnorm with layer 0 input_layernorm
    {
        const nw = try l.load("language_model.model.layers.0.input_layernorm.weight");
        const xb = (try tfix.load("qwen_basic_rms_x"));
        const x = try c.up(&r, xb);
        const y = try r.alloc(xb.len);
        var k = try m.kernel("rmsnorm", .{ 64, 1, 1 });
        try k.setBuffer(0, x);
        try k.setBuffer(1, nw);
        try k.setBuffer(2, y);
        try k.setU32(3, qb.hidden);
        try k.setF32(4, cfg.value.text_config.rms_norm_eps);
        try k.launch(.{ 3, 1, 1 });
        try compareBits("rmsnorm 3x5120", try c.fetch(&r, y, xb.len / 2), try c.aligned((try tfix.load("qwen_basic_rms_y"))));
    }
    // embedding rows
    {
        const t = try ql.table(c.gpa, &l, "language_model.model.embed_tokens", cfg.value.text_config.vocab_size, qb.hidden);
        const want = try c.aligned((try tfix.load("qwen_basic_emb_y")));
        var got: [3 * qb.hidden]u16 = undefined;
        for (emb_ids, 0..) |id, i| {
            try ops.embed(t, id);
            const row = try c.fetch(&r, ops.s.x, qb.hidden);
            @memcpy(got[i * qb.hidden ..][0..qb.hidden], row);
        }
        try compareBits("embed 3 rows", &got, want);
    }
    // final norm + lm_head + argmax
    {
        const nw = try l.load("language_model.model.norm.weight");
        const head = try ql.table(c.gpa, &l, "language_model.lm_head", cfg.value.text_config.vocab_size, qb.hidden);
        try r.upload(ops.s.x, (try tfix.load("qwen_basic_head_x")));
        const vocab = cfg.value.text_config.vocab_size;
        try ops.head(nw, head, vocab, false);
        const id = try ops.argmax();
        const got = try c.fetchF32(&r, ops.s.logits, vocab);
        const want = try c.alignedF32((try tfix.load("qwen_basic_head_y")));
        var mx: f32 = 0;
        var best: u32 = 0;
        for (want, 0..) |v, i| {
            mx = @max(mx, @abs(v));
            if (v > want[best]) best = @intCast(i);
        }
        var worst: f32 = 0;
        for (got, want) |g, w| worst = @max(worst, @abs(g - w) / mx);
        std.debug.print("lm_head {d} logits: worst {e:.2} of max|logit|, argmax {d} (numpy {d})\n", .{ vocab, worst, id, best });
        if (worst > 1e-4 or id != best) return error.TooInaccurate;
    }
    std.debug.print("qwen basic: ok\n", .{});
}
