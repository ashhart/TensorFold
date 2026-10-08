//! The embedding check and the captured-activation run of nem_fp64.zig.
const std = @import("std");
const f = @import("nem_fp64.zig");
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;

pub fn embed(c: *f.Ctx) !void {
    if (c.only.len > 0 and std.mem.indexOf(u8, "embed", c.only) == null) return;
    const m = c.m;
    const cfg = m.cfg;
    const h: usize = cfg.hidden_size;
    const hst = try f.Host.fetch(c.r, m.emb, cfg.vocab_size, h);
    defer hst.free();
    const n: usize = 16;
    const dev = try f.gpa.alloc(f64, n * h);
    defer f.gpa.free(dev);
    const ref = try f.gpa.alloc(f64, n * h);
    defer f.gpa.free(ref);
    for (0..n) |i| {
        const tok: u32 = if (c.real) c.ptoks[c.ptoks.len - 64 + i] else @intCast(c.prng.random().uintLessThan(usize, cfg.vocab_size));
        try f.run(&m.k.embed_tok, .{ 1, 1, 1 }, .{ m.emb.w, m.emb.s, m.emb.b, tok, c.yb, @as(u32, @intCast(h)) });
        try c.r.sync();
        const o = try c.dev16(c.yb, h);
        defer f.gpa.free(o);
        @memcpy(dev[i * h ..][0..h], o);
        for (0..h) |j| {
            const g = j / 64;
            const pk = hst.w[tok * (h / 8) + j / 8];
            const code: f64 = @floatFromInt((pk >> @as(u5, @intCast(4 * (j % 8)))) & 15);
            ref[i * h + j] = code * f.bf(hst.s[tok * (h / 64) + g]) + f.bf(hst.b[tok * (h / 64) + g]);
        }
    }
    f.compare(c, "embed", 0, "decode", "embed4_tok", n, h, h, dev, ref, true);
}

var gc: *f.Ctx = undefined;
var capturing = false;

/// nw.capture: records the activations the classes read (one row a call: the tail runs one token at a time).
fn onCapture(w: *nw.Win, m: *model.Model, li: usize, st: nw.Stage, n: u32) anyerror!void {
    if (!capturing or n != 1) return;
    const dim: usize = switch (st) {
        .xn => m.cfg.hidden_size,
        .yn => m.cfg.xd(),
        .att => m.cfg.num_attention_heads * m.cfg.head_dim,
    };
    const b = switch (st) {
        .xn => w.xn,
        .yn => w.yn,
        .att => w.att,
    };
    try m.r.sync();
    const row = try gc.dev16(b, dim);
    defer f.gpa.free(row);
    const g = try gc.caps.getOrPut(f.gpa, li * 4 + @intFromEnum(st));
    if (!g.found_existing) g.value_ptr.* = .empty;
    try g.value_ptr.appendSlice(f.gpa, row);
}

/// Prefills all but the last 64 tokens in windows of 512, then runs those one at a time, recording each class input.
pub fn captureRun(c: *f.Ctx) !void {
    const tail: usize = 64;
    const prompt = c.ptoks;
    if (prompt.len < tail + 2) return error.PromptTooShort;
    const head = prompt.len - tail;
    var t: usize = 0;
    while (t < head) {
        const n = @min(512, head - t);
        try c.w.forward(c.m, prompt[t .. t + n], true, 0);
        t += n;
    }
    try c.r.sync();
    gc = c;
    capturing = true;
    nw.capture = onCapture;
    for (0..tail) |i| {
        try c.w.forward(c.m, prompt[head + i ..][0..1], true, 1);
        try c.r.sync();
        const row = try c.dev16(c.w.xn, c.m.cfg.hidden_size);
        defer f.gpa.free(row);
        const g = try c.caps.getOrPut(f.gpa, 3);
        if (!g.found_existing) g.value_ptr.* = .empty;
        try g.value_ptr.appendSlice(f.gpa, row);
    }
    capturing = false;
    nw.capture = null;
}

