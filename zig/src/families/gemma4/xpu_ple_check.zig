//! The E4B's per-layer input path on the card against the host steps: xpu-gemma4 MODEL --ple-check IDS (BF16 linears).
const std = @import("std");
const core = @import("core");
const dev = @import("xpu_dev.zig");
const cfg = @import("xpu_config.zig");
const plan = @import("cuda_plan.zig");
const li = @import("cuda_layer_input.zig");
const fwd = @import("xpu_forward.zig");
const bfr = core.bf16_round;
const alloc = dev.alloc;

const g = cfg.geo;
const n_in = cfg.ple_all;

/// What a forward hands the check: embeddings, table rows, per-layer inputs, each layer's residual around its block.
pub const Capture = struct {
    rows: u32,
    emb: []f32,
    tok: []u16,
    pli: []u16,
    pre: []f32,
    post: []f32,

    pub const What = enum { embeddings, inputs, before, after };

    pub fn grab(c: *Capture, r: *fwd.Runner, what: What, layer: usize) !void {
        const t: usize = c.rows;
        switch (what) {
            .embeddings => try r.d.read(std.mem.sliceAsBytes(c.emb), r.a.x),
            .inputs => {
                try r.d.read(std.mem.sliceAsBytes(c.tok), r.a.tok);
                try r.d.read(std.mem.sliceAsBytes(c.pli), r.a.pli);
            },
            .before => try r.d.read(std.mem.sliceAsBytes(c.pre[layer * t * cfg.hidden ..][0 .. t * cfg.hidden]), r.a.x),
            .after => try r.d.read(std.mem.sliceAsBytes(c.post[layer * t * cfg.hidden ..][0 .. t * cfg.hidden]), r.a.x),
        }
    }
};

fn floats(t: dev.Tensor) ![]f32 {
    if (!std.mem.eql(u8, t.dtype, "BF16")) return error.NeedsBf16PerLayerLinears;
    const out = try alloc.alloc(f32, t.bytes.len / 2);
    for (out, 0..) |*o, i| o.* = bfr.value(std.mem.readInt(u16, t.bytes[2 * i ..][0..2], .little));
    return out;
}

const Stat = struct {
    n: usize = 0,
    differing: usize = 0,
    max_steps: f64 = 0,
    max_abs: f64 = 0,
    max_mag: f64 = 0,

    /// |a - b| in bf16 steps of the larger magnitude (a step is 2^-7 of the leading power of two).
    fn steps(a: f64, b: f64) f64 {
        const m = @max(@abs(a), @abs(b));
        if (m == 0) return 0;
        return @abs(a - b) / std.math.ldexp(@as(f64, 1), std.math.frexp(m).exponent - 8);
    }

    fn add(s: *Stat, card: u16, host: f32) void {
        s.n += 1;
        s.max_mag = @max(s.max_mag, @abs(@as(f64, bfr.value(card))));
        if (bfr.bits(host) == card) return;
        s.differing += 1;
        s.max_steps = @max(s.max_steps, steps(bfr.value(card), host));
        s.max_abs = @max(s.max_abs, @abs(@as(f64, bfr.value(card)) - host));
    }

    fn line(s: Stat, label: []const u8) void {
        const pct = @as(f64, @floatFromInt(s.differing)) * 100 / @as(f64, @floatFromInt(@max(s.n, 1)));
        std.debug.print("{s:<22} values {d:8}  differing {d:7} ({d:.3} %)  max {d:.1} bf16 step(s)  max abs {d:.5} (largest value {d:.3})\n", .{ label, s.n, s.differing, pct, s.max_steps, s.max_abs, s.max_mag });
    }
};

pub fn run(r: *fwd.Runner, ids: []const u32) !void {
    const t = ids.len;
    if (t == 0 or t > fwd.ROWS) return error.Usage;
    var c: Capture = .{
        .rows = @intCast(t),
        .emb = try alloc.alloc(f32, t * cfg.hidden),
        .tok = try alloc.alloc(u16, t * n_in),
        .pli = try alloc.alloc(u16, t * n_in),
        .pre = try alloc.alloc(f32, g.layers * t * cfg.hidden),
        .post = try alloc.alloc(f32, g.layers * t * cfg.hidden),
    };
    r.reset();
    r.cap = &c;
    try r.feed(ids, true);
    r.cap = null;
    const model: li.Model = .{
        .project = try floats(try r.ckpt.get(plan.prefix ++ "per_layer_model_projection.weight")),
        .norm = try floats(try r.ckpt.get(plan.prefix ++ "per_layer_projection_norm.weight")),
    };
    // 1. the per-layer inputs of every row, from the card's own embedding and table rows
    const emb = try alloc.alloc(f32, cfg.hidden);
    const table = try alloc.alloc(f32, n_in);
    const out = try alloc.alloc(f32, n_in);
    var s_in: Stat = .{};
    for (0..t) |row| {
        @memcpy(emb, c.emb[row * cfg.hidden ..][0..cfg.hidden]);
        for (table, c.tok[row * n_in ..][0..n_in]) |*d, b| d.* = bfr.value(b) / 16.0;
        li.inputs(g, model, emb, table, out);
        for (out, c.pli[row * n_in ..][0..n_in]) |h, card| s_in.add(card, h);
    }
    s_in.line("inputs()");
    // 2. each layer's block, from the card's own per-layer input and residual
    var s_all: Stat = .{};
    var worst: usize = 0;
    var worst_stat: Stat = .{};
    var scratch: [cfg.ple + cfg.hidden]f32 = undefined;
    const h = try alloc.alloc(f32, cfg.hidden);
    const input = try alloc.alloc(f32, cfg.ple);
    for (0..cfg.layers) |l| {
        var name: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&name, plan.prefix ++ "layers.{d}.", .{l});
        var buf: [192]u8 = undefined;
        const layer: li.Layer = .{
            .gate = try floats(try r.ckpt.get(try std.fmt.bufPrint(&buf, "{s}per_layer_input_gate.weight", .{p}))),
            .project = try floats(try r.ckpt.get(try std.fmt.bufPrint(&buf, "{s}per_layer_projection.weight", .{p}))),
            .norm = try floats(try r.ckpt.get(try std.fmt.bufPrint(&buf, "{s}post_per_layer_input_norm.weight", .{p}))),
            .scalar = (try floats(try r.ckpt.get(try std.fmt.bufPrint(&buf, "{s}layer_scalar", .{p}))))[0],
        };
        var s_l: Stat = .{};
        for (0..t) |row| {
            @memcpy(h, c.pre[(l * t + row) * cfg.hidden ..][0..cfg.hidden]);
            for (input, c.pli[row * n_in + l * cfg.ple ..][0..cfg.ple]) |*d, b| d.* = bfr.value(b);
            li.apply(g, layer, input, h, &scratch);
            for (h, c.post[(l * t + row) * cfg.hidden ..][0..cfg.hidden]) |host, card| {
                s_l.add(bfr.bits(card), host);
                s_all.add(bfr.bits(card), host);
            }
        }
        if (s_l.differing >= worst_stat.differing) {
            worst = l;
            worst_stat = s_l;
        }
        if (std.c.getenv("PLE_LAYERS") != null) {
            var label: [32]u8 = undefined;
            s_l.line(try std.fmt.bufPrint(&label, "apply() layer {d}", .{l}));
        }
    }
    s_all.line("apply() all layers");
    var label: [32]u8 = undefined;
    worst_stat.line(try std.fmt.bufPrint(&label, "  worst layer {d}", .{worst}));
}
