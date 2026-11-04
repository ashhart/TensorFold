//! BF16 vision transforms and full attention with explicit, noncausal visibility.
const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const u = @import("util.zig");

pub const Functions = struct {
    layer_norm: Function,
    gelu: Function,
    rope: Function,
    rows: Function,
    full: Function,

    pub fn load(ops: Module, pre: Module) u.Error!Functions {
        return .{
            .layer_norm = try ops.function("tf_layer_norm_bf16"),
            .gelu = try ops.function("tf_gelu_bf16"),
            .rope = try ops.function("tf_vision_rope_bf16"),
            .rows = try ops.function("tf_attention_rows_bf16"),
            .full = try pre.function("tf_fa_full_bf16"),
        };
    }
};

pub fn layerNorm(l: *const Launcher, x: u64, w: u64, b: u64, out: u64, rows: usize, width: usize, eps: f32, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(x);
    try a.add(w);
    try a.add(b);
    try a.add(out);
    try a.add(@as(c_int, @intCast(width)));
    try a.add(eps);
    try l.go(l.vision.layer_norm, u.dim(rows, 1, 1), u.dim(256, 1, 1), 0, s, &a);
}

pub fn gelu(l: *const Launcher, x: u64, out: u64, n: usize, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(x);
    try a.add(out);
    try a.add(@as(i64, @intCast(n)));
    try l.flat(l.vision.gelu, @intCast(n), s, &a);
}

pub fn rope(l: *const Launcher, x: u64, positions: u64, q: u64, k: u64, v: u64, rows: usize, heads: usize, d: usize, theta: f32, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(x);
    try a.add(positions);
    try a.add(q);
    try a.add(k);
    try a.add(v);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(theta);
    try l.flat(l.vision.rope, @intCast(rows * heads * d), s, &a);
}

pub fn attentionRows(l: *const Launcher, x: u64, out: u64, rows: usize, heads: usize, d: usize, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(x);
    try a.add(out);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(d)));
    try l.flat(l.vision.rows, @intCast(rows * heads * d), s, &a);
}

pub fn full(l: *const Launcher, q: u64, k: u64, v: u64, out: u64, rows: usize, heads: usize, d: usize, scale: f32, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(q);
    try a.add(k);
    try a.add(v);
    try a.add(out);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(scale);
    try a.add(@as(c_int, 0));
    for (0..2) |_| {
        try a.add(@as(i64, @intCast(heads * rows * d)));
        try a.add(@as(i64, @intCast(rows * d)));
        try a.add(@as(i64, @intCast(d)));
    }
    try l.go(l.vision.full, u.dim(u.cdiv(rows, 64), heads, 1), u.dim(256, 1, 1), 0, s, &a);
}
