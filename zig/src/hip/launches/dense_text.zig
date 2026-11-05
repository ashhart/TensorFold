//! BF16 dense text embeddings and the Torch BF16 RoPE rounding graph.
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const u = @import("util.zig");

pub const Functions = struct {
    embed: Function,
    rope: Function,
    pub fn load(m: Module) u.Error!Functions {
        return .{ .embed = try m.function("tf_embed_bf16"), .rope = try m.function("tf_rope_torch_bf16") };
    }
};

pub fn embed(l: *const Launcher, table: u64, ids: u64, out: u64, rows: usize, width: usize, vocab: usize, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(table);
    try a.add(ids);
    try a.add(out);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(width)));
    try a.add(@as(c_int, @intCast(vocab)));
    try l.flat(l.dense_text.embed, @intCast(rows * width), s, &a);
}

pub fn rope(l: *const Launcher, x: u64, positions: u64, out: u64, out_kind: c_int, rows: usize, heads: usize, d: usize, theta: f32, cache: bool, total: usize, s: u.S) u.Error!void {
    var a: u.Args = .{};
    try a.add(x);
    try a.add(positions);
    try a.add(out);
    try a.add(out_kind);
    try a.add(@as(c_int, @intCast(rows)));
    try a.add(@as(c_int, @intCast(heads)));
    try a.add(@as(c_int, @intCast(d)));
    try a.add(theta);
    try a.add(@as(c_int, @intFromBool(cache)));
    try a.add(@as(c_int, @intCast(total)));
    try l.flat(l.dense_text.rope, @intCast(rows * heads * d), s, &a);
}
