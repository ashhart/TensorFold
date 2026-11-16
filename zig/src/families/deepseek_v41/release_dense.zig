//! The release dense projections as calls (D§5.1, §5.2): an FP8 round trip an input, an MMA launch a projection.

const std = @import("std");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const kr = @import("dsv41_kernels").release;
const Arg = calls.Arg;
const Dt = calls.Dt;
const Emitter = block.Emitter;

/// The extension the runner dispatches to kernels_release.zig (kernels_replay_release.zig).
pub const ext = "tf_dsv41_rel_v1";

/// How a release projection's weight is stored: FP8 codes with 32x32 block scales, or BF16.
pub const Form = enum { fp8, bf16 };

/// An input's FP8 operands: E4M3 codes [m, k] and UE8M0 scales [m, k / 32].
pub const Operands = struct { codes: Arg, scales: Arg };

/// Two arguments are the same input when role, type, shape, strides and offset all match (D§4.1's views).
pub fn sameInput(a: Arg, b: Arg) bool {
    const x = a.t;
    const y = b.t;
    const role = switch (x.role) {
        .weight => |n| y.role == .weight and std.mem.eql(u8, n, y.role.weight),
        .buf => |n| y.role == .buf and std.mem.eql(u8, n, y.role.buf),
        .empty => y.role == .empty,
    };
    return role and x.dt == y.dt and x.offset == y.offset and std.mem.eql(i64, x.shape, y.shape) and std.mem.eql(i64, x.stride, y.stride);
}

/// x [m, k] bf16 (a view) round-tripped into "L.<tag>.q<i>" codes and "L.<tag>.qs<i>" scales.
pub fn actq(e: *Emitter, tag: []const u8, i: usize, x: Arg, m: i64, k: i64) !Operands {
    const codes = try e.buf("L.{s}.q{d}", .{ tag, i }, .u8, &.{ m, k });
    const scales = try e.buf("L.{s}.qs{d}", .{ tag, i }, .u8, &.{ m, @divExact(k, 32) });
    try e.ext(ext ++ ".actq", &.{ x, codes, scales });
    return .{ .codes = codes, .scales = scales };
}

/// The split-K partials a projection of m rows needs, or none: one buffer a stream, sized by its largest user.
fn part(e: *Emitter, m: i64, n: i64, k: i64) !Arg {
    const bytes = kr.partBytes(@intCast(m), @intCast(n), @intCast(k));
    if (bytes == 0) return .none;
    return e.buf("L.rel.{s}", .{if (e.br_side) "part_side" else "part"}, .f32, &.{@intCast(bytes / 4)});
}

/// A weight prefix without its layer ("L12.attn.ix_wk" -> "attn.ix_wk"): buffers are named a site, not a layer.
pub fn tagOf(w: []const u8) []const u8 {
    if (w.len < 2 or w[0] != 'L' or !std.ascii.isDigit(w[1])) return w;
    const dot = std.mem.indexOfScalar(u8, w, '.') orelse return w;
    return w[dot + 1 ..];
}

/// A projection's input: FP8 operands, or the BF16 rows themselves.
const Src = union(enum) { fp8: Operands, bf16: Arg };

/// One projection of `w` (N x K as the rank holds it) from its operands into y's columns [col, col + n).
fn project(e: *Emitter, w: []const u8, form: Form, src: Src, m: i64, k: i64, n: i64, y: Arg, col: i64) !void {
    const p = try part(e, m, n, k);
    switch (form) {
        .fp8 => try e.ext(ext ++ ".fp8", &.{
            src.fp8.codes,                                                    src.fp8.scales,
            try e.weight("{s}.w", .{w}, .u8, &.{ n, k }),                     try e.weight("{s}.s", .{w}, .u8, &.{ @divExact(n, 32), @divExact(k, 32) }),
            y,                                                                .{ .i = col },
            p,
        }),
        .bf16 => try e.ext(ext ++ ".bf16", &.{ src.bf16, try e.weight("{s}.w", .{w}, .bf16, &.{ n, k }), y, .{ .i = col }, p }),
    }
}

/// A weight prefix's form, from the loaded weights (gate_forward.widths).
pub fn formOf(e: *const Emitter, w: []const u8) !Form {
    return e.w.release.get(w) orelse error.NoReleaseForm;
}

/// The release body of Emitter.fused: one round trip a distinct FP8 input, then one launch a projection.
pub fn fused(e: *Emitter, tag: []const u8, ps: []const Emitter.Proj) !void {
    const m = e.n;
    const ops = try e.a.alloc(?Operands, ps.len);
    @memset(ops, null);
    var inputs: usize = 0;
    for (ps, 0..) |p, i| {
        if (try formOf(e, p.w) != .fp8) continue;
        // an input already round-tripped for an earlier projection of the group is reused
        for (ps[0..i], ops[0..i]) |q, o| if (o != null and sameInput(q.x, p.x)) {
            ops[i] = o;
            break;
        };
        if (ops[i] == null) {
            ops[i] = try actq(e, tag, inputs, p.x, m, @intCast(p.k));
            inputs += 1;
        }
    }
    for (ps, ops) |p, o| {
        const form = try formOf(e, p.w);
        try project(e, p.w, form, if (o) |v| Src{ .fp8 = v } else Src{ .bf16 = p.x }, m, @intCast(p.k), @intCast(p.n), p.y, @intCast(p.col));
    }
}

/// The release body of Emitter.upstream: one projection, the whole of y.
pub fn upstream(e: *Emitter, w: []const u8, x: Arg, k: i64, n: i64, y: Arg) !void {
    return fused(e, tagOf(w), &.{.{ .w = w, .x = x, .y = y, .k = @intCast(k), .n = @intCast(n) }});
}

/// The release body of a prefill matmul: y = x [m, K] W^T into `y`'s rows of `stride` from byte `offset`.
pub fn matmul(e: *Emitter, w: []const u8, x: Arg, k: i64, n: i64, role: calls.Role, dt: Dt, stride: i64, offset: i64) !void {
    const y = try e.view(role, dt, &.{ e.n, n }, &.{ stride, 1 }, offset);
    return fused(e, tagOf(w), &.{.{ .w = w, .x = x, .y = y, .k = @intCast(k), .n = @intCast(n) }});
}

test "a weight prefix's buffer tag drops its layer" {
    try std.testing.expectEqualStrings("attn.ix_wk", tagOf("L12.attn.ix_wk"));
    try std.testing.expectEqualStrings("attn.wo_a.1", tagOf("L0.attn.wo_a.1"));
    try std.testing.expectEqualStrings("head", tagOf("head"));
    try std.testing.expectEqualStrings("Lx.y", tagOf("Lx.y"));
}

test "release inputs are the same only when role, type, shape, strides and offset match" {
    const t = struct {
        fn at(name: []const u8, off: i64, comptime stride: i64) Arg {
            return .{ .t = .{ .role = .{ .buf = name }, .dt = .bf16, .shape = &.{ 3, 4096 }, .stride = &.{ stride, 1 }, .offset = off } };
        }
    };
    try std.testing.expect(sameInput(t.at("L.o", 0, 8192), t.at("L.o", 0, 8192)));
    try std.testing.expect(!sameInput(t.at("L.o", 0, 8192), t.at("L.o", 8192, 8192)));
    try std.testing.expect(!sameInput(t.at("L.o", 0, 8192), t.at("L.o", 0, 4096)));
    try std.testing.expect(!sameInput(t.at("L.o", 0, 8192), t.at("L.q", 0, 8192)));
    var w = t.at("L.o", 0, 8192);
    w.t.role = .{ .weight = "L.o" };
    try std.testing.expect(!sameInput(t.at("L.o", 0, 8192), w));
}

test "a release group: one round trip a distinct FP8 input, then a launch a projection, BF16 read as is" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cfg: @import("config.zig").Config = .{};
    var w: block.Widths = .{};
    try w.release.put(a, "L0.attn.wq_a", .fp8);
    try w.release.put(a, "L0.attn.wkv", .fp8);
    try w.release.put(a, "L0.attn.comp_wkv", .bf16);
    var e: Emitter = .{ .a = a, .cfg = &cfg, .w = &w, .o = .{ .format = .release_fp8, .world = 4 }, .n = 1, .start = 0 };
    const D: usize = cfg.hidden;
    const x = try e.buf("L.x", .{}, .bf16, &.{ 1, @intCast(D) });
    const qa = try e.buf("L.qa", .{}, .bf16, &.{ 1, cfg.q_lora });
    const kv = try e.buf("L.kv", .{}, .bf16, &.{ 1, cfg.head_dim });
    const cw = try e.buf("L.cw", .{}, .f32, &.{ 1, cfg.head_dim });
    try e.fused("x", &.{
        .{ .w = "L0.attn.wq_a", .x = x, .y = qa, .k = D, .n = cfg.q_lora },
        .{ .w = "L0.attn.wkv", .x = x, .y = kv, .k = D, .n = cfg.head_dim },
        .{ .w = "L0.attn.comp_wkv", .x = x, .y = cw, .k = D, .n = cfg.head_dim },
    });
    const cs = e.out.items;
    try std.testing.expectEqual(@as(usize, 4), cs.len);
    try std.testing.expectEqualStrings(ext ++ ".actq", cs[0].name);
    try std.testing.expectEqualStrings(ext ++ ".fp8", cs[1].name);
    try std.testing.expectEqualStrings(ext ++ ".fp8", cs[2].name);
    try std.testing.expectEqualStrings(ext ++ ".bf16", cs[3].name);
    // both FP8 projections read the one round trip; wq_a (K 5120, N 1280) splits, so it takes partials
    try std.testing.expectEqualStrings("L.x.q0", cs[1].args[0].arg.t.role.buf);
    try std.testing.expectEqualStrings("L.x.q0", cs[2].args[0].arg.t.role.buf);
    try std.testing.expectEqualStrings("L0.attn.wq_a.w", cs[1].args[2].arg.t.role.weight);
    try std.testing.expectEqualStrings("L.rel.part", cs[1].args[6].arg.t.role.buf);
    try std.testing.expectEqualStrings("L.x", cs[3].args[0].arg.t.role.buf);
    try std.testing.expectEqual(calls.Dt.f32, cs[3].args[2].arg.t.dt);
    // a prefix the loader did not record is refused by name
    try std.testing.expectError(error.NoReleaseForm, e.upstream("L0.attn.ix_wk", x, @intCast(D), 128, kv));
}
