//! GLM's dense latent attention for a window (dense_attn.zig's attendRows) against main's one-row launches, row by row.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const glm = tf.glm;

const H = 64;
const RANK = 512;
const TOP = 2048;
const lat = H * RANK; // a row's latent queries and outputs
const plane = H * TOP; // and its scores and probabilities
/// Main's glm_attn.metal, before the window kernels: the one-row scores and values every window row must equal.
pub const main_attn = @embedFile("baselines/glm_attn_main.metal.txt");

/// Rows: the window's dense rows; n: the heads; k: the first row's keys (edges: the 256-key join, the 1,024-key split).
const served = [_]c.Shape{ .{ .rows = 8, .n = H, .k = 400 }, .{ .rows = 1, .n = H, .k = 1 }, .{ .rows = 16, .n = H, .k = 1 }, .{ .rows = 3, .n = H, .k = 5 }, .{ .rows = 2, .n = H, .k = 31 }, .{ .rows = 4, .n = H, .k = 33 }, .{ .rows = 16, .n = H, .k = 240 }, .{ .rows = 2, .n = H, .k = 255 }, .{ .rows = 1, .n = H, .k = 256 }, .{ .rows = 3, .n = H, .k = 700 }, .{ .rows = 16, .n = H, .k = 1010 }, .{ .rows = 2, .n = H, .k = 1024 }, .{ .rows = 8, .n = H, .k = 1500 }, .{ .rows = 16, .n = H, .k = 2033 }, .{ .rows = 1, .n = H, .k = 2048 } };

pub const Fault = enum { none, halves, softmax, split, prefetch };

/// The loader's latent attention library on this GPU (nax.h inlined for it).
pub fn attnText(a: std.mem.Allocator, device: mtl.Device) ![]u8 {
    return tf.frags.source(device, a, sources.glm_attn);
}

/// kernels.zig's window softmax: ops/softmax.metal with glm_softmax_rows appended.
pub const softmax_rows_text = sources.ops_softmax ++ sources.glm_softmax_rows;

pub const Case = struct {
    a: std.mem.Allocator,
    fault: Fault,
    control: bool,
    kern: glm.kernels.Kernels = undefined,
    old: [3]mtl.Pipeline,
    cfg: glm.config.Config = .{},

    pub fn init(s: *bits.Suite, fault: Fault) !Case {
        const loader = try s.production();
        const attn = try attnText(s.a, s.device);
        defer s.a.free(attn);
        _ = try loader.require(attn);
        _ = try loader.require(softmax_rows_text);
        const faults = [_][3][]const u8{ .{ "", "", "" }, .{ "void glm_latent_scores_rows(", "n, n < 256,", "n, n < 255," }, .{ "void glm_softmax_rows(", "a.n0 + int(tg.y)", "a.n0 + int(tg.y) + 1" }, .{ "void glm_latent_values_rows(", "n > 1024 ? 1024 : 0", "n > 1024 ? 512 : 0" }, .{ "inline void glm_scores_at(", "mma_16x32<false, true>(c0, c1, ca, cb0, cb1);", "mma_16x32<false, true>(c0, c1, a, cb0, cb1);" } };
        const f = faults[@backingInt(fault)];
        const own_attn = if (fault == .halves or fault == .split or fault == .prefetch) try bits.edited(s.a, attn, f[0], f[1], f[2]) else try s.a.dupe(u8, attn);
        defer s.a.free(own_attn);
        const own_soft = if (fault == .softmax) try bits.edited(s.a, softmax_rows_text, f[0], f[1], f[2]) else try s.a.dupe(u8, softmax_rows_text);
        defer s.a.free(own_soft);
        var x: Case = .{ .a = s.a, .fault = fault, .control = fault != .none, .old = undefined };
        x.kern.latent_scores_rows = try bits.pipeline(s.device, own_attn, "glm_latent_scores_rows");
        errdefer x.kern.latent_scores_rows.deinit();
        x.kern.latent_values_rows = try bits.pipeline(s.device, own_attn, "glm_latent_values_rows");
        errdefer x.kern.latent_values_rows.deinit();
        x.kern.softmax_rows = try bits.pipeline(s.device, own_soft, "glm_softmax_rows");
        errdefer x.kern.softmax_rows.deinit();
        const frozen = try tf.frags.source(s.device, s.a, main_attn);
        defer s.a.free(frozen);
        x.old[0] = try bits.pipeline(s.device, frozen, "glm_latent_scores");
        errdefer x.old[0].deinit();
        x.old[1] = try bits.pipeline(s.device, try loader.require(sources.ops_softmax), "tf_softmax_bf16");
        errdefer x.old[1].deinit();
        x.old[2] = try bits.pipeline(s.device, frozen, "glm_latent_values");
        return x;
    }
    pub fn deinit(x: *Case) void {
        x.kern.latent_scores_rows.deinit();
        x.kern.latent_values_rows.deinit();
        x.kern.softmax_rows.deinit();
        for (x.old) |p| p.deinit();
    }
    pub fn name(fault: Fault) []const u8 {
        return switch (fault) {
            .none => "metal glm launch attn rows glm_latent_scores_rows, glm_softmax_rows, glm_latent_values_rows",
            .halves => "control: metal glm launch attn rows, the scores' two-half join from 255 keys",
            .softmax => "control: metal glm launch attn rows, each row's softmax one key long",
            .split => "control: metal glm launch attn rows, the values split at 512 keys",
            .prefetch => "control: metal glm launch attn rows, a scores step multiplying the prefetched next query block",
        };
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.fault), "main's glm_latent_scores, tf_softmax_bf16 and glm_latent_values a row", "main's one-row bytes in the scores, probabilities and outputs, NaN payloads included", &served, "the one-row kernels' 16-wide MMA steps, both joins and the softmax's sums in their order");
    }
    pub fn layout(_: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > 16 or s.n != H or s.k == 0 or s.k + s.rows - 1 > TOP) return null;
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * lat * 2, .bf16);
        _ = l.add(.input, (s.k + s.rows - 1) * RANK * 2, .bf16);
        _ = l.add(.output, s.rows * plane * 2, .bf16);
        _ = l.add(.output, s.rows * plane * 2, .bf16);
        _ = l.add(.output, s.rows * lat * 2, .bf16);
        return l;
    }
    /// The harness's queries and keys; gate 1 and every third shape plant NaN and both infinities in a query and keys.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const q = std.mem.bytesAsSlice(u16, data[0]);
        const keys = std.mem.bytesAsSlice(u16, data[1]);
        for (q) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.25);
        for (keys) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        q[3] = 0x7fc0;
        const early: usize = if (s.k > 2) 2 else s.k - 1;
        keys[early * RANK + 5] = 0x7f80;
        keys[(s.k + s.rows - 2) * RANK + 9] = 0xff80;
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const R = glm.weights.Ref;
        const ref = struct {
            fn of(v: rig_.Ref) R {
                return .{ .buf = v.buf, .off = v.off };
            }
        }.of;
        if (new) {
            const ctx: glm.forward.Ctx = .{ .k = &x.kern, .c = &x.cfg, .w = undefined, .s = undefined, .sc = undefined };
            return glm.forward.attendRows(&ctx, e, ref(r[1]), ref(r[0]), ref(r[2]), ref(r[3]), ref(r[4]), @intCast(s.k), @intCast(s.rows));
        }
        for (0..s.rows) |z| x.oneRow(e, r, z, @intCast(s.k + z));
    }
    /// Main's attendDense for window row z (forward.zig before the window kernels): its three launches as they were.
    fn oneRow(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, z: usize, n: u32) void {
        const bind = struct {
            fn at(enc: mtl.ComputeEncoder, v: rig_.Ref, extra: usize, i: usize) void {
                enc.setBuffer(v.buf, v.off + extra, i);
            }
        }.at;
        e.setPipeline(x.old[0]);
        bind(e, r[0], z * lat * 2, 0);
        bind(e, r[1], 0, 1);
        bind(e, r[2], z * plane * 2, 2);
        e.setValue([2]i32{ @intCast(n), @intFromBool(n < 256) }, 3);
        e.dispatchGroups(mtl.Size.of((n + 31) / 32, 1, 1), mtl.Size.of(2 * H, 1, 1));
        e.setPipeline(x.old[1]);
        bind(e, r[2], z * plane * 2, 0);
        e.setBytes(std.mem.asBytes(&[1]i32{@intCast(n)}), 1);
        bind(e, r[3], z * plane * 2, 2);
        e.dispatchGroups(mtl.Size.of(H, 1, 1), mtl.Size.of(((n + 3) / 4 + 31) / 32 * 32, 1, 1));
        e.setPipeline(x.old[2]);
        bind(e, r[3], z * plane * 2, 0);
        bind(e, r[1], 0, 1);
        bind(e, r[4], z * lat * 2, 2);
        e.setValue([2]i32{ @intCast(n), if (n > 1024) 1024 else 0 }, 3);
        e.dispatchGroups(mtl.Size.of(RANK / 32, 1, 1), mtl.Size.of(2 * H, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for (std.enums.values(Fault)) |fault| {
        if (s.wanted(only, Case.name(fault))) try s.add(Case, try Case.init(s, fault));
    }
}

/// Gate 5's registrations `glm launch attn <function>` and `glm launch softmax <function>`.
pub const prefix = "glm launch attn ";
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, _: []const u8) !bits.Texts {
    return .{ .here = try attnText(a, device), .pre_m5 = try tf.frags.precision.source(a, false, sources.glm_attn, sources.nax, false) };
}
pub const softmax = struct {
    pub const prefix = "glm launch softmax ";
    pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !bits.Texts {
        return .{ .here = softmax_rows_text, .pre_m5 = softmax_rows_text };
    }
};
