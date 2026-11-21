//! GLM's MLA norms in one launch (forward.zig's mlaNorms, glm_mla_norms) against main's four launches on x_proj rows.
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

const cfg: glm.config.Config = .{};
const Q = cfg.q_lora;
const R = cfg.kv_lora;
const IK = cfg.i_dim;
const IW = cfg.i_heads;
/// Main's glm_glue.metal: glm_rms, glm_layer_norm and glm_scale as their own launches.
const main_glue = @import("metal_glm_attn.zig").main_glue;

/// Four roles with the query norm (decode, x_proj rows xProj apart), three without (a prompt chunk's, rows 64-aligned).
pub const Roles = enum { four, three };
fn stride(r: Roles) usize {
    return if (r == .four) cfg.xProj() else std.mem.alignForward(usize, cfg.xProj(), 64);
}
/// Rows: x_proj rows; n: the roles; k: a row's stride.
const served = [2][5]c.Shape{
    .{ .{ .rows = 8, .n = 4, .k = stride(.four) }, .{ .rows = 1, .n = 4, .k = stride(.four) }, .{ .rows = 2, .n = 4, .k = stride(.four) }, .{ .rows = 5, .n = 4, .k = stride(.four) }, .{ .rows = 16, .n = 4, .k = stride(.four) } },
    .{ .{ .rows = 8, .n = 3, .k = stride(.three) }, .{ .rows = 1, .n = 3, .k = stride(.three) }, .{ .rows = 16, .n = 3, .k = stride(.three) }, .{ .rows = 64, .n = 3, .k = stride(.three) }, .{ .rows = 512, .n = 3, .k = stride(.three) } },
};

pub const Case = struct {
    roles: Roles,
    control: bool,
    kern: glm.kernels.Kernels = undefined,
    old: [3]mtl.Pipeline,

    pub fn init(s: *bits.Suite, roles: Roles, control: bool) !Case {
        const loader = try s.production();
        const glue = try loader.require(sources.glm_glue);
        const body = if (control) try bits.edited(s.a, glue, "void glm_mla_norms(", "glm_ln_row(x + a.ik_at, ik_w,", "glm_ln_row(x + a.ik_at, kv_w,") else try s.a.dupe(u8, glue);
        defer s.a.free(body);
        var x: Case = .{ .roles = roles, .control = control, .old = undefined };
        x.kern.mla_norms = try bits.pipeline(s.device, body, "glm_mla_norms");
        errdefer x.kern.mla_norms.deinit();
        var made: usize = 0;
        errdefer for (x.old[0..made]) |p| p.deinit();
        for ([_][]const u8{ "glm_rms", "glm_layer_norm", "glm_scale" }, 0..) |n, i| {
            x.old[i] = try bits.pipeline(s.device, main_glue, n);
            made += 1;
        }
        return x;
    }
    pub fn deinit(x: *Case) void {
        x.kern.mla_norms.deinit();
        for (x.old) |p| p.deinit();
    }
    pub fn name(roles: Roles, control: bool) []const u8 {
        if (control) return if (roles == .four) "control: metal glm launch mla norms four roles, the key norm with the kv norm's weights" else "control: metal glm launch mla norms three roles, the key norm with the kv norm's weights";
        return if (roles == .four) "metal glm launch mla norms glm_mla_norms four roles (kv, key, index weights, query)" else "metal glm launch mla norms glm_mla_norms three roles (kv, key, index weights)";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.roles, x.control), "main's glm_rms, glm_layer_norm and glm_scale launches", "main's bytes in the latent keys, indexer keys, index weights and query rows, NaN payloads included", &served[@backingInt(x.roles)], "MLX's rms and layer-norm sums in main's order, the weights' exact scale");
    }
    pub fn layout(x: *const Case, s: c.Shape) ?bits.Layout {
        const n: usize = if (x.roles == .four) 4 else 3;
        if (s.rows == 0 or s.n != n or s.k != stride(x.roles)) return null;
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * s.k * 2, .bf16);
        _ = l.add(.input, R * 2, .bf16);
        _ = l.add(.input, IK * 2, .bf16);
        _ = l.add(.input, IK * 2, .bf16);
        _ = l.add(.input, Q * 2, .bf16);
        _ = l.add(.output, s.rows * R * 2, .bf16);
        _ = l.add(.output, s.rows * IK * 2, .bf16);
        _ = l.add(.output, s.rows * IW * 2, .bf16);
        if (x.roles == .four) _ = l.add(.output, s.rows * Q * 2, .bf16);
        return l;
    }
    /// x_proj rows and norm weights; gate 1 and every third shape plant NaN and infinities in each role's columns.
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const xp = std.mem.bytesAsSlice(u16, data[0]);
        for (xp) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 6);
        for (data[1..5], 0..) |d, i| for (std.mem.bytesAsSlice(u16, d)) |*v| {
            v.* = bits.bf(if (i == 2) (rnd.float(f32) - 0.5) * 0.2 else 0.5 + rnd.float(f32));
        };
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const at = [_][2]usize{ .{ 0, Q + 3 }, .{ 1, Q + R + 5 }, .{ 2, Q + R + IK + 7 }, .{ 3, 11 }, .{ 3, Q + 40 } };
        for (at, [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80, 0xff80 }) |p, v| if (p[0] < s.rows) {
            xp[p[0] * s.k + p[1]] = v;
        };
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const rows: u32 = @intCast(s.rows);
        const xs: u32 = @intCast(s.k);
        const ref = struct {
            fn of(v: rig_.Ref, extra: usize) glm.weights.Ref {
                return .{ .buf = v.buf, .off = v.off + extra };
            }
        }.of;
        if (new) {
            var w: glm.weights.Mla = undefined;
            w.kv_norm = ref(r[1], 0);
            w.k_norm_w = ref(r[2], 0);
            w.k_norm_b = ref(r[3], 0);
            w.q_norm = ref(r[4], 0);
            return glm.forward.mlaNorms(&x.kern, &cfg, e, &w, ref(r[0], 0), xs, ref(r[5], 0), ref(r[6], 0), ref(r[7], 0), if (x.roles == .four) ref(r[8], 0) else null, rows);
        }
        const Rows = glm.forward.Rows;
        const bind = struct {
            fn at(enc: mtl.ComputeEncoder, v: rig_.Ref, extra: usize, i: usize) void {
                enc.setBuffer(v.buf, v.off + extra, i);
            }
        }.at;
        e.setPipeline(x.old[0]); // main's mlaCache: the kv norm into the cache, the key norm, the index weights
        bind(e, r[0], Q * 2, 0);
        bind(e, r[1], 0, 1);
        bind(e, r[5], 0, 2);
        e.setValue(Rows{ .rows = @intCast(rows), .width = R, .x_stride = @intCast(xs), .y_stride = R, .eps = cfg.eps }, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(((R + 3) / 4 + 31) / 32 * 32, 1, 1));
        e.setPipeline(x.old[1]);
        bind(e, r[0], (Q + R) * 2, 0);
        bind(e, r[2], 0, 1);
        bind(e, r[3], 0, 2);
        bind(e, r[6], 0, 3);
        e.setValue(Rows{ .rows = @intCast(rows), .width = IK, .x_stride = @intCast(xs), .y_stride = IK, .eps = 1e-6 }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
        e.setPipeline(x.old[2]);
        bind(e, r[0], (Q + R + IK) * 2, 0);
        bind(e, r[7], 0, 1);
        e.setValue(Rows{ .rows = @intCast(rows), .width = IW, .x_stride = @intCast(xs), .y_stride = IW, .eps = 1.0 / 64.0 }, 2);
        e.dispatchThreads(mtl.Size.of(IW, rows, 1), mtl.Size.of(IW, 1, 1));
        if (x.roles != .four) return;
        e.setPipeline(x.old[0]); // main's mlaAttend: the query norm from the same rows
        bind(e, r[0], 0, 0);
        bind(e, r[4], 0, 1);
        bind(e, r[8], 0, 2);
        e.setValue(Rows{ .rows = @intCast(rows), .width = Q, .x_stride = @intCast(xs), .y_stride = Q, .eps = cfg.eps }, 3);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(((Q + 3) / 4 + 31) / 32 * 32, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]Roles{ .four, .three }) |roles| for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(roles, control))) try s.add(Case, try Case.init(s, roles, control));
    };
}

/// Gate 5's registration `glm launch mla norms <function>`: the loader's glue library.
pub const prefix = "glm launch mla norms ";
pub fn limitsText(_: std.mem.Allocator, _: mtl.Device, _: []const u8) !bits.Texts {
    return .{ .here = sources.glm_glue, .pre_m5 = sources.glm_glue };
}
