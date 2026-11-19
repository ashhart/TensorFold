//! GLM's hyper-connection boundary (core/hc.zig's launches) against main's combine, expand and split, same streams.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const hc = tf.glm.hc;
const shape = tf.glm.kernels.hc_shape;

const D = shape.width;
const F = 4 * D;
const eps: f32 = 1e-5; // forward.zig's boundary passes the norm's eps
/// Main's hc_boundary.metal: one group a slice for the expand, one a row for the split.
pub const main_hc = @embedFile("baselines/hc_boundary_main.metal.txt");

/// A boundary's form: expanded into another buffer, in place (a prompt chunk), a first read, or a MoE block's combine.
pub const Mode = enum { expand, in_place, first, moe };
pub const Fault = enum { none, parts_weights, split_post, moe_last };
const topk = shape.topk;
/// Main's MoE combine before the boundary: the generated moe_combine with the shared expert's row in the last slot.
pub fn combineText(a: std.mem.Allocator) ![]u8 {
    const comb = comptime tf.glm.kernels.generatedSource("moe_combine");
    const first = try std.mem.replaceOwned(u8, a, comb, "const device bfloat* y = Y + size_t(r) * TOPK * D + d;", "const device bfloat* y = Y + size_t(r) * (TOPK + 1) * D + d;");
    defer a.free(first);
    return std.mem.replaceOwned(u8, a, first, "+ YS[size_t(r) * D + d];", "+ y[size_t(TOPK) * D];");
}

/// Rows: the boundary's rows (to 4: the expand's parts on six groups a slice); n: the streams; k: the width.
const served = [4][7]c.Shape{
    .{ .{ .rows = 4, .n = 4, .k = D }, .{ .rows = 1, .n = 4, .k = D }, .{ .rows = 2, .n = 4, .k = D }, .{ .rows = 3, .n = 4, .k = D }, .{ .rows = 5, .n = 4, .k = D }, .{ .rows = 8, .n = 4, .k = D }, .{ .rows = 16, .n = 4, .k = D } },
    .{ .{ .rows = 4, .n = 4, .k = D }, .{ .rows = 1, .n = 4, .k = D }, .{ .rows = 2, .n = 4, .k = D }, .{ .rows = 3, .n = 4, .k = D }, .{ .rows = 5, .n = 4, .k = D }, .{ .rows = 16, .n = 4, .k = D }, .{ .rows = 64, .n = 4, .k = D } },
    .{ .{ .rows = 4, .n = 4, .k = D }, .{ .rows = 1, .n = 4, .k = D }, .{ .rows = 2, .n = 4, .k = D }, .{ .rows = 3, .n = 4, .k = D }, .{ .rows = 5, .n = 4, .k = D }, .{ .rows = 8, .n = 4, .k = D }, .{ .rows = 16, .n = 4, .k = D } },
    .{ .{ .rows = 2, .n = 4, .k = D }, .{ .rows = 1, .n = 4, .k = D }, .{ .rows = 3, .n = 4, .k = D }, .{ .rows = 4, .n = 4, .k = D }, .{ .rows = 5, .n = 4, .k = D }, .{ .rows = 8, .n = 4, .k = D }, .{ .rows = 16, .n = 4, .k = D } },
};

/// The loader's boundary library: hc.zig's defines over core/hc_boundary.metal.
pub fn text(a: std.mem.Allocator) ![]u8 {
    return hc.source(a, shape);
}

const S = struct { x: usize, branch: usize, post: usize, comb: usize, fnp: usize, x_new: usize, part: usize, scale: usize, base: usize, norm: usize, normed: usize, wts: usize = 0, mixed: usize = 0 };

pub const Case = struct {
    a: std.mem.Allocator,
    mode: Mode,
    fault: Fault,
    control: bool,
    pipes: [hc.names.len]mtl.Pipeline,
    old: [4]mtl.Pipeline,
    combine: ?mtl.Pipeline = null,

    pub fn init(s: *bits.Suite, mode: Mode, fault: Fault) !Case {
        const loader = try s.production();
        const own = try text(s.a);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = switch (fault) {
            .none => try s.a.dupe(u8, own),
            .parts_weights => try bits.edited(s.a, own, "void tf_hc_expand_mix_parts(", "size_t(og * 8 + int(sg))", "size_t((og + 1) % 6 * 8 + int(sg))"),
            .split_post => try bits.edited(s.a, own, "void tf_hc_split(", "simd_shuffle(mix, ushort(S + llane))", "simd_shuffle(mix, ushort(llane))"),
            .moe_last => try bits.edited(s.a, own, "inline void hc_branch4(", "for (int k = 1; k < TF_HC_TOPK; k++)", "for (int k = 1; k < TF_HC_TOPK - 1; k++)"),
        };
        defer s.a.free(body);
        var x: Case = .{ .a = s.a, .mode = mode, .fault = fault, .control = fault != .none, .pipes = undefined, .old = undefined };
        var made: usize = 0;
        errdefer for (x.pipes[0..made]) |p| p.deinit();
        for (hc.names, 0..) |n, i| {
            x.pipes[i] = try bits.pipeline(s.device, body, n);
            made += 1;
        }
        const frozen = try std.fmt.allocPrint(s.a, "#define TF_D {d}\n#define TF_ITERS {d}\n#define TF_HC_EPS_INT {d}\n{s}", .{ shape.width, shape.sinkhorn, shape.eps_e9, main_hc });
        defer s.a.free(frozen);
        var old_made: usize = 0;
        errdefer for (x.old[0..old_made]) |p| p.deinit();
        for ([_][]const u8{ "tf_hc_expand_mix", "tf_hc_first_mix", "tf_hc_split", "tf_hc_expand_inplace" }, 0..) |n, i| {
            x.old[i] = try bits.pipeline(s.device, frozen, n);
            old_made += 1;
        }
        if (mode == .moe) {
            const comb = try combineText(s.a);
            defer s.a.free(comb);
            x.combine = try bits.pipeline(s.device, try loader.require(comb), comptime tf.glm.kernels.generatedFunction("moe_combine"));
        }
        return x;
    }
    pub fn deinit(x: *Case) void {
        for (x.pipes) |p| p.deinit();
        for (x.old) |p| p.deinit();
        if (x.combine) |p| p.deinit();
    }
    pub fn name(mode: Mode, fault: Fault) []const u8 {
        const names = [4][4][]const u8{
            .{ "metal glm launch hc boundary expand", "control: metal glm launch hc boundary expand, a parts group reading the next group's mix weights", "control: metal glm launch hc boundary expand, the split's post from the pre mixes", "" },
            .{ "metal glm launch hc boundary in place", "control: metal glm launch hc boundary in place, a parts group reading the next group's mix weights", "control: metal glm launch hc boundary in place, the split's post from the pre mixes", "" },
            .{ "metal glm launch hc boundary first read", "control: metal glm launch hc boundary first read, a parts group reading the next group's mix weights", "control: metal glm launch hc boundary first read, the split's post from the pre mixes", "" },
            .{ "metal glm launch hc boundary combining a MoE block", "control: metal glm launch hc boundary combining a MoE block, a parts group reading the next group's mix weights", "", "control: metal glm launch hc boundary combining a MoE block, the last pick left out" },
        };
        return names[@backingInt(mode)][@backingInt(fault)];
    }
    pub fn spec(x: *const Case) c.Spec {
        var sp = bits.spec(name(x.mode, x.fault), if (x.mode == .moe) "main's MoE combine, tf_hc_expand_mix and tf_hc_split" else "main's tf_hc_expand_mix and tf_hc_split", "main's bytes in the streams, partial sums, normed row, post and comb (raw words, signed zeros and NaN payloads included)", &served[@backingInt(x.mode)], "the expand's fma sequence, each part's simd and serial sums, the split's slice order and its Sinkhorn steps");
        sp.in_place = true;
        return sp;
    }
    fn slots(mode: Mode) S {
        return if (mode == .in_place) .{ .x = 0, .branch = 1, .post = 2, .comb = 3, .fnp = 4, .x_new = 0, .part = 5, .scale = 6, .base = 7, .norm = 8, .normed = 9 } else .{ .x = 0, .branch = 1, .post = 2, .comb = 3, .fnp = 4, .x_new = 5, .part = 6, .scale = 7, .base = 8, .norm = 9, .normed = 10, .wts = 11, .mixed = 12 };
    }
    pub fn layout(x: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.n != 4 or s.k != D) return null;
        var l: bits.Layout = .{};
        _ = l.add(if (x.mode == .in_place) .inout else .input, s.rows * F * 2, .bf16);
        _ = l.add(.input, s.rows * D * 2 * @as(usize, if (x.mode == .moe) topk + 1 else 1), .bf16);
        _ = l.add(.inout, s.rows * 4 * 4, .f32);
        _ = l.add(.inout, s.rows * 16 * 4, .f32);
        _ = l.add(.input, 24 * F * 2, .bf16);
        if (x.mode != .in_place) _ = l.add(.output, s.rows * F * 2, .bf16);
        _ = l.add(.output, s.rows * hc.partBytes(shape), .f32);
        _ = l.add(.input, 16, .f32);
        _ = l.add(.input, 24 * 4, .f32);
        _ = l.add(.input, D * 2, .bf16);
        _ = l.add(.output, s.rows * D * 2, .bf16);
        if (x.mode == .moe) {
            _ = l.add(.input, s.rows * topk * 4, .f32);
            _ = l.add(.scratch, s.rows * D * 2, .bf16);
        }
        return l;
    }
    /// tf-glm-hc-rows' draws; odd seeds add -0 and large streams; gate 1 and every third shape plant NaN, infinities.
    pub fn fill(x: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const i = slots(x.mode);
        const w: f32 = if (seed % 2 == 1) 16 else 1;
        const xs = std.mem.bytesAsSlice(u16, data[i.x]);
        for (xs, 0..) |*v, j| v.* = if (seed % 2 == 1 and j % 61 == 0) 0x8000 else if (seed % 2 == 1 and j % 67 == 0) bits.bf(3e4) else bits.bf((rnd.float(f32) - 0.5) * 4 * w);
        const br = std.mem.bytesAsSlice(u16, data[i.branch]);
        for (br) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2 * w);
        for (std.mem.bytesAsSlice(f32, data[i.post])) |*v| v.* = rnd.float(f32) * 2;
        for (std.mem.bytesAsSlice(f32, data[i.comb])) |*v| v.* = rnd.float(f32) * 0.5;
        for (std.mem.bytesAsSlice(u16, data[i.fnp])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.05 * w);
        @memcpy(std.mem.bytesAsSlice(f32, data[i.scale]), &[4]f32{ 0.9, 1.1, 0.7, 0 });
        if (x.mode == .moe) for (std.mem.bytesAsSlice(f32, data[i.wts])) |*v| {
            v.* = rnd.float(f32) * 0.5;
        };
        for (std.mem.bytesAsSlice(f32, data[i.base])) |*v| v.* = (rnd.float(f32) - 0.5) * w;
        for (std.mem.bytesAsSlice(u16, data[i.norm])) |*v| v.* = bits.bf(0.5 + rnd.float(f32));
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const at = [_][3]usize{ .{ 0, 0, 7 }, .{ 1, 1, 100 }, .{ 2, 0, 3 * D + 9 }, .{ 3, 0, 11 }, .{ 3, 0, D + 12 } };
        const vals = [_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7f80, 0xff80 };
        for (at, vals) |p, v| if (p[0] < s.rows) {
            if (p[1] == 0) xs[p[0] * F + p[2]] = v else br[p[0] * (br.len / s.rows) + p[2]] = v;
        };
    }
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const i = slots(x.mode);
        const rows: u32 = @intCast(s.rows);
        const expand = x.mode != .first;
        if (new and x.mode == .moe) return hc.boundary(e, x.pipes, shape, true, rows, eps, .{ .x_old = r[i.x], .branch = r[i.branch], .post = r[i.post], .comb = r[i.comb], .fn_packed = r[i.fnp], .x_new = r[i.x_new], .part = r[i.part], .scale = r[i.scale], .base = r[i.base], .norm = r[i.norm], .normed = r[i.normed], .wts = r[i.wts] });
        if (new) return hc.boundary(e, x.pipes, shape, expand, rows, eps, .{ .x_old = r[i.x], .branch = r[i.branch], .post = r[i.post], .comb = r[i.comb], .fn_packed = r[i.fnp], .x_new = r[i.x_new], .part = r[i.part], .scale = r[i.scale], .base = r[i.base], .norm = r[i.norm], .normed = r[i.normed] });
        const bind = struct {
            fn at(enc: mtl.ComputeEncoder, v: rig_.Ref, j: usize) void {
                enc.setBuffer(v.buf, v.off, j);
            }
        }.at;
        var branch = i.branch;
        if (x.combine) |p| { // moe.zig's combine_0 into the branch, as main launched it before the boundary
            e.setPipeline(p);
            for ([_]usize{ i.branch, i.branch, i.wts }, 0..) |j, k| bind(e, r[j], k);
            e.setBytes(std.mem.asBytes(&[2]i32{ @intCast(rows), topk }), 3);
            bind(e, r[i.mixed], 4);
            e.dispatchThreads(mtl.Size.of(s.rows * D, 1, 1), mtl.Size.of(256, 1, 1));
            branch = i.mixed;
        }
        const one = x.mode == .in_place; // main's boundary: the expand in place, then the first read's sums
        if (one) {
            e.setPipeline(x.old[3]);
            for ([_]usize{ i.x, branch, i.post, i.comb }, 0..) |j, k| bind(e, r[j], k);
            e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(D / 4, 1, 1));
        }
        e.setPipeline(x.old[if (expand and !one) 0 else 1]);
        for ([_]usize{ i.x, branch, i.post, i.comb, i.fnp, i.x_new, i.part }, 0..) |j, k| bind(e, r[j], k);
        e.dispatchGroups(mtl.Size.of(4 * D / 1024, s.rows, 1), mtl.Size.of(256, 1, 1));
        e.setPipeline(x.old[2]);
        for ([_]usize{ if (expand) i.x_new else i.x, i.part, i.scale, i.base, i.norm }, 0..) |j, k| bind(e, r[j], k);
        e.setValue(eps, 5);
        for ([_]usize{ i.normed, i.post, i.comb }, 6..) |j, k| bind(e, r[j], k);
        e.dispatchGroups(mtl.Size.of(s.rows, 1, 1), mtl.Size.of(1024, 1, 1));
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for (std.enums.values(Mode)) |mode| for (std.enums.values(Fault)) |fault| {
        if (Case.name(mode, fault).len == 0) continue;
        if (s.wanted(only, Case.name(mode, fault))) try s.add(Case, try Case.init(s, mode, fault));
    };
}

/// Gate 5's registrations `glm launch hc <function>`: the loader's boundary library.
pub const prefix = "glm launch hc ";
pub fn limitsText(a: std.mem.Allocator, _: mtl.Device, _: []const u8) !bits.Texts {
    const t = try text(a);
    return .{ .here = t, .pre_m5 = t };
}
