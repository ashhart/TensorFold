//! GLM's prompt experts and dense products in column groups (glm_mm_groups.metal) against core/affine_mm.zig's.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const bits = @import("metal_glm_bits.zig");
const rig_ = @import("rig.zig");
const c = contract.spec;
const g = contract.gates;
const affine_mm = tf.affine_mm;
const groups = tf.glm.mm_groups;

const E = 6; // experts in the pool
const TOPK = 2; // a token's picks: pairs / 2 tokens
const D = 4096;
const N = 2048;
const swiglu_limit: f32 = 10.0; // config.zig's

/// kernels.zig's prompt matmul library: core/affine_mm.metal at 4/64, bf16 out, the column-group gathers appended.
pub fn text(a: std.mem.Allocator, device: mtl.Device) ![]u8 {
    return groups.source(device, a);
}

/// The same library as a GPU without tensor units builds it (nax.h's simdgroup-matrix layout).
fn preM5(a: std.mem.Allocator) ![]u8 {
    const body = try tf.frags.precision.source(a, false, sources.core_affine_mm, sources.nax, false);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ "#define TF_BITS 4\n#define TF_GROUP 64\n#define TF_OUT_T bfloat\n", body, sources.glm_mm_groups });
}

/// Rows: sorted pairs (experts' means 2, 43 and 183: both tile heights of each launch); n: gate/up's width; k: hidden.
const served = [_]c.Shape{ .{ .rows = 1100, .n = N, .k = D }, .{ .rows = 12, .n = N, .k = D }, .{ .rows = 260, .n = N, .k = D } };

const Slot = enum(u8) { x, gw, gs, gb, uw, us, ub, dw, ds, db, offs, order, sums, act, y };

pub const Case = struct {
    control: bool,
    mm: affine_mm.Pipes,
    ours: groups.Pipes,

    pub fn init(s: *bits.Suite, control: bool) !Case {
        const loader = try s.production();
        const own = try text(s.a, s.device);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = if (control) try bits.edited(s.a, own, "inline int gt_order(", "col = g * GT_GROUP + rem % GT_GROUP;", "col = rem % GT_GROUP + 0 * g;") else try s.a.dupe(u8, own);
        defer s.a.free(body);
        var x: Case = .{ .control = control, .mm = undefined, .ours = undefined };
        var made: usize = 0;
        errdefer for (x.mm[0..made]) |p| p.deinit();
        for (affine_mm.names, &x.mm) |n, *p| {
            p.* = try bits.pipeline(s.device, own, n);
            made += 1;
        }
        var ours: usize = 0;
        errdefer for (x.ours[0..ours]) |p| p.deinit();
        for (groups.names, &x.ours) |n, *p| {
            p.* = try bits.pipeline(s.device, body, n);
            ours += 1;
        }
        return x;
    }
    pub fn deinit(x: *Case) void {
        for (x.mm) |p| p.deinit();
        for (x.ours) |p| p.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: metal glm launch prompt experts, every column group taking the first group's columns" else "metal glm launch prompt experts gate/up and down in column groups";
    }
    pub fn spec(x: *const Case) c.Spec {
        return bits.spec(name(x.control), "tf_affine_gather_glu and tf_affine_gather_scatter at 32 and 64 rows", "the one-launch gathers' bytes in every pair's activations and outputs", &served, "core/affine_mm.zig's tiles and sums; only the order the tiles run in changes");
    }
    pub fn layout(_: *const Case, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.rows > 1100 or s.rows % TOPK != 0 or s.n != N or s.k != D) return null;
        const w: tf.glm.affine.Width = .{ .bits = 4, .group = 64 };
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows / TOPK * D * 2, .bf16);
        for ([_][2]usize{ .{ E * N, D }, .{ E * N, D }, .{ E * D, N } }) |m| {
            _ = l.add(.input, tf.glm.affine.wBytes(m[0], m[1], w), .int);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
            _ = l.add(.input, tf.glm.affine.sBytes(m[0], m[1], w), .bf16);
        }
        _ = l.add(.input, E * 4, .int);
        _ = l.add(.input, s.rows * 4, .int);
        _ = l.add(.scratch, s.rows * (D / 64) * 4, .int);
        _ = l.add(.output, s.rows * N * 2, .bf16);
        _ = l.add(.output, s.rows * D * 2, .bf16);
        return l;
    }
    /// Each token two distinct experts of the pool, pairs sorted by expert in pair order (the prompt's sort's).
    pub fn fill(_: *const Case, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (std.mem.bytesAsSlice(u16, data[@backingInt(Slot.x)])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        for (0..3) |m| {
            const at = @backingInt(Slot.gw) + 3 * m;
            rnd.bytes(data[at]);
            for (std.mem.bytesAsSlice(u16, data[at + 1])) |*v| v.* = bits.bf(rnd.float(f32) * 0.004 + 0.0005);
            for (std.mem.bytesAsSlice(u16, data[at + 2])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.05);
        }
        var pick_buf: [1100]i32 = undefined;
        const picks = pick_buf[0..s.rows];
        for (0..s.rows / TOPK) |t| {
            const a = rnd.uintLessThan(usize, E);
            picks[t * TOPK] = @intCast(a);
            picks[t * TOPK + 1] = @intCast((a + 1 + rnd.uintLessThan(usize, E - 1)) % E);
        }
        const offs = std.mem.bytesAsSlice(i32, data[@backingInt(Slot.offs)]);
        const order = std.mem.bytesAsSlice(i32, data[@backingInt(Slot.order)]);
        var counts: [E]i32 = @splat(0);
        for (picks) |p| counts[@intCast(p)] += 1;
        var first: i32 = 0;
        for (offs, counts) |*o, n| {
            o.* = first;
            first += n;
        }
        var fill_at: [E]i32 = undefined;
        for (&fill_at, offs) |*fa, o| fa.* = o;
        for (picks, 0..) |p, i| {
            order[@intCast(fill_at[@intCast(p)])] = @intCast(i);
            fill_at[@intCast(p)] += 1;
        }
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const x = std.mem.bytesAsSlice(u16, data[@backingInt(Slot.x)]);
        const tokens = s.rows / TOPK;
        for ([_][2]usize{ .{ 0, 5 }, .{ 1, 700 }, .{ 2, 64 }, .{ 3, 9 }, .{ 3, 4000 } }, bits.special_bf16[0..5]) |p, v| if (p[0] < tokens) {
            x[p[0] * D + p[1]] = v;
        };
        // the SwiGLU's clip takes NaN and infinities out of the activations: down's scales carry them to its outputs
        const ds = std.mem.bytesAsSlice(u16, data[@backingInt(Slot.ds)]);
        for ([_]usize{ 3, 64 * (N / 64) + 1, 2000 * (N / 64), (E * D - 1) * (N / 64) }, bits.special_bf16[0..4]) |at, v| ds[at] = v;
    }
    /// prompt.zig's routed experts on one Mac at 4/64: row sums, gate/up GLU, the activations' sums, down scattered.
    pub fn encode(x: *const Case, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const n: u32 = @intCast(s.rows);
        const q = struct {
            fn mat(rr: []const rig_.Ref, at: Slot, rows: u32, k: u32) struct { w: rig_.Ref, s: rig_.Ref, b: rig_.Ref, n: u32, k: u32 } {
                const i = @backingInt(at);
                return .{ .w = rr[i], .s = rr[i + 1], .b = rr[i + 2], .n = rows, .k = k };
            }
        }.mat;
        const gate = q(r, .gw, N, D);
        const up = q(r, .uw, N, D);
        const down = q(r, .dw, D, N);
        const xr = r[@backingInt(Slot.x)];
        const sums = r[@backingInt(Slot.sums)];
        const offs = r[@backingInt(Slot.offs)];
        const order = r[@backingInt(Slot.order)];
        const act = r[@backingInt(Slot.act)];
        const y = r[@backingInt(Slot.y)];
        affine_mm.rowSums(e, x.mm, 64, xr, sums, n / TOPK, D);
        if (new) {
            groups.gatherGlu(e, x.ours, xr, sums, gate, up, offs, order, act, n, E, TOPK, swiglu_limit);
        } else affine_mm.gatherGlu(e, x.mm, xr, sums, gate, up, offs, order, act, n, E, TOPK, swiglu_limit);
        affine_mm.rowSums(e, x.mm, 64, act, sums, n, N);
        if (new) {
            groups.gatherTo(e, x.ours, act, sums, down, offs, y, order, n, E);
        } else affine_mm.gatherTo(e, x.mm, act, sums, down, offs, y, order, n, E);
    }
};

/// Rows, n, k: prompt.zig's deep products (a partial last column group at 2240 and 24896 columns).
const dense_served = [_]c.Shape{ .{ .rows = 200, .n = 4096, .k = 8192 }, .{ .rows = 130, .n = 2240, .k = 4096 }, .{ .rows = 70, .n = 24896, .k = 4096 } };

const DSlot = enum(u8) { x, w, s, b, sums, y };

pub const Dense = struct {
    control: bool,
    mm: affine_mm.Pipes,
    ours: groups.Pipes,

    pub fn init(s: *bits.Suite, control: bool) !Dense {
        const loader = try s.production();
        const own = try text(s.a, s.device);
        defer s.a.free(own);
        _ = try loader.require(own);
        const body = if (control) try bits.edited(s.a, own, "void glm_mm_grouped(", "const int row = (rem / gl) * 64", "const int row = (rem / GT_GROUP) * 64") else try s.a.dupe(u8, own);
        defer s.a.free(body);
        var x: Dense = .{ .control = control, .mm = undefined, .ours = undefined };
        var made: usize = 0;
        errdefer for (x.mm[0..made]) |p| p.deinit();
        for (affine_mm.names, &x.mm) |n, *p| {
            p.* = try bits.pipeline(s.device, own, n);
            made += 1;
        }
        var ours: usize = 0;
        errdefer for (x.ours[0..ours]) |p| p.deinit();
        for (groups.names, &x.ours) |n, *p| {
            p.* = try bits.pipeline(s.device, body, n);
            ours += 1;
        }
        return x;
    }
    pub fn deinit(x: *Dense) void {
        for (x.mm) |p| p.deinit();
        for (x.ours) |p| p.deinit();
    }
    pub fn name(control: bool) []const u8 {
        return if (control) "control: metal glm launch prompt dense, the last column group's row tiles counted as a whole group's" else "metal glm launch prompt dense products in column groups";
    }
    pub fn spec(x: *const Dense) c.Spec {
        return bits.spec(name(x.control), "tf_affine_mm at one batch", "tf_affine_mm's bytes in every output", &dense_served, "core/affine_mm.zig's tiles and sums; only the order the tiles run in changes");
    }
    pub fn layout(_: *const Dense, s: c.Shape) ?bits.Layout {
        if (s.rows == 0 or s.n % 64 != 0 or s.k % 64 != 0 or s.k < groups.dense_from) return null;
        const w: tf.glm.affine.Width = .{ .bits = 4, .group = 64 };
        var l: bits.Layout = .{};
        _ = l.add(.input, s.rows * s.k * 2, .bf16);
        _ = l.add(.input, tf.glm.affine.wBytes(s.n, s.k, w), .int);
        _ = l.add(.input, tf.glm.affine.sBytes(s.n, s.k, w), .bf16);
        _ = l.add(.input, tf.glm.affine.sBytes(s.n, s.k, w), .bf16);
        _ = l.add(.scratch, s.rows * (s.k / 64) * 4, .int);
        _ = l.add(.output, s.rows * s.n * 2, .bf16);
        return l;
    }
    pub fn fill(_: *const Dense, s: c.Shape, f: g.Fill, seed: u64, data: [][]u8) !void {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (std.mem.bytesAsSlice(u16, data[@backingInt(DSlot.x)])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 2);
        rnd.bytes(data[@backingInt(DSlot.w)]);
        for (std.mem.bytesAsSlice(u16, data[@backingInt(DSlot.s)])) |*v| v.* = bits.bf(rnd.float(f32) * 0.004 + 0.0005);
        for (std.mem.bytesAsSlice(u16, data[@backingInt(DSlot.b)])) |*v| v.* = bits.bf((rnd.float(f32) - 0.5) * 0.05);
        if (f == .stress or (f == .random and seed % 3 != 2)) return;
        const x = std.mem.bytesAsSlice(u16, data[@backingInt(DSlot.x)]);
        for ([_][2]usize{ .{ 0, 5 }, .{ 1, 700 }, .{ 2, 64 }, .{ 3, 9 }, .{ 69, 4000 } }, bits.special_bf16[0..5]) |p, v| if (p[0] < s.rows) {
            x[p[0] * s.k + p[1]] = v;
        };
    }
    /// prompt.zig's qmm: the rows' group sums, then the product (ours past mm_groups.dense_from deep).
    pub fn encode(x: *const Dense, e: mtl.ComputeEncoder, r: []const rig_.Ref, s: c.Shape, new: bool) void {
        const rows: u32 = @intCast(s.rows);
        const q = .{ .w = r[@backingInt(DSlot.w)], .s = r[@backingInt(DSlot.s)], .b = r[@backingInt(DSlot.b)], .n = @as(u32, @intCast(s.n)), .k = @as(u32, @intCast(s.k)) };
        const xr = r[@backingInt(DSlot.x)];
        const sums = r[@backingInt(DSlot.sums)];
        const y = r[@backingInt(DSlot.y)];
        affine_mm.rowSums(e, x.mm, 64, xr, sums, rows, q.k);
        if (new) {
            groups.dense(e, x.ours, xr, sums, q, y, rows, .{ .x_row = @intCast(q.k), .y_row = @intCast(q.n) });
        } else affine_mm.dense(e, x.mm, xr, sums, q, y, rows);
    }
};

pub fn collect(s: *bits.Suite, only: ?[]const u8) !void {
    for ([_]bool{ false, true }) |control| {
        if (s.wanted(only, Case.name(control))) try s.add(Case, try Case.init(s, control));
        if (s.wanted(only, Dense.name(control))) try s.add(Dense, try Dense.init(s, control));
    }
}

/// Gate 5's registrations `glm launch prompt <experts|dense> <function>`: the loader's prompt matmul library.
pub const prefix = "glm launch prompt ";
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, _: []const u8) !bits.Texts {
    return .{ .here = try text(a, device), .pre_m5 = try preM5(a) };
}
