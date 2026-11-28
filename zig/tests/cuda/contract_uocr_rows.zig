//! The decoder's row kernels under the contract: token embeddings and the image rows' scatter into them.
const std = @import("std");
const cuda = @import("cuda");
const rig_ = @import("contract_rig.zig");
const hub = @import("contract_uocr.zig");
const v = @import("contract_uocr_vision.zig");

const c = v.c;
const g = v.g;
const ks = hub.kernels;
const H = hub.hidden;
/// The test's embedding table rows (the kernels index it by id alone).
const table = 1024;
const image_token: u32 = 128815;

pub const Kind = enum { embed, scatter };

pub fn function(k: Kind) []const u8 {
    return @tagName(k);
}

fn control(fun: [:0]const u8, what: []const u8, gate: g.Gate) v.Control {
    return .{ .function = fun, .name = what, .gate = gate };
}

/// Each kernel's planted control (contract_uocr_glue.cuh).
pub const controls = [_]struct { Kind, v.Control }{
    .{ .embed, control("tf_ctl_uocr_embed_next", "control: cuda uocr tf_uocr_embed, each token reads the next id's row", .shapes) },
    .{ .scatter, control("tf_ctl_uocr_scatter_prev", "control: cuda uocr tf_uocr_scatter, each slot takes the previous row", .shapes) },
};

/// Rows: tokens or scattered rows; n: the row width.
pub fn spec(k: Kind) c.Spec {
    return switch (k) {
        .embed => .{ .name = "cuda uocr tf_uocr_embed", .accum = .none, .depth = v.noSums, .out = .bf16, .served = &.{ .{ .rows = 2048, .n = H, .k = 1 }, .{ .rows = 37, .n = H, .k = 1 }, .{ .rows = 1, .n = H, .k = 1 } } },
        .scatter => .{ .name = "cuda uocr tf_uocr_scatter", .accum = .none, .depth = v.noSums, .out = .bf16, .served = &.{ .{ .rows = 1000, .n = H, .k = 1 }, .{ .rows = 273, .n = H, .k = 1 }, .{ .rows = 100, .n = H, .k = 1 } } },
    };
}

pub const Row = struct {
    rig: rig_.Rig,
    k: ks.Kernels,
    kind: Kind,
    control: ?v.Control,
    shape: c.Shape = undefined,
    /// bf16 inputs (embed: table, prior rows; scatter: rows, prior slots).
    src: [2][]u16 = .{ &.{}, &.{} },
    /// The ids or destination slots.
    idx: []u32 = &.{},
    ins: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, k: ks.Kernels, kind: Kind, ctl: ?v.Control) Row {
        return .{ .rig = rig, .k = k, .kind = kind, .control = ctl };
    }
    pub fn deinit(t: *Row) void {
        t.free();
        t.rig.deinit();
    }
    fn free(t: *Row) void {
        const a = t.rig.gpu.gpa;
        for (&t.src) |*h| {
            a.free(h.*);
            h.* = &.{};
        }
        a.free(t.idx);
        t.idx = &.{};
    }
    pub fn kernel(t: *Row) g.Kernel {
        return v.mark(.{ .spec = spec(t.kind), .ctx = t, .vt = &vt }, t.control);
    }
    const vt = g.hooks(Row);

    /// Slots a scatter writes into.
    fn slots(t: *const Row) usize {
        return t.shape.rows + t.shape.rows / 2 + 3;
    }

    fn alloc(t: *Row, i: usize, n: usize, rng: std.Random, lo: f32, hi: f32) ![]u16 {
        t.src[i] = try t.rig.gpu.gpa.alloc(u16, n);
        v.fill(rng, t.src[i], lo, hi);
        return t.src[i];
    }

    pub fn prepare(t: *Row, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != H) return false;
        t.rig.reset();
        t.free();
        t.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const a = t.rig.gpu.gpa;
        const r = shape.rows;
        const P = v.planted;
        const nf = fill == .nonfinite;
        switch (t.kind) {
            .embed => {
                // every 7th token an image token, whose row keeps what it held; ids below the table's last row
                t.idx = try a.alloc(u32, r);
                for (t.idx, 0..) |*id, i| id.* = if (i % 7 == 3) image_token else rng.uintLessThan(u32, table - 1);
                const tab = try t.alloc(0, table * H, rng, -4, 4);
                const prior = try t.alloc(1, r * H, rng, -4, 4);
                if (nf) {
                    for (P, 0..) |val, i| if (i < r and i != 3) {
                        tab[t.idx[i] * H + i] = c.toBf16(val);
                    };
                    if (r > 3) prior[3 * H + 5] = c.toBf16(P[0]);
                }
            },
            .scatter => {
                const m = t.slots();
                // distinct slots: a permutation's first r entries
                t.idx = try a.alloc(u32, r);
                const perm = try a.alloc(u32, m);
                defer a.free(perm);
                for (perm, 0..) |*p, i| p.* = @intCast(i);
                rng.shuffle(u32, perm);
                @memcpy(t.idx, perm[0..r]);
                const y = try t.alloc(0, r * H, rng, -4, 4);
                _ = try t.alloc(1, m * H, rng, -4, 4);
                if (nf) for (P, 0..) |val, i| {
                    y[(i % r) * H + i] = c.toBf16(val);
                };
            },
        }
        for (t.src, 0..) |h, i| if (h.len > 0) {
            t.ins[i] = try t.rig.input(std.mem.sliceAsBytes(h));
        };
        if (t.idx.len > 0) t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.idx));
        t.out = try t.rig.output(t.count() * 2);
        return true;
    }

    pub fn run(t: *Row) !void {
        const s = t.rig.s;
        const r = t.shape.rows;
        // both leave rows alone: each launch starts from the prior rows copied into the output
        const bytes = t.count() * 2;
        try (cuda.DeviceBuffer{ .d = t.rig.gpu.d, .ptr = t.out, .len = bytes }).copyFrom(0, t.ins[1], bytes, s.handle);
        switch (t.kind) {
            .embed => try t.k.run("embed", s, .{ r, ks.blocks(H), 1 }, 256, .{ t.ins[2], t.ins[0], t.out, image_token }),
            .scatter => try t.k.run("scatter", s, .{ r, ks.blocks(H), 1 }, 256, .{ t.ins[0], t.ins[2], t.out }),
        }
        try s.synchronize();
    }

    pub fn count(t: *Row) usize {
        return switch (t.kind) {
            .embed => t.shape.rows * H,
            .scatter => t.slots() * H,
        };
    }

    fn at(t: *const Row, i: usize, j: usize) f64 {
        return c.fromBf16(t.src[i][j]);
    }

    pub fn reference(t: *Row, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        switch (t.kind) {
            .embed => for (want, 0..) |*o, i| {
                const id = t.idx[i / H];
                o.* = if (id == image_token) t.at(1, i) else t.at(0, id * H + i % H);
            },
            .scatter => {
                for (want, 0..) |*o, i| o.* = t.at(1, i);
                for (t.idx, 0..) |d, r| for (0..H) |col| {
                    want[d * H + col] = t.at(0, r * H + col);
                };
            },
        }
    }
};

/// The row kernels and their controls, appended to `list`.
pub fn register(ctx: hub.Ctx, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !hub.Held {
    return v.register(@This(), Row, ctx, a, list);
}
