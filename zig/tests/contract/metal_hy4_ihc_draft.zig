//! ihc.metal's pointwise kernels against float64: the streams' residual mix and DFly2's slot-0 fusion and residual add.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");
const rig_ = @import("rig.zig");
const h = @import("metal_hy4.zig");
const grp = @import("metal_hy4_ihc.zig");
const m = @import("metal_hy4_ihc_ref.zig");

const c = contract.spec;
const g = contract.gates;
const S = m.streams;
const pt = m.pt;
const step = m.step;
const bf = m.bf;
const bv = m.bv;

pub const Kind = enum { mix, fuse, residual };

pub const Case = struct { name: []const u8, kind: Kind, ctl: ?[]const u8 = null };

const base = "metal hy4 ihc ";
pub const cases = [_]Case{
    .{ .name = base ++ "tf_hy4_ihc_add", .kind = .mix },
    .{ .name = base ++ "tf_hy4_draft_fuse", .kind = .fuse },
    .{ .name = base ++ "tf_hy4_draft_residual", .kind = .residual },
    .{ .name = base ++ "tf_hy4_ihc_add", .kind = .mix, .ctl = "onegate" },
    .{ .name = base ++ "tf_hy4_draft_fuse", .kind = .fuse, .ctl = "gain0" },
    .{ .name = base ++ "tf_hy4_draft_residual", .kind = .residual, .ctl = "subtract" },
};

/// k: the mix's 4 streams; the fusion's 6-row block a lane; the residual's base stride in rows (block 1, column 6).
const served_mix = [_]c.Shape{ .{ .rows = h.window_rows, .n = h.hidden, .k = S }, .{ .rows = h.decode_rows, .n = h.hidden, .k = S }, .{ .rows = h.prompt_rows, .n = h.hidden, .k = S } };
const served_fuse = [_]c.Shape{.{ .rows = 1, .n = h.hidden, .k = h.window_rows }};
const served_residual = [_]c.Shape{ .{ .rows = h.window_rows, .n = h.hidden, .k = 1 }, .{ .rows = 1, .n = h.hidden, .k = h.window_rows } };

fn spec(v: Case, name: []const u8) c.Spec {
    var s: c.Spec = .{ .name = name, .accum = .none, .depth = h.fl.noSums, .out = .bf16, .served = &served_mix, .tiles = .{ .{}, .{ .size = 256, .step = 32, .min = 32 }, .{} } };
    switch (v.kind) {
        .mix => {},
        .fuse => {
            s.served = &served_fuse;
            s.in_place = true;
            s.tiles[1] = .{ .size = 128, .step = 32, .min = 32 };
        },
        .residual => {
            s.served = &served_residual;
            s.tiles[1] = .{ .size = 128, .step = 32, .min = 32 };
        },
    }
    return s;
}

fn function(k: Kind) []const u8 {
    return switch (k) {
        .mix => "tf_hy4_ihc_add",
        .fuse => "tf_hy4_draft_fuse",
        .residual => "tf_hy4_draft_residual",
    };
}

pub const Elem = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    srv: *const grp.K,
    ctl: ?grp.Ctl,
    /// The draft's own pipeline (Draft.open compiles ihc.metal for it) or a control's.
    pipe: ?mtl.Pipeline,
    v: Case,
    p: m.Params,
    label: [224]u8 = undefined,
    name: []const u8 = "",
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    y: []u16 = &.{},
    z: []u16 = &.{},
    post: []f32 = &.{},
    ins: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, srv: *const grp.K, v: Case, p: m.Params) !Elem {
        const k = if (v.ctl) |slug| grp.control(slug) else null;
        const pipe: ?mtl.Pipeline = if (k) |x| try grp.pipeline(a, device, x) else if (v.kind == .mix) null else try draftPipe(device, function(v.kind));
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .srv = srv, .ctl = k, .pipe = pipe, .v = v, .p = p };
    }
    pub fn deinit(t: *Elem) void {
        t.free();
        t.rig.deinit();
        if (t.pipe) |p| p.deinit();
    }
    fn free(t: *Elem) void {
        inline for (.{ "x", "y", "z", "post" }) |f| {
            t.a.free(@field(t, f));
            @field(t, f) = &.{};
        }
    }
    pub fn kernel(t: *Elem) g.Kernel {
        t.name = h.fl.named(t.v.name, if (t.ctl) |k| k.drops else null, &t.label);
        return .{ .spec = spec(t.v, t.name), .ctx = t, .vt = &vt, .control = if (t.ctl) |k| k.gate else null };
    }
    const vt = g.hooks(Elem);

    /// x: the residual streams, embedded blocks or base; y: the branch or delta; z: the fusion's gain; post: the mix's.
    pub fn prepare(t: *Elem, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0) return false;
        switch (t.v.kind) {
            .mix => if (shape.k != S) return false,
            .fuse => if (shape.k != h.window_rows) return false,
            .residual => if (shape.k != 1 and shape.k != h.window_rows) return false,
        }
        t.rig.reset();
        t.free();
        t.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        t.x = try t.a.alloc(u16, rows * n * shape.k);
        t.y = try t.a.alloc(u16, rows * n);
        t.z = try t.a.alloc(u16, if (t.v.kind == .fuse) n else 0);
        t.post = try t.a.alloc(f32, if (t.v.kind == .mix) rows * S else 0);
        for (t.x) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 4);
        for (t.y) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        for (t.z) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (t.post) |*v| v.* = 0.25 + rng.float(f32) * 1.5;
        if (fill == .nonfinite) t.plant();
        const xb = std.mem.sliceAsBytes(t.x);
        // the fusion's embedded rows are its output, restored before each run (it writes in place)
        if (t.v.kind == .fuse) t.out = try t.rig.output(xb.len) else t.ins[0] = try t.rig.input(xb);
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(t.y));
        if (t.v.kind == .fuse) t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.z));
        if (t.v.kind == .mix) t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(t.post));
        if (t.v.kind == .mix) t.out = try t.rig.output(xb.len);
        if (t.v.kind == .residual) t.out = try t.rig.output(rows * n * 2);
        return true;
    }

    /// NaN and both infinities in x (a row each, one row holding both), and in each of the other inputs.
    fn plant(t: *Elem) void {
        const n = t.shape.n;
        const k = t.shape.k;
        const rows = t.shape.rows;
        if (rows < 5 or n < 8) return;
        // the fusion updates each block's first row only: plant there
        const row = n * (if (t.v.kind == .fuse) k else if (t.v.kind == .residual) k else S);
        const spots = [_][2]usize{ .{ 0, 3 }, .{ 1, 5 }, .{ 2, 1 }, .{ 3, 4 }, .{ 3, 6 } };
        const vals = [_]f32{ h.fl.planted[0], h.fl.planted[1], h.fl.planted[2], h.fl.planted[1], h.fl.planted[2] };
        for (spots, vals) |q, v| t.x[q[0] * row + q[1]] = c.toBf16(v);
        t.y[4 * n + 2] = c.toBf16(h.fl.planted[0]);
        if (t.v.kind == .fuse) t.z[7] = c.toBf16(h.fl.planted[1]);
        if (t.v.kind == .mix) t.post[4 * S + 1] = h.fl.planted[2];
    }

    pub fn run(t: *Elem) !void {
        const rows: u32 = @intCast(t.shape.rows);
        const n: u32 = @intCast(t.shape.n);
        if (t.v.kind == .fuse) @memcpy(t.rig.bytes(t.out, t.x.len * 2), std.mem.sliceAsBytes(t.x));
        const p = t.rig.begin();
        const e = p.e;
        switch (t.v.kind) {
            .mix => {
                const k = grp.with(t.srv.*, t.pipe, .mix);
                const R = h.hy.ihc_metal.Ref;
                const shape: h.hy.ihc.Shape = .{ .rows = rows, .width = n, .gates = 2 * S, .eps = t.p.eps, .gate_eps = t.p.gate_eps, .post_magnitude = t.p.magnitude };
                try k.add(e, .{ .residual = R{ .buffer = t.ins[0].buf, .offset = t.ins[0].off }, .branch = .{ .buffer = t.ins[1].buf, .offset = t.ins[1].off }, .post = .{ .buffer = t.ins[2].buf, .offset = t.ins[2].off }, .output = .{ .buffer = t.out.buf, .offset = t.out.off } }, shape);
            },
            // Draft.encode's fusion and Draft.add, as metal_draft.zig encodes them
            .fuse => {
                e.setPipeline(t.pipe.?);
                e.setBuffer(t.out.buf, t.out.off, 0);
                e.setBuffer(t.ins[1].buf, t.ins[1].off, 1);
                e.setBuffer(t.ins[2].buf, t.ins[2].off, 2);
                e.setValue([2]u32{ n, @intCast(t.shape.k) }, 3);
                e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(128, 1, 1));
            },
            .residual => {
                e.setPipeline(t.pipe.?);
                for ([_]rig_.Ref{ t.ins[0], t.ins[1], t.out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue([2]u32{ n, n * @as(u32, @intCast(t.shape.k)) }, 3);
                e.dispatchThreads(mtl.Size.of(n, rows, 1), mtl.Size.of(128, 1, 1));
            },
        }
        try t.rig.finish(p);
    }

    pub fn count(t: *Elem) usize {
        return switch (t.v.kind) {
            .mix => t.shape.rows * S * t.shape.n,
            .fuse => t.x.len,
            .residual => t.shape.rows * t.shape.n,
        };
    }

    /// Output i's float64 value at the kernel's bf16 points, and its range through each fp32 step.
    fn at(t: *const Elem, i: usize) struct { want: f64, r: m.Range } {
        const n = t.shape.n;
        switch (t.v.kind) {
            .mix => {
                const row = i / (S * n);
                const col = i % (S * n);
                const ps: f64 = t.post[row * S + col / n];
                const b = bv(t.y[row * n + col % n]);
                const x = bv(t.x[i]);
                return .{ .want = bf(x + ps * b), .r = step(step(pt(b).times(ps), 1).plus(pt(x)), 1).bf16() };
            },
            .fuse => {
                const block = n * t.shape.k;
                const x = bv(t.x[i]);
                if (i % block >= n) return .{ .want = x, .r = pt(x) };
                const col = i % block;
                // bf16(delta gain): a product of two bf16 values, exact in fp32
                const upd = bf(bv(t.y[(i / block) * n + col]) * bv(t.z[col]));
                return .{ .want = bf(x + upd), .r = step(pt(x + upd), 1).bf16() };
            },
            .residual => {
                const row = i / n;
                const x = bv(t.x[row * n * t.shape.k + i % n]);
                const y = bv(t.y[i]);
                return .{ .want = bf(x + y), .r = step(pt(x + y), 1).bf16() };
            },
        }
    }

    pub fn reference(t: *Elem, want: []f64, mag: []f64) !void {
        for (want, 0..) |*w, i| w.* = t.at(i).want;
        @memset(mag, 0);
    }

    /// The products of two bf16 values are exact in fp32; each sum rounds once to fp32, then to bf16.
    pub fn intervals(t: *Elem, low: []f64, high: []f64) !void {
        for (low, high, 0..) |*l, *hh, i| {
            const r = t.at(i).r;
            l.* = r.lo;
            hh.* = r.hi;
        }
    }

    pub fn bounds(t: *Elem, want: []const f64, _: []const f64, allow: []f64) !void {
        try h.fl.allowFrom(t.a, t, want, allow);
    }
};

/// A draft kernel compiled as Draft.open compiles it: ihc.metal with MLX's options.
fn draftPipe(device: mtl.Device, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, sources.hy4_flash_ihc, mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}
