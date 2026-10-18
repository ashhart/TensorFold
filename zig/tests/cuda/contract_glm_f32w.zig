//! FP4-Local's float32 norm weights (D1) under the contract: rmsnorm_4096_f32w and hc_finish_f32w, each output alone.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_glm_f32w_ref.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const d = ref.d;

/// norm_*: rmsnorm_4096_f32w's rows and their 64-sums; hc_*: hc_finish_f32w's normed rows, 64-sums, post and comb.
pub const Which = enum { norm_y, norm_sums, hc_y, hc_sums, hc_post, hc_comb };

/// A thread's 16 squares in one fma chain, the warp's butterfly, then 8 warps' partials by butterfly.
fn normDepth(_: c.Shape) usize {
    return 16 + 5 + 3;
}

/// hc_finish's 16 squares in fma and mul-add pairs (23 roundings), then the same butterflies.
fn hcDepth(_: c.Shape) usize {
    return 23 + 5 + 3;
}

/// A thread's 8 outputs in one chain, then 8 threads by xor 4, 2 and 1.
fn groupDepth(_: c.Shape) usize {
    return 7 + 3;
}

/// The 16 K blocks' partials of a logit and of the squares, each one chain.
fn mixDepth(_: c.Shape) usize {
    return ref.nb;
}

const rms_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const hc_why = "three bf16 roundings in the bound: fp16 squares overflow to a zero output, 64 bounds; bf16 sums move it ulps";
const sums_why = "64 terms of at most 8 bits: fp16 holds every partial, so bf16's miss is the one shown";
const mix_why = "16 integer partials: fp16 holds every one, so bf16's miss is the one shown";

pub fn spec(w: Which) c.Spec {
    const rows4096: []const c.Shape = &.{ .{ .rows = 1, .n = d, .k = d }, .{ .rows = 70, .n = d, .k = d } };
    const groups: []const c.Shape = &.{ .{ .rows = 1, .n = 64, .k = 64 }, .{ .rows = 70, .n = 64, .k = 64 } };
    return switch (w) {
        .norm_y => .{ .name = "cuda glm tf_glm_rmsnorm_4096_f32w (float32 norm weight)", .accum = .f32, .depth = normDepth, .ops = 8, .mids = 1, .out = .bf16, .served = rows4096, .sums = .{ .bf16 = 0, .why = rms_why } },
        .norm_sums => .{ .name = "cuda glm tf_glm_rmsnorm_4096_f32w (its 64-sums)", .accum = .f32, .depth = groupDepth, .out = .f32, .served = groups, .sums = .{ .f16 = 0, .why = sums_why } },
        .hc_y => .{ .name = "cuda glm tf_glm_hc_finish_f32w (normed rows)", .accum = .f32, .depth = hcDepth, .ops = 8, .mids = 3, .out = .bf16, .served = rows4096, .sums = .{ .f16 = 0, .bf16 = 0, .why = hc_why } },
        .hc_sums => .{ .name = "cuda glm tf_glm_hc_finish_f32w (its 64-sums)", .accum = .f32, .depth = groupDepth, .out = .f32, .served = groups, .sums = .{ .f16 = 0, .why = sums_why } },
        .hc_post => .{ .name = "cuda glm tf_glm_hc_finish_f32w (post)", .accum = .f32, .depth = mixDepth, .ops = 16, .out = .f32, .served = &.{ .{ .rows = 1, .n = 4, .k = ref.nb }, .{ .rows = 70, .n = 4, .k = ref.nb } }, .sums = .{ .f16 = 0, .why = mix_why } },
        .hc_comb => .{ .name = "cuda glm tf_glm_hc_finish_f32w (comb, Sinkhorn)", .accum = .f32, .depth = mixDepth, .ops = 256, .out = .f32, .served = &.{ .{ .rows = 1, .n = 16, .k = ref.nb }, .{ .rows = 70, .n = 16, .k = ref.nb } }, .sums = .{ .f16 = 0, .why = mix_why } },
    };
}

pub const F32w = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    in: ?ref.In = null,
    arena: std.heap.ArenaAllocator,
    ptrs: [9]u64 = @splat(0), // norm: x, gamma, y, xs; hc: x, part, base, scale, gamma, y, xs, post, comb

    pub fn init(rig: rig_.Rig, which: Which, f: cuda.Function) F32w {
        return .{ .rig = rig, .which = which, .f = f, .arena = .init(rig.gpu.gpa) };
    }

    pub fn deinit(self: *F32w) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *F32w) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(F32w);

    fn hc(self: *const F32w) bool {
        return switch (self.which) {
            .norm_y, .norm_sums => false,
            else => true,
        };
    }

    /// The inputs (`ref.fill`), the checked output, and the kernel's other outputs as scratch.
    pub fn prepare(self: *F32w, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const served = spec(self.which).served[0];
        if (shape.n != served.n or shape.k != served.k) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const exact = self.which == .norm_sums or self.which == .hc_sums;
        const in = try ref.fill(self.arena.allocator(), self.hc(), shape.rows, fill, seed, exact);
        self.in = in;
        const r = &self.rig;
        const n = shape.rows;
        const y = if (self.which == .norm_y or self.which == .hc_y) try r.output(n * d * 2) else try r.scratch(n * d * 2);
        const xs = if (exact) try r.output(n * (d / 64) * 4) else if (self.hc()) try r.scratch(n * (d / 64) * 4) else 0;
        const x = try r.input(std.mem.sliceAsBytes(in.x));
        const gm = try r.input(std.mem.sliceAsBytes(in.gamma));
        if (!self.hc()) {
            self.ptrs = .{ x, gm, y, xs, 0, 0, 0, 0, 0 };
            return true;
        }
        const part = try r.input(std.mem.sliceAsBytes(in.part));
        const base = try r.input(std.mem.asBytes(&in.base));
        const sc = try r.input(std.mem.asBytes(&ref.scale));
        const post = if (self.which == .hc_post) try r.output(n * 4 * 4) else try r.scratch(n * 4 * 4);
        const comb = if (self.which == .hc_comb) try r.output(n * 16 * 4) else try r.scratch(n * 16 * 4);
        self.ptrs = .{ x, part, base, sc, gm, y, xs, post, comb };
        return true;
    }

    /// One launch, 256 threads a row: the norm's (x, stride, w, out, stride, xs, eps, sums) or hc_finish's arguments.
    pub fn run(self: *F32w) !void {
        const p = self.ptrs;
        const in = self.in.?;
        var a: cuda.Args = .{};
        if (self.hc()) {
            inline for (p) |v| a.add(v);
            inline for (.{ in.eps, in.hc_eps, @as(c_int, ref.nb), @as(c_int, ref.iters) }) |v| a.add(v);
        } else {
            inline for (.{ p[0], @as(i64, d), p[1], p[2], @as(i64, d), p[3], in.eps, @as(c_int, @intFromBool(p[3] != 0)) }) |v| a.add(v);
        }
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(in.rows) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *F32w) usize {
        const n = self.in.?.rows;
        return n * @as(usize, switch (self.which) {
            .norm_y, .hc_y => d,
            .norm_sums, .hc_sums => d / 64,
            .hc_post => 4,
            .hc_comb => 16,
        });
    }

    fn outputs(self: *F32w, m: ref.Sum, want: []f64, mag: []f64) !void {
        const in = self.in.?;
        const a = self.arena.allocator();
        switch (self.which) {
            .norm_y, .hc_y => try ref.normed(a, in, m, want, mag),
            .norm_sums, .hc_sums => try ref.sums(a, in, m, want, mag),
            .hc_post => ref.post(in, m, want, mag),
            .hc_comb => ref.comb(in, m, want, mag),
        }
    }

    pub fn reference(self: *F32w, want: []f64, mag: []f64) !void {
        return self.outputs(.{}, want, mag);
    }

    pub fn simulate(self: *F32w, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        return self.outputs(.{ .acc = acc, .order = order }, out, mag);
    }
};
