//! FP4-Local's float32 norm weights under the contract: rmsnorm_4096_f32w and hc_finish_f32w, each output alone.
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
fn normDepth(shape: c.Shape) usize {
    return if (shape.k == 512) 4 + 5 + 2 else if (shape.k == 1536) 16 + 5 + 2 else 16 + 5 + 3;
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
    return specWidth(w, d);
}

fn rowsFor(comptime width: usize) []const c.Shape {
    return &.{ .{ .rows = 1, .n = width, .k = width }, .{ .rows = 70, .n = width, .k = width }, .{ .rows = 2048, .n = width, .k = width } };
}

pub fn specWidth(w: Which, width: usize) c.Spec {
    const rows4096 = switch (width) {
        512 => rowsFor(512),
        1536 => rowsFor(1536),
        else => rowsFor(d),
    };
    const groups = rowsFor(64);
    return switch (w) {
        .norm_y => .{ .name = "cuda glm tf_glm_rmsnorm_4096_f32w (float32 norm weight)", .accum = .f32, .depth = normDepth, .ops = 8, .mids = 1, .out = .bf16, .served = rows4096, .sums = .{ .bf16 = 0, .why = rms_why } },
        .norm_sums => .{ .name = "cuda glm tf_glm_rmsnorm_4096_f32w (its 64-sums)", .accum = .f32, .depth = groupDepth, .out = .f32, .served = groups, .sums = .{ .f16 = 0, .why = sums_why } },
        .hc_y => .{ .name = "cuda glm tf_glm_hc_finish_f32w (normed rows)", .accum = .f32, .depth = hcDepth, .ops = 8, .mids = 3, .out = .bf16, .served = rows4096, .sums = .{ .f16 = 0, .bf16 = 0, .why = hc_why } },
        .hc_sums => .{ .name = "cuda glm tf_glm_hc_finish_f32w (its 64-sums)", .accum = .f32, .depth = groupDepth, .out = .f32, .served = groups, .sums = .{ .f16 = 0, .why = sums_why } },
        .hc_post => .{ .name = "cuda glm tf_glm_hc_finish_f32w (post)", .accum = .f32, .depth = mixDepth, .ops = 16, .out = .f32, .served = &.{ .{ .rows = 1, .n = 4, .k = ref.nb }, .{ .rows = 70, .n = 4, .k = ref.nb }, .{ .rows = 2048, .n = 4, .k = ref.nb } }, .sums = .{ .f16 = 0, .why = mix_why } },
        .hc_comb => .{ .name = "cuda glm tf_glm_hc_finish_f32w (comb, Sinkhorn)", .accum = .f32, .depth = mixDepth, .ops = 256, .out = .f32, .served = &.{ .{ .rows = 1, .n = 16, .k = ref.nb }, .{ .rows = 70, .n = 16, .k = ref.nb }, .{ .rows = 2048, .n = 16, .k = ref.nb } }, .sums = .{ .f16 = 0, .why = mix_why } },
    };
}

/// `gamma` rounded to bf16 and back: what a kernel reading a bf16 weight multiplies by; the reference keeps float32.
fn narrowed(a: std.mem.Allocator, gamma: []const f32) ![]f32 {
    const out = try a.alloc(f32, gamma.len);
    for (out, gamma) |*o, v| o.* = @floatCast(ref.bf(v));
    return out;
}

pub const F32w = struct {
    rig: rig_.Rig,
    which: Which,
    f: cuda.Function,
    in: ?ref.In = null,
    arena: std.heap.ArenaAllocator,
    ptrs: [9]u64 = @splat(0), // norm: x, gamma, y, xs; hc: x, part, base, scale, gamma, y, xs, post, comb
    width: usize = d,
    baseline: ?cuda.Function = null,
    baseline_ptrs: [9]u64 = @splat(0),
    baseline_output: u64 = 0,
    bf16_gamma: bool = false, // gate 3's control: the weight uploaded as bf16 values, as a bf16-weight kernel reads it

    pub fn init(rig: rig_.Rig, which: Which, f: cuda.Function) F32w {
        return .{ .rig = rig, .which = which, .f = f, .arena = .init(rig.gpu.gpa) };
    }

    pub fn deinit(self: *F32w) void {
        cuda.completion.drain(self.rig.gpu.d, self.rig.s, "GLM norm contract release before completion");
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *F32w) g.Kernel {
        var declared = specWidth(self.which, self.width);
        if (self.baseline != null) {
            declared.name = bf16Name(self.which, self.width);
            declared.replaces = "main 958efa3bf GLM BF16 norm, frozen in the contract controls";
        }
        if (self.bf16_gamma) declared.name = "control: a float32 norm weight read as bf16";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.bf16_gamma) .shapes else null };
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
        const served = specWidth(self.which, self.width).served[0];
        if (shape.n != served.n or shape.k != served.k) return false;
        cuda.completion.drain(self.rig.gpu.d, self.rig.s, "GLM norm contract reset before completion");
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const exact = self.which == .norm_sums or self.which == .hc_sums;
        var in = try ref.fillWidth(self.arena.allocator(), self.hc(), self.width, shape.rows, fill, seed, exact);
        if (self.bf16_gamma) ref.narrowControl(&in);
        if (self.baseline != null) {
            for (in.gamma) |*v| v.* = @floatCast(ref.bf(v.*));
        }
        self.in = in;
        const r = &self.rig;
        const n = shape.rows;
        const y = if (self.which == .norm_y or self.which == .hc_y) try r.output(n * self.width * 2) else try r.scratch(n * self.width * 2);
        const xs = if (exact) try r.output(n * (self.width / 64) * 4) else if (self.hc()) try r.scratch(n * (self.width / 64) * 4) else 0;
        const x = try r.input(std.mem.sliceAsBytes(in.x));
        const gamma = if (self.bf16_gamma) try narrowed(self.arena.allocator(), in.gamma) else in.gamma;
        const gm = if (self.baseline != null) blk: {
            const bits = try self.arena.allocator().alloc(u16, gamma.len);
            for (bits, gamma) |*v, xw| v.* = c.toBf16(xw);
            break :blk try r.input(std.mem.sliceAsBytes(bits));
        } else try r.input(std.mem.sliceAsBytes(gamma));
        if (!self.hc()) {
            self.ptrs = .{ x, gm, y, xs, 0, 0, 0, 0, 0 };
            try self.baselineOutputs();
            return true;
        }
        const part = try r.input(std.mem.sliceAsBytes(in.part));
        const base = try r.input(std.mem.asBytes(&in.base));
        const sc = try r.input(std.mem.asBytes(&ref.scale));
        const post = if (self.which == .hc_post) try r.output(n * 4 * 4) else try r.scratch(n * 4 * 4);
        const comb = if (self.which == .hc_comb) try r.output(n * 16 * 4) else try r.scratch(n * 16 * 4);
        self.ptrs = .{ x, part, base, sc, gm, y, xs, post, comb };
        try self.baselineOutputs();
        return true;
    }

    fn baselineOutputs(self: *F32w) !void {
        if (self.baseline == null) return;
        self.baseline_ptrs = self.ptrs;
        const n = self.in.?.rows;
        const at: usize = if (self.hc()) 5 else 2;
        self.baseline_ptrs[at] = try self.rig.scratch(n * self.width * 2);
        if (self.ptrs[at + 1] != 0) self.baseline_ptrs[at + 1] = try self.rig.scratch(n * (self.width / 64) * 4);
        if (self.hc()) {
            self.baseline_ptrs[7] = try self.rig.scratch(n * 4 * 4);
            self.baseline_ptrs[8] = try self.rig.scratch(n * 16 * 4);
        }
        self.baseline_output = self.baseline_ptrs[
            switch (self.which) {
                .norm_y => 2,
                .norm_sums => 3,
                .hc_y => 5,
                .hc_sums => 6,
                .hc_post => 7,
                .hc_comb => 8,
            }
        ];
    }

    fn launch(self: *F32w, f: cuda.Function, p: [9]u64) !void {
        const in = self.in.?;
        var a: cuda.Args = .{};
        if (self.hc()) {
            inline for (p) |v| a.add(v);
            inline for (.{ in.eps, in.hc_eps, @as(c_int, ref.nb), @as(c_int, ref.iters) }) |v| a.add(v);
        } else {
            inline for (.{ p[0], @as(i64, @intCast(self.width)), p[1], p[2], @as(i64, @intCast(self.width)), p[3], in.eps, @as(c_int, @intFromBool(p[3] != 0)) }) |v| a.add(v);
        }
        const threads: u32 = if (!self.hc() and self.width != d) 128 else 256;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(in.rows) }, .block = .{ .x = threads } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *F32w) !void {
        return self.launch(self.f, self.ptrs);
    }

    /// The old kernel reads the identical input allocations and writes separate guarded scratch allocations.
    pub fn replaced(self: *F32w) ![]const u8 {
        try self.launch(self.baseline orelse return error.NoBaseline, self.baseline_ptrs);
        return self.rig.bytes(self.baseline_output, self.count() * c.size(specWidth(self.which, self.width).out));
    }

    pub fn count(self: *F32w) usize {
        const n = self.in.?.rows;
        return n * @as(usize, switch (self.which) {
            .norm_y, .hc_y => self.width,
            .norm_sums, .hc_sums => self.width / 64,
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

/// BF16 kernels changed only by the template refactor: each output is compared with the frozen old entrypoint.
pub const Bf16Entry = struct { which: Which, width: usize, symbol: [:0]const u8, old: [:0]const u8 };
pub const bf16_entries = blk: {
    var entries: [10]Bf16Entry = undefined;
    var at: usize = 0;
    for (.{ 512, 1536, 4096 }) |width| for (.{ Which.norm_y, Which.norm_sums }) |which| {
        entries[at] = .{ .which = which, .width = width, .symbol = "tf_glm_rmsnorm_" ++ std.fmt.comptimePrint("{d}", .{width}), .old = "tf_ctl_glm_old_rmsnorm_" ++ std.fmt.comptimePrint("{d}", .{width}) };
        at += 1;
    };
    for (.{ Which.hc_y, Which.hc_sums, Which.hc_post, Which.hc_comb }) |which| {
        entries[at] = .{ .which = which, .width = d, .symbol = "tf_glm_hc_finish", .old = "tf_ctl_glm_old_hc_finish" };
        at += 1;
    }
    break :blk entries;
};

fn bf16Name(w: Which, width: usize) []const u8 {
    return switch (w) {
        .norm_y => switch (width) {
            512 => "cuda glm rmsnorm_512 BF16 replacement",
            1536 => "cuda glm rmsnorm_1536 BF16 replacement",
            else => "cuda glm rmsnorm_4096 BF16 replacement",
        },
        .norm_sums => switch (width) {
            512 => "cuda glm rmsnorm_512 BF16 replacement 64-sums",
            1536 => "cuda glm rmsnorm_1536 BF16 replacement 64-sums",
            else => "cuda glm rmsnorm_4096 BF16 replacement 64-sums",
        },
        .hc_y => "cuda glm hc_finish BF16 replacement normed rows",
        .hc_sums => "cuda glm hc_finish BF16 replacement 64-sums",
        .hc_post => "cuda glm hc_finish BF16 replacement post",
        .hc_comb => "cuda glm hc_finish BF16 replacement comb",
    };
}
