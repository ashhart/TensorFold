//! TF_GLM_MOE_DIRECT's kernels on CUDA (#3570): tf_glm_topk's flat picks by rule, tf_glm_combine_tail bit-equal.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const glue = glm.glue;
const top_k = glm.exl3.top_k;
const slots = top_k + 1;
const experts = 288; // GLM-5.3-Flash's routed experts; the shared expert's id is 288
const dims = 4096;
const scale: f32 = 2.5; // GLM-5.3-Flash's routed_scaling_factor

fn noSums(_: c.Shape) usize {
    return 0;
}

/// A seeded 64-bit stream (splitmix64), cheap enough for a chunk's slot rows.
const Mix = struct {
    x: u64,

    fn next(m: *Mix) u64 {
        m.x +%= 0x9e3779b97f4a7c15;
        var z = m.x;
        z = (z ^ (z >> 30)) *% 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) *% 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

    /// One of `steps` evenly spaced values from lo up to, not including, hi.
    fn grid(m: *Mix, lo: f32, hi: f32, steps: u32) f32 {
        const i: f32 = @floatFromInt(m.next() % steps);
        return lo + i * (hi - lo) / @as(f32, @floatFromInt(steps));
    }
};

const pick_rule = "flat equal to pick's routed columns, and pick and wts bit-equal to the launch without flat, " ++
    "non-finite logits included; on finite logits the top_k highest sigmoid + bias in order, ties to the lower " ++
    "expert id, each weight its sigmoid normed over the picks times the scale, then the shared expert at weight 1";

/// One warp a row, from one decode row to one EXL3 chunk.
const pick_served = [_]c.Shape{
    .{ .rows = 1, .n = experts, .k = top_k },
    .{ .rows = 4, .n = experts, .k = top_k },
    .{ .rows = 32, .n = experts, .k = top_k },
    .{ .rows = 129, .n = experts, .k = top_k },
    .{ .rows = glm.exl3.chunk_rows, .n = experts, .k = top_k },
};

pub fn pickSpec(control: bool) c.Spec {
    const name = if (control) "cuda glm tf_glm_topk control: no flat pointer, the flat picks left unwritten" else "cuda glm tf_glm_topk with flat picks (TF_GLM_MOE_DIRECT)";
    return .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = pick_rule }, .served = &pick_served };
}

/// tf_glm_topk as glue.select launches it, with the flat picks; its pick and wts against the launch without them.
pub const Pick = struct {
    rig: rig_.Rig,
    set: glue.Set,
    control: bool = false,
    shape: c.Shape = undefined,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    finite: bool = true,
    in: [2]u64 = .{ 0, 0 },
    out: [3]u64 = .{ 0, 0, 0 }, // pick, wts, flat
    base: u64 = 0, // pick then wts of the launch without flat
    got: std.ArrayList(u8) = .empty,
    msg: [200]u8 = undefined,

    const vt = g.hooks(Pick);

    pub fn deinit(self: *Pick) void {
        const a = self.rig.gpu.gpa;
        a.free(self.logits);
        a.free(self.bias);
        self.got.deinit(a);
        self.rig.deinit();
    }

    pub fn kernel(self: *Pick) g.Kernel {
        return .{ .spec = pickSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    /// Logits on a 1/16 grid, biases on a 1/64 grid shared by experts 2j and 2j + 1; the top-bias pair ties first.
    pub fn prepare(self: *Pick, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != experts or shape.k != top_k) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.logits);
        a.free(self.bias);
        self.logits = &.{};
        self.bias = &.{};
        self.shape = shape;
        self.logits = try a.alloc(f32, shape.rows * experts);
        self.bias = try a.alloc(f32, experts);
        var m: Mix = .{ .x = seed };
        for (0..experts / 2) |j| {
            const b = m.grid(-0.5, 0.5, 64);
            self.bias[2 * j] = b;
            self.bias[2 * j + 1] = b;
        }
        var top: usize = 0;
        for (0..experts / 2) |j| if (self.bias[2 * j] > self.bias[2 * top]) {
            top = j;
        };
        for (self.logits) |*x| x.* = m.grid(-8, 8, 256);
        for (0..shape.rows) |r| {
            self.logits[r * experts + 2 * top] = 9;
            self.logits[r * experts + 2 * top + 1] = 9;
        }
        self.finite = fill != .nonfinite;
        if (!self.finite) { // rows 0-3: NaN, +inf, -inf and both infinities at a few experts each
            const put = [_][2]f32{ .{ std.math.nan(f32), std.math.nan(f32) }, .{ std.math.inf(f32), std.math.inf(f32) }, .{ -std.math.inf(f32), -std.math.inf(f32) }, .{ std.math.inf(f32), -std.math.inf(f32) } };
            for (put, 0..) |v, r| if (r < shape.rows) {
                for (0..4) |e| self.logits[r * experts + e * 71] = v[e % 2];
            };
        }
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.logits));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.bias));
        self.out[0] = try self.rig.output(shape.rows * slots * 4);
        self.out[1] = try self.rig.output(shape.rows * slots * 4);
        self.out[2] = try self.rig.output(shape.rows * top_k * 4);
        self.base = try self.rig.scratch(2 * shape.rows * slots * 4);
        return true;
    }

    /// The control launches as before #3570, without the flat pointer.
    pub fn run(self: *Pick) !void {
        try glue.select(self.set, self.rig.s, self.in[0], self.in[1], self.out[0], self.out[1], self.shape.rows, top_k, experts, scale, true, if (self.control) null else self.out[2]);
        try self.rig.s.synchronize();
    }

    fn fail(self: *Pick, comptime fmt: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.msg, fmt, args) catch "(message too long)";
    }

    /// The stated rule (see pick_rule); the ranking only on finite logits, against float64 within 1e-5 of a choice.
    pub fn check(self: *Pick, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const rows = self.shape.rows;
        const pb = rows * slots * 4;
        self.got.clearRetainingCapacity();
        try self.got.appendSlice(a, try self.rig.outputBytes());
        const got = self.got.items;
        try glue.select(self.set, self.rig.s, self.in[0], self.in[1], self.base, self.base + pb, rows, top_k, experts, scale, true, null);
        const base = try self.rig.bytes(self.base, 2 * pb);
        if (std.mem.indexOfDiff(u8, got[0 .. 2 * pb], base)) |at| return self.fail("pick and wts byte {d} differs from the launch without flat", .{at});
        const pick = std.mem.bytesAsSlice(i32, got[0..pb]);
        const wts = std.mem.bytesAsSlice(f32, got[pb .. 2 * pb]);
        const flat = std.mem.bytesAsSlice(i32, got[2 * pb ..][0 .. rows * top_k * 4]);
        for (0..rows) |r| {
            for (0..top_k) |k| if (flat[r * top_k + k] != pick[r * slots + k])
                return self.fail("row {d} flat pick {d} is {d}, pick's is {d}", .{ r, k, flat[r * top_k + k], pick[r * slots + k] });
            if (pick[r * slots + top_k] != experts or wts[r * slots + top_k] != 1)
                return self.fail("row {d}'s shared slot holds {d} at weight {e}", .{ r, pick[r * slots + top_k], wts[r * slots + top_k] });
        }
        if (gate == .nonfinite or !self.finite) return null;
        for (0..rows) |r| if (self.rule(r, pick[r * slots ..][0..top_k], wts[r * slots ..][0..top_k])) |why| return why;
        return null;
    }

    fn sigmoid(self: *const Pick, r: usize, e: usize) f64 {
        return 1 / (1 + @exp(-@as(f64, self.logits[r * experts + e])));
    }

    fn choice(self: *const Pick, r: usize, e: usize) f64 {
        return self.sigmoid(r, e) + @as(f64, self.bias[e]);
    }

    /// Row `r`'s picks in order: none clearly below an unpicked expert, an exact tie to the lower id, and the weights.
    fn rule(self: *Pick, r: usize, pick: anytype, wts: anytype) ?[]const u8 {
        var taken: [experts]bool = @splat(false);
        var total: f64 = 0;
        for (0..top_k) |k| {
            const p = pick[k];
            if (p < 0 or p >= experts or taken[@intCast(p)]) return self.fail("row {d} pick {d} is {d}", .{ r, k, p });
            const e: usize = @intCast(p);
            taken[e] = true;
            const mine = self.choice(r, e);
            for (0..experts) |o| if (!taken[o]) {
                const theirs = self.choice(r, o);
                if (theirs > mine + 1e-5) return self.fail("row {d} pick {d} is {d} ({d:.6}), expert {d} scores {d:.6}", .{ r, k, e, mine, o, theirs });
                const tie = self.logits[r * experts + o] == self.logits[r * experts + e] and self.bias[o] == self.bias[e];
                if (tie and o < e) return self.fail("row {d} pick {d} is {d}, tied with lower expert {d}", .{ r, k, e, o });
            };
            total += self.sigmoid(r, e);
        }
        for (0..top_k) |k| {
            const want = self.sigmoid(r, @intCast(pick[k])) / total * @as(f64, scale);
            const w: f64 = wts[k];
            if (!(@abs(w - want) <= 1e-5 * want + 1e-7)) return self.fail("row {d} weight {d} is {e}, float64 gives {e}", .{ r, k, w, want });
        }
        return null;
    }

    pub fn count(self: *Pick) usize {
        return self.shape.rows * (2 * slots + top_k);
    }

    pub fn reference(_: *Pick, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};

const sum_rule = "every byte equal to tf_glm_put_slot, the routed slot rows' copy and tf_glm_combine's on the same " ++
    "rows, weights and shared row, non-finite inputs included";

/// Every decode row count up to one EXL3 chunk (cuda_moe.direct's limit), at GLM's dims and 8 + 1 slots.
const sum_served = blk: {
    @setEvalBranchQuota(8 * glm.exl3.chunk_rows);
    var s: [glm.exl3.chunk_rows]c.Shape = undefined;
    for (&s, 1..) |*x, r| x.* = .{ .rows = r, .n = dims, .k = slots };
    break :blk s;
};

pub fn sumSpec(control: bool) c.Spec {
    const name = if (control) "cuda glm tf_glm_combine_tail control: slot 0's routed rows in the shared row's place" else "cuda glm tf_glm_combine_tail (TF_GLM_MOE_DIRECT's slot sum)";
    return .{ .name = name, .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = sum_rule }, .replaces = "tf_glm_put_slot, the EXL3 slot-row copy and tf_glm_combine", .served = &sum_served };
}

/// tf_glm_combine_tail as cuda_moe.fill launches it on the EXL3 experts' slot rows and the shared expert's row.
pub const Combine = struct {
    rig: rig_.Rig,
    set: glue.Set,
    put_slot: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    y: u64 = 0, // fp32 [rows][top_k][d], as the experts leave them
    tail: u64 = 0, // fp32 [rows][d], the shared expert's row
    wts: u64 = 0, // fp32 [rows][slots]
    out: u64 = 0,
    ey: u64 = 0, // the slot rows the replaced path builds: fp32 [rows][slots][d]
    base: u64 = 0,
    pattern: [4096]f32 = undefined,
    got: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    const vt = g.hooks(Combine);

    pub fn deinit(self: *Combine) void {
        self.got.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Combine) g.Kernel {
        return .{ .spec = sumSpec(self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    /// A seeded 4,096-value pattern, each row turned its own way; weights as top-k leaves them; NaN, infs in rows 0-2.
    pub fn prepare(self: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k != slots) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const rows = shape.rows;
        const d = shape.n;
        var m: Mix = .{ .x = seed };
        for (&self.pattern) |*v| v.* = m.grid(-4, 4, 8192);
        const ys = try a.alloc(f32, rows * top_k * d);
        defer a.free(ys);
        const tail = try a.alloc(f32, rows * d);
        defer a.free(tail);
        const w = try a.alloc(f32, rows * slots);
        defer a.free(w);
        for (0..rows * top_k) |i| self.row(ys[i * d ..][0..d], m.next());
        for (0..rows) |r| self.row(tail[r * d ..][0..d], m.next());
        for (0..rows) |r| {
            for (0..top_k) |k| w[r * slots + k] = m.grid(0, scale, 4096);
            w[r * slots + top_k] = 1;
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, r| if (r < rows) {
                ys[(r * top_k + r) * d + 8 * r + 1] = v;
                tail[r * d + 3] = v;
            };
        }
        self.y = try self.rig.input(std.mem.sliceAsBytes(ys));
        self.tail = try self.rig.input(std.mem.sliceAsBytes(tail));
        self.wts = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(rows * d * 4);
        self.ey = try self.rig.scratch(rows * slots * d * 4);
        self.base = try self.rig.scratch(rows * d * 4);
        return true;
    }

    fn row(self: *const Combine, dst: []f32, turn: u64) void {
        var at: usize = 0;
        var from: usize = @intCast(turn % self.pattern.len);
        while (at < dst.len) {
            const n = @min(dst.len - at, self.pattern.len - from);
            @memcpy(dst[at..][0..n], self.pattern[from..][0..n]);
            at += n;
            from = 0;
        }
    }

    pub fn run(self: *Combine) !void {
        const tail = if (self.control) self.y else self.tail;
        try glue.combineTail(self.set, self.rig.s, self.y, tail, self.wts, self.out, self.shape.rows, self.shape.n, slots);
        try self.rig.s.synchronize();
    }

    /// What a decode window ran before #3570: compact's slot-row copy, put_slot of the shared row, tf_glm_combine.
    pub fn replaced(self: *Combine) ![]const u8 {
        const rows = self.shape.rows;
        const d = self.shape.n;
        const s = self.rig.s;
        const m: cuda.abi.Memcpy2D = .{ .src = self.y, .src_pitch = top_k * d * 4, .dst = self.ey, .dst_pitch = slots * d * 4, .width = top_k * d * 4, .height = rows };
        try s.d.check(s.d.api.cuMemcpy2DAsync_v2(&m, s.handle), "the routed slot rows");
        var a: cuda.Args = .{};
        a.add(self.tail);
        a.add(self.ey);
        for ([_]usize{ rows, d, slots, top_k, 0 }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.put_slot, .{ .grid = .{ .x = @intCast((rows * d + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
        try glue.combine(self.set, s, self.ey, true, self.wts, self.base, rows, d, slots);
        return self.rig.bytes(self.base, rows * d * 4);
    }

    /// Gate 1's stated rule: the bytes on the planted inputs equal the replaced path's.
    pub fn check(self: *Combine, _: g.Gate) !?[]const u8 {
        self.got.clearRetainingCapacity();
        try self.got.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.replaced();
        const at = std.mem.indexOfDiff(u8, self.got.items, want) orelse return null;
        return std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from the replaced path's", .{ at, want.len }) catch "differs";
    }

    pub fn count(self: *Combine) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(_: *Combine, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};

/// The two kernels and their controls on the heap, from their own glm_glue and glm_ops images.
pub const Set = struct {
    gpa: std.mem.Allocator,
    glue_mod: cuda.Module,
    ops_mod: cuda.Module,
    fns: glue.Kernels,
    picks: [2]Pick = undefined,
    sums: [2]Combine = undefined,

    pub fn init(gpu: check_.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var glue_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
        errdefer glue_mod.unload();
        var ops_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
        errdefer ops_mod.unload();
        set.* = .{ .gpa = gpu.gpa, .glue_mod = glue_mod, .ops_mod = ops_mod, .fns = try glue.Kernels.load(glue_mod) };
        const put_slot = try ops_mod.function("tf_glm_put_slot");
        for (&set.picks, [_]bool{ false, true }) |*x, control| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .set = .{ .glue = &set.fns }, .control = control };
        for (&set.sums, [_]bool{ false, true }) |*x, control| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .set = .{ .glue = &set.fns }, .put_slot = put_slot, .control = control };
        for (&set.picks) |*x| try list.append(a, x.kernel());
        for (&set.sums) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.sums) |*x| x.deinit();
        for (&set.picks) |*x| x.deinit();
        set.ops_mod.unload();
        set.glue_mod.unload();
        set.gpa.destroy(set);
    }
};
