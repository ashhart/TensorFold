//! HY4's real wide MLA chunks: raw numerator/maxima/mass and their sink merge on every head/key form.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mla = @import("contract_mla_chunks.zig");
const pages = @import("contract_glm_pages.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lw = 512;
const kw = 576;
const chunk = 512;
const scale: f32 = 0.0625;

/// n is heads * 512 and k is the serving key_stride. Counts form causal rows ending at k.
pub const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2 * lw, .k = 1 },
    .{ .rows = 3, .n = 16 * lw, .k = 512 },
    .{ .rows = 8, .n = 17 * lw, .k = 513 },
    .{ .rows = 8, .n = 32 * lw, .k = 33 },
    .{ .rows = 4, .n = 33 * lw, .k = 1024 },
    .{ .rows = 4, .n = 64 * lw, .k = 2 * chunk },
};
fn depth(s: c.Shape) usize {
    return kw + chunk + (s.k + chunk - 1) / chunk + 1;
}

pub const Entry = struct {
    rig: rig_.Rig,
    source: mla.Chunks,
    f: cuda.Function,
    merge: cuda.Function,
    control: bool = false,
    shape: c.Shape = undefined,
    cache: []u16 = &.{},
    slots: []i32 = &.{},
    counts: []i32 = &.{},
    sinks: []f32 = &.{},
    nch: usize = 0,
    dev: [5]u64 = @splat(0), // q, reversed cache, slots, counts, sinks
    parts: [3]u64 = @splat(0),
    out: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn kernel(self: *Entry) g.Kernel {
        return .{ .spec = .{ .name = if (self.control) "control: HY4 attend chunks with reversed slots against a forward cache" else "cuda hy4 tf_hy4_attend_chunks raw numerator/maxima/mass + tf_hy4_attend_merge", .accum = .none, .depth = depth, .out = .raw, .nan = .{ .stated = "all chunk partial bytes are compared directly; merged finite outputs meet the wide-MLA bound and non-finite classes match float64" }, .served = &shapes }, .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }
    const vt = g.hooks(Entry);

    fn free(self: *Entry) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.cache, &self.slots, &self.counts, &self.sinks }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn deinit(self: *Entry) void {
        self.free();
        self.source.deinit();
        self.rig.deinit();
    }
    fn heads(self: *const Entry) usize {
        return self.shape.n / lw;
    }
    fn partialHeads(self: *const Entry) usize {
        return self.nch * self.shape.rows * self.heads();
    }
    fn semanticCount(self: *const Entry) usize {
        return self.shape.rows * self.heads() * lw;
    }
    pub fn count(self: *Entry) usize {
        return self.partialHeads() * (lw + 2) * 4 + self.semanticCount() * 2;
    }

    pub fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % lw != 0 or shape.n / lw > 64 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        self.nch = (shape.k + chunk - 1) / chunk;
        // Shift the wide-MLA generator so its fullest causal row ends at this key_stride.
        const source_shape: c.Shape = if (shape.k > shape.rows)
            .{ .rows = shape.rows, .n = shape.n, .k = shape.k - shape.rows + @min(shape.k - shape.rows, shape.rows / 2) }
        else
            shape;
        if (!try self.source.prepare(source_shape, if (fill == .nonfinite) .random else fill, seed)) return false;
        if (fill == .nonfinite) {
            // Query-row/head plants avoid shared-key poisoning and retain a finite second head for one-key shapes.
            self.source.q[3] = c.toBf16(std.math.nan(f32));
            if (shape.rows > 1) self.source.q[self.heads() * kw + 4] = c.toBf16(std.math.inf(f32));
            if (shape.rows > 2) self.source.q[2 * self.heads() * kw + 5] = c.toBf16(-std.math.inf(f32));
        }
        self.cache = try a.alloc(u16, shape.k * kw);
        self.slots = try a.alloc(i32, shape.rows * shape.k);
        self.counts = try a.alloc(i32, shape.rows);
        self.sinks = try a.alloc(f32, self.heads());
        @memset(self.sinks, -1.0e30); // zero sink weight: the merge leaves the chunk reference unchanged
        for (0..shape.k) |i| @memcpy(self.cache[(shape.k - 1 - i) * kw ..][0..kw], self.source.cache[i * kw ..][0..kw]);
        for (self.slots, 0..) |*x, i| x.* = @intCast(shape.k - 1 - i % shape.k);
        for (self.counts, 0..) |*x, r| x.* = @intCast(@min(shape.k, self.source.at + r + 1));
        if (self.control) {
            for (self.slots, 0..) |*x, i| x.* = @intCast(i % shape.k);
        }
        self.dev = .{
            try self.rig.input(std.mem.sliceAsBytes(self.source.q)), try self.rig.input(std.mem.sliceAsBytes(self.cache)),
            try self.rig.input(std.mem.sliceAsBytes(self.slots)),    try self.rig.input(std.mem.sliceAsBytes(self.counts)),
            try self.rig.input(std.mem.sliceAsBytes(self.sinks)),
        };
        const n = self.partialHeads();
        self.parts = .{ try self.rig.output(n * lw * 4), try self.rig.output(n * 4), try self.rig.output(n * 4) };
        self.out = try self.rig.output(self.semanticCount() * 2);
        return true;
    }

    pub fn run(self: *Entry) !void {
        const rows64: i64 = @intCast(self.shape.rows);
        const heads64: i64 = @intCast(self.heads());
        var args: cuda.Args = .{};
        inline for (.{ self.dev[0], self.dev[1], self.dev[2], self.dev[3], self.parts[0], self.parts[1], self.parts[2], rows64, heads64, @as(i64, @intCast(self.shape.k)), scale }) |x| args.add(x);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((self.heads() + cuda.mla.heads_tile - 1) / cuda.mla.heads_tile), .z = @intCast(self.nch) }, .block = .{ .x = 256 }, .shared = cuda.mla.chunksSmem(cuda.mla.heads_tile) }, self.rig.s, &args);
        var merge: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.dev[4], self.out, rows64, heads64, @as(c_int, @intCast(self.nch)) }) |x| merge.add(x);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(self.heads()) }, .block = .{ .x = 128 } }, self.rig.s, &merge);
        try self.rig.s.synchronize();
    }

    fn attend(self: *const Entry, r: usize, h: usize, out: []f64, tol: []f64) !void {
        const a = self.rig.gpu.gpa;
        const n: usize = @intCast(self.counts[r]);
        const terms = try a.alloc(f64, 4 * n + kw);
        defer a.free(terms);
        const scores = terms[0..n];
        const ds = terms[n .. 2 * n];
        const p = terms[2 * n .. 3 * n];
        const col = terms[3 * n .. 4 * n];
        const products = terms[4 * n ..];
        var most = -std.math.inf(f64);
        for (0..n) |i| {
            var abs: f64 = 0;
            for (products, 0..) |*x, d| {
                x.* = c.fromBf16(self.source.q[(r * self.heads() + h) * kw + d]) * c.fromBf16(self.source.cache[i * kw + d]);
                abs += @abs(x.*);
            }
            scores[i] = pages.sumIn(.none, .sequential, products) * scale;
            ds[i] = abs * scale * (kw + 8) * 0x1p-24;
            most = pages.maxNan(most, scores[i]);
        }
        for (p, scores) |*x, score| x.* = @exp(score - most);
        const mass = pages.sumIn(.none, .sequential, p);
        for (out, tol, 0..) |*o, *t, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, weight, i| {
                const v = c.fromBf16(self.source.cache[i * kw + d]);
                x.* = weight * v;
                spread += weight * @abs(v) * (0x1p-8 + ds[i]);
            }
            o.* = pages.sumIn(.none, .sequential, col) / mass;
            t[d] = @abs(o.*) * 0x1p-8 + spread / mass + 1e-30;
        }
    }

    pub fn reference(_: *Entry, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Entry, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const n = self.semanticCount();
        const work = try a.alloc(f64, 2 * n);
        defer a.free(work);
        const want = work[0..n];
        const tol = work[n..];
        const raw = try self.rig.outputBytes();
        const merged = raw[self.partialHeads() * (lw + 2) * 4 ..];
        var planted: usize = 0;
        for (0..self.shape.rows) |r| for (0..self.heads()) |h| {
            const at = (r * self.heads() + h) * lw;
            try self.attend(r, h, want[at..][0..lw], tol[at..][0..lw]);
            for (want[at..][0..lw], tol[at..][0..lw], 0..) |w, t, d| {
                const got = c.read(.bf16, merged, at + d);
                if (c.classOf(w) != .finite) planted += 1;
                if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} column {d}: {t} where float64 gives {t}", .{ r, h, d, c.classOf(got), c.classOf(w) });
                if (gate != .nonfinite and std.math.isFinite(w) and !(@abs(got - w) <= t)) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} column {d}: error {e} exceeds {e}", .{ r, h, d, @abs(got - w), t });
            }
        };
        if (gate == .nonfinite and (planted == 0 or planted == n)) return try std.fmt.bufPrint(&self.msg, "planted non-finite values cover {d} of {d} outputs", .{ planted, n });
        return null;
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    items: [2]Entry = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.hy4_mla_attention);
        const f = try self.mod.?.function("tf_hy4_attend_chunks");
        try f.allowDynamicShared(cuda.mla.chunksSmem(cuda.mla.heads_tile));
        const merge = try self.mod.?.function("tf_hy4_attend_merge");
        for (&self.items, [_]bool{ false, true }) |*item, control| {
            item.* = .{
                .rig = .{ .gpu = gpu, .s = s },
                .source = .{ .rig = .{ .gpu = gpu, .s = s }, .f = undefined, .merge = undefined },
                .f = f,
                .merge = merge,
                .control = control,
            };
            self.made += 1;
            try kernels.append(a, item.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        if (self.mod) |m| m.unload();
        self.a.destroy(self);
    }
};
