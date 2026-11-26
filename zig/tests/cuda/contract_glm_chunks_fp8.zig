//! GLM's real E4M3 chunk kernels: raw partials and their served merge under the identity A/B contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lw = 512;
const heads = 32;
const chunk = 512;
const scale: f32 = 0.0625;
const e4m3 = core.quant.fp8.e4m3;

pub const Form = struct { hb: u32, sparse: bool, control: bool = false };
pub const forms = [_]Form{
    .{ .hb = 16, .sparse = false },                  .{ .hb = 32, .sparse = false },
    .{ .hb = 16, .sparse = true },                   .{ .hb = 32, .sparse = true },
    .{ .hb = 16, .sparse = false, .control = true },
};

fn served(f: Form) []const c.Shape {
    return if (f.hb == 16) (if (f.sparse) &.{.{ .rows = 8, .n = heads * lw, .k = glm.sparse.width }} else &.{.{ .rows = 8, .n = heads * lw, .k = 2 * chunk }}) else (if (f.sparse) &.{.{ .rows = 64, .n = heads * lw, .k = 300 }} else &.{.{ .rows = 64, .n = heads * lw, .k = chunk }});
}

fn depth(s: c.Shape) usize {
    return lw + chunk + s.k / chunk + 1;
}

fn noDepth(_: c.Shape) usize {
    return 0;
}

pub const Entry = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    merge: cuda.Function,
    form: Form,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    codes: []u8 = &.{},
    scales: []f32 = &.{},
    tokens: []i32 = &.{},
    counts: []i32 = &.{},
    at: usize = 0,
    nch: usize = 0,
    dev: [6]u64 = @splat(0), // q, codes, scales, tokens, counts, position
    parts: [3]u64 = @splat(0),
    out: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn kernel(self: *Entry) g.Kernel {
        const name = if (self.form.control)
            "control: GLM FP8 chunks-16 scales multiplied by 1.5"
        else if (self.form.sparse)
            std.fmt.comptimePrint("cuda glm tf_glm_chunks_{d}_fp8 sparse raw numerator/maxima/mass + tf_glm_merge", .{self.form.hb})
        else
            std.fmt.comptimePrint("cuda glm tf_glm_chunks_{d}_fp8 dense raw numerator/maxima/mass + tf_glm_merge", .{self.form.hb});
        return .{ .spec = .{ .name = name, .accum = .none, .depth = if (self.form.control) noDepth else depth, .out = .raw, .nan = .{ .stated = "raw FP8 partial bits are compared directly; merged BF16 classes match float64 and finite values meet the MLA bound, count-zero rows retain poison" }, .served = served(self.form) }, .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Entry);

    fn free(self: *Entry) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ &self.q, &self.codes, &self.scales, &self.tokens, &self.counts }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
    }
    pub fn deinit(self: *Entry) void {
        self.free();
        self.rig.deinit();
    }
    fn span(self: *const Entry) usize {
        return if (self.form.sparse) glm.sparse.width else self.at + self.shape.rows;
    }
    fn seen(self: *const Entry, r: usize) usize {
        return if (self.form.sparse) @intCast(self.counts[r]) else self.at + r + 1;
    }
    fn key(self: *const Entry, r: usize, i: usize) usize {
        return if (self.form.sparse) @intCast(self.tokens[r * self.shape.k + i]) else i;
    }
    fn partialHeads(self: *const Entry) usize {
        return self.nch * self.shape.rows * heads;
    }
    fn semanticCount(self: *const Entry) usize {
        return self.shape.rows * heads * lw;
    }
    pub fn count(self: *Entry) usize {
        return self.partialHeads() * (lw + 2) * 4 + self.semanticCount() * 2;
    }

    fn putCode(v: f32) u8 {
        var code: u8 = 0;
        while (code < 0x7f) : (code += 1) if (e4m3(code) == @abs(v)) break;
        return code | @as(u8, if (std.math.signbit(v)) 0x80 else 0);
    }

    pub fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n != heads * lw or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = if (self.form.sparse) 0 else shape.k - @min(shape.k, shape.rows / 2);
        self.nch = ((if (self.form.sparse) shape.k else self.at + shape.rows) + chunk - 1) / chunk;
        self.q = try a.alloc(u16, shape.rows * heads * lw);
        self.codes = try a.alloc(u8, self.span() * lw);
        self.scales = try a.alloc(f32, self.span());
        self.tokens = try a.alloc(i32, shape.rows * shape.k);
        self.counts = try a.alloc(i32, shape.rows);
        for (self.q) |*x| x.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (self.codes) |*x| {
            x.* = rng.int(u8);
            if (x.* & 0x7f == 0x7f) x.* ^= 1;
        }
        for (self.scales) |*x| x.* = std.math.ldexp(@as(f32, 1), -@as(i32, @intCast(5 + rng.uintLessThan(u32, 4))));
        for (self.counts, 0..) |*x, r| x.* = if (self.form.sparse and r == 1 and shape.rows > 1) 0 else @intCast(shape.k);
        for (self.tokens, 0..) |*x, i| x.* = @intCast((i * 4096 / shape.k + rng.uintLessThan(usize, 4096 / shape.k)) % self.span());
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            if (self.form.sparse) {
                // Keep key rows finite in every list: plants differ by query row and all other rows stay finite.
                if (shape.rows > 2) self.q[2 * heads * lw + 4] = c.toBf16(std.math.inf(f32));
                if (shape.rows > 3) self.q[3 * heads * lw + 5] = c.toBf16(-std.math.inf(f32));
            } else {
                // Only the fullest causal row sees the final key; earlier rows stay finite.
                const last = self.span() - 1;
                self.codes[last * lw + 9] = 0x7f;
                self.scales[last] = std.math.inf(f32);
            }
        }
        if (fill == .stress) {
            for (self.codes, 0..) |*x, i| x.* = ([_]u8{ 0, 0x80, 1, 0x81, 2, 0x82, 0x7e, 0xfe })[i % 8];
        }
        if (fill == .random) self.adversarial();
        const launch_scales = if (self.form.control) blk: {
            const changed = try a.dupe(f32, self.scales);
            for (changed) |*x| x.* *= 1.5;
            break :blk changed;
        } else self.scales;
        defer if (self.form.control) a.free(launch_scales);
        const p: i32 = @intCast(self.at);
        self.dev = .{
            try self.rig.input(std.mem.sliceAsBytes(self.q)),        try self.rig.input(self.codes),
            try self.rig.input(std.mem.sliceAsBytes(launch_scales)), try self.rig.input(std.mem.sliceAsBytes(self.tokens)),
            try self.rig.input(std.mem.sliceAsBytes(self.counts)),   try self.rig.input(std.mem.asBytes(&p)),
        };
        const n = self.partialHeads();
        self.parts = .{ try self.rig.output(n * lw * 4), try self.rig.output(n * 4), try self.rig.output(n * 4) };
        self.out = try self.rig.output(self.semanticCount() * 2);
        return true;
    }

    fn adversarial(self: *Entry) void {
        const r = self.shape.rows - 1;
        const n = self.seen(r);
        const q = self.q[(r * heads) * lw ..][0..lw];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        const tile = @min(chunk, (n - 1) / 2);
        if (tile == 0) return;
        const spots = [3]usize{ tile - 1, tile, @min(2 * tile, n - 1) };
        for (0..n) |i| {
            const row = self.key(r, i);
            @memset(self.codes[row * lw ..][0..lw], 0);
            self.scales[row] = 1;
        }
        for (spots, 0..) |i, which| {
            const row = self.key(r, i);
            const kv = self.codes[row * lw ..][0..lw];
            kv[0] = putCode(if (which < 2) 16 else 32);
            kv[1] = 0x80;
            kv[2] = 1;
            kv[3] = 0x81;
        }
    }

    pub fn run(self: *Entry) !void {
        const rows64: i64 = @intCast(self.shape.rows);
        const w: i64 = if (self.form.sparse) @intCast(self.shape.k) else 0;
        var args: cuda.Args = .{};
        inline for (.{ self.dev[0], self.dev[1], self.dev[2], self.dev[5], self.dev[3], self.dev[4], self.parts[0], self.parts[1], self.parts[2], rows64, @as(i64, heads), w, scale, @as(c_int, @intFromBool(self.form.sparse)) }) |x| args.add(x);
        const shared = if (self.form.hb == 32) glm.glue.chunksSmem(32) else glm.glue.chunksSmem(16);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = (heads + self.form.hb - 1) / self.form.hb, .z = @intCast(self.nch) }, .block = .{ .x = 256 }, .shared = shared }, self.rig.s, &args);
        var merge: cuda.Args = .{};
        inline for (.{ self.parts[0], self.parts[1], self.parts[2], self.out, self.dev[4], rows64, @as(i64, heads), @as(c_int, @intCast(self.nch)), @as(c_int, @intFromBool(self.form.sparse)) }) |x| merge.add(x);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = heads }, .block = .{ .x = 128 } }, self.rig.s, &merge);
        try self.rig.s.synchronize();
    }

    fn value(self: *const Entry, r: usize, i: usize, d: usize) f64 {
        const row = self.key(r, i);
        return c.fromBf16(c.toBf16(@floatCast(e4m3(self.codes[row * lw + d]) * self.scales[row])));
    }
    fn attend(self: *const Entry, r: usize, h: usize, out: []f64, tol: []f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.seen(r);
        const terms = try a.alloc(f64, 4 * n + lw);
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
                x.* = c.fromBf16(self.q[(r * heads + h) * lw + d]) * self.value(r, i, d);
                abs += @abs(x.*);
            }
            scores[i] = pages.sumIn(.none, .sequential, products) * scale;
            ds[i] = abs * scale * 520 * 0x1p-24;
            most = pages.maxNan(most, scores[i]);
        }
        for (p, scores) |*x, score| x.* = @exp(score - most);
        const mass = pages.sumIn(.none, .sequential, p);
        for (out, tol, 0..) |*o, *t, d| {
            var spread: f64 = 0;
            for (col, p, 0..) |*x, weight, i| {
                const v = self.value(r, i, d);
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
        const want = try a.alloc(f64, 2 * n);
        defer a.free(want);
        const tol = want[n..];
        const raw = try self.rig.outputBytes();
        const merged = raw[self.partialHeads() * (lw + 2) * 4 ..];
        var planted: usize = 0;
        var considered: usize = 0;
        for (0..self.shape.rows) |r| {
            if (self.form.sparse and self.counts[r] == 0) for (0..heads * lw) |i| {
                const got = std.mem.readInt(u16, merged[(r * heads * lw + i) * 2 ..][0..2], .little);
                if (got != 0xffff) return try std.fmt.bufPrint(&self.msg, "count-zero row {d} element {d} was not left alone", .{ r, i });
            } else for (0..heads) |h| {
                const at = (r * heads + h) * lw;
                try self.attend(r, h, want[at..][0..lw], tol[at..][0..lw]);
                for (want[at..][0..lw], tol[at..][0..lw], 0..) |w, t, d| {
                    const got = c.read(.bf16, merged, at + d);
                    considered += 1;
                    if (c.classOf(w) != .finite) planted += 1;
                    if (c.classOf(got) != c.classOf(w)) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} column {d}: {t} where float64 gives {t}", .{ r, h, d, c.classOf(got), c.classOf(w) });
                    if (gate != .nonfinite and std.math.isFinite(w) and !(@abs(got - w) <= t)) return try std.fmt.bufPrint(&self.msg, "row {d} head {d} column {d}: error {e} exceeds {e}", .{ r, h, d, @abs(got - w), t });
                }
            };
        }
        if (gate == .nonfinite and (planted == 0 or planted == considered)) return try std.fmt.bufPrint(&self.msg, "planted non-finite values cover {d} of {d} outputs", .{ planted, considered });
        return null;
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    items: [forms.len]Entry = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
        const merge = try self.mod.?.function("tf_glm_merge");
        for (&self.items, forms) |*item, form| {
            const f = try self.mod.?.function(if (form.hb == 16) "tf_glm_chunks_16_fp8" else "tf_glm_chunks_32_fp8");
            try f.allowDynamicShared(if (form.hb == 16) glm.glue.chunksSmem(16) else glm.glue.chunksSmem(32));
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .merge = merge, .form = form };
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
