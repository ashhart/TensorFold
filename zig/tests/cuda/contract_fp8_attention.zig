//! Opt-in FP8 prompt attention through the production wrapper, with distinct mode and current-formula references.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const attn = @import("contract_attn.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Native = cuda.fp8_attention_prompt.Native;
const Mode = cuda.fp8_attention_prompt.Mode;
const D = 256;
const scale: f32 = 1.0 / 16.0;
const decode = core.quant.fp8.e4m3;
const round8 = core.quant.fp8.toE4m3;

pub const Fault = enum { none, guard, missing_tail, erased_nonfinite, narrow_sum };
pub const Config = struct { mode: Mode = .residual, paged: bool = false, fault: Fault = .none, quality: bool = true, prepare_only: bool = false };
pub const entries = [_]Config{
    .{ .mode = .single },            .{ .mode = .single, .paged = true },
    .{ .mode = .residual },          .{ .mode = .residual, .paged = true },
    .{ .fault = .guard },            .{ .fault = .missing_tail, .paged = true },
    .{ .fault = .erased_nonfinite }, .{ .fault = .narrow_sum },
};
const served = [_]c.Shape{
    .{ .rows = 33, .n = 24, .k = 65 },  .{ .rows = 1, .n = 2, .k = 0 },
    .{ .rows = 65, .n = 2, .k = 127 },  .{ .rows = 1, .n = 2, .k = 32767 },
    .{ .rows = 1, .n = 2, .k = 65535 }, .{ .rows = 1, .n = 2, .k = 131071 },
};
fn depth(s: c.Shape) usize {
    return 256 + (s.k + s.rows + 63) / 64 * 24;
}
pub fn spec(config: Config) c.Spec {
    return .{
        .name = switch (config.fault) {
            .guard => "CONTROL fp8 attention guard write",
            .missing_tail => "CONTROL fp8 attention missing final query",
            .erased_nonfinite => "CONTROL fp8 attention erased exceptional input",
            .narrow_sum => "CONTROL fp8 attention BF16 sum replay",
            .none => if (config.mode == .single)
                (if (config.paged) "cuda FP8 QK single paged" else "cuda FP8 QK single whole")
            else
                (if (config.paged) "cuda FP8 QK residual paged" else "cuda FP8 QK residual whole"),
        },
        .accum = .f32,
        .depth = depth,
        .ops = 5,
        .out = .bf16,
        .nan = .{ .stated = "visible exceptional operands use original-input FP64 attention; hidden keys are excluded" },
        .tiles = .{ .{ .size = 32 }, .{ .size = 2, .step = 2, .min = 2 }, .{ .size = 64, .min = 0 } },
        .served = &served,
        .sums = .{ .why = "equal-score stress proves P x V accumulation; native QK sums also require the score-probe receipt" },
    };
}

pub fn run(gpu: @import("check.zig").Gpu) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedFp8AttentionDevice;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: Cases = .{};
    defer cases.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try cases.append(&kernels, gpu, stream, sm, cuda.kernels.fp8_attention_prompt);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0xf82433, &tally);
    const limits = try checkLimits(gpu, sm);
    std.debug.print("SUMMARY FP8 attention: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limit problems\n", .{ tally.passed, tally.failed, tally.caught, tally.missed, limits });
    if (tally.problems() + limits != 0) return error.TestFailed;
}

fn checkLimits(gpu: @import("check.zig").Gpu, sm: u32) !usize {
    const registry = core.contract.registry;
    const index: usize = if (sm == 120) 3 else 4;
    const release = try core.contract.cubin.releaseFor(gpu.gpa, cuda.kernels.fp8_attention_prompt, sm);
    var module = try cuda.Module.load(gpu.d, cuda.kernels.fp8_attention_prompt);
    defer module.unload();
    var found: usize = 0;
    var problems: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "fp8_attention_prompt")) continue;
        found += 1;
        const symbol = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(symbol);
        const f = try module.function(symbol);
        const regs = try f.attribute(.num_regs);
        const shared_bytes = try f.attribute(.shared_size_bytes);
        const bound = try f.attribute(.max_threads_per_block);
        const expected = registry.usageFor(record, release)[index];
        const ok = expected.regs != 0 and regs == expected.regs and shared_bytes == expected.static and
            bound >= record.launch.threads;
        problems += @intFromBool(!ok);
        std.debug.print("{s} FP8 attention limits {s}: regs={d} static={d} bound={d}\n", .{ if (ok) "PASS" else "FAIL", record.launch.symbol, regs, shared_bytes, bound });
    }
    if (found != 6) {
        std.debug.print("FAIL FP8 attention: {d} of 6 launch records registered\n", .{found});
        problems += 1;
    }
    return problems;
}

pub const Cases = struct {
    items: [entries.len]Case = undefined,
    initialized: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: @import("check.zig").Gpu, stream: cuda.Stream, sm: u32, image: []const u8) !void {
        if (sm != 120 and sm != 121) return;
        for (&self.items, entries) |*item, config| {
            item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config, .image = image };
            self.initialized += 1;
            try list.append(gpu.gpa, item.kernel());
        }
    }
    pub fn deinit(self: *Cases) void {
        for (self.items[0..self.initialized]) |*item| item.deinit();
        self.initialized = 0;
    }
};

pub const Case = struct {
    rig: Rig,
    config: Config,
    image: []const u8,
    arena: ?std.heap.ArenaAllocator = null,
    native: ?Native = null,
    baseline: ?cuda.Module = null,
    reference_state: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    fill: g.Fill = .random,
    q: []u16 = &.{},
    quantized: []f64 = &.{},
    q_bad: []bool = &.{},
    cache: []u8 = &.{},
    pages: []i32 = &.{},
    bad_key: []usize = &.{},
    input: cuda.fp8_attention_prompt.Input = undefined,
    baseline_out: u64 = 0,
    original_formula: bool = false,
    msg: [256]u8 = undefined,

    pub fn clear(self: *Case) void {
        self.rig.s.synchronize() catch {};
        if (self.native) |*native| native.deinit();
        self.native = null;
        if (self.baseline) |*module| module.unload();
        self.baseline = null;
        if (self.reference_state) |*r| r.deinit();
        self.reference_state = null;
        self.rig.reset();
        if (self.arena) |*a| a.deinit();
        self.arena = null;
    }
    pub fn deinit(self: *Case) void {
        self.clear();
        self.rig.deinit();
    }
    fn release(ctx: *anyopaque) void {
        const self: *Case = @ptrCast(@alignCast(ctx));
        self.clear();
    }
    const vt = blk: {
        var h = g.hooks(Case);
        h.release = release;
        break :blk h;
    };
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.config), .ctx = self, .vt = &vt, .control = switch (self.config.fault) {
            .none => null,
            .guard => .repeat,
            .missing_tail => .shapes,
            .erased_nonfinite => .nonfinite,
            .narrow_sum => .sums,
        } };
    }
    fn kvHeads(self: *const Case) usize {
        return if (self.shape.n == 24) 4 else 1;
    }
    fn capacity(self: *const Case) usize {
        return self.shape.k + self.shape.rows;
    }
    fn pageTokens(self: *const Case) usize {
        return if (self.config.paged) 48 else self.capacity();
    }
    fn planeRows(self: *const Case) usize {
        return self.kvHeads() * self.pageTokens();
    }
    fn pageBytes(self: *const Case) usize {
        return std.mem.alignForward(usize, self.planeRows() * (2 * D + 8), 256);
    }
    fn address(self: *const Case, head: usize, key: usize, is_value: bool, scales: bool) usize {
        const n = self.planeRows();
        const page = if (self.config.paged) @as(usize, @intCast(self.pages[key / 48])) * self.pageBytes() else 0;
        const row = head * self.pageTokens() + key % self.pageTokens();
        return page + if (scales) 2 * n * D + @as(usize, @intFromBool(is_value)) * n * 4 + row * 4 else @as(usize, @intFromBool(is_value)) * n * D + row * D;
    }
    fn setScale(self: *Case, head: usize, key: usize, is_value: bool, x: f32) void {
        std.mem.writeInt(u32, self.cache[self.address(head, key, is_value, true)..][0..4], @bitCast(x), .little);
    }
    fn value(self: *const Case, head: usize, key: usize, d: usize, v: bool, rounded: bool) f64 {
        const a = self.address(head, key, v, false);
        const s: f32 = @bitCast(std.mem.readInt(u32, self.cache[self.address(head, key, v, true)..][0..4], .little));
        const raw = decode(self.cache[a + d]);
        const x: f32 = @as(f32, @floatCast(raw)) * s;
        return if (rounded) c.fromBf16(c.toBf16(x)) else raw * s;
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const maximum_rows = if (self.config.prepare_only) @import("qwen_prefill_config").max_rows else 65;
        if ((shape.n != 2 and shape.n != 24) or shape.rows == 0 or shape.rows > maximum_rows or
            shape.k + shape.rows > 131072) return false;
        if (self.config.fault == .missing_tail and shape.rows == 1) return false;
        self.clear();
        self.shape = shape;
        self.fill = fill;
        self.original_formula = false;
        self.arena = .init(self.rig.gpu.gpa);
        errdefer self.clear();
        const a = self.arena.?.allocator();
        self.q = try a.alloc(u16, self.count());
        self.quantized = try a.alloc(f64, self.count());
        self.q_bad = try a.alloc(bool, shape.rows * shape.n);
        self.bad_key = try a.alloc(usize, self.kvHeads());
        @memset(self.bad_key, std.math.maxInt(usize));
        const page_count = if (self.config.paged) (self.capacity() + 47) / 48 else 0;
        self.pages = try a.alloc(i32, page_count);
        for (self.pages, 0..) |*p, i| p.* = @intCast(page_count - 1 - i);
        self.cache = try a.alloc(u8, if (self.config.paged) page_count * self.pageBytes() else self.pageBytes());
        @memset(self.cache, 0x7f);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.q) |*q| q.* = c.toBf16((rng.float(f32) * 2 - 1) * 3);
        for (0..self.kvHeads()) |head| for (0..self.capacity()) |key| for (0..2) |v| {
            self.setScale(head, key, v != 0, (1 + @as(f32, @floatFromInt((key + head * 3) % 7))) / 64);
            for (0..D) |d| self.cache[self.address(head, key, v != 0, false) + d] =
                (0x28 + rng.uintLessThan(u8, 40)) | (if ((d + key) % 3 == 0) @as(u8, 128) else 0);
        };
        if (fill == .stress) try self.stress(rng) else if (fill == .nonfinite) self.plant();
        if (self.config.prepare_only) {
            @memset(self.quantized, 0);
            @memset(self.q_bad, false);
        } else self.quantize();
        self.reference_state = try attn.Ref.init(self.rig.gpu.gpa, D, self.capacity());
        var qsource = self.q;
        var cache_source = self.cache;
        if (self.config.fault == .erased_nonfinite) {
            qsource = try a.dupe(u16, self.q);
            cache_source = try a.dupe(u8, self.cache);
            for (qsource) |*q| if (!std.math.isFinite(c.fromBf16(q.*))) {
                q.* = 0;
            };
            for (0..self.kvHeads()) |h| for (0..self.capacity()) |key| for (0..2) |v| {
                const sa = self.address(h, key, v != 0, true);
                const s: f32 = @bitCast(std.mem.readInt(u32, cache_source[sa..][0..4], .little));
                if (!std.math.isFinite(s)) std.mem.writeInt(u32, cache_source[sa..][0..4], @bitCast(@as(f32, 0)), .little);
                for (cache_source[self.address(h, key, v != 0, false)..][0..D]) |*b|
                    if (b.* & 127 == 127) {
                        b.* = 0;
                    };
            };
        }
        const cache_ptr = try self.rig.input(cache_source);
        const n = self.planeRows();
        self.input = .{
            .q = try self.rig.input(std.mem.sliceAsBytes(qsource)),
            .k = cache_ptr,
            .v = cache_ptr + n * D,
            .k_scales = cache_ptr + 2 * n * D,
            .v_scales = cache_ptr + 2 * n * D + n * 4,
            .output = try self.rig.output(self.count() * 2),
            .pages = if (page_count == 0) 0 else try self.rig.input(std.mem.sliceAsBytes(self.pages)),
            .shape = .{ .rows = @intCast(shape.rows), .prefix = @intCast(shape.k), .heads = @intCast(shape.n), .kv_heads = @intCast(self.kvHeads()), .capacity = @intCast(self.capacity()), .scale = scale, .page_tokens = if (self.config.paged) 48 else 0, .page_bytes = if (self.config.paged) self.pageBytes() else 0 },
        };
        self.baseline_out = try self.rig.scratch(self.count() * 2);
        if (!self.config.prepare_only)
            self.native = try Native.init(self.rig.gpu.d, self.image, try self.rig.gpu.ctx.capability());
        self.baseline = try cuda.Module.load(self.rig.gpu.d, cuda.kernels.prefill_attention);
        return true;
    }
    fn plant(self: *Case) void {
        self.q[0] = c.toBf16(std.math.nan(f32));
        if (self.shape.rows > 1) self.q[self.shape.n * D] = c.toBf16(std.math.inf(f32));
        if (self.shape.rows > 2) self.q[2 * self.shape.n * D] = c.toBf16(-std.math.inf(f32));
        const key = @min(self.capacity() - 1, self.shape.k + 1);
        self.cache[self.address(0, key, true, false) + 3] = 127;
        self.bad_key[0] = key;
    }
    fn stress(self: *Case, rng: std.Random) !void {
        @memset(self.q, 0);
        const a = self.arena.?.allocator();
        const n = @min(self.shape.k, 2048);
        const values = try a.alloc(f32, n);
        attn.mirrored(rng, values);
        for (0..self.kvHeads()) |h| for (0..self.capacity()) |key| {
            self.setScale(h, key, false, 1);
            @memset(self.cache[self.address(h, key, false, false)..][0..D], 0);
            self.setScale(h, key, true, if (key < n) @abs(c.fromBf16(c.toBf16(values[key]))) else 1);
            @memset(self.cache[self.address(h, key, true, false)..][0..D], if (key >= n) 0 else if (values[key] < 0) 0xb8 else 0x38);
        };
    }
    fn quantize(self: *Case) void {
        for (0..self.shape.rows * self.shape.n) |row| {
            const input = self.q[row * D ..][0..D];
            const out = self.quantized[row * D ..][0..D];
            var maximum: f32 = 0;
            var bad = false;
            for (input) |q| {
                const v = c.fromBf16(q);
                bad = bad or !std.math.isFinite(v);
                maximum = @max(maximum, @abs(v));
            }
            self.q_bad[row] = bad;
            if (bad) {
                @memset(out, 0);
                continue;
            }
            const hs: f32 = if (maximum > 0) @max(maximum / 448, 0x1p-126) else 1;
            const inv: f32 = 1 / hs;
            var residual: [D]f32 = undefined;
            var low_max: f32 = 0;
            for (input, out, &residual) |q, *v, *r| {
                const x = c.fromBf16(q);
                const rounded: f32 = @floatCast(round8(@as(f32, x * inv)));
                v.* = @as(f64, rounded) * hs;
                const represented: f32 = rounded * hs;
                r.* = x - represented;
                low_max = @max(low_max, @abs(r.*));
            }
            if (self.config.mode == .residual) {
                const ls: f32 = if (low_max > 0) @max(low_max / 448, 0x1p-126) else 1;
                const inverse: f32 = 1 / ls;
                for (out, residual) |*v, r| v.* += round8(@as(f32, r * inverse)) * ls;
            }
        }
    }
    pub fn count(self: *const Case) usize {
        return self.shape.rows * self.shape.n * D;
    }
    pub fn load(self: *Case, i: usize, h: *attn.Head) !void {
        const row = i / self.shape.n;
        const head = i % self.shape.n;
        const kh = head / (self.shape.n / self.kvHeads());
        h.n = self.shape.k + row + 1;
        const original = self.original_formula or self.q_bad[i] or self.bad_key[kh] < h.n;
        for (h.q, 0..) |*q, d| q.* = if (original) c.fromBf16(self.q[i * D + d]) else self.quantized[i * D + d];
        for (0..h.n) |key| for (0..D) |d| {
            h.k[key * D + d] = self.value(kh, key, d, false, original);
            h.v[key * D + d] = self.value(kh, key, d, true, true);
        };
    }
    pub fn run(self: *Case) !void {
        var input = self.input;
        if (self.config.fault == .missing_tail) input.shape.rows -= 1;
        try self.native.?.run(self.rig.s, self.config.mode, input);
        try self.rig.s.synchronize();
        if (self.config.fault == .guard) {
            const out: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.input.output - 1, .len = 1, .borrowed = true };
            try out.upload(0, &.{0});
        } else if (self.config.fault == .narrow_sum and self.fill == .stress) {
            const a = self.arena.?.allocator();
            const simulated = try a.alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, simulated);
            const bits = try a.alloc(u16, self.count());
            for (bits, simulated) |*b, v| b.* = c.toBf16(@floatCast(v));
            const out: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.input.output, .len = bits.len * 2, .borrowed = true };
            try out.upload(0, std.mem.sliceAsBytes(bits));
        }
    }
    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        try self.reference_state.?.reference(self, self.shape.rows * self.shape.n, scale, want, magnitude);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.reference_state.?.simulate(self, self.shape.rows * self.shape.n, scale, acc, order, out);
    }
    pub fn bounds(self: *Case, want: []const f64, magnitude: []const f64, out: []f64) !void {
        for (out, want, magnitude) |*bound, value_, mag| bound.* = if (self.fill == .stress)
            c.boundExact(spec(self.config), value_)
        else
            0x1p-8 * @abs(value_) + 0x1p-7 * mag + 1e-6;
    }
    pub fn check(self: *Case, at: g.Gate) !?[]const u8 {
        if (try self.reference_state.?.check(self, self.shape.rows * self.shape.n, scale, try self.rig.outputBytes(), at)) |why|
            return why;
        if (at == .shapes and self.config.quality and self.config.fault == .none) return self.quality();
        return null;
    }
    pub fn launchBaseline(self: *Case) !void {
        const name = if (self.config.paged)
            "_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi"
        else
            "_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
        const f = try self.baseline.?.function(name);
        try f.allowDynamicShared(98304);
        var args: cuda.Args = .{};
        const p = self.input;
        for ([_]u64{ p.q, p.k, p.v, p.k_scales, p.v_scales, self.baseline_out }) |v| args.add(v);
        for ([_]usize{ self.shape.k, self.shape.rows, self.shape.n, self.kvHeads(), self.shape.n / self.kvHeads() }) |v|
            args.add(@as(i32, @intCast(v)));
        args.add(scale);
        args.add(@as(i32, @intCast(self.capacity())));
        if (self.config.paged) {
            args.add(p.pages);
            args.add(@as(i64, @intCast(self.pageBytes())));
            args.add(@as(i32, 48));
        }
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((self.shape.rows + 63) / 64), .y = @intCast(self.shape.n / 2) }, .block = .{ .x = 256 }, .shared = 98304 }, self.rig.s, &args);
    }
    pub fn quality(self: *Case) !?[]const u8 {
        try self.launchBaseline();
        try self.rig.s.synchronize();
        const native_bytes = try self.rig.outputBytes();
        const baseline_bytes = try self.rig.bytes(self.baseline_out, self.count() * 2);
        const a = self.arena.?.allocator();
        const want = try a.alloc(f64, self.count());
        const magnitude = try a.alloc(f64, self.count());
        self.original_formula = true;
        defer self.original_formula = false;
        try self.reference(want, magnitude);
        var native_sq: f64 = 0;
        var base_sq: f64 = 0;
        var native_max: f64 = 0;
        var base_max: f64 = 0;
        var ns: [D]f64 = @splat(0);
        var bs: [D]f64 = @splat(0);
        var nm: [D]f64 = @splat(0);
        var bm: [D]f64 = @splat(0);
        for (want, 0..) |v, i| {
            const x = @abs(c.read(.bf16, native_bytes, i) - v);
            const y = @abs(c.read(.bf16, baseline_bytes, i) - v);
            native_sq += x * x;
            base_sq += y * y;
            native_max = @max(native_max, x);
            base_max = @max(base_max, y);
            ns[i % D] += x * x;
            bs[i % D] += y * y;
            nm[i % D] = @max(nm[i % D], x);
            bm[i % D] = @max(bm[i % D], y);
        }
        std.debug.print("FP8 attention original-formula errors mode={t} paged={} native_sq={e} current_sq={e} native_max={e} current_max={e}\n", .{ self.config.mode, self.config.paged, native_sq, base_sq, native_max, base_max });
        var worse: usize = 0;
        var worst: usize = 0;
        for (ns, bs, nm, bm, 0..) |x, y, xx, yy, column| {
            if (x > y or xx > yy) worse += 1;
            if (xx - yy > nm[worst] - bm[worst]) worst = column;
        }
        std.debug.print("FP8 attention columns worse={d}, worst={d}, native={e}, current={e}\n", .{ worse, worst, nm[worst], bm[worst] });
        if (worse != 0) return try std.fmt.bufPrint(&self.msg, "original-formula error worsened in {d} columns; worst {d}: {e}/{e}", .{ worse, worst, nm[worst], bm[worst] });
        return null;
    }
};
