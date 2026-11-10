//! Mode 6 is a byte-preserving replacement; inherited exceptional outputs are reported separately from FP64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const data = @import("contract_fp8_attention.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Tuned = cuda.attention_tuned.Tuned;
const mode = cuda.attention_tuned.Mode.compact_w4h2s1;
const D = 256;

pub const Fault = enum { none, guard, missing_tail, wrong_page, erased_nonfinite, narrow_sum };
pub const Config = struct { paged: bool = false, fault: Fault = .none };
pub const entries = [_]Config{
    .{},                                      .{ .paged = true },              .{ .fault = .guard },      .{ .fault = .missing_tail },
    .{ .paged = true, .fault = .wrong_page }, .{ .fault = .erased_nonfinite }, .{ .fault = .narrow_sum },
};
const served = [_]c.Shape{
    .{ .rows = 33, .n = 24, .k = 65 },
    .{ .rows = 3584, .n = 24, .k = 4096 },
    .{ .rows = 4096, .n = 24, .k = 0 },
    .{ .rows = @import("qwen_prefill_config").max_rows, .n = 24, .k = 0 },
    .{ .rows = 1, .n = 2, .k = 0 },
    .{ .rows = 15, .n = 2, .k = 63 },
    .{ .rows = 16, .n = 2, .k = 64 },
    .{ .rows = 17, .n = 24, .k = 127 },
    .{ .rows = 31, .n = 2, .k = 127 },
    .{ .rows = 32, .n = 24, .k = 127 },
    .{ .rows = 63, .n = 2, .k = 127 },
    .{ .rows = 64, .n = 24, .k = 127 },
    .{ .rows = 65, .n = 2, .k = 127 },
    .{ .rows = 1, .n = 2, .k = 32767 },
    .{ .rows = 1, .n = 2, .k = 65535 },
    .{ .rows = 1, .n = 2, .k = 131071 },
};
fn depth(s: c.Shape) usize {
    return 256 + (s.k + s.rows + 63) / 64 * 24;
}
pub fn spec(config: Config) c.Spec {
    return .{
        .name = switch (config.fault) {
            .none => if (config.paged) "cuda compact exact attention paged" else "cuda compact exact attention whole",
            .guard => "CONTROL compact attention guard write",
            .missing_tail => "CONTROL compact attention omitted final query",
            .wrong_page => "CONTROL compact attention swapped page table",
            .erased_nonfinite => "CONTROL compact attention erased exceptional query",
            .narrow_sum => "CONTROL compact attention BF16 accumulator replay",
        },
        .accum = .f32,
        .depth = depth,
        .ops = 3,
        .out = .bf16,
        .replaces = "current hpattn8_kernel or hpattn8_whole_kernel<256,8,2,2>",
        .nan = .{ .stated = "same bytes as current prompt attention, including inherited exceptional behavior; FP64 discrepancies are reported" },
        .tiles = .{ .{ .size = 32 }, .{ .size = 2, .step = 2, .min = 2 }, .{ .size = 64, .min = 0 } },
        .served = &served,
        .sums = .{ .why = "equal-score stress preserves the current BF16-P and FP32-PV accumulation" },
    };
}

pub const Cases = struct {
    items: [entries.len]Case = undefined,
    initialized: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: @import("check.zig").Gpu, stream: cuda.Stream, sm: u32, image: []const u8) !void {
        if (sm != 120 and sm != 121) return;
        for (&self.items, entries) |*item, config| {
            item.* = .{ .base = .{ .rig = .{ .gpu = gpu, .s = stream }, .image = &.{}, .config = .{ .paged = config.paged, .quality = false, .prepare_only = true } }, .config = config, .image = image };
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
    base: data.Case,
    config: Config,
    image: []const u8,
    rig: *Rig = undefined,
    tuned: ?Tuned = null,
    candidate: cuda.attention_tuned.Input = undefined,
    msg: [256]u8 = undefined,

    fn clear(self: *Case) void {
        self.base.rig.s.synchronize() catch {};
        if (self.tuned) |*t| t.deinit();
        self.tuned = null;
        self.base.clear();
    }
    pub fn deinit(self: *Case) void {
        self.clear();
        self.base.rig.deinit();
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
            .missing_tail, .wrong_page => .shapes,
            .erased_nonfinite => .nonfinite,
            .narrow_sum => .sums,
        } };
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.clear();
        self.rig = &self.base.rig;
        if (self.config.fault == .missing_tail and shape.rows < 2) return false;
        if (!try self.base.prepare(shape, fill, seed)) return false;
        self.base.original_formula = true;
        const a = self.base.arena.?.allocator();
        self.candidate = self.base.input;
        if (self.config.fault == .wrong_page) {
            if (self.base.pages.len < 2) return false;
            const pages = try a.dupe(i32, self.base.pages);
            std.mem.swap(i32, &pages[0], &pages[1]);
            self.candidate.pages = try self.rig.input(std.mem.sliceAsBytes(pages));
        } else if (self.config.fault == .erased_nonfinite) {
            const q = try a.dupe(u16, self.base.q);
            for (q) |*v| if (!std.math.isFinite(c.fromBf16(v.*))) {
                v.* = 0;
            };
            self.candidate.q = try self.rig.input(std.mem.sliceAsBytes(q));
        }
        self.tuned = try Tuned.initCompact(self.rig.gpu.d, self.image, try self.rig.gpu.ctx.capability());
        return true;
    }
    pub fn run(self: *Case) !void {
        var input = self.candidate;
        if (self.config.fault == .missing_tail) input.shape.rows -= 1;
        try self.tuned.?.run(self.rig.s, mode, input);
        try self.rig.s.synchronize();
        if (self.config.fault == .guard) {
            const output: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = input.output - 1, .len = 1, .borrowed = true };
            try output.upload(0, &.{0});
        } else if (self.config.fault == .narrow_sum and self.base.fill == .stress) {
            const a = self.base.arena.?.allocator();
            const values = try a.alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const output = try a.alloc(u16, self.count());
            for (output, values) |*v, x| v.* = c.toBf16(@floatCast(x));
            const buffer: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = input.output, .len = output.len * 2, .borrowed = true };
            try buffer.upload(0, std.mem.sliceAsBytes(output));
        }
    }
    pub fn count(self: *const Case) usize {
        return self.base.count();
    }
    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        try self.base.reference(want, magnitude);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.base.simulate(acc, order, out);
    }
    pub fn bounds(self: *Case, want: []const f64, magnitude: []const f64, out: []f64) !void {
        try self.base.bounds(want, magnitude, out);
    }
    pub fn replaced(self: *Case) ![]const u8 {
        try self.base.launchBaseline();
        return self.rig.bytes(self.base.baseline_out, self.count() * 2);
    }
    pub fn check(self: *Case, gate: g.Gate) !?[]const u8 {
        if (gate != .nonfinite) return null;
        const output = try self.rig.outputBytes();
        const old = try self.replaced();
        if (!std.mem.eql(u8, output, old)) return try std.fmt.bufPrint(&self.msg, "exceptional outputs differ from current kernel at byte {d}", .{std.mem.indexOfDiff(u8, output, old).?});
        if (try self.base.reference_state.?.check(&self.base, self.base.shape.rows * self.base.shape.n, 1.0 / 16.0, output, .nonfinite)) |why| {
            std.debug.print("INHERITED current prompt-attention FP64 discrepancy: {s}\n", .{why});
        }
        return null;
    }
};

fn checkLimits(gpu: @import("check.zig").Gpu, image: []const u8, sm: u32) !usize {
    const registry = core.contract.registry;
    const index: usize = if (sm == 120) 3 else 4;
    const release = try core.contract.cubin.releaseFor(gpu.gpa, image, sm);
    var module = try cuda.Module.load(gpu.d, image);
    defer module.unload();
    var count: usize = 0;
    var problems: usize = 0;
    for (registry.all_cuda) |record| {
        if (!std.mem.eql(u8, record.launch.image, "prefill_attention") or
            !std.mem.startsWith(u8, record.launch.symbol, "tf_compact_w4h2s1_")) continue;
        count += 1;
        const name = try gpu.gpa.dupeSentinel(u8, record.launch.symbol, 0);
        defer gpu.gpa.free(name);
        const f = try module.function(name);
        const regs = try f.attribute(.num_regs);
        const shared_bytes = try f.attribute(.shared_size_bytes);
        const local = try f.attribute(.local_size_bytes);
        const bound = try f.attribute(.max_threads_per_block);
        const expected = registry.usageFor(record, release)[index];
        const ok = expected.regs != 0 and expected.regs == regs and expected.static == shared_bytes and
            expected.local == local and bound >= 128 and record.launch.dynamic == 49152;
        problems += @intFromBool(!ok);
        std.debug.print("{s} compact attention limits {s}: regs={d} static={d} local={d} bound={d}\n", .{ if (ok) "PASS" else "FAIL", record.launch.symbol, regs, shared_bytes, local, bound });
    }
    if (count != 2) {
        problems += 1;
        std.debug.print("FAIL compact attention: expected two launch records, found {d}\n", .{count});
    }
    return problems;
}

pub fn run(gpu: @import("check.zig").Gpu, image: []const u8) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedExactAttentionDevice;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: Cases = .{};
    defer cases.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try cases.append(&kernels, gpu, stream, sm, image);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0xea2433, &tally);
    const limits = try checkLimits(gpu, image, sm);
    std.debug.print("SUMMARY compact exact attention: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limits\n", .{ tally.passed, tally.failed, tally.caught, tally.missed, limits });
    if (tally.problems() + limits != 0) return error.TestFailed;
}
