//! Compact DFlash taps preserve the legacy ring's BF16 rows through prompt, wrap and selected-path gathers.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen = @import("qwen35");
const Rig = @import("contract_rig.zig").Rig;
const Gpu = @import("check.zig").Gpu;
const c = core.contract.spec;
const g = core.contract.gates;
const capacity = 2048 + 64;
const legacy = 2048 + @max(8192, @import("qwen_prefill_config").max_rows);
const taps = 5;
const Seg = qwen.cache.Seg;
const Selected = extern struct { n: c_int, rows: [64]c_int };
const pick = [_]u32{ 0, 7, 31, 63 };

pub const Fault = enum { none, tail, nonfinite, guard };
const forms = [_]Fault{ .none, .tail, .nonfinite, .guard };
const served = [_]c.Shape{
    .{ .rows = 8192, .n = 5120, .k = 0 },
    .{ .rows = 4096, .n = 256, .k = 2097 },
    .{ .rows = 2113, .n = 256, .k = 4217 },
    .{ .rows = 2112, .n = 256, .k = 0 },
    .{ .rows = 2048, .n = 256, .k = 0 },
    .{ .rows = 65, .n = 256, .k = 2111 },
};
fn depth(_: c.Shape) usize {
    return 1;
}
pub fn spec(fault: Fault) c.Spec {
    return .{
        .name = switch (fault) {
            .none => "cuda qwen35 compact target tap ring",
            .tail => "CONTROL compact taps omit append tail",
            .nonfinite => "CONTROL compact taps erase NaN",
            .guard => "CONTROL compact taps overwrite guard",
        },
        .accum = .none,
        .depth = depth,
        .out = .raw,
        .replaces = "tf_q35_tap with the legacy prompt-sized ring and identical BF16 gather kernels",
        .nan = .{ .stated = "BF16 of the FP32 residual add; NaN payloads equal the legacy CUDA path" },
        .served = &served,
        .tiles = .{ .{ .size = 64 }, .{ .size = 256, .step = 256, .min = 256 }, .{} },
        .sums = .{ .why = "one FP32 addition of two BF16 inputs, then BF16 rounding; no reduction accumulator" },
    };
}

const Pass = struct {
    rows: usize,
    row0: usize,
    input: usize,
    start: usize,
    read: usize,
    reads: usize,
    selected: bool = false,
    offset: usize = 0,
    map: u64 = 0,
    params: u64 = 0,
    segs: [2]u64 = @splat(0),
};

pub const Case = struct {
    rig: Rig,
    fault: Fault,
    tap: cuda.Function,
    gather: cuda.Function,
    selected: cuda.Function,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    input: [2]u64 = @splat(0),
    rings: [2]u64 = @splat(0),
    outputs: [2]u64 = @splat(0),
    passes: [3]Pass = undefined,
    want: []u16 = &.{},
    words: usize = 0,
    msg: [220]u8 = undefined,

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
        self.rig.gpu.gpa.free(self.want);
        self.want = &.{};
    }
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = spec(self.fault), .ctx = self, .vt = &vt, .control = switch (self.fault) {
            .none => null,
            .tail => .shapes,
            .nonfinite => .nonfinite,
            .guard => .repeat,
        } };
    }
    const vt = g.hooks(Case);

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 8192 or shape.n == 0 or shape.n % 256 != 0 or shape.n > 5120) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.want);
        self.want = &.{};
        self.shape = shape;
        const prompt_read = if (shape.rows > 1) shape.rows - 1 else 1;
        const prompt_keep = @min(prompt_read, 2048);
        const final_keep = @min(shape.rows + 72, 2048);
        self.passes = .{
            .{ .rows = shape.rows, .row0 = 0, .input = 0, .start = shape.k, .read = shape.k + prompt_read - prompt_keep, .reads = prompt_keep },
            .{ .rows = 64, .row0 = 7, .input = shape.rows, .start = shape.k + shape.rows, .read = shape.k + shape.rows, .reads = pick.len, .selected = true },
            // Reject 56 rows of the prior verify, then replace them with this changed-input frame.
            .{ .rows = 64, .row0 = 7, .input = shape.rows + 71, .start = shape.k + shape.rows + 8, .read = shape.k + shape.rows + 72 - final_keep, .reads = final_keep },
        };
        const pitch = taps * shape.n;
        self.words = 0;
        for (&self.passes) |*p| {
            p.offset = self.words;
            self.words += p.reads * pitch;
        }
        self.want = try a.alloc(u16, self.words);
        const input_rows = shape.rows + 142 + taps;
        const h = try a.alloc(u16, input_rows * shape.n);
        defer a.free(h);
        const r = try a.alloc(u16, h.len);
        defer a.free(r);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (h, r, 0..) |*x, *y, i| {
            x.* = c.toBf16(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -4096, 4096))) / 256);
            y.* = c.toBf16(@as(f32, @floatFromInt(rng.intRangeAtMost(i32, -4096, 4096))) / 512);
            if (fill == .nonfinite) switch (i % shape.n) {
                0 => x.* = 0x7fc1,
                1 => {
                    x.* = 0x7f80;
                    y.* = 0xff80;
                },
                2 => x.* = 0x7f80,
                3 => x.* = 0xff80,
                else => {},
            };
        }
        self.input = .{ try self.rig.input(std.mem.sliceAsBytes(h)), try self.rig.input(std.mem.sliceAsBytes(r)) };
        self.rings = .{ try self.rig.scratch(capacity * pitch * 2), try self.rig.scratch(legacy * pitch * 2) };
        self.outputs = .{ try self.rig.output(self.words * 2), try self.rig.scratch(self.words * 2) };
        for (&self.passes) |*p| {
            const map = try a.alloc(c_int, p.rows + p.row0 + 7);
            defer a.free(map);
            @memset(map, 1);
            @memset(map[0..p.row0], 0);
            p.map = try self.rig.input(std.mem.sliceAsBytes(map));
            const params = [_]c_int{ 0, 0, @intCast(p.rows + p.row0) };
            p.params = try self.rig.input(std.mem.sliceAsBytes(&params));
            for (0..2) |which| {
                var sg = std.mem.zeroes(Seg);
                sg.tap = self.rings[which];
                sg.len = @intCast(p.start);
                sg.row0 = @intCast(p.row0);
                sg.rows = @intCast(p.rows);
                sg.tap_rows = @intCast(if (which == 0) capacity else legacy);
                sg.taps = taps;
                const segs = [_]Seg{ std.mem.zeroes(Seg), sg };
                p.segs[which] = try self.rig.input(std.mem.sliceAsBytes(&segs));
            }
        }
        self.referenceWords(h, r);
        return true;
    }

    fn source(self: *const Case, position: usize, through: usize) usize {
        var i = through + 1;
        while (i > 0) {
            i -= 1;
            const p = self.passes[i];
            if (position >= p.start and position < p.start + p.rows) return p.input + p.row0 + position - p.start;
        }
        unreachable;
    }
    fn referenceWords(self: *Case, h: []const u16, r: []const u16) void {
        const d = self.shape.n;
        for (self.passes, 0..) |p, pi| for (0..p.reads) |row| {
            const position = p.read + if (p.selected) pick[row] else row;
            const src = self.source(position, pi);
            for (0..taps) |tap| for (0..d) |col| {
                const index = (src + tap) * d + col;
                const sum: f32 = @floatCast(@as(f64, c.fromBf16(h[index])) + @as(f64, c.fromBf16(r[index])));
                self.want[p.offset + (row * taps + tap) * d + col] = c.toBf16(sum);
            };
        };
    }

    fn launch(self: *Case, f: cuda.Function, rows: usize, cols: usize, values: anytype) !void {
        var args: cuda.Args = .{};
        inline for (values) |value| args.add(value);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(cols) }, .block = .{ .x = 256 } }, self.rig.s, &args);
    }
    fn execute(self: *Case, which: usize) !void {
        const d = self.shape.n;
        const pitch = taps * d;
        for (self.passes, 0..) |p, pi| {
            // A captured bucket may launch extra blocks; P[2] must exclude them before touching rowseg.
            const blocks = p.rows + p.row0 + 7 - @as(usize, if (which == 0 and self.fault == .tail and pi == 2) 8 else 0);
            for (0..taps) |tap| try self.launch(self.tap, blocks, (d + 2047) / 2048, .{
                self.input[0] + (p.input + tap) * d * 2, self.input[1] + (p.input + tap) * d * 2,
                p.segs[which],                           p.map,
                @as(c_int, @intCast(tap)),               @as(c_int, @intCast(d)),
                p.params,
            });
            const cap: c_int = @intCast(if (which == 0) capacity else legacy);
            const out = self.outputs[which] + p.offset * 2;
            if (p.selected) {
                var selection: Selected = .{ .n = pick.len, .rows = @splat(0) };
                for (pick, 0..) |r, i| selection.rows[i] = @intCast(r);
                try self.launch(self.selected, p.reads, 1, .{ self.rings[which], cap, @as(c_longlong, @intCast(p.read)), selection, @as(c_int, @intCast(pitch)), out });
            } else try self.launch(self.gather, p.reads, 1, .{ self.rings[which], cap, @as(c_longlong, @intCast(p.read)), @as(c_int, @intCast(pitch)), out });
        }
    }
    pub fn run(self: *Case) !void {
        for (0..2) |which| try self.rig.zero(self.rings[which], (if (which == 0) capacity else legacy) * taps * self.shape.n * 2);
        try self.execute(0);
        if (self.fault == .nonfinite) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.outputs[0], .len = 2 };
            try b.fill8(0, self.rig.s.handle);
        } else if (self.fault == .guard) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.outputs[0] + self.words * 2, .len = 2 };
            try b.fill8(0, self.rig.s.handle);
        }
        try self.rig.s.synchronize();
        if (self.fault == .none) if (try self.check(.shapes)) |why| {
            std.debug.print("FAIL compact taps independent FP64 reference: {s}\n", .{why});
            return error.TestFailed;
        };
    }
    pub fn replaced(self: *Case) ![]const u8 {
        try self.execute(1);
        return self.rig.bytes(self.outputs[1], self.words * 2);
    }
    pub fn count(self: *Case) usize {
        return self.words * 2;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (std.mem.sliceAsBytes(self.want), want, mag) |value, *w, *m| {
            w.* = @floatFromInt(value);
            m.* = 0;
        }
    }
    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const raw = try self.rig.outputBytes();
        const got = std.mem.bytesAsSlice(u16, raw);
        for (got, self.want, 0..) |actual, expected, i| {
            const ok = if (std.math.isNan(c.fromBf16(expected))) std.math.isNan(c.fromBf16(actual)) else actual == expected;
            if (!ok) return try std.fmt.bufPrint(&self.msg, "BF16 word {d}: 0x{x:0>4}, expected FP64/FP32/RNE 0x{x:0>4}", .{ i, actual, expected });
        }
        return null;
    }
};

pub const Cases = struct {
    module: ?cuda.Module = null,
    draft: ?cuda.Module = null,
    items: [forms.len]Case = undefined,
    count: usize = 0,
    pub fn append(self: *Cases, list: *std.ArrayList(g.Kernel), gpu: Gpu, stream: cuda.Stream) !void {
        self.module = try cuda.Module.load(gpu.d, cuda.kernels.qwen35);
        self.draft = try cuda.Module.load(gpu.d, cuda.kernels.qwen35_draft);
        for (&self.items, forms) |*item, fault| {
            item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .fault = fault, .tap = try self.module.?.function("tf_q35_tap"), .gather = try self.draft.?.function("tf_q35d_gather_taps"), .selected = try self.draft.?.function("tf_q35d_gather_rows") };
            self.count += 1;
            try list.append(gpu.gpa, item.kernel());
        }
    }
    pub fn deinit(self: *Cases) void {
        for (self.items[0..self.count]) |*item| item.deinit();
        if (self.draft) |*mod| mod.unload();
        if (self.module) |*mod| mod.unload();
        self.* = .{};
    }
};

fn checkLimits(gpu: Gpu) !usize {
    const registry = core.contract.registry;
    const launches = [_]struct { image: []const u8, symbol: []const u8, threads: u32 = 256 }{
        .{ .image = "qwen35", .symbol = "tf_q35_tap" },
        .{ .image = "qwen35_draft", .symbol = "tf_q35d_gather_taps" },
        .{ .image = "qwen35_draft", .symbol = "tf_q35d_gather_rows" },
    };
    const sm = try gpu.ctx.capability();
    const chip: usize = for (core.contract.limits.sms, 0..) |target, i| {
        if (target.sm == sm) break i;
    } else return error.UnsupportedTapDevice;
    var problems: usize = 0;
    for (launches) |launch| {
        const image = if (std.mem.eql(u8, launch.image, "qwen35")) cuda.kernels.qwen35 else cuda.kernels.qwen35_draft;
        var module = try cuda.Module.load(gpu.d, image);
        defer module.unload();
        const release = try core.contract.cubin.releaseFor(gpu.gpa, image, sm);
        const name = try gpu.gpa.dupeSentinel(u8, launch.symbol, 0);
        defer gpu.gpa.free(name);
        const f = try module.function(name);
        const actual: core.contract.limits.Usage = .{
            .regs = @intCast(try f.attribute(.num_regs)),
            .static = @intCast(try f.attribute(.shared_size_bytes)),
            .local = @intCast(try f.attribute(.local_size_bytes)),
        };
        const bound = try f.attribute(.max_threads_per_block);
        var seen: usize = 0;
        var expected: core.contract.limits.Usage = .{ .regs = 0, .static = 0 };
        for (registry.all_cuda) |record| {
            if (!std.mem.eql(u8, record.launch.image, launch.image) or
                !std.mem.eql(u8, record.launch.symbol, launch.symbol)) continue;
            seen += 1;
            expected = registry.usageFor(record, release)[chip];
        }
        const ok = seen == 1 and expected.regs > 0 and actual.regs == expected.regs and
            actual.static == expected.static and actual.local == expected.local and bound >= launch.threads;
        problems += @intFromBool(!ok);
        std.debug.print("{s} compact taps limits {s}: regs={d} static={d} local={d} bound={d}, records={d}\n", .{ if (ok) "PASS" else "FAIL", launch.symbol, actual.regs, actual.static, actual.local, bound, seen });
    }
    return problems;
}

pub fn run(gpu: Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: Cases = .{};
    defer cases.deinit();
    var kernels: std.ArrayList(g.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    try cases.append(&kernels, gpu, stream);
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, kernels.items, 0x7a92433, &tally);
    const limits = try checkLimits(gpu);
    std.debug.print("SUMMARY compact target taps: {d} passed, {d} failed, {d} controls caught, {d} missed, {d} limits\n", .{ tally.passed, tally.failed, tally.caught, tally.missed, limits });
    if (tally.problems() + limits != 0) return error.TestFailed;
}

test "compact target tap contract covers the native BF16 write and both gather kernels" {
    std.mem.doNotOptimizeAway(&run);
    const s = spec(.none);
    try std.testing.expect(s.replaces != null and Case.vt.replaced != null);
    try std.testing.expectEqual(@as(usize, 8192), served[0].rows);
    try std.testing.expectEqual(@as(usize, 5120), served[0].n);
    try std.testing.expectEqual(@as(usize, 64), @typeInfo(@TypeOf(@as(Selected, undefined).rows)).array.len);
}
