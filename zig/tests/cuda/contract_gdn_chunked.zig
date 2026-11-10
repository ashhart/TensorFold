//! Chunkwise GDN state/output gates against scalar FP64, including the legacy fallback's exceptional semantics.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ref = @import("gdn_chunked_reference.zig");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;
const Chunked = cuda.gdn_chunked.Chunked;
const Kind = enum { output, state };
const Fault = enum { none, guard, tail, erase_nonfinite, narrow };
const Config = struct { kind: Kind, fault: Fault = .none };
const entries = [_]Config{
    .{ .kind = .output },                           .{ .kind = .state },
    .{ .kind = .output, .fault = .guard },          .{ .kind = .output, .fault = .tail },
    .{ .kind = .state, .fault = .erase_nonfinite }, .{ .kind = .output, .fault = .narrow },
    .{ .kind = .state, .fault = .narrow },
};
const served = [_]c.Shape{
    .{ .rows = 1, .n = 3, .k = 128 },  .{ .rows = 15, .n = 3, .k = 128 },
    .{ .rows = 16, .n = 3, .k = 128 }, .{ .rows = 17, .n = 3, .k = 128 },
    .{ .rows = 31, .n = 3, .k = 128 }, .{ .rows = 32, .n = 3, .k = 128 },
    .{ .rows = 33, .n = 3, .k = 128 }, .{ .rows = 65, .n = 3, .k = 128 },
};
const serial_symbol = "_ZN14tf_gdn_prefill12chain_kernelI13__nv_bfloat16S1_Li64ELi32EEEvPKT_S4_PKS1_PKfPKT0_S8_PfPS1_iii";
fn depth(s: c.Shape) usize {
    return s.rows * 512;
}
fn specification(config: Config) c.Spec {
    return .{
        .name = switch (config.fault) {
            .none => if (config.kind == .state) "cuda GDN chunk state" else "cuda GDN chunk output",
            .guard => "cuda GDN chunk CONTROL guard",
            .tail => "cuda GDN chunk CONTROL tail",
            .erase_nonfinite => "cuda GDN chunk CONTROL erased nonfinite",
            .narrow => if (config.kind == .state) "cuda GDN chunk CONTROL narrow state" else "cuda GDN chunk CONTROL narrow output",
        },
        .accum = .f32,
        .out = if (config.kind == .state) .f32 else .bf16,
        .depth = depth,
        .ops = 8,
        .nan = .{ .stated = "nonfinite heads use the original scalar recurrence; unaffected values retain scalar classes" },
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = &served,
    };
}

pub fn run(gpu: @import("check.zig").Gpu, image: []const u8, serial_image: []const u8) !void {
    return runTensor(gpu, image, serial_image, 0);
}
pub fn runTensor(gpu: @import("check.zig").Gpu, image: []const u8, serial_image: []const u8, tile: usize) !void {
    if (tile != 0 and tile != 16 and tile != 32) return error.UnsupportedGdnChunkTile;
    if (try gpu.ctx.capability() < 80) return error.UnsupportedGdnChunkDevice;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var cases: [entries.len]Case = undefined;
    var kernels: [entries.len]g.Kernel = undefined;
    for (&cases, &kernels, entries) |*item, *kernel, config| {
        item.* = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = config, .image = image, .serial_image = serial_image, .tile = tile };
        kernel.* = item.kernel();
    }
    defer for (&cases) |*item| item.deinit();
    var tally: g.Tally = .{};
    try g.runAll(gpu.gpa, &kernels, 0x6d646e, &tally);
    const limits = try checkLimits(gpu, image, tile);
    std.debug.print("SUMMARY GDN chunk: {d} kernels pass, {d} fail, {d} controls caught, {d} missed, {d} limit problems\n", .{
        tally.passed, tally.failed, tally.caught, tally.missed, limits,
    });
    if (tally.problems() + limits != 0) return error.TestFailed;
}
fn checkLimits(gpu: @import("check.zig").Gpu, image: []const u8, tile: usize) !usize {
    var module = try cuda.Module.load(gpu.d, image);
    defer module.unload();
    var problems: usize = 0;
    for (cuda.gdn_chunked.symbols, [_]i32{ 256, 128, 32, 256, 128 }) |name, launched| {
        const f = try module.function(name);
        const threads = try f.attribute(.max_threads_per_block);
        const shared = try f.attribute(.shared_size_bytes);
        const regs = try f.attribute(.num_regs);
        const local = try f.attribute(.local_size_bytes);
        const ok = threads >= launched and shared <= 99 * 1024 and regs > 0 and regs <= 255;
        problems += @intFromBool(!ok);
        std.debug.print("{s} GDN chunk limits {s}: threads={d} static={d} registers={d} local={d}\n", .{
            if (ok) "PASS" else "FAIL", name, threads, shared, regs, local,
        });
    }
    if (tile != 0) {
        const name: [:0]const u8 = if (tile == 16) "tf_gdn_chunk_tensor_16" else "tf_gdn_chunk_tensor_32";
        const f = try module.function(name);
        const shared = try f.attribute(.shared_size_bytes);
        const threads = try f.attribute(.max_threads_per_block);
        const regs = try f.attribute(.num_regs);
        const ok = threads >= 256 and shared == @as(i32, if (tile == 16) 26128 else 41360);
        problems += @intFromBool(!ok);
        std.debug.print("{s} GDN tensor limits {s}: threads={d} shared={d} registers={d}\n", .{ if (ok) "PASS" else "FAIL", name, threads, shared, regs });
    }
    return problems;
}
pub const Case = struct {
    rig: Rig,
    config: Config,
    image: []const u8,
    serial_image: []const u8,
    tile: usize = 0,
    arena: ?std.heap.ArenaAllocator = null,
    fixture: ?ref.Fixture = null,
    op: ?Chunked = null,
    legacy: ?cuda.Module = null,
    serial: ?cuda.Function = null,
    b: cuda.gdn_chunked.Buffers = undefined,
    old_state: u64 = 0,
    old_output: u64 = 0,
    shape: cuda.gdn_chunked.Shape = .{ .rows = 1, .hk = 1, .hv = 3 },
    fill: g.Fill = .random,
    want: []f64 = &.{},
    magnitude: []f64 = &.{},
    host: []u8 = &.{},

    fn clear(self: *Case) void {
        self.rig.s.synchronize() catch {};
        if (self.op) |*op| op.unload();
        self.op = null;
        if (self.legacy) |*module| module.unload();
        self.legacy = null;
        self.rig.reset();
        if (self.fixture) |*fixture| fixture.deinit();
        self.fixture = null;
        if (self.arena) |*arena| arena.deinit();
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
    fn output(ctx: *anyopaque) ![]const u8 {
        const self: *Case = @ptrCast(@alignCast(ctx));
        const ptr = if (self.config.kind == .state) self.b.state else self.b.output;
        @memcpy(self.host, try self.rig.bytes(ptr, self.host.len));
        return self.host;
    }
    const vt = blk: {
        var hooks = g.hooks(Case);
        hooks.output = output;
        hooks.release = release;
        break :blk hooks;
    };
    pub fn kernel(self: *Case) g.Kernel {
        return .{ .spec = specification(self.config), .ctx = self, .vt = &vt, .control = switch (self.config.fault) {
            .none => null,
            .guard => .repeat,
            .tail => .shapes,
            .erase_nonfinite => .nonfinite,
            .narrow => .sums,
        } };
    }
    pub fn count(self: *const Case) usize {
        return self.shape.hv * ref.width * (if (self.config.kind == .state) ref.width else self.shape.rows);
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, _: u64) !bool {
        if (shape.rows == 0 or shape.rows > 65 or shape.n != 3 or shape.k != 128) return false;
        self.clear();
        errdefer self.clear();
        self.shape = .{ .rows = shape.rows, .hk = 1, .hv = shape.n };
        self.fill = fill;
        self.arena = .init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.fixture = try ref.Fixture.init(self.rig.gpu.gpa, shape.rows, 1, shape.n);
        const x = &self.fixture.?.x;
        if (fill == .stress) {
            @memset(@constCast(x.q), ref.narrow(1));
            @memset(@constCast(x.k), ref.narrow(if (self.config.kind == .state) 0.25 else 0));
            @memset(@constCast(x.v), 0);
            @memset(@constCast(x.decay), 1);
            @memset(@constCast(x.beta), ref.narrow(if (self.config.kind == .state) 1 else 0));
            for (@constCast(x.state), 0..) |*s, i| {
                const key = i % ref.width;
                s.* = if (key == 0) 131200 else if (key < 64) 131072 else -131072;
            }
        } else if (fill == .nonfinite) {
            for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |bad, head|
                @constCast(x.v)[((head + 1) * x.hv + head) * ref.width + head] = ref.narrow(bad);
        }
        self.want = try a.alloc(f64, self.count());
        self.magnitude = try a.alloc(f64, self.count());
        const values = try a.alloc(f64, x.rows);
        const mags = try a.alloc(f64, x.rows);
        for (0..x.hv) |head| for (0..ref.width) |value| {
            var state: [ref.width]f64 = undefined;
            var state_mag: [ref.width]f64 = undefined;
            ref.serial(x.*, head, value, values, &state);
            ref.magnitude(x.*, head, value, mags, &state_mag);
            if (self.config.kind == .state) {
                @memcpy(self.want[(head * ref.width + value) * ref.width ..][0..ref.width], &state);
                @memcpy(self.magnitude[(head * ref.width + value) * ref.width ..][0..ref.width], &state_mag);
            } else for (0..x.rows) |row| {
                const i = (row * x.hv + head) * ref.width + value;
                self.want[i] = values[row];
                self.magnitude[i] = mags[row];
            }
        };
        const gpu_v = try a.dupe(u16, x.v);
        if (self.config.fault == .erase_nonfinite) for (gpu_v) |*v| {
            if (!std.math.isFinite(ref.bf16(v.*))) v.* = 0;
        };
        self.b = .{
            .q = try self.rig.input(std.mem.sliceAsBytes(x.q)),
            .k = try self.rig.input(std.mem.sliceAsBytes(x.k)),
            .v = try self.rig.input(std.mem.sliceAsBytes(gpu_v)),
            .decay = try self.rig.input(std.mem.sliceAsBytes(x.decay)),
            .beta = try self.rig.input(std.mem.sliceAsBytes(x.beta)),
            .initial = try self.rig.input(std.mem.sliceAsBytes(x.state)),
            .state = try self.rig.output(x.hv * ref.width * ref.width * 4),
            .output = try self.rig.output(x.rows * x.hv * ref.width * 2),
        };
        self.old_state = try self.rig.output(x.state.len * 4);
        self.old_output = try self.rig.output(x.rows * x.hv * ref.width * 2);
        self.op = if (self.tile == 0) try Chunked.load(self.rig.gpu.d, self.image, self.shape) else try Chunked.loadTensor(self.rig.gpu.d, self.image, self.shape, self.tile);
        const bytes = self.op.?.scratch.len;
        self.op.?.scratch.free();
        self.op.?.scratch = .{ .d = self.rig.gpu.d, .ptr = try self.rig.scratch(bytes), .len = bytes, .borrowed = true };
        self.legacy = try cuda.Module.load(self.rig.gpu.d, self.serial_image);
        self.serial = try self.legacy.?.function(serial_symbol);
        self.host = try a.alloc(u8, self.count() * c.size(specification(self.config).out));
        return true;
    }
    pub fn run(self: *Case) !void {
        var shape = self.shape;
        if (self.config.fault == .tail and shape.rows > 1) shape.rows -= 1;
        try self.op.?.run(self.rig.s, shape, self.b);
        var args: cuda.Args = .{};
        for ([_]u64{ self.b.q, self.b.k, self.b.v, self.b.decay, self.b.beta, self.b.initial, self.old_state, self.old_output }) |ptr| args.add(ptr);
        for ([_]usize{ self.shape.rows, self.shape.hk, self.shape.hv }) |v| args.add(@as(i32, @intCast(v)));
        try cuda.launch.launch(self.serial.?, .{ .grid = .{ .x = @intCast(self.shape.hv), .y = 2 }, .block = .{ .x = 128 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
        if (self.config.fault == .none) try self.checkFallback();
        if (self.config.fault == .guard) {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.b.output - 1, .len = 1, .borrowed = true };
            try b.upload(0, &.{0});
        }
        if (self.config.fault == .narrow) {
            const a = self.arena.?.allocator();
            const values = try a.alloc(f64, self.count());
            try self.simulate(.bf16, .sequential, values);
            const data = try a.alloc(u8, self.host.len);
            for (values, 0..) |value, i| if (self.config.kind == .state) {
                std.mem.writeInt(u32, data[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(value))), .little);
            } else std.mem.writeInt(u16, data[i * 2 ..][0..2], ref.narrow(@floatCast(value)), .little);
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = if (self.config.kind == .state) self.b.state else self.b.output, .len = data.len, .borrowed = true };
            try b.upload(0, data);
        }
    }
    pub fn reference(self: *Case, want: []f64, magnitude: []f64) !void {
        @memcpy(want, self.want);
        @memcpy(magnitude, self.magnitude);
    }
    fn checkFallback(self: *Case) !void {
        const a = self.arena.?.allocator();
        const layout = try cuda.gdn_chunked.Layout.init(self.shape);
        const flags = try a.dupe(u8, try self.rig.bytes(self.op.?.scratch.ptr + layout.flags, self.shape.hv * 4));
        for (0..self.shape.hv) |head| {
            const fallback = std.mem.readInt(u32, flags[head * 4 ..][0..4], .little) != 0;
            if (fallback != (self.fill == .nonfinite)) return error.UnexpectedGdnFallback;
            if (!fallback) continue;
            const state_at = head * ref.width * ref.width * 4;
            const state_bytes = ref.width * ref.width * 4;
            const old = try a.dupe(u8, try self.rig.bytes(self.old_state + state_at, state_bytes));
            const current = try self.rig.bytes(self.b.state + state_at, state_bytes);
            if (!equivalent(.f32, current, old)) return error.GdnFallbackStateChanged;
            for (0..self.shape.rows) |row| {
                const at = (row * self.shape.hv + head) * ref.width * 2;
                const expected = try a.dupe(u8, try self.rig.bytes(self.old_output + at, ref.width * 2));
                if (!equivalent(.bf16, try self.rig.bytes(self.b.output + at, ref.width * 2), expected))
                    return error.GdnFallbackOutputChanged;
            }
        }
    }
    pub fn simulate(self: *Case, accum: c.Accum, order: g.Order, out: []f64) !void {
        if (self.fill != .stress or self.shape.rows != 1) return error.BadGdnStress;
        const x = self.fixture.?.x;
        var terms: [ref.width]f64 = undefined;
        if (self.config.kind == .output) {
            for (&terms, x.state[0..ref.width]) |*p, s| p.* = s;
            @memset(out, core.contract.rows.sumIn(accum, order, &terms));
        } else {
            for (&terms, x.state[0..ref.width]) |*p, s| p.* = s * 0.25;
            const predicted = core.contract.rows.sumIn(accum, order, &terms);
            for (out, x.state) |*value, state| value.* = state - predicted * 0.25;
        }
    }
};

fn equivalent(kind: c.Elem, a: []const u8, b: []const u8) bool {
    for (0..a.len / c.size(kind)) |i| {
        const first = c.read(kind, a, i);
        const second = c.read(kind, b, i);
        if (std.math.isNan(first) and std.math.isNan(second)) continue;
        if (!std.mem.eql(u8, a[i * c.size(kind) ..][0..c.size(kind)], b[i * c.size(kind) ..][0..c.size(kind)])) return false;
    }
    return true;
}

test "GDN chunk gate fixtures retain true FP16/BF16 stress discrimination" {
    var terms: [128]f64 = undefined;
    for (&terms, 0..) |*t, i| t.* = if (i == 0) 131200 else if (i < 64) 131072 else -131072;
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(@as(f64, 128), core.contract.rows.sumIn(.f32, order, &terms));
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const narrowed = core.contract.rows.sumIn(acc, order, &terms);
            const limit = c.boundExact(specification(.{ .kind = .output }), 128);
            if (std.math.isFinite(narrowed) and @abs(narrowed - 128) < 100 * limit)
                std.debug.print("GDN stress {t}/{t}: sum={d}, want=128, allowance={d}\n", .{ order, acc, narrowed, limit });
            try std.testing.expect(!std.math.isFinite(narrowed) or @abs(narrowed - 128) >= 100 * limit);
        }
    }
    for (&terms) |*term| term.* *= 0.25;
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        try std.testing.expectEqual(@as(f64, 32), core.contract.rows.sumIn(.f32, order, &terms));
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            const got = 131200 - core.contract.rows.sumIn(acc, order, &terms) * 0.25;
            const want: f64 = 131192;
            const limit = c.boundExact(specification(.{ .kind = .state }), want);
            try std.testing.expect(!std.math.isFinite(got) or @abs(got - want) >= 100 * limit);
        }
    }
}
