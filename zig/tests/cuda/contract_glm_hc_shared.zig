//! Decode HC partials: the shared-fn replacement preserves all 32 slots, including unwritten padding.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
pub const rows = [_]usize{ 1, 3, 4, 8, 16, 33, 63 };
fn depth(_: c.Shape) usize {
    return 32;
}
pub fn spec() c.Spec {
    const served = comptime blk: {
        var out: [rows.len]c.Shape = undefined;
        for (rows, &out) |n, *s| s.* = .{ .rows = n, .n = 24, .k = 16384 };
        break :blk out;
    };
    return .{ .name = "cuda GLM shared HC tf_glm_hc_partial8_shared", .accum = .f32, .depth = depth, .out = .raw, .replaces = "tf_glm_hc_partial8", .nan = .{ .stated = "Every part byte equals partial8 on the same BF16 inputs, including NaN payloads, infinities and padding" }, .served = &served, .sums = .{ .why = "Exact replacement bytes; accumulation stays the original FP32 instruction order" } };
}

/// Both operands include signs, zeros and cancellation; the non-finite fixture plants values in x and fn.
pub fn fillInputs(x: []u16, w: []u16, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (x) |*h| h.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 2048))) - 1024)) / 256.0);
    for (w, 0..) |*h, i| h.* = c.toBf16(if (fill == .stress) (if (i % 2 == 0) @as(f32, 1) else -1) else rng.float(f32) * 2 - 1);
    const special = [_]u16{ 0x0000, 0x8000, 0x0080, 0x8080, 0x3f80, 0xbf80 };
    for (special, 0..) |h, i| {
        x[i * 128] = h;
        w[i * 128] = h;
    }
    if (fill == .nonfinite) {
        const nonfinite = [_]u16{ 0x7fc1, 0xffc2, 0x7f80, 0xff80 };
        for (nonfinite, 0..) |h, i| {
            x[x.len - 1 - i * 128] = h;
            w[(i * 5 + 1) * 16384 + 17 + i * 128] = h;
        }
    }
}
pub fn sameBytes(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub const Partial = struct {
    rig: rig_.Rig,
    fast: cuda.Function,
    old: cuda.Function,
    shape: c.Shape = undefined,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    corrupt: bool = false,
    nonfinite: bool = false,
    pub fn deinit(self: *Partial) void {
        self.rig.deinit();
    }
    pub fn kernel(self: *Partial) g.Kernel {
        var s = spec();
        if (self.nonfinite) s.name = "cuda GLM shared HC nonfinite tf_glm_hc_partial8_shared";
        if (self.corrupt) s.name = "control: GLM shared HC part byte changed";
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.corrupt) .shapes else null };
    }
    const vt = g.hooks(Partial);
    pub fn count(self: *Partial) usize {
        return self.shape.rows * 16 * 32 * 4;
    }
    pub fn prepare(self: *Partial, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows >= 64 or shape.n != 24 or shape.k != 16384) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        const x = try a.alloc(u16, shape.rows * shape.k);
        defer a.free(x);
        const w = try a.alloc(u16, 24 * shape.k);
        defer a.free(w);
        fillInputs(x, w, if (self.nonfinite) .nonfinite else fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(self.count());
        self.base = try self.rig.scratch(self.count());
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.count() };
        try b.fill8(0xff, self.rig.s.handle);
        return true;
    }
    fn launch(self: *Partial, shared: bool) !void {
        var a: cuda.Args = .{};
        inline for (.{ self.x, self.w, if (shared) self.out else self.base, @as(i64, 16384), @as(c_int, 16) }) |v| a.add(v);
        if (shared) a.add(@as(i64, @intCast(self.shape.rows)));
        try cuda.launch.launch(if (shared) self.fast else self.old, .{ .grid = .{ .x = @intCast(if (shared) (self.shape.rows + 3) / 4 else self.shape.rows), .y = 16 }, .block = .{ .x = 1024 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }
    pub fn run(self: *Partial) !void {
        try self.launch(true);
        if (self.corrupt) {
            const bytes = try self.rig.outputBytes();
            var byte = [_]u8{bytes[0] ^ 1};
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.count() };
            try b.upload(0, &byte);
        }
    }
    pub fn replaced(self: *Partial) ![]const u8 {
        try self.launch(false);
        return self.rig.bytes(self.base, self.count());
    }
    pub fn reference(_: *Partial, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
    pub fn check(self: *Partial, _: g.Gate) !?[]const u8 {
        const got = try self.rig.gpu.gpa.dupe(u8, try self.rig.outputBytes());
        defer self.rig.gpu.gpa.free(got);
        return if (sameBytes(got, try self.replaced())) null else "part bytes differ from tf_glm_hc_partial8";
    }
};

pub const Pair = struct {
    cells: [3]Partial,
    pub fn init(rig: rig_.Rig, fast: cuda.Function, old: cuda.Function) Pair {
        return .{ .cells = .{ .{ .rig = rig, .fast = fast, .old = old }, .{ .rig = rig, .fast = fast, .old = old, .nonfinite = true }, .{ .rig = rig, .fast = fast, .old = old, .corrupt = true } } };
    }
    pub fn deinit(self: *Pair) void {
        for (&self.cells) |*x| x.deinit();
    }
    pub fn append(self: *Pair, a: std.mem.Allocator, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&self.cells) |*x| try kernels.append(a, x.kernel());
    }
};
