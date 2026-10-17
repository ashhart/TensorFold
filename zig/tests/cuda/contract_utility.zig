//! Flash Next's sum and history utilities through the guarded numerical contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
pub const ref = @import("contract_utility_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Sum = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    bf16: bool,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    terms: []f32 = &.{},
    part: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Sum) void {
        self.rig.gpu.gpa.free(self.terms);
        self.rig.deinit();
    }

    pub fn kernel(self: *Sum) g.Kernel {
        return .{ .spec = ref.sumSpec(self.bf16), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Sum);

    pub fn prepare(self: *Sum, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.sumShape(shape)) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.terms);
        self.terms = &.{};
        self.shape = shape;
        self.terms = try a.alloc(f32, self.count() * shape.k);
        ref.sumFill(self.terms, shape, fill, seed);
        self.part = try self.rig.input(std.mem.sliceAsBytes(self.terms));
        self.out = try self.rig.output(self.count() * c.size(ref.sumSpec(self.bf16).out));
        return true;
    }

    pub fn run(self: *Sum) !void {
        var args: cuda.Args = .{};
        args.add(self.part);
        args.add(self.out);
        args.add(@as(c_int, if (self.bf16) 1 else 0));
        args.add(@as(c_longlong, @intCast(self.count())));
        args.add(@as(c_int, @intCast(self.shape.k)));
        try cuda.launch.launch(self.f, .{
            .grid = .{ .x = @intCast((self.count() + 255) / 256) },
            .block = .{ .x = 256 },
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Sum) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Sum, want: []f64, mag: []f64) !void {
        ref.sumReference(self.terms, self.shape, want, mag);
    }

    pub fn simulate(self: *Sum, acc: c.Accum, order: g.Order, out: []f64) !void {
        ref.sumSimulate(self.terms, self.shape, acc, order, out);
    }

    /// Retain the float64 error bound while additionally checking ordered f32 and single final RNE.
    pub fn check(self: *Sum, at: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a sum launch changed a guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.terms), try self.rig.inputBytes()))
            return "a sum launch changed its immutable slice inputs";
        return ref.sumCheck(self.rig.gpu.gpa, self.terms, self.shape, self.bf16, try self.rig.outputBytes(), at == .nonfinite);
    }
};

pub const Shift = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    padded: bool,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    stride: usize = 0,
    initial: []u16 = &.{},
    source: []u16 = &.{},
    want: []u16 = &.{},
    win: u64 = 0,
    rows: u64 = 0,

    pub fn deinit(self: *Shift) void {
        const a = self.rig.gpu.gpa;
        a.free(self.initial);
        a.free(self.source);
        a.free(self.want);
        self.rig.deinit();
    }

    pub fn kernel(self: *Shift) g.Kernel {
        return .{ .spec = ref.shiftSpec(self.padded), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Shift);

    pub fn prepare(self: *Shift, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.shiftShape(shape)) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.initial);
        a.free(self.source);
        a.free(self.want);
        self.initial = &.{};
        self.source = &.{};
        self.want = &.{};
        self.shape = shape;
        self.stride = shape.n + (if (self.padded) @as(usize, 6240) else 0);
        self.initial = try a.alloc(u16, self.count());
        self.source = try a.alloc(u16, shape.rows * self.stride);
        ref.copyFill(self.initial, self.source, shape, self.stride, fill, seed);
        self.want = try ref.copyReference(a, self.initial, self.source, shape, self.stride);
        // Mutable history is an output state, never classified as immutable input.
        self.win = try self.rig.output(self.count() * 2);
        self.rows = try self.rig.input(std.mem.sliceAsBytes(self.source));
        return true;
    }

    pub fn run(self: *Shift) !void {
        try self.rig.s.synchronize();
        const history: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.win, .len = self.count() * 2 };
        // Every repeat starts from the same history; five successive shifts would test another operation.
        try history.upload(0, std.mem.sliceAsBytes(self.initial));
        var args: cuda.Args = .{};
        args.add(self.win);
        args.add(self.rows);
        args.add(@as(c_longlong, @intCast(self.stride)));
        args.add(@as(c_int, @intCast(self.shape.rows)));
        args.add(@as(c_int, @intCast(self.shape.k)));
        args.add(@as(c_int, @intCast(self.shape.n)));
        try cuda.launch.launch(self.f, .{
            .grid = .{ .x = @intCast((self.shape.n + 255) / 256) },
            .block = .{ .x = 256 },
        }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Shift) usize {
        return self.shape.k * self.shape.n;
    }

    pub fn reference(self: *Shift, want: []f64, mag: []f64) !void {
        for (want, mag, self.want) |*w, *m, bits| {
            w.* = c.fromBf16(bits);
            m.* = 0;
        }
    }

    pub fn check(self: *Shift, at: g.Gate) !?[]const u8 {
        if (!try self.rig.intact()) return "a history launch changed a guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.source), try self.rig.inputBytes()))
            return "a history launch changed its immutable source rows";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.want), try self.rig.outputBytes()))
            return "history bytes differ from append-and-retain oracle";
        if (at == .nonfinite) {
            var nonfinite: usize = 0;
            for (self.want) |bits| if (!std.math.isFinite(c.fromBf16(bits))) {
                nonfinite += 1;
            };
            if (nonfinite == 0 or nonfinite == self.want.len)
                return "copy fixture did not retain both finite and nonfinite values";
        }
        return null;
    }
};
