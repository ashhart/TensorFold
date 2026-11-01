//! Nemotron's pocket (nemotron_ops.cu) under the contract: relu^2 up and down, block-row and tiled, two controls.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const width = nemotron.kernels.pocket_width;
const row_mul = 8; // a row's expert slots: the pocket adds into slot 6, the shared expert's first half
const row_add = 6;

pub const Kind = enum { in, out };

/// A planted fault: tf_pocket_in's relu by fmaxf (contract_controls.cu), or tf_pocket_out reading one neuron short.
pub const Fault = enum { none, fmax, short };

pub const Entry = struct { name: []const u8, kind: Kind, tiled: bool, f32_out: bool = true, fault: Fault = .none };

pub const entries = [_]Entry{
    .{ .name = "cuda nemotron pocket in tf_pocket_in (relu^2)", .kind = .in, .tiled = false },
    .{ .name = "cuda nemotron pocket in tf_pocket_in_rows (relu^2, eight neurons by eight rows a warp)", .kind = .in, .tiled = true },
    .{ .name = "cuda nemotron pocket out tf_pocket_out (fp32 out)", .kind = .out, .tiled = false },
    .{ .name = "cuda nemotron pocket out tf_pocket_out (bf16 out)", .kind = .out, .tiled = false, .f32_out = false },
    .{ .name = "cuda nemotron pocket out tf_pocket_out_rows (fp32 out, 16 rows a thread)", .kind = .out, .tiled = true },
    .{ .name = "cuda nemotron pocket out tf_pocket_out_rows (bf16 out, 16 rows a thread)", .kind = .out, .tiled = true, .f32_out = false },
    .{ .name = "control: pocket in with its relu by fmaxf, which drops a NaN", .kind = .in, .tiled = false, .fault = .fmax },
    .{ .name = "control: pocket out launched one neuron short", .kind = .out, .tiled = false, .fault = .short },
};

/// in: each lane's strided share of the d products, then the warp's five-step butterfly.
fn inDepth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 5;
}

/// out: the neurons in order, then the add into the slot's value.
fn outDepth(s: c.Shape) usize {
    return s.k + 1;
}

/// in's shapes are (rows, neurons in use, d), out's (rows, d, neurons); d is Lightning's 2,688 and Super's 4,096.
const served = [2][2][]const c.Shape{
    .{ &.{ .{ .rows = 1, .n = width, .k = 2688 }, .{ .rows = 200, .n = 37, .k = 4096 } }, &.{ .{ .rows = 256, .n = width, .k = 2688 }, .{ .rows = 600, .n = 29, .k = 4096 } } },
    .{ &.{ .{ .rows = 1, .n = 2688, .k = width }, .{ .rows = 63, .n = 4096, .k = 37 } }, &.{ .{ .rows = 64, .n = 2688, .k = width }, .{ .rows = 300, .n = 4096, .k = 29 } } },
};

pub fn spec(e: Entry) c.Spec {
    const shapes = served[@backingInt(e.kind)][@intFromBool(e.tiled)];
    return switch (e.kind) {
        // act = z * z, one fp32 rounding after the sum
        .in => .{ .name = e.name, .accum = .f32, .depth = inDepth, .ops = 1, .out = .f32, .tiles = .{ .{ .size = if (e.tiled) 8 else 1 }, .{ .size = if (e.tiled) width else 4 }, .{ .size = 32 } }, .served = shapes },
        .out => .{ .name = e.name, .accum = .f32, .depth = outDepth, .out = if (e.f32_out) .f32 else .bf16, .in_place = true, .tiles = .{ .{ .size = if (e.tiled) 16 else 1 }, .{ .size = 128 }, .{ .size = if (e.tiled) 8 else 16 } }, .served = shapes },
    };
}

/// The module's four kernels and the fmaxf control.
const Fns = struct { in: [2]cuda.Function, out: [2]cuda.Function, fmax: cuda.Function };

/// A pocket of `width` neurons, the first `used` in use and the rest NaN so a read of one shows.
pub const Pocket = struct {
    rig: rig_.Rig,
    entry: Entry,
    fns: *const Fns,
    shape: c.Shape = undefined,
    y: []u16 = &.{}, // in: [rows][d]
    w: []f32 = &.{}, // up or down: [width][d]
    act: []f32 = &.{}, // out: [rows][width]
    start: []f32 = &.{}, // out: the output's values before the launch, [rows * row_mul][d], bf16 values for bf16 out
    start_bytes: []u8 = &.{},
    yd: u64 = 0,
    wd: u64 = 0,
    actd: u64 = 0,
    used: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Pocket) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Pocket) void {
        const a = self.rig.gpu.gpa;
        a.free(self.y);
        a.free(self.w);
        a.free(self.act);
        a.free(self.start);
        a.free(self.start_bytes);
        self.y = &.{};
        self.w = &.{};
        self.act = &.{};
        self.start = &.{};
        self.start_bytes = &.{};
    }

    pub fn kernel(self: *Pocket) g.Kernel {
        const control: ?g.Gate = switch (self.entry.fault) {
            .none => null,
            .fmax => .nonfinite,
            .short => .shapes,
        };
        return .{ .spec = spec(self.entry), .ctx = self, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Pocket);

    /// Neurons in use and the length of each neuron's row.
    fn neurons(self: *const Pocket) usize {
        return if (self.entry.kind == .in) self.shape.n else self.shape.k;
    }

    fn d(self: *const Pocket) usize {
        return if (self.entry.kind == .in) self.shape.k else self.shape.n;
    }

    fn esize(self: *const Pocket) usize {
        return if (self.entry.f32_out) 4 else 2;
    }

    /// Random rows and weights; non-finite: NaN, +Inf and -Inf in the first rows' inputs; stress: mirrored integers.
    pub fn prepare(self: *Pocket, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const used = self.neurons();
        if (used > width) return false;
        const a = self.rig.gpu.gpa;
        const dd = self.d();
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.w = try a.alloc(f32, width * dd);
        for (self.w[0 .. used * dd]) |*v| v.* = (rng.float(f32) - 0.5) * 0.25;
        @memset(self.w[used * dd ..], std.math.nan(f32));
        switch (self.entry.kind) {
            .in => {
                self.y = try a.alloc(u16, rows * dd);
                for (self.y) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
            },
            .out => {
                self.act = try a.alloc(f32, rows * width);
                for (0..rows) |r| for (0..width) |j| {
                    self.act[r * width + j] = if (j < used) rng.float(f32) else std.math.nan(f32);
                };
                self.start = try a.alloc(f32, rows * row_mul * dd);
                for (self.start) |*v| v.* = rng.float(f32) * 2 - 1;
            },
        }
        switch (fill) {
            .random => {},
            .nonfinite => {
                // a row holding NaN, one holding +Inf, one holding both infinities (in: in y; out: in act)
                const k = if (self.entry.kind == .in) dd else used;
                const planted = [_]struct { usize, usize, f32 }{ .{ 0, k / 3, std.math.nan(f32) }, .{ 1, k / 2, std.math.inf(f32) }, .{ 2, 0, std.math.inf(f32) }, .{ 2, k - 1, -std.math.inf(f32) } };
                for (planted) |p| if (p[0] < rows) switch (self.entry.kind) {
                    .in => self.y[p[0] * dd + p[1]] = c.toBf16(p[2]),
                    .out => self.act[p[0] * width + p[1]] = p[2],
                };
            },
            .stress => self.stress(rng),
        }
        if (self.entry.kind == .out) {
            self.start_bytes = try a.alloc(u8, self.start.len * self.esize());
            if (self.entry.f32_out) {
                @memcpy(self.start_bytes, std.mem.sliceAsBytes(self.start));
            } else for (std.mem.bytesAsSlice(u16, self.start_bytes), self.start) |*h, *v| {
                h.* = c.toBf16(v.*);
                v.* = @floatCast(c.fromBf16(h.*));
            }
        }
        const in_use: i32 = @intCast(if (self.entry.fault == .short) used - 1 else used);
        self.used = try self.rig.input(std.mem.asBytes(&in_use));
        self.wd = try self.rig.input(std.mem.sliceAsBytes(self.w));
        switch (self.entry.kind) {
            .in => {
                self.yd = try self.rig.input(std.mem.sliceAsBytes(self.y));
                self.out = try self.rig.output(rows * width * 4);
            },
            .out => {
                self.actd = try self.rig.input(std.mem.sliceAsBytes(self.act));
                self.out = try self.rig.output(self.start_bytes.len);
            },
        }
        return true;
    }

    /// Gate 4: mirrored integer weights and inputs (out pairs j with j + 16), every partial sum an exact small integer.
    fn stress(self: *Pocket, rng: std.Random) void {
        const dd = self.d();
        const used = self.neurons();
        switch (self.entry.kind) {
            .in => {
                const h = dd / 2;
                for (0..used) |j| for (0..h) |i| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
                    self.w[j * dd + i] = v;
                    self.w[j * dd + h + i] = v;
                };
                for (0..self.shape.rows) |r| for (0..h) |i| {
                    const v: f32 = @floatFromInt(4 * (1 + 2 * rng.uintLessThan(u32, 64)));
                    self.y[r * dd + i] = c.toBf16(v);
                    self.y[r * dd + h + i] = c.toBf16(if (i < 8) 4 - v else -v);
                };
            },
            .out => {
                // neuron j + 16 mirrors j inside each 32: halves of 32 lanes would add each mirror to its own value
                for (0..used) |j| if (j % 32 < 16) for (0..dd) |i| {
                    const v: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 2));
                    self.w[j * dd + i] = v;
                    self.w[(j + 16) * dd + i] = v;
                };
                for (0..self.shape.rows) |r| for (0..used) |j| if (j % 32 < 16) {
                    const v: f32 = @floatFromInt(4 * (1 + 2 * rng.uintLessThan(u32, 1024)));
                    self.act[r * width + j] = v;
                    self.act[r * width + j + 16] = if (j < 8) 4 - v else -v;
                };
                @memset(self.start, 0);
            },
        }
    }

    /// pocketRows' launch of this form; out first restores the output it adds into.
    pub fn run(self: *Pocket) !void {
        const rows = self.shape.rows;
        const dd = self.d();
        var a: cuda.Args = .{};
        switch (self.entry.kind) {
            .in => {
                try self.rig.zero(self.out, rows * width * 4); // act past the neurons in use stays 0
                a.add(self.yd);
                a.add(@as(i32, @intCast(dd)));
                a.add(self.wd);
                a.add(self.used);
                a.add(self.out);
                a.add(@as(i32, @intCast(rows)));
                const f = if (self.entry.fault == .fmax) self.fns.fmax else self.fns.in[@intFromBool(self.entry.tiled)];
                const grid: cuda.launch.Dim3 = if (self.entry.tiled) .{ .x = (width + 63) / 64, .y = @intCast((rows + 7) / 8) } else .{ .x = (width + 3) / 4, .y = @intCast(rows) };
                try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = if (self.entry.tiled) 256 else 128 } }, self.rig.s, &a);
            },
            .out => {
                try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = self.out, .len = self.start_bytes.len }).upload(0, self.start_bytes);
                a.add(self.actd);
                a.add(self.wd);
                a.add(self.used);
                a.add(@as(i32, @intCast(dd)));
                a.add(self.out);
                a.add(@as(i32, @intFromBool(self.entry.f32_out)));
                for ([_]usize{ rows, row_mul, row_add }) |v| a.add(@as(i32, @intCast(v)));
                const grid: cuda.launch.Dim3 = .{ .x = @intCast((dd + 127) / 128), .y = @intCast(if (self.entry.tiled) (rows + 15) / 16 else rows) };
                try cuda.launch.launch(self.fns.out[@intFromBool(self.entry.tiled)], .{ .grid = grid, .block = .{ .x = 128 } }, self.rig.s, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Pocket) usize {
        return switch (self.entry.kind) {
            .in => self.shape.rows * width,
            .out => self.start.len,
        };
    }

    /// The products output o sums, in order; null where the launch leaves it as it was.
    fn terms(self: *const Pocket, o: usize, t: []f64) ?[]f64 {
        const dd = self.d();
        const used = self.neurons();
        switch (self.entry.kind) {
            .in => {
                const r = o / width;
                const j = o % width;
                if (j >= used) return null;
                for (t[0..dd], 0..) |*v, i| v.* = @as(f64, self.w[j * dd + i]) * c.fromBf16(self.y[r * dd + i]);
                return t[0..dd];
            },
            .out => {
                const slot = o / dd;
                if (slot % row_mul != row_add or slot / row_mul >= self.shape.rows) return null;
                const r = slot / row_mul;
                const col = o % dd;
                for (t[0..used], 0..) |*v, j| v.* = @as(f64, self.act[r * width + j]) * self.w[j * dd + col];
                return t[0..used];
            },
        }
    }

    /// in: relu^2, as float64 keeps a NaN; out: the slot's value plus the sum.
    fn epilogue(self: *const Pocket, o: usize, sum: f64) f64 {
        return switch (self.entry.kind) {
            .in => blk: {
                const u = if (std.math.isNan(sum)) sum else @max(sum, 0);
                break :blk u * u;
            },
            .out => self.start[o] + sum,
        };
    }

    /// What the launch leaves where it writes nothing: act's 0, the slot's own value.
    fn untouched(self: *const Pocket, o: usize) f64 {
        return if (self.entry.kind == .in) 0 else self.start[o];
    }

    pub fn reference(self: *Pocket, want: []f64, mag: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, @max(self.d(), width));
        defer self.rig.gpu.gpa.free(t);
        const e = c.gamma(@floatFromInt(spec(self.entry).depth(self.shape)), c.unit(.f32));
        for (want, mag, 0..) |*w, *m, o| {
            const ts = self.terms(o, t) orelse {
                w.* = self.untouched(o);
                m.* = 0;
                continue;
            };
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (ts) |v| {
                sum += v;
                abs += @abs(v);
            }
            w.* = self.epilogue(o, sum);
            m.* = switch (self.entry.kind) {
                // the sum's error e |terms| reaches relu^2 as 2 (|u| + e |terms|), bounded near 0 too
                .in => 2 * (@sqrt(w.*) + e * abs) * abs,
                .out => @abs(self.start[o]) + abs,
            };
        }
    }

    pub fn simulate(self: *Pocket, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.rig.gpu.gpa.alloc(f64, @max(self.d(), width));
        defer self.rig.gpu.gpa.free(t);
        for (out, 0..) |*v, o| {
            const ts = self.terms(o, t) orelse {
                v.* = self.untouched(o);
                continue;
            };
            v.* = self.epilogue(o, contract.rows.sumIn(acc, order, ts));
        }
    }
};

/// Every registration, on the heap: the gates hold pointers to them.
pub const Set = struct {
    gpa: std.mem.Allocator,
    mod: cuda.Module,
    ctl: cuda.Module,
    fns: Fns = undefined,
    items: [entries.len]Pocket = undefined,

    const controls = struct {
        pub const bytes align(16) = @embedFile("fatbin_contract_controls").*;
    };

    /// The set, its kernels appended to `list` (each control after the kernels it breaks).
    pub fn init(gpu: check.Gpu, s: cuda.Stream, a: std.mem.Allocator, list: *std.ArrayList(g.Kernel)) !*Set {
        const set = try gpu.gpa.create(Set);
        errdefer gpu.gpa.destroy(set);
        var mod = try cuda.Module.load(gpu.d, cuda.kernels.nemotron_ops);
        errdefer mod.unload();
        var ctl = try cuda.Module.load(gpu.d, &controls.bytes);
        errdefer ctl.unload();
        set.* = .{ .gpa = gpu.gpa, .mod = mod, .ctl = ctl };
        set.fns = .{
            .in = .{ try mod.function("tf_pocket_in"), try mod.function("tf_pocket_in_rows") },
            .out = .{ try mod.function("tf_pocket_out"), try mod.function("tf_pocket_out_rows") },
            .fmax = try ctl.function("tf_ctl_pocket_in_fmax"),
        };
        for (&set.items, entries) |*x, e| x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .entry = e, .fns = &set.fns };
        for (&set.items) |*x| try list.append(a, x.kernel());
        return set;
    }

    pub fn deinit(set: *Set) void {
        for (&set.items) |*x| x.deinit();
        set.ctl.unload();
        set.mod.unload();
        set.gpa.destroy(set);
    }
};

/// The pocket's gates 1-4 and its controls alone.
pub fn run(gpu: check.Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var list: std.ArrayList(g.Kernel) = .empty;
    defer list.deinit(a);
    const set = try Set.init(gpu, s, a, &list);
    defer set.deinit();
    var t: g.Tally = .{};
    try g.runAll(a, list.items, 0x706f636b, &t);
    std.debug.print("SUMMARY pocket: {d} kernels, {d} pass gates 1-4, {d} fail, {d} known findings; {d} controls: {d} red as planted, {d} missed\n", .{ t.kernels, t.passed, t.failed, t.known, t.controls, t.caught, t.missed });
    try check.expect(t.problems() == 0, "pocket-contract: {d} gate failures, {d} controls missed", .{ t.failed, t.missed });
}

test "every control is named one, and only the controls plant a fault" {
    for (entries) |e| try std.testing.expectEqual(e.fault != .none, std.mem.startsWith(u8, e.name, "control: "));
}
