//! Nemotron's Mamba scan under the contract on CUDA: a window's kept rows replayed, then its own, 32 steps a stage.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const ref = nemotron.glue_ref;

const dh = 64; // value rows a head
const groups = 8;
const ds = 128; // state a value row
const rmax = 32; // a window plane's rows
const lo: f32 = 0;
const hi = std.math.inf(f32);
const rule = "glue_ref.scan's bytes (y, the window's dt, the state after the kept rows), NaN wherever it gives NaN";

fn noSums(_: c.Shape) usize {
    return 0;
}

/// Super's 128 heads: rows window rows after k kept ones, at main's widest window and either side of each stage.
pub fn spec() c.Spec {
    const served = &[_]c.Shape{ .{ .rows = 16, .n = 128, .k = 15 }, .{ .rows = 1, .n = 128, .k = 0 }, .{ .rows = 1, .n = 128, .k = 31 }, .{ .rows = 2, .n = 128, .k = 31 }, .{ .rows = 32, .n = 128, .k = 0 }, .{ .rows = 32, .n = 128, .k = 31 } };
    return .{ .name = "cuda nemotron mamba scan tf_nemo_scan", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = rule }, .tiles = .{ .{ .size = 32 }, .{}, .{ .size = 32 } }, .served = served };
}

/// shape.rows window rows after shape.k kept rows of the other parity, over shape.n heads; y, state and dt out.
pub const Scan = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    host: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    m: ref.Shape = undefined,
    parity: usize = 0,
    p: []u16 = &.{},
    xc: []u16 = &.{},
    st: []f32 = &.{},
    dt: []f32 = &.{},
    al: []f32 = &.{},
    dsk: []f32 = &.{},
    dtb: []f32 = &.{},
    in: [8]u64 = undefined, // p, xc, a, dsk, dtb, meta, then the state and dt each run starts from
    y: u64 = 0,
    st_out: u64 = 0,
    dt_out: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Scan) void {
        self.host.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Scan) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Scan);

    fn bf16s(a: std.mem.Allocator, rng: std.Random, n: usize, scale: f32) ![]u16 {
        const out = try a.alloc(u16, n);
        for (out) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * scale);
        return out;
    }

    fn f32s(a: std.mem.Allocator, rng: std.Random, n: usize, from: f32, to: f32) ![]f32 {
        const out = try a.alloc(f32, n);
        for (out) |*v| v.* = from + rng.float(f32) * (to - from);
        return out;
    }

    /// Random rows as the engine's ranges give them; the seed picks the parity; non-finite: plant's four values.
    pub fn prepare(self: *Scan, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > rmax or shape.k >= rmax or shape.n == 0 or shape.n % groups != 0) return false;
        self.rig.reset();
        _ = self.host.reset(.retain_capacity);
        const a = self.host.allocator();
        const xd = shape.n * dh;
        const cd = xd + 2 * groups * ds;
        self.shape = shape;
        self.m = .{ .proj = xd + cd + shape.n, .xd = xd, .cd = cd, .heads = shape.n, .dh = dh, .groups = groups, .rmax = rmax };
        self.parity = seed % 2;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.p = try bf16s(a, rng, shape.rows * self.m.proj, 2);
        self.xc = try bf16s(a, rng, 2 * rmax * cd, 1);
        self.st = try f32s(a, rng, shape.n * dh * ds, -0.2, 0.2);
        self.dt = try f32s(a, rng, 2 * rmax * shape.n, 0.001, 0.1);
        self.al = try f32s(a, rng, shape.n, -2, -0.05);
        self.dsk = try f32s(a, rng, shape.n, 0.5, 1.5);
        self.dtb = try f32s(a, rng, shape.n, -4, 1);
        if (fill == .nonfinite) self.plant();
        const r = &self.rig;
        const meta = [4]i32{ 100, @intCast(self.parity), @intCast(shape.k), 0 };
        const ins = [_][]const u8{ std.mem.sliceAsBytes(self.p), std.mem.sliceAsBytes(self.xc), std.mem.sliceAsBytes(self.al), std.mem.sliceAsBytes(self.dsk), std.mem.sliceAsBytes(self.dtb), std.mem.sliceAsBytes(&meta), std.mem.sliceAsBytes(self.st), std.mem.sliceAsBytes(self.dt) };
        for (ins, &self.in) |bytes, *at| at.* = try r.input(bytes);
        self.y = try r.output(shape.rows * xd * 2);
        self.st_out = try r.output(self.st.len * 4);
        self.dt_out = try r.output(self.dt.len * 4);
        return true;
    }

    /// NaN in window row 0's x (head 0), +Inf in row 1's gate (head 1) and row 2's dt (head 2), -Inf in head 3's state.
    fn plant(self: *Scan) void {
        const m = self.m;
        const window = self.parity * rmax * m.cd;
        self.xc[window] = c.toBf16(std.math.nan(f32));
        self.p[m.proj + dh] = c.toBf16(std.math.inf(f32));
        self.p[2 * m.proj + m.xd + m.cd + 2] = c.toBf16(std.math.inf(f32));
        self.st[3 * dh * ds] = -std.math.inf(f32);
    }

    /// The state and dt planes back to the prepared ones, then one launch: a run never reads the last run's writes.
    pub fn run(self: *Scan) !void {
        const m = self.m;
        for ([_]u64{ self.st_out, self.dt_out }, self.in[6..], [_]usize{ self.st.len * 4, self.dt.len * 4 }) |to, from, n| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = to, .len = n };
            try b.copyFrom(0, from, n, self.rig.s.handle);
        }
        var a: cuda.Args = .{};
        for ([_]u64{ self.in[0], self.in[1], self.dt_out, self.st_out, self.in[2], self.in[3], self.in[4], self.in[5], self.y }) |p| a.add(p);
        a.add(@as(c_int, @intCast(self.shape.rows)));
        a.add(lo);
        a.add(hi);
        for ([_]usize{ m.proj, m.xd, m.cd, m.xd + m.cd, m.heads, m.dh, m.groups, m.rmax }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(m.heads), .y = dh / 32, .z = 1 }, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Scan) usize {
        return self.shape.rows * self.m.xd * 2 + (self.st.len + self.dt.len) * 4;
    }

    pub fn reference(_: *Scan, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// The first element of `got` whose bits are not `want`'s, where not both are NaN (bf16 words or f32).
    fn differs(comptime T: type, got: []const u8, want: []const T) ?usize {
        for (want, 0..) |w, i| {
            const v = std.mem.bytesToValue(T, got[i * @sizeOf(T) ..][0..@sizeOf(T)]);
            if (!std.mem.eql(u8, std.mem.asBytes(&v), std.mem.asBytes(&w)) and !(isNan(v) and isNan(w))) return i;
        }
        return null;
    }

    fn isNan(v: anytype) bool {
        return std.math.isNan(if (@TypeOf(v) == u16) c.fromBf16(v) else v);
    }

    /// The host twin's outputs on the same inputs; at gate 1 some y must be non-finite, or the plants prove nothing.
    pub fn check(self: *Scan, at: g.Gate) !?[]const u8 {
        const a = self.host.allocator();
        const got = try self.rig.outputBytes();
        const y = try a.alloc(u16, self.shape.rows * self.m.xd);
        const st = try a.dupe(f32, self.st);
        const dt = try a.dupe(f32, self.dt);
        ref.scan(self.p, self.xc, dt, st, self.al, self.dsk, self.dtb, y, self.parity, self.shape.k, self.shape.rows, lo, hi, self.m);
        const y_bytes = y.len * 2;
        const st_bytes = st.len * 4;
        if (differs(u16, got, y)) |i| return try std.fmt.bufPrint(&self.msg, "y {d} of {d} differs from the host twin's", .{ i, y.len });
        if (differs(f32, got[y_bytes..], st)) |i| return try std.fmt.bufPrint(&self.msg, "state {d} of {d} differs from the host twin's", .{ i, st.len });
        if (differs(f32, got[y_bytes + st_bytes ..], dt)) |i| return try std.fmt.bufPrint(&self.msg, "dt {d} of {d} differs from the host twin's", .{ i, dt.len });
        if (at != .nonfinite) return null;
        for (y) |v| if (c.classOf(c.fromBf16(v)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no y is non-finite: the planted values prove nothing", .{});
    }
};
