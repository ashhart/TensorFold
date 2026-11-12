//! A rolled-back stream's windows in one launch (fn_late_roll) against a copy and fn_shift_window a window.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const nk = 16;
const nv = 48;
const dk = 128;
const dv = 128;
const taps = 4;
const conv_c = 2 * nk * dk + nv * dv;
const proj_w = conv_c + nv * dv + 2 * nv;
const hist_len = (taps - 1) * conv_c * 2;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// rows: the rollback's jobs (a stream's DeltaNet layers' windows); n and k fixed by the model.
pub fn spec() c.Spec {
    return .{ .name = "cuda flashnext rollback windows fn_late_roll", .accum = .none, .depth = noSums, .out = .raw, .nan = .{ .stated = "every window bit-equal to its saved copy then fn_shift_window by the kept rows" }, .replaces = "a copy and fn_shift_window a window", .served = &.{ .{ .rows = 2, .n = conv_c, .k = taps - 1 }, .{ .rows = 37, .n = conv_c, .k = taps - 1 } } };
}

pub const Fns = struct { shift: cuda.Function, roll: cuda.Function };

pub const Roll = struct {
    rig: rig_.Rig,
    f: *const Fns,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    keep: usize = 0,
    ins: [2]u64 = @splat(0),
    region: u64 = 0,
    base: u64 = 0,
    len: usize = 0,
    fresh: ?[]u8 = null,
    seen: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Roll) void {
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.seen.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    const vt = g.hooks(Roll);

    pub fn kernel(self: *Roll) g.Kernel {
        return .{ .spec = spec(), .ctx = self, .vt = &vt };
    }

    /// Jobs of conv windows (taps - 1 rows of conv_c), each `keep` rows on from its saved copy over the pass's rows.
    pub fn prepare(self: *Roll, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (self.fresh) |f| self.rig.gpu.gpa.free(f);
        self.fresh = null;
        self.shape = shape;
        if (shape.rows < 2 or shape.rows > 40 or shape.n != conv_c or shape.k != taps - 1) return false;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x726f6c6c);
        const r = prng.random();
        self.keep = 1 + r.uintLessThan(usize, 32);
        const rows_n = self.keep + taps;
        const hist = try a.alloc(u16, shape.rows * hist_len / 2);
        defer a.free(hist);
        for (hist) |*x| x.* = c.toBf16(r.float(f32) * 2 - 1);
        const rows = try a.alloc(u16, shape.rows * rows_n * proj_w);
        defer a.free(rows);
        for (rows) |*x| x.* = c.toBf16(r.float(f32) * 2 - 1);
        if (fill == .nonfinite) hist[5] = c.toBf16(std.math.inf(f32));
        self.ins[0] = try self.rig.input(std.mem.sliceAsBytes(hist));
        self.ins[1] = try self.rig.input(std.mem.sliceAsBytes(rows));
        self.len = shape.rows * hist_len;
        const fresh = try a.alloc(u8, self.len);
        self.fresh = fresh;
        @memset(fresh, 0xff);
        self.region = try self.rig.output(self.len);
        self.base = try self.rig.scratch(self.len);
        return true;
    }

    fn cint(v: usize) c_int {
        return @intCast(v);
    }

    /// The rollback's launch as cuda_mtp_roll.lateRoll makes it.
    fn ours(self: *Roll, region: u64) !void {
        var t: @import("flashnext").batch_state.RollJobs = .{ .n = @intCast(self.shape.rows) };
        for (t.j[0..self.shape.rows], 0..) |*j, i| j.* = .{ .win = region + i * hist_len, .hist = self.ins[0] + i * hist_len, .rows = self.ins[1] + i * (self.keep + taps) * proj_w * 2, .stride = proj_w, .taps = taps - 1, .c = conv_c };
        var a: cuda.Args = .{};
        a.add(t);
        a.add(cint(self.keep));
        try cuda.launch.launch(self.f.roll, .{ .grid = .{ .x = @intCast((conv_c + 255) / 256), .y = @intCast(self.shape.rows) }, .block = .{ .x = 256 } }, self.rig.s, &a);
    }

    /// Each window's saved copy, then fn_shift_window by the kept rows: the launches the one launch replaces.
    fn theirs(self: *Roll, region: u64) !void {
        for (0..self.shape.rows) |i| {
            const win = region + i * hist_len;
            try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = win, .len = hist_len }).copyFrom(0, self.ins[0] + i * hist_len, hist_len, self.rig.s.handle);
            var a: cuda.Args = .{};
            a.add(win);
            a.add(self.ins[1] + i * (self.keep + taps) * proj_w * 2);
            a.add(@as(c_longlong, proj_w));
            a.add(cint(self.keep));
            a.add(cint(taps - 1));
            a.add(cint(conv_c));
            try cuda.launch.launch(self.f.shift, .{ .grid = .{ .x = @intCast((conv_c + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        }
    }

    fn restore(self: *Roll, at: u64) !void {
        try (cuda.DeviceBuffer{ .d = self.rig.gpu.d, .ptr = at, .len = self.len }).upload(0, self.fresh.?);
    }

    pub fn run(self: *Roll) !void {
        try self.restore(self.region);
        try self.ours(self.region);
        try self.rig.s.synchronize();
    }

    fn baseline(self: *Roll) ![]const u8 {
        try self.restore(self.base);
        try self.theirs(self.base);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.len);
    }

    pub fn replaced(self: *Roll) ![]const u8 {
        return self.baseline();
    }

    /// Gate 1's stated rule: every window on the planted inputs equals the copy and shift's bytes.
    pub fn check(self: *Roll, gate: g.Gate) !?[]const u8 {
        _ = gate;
        self.seen.clearRetainingCapacity();
        try self.seen.appendSlice(self.rig.gpu.gpa, try self.rig.outputBytes());
        const want = try self.baseline();
        if (std.mem.eql(u8, self.seen.items, want)) return null;
        return try std.fmt.bufPrint(&self.msg, "bytes differ from the copy and shift's from byte {d}", .{std.mem.indexOfDiff(u8, self.seen.items, want).?});
    }

    pub fn count(self: *Roll) usize {
        return self.len;
    }

    pub fn reference(_: *Roll, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }
};
