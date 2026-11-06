//! rms_tiles.cuh under the contract: tf_rms_rows_sums' replacement at Flash Next's MTP row, its y and its group sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

const width = cuda.norms.tiles_width;
const group = 32;
const groups = width / group;

/// The normed bf16 row, or its fp32 sums of each 32 stored values.
pub const View = enum { y, xs };

/// Every 1,024th square a chain, then 1,024 chains' tree.
fn treeDepth(_: c.Shape) usize {
    return width / 1024 + 12;
}

fn groupDepth(_: c.Shape) usize {
    return group;
}

/// Rows 1 to 16, each through rmsSums, so the device's table picks the tiles.
const served = blk: {
    var s: [16]c.Shape = undefined;
    for (&s, 1..) |*x, r| x.* = .{ .rows = r, .n = width, .k = width };
    break :blk s;
};

const y_why = "a bf16 output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";

/// Gate 4's weights for the sums: ones rows store y = w, so each group adds 2^20, fifteen ones, -2^20, fifteen ones.
fn stressWeight(i: usize) f32 {
    return switch (i % group) {
        0 => 0x1p20,
        16 => -0x1p20,
        else => 1,
    };
}

pub fn spec(view: View) c.Spec {
    return switch (view) {
        .y => .{ .name = "cuda RMS norm tf_rms_rows_sums_tiles (bf16 y)", .accum = .f32, .depth = treeDepth, .ops = 8, .out = .bf16, .replaces = "tf_rms_rows_sums", .served = &served, .sums = .{ .bf16 = 0, .why = y_why } },
        .xs => .{ .name = "cuda RMS norm tf_rms_rows_sums_tiles (fp32 group sums)", .accum = .f32, .depth = groupDepth, .out = .f32, .replaces = "tf_rms_rows_sums", .served = &served },
    };
}

pub const Tiles = struct {
    rig: rig_.Rig,
    view: View,
    norms: *const cuda.norms.Norms,
    r: ?rows.Rows = null,
    terms: std.ArrayList(f64) = .empty,
    x: u64 = 0,
    w: u64 = 0,
    y: u64 = 0,
    xs: u64 = 0,
    base_y: u64 = 0,
    base_xs: u64 = 0,

    pub fn deinit(self: *Tiles) void {
        if (self.r) |r| r.deinit();
        self.terms.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Tiles) g.Kernel {
        return .{ .spec = spec(self.view), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Tiles);

    /// The current kernel: a copy of the norms with no table row.
    fn current(self: *Tiles, y: u64, xs: u64) !void {
        var base = self.norms.*;
        base.sm = 0;
        try base.rmsSums(self.rig.s, self.x, width, self.w, y, xs, self.r.?.shape.rows, width, group, self.r.?.eps);
    }

    pub fn prepare(self: *Tiles, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != width or shape.n != width) return false;
        if (self.r) |r| r.deinit();
        self.r = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const r = try rows.Rows.init(a, .rms, shape);
        self.r = r;
        r.fill(fill, seed);
        const wf = try a.alloc(f32, r.w.len);
        defer a.free(wf);
        for (wf, r.w) |*o, b| o.* = c.fromBf16(b);
        if (self.view == .xs and fill == .stress) { // sums exact in fp32 in any order, far off in bf16 and fp16
            @memset(r.x, 0x3f80);
            for (wf, 0..) |*o, i| o.* = stressWeight(i);
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(r.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(wf));
        const ny = shape.rows * width * 2;
        const nxs = shape.rows * groups * 4;
        self.y = if (self.view == .y) try self.rig.output(ny) else try self.rig.scratch(ny);
        self.xs = if (self.view == .xs) try self.rig.output(nxs) else try self.rig.scratch(nxs);
        self.base_y = try self.rig.scratch(ny);
        self.base_xs = try self.rig.scratch(nxs);
        if (self.view == .xs) { // the stored y the sums add, from the current kernel; the y view proves the bytes equal
            try self.current(self.base_y, self.base_xs);
            const stored = try self.rig.bytes(self.base_y, ny);
            try self.terms.resize(a, shape.rows * width);
            for (self.terms.items, std.mem.bytesAsSlice(u16, stored)) |*t, b| t.* = c.fromBf16(b);
        }
        return true;
    }

    pub fn run(self: *Tiles) !void {
        const r = self.r.?;
        try self.norms.rmsSums(self.rig.s, self.x, width, self.w, self.y, self.xs, r.shape.rows, width, group, r.eps);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Tiles) ![]const u8 {
        try self.current(self.base_y, self.base_xs);
        const n = self.r.?.shape.rows;
        return switch (self.view) {
            .y => self.rig.bytes(self.base_y, n * width * 2),
            .xs => self.rig.bytes(self.base_xs, n * groups * 4),
        };
    }

    pub fn count(self: *Tiles) usize {
        const n = self.r.?.shape.rows;
        return if (self.view == .y) n * width else n * groups;
    }

    pub fn reference(self: *Tiles, want: []f64, mag: []f64) !void {
        if (self.view == .y) return self.r.?.reference(want, mag);
        for (want, mag, 0..) |*w, *m, j| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (self.terms.items[j * group ..][0..group]) |t| {
                sum += t;
                abs += @abs(t);
            }
            w.* = sum;
            m.* = abs;
        }
    }

    pub fn simulate(self: *Tiles, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.view == .y) return self.r.?.simulate(acc, order, out);
        for (out, 0..) |*o, j| o.* = rows.sumIn(acc, order, self.terms.items[j * group ..][0..group]);
    }
};

pub fn init(gpu: check.Gpu, s: cuda.Stream, norms: *const cuda.norms.Norms, view: View) Tiles {
    return .{ .rig = .{ .gpu = gpu, .s = s }, .view = view, .norms = norms };
}
