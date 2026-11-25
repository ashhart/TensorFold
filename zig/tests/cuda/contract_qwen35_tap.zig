//! Qwen3.5 target tap capture: full shared-drafter writes and the MTP suffix boundary.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check_ = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;

const tap_rows = 6144; // cuda_cache.zig: 2,048 retained rows plus one 4,096-row target chunk.
const hidden = 8; // tf_q35_tap's D need only be a multiple of eight.
const rows = 8;
const start: usize = 6138; // six rows before the planted absolute boundary at 6,143.
const retained: usize = 3;
const Seg = extern struct {
    rec: u64 = 0,
    conv: u64 = 0,
    kv: u64 = 0,
    tap: u64,
    log: u64 = 0,
    before: u64 = 0,
    cap: c_int = 8192,
    len: c_int = @intCast(start),
    row0: c_int = 0,
    rows: c_int = @intCast(rows),
    tap_rows: c_int = @intCast(tap_rows),
    taps: c_int = 1,
    delta: c_int = 0,
    pt: u64 = 0,
    ps: u64 = 0,
    ptok: c_int = 0,
    pn: c_int = 0,
    plog: u64 = 0,
};

pub const Kind = enum { full, mtp_suffix, full_truncated_control, mtp_boundary_control };

fn depth(_: c.Shape) usize {
    return 0;
}

pub fn spec(kind: Kind) c.Spec {
    const name: []const u8 = switch (kind) {
        .full => "cuda qwen35 tap full capture (DFlash2/shared taps)",
        .mtp_suffix => "cuda qwen35 tap filtered MTP suffix including position 6143",
        .full_truncated_control => "control qwen35 tap incorrectly truncating shared capture",
        .mtp_boundary_control => "control qwen35 tap omitting MTP's first retained row",
    };
    return .{
        .name = name,
        .accum = .none,
        .depth = depth,
        .out = .raw,
        .nan = .{ .stated = "exact BF16 residual bytes and untouched poison slots" },
        .served = &.{.{ .rows = rows, .n = hidden, .k = 1 }},
    };
}

pub const Case = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    kind: Kind,
    h: []u16 = &.{},
    r: []u16 = &.{},
    expected: []u8 = &.{},
    input_copy: []u8 = &.{},
    tap: u64 = 0,
    segs: u64 = 0,
    rowseg: u64 = 0,
    hdev: u64 = 0,
    rdev: u64 = 0,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        const a = self.rig.gpu.gpa;
        a.free(self.h);
        a.free(self.r);
        a.free(self.expected);
        a.free(self.input_copy);
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const control = self.kind == .full_truncated_control or self.kind == .mtp_boundary_control;
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (control) .shapes else null };
    }

    pub fn prepare(self: *Case, shape: c.Shape, _: g.Fill, _: u64) !bool {
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.h);
        a.free(self.r);
        a.free(self.expected);
        a.free(self.input_copy);
        self.h = &.{};
        self.r = &.{};
        self.expected = &.{};
        self.input_copy = &.{};
        if (shape.rows != rows or shape.n != hidden or shape.k != 1) return false;

        self.h = try a.alloc(u16, rows * hidden);
        self.r = try a.alloc(u16, rows * hidden);
        for (0..rows * hidden) |i| {
            self.h[i] = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 8);
            self.r[i] = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16);
        }
        self.tap = try self.rig.output(tap_rows * hidden * 2);
        self.expected = try a.alloc(u8, tap_rows * hidden * 2);
        @memset(self.expected, 0xff);
        const expected_keep = switch (self.kind) {
            .full => 0,
            .mtp_suffix => retained,
            .full_truncated_control => 0,
            .mtp_boundary_control => retained,
        };
        const cutoff = start + rows - expected_keep;
        for (0..rows) |row| {
            const pos = start + row;
            const should_write = expected_keep == 0 or pos >= cutoff;
            if (!should_write) continue;
            const slot = pos % tap_rows;
            for (0..hidden) |col| {
                const at = row * hidden + col;
                const sum = core.bf16_round.bits(c.fromBf16(self.h[at]) + c.fromBf16(self.r[at]));
                std.mem.writeInt(u16, self.expected[(slot * hidden + col) * 2 ..][0..2], sum, .little);
            }
        }

        self.hdev = try self.rig.input(std.mem.sliceAsBytes(self.h));
        self.rdev = try self.rig.input(std.mem.sliceAsBytes(self.r));
        const seg = Seg{ .tap = self.tap };
        self.segs = try self.rig.input(std.mem.asBytes(&seg));
        const row_index: [rows]c_int = @splat(0);
        self.rowseg = try self.rig.input(std.mem.asBytes(&row_index));
        self.input_copy = try a.alloc(u8, self.h.len * 4 + @sizeOf(Seg) + @sizeOf(@TypeOf(row_index)));
        var at: usize = 0;
        for ([_][]const u8{ std.mem.sliceAsBytes(self.h), std.mem.sliceAsBytes(self.r), std.mem.asBytes(&seg), std.mem.asBytes(&row_index) }) |part| {
            @memcpy(self.input_copy[at..][0..part.len], part);
            at += part.len;
        }
        return true;
    }

    pub fn run(self: *Case) !void {
        const keep: c_int = switch (self.kind) {
            .full => 0,
            .mtp_suffix => retained,
            .full_truncated_control => retained,
            .mtp_boundary_control => retained - 1,
        };
        var args: cuda.Args = .{};
        args.add(self.hdev);
        args.add(self.rdev);
        args.add(self.segs);
        args.add(self.rowseg);
        args.add(@as(c_int, 0)); // tap index
        args.add(@as(c_int, @intCast(hidden)));
        args.add(keep);
        args.add(@as(u64, 0)); // no partial-row params
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = rows, .y = 1 }, .block = .{ .x = 256 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.expected.len;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (want, 0..) |*v, i| v.* = @floatFromInt(self.expected[i]);
        @memset(mag, 0);
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        if (!std.mem.eql(u8, got, self.expected)) {
            const at = std.mem.indexOfDiff(u8, got, self.expected).?;
            return if (at / (hidden * 2) % tap_rows == 6143 % tap_rows)
                "tap bytes differ at or around the planted position-6143 boundary"
            else
                "tap bytes differ or a skipped ring slot was overwritten";
        }
        if (!std.mem.eql(u8, try self.rig.inputBytes(), self.input_copy)) return "tap capture changed an input";
        if (!try self.rig.intact()) return "tap capture changed an input or guard";
        return null;
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    cases: [4]Case = undefined,
    used: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check_.Gpu, s: cuda.Stream, mod: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        const f = try mod.function("tf_q35_tap");
        for (std.enums.values(Kind), &self.cases) |kind, *item| {
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = f, .kind = kind };
            self.used += 1;
            try kernels.append(a, item.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.cases[0..self.used]) |*item| item.deinit();
        self.a.destroy(self);
    }
};
