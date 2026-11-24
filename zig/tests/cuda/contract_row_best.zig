//! The baseline proposal row reduction through guards, first-ID checks and the independent float64 LSE oracle.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
pub const ref = @import("contract_row_best_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

/// fn_row_best; its split over 32 blocks a row (fn_row_best_max, fn_row_best_sum); a control: the split over V - 1.
pub const Kind = enum { whole, split, control };

fn spec(kind: Kind) c.Spec {
    var s = ref.spec;
    if (kind == .whole) return s;
    s.name = if (kind == .split) "cuda Flash Next fn_row_best_max + fn_row_best_sum (replaces fn_row_best)" else "control: the split row best over V - 1 values posing as fn_row_best";
    s.replaces = "fn_row_best";
    return s;
}

pub const Best = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    kind: Kind = .whole,
    split: ?[2]cuda.Function = null,
    shape: c.Shape = undefined,
    bits: []u16 = &.{},
    input: u64 = 0,
    output: u64 = 0,
    probe: u64 = 0,
    scratch: u64 = 0,
    base: u64 = 0,

    pub fn deinit(self: *Best) void {
        self.rig.gpu.gpa.free(self.bits);
        self.rig.deinit();
    }

    const vt = g.hooks(Best);

    pub fn kernel(self: *Best) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = if (self.kind == .control) .shapes else null };
    }

    pub fn prepare(self: *Best, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!ref.admits(shape)) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        a.free(self.bits);
        self.bits = &.{};
        self.shape = shape;
        self.bits = try a.alloc(u16, shape.rows * (shape.k + 17));
        ref.fill(self.bits, shape, fill, seed);
        self.input = try self.rig.input(std.mem.sliceAsBytes(self.bits));
        self.output = try self.rig.output(self.count() * 4);
        self.probe = try self.rig.scratch(16);
        if (self.kind != .whole) {
            self.scratch = try self.rig.scratch(shape.rows * (32 * 12 + 4));
            self.base = try self.rig.scratch(self.count() * 4);
        }
        return true;
    }

    fn launch(self: *Best, input: u64, output: u64, rows: usize) !void {
        var args: cuda.Args = .{};
        args.add(input);
        args.add(@as(c_int, @intCast(self.shape.k)));
        args.add(@as(c_longlong, @intCast(self.shape.k + 17)));
        args.add(output);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    fn launchSplit(self: *Best, v: usize) !void {
        const fns = self.split.?;
        const rows = self.shape.rows;
        const sums = self.scratch + rows * 32 * 8;
        const cfg: cuda.launch.Config = .{ .grid = .{ .x = @intCast(rows), .y = 32 }, .block = .{ .x = 32 } };
        var first: cuda.Args = .{};
        first.add(self.input);
        first.add(@as(c_int, @intCast(v)));
        first.add(@as(c_longlong, @intCast(self.shape.k + 17)));
        first.add(self.scratch);
        try cuda.launch.launch(fns[0], cfg, self.rig.s, &first);
        var second: cuda.Args = .{};
        second.add(self.input);
        second.add(@as(c_int, @intCast(v)));
        second.add(@as(c_longlong, @intCast(self.shape.k + 17)));
        second.add(self.scratch);
        second.add(sums);
        second.add(sums + rows * 32 * 4);
        second.add(self.output);
        try cuda.launch.launch(fns[1], cfg, self.rig.s, &second);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Best) !void {
        switch (self.kind) {
            .whole => try self.launch(self.input, self.output, self.shape.rows),
            .split => try self.launchSplit(self.shape.k),
            .control => try self.launchSplit(@max(self.shape.k, 2) - 1),
        }
    }

    pub fn replaced(self: *Best) ![]const u8 {
        try self.launch(self.input, self.base, self.shape.rows);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Best) usize {
        return self.shape.rows * 4;
    }

    pub fn reference(self: *Best, want: []f64, mag: []f64) !void {
        ref.reference(self.bits, self.shape, want, mag);
    }

    pub fn simulate(self: *Best, acc: c.Accum, order: g.Order, out: []f64) !void {
        try ref.simulate(self.rig.gpu.gpa, self.bits, self.shape, acc, order, out);
    }

    pub fn check(self: *Best, gate: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const actual = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(actual);
        if (ref.check(self.bits, self.shape, actual)) |why| return why;
        if (gate == .shapes and self.kind == .whole) for (0..self.shape.rows) |row| {
            try self.launch(self.input + row * (self.shape.k + 17) * 2, self.probe, 1);
            if (!std.mem.eql(u8, actual[row * 16 ..][0..16], try self.rig.bytes(self.probe, 16)))
                return "a batched reduction row differs from the same input reduced alone";
        };
        if (!try self.rig.intact()) return "row reduction changed a payload guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.bits), try self.rig.inputBytes())) return "row reduction changed immutable logits";
        return null;
    }
};
