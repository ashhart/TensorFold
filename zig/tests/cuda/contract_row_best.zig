//! The baseline proposal row reduction through guards, first-ID checks and the independent float64 LSE oracle.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
pub const ref = @import("contract_row_best_ref.zig");
const c = core.contract.spec;
const g = core.contract.gates;

pub const Best = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    shape: c.Shape = undefined,
    bits: []u16 = &.{},
    input: u64 = 0,
    output: u64 = 0,
    probe: u64 = 0,

    pub fn deinit(self: *Best) void {
        self.rig.gpu.gpa.free(self.bits);
        self.rig.deinit();
    }

    const vt = g.hooks(Best);

    pub fn kernel(self: *Best) g.Kernel {
        return .{ .spec = ref.spec, .ctx = self, .vt = &vt };
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

    pub fn run(self: *Best) !void {
        try self.launch(self.input, self.output, self.shape.rows);
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
        if (gate == .shapes) for (0..self.shape.rows) |row| {
            try self.launch(self.input + row * (self.shape.k + 17) * 2, self.probe, 1);
            if (!std.mem.eql(u8, actual[row * 16 ..][0..16], try self.rig.bytes(self.probe, 16)))
                return "a batched reduction row differs from the same input reduced alone";
        };
        if (!try self.rig.intact()) return "row reduction changed a payload guard";
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.bits), try self.rig.inputBytes())) return "row reduction changed immutable logits";
        return null;
    }
};
