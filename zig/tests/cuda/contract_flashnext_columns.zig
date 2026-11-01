//! Rank-block column copies preserve raw 16-bit words, including both halves of FP32 values and untouched slots.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Rig = @import("contract_rig.zig").Rig;
const c = core.contract.spec;
const g = core.contract.gates;

fn noSums(_: c.Shape) usize {
    return 0;
}

pub const Columns = struct {
    rig: Rig,
    f: cuda.Function,
    slot: bool = false,
    control: bool = false,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    input: std.ArrayList(u16) = .empty,
    src: u64 = 0,
    dst: u64 = 0,
    msg: [160]u8 = undefined,
    const vt = g.hooks(Columns);

    pub fn deinit(self: *Columns) void {
        self.input.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    pub fn kernel(self: *Columns) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null, .spec = .{
            .name = if (self.control) "control: Flash Next columns read rank zero twice" else if (self.slot)
                "cuda flashnext columns into the shared expert slot"
            else
                "cuda flashnext rank blocks into full input and output rows",
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "every input word is copied without arithmetic; all other slots retain their bytes" },
            .tiles = .{ .{ .size = 16 }, .{ .size = 256, .step = 1, .min = 1 }, .{} },
            .served = &.{ .{ .rows = 1, .n = 320, .k = 1 }, .{ .rows = 17, .n = 2560, .k = 1 }, .{ .rows = 1536, .n = 3072, .k = 1 } },
        } };
    }

    pub fn prepare(self: *Columns, shape: c.Shape, _: g.Fill, seed: u64) !bool {
        self.rig.reset();
        self.shape = shape;
        if (shape.rows == 0 or shape.n == 0) return false;
        try self.input.resize(self.rig.gpu.gpa, 2 * shape.rows * shape.n);
        var rng = std.Random.DefaultPrng.init(seed);
        for (self.input.items) |*v| v.* = rng.random().int(u16);
        self.src = try self.rig.input(std.mem.sliceAsBytes(self.input.items));
        self.dst = try self.rig.output(self.count());
        return true;
    }

    pub fn run(self: *Columns) !void {
        const s = self.shape;
        const slots: usize = if (self.slot) 11 else 1;
        const pitch = 2 * s.n * slots;
        for (0..2) |rank| {
            var a: cuda.Args = .{};
            a.add(self.src + (if (self.control) 0 else rank) * s.rows * s.n * 2);
            a.add(@as(i64, @intCast(s.n)));
            a.add(self.dst + ((slots - 1) * 2 * s.n + rank * s.n) * 2);
            a.add(@as(i64, @intCast(pitch)));
            a.add(@as(i32, @intCast(s.n)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((s.n + 255) / 256), .y = @intCast(s.rows) }, .block = .{ .x = 256 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Columns) usize {
        return self.shape.rows * self.shape.n * 4 * @as(usize, if (self.slot) 11 else 1);
    }

    pub fn reference(_: *Columns, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *Columns, _: g.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        const s = self.shape;
        const slots: usize = if (self.slot) 11 else 1;
        for (0..s.rows) |row| for (0..slots) |slot| for (0..2) |rank| for (0..s.n) |col| {
            const index = ((row * slots + slot) * 2 + rank) * s.n + col;
            const want: u16 = if (slot == slots - 1) self.input.items[(rank * s.rows + row) * s.n + col] else 0xffff;
            if (std.mem.readInt(u16, got[index * 2 ..][0..2], .little) != want)
                return try std.fmt.bufPrint(&self.msg, "row {d} slot {d} rank {d} column {d} differs", .{ row, slot, rank, col });
        };
        return null;
    }
};
