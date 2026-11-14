//! GLM paged E4M3 writer under the CUDA contract: codes and scales follow one shuffled logical table.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const rig_ = @import("contract_rig.zig");
const pages = @import("contract_glm_pages.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const lw = pages.lw;
const row_bytes = lw + 4;
const pool_bytes = (pages.rows_a_page / 4) * pages.dk * 2;
pub const part_bytes = pages.rows_a_page * row_bytes + pool_bytes;
pub const page_bytes = 2 * part_bytes;

fn noSums(_: c.Shape) usize {
    return 0;
}

pub const spec: c.Spec = .{
    .name = "cuda glm tf_glm_lat_write_fp8_paged codes and scales on pages out of order",
    .accum = .none,
    .depth = noSums,
    .out = .raw,
    .nan = .{ .stated = "finite rows match the E4M3 CPU codec exactly; a non-finite row still writes a positive non-NaN scale and stays within its page" },
    .served = &.{ .{ .rows = 8, .n = lw, .k = 1 }, .{ .rows = 513, .n = lw, .k = 1 } },
};

fn copy(r: *rig_.Rig, dst: u64, src: u64, len: usize) !void {
    try (cuda.DeviceBuffer{ .d = r.gpu.d, .ptr = dst, .len = len }).copyFrom(0, src, len, r.s.handle);
}

pub const LatWrite = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    wrong_table: bool = false,
    shape: c.Shape = undefined,
    lat: []u16 = &.{},
    ids: []i32 = &.{},
    expected: []u8 = &.{},
    finite: []bool = &.{},
    at: usize = 0,
    input: u64 = 0,
    pos: u64 = 0,
    pt: u64 = 0,
    arena: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *LatWrite) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *LatWrite) void {
        const a = self.rig.gpu.gpa;
        a.free(self.lat);
        a.free(self.ids);
        a.free(self.expected);
        a.free(self.finite);
        self.lat = &.{};
        self.ids = &.{};
        self.expected = &.{};
        self.finite = &.{};
    }

    pub fn kernel(self: *LatWrite) g.Kernel {
        var declared = spec;
        if (self.wrong_table)
            declared.name = "control: GLM tf_glm_lat_write_fp8_paged uses a swapped logical table";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.wrong_table) .shapes else null };
    }

    const vt = g.hooks(LatWrite);

    pub fn prepare(self: *LatWrite, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != lw or shape.rows == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.at = if (shape.rows <= 64) pages.rows_a_page - shape.rows / 2 else 300;
        self.lat = try a.alloc(u16, shape.rows * lw);
        for (self.lat) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        @memset(self.lat[0..lw], 0); // zero row: scale is exactly one.
        if (shape.rows > 1) @memset(self.lat[lw .. 2 * lw], 1); // BF16 subnormal row: finite scale floor.
        if (shape.rows > 2) {
            self.lat[2 * lw] = c.toBf16(448);
            self.lat[2 * lw + 1] = c.toBf16(-448);
            self.lat[2 * lw + 2] = c.toBf16(1.0625); // round-even fixture.
        }
        if (fill == .nonfinite and shape.rows > 3) self.lat[3 * lw + 7] = c.toBf16(std.math.inf(f32));
        self.expected = try a.alloc(u8, shape.rows * row_bytes);
        @memset(self.expected, 0);
        self.finite = try a.alloc(bool, shape.rows);
        for (0..shape.rows) |row| {
            const codes = self.expected[row * row_bytes ..][0..lw];
            if (glm.kv8_ref.pack(self.lat[row * lw ..][0..lw], codes)) |scale| {
                self.finite[row] = true;
                @memcpy(self.expected[row * row_bytes + lw ..][0..4], std.mem.asBytes(&scale));
            } else |err| switch (err) {
                error.NonFiniteRow => self.finite[row] = false,
                else => return err,
            }
        }
        const n_pages = (self.at + shape.rows + pages.rows_a_page - 1) / pages.rows_a_page;
        self.ids = try pages.shuffled(a, rng, n_pages);
        const p: i32 = @intCast(self.at);
        self.input = try self.rig.input(std.mem.sliceAsBytes(self.lat));
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.pt = try pages.uploadTable(&self.rig, self.ids, self.wrong_table);
        self.arena = try self.rig.scratch((n_pages + 3) * page_bytes);
        self.out = try self.rig.output(shape.rows * row_bytes);
        return true;
    }

    pub fn run(self: *LatWrite) !void {
        const cache = self.arena + part_bytes;
        const scales = cache + pages.rows_a_page * lw;
        var args: cuda.Args = .{};
        inline for (.{ self.input, @as(i64, lw), cache, scales, self.pos, self.pt, @as(i64, page_bytes) }) |v| args.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = 512 } }, self.rig.s, &args);
        for (0..self.shape.rows) |r| {
            const logical = self.at + r;
            const physical: u64 = @intCast(self.ids[logical / pages.rows_a_page]);
            const row: u64 = @intCast(logical % pages.rows_a_page);
            const page = physical * page_bytes;
            try copy(&self.rig, self.out + r * row_bytes, cache + page + row * lw, lw);
            try copy(&self.rig, self.out + r * row_bytes + lw, scales + page + row * 4, 4);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *LatWrite) usize {
        return self.shape.rows * row_bytes;
    }

    pub fn reference(_: *LatWrite, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    pub fn check(self: *LatWrite, _: c.Gate) !?[]const u8 {
        const got = try self.rig.outputBytes();
        for (0..self.shape.rows) |row| {
            const actual = got[row * row_bytes ..][0..row_bytes];
            if (self.finite[row]) {
                if (!std.mem.eql(u8, actual, self.expected[row * row_bytes ..][0..row_bytes]))
                    return "finite row codes or scale differ from the E4M3 reference";
            } else {
                const bits = std.mem.readInt(u32, actual[lw..][0..4], .little);
                const scale: f32 = @bitCast(bits);
                if (std.math.isNan(scale) or scale <= 0) return "non-finite row wrote an invalid scale";
            }
        }
        return null;
    }
};
