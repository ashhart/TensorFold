//! Exceptional groups are an admission verdict, never an int8 numeric result.
const std = @import("std");
const cuda = @import("cuda");
const Case = @import("contract_w4a8.zig").Case;
const c = @import("core").contract.spec;
fn bytes(self: *Case, n: usize, fill: u8) !u64 {
    const host = try self.arena.allocator().alloc(u8, n);
    @memset(host, fill);
    return self.rig.input(host);
}
fn upload(self: *Case, ptr: u64, data: []const u8) !void {
    const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = ptr, .len = data.len, .borrowed = true };
    try b.upload(0, data);
}
pub fn prepare(self: *Case) !void {
    const s = self.shape;
    self.dx = try self.rig.input(std.mem.sliceAsBytes(self.x));
    const size = cuda.w4a8.Kernels.scratch(s.rows, s.k);
    self.dq = try self.rig.scratch(size);
    self.da = self.dq + s.rows * s.k;
    self.ds = self.da + s.rows * (s.k / 64) * 4;
    self.out = try self.rig.output(s.rows * s.n * 2);
    // Even a broken flag remains a safe numeric launch, so the test detects consumption rather than a driver fault.
    const layout = cuda.qlinear.Affine4.layout(s.n, s.k);
    self.weight = .{ .w = try bytes(self, layout.words * 4, 0), .s = try bytes(self, layout.scales * 2, 0), .b = try bytes(self, layout.scales * 2, 0), .n = s.n, .k = s.k, .npad = layout.npad };
}
pub fn run(self: *Case) !void {
    const s = self.shape;
    const stride = if (self.kind == .prefill) s.k else s.k + 64;
    const work = try self.arena.allocator().alloc(u8, cuda.w4a8.Kernels.scratch(s.rows, s.k));
    @memset(work, 0xa5);
    const out = try self.arena.allocator().alloc(u8, s.rows * s.n * 2);
    @memset(out, 0xa5);
    const plants = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    self.exception_failure = null;
    for (0..s.rows) |row| for (0..s.k / 64) |group| for ([_]usize{ 0, 31, 32, 63 }) |lane| for (plants) |plant| {
        @memset(self.x, c.toBf16(1));
        self.x[row * stride + group * 64 + lane] = c.toBf16(plant);
        try upload(self, self.dx, std.mem.sliceAsBytes(self.x));
        try upload(self, self.dq, work);
        try upload(self, self.out, out);
        var flag: u32 = 0;
        if (self.kind == .quantize) {
            try self.kernels.quantizeRows(self.rig.s, self.dx, self.dq, self.da, self.ds, s.rows, s.k, stride, self.flag, &flag);
        } else try self.kernels.prompt(self.rig.s, self.dx, self.weight, self.out, s.rows, self.dq, self.flag, &flag);
        if (self.fault == .missed_flag) try upload(self, self.flag, &.{ 0, 0, 0, 0 });
        const raw = try self.rig.bytes(self.flag, 4);
        if (flag == 0 or std.mem.readInt(u32, raw[0..4], .little) == 0) {
            self.exception_failure = "exceptional group did not raise the chunk flag";
            return;
        }
        const code = try self.rig.bytes(self.dq + row * s.k + group * 64, 64);
        if (!std.mem.allEqual(u8, code, 0xa5)) {
            self.exception_failure = "exceptional group wrote int8 codes instead of discarding it";
            return;
        }
        const scale = try self.rig.bytes(self.da + (row * (s.k / 64) + group) * 4, 4);
        const sum = try self.rig.bytes(self.ds + (row * (s.k / 64) + group) * 4, 4);
        if (!std.mem.allEqual(u8, scale, 0xa5) or !std.mem.allEqual(u8, sum, 0xa5)) {
            self.exception_failure = "exceptional group wrote a scale or sum";
            return;
        }
        if (self.kind == .prefill and !std.mem.eql(u8, out, try self.rig.bytes(self.out, out.len))) {
            self.exception_failure = "an exceptional chunk consumed int8 outputs";
            return;
        }
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(self.x), try self.rig.bytes(self.dx, self.x.len * 2)) or !try self.rig.intact()) {
            self.exception_failure = "exceptional quantization changed an input or guard";
            return;
        }
    };
}
