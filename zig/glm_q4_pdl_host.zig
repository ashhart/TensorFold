//! GLM's Q4 draft-head launch honors the process-wide PDL override without a device.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const q4 = glm.q4;

const c_env = struct {
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
};

const Seen = struct {
    var plain: usize = 0;
    var dependent: usize = 0;

    fn run(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        plain += 1;
        return cuda.abi.success;
    }

    fn ex(c: *const cuda.abi.LaunchConfig, _: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| if (at.id == .programmatic_stream_serialization and at.value.int == 1) {
            dependent += 1;
            return cuda.abi.success;
        };
        plain += 1;
        return cuda.abi.success;
    }
};

test "GLM Q4 PDL: exact 0 makes the GB10 draft-head launch plain" {
    try std.testing.expectEqual(@as(c_int, 0), c_env.setenv("TF_GLM_PDL", "0", 1));
    defer _ = c_env.unsetenv("TF_GLM_PDL");

    var d: cuda.Driver = .{ .lib = undefined, .api = undefined, .func_name = null };
    d.api.cuLaunchKernel = Seen.run;
    d.api.cuLaunchKernelEx = Seen.ex;
    const f: cuda.Function = .{ .d = &d, .handle = @ptrFromInt(10) };
    const ks: q4.Kernels = .{ .group = f, .pack_dense = f, .transpose = f, .gb10 = true };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const weight: q4.Q4 = .{ .w = 0x1000, .s = 0x2000, .b = 0x3000, .n = 128, .k = 64, .npad = 128 };

    try q4.matmul(&ks, stream, 0x4000, 64, 0x5000, weight, 0x6000, 1);
    try std.testing.expectEqual(@as(usize, 1), Seen.plain);
    try std.testing.expectEqual(@as(usize, 0), Seen.dependent);
}
