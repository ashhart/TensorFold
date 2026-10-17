//! TensorFold's Intel GPU runtime for Nemotron: ze.zig's Level Zero table, OpenCL C kernels as embedded SPIR-V.

pub const ze = @import("ze");
pub const Driver = @import("context.zig").Driver;
pub const Error = @import("context.zig").Error;
pub const context = @import("context.zig");
pub const Context = context.Context;
pub const DeviceBuffer = @import("memory.zig").DeviceBuffer;
pub const HostBuffer = @import("memory.zig").HostBuffer;
pub const Stream = @import("stream.zig").Stream;
pub const Module = @import("module.zig").Module;
pub const Kernel = @import("module.zig").Kernel;
pub const launch = @import("launch.zig");
pub const rt = @import("rt.zig");
pub const loader = @import("loader.zig");
pub const Pool = @import("pool.zig").Pool;
pub const stop = @import("stop.zig");
pub const decode = @import("decode.zig");
pub const kernels = @import("kernels.zig");

test {
    _ = context;
    _ = launch;
    _ = decode;
}
