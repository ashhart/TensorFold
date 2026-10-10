//! What the op tests call `rt`: a Runtime over a freshly opened device plus the runtime's types.

const std = @import("std");
const xpu = @import("xpu");

pub const Error = xpu.rt.Error;
pub const Buffer = xpu.rt.Buffer;
pub const Module = xpu.rt.Module;
pub const Kernel = xpu.rt.Kernel;
pub const profDump = xpu.rt.profDump;
pub const panicDrain = xpu.rt.panicDrain;

pub const Runtime = xpu.rt.Runtime;

/// Opens the loader, the device (TF_DEVICE, else the first Arc card) and a stream; they live until the process ends.
pub fn open() Error!xpu.rt.Runtime {
    const gpa = std.heap.page_allocator;
    const drv = gpa.create(xpu.Driver) catch return error.OutOfMemory;
    drv.* = try xpu.Driver.open();
    const ctx = gpa.create(xpu.Context) catch return error.OutOfMemory;
    const ordinal: ?u32 = if (std.c.getenv("TF_DEVICE")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch null else null;
    ctx.* = try xpu.Context.init(drv, ordinal);
    const stream = try xpu.Stream.init(ctx);
    return xpu.rt.Runtime.init(ctx, stream);
}
