//! CUDA launch for Hy4-flash's FP32 sparse-index key pooling.
const cuda = @import("cuda");
const groups = @import("index_groups.zig");

pub const symbol = "tf_hy4_pool_groups";
pub const threads = 128;

pub const Bindings = struct { values: u64, gates: u64, position_bias: u64, pooled: u64 };

pub fn open(driver: *const cuda.Driver) !cuda.Module {
    return cuda.Module.load(driver, cuda.kernels.hy4_indexpool);
}

pub fn load(module: cuda.Module) !cuda.Function {
    return module.function(symbol);
}

pub fn run(kernel: cuda.Function, stream: cuda.Stream, refs: Bindings, shape: groups.PoolShape) !void {
    if (shape.groups == 0 or shape.width == 0) return;
    var args: cuda.Args = .{};
    inline for (.{ refs.values, refs.gates, refs.position_bias, refs.pooled, shape.groups, shape.width }) |arg| {
        args.add(arg);
    }
    const columns = (shape.width - 1) / threads + 1;
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = shape.groups, .y = columns }, .block = .{ .x = threads } }, stream, &args);
}
