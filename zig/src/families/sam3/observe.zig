pub const Tensor = @import("tensor_metal").Tensor;

pub const Observer = struct {
    context: *anyopaque,
    write: *const fn (*anyopaque, []const u8, Tensor, []const usize) anyerror!void,
};
