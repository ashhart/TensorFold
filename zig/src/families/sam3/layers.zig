const ops = @import("ops.zig");

pub const Layer = struct {
    runtime: *ops.Runtime,
    prefix: []const u8,

    pub fn named(self: Layer, suffix: []const u8) ![]const u8 {
        return self.runtime.name("{s}.{s}", .{ self.prefix, suffix });
    }

    pub fn child(self: Layer, suffix: []const u8) !Layer {
        return .{ .runtime = self.runtime, .prefix = try self.named(suffix) };
    }

    pub fn indexed(self: Layer, suffix: []const u8, index: usize) !Layer {
        return self.child(try self.runtime.name("{s}.{d}", .{ suffix, index }));
    }

    pub fn parameter(self: Layer, suffix: []const u8) !ops.Tensor {
        return self.runtime.weight(try self.named(suffix));
    }

    pub fn linear(self: Layer, suffix: []const u8, input: ops.Tensor) !ops.Tensor {
        const name = try self.named(suffix);
        const matrix = try self.runtime.store.get(try self.runtime.name("{s}.weight", .{name}));
        if (matrix.rank != 2 or matrix.dim(0) == 0 or matrix.dim(1) == 0 or input.count % matrix.dim(1) != 0)
            return error.SamProjectionShape;
        return self.runtime.linear(name, input, matrix.dim(1));
    }

    pub fn norm(self: Layer, suffix: []const u8, input: ops.Tensor, epsilon: f32) !ops.Tensor {
        const name = try self.named(suffix);
        const scale = try self.runtime.store.get(try self.runtime.name("{s}.weight", .{name}));
        if (scale.rank != 1 or scale.numel() == 0 or input.count % scale.numel() != 0) return error.SamNormShape;
        return self.runtime.norm(name, input, scale.numel(), epsilon);
    }

    pub fn mlp(self: Layer, suffix: []const u8, input: ops.Tensor, depth: usize) !ops.Tensor {
        const network = try self.child(suffix);
        var value = input;
        for (0..depth) |index| {
            value = try network.linear(try self.runtime.name("layers.{d}", .{index}), value);
            if (index + 1 < depth)
                value = try self.runtime.unary(value, 0);
        }
        return value;
    }

    pub fn feedForward(self: Layer, input: ops.Tensor, gelu: bool) !ops.Tensor {
        const hidden = try self.linear("linear1", input);
        return self.linear("linear2", if (gelu) try self.runtime.gelu(hidden) else try self.runtime.unary(hidden, 0));
    }
};
