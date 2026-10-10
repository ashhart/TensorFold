//! The weight formats behind one interface: a format module provides what `conforms` lists; nothing above names one.

const std = @import("std");

pub const types = @import("types.zig");
pub const slice = @import("slice.zig");
pub const convert = @import("convert.zig");
pub const mlx = @import("mlx.zig");
pub const dense = @import("dense.zig");

const Allocator = std.mem.Allocator;

pub const Format = types.Format;
pub const Decoder = types.Decoder;
pub const Tables = types.Tables;
pub const Tensor = types.Tensor;
pub const DType = types.DType;
pub const Buf = types.Buf;
pub const Uploader = types.Uploader;
pub const Sources = types.Sources;
pub const Rank = slice.Rank;
pub const Span = slice.Span;
pub const join = types.join;

/// The module of a format.
pub fn Module(comptime f: Format) type {
    return switch (f) {
        .mlx => mlx,
        .dense => dense,
    };
}

/// The order a tensor is offered to the formats: the first that recognizes it reads it.
const read_order = [_]Format{ .dense, .mlx };

/// What a format module provides, checked at compile time.
pub fn conforms(comptime M: type) void {
    const required = .{
        "id", "decoder", // which format, and the device decoder its kernels use
        "Config", "detect", // the checkpoint's own declaration of the format (null: not this format)
        "Host", "matches", "read", "rows", "bytes", "halves", // a tensor's host side: recognized, read and split
        "sliceRows", "sliceCols", "sliceStack", // tensor parallelism: output rows, input columns, experts
        "Device", "upload", // load: onto the device, repacked there when the decoder reads a layout of its own
        "Matrix", "View", "view", // what the kernels read of an uploaded tensor
        "elements", "reference", // fp64 dequantization of every weight, for the tests
    };
    inline for (required) |name| {
        if (!@hasDecl(M, name)) @compileError("a format module needs `" ++ name ++ "`");
    }
}

/// The checkpoint's declared quantization.
pub const Config = union(Format) {
    mlx: mlx.Config,
    dense: dense.Config,

    /// The tensor's MLX width (the table's own, else the global one).
    pub fn width(c: Config, key: []const u8) error{UnsupportedQuantization}!mlx.Width {
        return switch (c) {
            .mlx => |q| q.width(key),
            .dense => error.UnsupportedQuantization,
        };
    }
};

/// The format the config declares, or `error.UnsupportedQuantization`.
pub fn detect(a: Allocator, src: Sources) (Allocator.Error || error{UnsupportedQuantization})!Config {
    inline for (read_order) |f| {
        if (try Module(f).detect(a, src)) |c| return @unionInit(Config, @tagName(f), c);
    }
    return error.UnsupportedQuantization;
}

/// A projection read from the checkpoint, still on the host.
pub const Host = union(Format) {
    mlx: mlx.Host,
    dense: dense.Host,

    /// Output rows N.
    pub fn rows(h: Host) usize {
        return switch (h) {
            inline else => |x, tag| Module(tag).rows(x),
        };
    }

    /// The bytes of its tensors.
    pub fn bytes(h: Host) usize {
        return switch (h) {
            inline else => |x, tag| Module(tag).bytes(x),
        };
    }
};

fn wrap(comptime f: Format, v: Module(f).Host) Host {
    return @unionInit(Host, @tagName(f), v);
}

/// The projection `key` names: the first format that recognizes the tensors reads it.
pub fn read(a: Allocator, t: anytype, key: []const u8) !Host {
    inline for (read_order) |f| {
        if (Module(f).matches(t, key)) return wrap(f, try Module(f).read(a, t, key));
    }
    return types.refuse(key, "no weight format reads this tensor");
}

/// A fused [embedding | hidden] projection as its two halves along K.
pub fn halves(a: Allocator, h: Host) ![2]Host {
    switch (h) {
        inline else => |x, tag| {
            const pair = try Module(tag).halves(a, x);
            return .{ wrap(tag, pair[0]), wrap(tag, pair[1]) };
        },
    }
}

/// Output rows `spans`: a column-split projection's share on one rank.
pub fn sliceRows(a: Allocator, h: Host, spans: []const Span) slice.Error!Host {
    switch (h) {
        inline else => |x, tag| return wrap(tag, try Module(tag).sliceRows(a, x, spans)),
    }
}

/// One rank's input groups of a row-split projection.
pub fn sliceCols(a: Allocator, h: Host, r: Rank, what: []const u8) slice.Error!Host {
    switch (h) {
        inline else => |x, tag| return wrap(tag, try Module(tag).sliceCols(a, x, r, what)),
    }
}

/// A rank's `part` experts of a stack, and the shared one on rank 0.
pub fn sliceStack(a: Allocator, h: Host, r: Rank, part: usize) slice.Error!Host {
    switch (h) {
        inline else => |x, tag| return wrap(tag, try Module(tag).sliceStack(a, x, r, part)),
    }
}

/// A projection on the device, as its format uploaded it.
pub const Device = union(Format) {
    mlx: mlx.Device,
    dense: dense.Device,
};

pub fn upload(u: Uploader, h: Host) !Device {
    switch (h) {
        inline else => |x, tag| return @unionInit(Device, @tagName(tag), try Module(tag).upload(u, x)),
    }
}

/// What the kernels read of a projection, by format.
pub const Handle = union(Format) {
    mlx: mlx.Matrix,
    dense: dense.Matrix,
};

/// One product (N, K): the format's handle and its shape. A stack is one expert's shape at the stack's addresses.
pub const Projection = struct {
    n: u32,
    k: u32,
    /// Split along K: the product stays fp32, one rank's share of a sum.
    partial: bool = false,
    handle: Handle,

    pub fn format(p: Projection) Format {
        return std.meta.activeTag(p.handle);
    }

    pub fn decoder(p: Projection) Decoder {
        return switch (p.format()) {
            inline else => |f| Module(f).decoder,
        };
    }
};

/// The view of an uploaded projection, or of a stack of them (`stacked`).
pub fn view(d: Device, stacked: bool) error{UnsupportedTables}!Projection {
    switch (d) {
        inline else => |x, tag| {
            const v = try Module(tag).view(x, stacked);
            return .{ .n = v.n, .k = v.k, .handle = @unionInit(Handle, @tagName(tag), v.matrix) };
        },
    }
}

test "every format module provides the interface" {
    comptime conforms(mlx);
    comptime conforms(dense);
}

test "a projection knows its format and decoder" {
    try std.testing.expectEqual(Decoder.mlx, (Projection{ .n = 1, .k = 32, .handle = .{ .mlx = undefined } }).decoder());
    try std.testing.expectEqual(Format.mlx, (Projection{ .n = 1, .k = 32, .handle = .{ .mlx = undefined } }).format());
}

test {
    _ = @import("exact_test.zig");
    _ = slice;
    _ = convert;
    _ = mlx;
    _ = dense;
}
