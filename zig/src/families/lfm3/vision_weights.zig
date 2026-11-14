//! LFM3-VL's 138 dense BF16 vision tensors, borrowed from the owned checkpoint without conversions.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const st = @import("../../core/safetensors.zig");
const layout = @import("vision_layout.zig");

comptime {
    std.debug.assert(layout.max_rank == st.max_rank);
}

pub const width = layout.width;
pub const intermediate = layout.intermediate;
pub const depth = layout.depth;
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const Layouts = layout.Layout(Ref);
pub const Conv = Layouts.Conv;
pub const Attention = Layouts.Attention;
pub const Layer = Layouts.Layer;
pub const Weights = Layouts.Weights;

/// The loaded checkpoint as the layout reads it: each tensor with the buffer it sits in.
const Index = struct {
    checkpoint: *const ck.Checkpoint,

    const Names = struct {
        inner: std.StringHashMapUnmanaged(ck.Tensor).KeyIterator,

        pub fn next(self: *Names) ?[]const u8 {
            const name = self.inner.next() orelse return null;
            return name.*;
        }
    };

    pub fn find(self: Index, name: []const u8) !?layout.Found(Ref) {
        const t = self.checkpoint.get(name) catch |err| switch (err) {
            error.MissingTensor => return null,
            else => return err,
        };
        return .{
            .ref = .{ .buf = t.buffer, .off = t.offset },
            .bf16 = t.dtype == .bf16,
            .rank = t.rank,
            .shape = t.shape,
            .offset = t.offset,
            .bytes = t.bytes,
            .storage = t.buffer.length(),
        };
    }

    pub fn keys(self: Index) Names {
        return .{ .inner = self.checkpoint.tensors.keyIterator() };
    }
};

pub fn load(checkpoint: *const ck.Checkpoint) !Weights {
    return layout.load(Ref, checkpoint.allocator, Index{ .checkpoint = checkpoint });
}
