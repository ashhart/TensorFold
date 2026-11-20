const std = @import("std");
const st = @import("safetensors");

pub const Masks = struct {
    masks: st.Tensor,
    ids: ?st.Tensor,
    indices: ?st.Tensor,
    active: ?st.Tensor,
    frames: usize,
    objects: usize,
    height: usize,
    width: usize,
    pixels: usize,

    pub fn fromFile(file: *const st.File) !Masks {
        return init(file.get("masks") orelse return error.MissingMasks, file.get("object_ids"), file.get("frame_indices"), file.get("active"), file.get("complete"));
    }

    pub fn init(masks: st.Tensor, ids: ?st.Tensor, indices: ?st.Tensor, active: ?st.Tensor, complete: ?st.Tensor) !Masks {
        if (masks.dtype != .u8 or (masks.rank != 3 and masks.rank != 4)) return error.MaskShapeMismatch;
        const frames = masks.dim(0);
        const objects = if (masks.rank == 4) masks.dim(1) else 1;
        const height = masks.dim(masks.rank - 2);
        const width = masks.dim(masks.rank - 1);
        if (frames == 0 or objects == 0 or height == 0 or width == 0) return error.EmptyMaskComparison;
        const pixels = try std.math.mul(usize, height, width);
        const bytes = try std.math.mul(usize, try std.math.mul(usize, frames, objects), pixels);
        if (masks.bytes.len != bytes) return error.MaskShapeMismatch;
        const has_metadata = ids != null or indices != null or active != null or complete != null;
        if (has_metadata or masks.rank == 4) {
            const object_ids = ids orelse return error.IncompleteMaskMetadata;
            const frame_indices = indices orelse return error.IncompleteMaskMetadata;
            const activity = active orelse return error.IncompleteMaskMetadata;
            const done = complete orelse return error.IncompleteMaskMetadata;
            if (!object_ids.is(.u64, &.{objects}) or !frame_indices.is(.u64, &.{frames}) or
                !activity.is(.u8, &.{ frames, objects }) or !done.is(.u8, &.{1})) return error.InvalidMaskMetadata;
            if (done.bytes.len != 1 or done.bytes[0] != 1) return error.IncompleteMaskCapture;
            for (0..objects) |i| for (0..i) |j| {
                if (integer(object_ids, i) == integer(object_ids, j)) return error.DuplicateObjectId;
            };
            for (0..frames) |i| for (0..i) |j| {
                if (integer(frame_indices, i) == integer(frame_indices, j)) return error.DuplicateFrameIndex;
            };
            for (activity.bytes) |value| if (value > 1) return error.InvalidMaskActivity;
        }
        return .{ .masks = masks, .ids = ids, .indices = indices, .active = active, .frames = frames, .objects = objects, .height = height, .width = width, .pixels = pixels };
    }

    pub fn frameIndex(self: Masks, frame: usize) u64 {
        return if (self.indices) |indices| integer(indices, frame) else frame;
    }

    pub fn present(self: Masks, frame: usize, object: usize) bool {
        return if (self.active) |active| active.bytes[frame * self.objects + object] == 1 else true;
    }

    pub fn mask(self: Masks, frame: usize, object: usize) []const u8 {
        return self.masks.bytes[(frame * self.objects + object) * self.pixels ..][0..self.pixels];
    }

    pub fn match(self: Masks, other: Masks) !void {
        if (self.frames != other.frames or self.objects != other.objects or self.width != other.width or self.height != other.height) return error.MaskShapeMismatch;
        if ((self.ids == null or other.ids == null) and self.objects != 1) return error.IncompleteMaskMetadata;
        if (self.ids) |left| if (other.ids) |right| {
            if (!std.mem.eql(u8, left.bytes, right.bytes)) return error.ReferenceObjectMismatch;
        };
        for (0..self.frames) |frame| {
            if (self.frameIndex(frame) != other.frameIndex(frame)) return error.UnexpectedFrame;
            for (0..self.objects) |object| if (self.present(frame, object) != other.present(frame, object)) return error.ReferenceObjectMismatch;
        }
    }
};

fn integer(value: st.Tensor, index: usize) u64 {
    return std.mem.readInt(u64, value.bytes[index * 8 ..][0..8], .little);
}

fn tensor(dtype: st.DType, shape: []const usize, bytes: []const u8) st.Tensor {
    var out: st.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(out.shape[0..shape.len], shape);
    return out;
}

test "mask comparisons cannot pass changed IDs, anchors, lifecycle, or incomplete captures" {
    const masks = tensor(.u8, &.{ 2, 1, 2 }, &.{ 255, 0, 0, 255 });
    const ids = tensor(.u64, &.{1}, &.{ 42, 0, 0, 0, 0, 0, 0, 0 });
    const indices = tensor(.u64, &.{2}, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 });
    const active = tensor(.u8, &.{ 2, 1 }, &.{ 1, 1 });
    const complete = tensor(.u8, &.{1}, &.{1});
    const left = try Masks.init(masks, ids, indices, active, complete);
    try left.match(try Masks.init(masks, null, null, null, null));
    try std.testing.expectError(error.IncompleteMaskMetadata, Masks.init(masks, ids, indices, active, null));
    try std.testing.expectError(error.IncompleteMaskCapture, Masks.init(masks, ids, indices, active, tensor(.u8, &.{1}, &.{0})));
    const changed_id = tensor(.u64, &.{1}, &.{ 9, 0, 0, 0, 0, 0, 0, 0 });
    try std.testing.expectError(error.ReferenceObjectMismatch, left.match(try Masks.init(masks, changed_id, indices, active, complete)));
    const changed_activity = tensor(.u8, &.{ 2, 1 }, &.{ 1, 0 });
    try std.testing.expectError(error.ReferenceObjectMismatch, left.match(try Masks.init(masks, ids, indices, changed_activity, complete)));
    const reversed = tensor(.u64, &.{2}, &.{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try std.testing.expectError(error.UnexpectedFrame, left.match(try Masks.init(masks, ids, reversed, active, complete)));
}
