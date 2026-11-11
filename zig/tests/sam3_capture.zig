const std = @import("std");
const sam = @import("sam3");

pub const Capture = struct {
    allocator: std.mem.Allocator,
    file: std.Io.File,
    ids: []u64,
    active: []u8,
    row: []u8,
    pixels: usize,
    size: sam.prompts.Size,
    frames: usize,
    start: usize,
    reverse: bool,
    written: usize = 0,
    data: usize,
    active_offset: usize,
    masks_offset: usize,
    complete_offset: usize,

    pub fn create(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, path: []const u8, ids: []const u64, size: sam.prompts.Size, frames: usize, start: usize, reverse: bool) !Capture {
        if (frames == 0 or ids.len == 0) return error.EmptyMaskCapture;
        if (reverse and frames - 1 > start) return error.FrameOutOfRange;
        for (ids, 0..) |id, i| for (ids[0..i]) |other| {
            if (id == other) return error.DuplicateObjectId;
        };
        const pixels = try size.pixels();
        const id_bytes = try std.math.mul(usize, ids.len, 8);
        const index_end = try std.math.add(usize, id_bytes, try std.math.mul(usize, frames, 8));
        const active_offset = try std.math.add(usize, index_end, 1);
        const active_count = try std.math.mul(usize, frames, ids.len);
        const masks_offset = try std.math.add(usize, active_offset, active_count);
        const row_size = try std.math.mul(usize, pixels, ids.len);
        const end = try std.math.add(usize, masks_offset, try std.math.mul(usize, frames, row_size));
        const shape = [_]usize{ frames, ids.len, size.height, size.width };
        const single_shape = [_]usize{ frames, size.height, size.width };
        const header = try std.json.Stringify.valueAlloc(a, .{
            .object_ids = .{ .dtype = "U64", .shape = .{ids.len}, .data_offsets = .{ 0, id_bytes } },
            .frame_indices = .{ .dtype = "U64", .shape = .{frames}, .data_offsets = .{ id_bytes, index_end } },
            .complete = .{ .dtype = "U8", .shape = .{1}, .data_offsets = .{ index_end, active_offset } },
            .active = .{ .dtype = "U8", .shape = .{ frames, ids.len }, .data_offsets = .{ active_offset, masks_offset } },
            .masks = .{ .dtype = "U8", .shape = if (ids.len == 1) &single_shape else &shape, .data_offsets = .{ masks_offset, end } },
        }, .{});
        defer a.free(header);
        const header_len = std.mem.alignForward(usize, header.len, 8);
        const prefix = try a.alloc(u8, 8 + header_len + masks_offset);
        defer a.free(prefix);
        @memset(prefix, 0);
        std.mem.writeInt(u64, prefix[0..8], header_len, .little);
        @memset(prefix[8..][0..header_len], ' ');
        @memcpy(prefix[8..][0..header.len], header);
        const data = 8 + header_len;
        for (ids, 0..) |id, i| std.mem.writeInt(u64, prefix[data + i * 8 ..][0..8], id, .little);
        for (0..frames) |i| {
            const index = if (reverse) start - i else try std.math.add(usize, start, i);
            std.mem.writeInt(u64, prefix[data + id_bytes + i * 8 ..][0..8], index, .little);
        }
        const own_ids = try a.dupe(u64, ids);
        errdefer a.free(own_ids);
        const active = try a.alloc(u8, active_count);
        errdefer a.free(active);
        @memset(active, 0);
        const row = try a.alloc(u8, row_size);
        errdefer a.free(row);
        const file = try dir.createFile(io, path, .{ .exclusive = true });
        errdefer file.close(io);
        try file.writeStreamingAll(io, prefix);
        return .{
            .allocator = a,
            .file = file,
            .ids = own_ids,
            .active = active,
            .row = row,
            .pixels = pixels,
            .size = size,
            .frames = frames,
            .start = start,
            .reverse = reverse,
            .data = data,
            .active_offset = active_offset,
            .masks_offset = masks_offset,
            .complete_offset = index_end,
        };
    }

    pub fn deinit(self: *Capture, io: std.Io) void {
        self.file.close(io);
        self.allocator.free(self.ids);
        self.allocator.free(self.active);
        self.allocator.free(self.row);
        self.* = undefined;
    }

    pub fn add(self: *Capture, io: std.Io, prediction: sam.session.Prediction, lifecycle: bool) !void {
        if (self.written == self.frames) return error.ExcessCapturedFrames;
        const index = if (self.reverse) self.start - self.written else self.start + self.written;
        if (prediction.frame_index != index) return error.UnexpectedFrame;
        const active = self.active[self.written * self.ids.len ..][0..self.ids.len];
        @memset(active, 0);
        @memset(self.row, 0);
        for (prediction.objects) |object| {
            const slot = std.mem.indexOfScalar(u64, self.ids, object.id) orelse return error.UnexpectedObject;
            if (active[slot] != 0) return error.DuplicateObjectId;
            if (object.mask.pixels.len != self.pixels or !std.meta.eql(object.mask.size, self.size)) return error.MaskShapeMismatch;
            for (object.mask.pixels) |value| if (value != 0 and value != 255) return error.NonBinaryMask;
            active[slot] = 1;
            @memcpy(self.row[slot * self.pixels ..][0..self.pixels], object.mask.pixels);
        }
        if (!lifecycle and std.mem.indexOfScalar(u8, active, 0) != null) return error.MissingObject;
        try self.file.writePositionalAll(io, self.row, self.data + self.masks_offset + self.written * self.row.len);
        self.written += 1;
    }

    pub fn finish(self: *Capture, io: std.Io) !void {
        if (self.written != self.frames) return error.MissingCapturedFrames;
        try self.file.writePositionalAll(io, self.active, self.data + self.active_offset);
        try self.file.writePositionalAll(io, &.{1}, self.data + self.complete_offset);
    }
};

test "capture requires every frame and object, preserving ID and reverse frame mapping" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    var capture = try Capture.create(a, io, dir.dir, "masks.safetensors", &.{ 42, 9 }, .{ .width = 2, .height = 1 }, 2, 7, true);
    defer capture.deinit(io);
    var white = [_]u8{ 255, 0 };
    var black = [_]u8{ 0, 0 };
    var objects = [_]sam.session.TrackedMask{
        .{ .id = 9, .mask = .{ .size = .{ .width = 2, .height = 1 }, .pixels = &white, .score = 1, .box = .{ 0, 0, 1, 1 } } },
        .{ .id = 42, .mask = .{ .size = .{ .width = 2, .height = 1 }, .pixels = &black, .score = 1, .box = .{ 0, 0, 0, 0 } } },
    };
    var prediction = sam.session.Prediction{ .allocator = a, .frame_index = 7, .objects = &objects };
    try std.testing.expectError(error.MissingCapturedFrames, capture.finish(io));
    prediction.objects = objects[0..1];
    try std.testing.expectError(error.MissingObject, capture.add(io, prediction, false));
    prediction.objects = &objects;
    objects[1].id = 9;
    try std.testing.expectError(error.DuplicateObjectId, capture.add(io, prediction, false));
    objects[1].id = 42;
    try capture.add(io, prediction, false);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 255, 0 }, capture.row);
    prediction.frame_index = 6;
    prediction.objects = objects[1..];
    try capture.add(io, prediction, true);
    try capture.finish(io);
    try std.testing.expectEqualSlices(u8, &.{ 1, 1, 1, 0 }, capture.active);
}
