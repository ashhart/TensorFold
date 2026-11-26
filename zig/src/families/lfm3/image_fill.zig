//! A prompt pass's images: opened once for the pass, spliced by every chunk, freed when it closes.
const std = @import("std");
const lanes = @import("lanes");

/// The engine's chunk, narrowed by a host cap: `max_rows == 0` leaves the engine's cut.
pub fn rows(chunk: usize, max_rows: u32) usize {
    return if (max_rows == 0) chunk else @min(chunk, max_rows);
}

/// A pass's provider and its images, generic over both so the chunk path runs without a device.
pub fn Fill(comptime Provider: type, comptime Images: type) type {
    return struct {
        provider: ?Provider = null,
        stream: *const lanes.Stream,
        images: ?Images = null,
        open: bool = false,

        const Self = @This();

        /// The pass's images from `from` on: the provider projects them here, before the first chunk.
        pub fn begin(self: *Self, from: usize) !void {
            std.debug.assert(!self.open);
            self.open = true;
            if (self.provider) |p| self.images = try p.begin(p.ctx, self.stream, from);
        }

        /// One chunk's images: the pass's own, never a new projection.
        pub fn chunk(self: *Self, at: usize, end: usize) !?Images {
            _ = at;
            _ = end;
            std.debug.assert(self.open);
            return self.images;
        }

        /// The pass is over (its last chunk, a cancel, a failure or the stream leaving): its rows go.
        pub fn close(self: *Self) void {
            if (!self.open) return;
            self.open = false;
            if (self.images != null) self.provider.?.end(self.provider.?.ctx, self.stream);
            self.images = null;
        }
    };
}

test {
    _ = @import("image_fill_test.zig");
}
