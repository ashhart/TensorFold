//! A prompt's picture rows: the tower's BF16 output for each image, and the part of it a prefill chunk takes.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const state = @import("state.zig");

/// The tower behind the backend: one image's rows as a new BF16 buffer [tokens, hidden], the caller's to free.
pub const Encoder = struct {
    ctx: *anyopaque,
    encode: *const fn (ctx: *anyopaque, image: lanes.stream.Image) anyerror!mtl.Buffer,
};

/// The SHA-256 of a picture's BF16 rows as lowercase hex, the tower output the splice takes.
pub fn digestHex(rows: []const u8) [64]u8 {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(rows, &sum, .{});
    return std.fmt.bytesToHex(sum, .lower);
}

pub const Rows = struct {
    gpa: std.mem.Allocator,
    whole: []fwd.ImageRows,
    made: usize = 0,
    part: []fwd.ImageRows,

    /// Encodes each picture of `stream` at or after row `first` (a restored prefix already holds the earlier ones).
    pub fn init(gpa: std.mem.Allocator, encoder: ?Encoder, stream: *const lanes.Stream, first: usize) !Rows {
        const prompt_len = stream.prompt().len;
        const whole = try gpa.alloc(fwd.ImageRows, stream.images.len);
        const part = gpa.alloc(fwd.ImageRows, stream.images.len) catch |err| {
            gpa.free(whole);
            return err;
        };
        var self: Rows = .{ .gpa = gpa, .whole = whole, .part = part };
        errdefer self.deinit();
        for (stream.images) |im| {
            if (im.at > prompt_len or im.tokens > prompt_len - im.at) return error.BadInklingImageRows;
            if (im.at + im.tokens <= first) continue;
            const e = encoder orelse return error.InklingVisionDisabled;
            if (im.tokens != @as(u64, im.gh) * im.gw) return error.BadInklingImageGrid;
            const buffer = try e.encode(e.ctx, im);
            self.whole[self.made] = .{ .row = im.at, .rows = im.tokens, .buf = buffer };
            self.made += 1;
            if (buffer.length() < @as(usize, im.tokens) * state.hidden * 2) return error.BadInklingImageRows;
            const bytes = buffer.slice(u8, @as(usize, im.tokens) * state.hidden * 2);
            std.log.info("inkling2 image tower: picture={x:0>16} rows={d} sha256={s}", .{ im.hash, im.tokens, digestHex(bytes) });
        }
        return self;
    }

    /// The pictures' rows inside prompt positions [start, end), as rows of that chunk.
    pub fn chunk(self: *Rows, start: usize, end: usize) []const fwd.ImageRows {
        var count: usize = 0;
        for (self.whole[0..self.made]) |im| {
            const from = @max(start, im.row);
            const to = @min(end, im.row + im.rows);
            if (from >= to) continue;
            self.part[count] = .{ .row = @intCast(from - start), .rows = @intCast(to - from), .buf = im.buf, .off = im.off + (from - im.row) * state.hidden * 2 };
            count += 1;
        }
        return self.part[0..count];
    }

    pub fn deinit(self: *Rows) void {
        for (self.whole[0..self.made]) |im| im.buf.deinit();
        self.gpa.free(self.whole);
        self.gpa.free(self.part);
    }
};
