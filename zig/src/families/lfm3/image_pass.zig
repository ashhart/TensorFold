//! A prompt pass's images: each distinct one through the tower once, at the rows its chunks splice.
const std = @import("std");

/// An image's projected rows in the pass buffer, at a prompt position.
pub const Span = struct { at: u32, rows: u32, source: u32 };

/// The rows a pass resumed at `from` needs: each distinct image whose span ends after it, once.
pub fn rows(images: anytype, from: usize) usize {
    var total: usize = 0;
    for (images, 0..) |im, i| {
        if (@as(usize, im.at) + im.tokens <= from) continue;
        var seen = false;
        if (im.hash != 0) for (images[0..i]) |prior| {
            if (@as(usize, prior.at) + prior.tokens <= from or prior.hash != im.hash) continue;
            seen = true;
            break;
        };
        if (!seen) total += im.tokens;
    }
    return total;
}

/// Projects each pending distinct image once, in prompt order; a repeated image reuses the first one's rows.
pub fn encode(gpa: std.mem.Allocator, images: anytype, from: usize, spans: *std.ArrayList(Span), project: anytype) !void {
    var firsts: std.ArrayList(First) = .empty;
    defer firsts.deinit(gpa);
    try firsts.ensureTotalCapacity(gpa, images.len);
    try spans.ensureTotalCapacity(gpa, images.len);
    var row: u32 = 0;
    for (images) |im| {
        if (@as(usize, im.at) + im.tokens <= from) continue;
        var source: ?u32 = null;
        if (im.hash != 0) for (firsts.items) |first| if (first.hash == im.hash) {
            source = first.row;
            break;
        };
        if (source == null) {
            source = row;
            try project.project(im, row);
            row += im.tokens;
            firsts.appendAssumeCapacity(.{ .hash = im.hash, .row = source.? });
        }
        spans.appendAssumeCapacity(.{ .at = im.at, .rows = im.tokens, .source = source.? });
    }
}

const First = struct { hash: u64, row: u32 };

test {
    _ = @import("image_pass_test.zig");
}
