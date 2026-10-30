//! An EXL3 checkpoint's bf16 embedding rows in mapped host memory, only when the card has no room for them.
const std = @import("std");
const cuda = @import("cuda");

/// What a load has to leave room for: the server's room before it, one stream's window, and the drafter's directory.
pub const Fit = struct { room: u64, context: usize, drafter: ?[]const u8 };

/// The load's other bytes (kernels, kit, startup costs): ~112 MiB on a 3090 Ti; less refuses as main does, more maps.
const others: u64 = 96 << 20;

/// Whether the rows go to host memory: a discrete card whose room holds the device bytes and one stream only so.
pub fn mapped(unified: bool, room: u64, device: u64, stream: u64, beside: u64) bool {
    return !unified and room < device +| stream +| beside;
}

/// The weights' bytes on the card; an EXL3 checkpoint's embedding rows' host memory is made when `fit`'s room is short.
pub fn measure(e: anytype, io: std.Io, ctx: *const cuda.Context, ck: anytype, fit: ?Fit) !usize {
    e.host_embed = null;
    if (!e.g.exl3) return @import("cuda_weights.zig").total(e.g);
    const x3w = @import("cuda_exl3_weights.zig");
    const bytes = try x3w.measure(e, ck);
    const f = fit orelse return bytes;
    const unified = (try ctx.attribute(.integrated)) != 0;
    const stream = e.streamBytes(f.context);
    const beside = if (f.drafter) |dir| try drafter(e, io, dir) else 0;
    if (!mapped(unified, f.room, bytes + try e.scratch(.measure) + others, stream, beside)) return bytes;
    e.host_embed = try cuda.HostBuffer.allocMapped(e.d, e.g.vocab * e.g.hidden * 2);
    std.log.info("qwen3_5: embedding rows read from host memory: no room beside a {d}-token stream", .{f.context});
    return x3w.measure(e, ck);
}

/// A DFlash2 drafter's device bytes: its weights (bf16 linears made 4-bit, 5/16) and one stream's ring, taps and frame.
fn drafter(e: anytype, io: std.Io, dir: []const u8) !u64 {
    const draft = @import("cuda_draft.zig");
    const c = draft.Config.read(e.gpa, io, dir, e.g) catch return 0; // not a DFlash2 drafter: its open says why
    var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (try it.next(io)) |f| {
        if (std.mem.endsWith(u8, f.name, ".safetensors")) total += (try d.statFile(io, f.name, .{})).size;
    }
    return total / 16 * 5 + try draft.Draft.streamBytesFor(c, e.g.hidden);
}

/// The rows copied to the pinned host memory the device reads in place.
pub fn rows(host: cuda.HostBuffer, bytes: []const u8) !u64 {
    if (bytes.len != host.bytes.len) return error.EmbeddingShape;
    @memcpy(host.bytes, bytes);
    return host.device();
}

test "the rows stay on the card unless a discrete card's room is short of the device bytes and one stream" {
    try std.testing.expect(!mapped(false, 20, 10, 5, 5));
    try std.testing.expect(mapped(false, 19, 10, 5, 5));
    try std.testing.expect(!mapped(true, 19, 10, 5, 5)); // unified memory: host memory is the same memory
    try std.testing.expect(mapped(false, 0, std.math.maxInt(u64), 1, 1));
}
