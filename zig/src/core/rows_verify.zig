//! TF_IMPRINT_VERIFY=1: a state just written is read back from disk and its image compared, piece by piece, with RAM's.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const image = @import("rows_image.zig");
const read = @import("rows_read.zig");
const Rows = @import("rows_store.zig").Rows;

const piece: u64 = 16 << 20;

/// Disk state `i` read into a new state and compared with `saved`, the kept state it was written from; a line logged.
pub fn state(r: *Rows, i: usize, saved: pc.Saved) void {
    const at = r.states.items[i].at;
    const quiet = @import("builtin").is_test;
    compare(r, i, saved) catch |err| {
        r.counts.differed += 1;
        if (!quiet) std.log.err("imprint verify: {d} tokens read back differ or failed ({s})", .{ at, @errorName(err) });
        return;
    };
    r.counts.verified += 1;
    if (!quiet) std.log.info("imprint verify: {d} tokens read back byte-equal to the kept state ({d} so far)", .{ at, r.counts.verified });
}

fn compare(r: *Rows, i: usize, saved: pc.Saved) !void {
    const at = r.states.items[i].at;
    const parts = try r.parts(r.gpa, at);
    defer r.gpa.free(parts);
    const back = try read.read(r, i);
    defer r.family.vtable.drop(r.family.ptr, back);
    const a = try r.gpa.alloc(u8, piece);
    defer r.gpa.free(a);
    const b = try r.gpa.alloc(u8, piece);
    defer r.gpa.free(b);
    var off: u64 = 0;
    for (parts) |p| {
        defer off += p.bytes;
        var done: u64 = 0;
        while (done < p.bytes) {
            const n = @min(piece, p.bytes - done);
            const g = [_]image.Range{.{ .off = off + done, .len = n }};
            try r.img.gather(r.family.ptr, saved, &g, a[0..@intCast(n)]);
            try r.img.gather(r.family.ptr, back, &g, b[0..@intCast(n)]);
            if (!std.mem.eql(u8, a[0..@intCast(n)], b[0..@intCast(n)])) return error.ImageDiffers;
            done += n;
        }
    }
}
