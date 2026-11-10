//! States the rows store drops as it opens, counted by cause and said one line a cause, then the blocks it removes.
const pc = @import("prompt_cache.zig");

/// The cause of a state that names a block no file holds (a block deleted, or lost before it was written whole).
pub const missing = error.RowsBlockMissing;

const unnamed = " (a state's file is gone, or a write was cut short)";

pub const Dropped = struct {
    causes: [8]anyerror = undefined,
    counts: [8]u32 = @splat(0),
    blocks: usize = 0, // blocks no kept state names, removed as the store opens

    /// One state dropped for `err`; past eight causes, a new one is said at once.
    pub fn add(d: *Dropped, err: anyerror) void {
        for (&d.causes, &d.counts) |*c, *n| if (n.* == 0 or c.* == err) {
            c.* = err;
            n.* += 1;
            return;
        };
        say(err, 1);
    }

    /// Each cause's line, in `rows_read.bad`'s voice, then the blocks'; then the counts start again.
    pub fn flush(d: *Dropped) void {
        for (d.causes, d.counts) |c, n| if (n > 0) say(c, n);
        if (d.blocks == 1) pc.note("imprint: a disk block no kept state names is removed" ++ unnamed, .{});
        if (d.blocks > 1) pc.note("imprint: {d} disk blocks no kept state names are removed" ++ unnamed, .{d.blocks});
        d.counts = @splat(0);
        d.blocks = 0;
    }
};

fn say(err: anyerror, n: u32) void {
    if (err == missing) {
        if (n == 1) pc.note("imprint: a disk state is missing one of its blocks; it is forgotten", .{}) else pc.note("imprint: {d} disk states are missing one of their blocks; they are forgotten", .{n});
    } else if (n == 1) {
        pc.note("imprint: a disk state failed its check ({s}); it is forgotten", .{@errorName(err)});
    } else pc.note("imprint: {d} disk states failed their check ({s}); they are forgotten", .{ n, @errorName(err) });
}
