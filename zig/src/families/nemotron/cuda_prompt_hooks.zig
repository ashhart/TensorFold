//! The hooks a prompt pass takes: a cancel check, the rows that keep states, and taps on each chunk's hidden rows.

/// Asked before each prompt chunk (or segmented call): true stops the prompt with error.Cancelled.
pub const Cancel = struct {
    ptr: *anyopaque,
    check: *const fn (ptr: *anyopaque) bool,
    yields: bool = false, // a background pass's hook: it steps aside for real work, so its chunks stay short

    pub fn now(c: ?Cancel) bool {
        const x = c orelse return false;
        return x.check(x.ptr);
    }

    /// A background pass: chunks of state.yield_rows, one in flight, and what it prefilled kept when it stops.
    pub fn background(c: ?Cancel) bool {
        return if (c) |x| x.yields else false;
    }
};

/// Rows where a prompt pass keeps its state (ascending): one inside a chunk is called before it, one at its end after.
pub const Keep = struct {
    at: []const u32,
    ctx: *anyopaque,
    call: *const fn (*anyopaque, u32) void,
    cut: bool = false, // every row ends a chunk instead: the reference the marks test compares the armed states to
};

/// Each prompt chunk's rows as it ends: `rows` normed final hidden states at `hidden`, from prompt row `row`.
pub const Taps = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, hidden: u64, row: usize, rows: usize) anyerror!void };
