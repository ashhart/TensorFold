//! Secondary slots gather target taps in bounded drafter-sized pieces.
pub fn absorb(context: anytype, rows: usize) !void {
    var first: usize = 0;
    while (first < rows) {
        const count = @min(128, rows - first);
        try context.gather(first, count);
        first += count;
    }
}
