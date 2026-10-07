//! A GPU round's head depth: all its levels while those past 4 bring a tenth of the tokens, else 4 until rounds keep 5+.

pub const Depth = struct {
    levels: usize,
    most: usize,
    fixed: bool,
    kept: [8]u32 = @splat(0), // the last rounds' kept tokens at the current depth
    seen: usize = 0,

    const shallow = 4;

    pub fn observe(d: *Depth, kept: u32) void {
        d.kept[d.seen % d.kept.len] = kept;
        d.seen += 1;
    }

    pub fn next(d: *Depth) usize {
        if (d.fixed or d.seen < d.kept.len or d.most <= shallow) return d.levels;
        var all: u32 = 0;
        var past: u32 = 0; // tokens the levels past 4 brought
        var long: u32 = 0; // rounds that kept every token of a 4-level window
        for (d.kept) |k| {
            all += k;
            past += k -| (shallow + 1);
            long += @intFromBool(k >= shallow + 1);
        }
        const deep = d.levels > shallow;
        const switch_ = if (deep) past * 10 < all else long * 2 > d.kept.len;
        if (switch_) {
            d.levels = if (deep) shallow else d.most;
            d.seen = 0;
        }
        return d.levels;
    }
};
