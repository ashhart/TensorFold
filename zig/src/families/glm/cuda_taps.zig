//! DFlash2's taps on the target's forward: after each tapped layer, its rows' 4-stream mean into one tap slot.
const cuda = @import("cuda");

/// The target layers DFlash2 reads (incoai/GLM-5.3-Flash-DFlash2's target_layer_ids), in tap-slot order.
pub const layers = [_]usize{ 5, 14, 24, 33, 42 };

/// df_tap's D: the hidden size of a residual stream and of a tap slot.
pub const hidden = 4096;

/// Layer `j`'s tap slot, or null when DFlash2 does not read layer `j`.
pub fn slot(j: usize) ?i32 {
    for (layers, 0..) |l, i| if (l == j) return @intCast(i);
    return null;
}

/// Where a forward's taps go: rows [R][5 * hidden] bf16 at `out`, through glm_dflash's df_tap.
pub const Taps = struct {
    f: cuda.Function,
    out: u64,

    /// After layer `j` (nothing off the tapped layers): rows lo..lo + n of residual `x` (bf16 [rows][4][hidden]).
    pub fn at(t: Taps, s: cuda.Stream, j: usize, x: u64, lo: usize, n: usize) !void {
        const k = slot(j) orelse return;
        if (n == 0) return;
        var a: cuda.Args = .{};
        a.add(x + lo * 4 * hidden * 2);
        a.add(t.out + lo * layers.len * hidden * 2);
        a.add(k);
        a.add(@as(i32, 0)); // the mean of the 4 streams (df_tap's mode 0)
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(n), .y = hidden / 1024 }, .block = .{ .x = 256 } }, s, &a);
    }
};
