//! DFlash2's taps on the target's forward: after each tapped layer, its rows' 4-stream mean into one tap slot.
const cuda = @import("cuda");

/// The target layers DFlash2 reads (incoai/GLM-5.3-Flash-DFlash2's target_layer_ids), in tap-slot order.
pub const layers = [_]usize{ 5, 14, 24, 33, 42 };

/// Where a forward's taps go: rows [R][5 * hidden] bf16 at `out`, through glm_dflash's df_tap.
pub const Taps = struct {
    f: cuda.Function,
    out: u64,

    /// After layer `j` of an `R`-row pass on residual `x` (bf16 [R][4][4096]): its tap slot, if `j` is tapped.
    pub fn at(t: Taps, s: cuda.Stream, j: usize, x: u64, R: usize) !void {
        const slot: i32 = for (layers, 0..) |l, i| {
            if (l == j) break @intCast(i);
        } else return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(t.out);
        a.add(slot);
        a.add(@as(i32, 0)); // the mean of the 4 streams (df_tap's mode 0)
        try cuda.launch.launch(t.f, .{ .grid = .{ .x = @intCast(R), .y = 4096 / 1024 }, .block = .{ .x = 256 } }, s, &a);
    }
};
