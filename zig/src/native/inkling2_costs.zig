//! Inkling 2 Mini's verify forward ms by total rows, timed at load: the shared draft pricing's cost table.
const std = @import("std");
const tf = @import("tensorfold");
const fl = tf.inkling2_metal;
const lanes = tf.lanes;

/// Rows timed: one window (1..16) and the wider forwards of several streams; the rule interpolates between them.
pub const widths = [_]u32{ 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64 };

pub const Table = struct {
    costs: [widths.len]lanes.config.Cost = undefined,
    timed: usize = 0,

    pub fn slice(t: *const Table) []const lanes.config.Cost {
        return t.costs[0..t.timed];
    }
};

/// Each width as one item on a throwaway state at position 0, the second of two runs timed; ids spread over the vocab.
pub fn measure(model: *fl.Model, io: std.Io, table: *Table) !void {
    var state = try fl.state.State.init(model.gpa, model.device, &model.config, .{ .capacity = 128, .rows = widths[widths.len - 1] });
    defer state.deinit();
    var ids: [widths[widths.len - 1]]u32 = undefined;
    const vocab: u64 = model.config.text.unpadded_vocab;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 40503 + 977) % vocab);
    table.timed = 0;
    for (widths) |rows| {
        var ms: f64 = 0;
        for (0..2) |run| {
            model.reset(&state);
            const t0 = std.Io.Clock.awake.now(io);
            try model.forward(&.{.{ .state = &state, .ids = ids[0..rows], .logit_rows = rows }});
            if (run == 1) ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        }
        table.costs[table.timed] = .{ .width = rows, .ms = ms };
        table.timed += 1;
    }
}
