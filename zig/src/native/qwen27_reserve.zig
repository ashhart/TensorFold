//! What a Qwen3.8-27B-class start reserves beside its weights: `serve` charges it, and `info` prints it unloaded.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const memory = q.prompt_memory;
const Config = q.config.Config;

/// The rows the deferred load gives the frame before startup widens it.
pub const loaded_rows: u32 = 128;

const env: @import("sys").environment.Process = .{};

/// A development cap on the working set startup counts.
pub fn cap() ?[]const u8 {
    return env.get("TF_QWEN27_WORKING_SET_BYTES");
}

/// The prompt fit startup charges: `current_rows` is the frame the model holds, `left` the room beside it.
pub fn fit(c: Config, capacity: u32, window: u32, streams: u32, units: bool, ring: usize, planes: usize, current_rows: u32, left: usize) memory.Fit {
    return .{ .config = c, .context = capacity + window, .slots = @max(streams, 1), .shared = units, .ring = ring, .planes = planes, .current_rows = current_rows, .left = left };
}

fn rowsValue() ?[]const u8 {
    return env.get("TF_PROMPT_ROWS");
}

/// The prompt rows asked for: a valid TF_PROMPT_ROWS, else the chip's default.
pub fn requested(generation: u32, moe: bool) u32 {
    return q.prompt_rows.requested(generation, moe, rowsValue());
}

/// The prompt rows startup takes: the request when it fits (always when named), never under the window.
pub fn rows(f: memory.Fit, generation: u32, window: u32) !u32 {
    return @max(window, try f.choose(requested(generation, f.config.moe()), q.prompt_rows.override(rowsValue()) != null));
}

pub const Budget = struct {
    context: u32,
    streams: u32,
    rows: u32,
    /// The streams that fit beside the weights and the margin; 0 when not one does.
    slots: u32,
    /// Weights, the widened frame, and the charged streams' KV, state and prompt scratch.
    total: usize,
    working_set: usize,
};

/// Startup's charge for `dir` on this Mac, no drafter or images; `weights` bytes stand in for the loaded model.
pub fn budget(a: std.mem.Allocator, io: std.Io, dir: []const u8, weights: u64, context: u32, streams: u32, fixed: bool) !Budget {
    const path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    const c = try q.config.parse(a, text, null);
    if (context == 0 or context > c.max_position) return error.ContextFull;
    const dev = try mtl.Device.init();
    defer dev.deinit();
    const units = dev.tensorUnits();
    const window = try @import("qwen27_lanes.zig").windowRows(units);
    const current = @max(loaded_rows, q.model.promptRows(c, 0));
    const held = @as(usize, @intCast(weights)) + memory.frameBytes(c, current);
    const f = fit(c, context, window, streams, units, 0, q.taps.layers.len, current, try memory.room(dev.maxWorkingSet(), held, cap()));
    const r = try rows(f, dev.generation(), window);
    const slots = f.streams(r, fixed) catch |err| if (err == error.TooManyStreams) 0 else return err;
    return .{ .context = context, .streams = streams, .rows = r, .slots = slots, .total = held + try f.cost(r, if (slots == 0) streams else slots), .working_set = try memory.workingSet(dev.maxWorkingSet(), cap()) };
}
