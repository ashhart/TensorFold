//! Public Qwen native load options, context limits and owned backend results.
const std = @import("std");
const lanes = @import("lanes");
const E = @import("cuda_engine.zig");
const vision = @import("cuda_vision.zig");
const reuse = @import("cuda_reuse.zig");
const cuda_rope = @import("cuda_rope.zig");
const kv_pool = @import("core").kv_pool;
const be = lanes.backend;

/// The server's reading of the room left now; null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `room` and `drafter` size an EXL3 checkpoint's load: its embedding rows go to host memory when the card is short.
pub const Options = struct { context: usize, drafts: bool, cache_features: bool = false, lanes: u32 = 1, segments: usize = 1, slide: bool = false, vision: ?vision.Limits = null, kv8: bool = false, prompt_int8: bool = false, precision: ?enum { full } = null, rope_yarn: ?f64 = null, room: ?Room = null, drafter: ?[]const u8 = null };

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    learns: ?*E.Engine = null,
    learn_bytes: usize = 0,
    backend: be.Backend,
    facts: lanes.Model,
    rows: u32,
    prompt_rows: u32 = E.max_rows,
    /// Device bytes each admitted stream allocates (recurrences, convolution tails, keys and values).
    stream_bytes: usize,
    free_streams: u32 = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// Called with the admitted stream count: that many caches allocated and touched now, as memory commits on touch.
    reserve: *const fn (*anyopaque, u32) anyerror!void,
    /// --vision: the tower the server's image requests reach (native/cuda_vision_offer.zig).
    vision: ?*vision.Vision = null,
    /// What the prompt cache's saves copy: the engine and the streams' caches.
    target: *reuse.Target,
    /// TF_QWEN_KV_PAGES: the streams beside the first take pages of one pool (null: a window each).
    layout: ?kv_pool.Layout = null,
    /// TF_QWEN_KV_SHARE: the family keeps prompts as pool pages itself; the host makes no prompt cache of its own.
    own_prompt_cache: bool = false,
    /// TF_QWEN_KV_SHARE or TF_QWEN_KV_PAGES_ALL: every stream runs on the pool's pages; the engine's window sits idle.
    all_on_pages: bool = false,
    /// TF_QWEN_KV_PAGES_ALL=1: the engine's own stream takes pages too, a window of them set aside by the plan.
    engine_paged: bool = false,
    /// An integrated GPU: kept states in device memory on the host budget (Target.device), the startup line says so.
    device_snaps: bool = false,
    /// A Clef checkpoint's readouts (the engine then answers only those); null for every other checkpoint.
    readouts: ?lanes.readout.Info = null,
};

/// The window --context may ask for: the config's, or the YaRN window its block or --rope-yarn makes (cuda_rope.zig).
pub fn contextCeiling(a: std.mem.Allocator, io: std.Io, dir: []const u8, flag: ?f64) !i64 {
    return @intCast((try cuda_rope.resolveDir(a, io, dir, flag)).ceiling);
}
