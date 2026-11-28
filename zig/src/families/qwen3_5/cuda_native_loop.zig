//! SenseNova's text-only CUDA entry, using the shared Qwen engine.
const std = @import("std");
const cuda = @import("cuda");
const native = @import("cuda_native.zig");

pub const model_type = "sensenova_loop";
pub const formats: []const []const u8 = &.{"bf16"};
pub const default_context = native.default_context;
pub const max_segments = native.max_segments;
pub const prompt_rows = native.prompt_rows;
pub const takes_drafter = false;
pub const drafter_refusal = "--drafter: SenseNova Loop on CUDA supports plain decoding only";
pub const explains_open = true;
pub const Options = native.Options;
pub const contextCeiling = native.contextCeiling;
pub const window_rows = native.window_rows;
pub const textWeightBytes = native.textWeightBytes;
pub const visionBytes = native.visionBytes;
pub const kvHostRoom = native.kvHostRoom;
pub const kvHostRefusal = native.kvHostRefusal;
pub const attach = native.attach;
pub const pagedLayout = native.pagedLayout;
pub const fits = native.fits;
pub const snap_bytes = native.snap_bytes;
pub const snap_save = native.snap_save;
pub const snap_restore = native.snap_restore;
pub const snap_drop = native.snap_drop;
pub const snap_spare = native.snap_spare;
pub const snap_trim = native.snap_trim;
pub const snap_reuses = native.snap_reuses;
pub const snap_host = native.snap_host;
pub const snap_lookahead = native.snap_lookahead;
pub const snap_grid = native.snap_grid;
pub const warm_quiet = native.warm_quiet;
pub const warm_usable = native.warm_usable;
pub const request_starts = native.request_starts;
pub const request_starts_switch = native.request_starts_switch;
pub const explain = native.explain;

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, options: Options) !native.Loaded {
    if (options.loops) |count| if (count < 1 or count > 4) return error.BadLoops;
    if (options.drafter != null) return error.SenseNovaDrafter;
    if (options.vision != null) return error.SenseNovaVision;
    if (options.slide) return error.SenseNovaLearning;
    if (options.tp != 1 or options.rank != 0) return error.SenseNovaRanks;
    var plain = options;
    plain.drafts = false;
    var loaded = try native.open(gpa, io, ctx, dir, kernels, plain);
    loaded.facts.copy_drafts = false;
    loaded.facts.context_copies = false;
    loaded.facts.root_copies = false;
    return loaded;
}
