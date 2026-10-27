//! Flash Next on our Metal runtime: the Python engine's recorded kernels (a dump from tools/zig/flashnext_dump.py)...
const std = @import("std");
const platform = @import("sys");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const roles_gen = @import("roles_gen.zig");
const segments = @import("../../core/segments.zig");
const tpm = @import("tp.zig");
const mark_slots = @import("marks.zig");
const verdict_kernels = @import("verdict.zig");
pub const split = @import("split.zig");
pub const dense = @import("dense.zig");
pub const lane_w = @import("lane_w.zig");
pub const ple_scale = @import("ple_scale.zig");
pub const rows_load = @import("rows_load.zig");
pub const variants = @import("variants.zig");
pub const prompt_w = @import("prompt_w.zig");
pub const engine_widths = @import("engine_widths.zig");
pub const gdn_step = @import("gdn.zig");
pub const Tp2 = tpm.Tp2;
pub const frags = @import("../../core/frags.zig");
pub const vis_prompt = @import("vis_prompt.zig");

pub const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

pub const D = @import("flashnext_memory").prompt.D;
pub const WIDE = @import("flashnext_memory").prompt.WIDE;
pub const LAYERS = 48;
pub const VOCAB = 248320;
pub const window_config = @import("window_config.zig");
pub const WindowSettings = window_config.Settings;
pub const TRAINED = window_config.trained;
pub const CAP_MAX = window_config.maximum;
/// The default capacity for callers that do not request a long window.
pub const CAP = TRAINED;
pub const PLE_TAIL = @import("flashnext_memory").prompt.PLE_TAIL;
pub const GROUPS = 8;
pub const MAXR = 16; // a decode window's rows at most
const HEAD_TILES = VOCAB / 32; // the vocabulary head's 32-column tiles
pub const CS_ROW = @import("flashnext_memory").prompt.CS_ROW; // a DeltaNet conv state row (bytes)
pub const SO_ROW = 48 * 128 * 128 * 4; // a DeltaNet recurrent state row (bytes)

pub const Buf = struct {
    b: mtl.Buffer,
    off: usize = 0,

    /// The same buffer `bytes` further on.
    pub fn at(x: Buf, bytes: usize) Buf {
        return .{ .b = x.b, .off = x.off + bytes };
    }
};
pub const Entry = struct { fd: platform.c.fd_t, at: usize, len: usize };
pub const Variant = struct { inputs: [][]const u8, outputs: [][]const u8, meta: [][]const u8, pipe: mtl.Pipeline, file: []const u8 = "", name: []const u8 = "", text: []const u8 = "" };
pub const Site = struct { v: *Variant, grid: mtl.Size, tg: mtl.Size };

pub const glue_source = @import("replay_glue_source.zig").text;

/// Routed and shared experts at full width (FZ_XNEW=1): a 6-bit group read as six aligned words, 16 lanes a gate/up...
const xnew_source = @import("replay_expert_source.zig").text;

/// The hyper-connection's up projection for windows of 3-8 rows (`Run.hc_up`), compiled after the recorded...
const hc_up_source = @import("replay_hc_source.zig").text;

pub fn readAll(fd: platform.c.fd_t, dest: []u8, at: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = platform.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return error.ShortRead;
        done += @intCast(n);
    }
}

pub const Run = @import("replay_run.zig").Run;

/// Past 512 complete 4-key blocks the attention reads each row's 512 best blocks and its tail: the recorded pool,...
pub const RING_WORDS = @import("replay_buffers.zig").RING_WORDS;
pub const AR_HLEN = @import("replay_buffers.zig").AR_HLEN;
pub const AR_WORDS = @import("replay_buffers.zig").AR_WORDS;
pub const AR_POS = @import("replay_buffers.zig").AR_POS;
pub const AR_NK = @import("replay_buffers.zig").AR_NK;
pub const AR_KV = @import("replay_buffers.zig").AR_KV;
pub const AR_ABS_POS = @import("replay_buffers.zig").AR_ABS_POS;
pub const AR_ABS_NK = @import("replay_buffers.zig").AR_ABS_NK;
pub const AR_ABS_KV = @import("replay_buffers.zig").AR_ABS_KV;
pub const AR_CHAIN = @import("replay_buffers.zig").AR_CHAIN;
pub const AR_CHAIN_STRIDE = @import("replay_buffers.zig").AR_CHAIN_STRIDE;
pub const TOP = @import("replay_buffers.zig").TOP;
pub const KW = @import("replay_buffers.zig").KW;
pub const CATCH = @import("replay_buffers.zig").CATCH;
pub const catchSlot = @import("replay_buffers.zig").catchSlot;
pub const Select = @import("replay_buffers.zig").Select;
pub const SEL_POOLED = @import("replay_buffers.zig").SEL_POOLED;
pub const POOL_BLOCKS = @import("replay_buffers.zig").POOL_BLOCKS;
pub const GSelect = @import("replay_buffers.zig").GSelect;
pub const Hc = @import("replay_buffers.zig").Hc;
pub const Lane = @import("replay_buffers.zig").Lane;
pub const Layer = @import("replay_buffers.zig").Layer;
pub const hcOf = @import("replay_buffers.zig").hcOf;
pub const laneOf = @import("replay_buffers.zig").laneOf;
pub const Ple = @import("replay_buffers.zig").Ple;
pub const Tmp = @import("replay_buffers.zig").Tmp;
pub const ropeBuf = @import("replay_buffers.zig").ropeBuf;
pub const yarn = @import("replay_buffers.zig").yarn;
pub const f32Buf = @import("replay_buffers.zig").f32Buf;
pub const i32Buf = @import("replay_buffers.zig").i32Buf;
pub const Slot = @import("replay_buffers.zig").Slot;
pub const draftCount = @import("replay_buffers.zig").draftCount;
pub const Mtp = @import("replay_buffers.zig").Mtp;
pub const MARKS = @import("replay_buffers.zig").MARKS;
pub const Marks = @import("replay_buffers.zig").Marks;
const linearIndex = @import("replay_buffers.zig").linearIndex;
test {
    _ = @import("replay_buffers.zig");
    _ = tpm; // tp.zig's tests: the window layout and the host service's wraps
    _ = @import("drafts.zig"); // the depth rule's tests
    _ = @import("replay_held_test.zig"); // Run.release and the held list (#732)
    _ = vis_prompt; // an image prompt's frame, its spans and its cache key
    _ = @import("grammar.zig"); // structured output: masked rows on the GPU against the host draw
}
pub const Model = @import("replay_model.zig").Model;

/// FZ_PROFILE's knock-out bit for speed-up mode's exchanges: the bit after `Run.class`'s eight.
pub const TP_CLASS: u32 = 1 << 8;

pub const DepthRule = @import("drafts.zig").DepthRule; // drafts a round from recent landing
pub const copyDrafts = @import("drafts.zig").copyDrafts;

/// Prompt chunks of up to PMAX rows: projections on the 6-bit tensor-unit kernels, experts sorted by expert and...
pub const PMAX = @import("flashnext_memory").prompt.PMAX; // the prompt buffers' rows; `step` cuts the chunks
/// A prompt selector's scores: PMAX rows by the trained window's blocks (2 GiB). Past it chunks take fewer rows.
pub const SC_WORDS: usize = PMAX * (TRAINED / 4);

/// The rows a prompt chunk from `pos` may take (at most `want`): every row scores the blocks before its end.
pub fn chunkRows(pos: usize, want: usize) usize {
    var rows = want;
    while (rows > 64 and rows * ((pos + rows) / 4 + 1) > SC_WORDS) rows /= 2;
    return rows;
}
pub const Prompt = @import("replay_prompt.zig").Prompt;

pub fn jsonInt(v: std.json.Value) i64 {
    return switch (v) {
        .integer => |x| x,
        .float => |x| @intFromFloat(x),
        else => 0,
    };
}

test "YaRN x4 frequencies and scale: transformers' table for base 1e7 over the trained 262,144" {
    const y = yarn(1e7, 4, TRAINED);
    const want = [_]f32{ 1.0, 0.6042963862419128, 0.36517414450645447, 0.22067339718341827, 0.1333521455526352, 0.08058421313762665, 0.04869675263762474, 0.02942727319896221, 0.017782794311642647, 0.010746078565716743, 0.006493816617876291, 0.003924189601093531, 0.0023713738191872835, 0.0014330126577988267, 0.0008659643353894353, 0.00047423981595784426, 0.00025693507632240653, 0.00013734975073020905, 7.217386882985011e-05, 3.707224823301658e-05, 1.844922371674329e-05, 8.759770025790203e-06, 3.849816039291909e-06, 2.3264301489689387e-06, 1.405853367941745e-06, 8.495520660289912e-07, 5.133812237545499e-07, 3.102344408034696e-07, 1.8747356023141037e-07, 1.1328959459433463e-07, 6.846049416253663e-08, 4.1370427084075345e-08 };
    for (want, y.freq) |w, g| try std.testing.expectApproxEqRel(w, g, 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 1.138629436111989), y.scale, 1e-6);
}

test "chunk rows shrink only past the trained window" {
    try std.testing.expectEqual(@as(usize, PMAX), chunkRows(0, PMAX));
    try std.testing.expectEqual(@as(usize, PMAX), chunkRows(200_000, PMAX));
    try std.testing.expect(chunkRows(1_000_000, PMAX) <= 2048);
}

pub const vision_memory = @import("vision_memory.zig");
