//! Named environment validation before model, driver and transport work; successful values keep their old semantics.
const std = @import("std");
const sys = @import("sys.zig");

pub const Error = error{InvalidEnvironment};
pub const Scope = enum { common, cuda, metal, xpu, slide, hooks, dsv41, gemma4, drama3, glm, flashnext };
pub const Kind = enum { u32, u64, size, percent, positive, device, segments, mib, bit_width, word, prompt, flag, toggle, real, positive_real, signed_real, range, path };
pub const Rule = struct {
    name: [:0]const u8,
    scope: Scope,
    kind: Kind,
    forms: []const u8,
    trim: []const u8 = "",
    first_digit: bool = false,
    /// A whole number's own bounds inside its kind's; a real kind applies `max` too when it is set.
    min: u64 = 0,
    max: u64 = std.math.maxInt(u64),
    /// When nonempty, the only whole numbers accepted.
    only: []const u64 = &.{},
    /// A `.word` setting's accepted texts; the parsed value is the matching index.
    words: []const []const u8 = &.{},
    /// A legacy name that is ignored, and so never refused, while its canonical name is set.
    alias_of: ?[:0]const u8 = null,
    /// False where the old reader took only 0, false and off as off: "no" is refused, not read as off.
    no_word: bool = true,
};

pub fn isReal(kind: Kind) bool {
    return kind == .real or kind == .positive_real or kind == .signed_real;
}

pub const rules = [_]Rule{
    .{ .name = "TF_STARTUP_RECEIPT", .scope = .common, .kind = .flag, .forms = "0 or 1 (opt-in startup and first completion receipts)" },
    .{ .name = "TENSORFOLD_VIDEO_TOKENS", .scope = .common, .kind = .u64, .forms = "a nonnegative whole video-token budget (default 16384; unsigned 64-bit count)" },
    .{ .name = "TENSORFOLD_MAX_PDF_PAGES", .scope = .common, .kind = .positive, .forms = "a whole number from 1 to 4294967295 (capped by --vision-max-images)" },
    .{ .name = "TENSORFOLD_PRESSURE_CACHE_PERCENT", .scope = .common, .kind = .percent, .trim = " \t", .forms = "a whole percentage from 0 to 100" },
    .{ .name = "TENSORFOLD_PREFIX_WAIT_MS", .scope = .common, .kind = .u32, .trim = " \t", .first_digit = true, .forms = "whole milliseconds from 0 to 4294967295 (0 disables waiting)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "a whole unsigned 64-bit read count (0 disables reading ahead; capped at 65536)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_GIB", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "a whole unsigned 64-bit GiB count (capped at 16777216)" },
    .{ .name = "TENSORFOLD_RESTORE_PREFETCH_MS", .scope = .common, .kind = .u64, .trim = " \t", .first_digit = true, .forms = "whole unsigned 64-bit milliseconds (capped at 1099511627776)" },
    .{ .name = "TENSORFOLD_TOOL_ID_KEY_FILE", .scope = .common, .kind = .path, .forms = "a path without control characters to a file only its owner can read, holding at least 32 key bytes" },
    .{ .name = "TENSORFOLD_CUDA_KERNELS", .scope = .cuda, .kind = .path, .forms = "a captured kernel set's folder without control characters, or native for the engine's own kernels" },
    .{ .name = "TF_CUDA_DEVICE", .scope = .cuda, .kind = .device, .trim = " ", .forms = "a whole GPU ordinal from 0 to 4294967295, or empty for the default" },
    .{ .name = "TF_CUDA_SEGMENTS", .scope = .cuda, .kind = .segments, .trim = " ", .forms = "a whole number from 1 to 4, or empty for the default" },
    .{ .name = "TF_CUDA_CARVEOUT", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_CUDA_CARVEOUT_MIB", .scope = .cuda, .kind = .mib, .forms = "a positive whole MiB count fitting a native byte size, or empty for 1792 MiB" },
    .{ .name = "TF_MTP_DRAFTS", .scope = .cuda, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (clamped to the drafter's window)" },
    .{ .name = "TF_DSPINE_WINDOW", .scope = .cuda, .kind = .size, .min = 16, .max = 8192, .forms = "a whole row count from 16 to 8192, at most the DSpine drafter's own sliding_window" },
    .{ .name = "TF_DRAFT_NODES", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative native-size node count (capped by the drafter's window)" },
    .{ .name = "TF_QMMF_HALF_BELOW", .scope = .cuda, .kind = .size, .forms = "a whole nonnegative native-size block count (0 disables half tiles)" },
    .{ .name = "TENSORFOLD_PROMPT_CACHE_KEEP_OTHERS", .scope = .cuda, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_PROMPT_INTERMEDIATE", .scope = .cuda, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_ROCE_GID_INDEX", .scope = .cuda, .kind = .u32, .forms = "a whole GID index from 0 to 2147483647" },
    .{ .name = "TF_GEMMA4_CHUNK_SOLO", .scope = .gemma4, .kind = .size, .forms = "a solo prompt pass's rows, a multiple of 512 up to 4096" },
    .{ .name = "TF_DRAMA_PREFILL_ROWS", .scope = .drama3, .kind = .u32, .trim = " ", .forms = "whole prompt rows a chunk, 1 through 1024 (checked when a Drama 3 checkpoint opens)" },
    .{ .name = "TF_DRAMA_REFERENCE_CACHE", .scope = .drama3, .kind = .u32, .trim = " ", .forms = "kept reference states, 0 (off) through 16 (checked when a Drama 3 checkpoint opens)" },
    .{ .name = "TF_Q27_COPY_ROWS", .scope = .metal, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (0 disables copied rows)" },
    .{ .name = "TF_Q27_DRAFT_SLOTS", .scope = .metal, .kind = .u32, .forms = "a whole number from 0 to 4294967295 (clamped to the current stream bounds)" },
    .{ .name = "TF_HEAD_GEOMETRY", .scope = .metal, .kind = .size, .forms = "a whole nonnegative native-size geometry count" },
    .{ .name = "TF_PROMPT_ROWS", .scope = .metal, .kind = .prompt, .forms = "a multiple of 16 from 128 to 2048 for native serving" },
    .{ .name = "TF_ANE_COLUMNS", .scope = .metal, .kind = .u32, .forms = "a whole column count from 0 to 4294967295 (0 disables the share)" },
    .{ .name = "TF_FLASHNEXT_SCALE_INDEX", .scope = .metal, .kind = .flag, .forms = "0 or 1 (1: Flash Next's 4-bit expert kernels read scales and biases through a 16-bit index)" },
    .{ .name = "GLM_EHPROJ_NAX", .scope = .metal, .kind = .flag, .forms = "0 or 1 (0 serves GLM's MTP eh_proj on the scalar kernel)" },
    .{ .name = "TF_FLASHNEXT_GROUPED_ROWS", .scope = .metal, .kind = .size, .forms = "0 (never) or the rows, 1 to 32, from which a window's experts run grouped", .max = 32 },
    .{ .name = "TF_DSV41_SMALL", .scope = .metal, .kind = .flag, .forms = "0 or 1 (1 reads a DeepSeek-V4.1 Metal checkpoint's config at its own small shapes)" },
    .{ .name = "TF_DSV41_ASSETS", .scope = .metal, .kind = .path, .forms = "a directory path without control characters holding DeepSeek-V4.1's Engram token map" },
    .{ .name = "TF_DSV41_ENGRAM_DIR", .scope = .metal, .kind = .path, .forms = "a directory path without control characters holding DeepSeek-V4.1's packed Engram tables" },
    .{ .name = "TF_DSV41_ENGRAM_VERIFY", .scope = .metal, .kind = .word, .words = &.{ "sample", "full" }, .forms = "sample or full (full also checks every byte of the Engram tables at boot)" },
    .{ .name = "TF_DSV41_ENGRAM_CACHE", .scope = .metal, .kind = .u32, .forms = "a whole Engram record count a layer from 0 to 4294967295 (default 65536)" },
    .{ .name = "XPU_A8_BITS", .scope = .xpu, .kind = .bit_width, .forms = "4 or 8" },
    .{ .name = "LFM3_FUSE", .scope = .xpu, .kind = .u32, .forms = "a whole fusion mask from 0 to 4294967295" },
    .{ .name = "TF_SLIDE_POCKET_KEY", .scope = .slide, .kind = .word, .words = &.{ "auto", "reply", "subject", "both" }, .forms = "auto, reply, subject or both" },
    .{ .name = "TF_SLIDE_FIT_GIB", .scope = .slide, .kind = .size, .forms = "a whole nonnegative native-size GiB count" },
    .{ .name = "TF_SLIDE_POCKET_DUMP", .scope = .slide, .kind = .path, .forms = "a directory path without control characters, where each lesson's read rows are written" },
    .{ .name = "TF_SLIDE_FORGET_FAIL", .scope = .hooks, .kind = .u32, .trim = " ", .forms = "a whole fact index from 0 to 4294967295 (0 disables the planted failure)" },
    .{ .name = "TF_GLM_HOST_CACHE", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_VISION_HOST", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_PROF", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_LINK_WAIT", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_COPY_SHARE", .scope = .glm, .kind = .flag, .forms = "0 or 1 (1, with TF_GLM_COPIES=1: copies in shared rounds take the lone ramp)" },
    .{ .name = "TF_GLM_PDL", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_PROMPT_STEPS", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_SHARED", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_DSEL", .scope = .glm, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_GLM_DFLASH2_LOG", .scope = .glm, .kind = .u64, .forms = "a whole unsigned 64-bit round count (0 logs no drafts)" },
    .{ .name = "TF_GLM_DRAFT_CONFIDENCE", .scope = .glm, .kind = .real, .max = 1, .forms = "a finite chain-chance cut from 0 to 1 (default 0.7)" },
    // 2^31 - 32: GLM's slots, the window and its 32 verify rows (glm/cuda_native.zig's capacity), keep i32 positions
    .{ .name = "TF_GLM_CONTEXT_CEILING", .scope = .glm, .kind = .u64, .first_digit = true, .min = 1, .max = 2147483616, .forms = "a whole token count from 1 to 2147483616 (the window GLM-5.3-Flash may serve in place of config.json's)" },
    .{ .name = "TF_DSV41_MHC_DEFER", .scope = .dsv41, .kind = .toggle, .trim = " \t", .no_word = false, .forms = "0, 1, true, false, on, off or yes (any case)" },
    .{ .name = "TF_DSV41_COEF_LATE", .scope = .dsv41, .kind = .toggle, .trim = " \t", .no_word = false, .forms = "0, 1, true, false, on, off or yes (any case)" },
    .{ .name = "TF_DSV41_LOOKUP", .scope = .dsv41, .kind = .toggle, .trim = " \t", .forms = "0, 1, true, false, on, off, yes or no (any case); blank turns lookup drafts off" },
    .{ .name = "TF_DSV41_LOOKUP_PLAN", .scope = .dsv41, .kind = .toggle, .trim = " \t", .forms = "0, 1, true, false, on, off, yes or no (any case)" },
    .{ .name = "TF_DSV41_GM_V3", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_OWN_PREFILL", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_DRAFTS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_GRAPHS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_AOT_GUARD", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ENGRAM_PREFETCH", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_BIAS_VL", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1 (either selects the vision bias; unset or blank does not)" },
    .{ .name = "TF_DSV41_FLOOR_PRICE", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SESSIONS", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SAMPLING", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SLOT_DRAFTS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_FORMS_INPLACE", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_EX_STACKS", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_STREAM_LAZY", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_GRAPH_LOG", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_WIN_PROF", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_BOOT_TRACE", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_PREFILL_PIECES", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_GREEDY_GPU", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_DRAFT_TIMED", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SYNC_EACH", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ENGRAM_POLL", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ENGRAM_AIO", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ENGRAM_URING", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_KV_SPLIT", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_NEEDS_POOL", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_KV_SPLIT_DEDUP", .scope = .dsv41, .kind = .flag, .trim = " \t", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_KV_SPLIT_EARLY", .scope = .dsv41, .kind = .flag, .trim = " \t", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_KV_SPLIT_HISTORY", .scope = .dsv41, .kind = .flag, .trim = " \t", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_PREFILL_ADAPT", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_R1", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_PREFETCH_AHEAD", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_KV_NORM_STORE", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ROUTER_GROUP_ROT", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SCHEMA_PROMPT", .scope = .dsv41, .kind = .flag, .trim = " \t\r\n", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_TOOL_GRAMMAR", .scope = .dsv41, .kind = .flag, .trim = " ", .alias_of = "TENSORFOLD_TOOL_GRAMMAR", .forms = "0 or 1 (ignored while TENSORFOLD_TOOL_GRAMMAR is set)" },
    .{ .name = "TF_DSV41_PIECE_RUNS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_REPLAY_RUNS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_JOIN_DRAFTS", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_TAIL_JOIN", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_PF_WS_SHARE", .scope = .dsv41, .kind = .flag, .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SESSION_SKIP_COVERED", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SESSION_DELTA", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_SESSION_SLOTS", .scope = .dsv41, .kind = .flag, .trim = " ", .forms = "0 or 1" },
    .{ .name = "TF_DSV41_ENGRAM_QD", .scope = .dsv41, .kind = .u32, .forms = "a whole reader-thread count from 0 to 4294967295 (0 reads serially)" },
    .{ .name = "TF_DSV41_ENGRAM_READERS", .scope = .dsv41, .kind = .u32, .max = 32, .forms = "a whole reader-thread count from 0 to 32 (0 reads serially)" },
    .{ .name = "TF_DSV41_GEN_STREAMS", .scope = .dsv41, .kind = .size, .min = 1, .forms = "a whole request count from 1 to the native size maximum" },
    .{ .name = "TF_DSV41_CONTEXT", .scope = .dsv41, .kind = .u64, .min = 1, .max = std.math.maxInt(i64), .forms = "a whole position count from 1 to 9223372036854775807" },
    .{ .name = "TF_DSV41_SLOTS", .scope = .dsv41, .kind = .u32, .min = 1, .max = 16, .forms = "a whole slot count from 1 to 16" },
    .{ .name = "TF_DSV41_NUCLEUS", .scope = .dsv41, .kind = .u32, .forms = "a whole candidate count from 0 to 4294967295 (0 disables the nucleus)" },
    .{ .name = "TF_DSV41_POOL_TOKENS", .scope = .dsv41, .kind = .u64, .forms = "a whole unsigned 64-bit token count" },
    .{ .name = "TF_DSV41_KV_SPLIT_UNION_MIB", .scope = .dsv41, .kind = .u64, .max = std.math.maxInt(i64) >> 20, .forms = "a whole MiB count from 0 to 8796093022207" },
    .{ .name = "TF_DSV41_PREFILL_ROWS", .scope = .dsv41, .kind = .u64, .trim = " ", .forms = "a whole unsigned 64-bit row count" },
    .{ .name = "TF_DSV41_INDEX_BUDGET_MIB", .scope = .dsv41, .kind = .u64, .trim = " ", .min = 1, .max = 4096, .forms = "a whole MiB count from 1 to 4096" },
    .{ .name = "TF_DSV41_BATCH_SHORT", .scope = .dsv41, .kind = .u64, .trim = " ", .forms = "a whole unsigned 64-bit token count" },
    .{ .name = "TF_DSV41_LONG_PROMPT", .scope = .dsv41, .kind = .u64, .trim = " ", .min = 1, .forms = "a whole token count of at least 1" },
    .{ .name = "TF_DSV41_PREFILL_CONCURRENT", .scope = .dsv41, .kind = .u64, .trim = " ", .forms = "a whole unsigned 64-bit prompt count" },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_MIN", .scope = .dsv41, .kind = .u64, .trim = " ", .min = 16, .forms = "a whole row count of at least 16" },
    .{ .name = "TF_DSV41_FLOOR_HARD_GIB", .scope = .dsv41, .kind = .positive_real, .trim = " ", .forms = "a finite GiB amount greater than 0" },
    .{ .name = "TF_DSV41_PF_WS_FLOOR_GIB", .scope = .dsv41, .kind = .real, .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_GRAPH_FLOOR_GIB", .scope = .dsv41, .kind = .real, .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_GRAPH_COST_MIB", .scope = .dsv41, .kind = .real, .forms = "a finite MiB amount of at least 0" },
    .{ .name = "TF_DSV41_EXPERT_TOPP", .scope = .dsv41, .kind = .signed_real, .forms = "a finite number" },
    .{ .name = "TF_DSV41_PREFILL_SHARE", .scope = .dsv41, .kind = .positive_real, .trim = " ", .max = 1, .forms = "a finite share greater than 0 and at most 1" },
    .{ .name = "TF_DSV41_FLOOR_GIB", .scope = .dsv41, .kind = .positive_real, .trim = " ", .forms = "a finite GiB amount greater than 0" },
    .{ .name = "TF_DSV41_FLOOR_PF_CHUNK_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_FLOOR_PF_INDEX_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_FLOOR_PF_HOST_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_FLOOR_PF_OTHER_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_ADMIT_FREE_FLOOR_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_FLOOR_REFUSE_S", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite number of seconds of at least 0" },
    .{ .name = "TF_DSV41_ADMIT_CACHE_KEEP_GB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_PREFILL_ADAPT_UP_GIB", .scope = .dsv41, .kind = .real, .trim = " ", .forms = "a finite GiB amount of at least 0" },
    .{ .name = "TF_DSV41_IMAGES", .scope = .dsv41, .kind = .word, .trim = " ", .words = &.{ "0", "native" }, .forms = "native or 0" },
    .{ .name = "TF_DSV41_LAYERS", .scope = .dsv41, .kind = .range, .forms = "lo-hi: two whole layer numbers from 0 to 4294967295 with lo <= hi" },
    .{ .name = "TF_DSV41_ROWS_CAP", .scope = .dsv41, .kind = .u32, .only = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64 }, .forms = "1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48 or 64" },
    .{ .name = "TF_FLASHNEXT_PROMPT_BF16", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_WINDOW_GRAPH", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_LANE_PARTS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_KEEP_INPASS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_FIRST_EARLY", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_GPU_TOPK", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_GDN_FUSED", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_GDN_LANES", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_BATCH_HEADS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_SAMPLED_HEADS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_SHARED_DRAFTS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_SHARED_SIDE", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_HC_FUSE", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 1)" },
    .{ .name = "TF_FLASHNEXT_KEEP_LATE", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (unset or blank leaves it unset)" },
    .{ .name = "TF_FLASHNEXT_DECODE_GRAPH", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_ATTN_LANES", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_WARM_AHEAD", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_TABLE_WARM", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_PROFILE", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_VERIFY_HOST_LOGITS", .scope = .flashnext, .kind = .flag, .forms = "0 or 1 (default 0)" },
    .{ .name = "TF_FLASHNEXT_DRAFTS", .scope = .flashnext, .kind = .size, .forms = "a whole draft count from 0 to the native size maximum (default 6; clamped to 1 through 15)" },
    .{ .name = "TF_FLASHNEXT_DRAFTS_GREEDY", .scope = .flashnext, .kind = .size, .forms = "a whole draft count from 0 to the native size maximum (default 8; clamped to 1 through 15)" },
    .{ .name = "TF_FLASHNEXT_GREEDY_WIDE", .scope = .flashnext, .kind = .size, .forms = "a whole round width from 0 to the native size maximum (default 4)" },
    .{ .name = "TF_FLASHNEXT_WINDOW_ROWS", .scope = .flashnext, .kind = .size, .forms = "a whole row count from 0 to the native size maximum (default the shared round's rows; capped by them)" },
    .{ .name = "TF_FLASHNEXT_READOUT_ROWS", .scope = .flashnext, .kind = .size, .forms = "a whole row count from 0 to the native size maximum (default 64; 0 never takes the tiles)" },
    .{ .name = "TF_FLASHNEXT_FACT_ROWS", .scope = .flashnext, .kind = .size, .forms = "a whole row count from 0 to the native size maximum (default 64; 0 never takes the tiles)" },
    .{ .name = "TF_FLASHNEXT_KV_PAGE_TEST", .scope = .flashnext, .kind = .size, .forms = "a page size the kernels take, whole blocks of the indexer's ratio up to the window (unset: off)" },
    .{ .name = "TF_FLASHNEXT_CONFIDENCE", .scope = .flashnext, .kind = .signed_real, .forms = "a finite number (default -0.4)" },
    .{ .name = "TF_FLASHNEXT_PROMPT", .scope = .flashnext, .kind = .word, .words = &.{"rows"}, .forms = "rows for the rows path (unset or blank keeps the plain path)" },
};

pub fn rule(name: []const u8) Rule {
    for (rules) |r| if (std.mem.eql(u8, name, r.name)) return r;
    @panic("unregistered strict environment parser");
}

/// A compile error unless `name` has a row, so a strict read can never reach `rule`'s panic at run time.
pub fn registered(comptime name: []const u8) void {
    @setEvalBranchQuota(20 * rules.len * 64);
    for (rules) |r| if (std.mem.eql(u8, r.name, name)) return;
    @compileError(name ++ " has no row in platform/environment.zig");
}

/// Null means unset, empty or whitespace-only (main's readers used their defaults then), never malformed.
pub fn parse(name: []const u8, raw: ?[]const u8) Error!?u64 {
    const r = rule(name);
    if (std.mem.trim(u8, raw orelse return null, " \t\r\n").len == 0) return null;
    const text = std.mem.trim(u8, raw.?, r.trim);
    if (r.first_digit and (text[0] < '0' or text[0] > '9')) return error.InvalidEnvironment;
    if (r.kind == .word) {
        for (r.words, 0..) |v, i| if (std.mem.eql(u8, text, v)) return i;
        return error.InvalidEnvironment;
    }
    if (r.kind == .toggle) return toggle(r, text) orelse error.InvalidEnvironment;
    // a path travels as 0, its text read by `path`; the file itself is its reader's to check
    if (r.kind == .path) return if (for (text) |ch| {
        if (ch < 0x20 or ch == 0x7f) break true;
    } else false) error.InvalidEnvironment else 0;
    // a real travels as its f64 bits and a range as lo above hi; `value` and `range` decode them
    if (isReal(r.kind)) return @bitCast(try real(r, text));
    if (r.kind == .range) {
        var it = std.mem.splitScalar(u8, text, '-');
        const lo = std.fmt.parseInt(u32, it.next().?, 10) catch return error.InvalidEnvironment;
        const hi = std.fmt.parseInt(u32, it.next() orelse "", 10) catch return error.InvalidEnvironment;
        if (it.next() != null or hi < lo) return error.InvalidEnvironment;
        return @as(u64, lo) << 32 | hi;
    }
    if (r.kind == .flag or r.kind == .bit_width) {
        const a: []const u8 = if (r.kind == .flag) "0" else "4";
        const b: []const u8 = if (r.kind == .flag) "1" else "8";
        if (!std.mem.eql(u8, text, a) and !std.mem.eql(u8, text, b)) return error.InvalidEnvironment;
    }
    const n = std.fmt.parseInt(u64, text, 10) catch return error.InvalidEnvironment;
    const max: u64 = switch (r.kind) {
        .u32, .positive, .device => std.math.maxInt(u32),
        .percent => 100,
        .size, .mib => std.math.maxInt(usize),
        .segments => 4,
        else => std.math.maxInt(u64),
    };
    if (n > max or n < r.min or n > r.max) return error.InvalidEnvironment;
    if (r.only.len > 0 and std.mem.indexOfScalar(u64, r.only, n) == null) return error.InvalidEnvironment;
    if ((r.kind == .positive or r.kind == .segments or r.kind == .mib) and n == 0) return error.InvalidEnvironment;
    if (r.kind == .mib and n > std.math.maxInt(usize) / (1 << 20)) return error.InvalidEnvironment;
    if (r.kind == .prompt and (n < 128 or n > 2048 or n % 16 != 0)) return error.InvalidEnvironment;
    if (std.mem.eql(u8, name, "TF_ROCE_GID_INDEX") and n > std.math.maxInt(c_int)) return error.InvalidEnvironment;
    return n;
}

/// A switch: 0 or 1, or the word forms older readers took, in any case; 0 is off.
fn toggle(r: Rule, text: []const u8) ?u64 {
    if (!r.no_word and std.ascii.eqlIgnoreCase(text, "no")) return null;
    for ([_][]const u8{ "0", "false", "off", "no", "1", "true", "on", "yes" }, 0..) |v, i|
        if (std.ascii.eqlIgnoreCase(text, v)) return i / 4;
    return null;
}

/// Finite only; `.real` is at least 0, `.positive_real` above 0, and a set `max` caps either.
fn real(r: Rule, text: []const u8) Error!f64 {
    const x = std.fmt.parseFloat(f64, text) catch return error.InvalidEnvironment;
    const low = switch (r.kind) {
        .real => x >= 0,
        .positive_real => x > 0,
        else => true,
    };
    if (!std.math.isFinite(x) or !low) return error.InvalidEnvironment;
    if (r.max != std.math.maxInt(u64) and x > @as(f64, @floatFromInt(r.max))) return error.InvalidEnvironment;
    return x;
}

/// A `.path` setting's text is never shown: a key pasted into it by mistake would reach the log.
pub fn describe(a: std.mem.Allocator, name: []const u8, raw: []const u8) ![]const u8 {
    if (rule(name).kind == .path) return std.fmt.allocPrint(a, "{s}: expected {s}", .{ name, rule(name).forms });
    return std.fmt.allocPrint(a, "{s}=\"{f}\": expected {s}", .{ name, std.zig.fmtString(raw), rule(name).forms });
}

/// A direct family/tool entry point gets the same refusal even when it has no server problem buffer.
fn checked(name: [:0]const u8, raw: ?[]const u8) Error!?u64 {
    return parse(name, raw) catch {
        if (!@import("builtin").is_test) if (rule(name).kind == .path) std.log.err("{s}: expected {s}", .{ name, rule(name).forms }) else std.log.err("{s}=\"{f}\": expected {s}", .{ name, std.zig.fmtString(raw.?), rule(name).forms });
        return error.InvalidEnvironment;
    };
}

pub fn check(name: [:0]const u8, raw: ?[]const u8) Error!void {
    _ = try checked(name, raw);
}

/// One setting in its reader's type: a switch as bool, a real as a float, any other kind as its whole number.
pub fn value(comptime T: type, name: [:0]const u8, raw: ?[]const u8, default: T) Error!T {
    std.debug.assert(isReal(rule(name).kind) == (@typeInfo(Plain(T)) == .float) and rule(name).kind != .range and rule(name).kind != .path);
    return decode(T, (try checked(name, raw)) orelse return default);
}

fn Plain(comptime T: type) type {
    return if (@typeInfo(T) == .optional) @typeInfo(T).optional.child else T;
}

fn decode(comptime T: type, n: u64) T {
    return switch (@typeInfo(T)) {
        .optional => |o| decode(o.child, n),
        .bool => n != 0,
        .float => @floatCast(@as(f64, @bitCast(n))),
        else => @intCast(n),
    };
}

pub fn read(comptime T: type, name: [:0]const u8, default: T) Error!T {
    return value(T, name, sys.getenv(name), default);
}

/// A `.range` setting's lo and hi, or null when unset or blank.
pub fn range(name: [:0]const u8, raw: ?[]const u8) Error!?[2]u32 {
    const n = (try checked(name, raw)) orelse return null;
    return .{ @intCast(n >> 32), @truncate(n) };
}

/// A `.path` setting's text, or null when unset or blank; nothing is logged, so the caller words the refusal.
pub fn path(name: [:0]const u8, raw: ?[]const u8) Error!?[]const u8 {
    std.debug.assert(rule(name).kind == .path);
    _ = (try parse(name, raw)) orelse return null;
    return std.mem.trim(u8, raw.?, rule(name).trim);
}

/// One scope's rules in table order, without allocating: the first malformed value is logged and refused.
pub fn checkScope(scope: Scope, environment: anytype) Error!void {
    for (rules) |r| if (r.scope == scope and !masked(r, environment)) {
        _ = try checked(r.name, environment.get(r.name));
    };
}

fn masked(r: Rule, environment: anytype) bool {
    return if (r.alias_of) |canonical| environment.get(canonical) != null else false;
}

/// Every CUDA entry point shares flag precedence, accepted text and diagnostics.
pub fn device(flag: ?u32, raw: ?[]const u8) Error!u32 {
    if (flag) |n| return n;
    return @intCast((try checked("TF_CUDA_DEVICE", raw)) orelse 0);
}

pub const Flags = struct {
    cuda: bool = false,
    metal: bool = false,
    xpu: bool = false,
    slide: bool = false,
    hooks: bool = false,
    dsv41: bool = false,
    gemma4: bool = false,
    drama3: bool = false,
    glm: bool = false,
    flashnext: bool = false,
    device: ?u32 = null,
    segments: ?u32 = null,
    carveout: bool = false,

    fn includes(f: Flags, scope: Scope) bool {
        return switch (scope) {
            .common => true,
            .cuda => f.cuda,
            .metal => f.metal,
            .xpu => f.xpu,
            .slide => f.slide,
            .hooks => f.hooks,
            .dsv41 => f.dsv41,
            .gemma4 => f.gemma4,
            .drama3 => f.drama3,
            .glm => f.glm,
            .flashnext => f.flashnext,
        };
    }
};

/// The environment is injectable so the actual startup validator runs against host fixtures without a GPU.
pub fn refusal(a: std.mem.Allocator, environment: anytype, flags: Flags) !?[]const u8 {
    for (rules) |r| {
        if (!flags.includes(r.scope) or masked(r, environment)) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_DEVICE") and flags.device != null) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_SEGMENTS") and flags.segments != null) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_CARVEOUT") and flags.carveout) continue;
        if (std.mem.eql(u8, r.name, "TF_CUDA_CARVEOUT_MIB") and !flags.carveout and
            !std.mem.eql(u8, environment.get("TF_CUDA_CARVEOUT") orelse "0", "1")) continue;
        if (std.mem.eql(u8, r.name, "TF_DRAFT_NODES") and environment.get("TF_NO_TREES") != null) continue;
        const raw = environment.get(r.name) orelse continue;
        _ = parse(r.name, raw) catch return try describe(a, r.name, raw);
    }
    return null;
}

pub const Process = struct {
    pub fn get(_: Process, name: [:0]const u8) ?[]const u8 {
        return sys.getenv(name);
    }
};

test {
    _ = @import("environment_test.zig");
    _ = @import("environment_flashnext_test.zig");
}
