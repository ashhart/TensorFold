//! SAFE registry rows whose legacy call-site readers retain their defaults, clamps and public errors.
const Rule = @import("environment.zig").Rule;

pub const rules = [_]Rule{
    .{ .name = "TF_LOAD_PREFETCH", .scope = .common, .kind = .size, .max = 16, .blank_default = false, .forms = "a whole reader-thread count from 0 to 16 (unset: the caller's default)" },
    .{ .name = "TENSORFOLD_PREFILL_SHARE", .scope = .common, .kind = .positive_real, .max = 1, .forms = "a finite share from 0.01 to 1 (unset or exactly empty uses the caller default; startup enforces the lower bound)" },
    .{ .name = "TENSORFOLD_PREFILL_BUDGET", .scope = .common, .kind = .u32, .trim = " \t", .first_digit = true, .forms = "a whole row budget from 0 to 4294967295 (0 disables the budget)" },
    .{ .name = "TENSORFOLD_PREFILL_POLICY", .scope = .common, .kind = .word, .trim = " \t", .words = &.{"fcfs"}, .forms = "fcfs (unset, blank, or whitespace: the stock order)" },
    .{ .name = "TENSORFOLD_PREFILL_TOKENS", .scope = .common, .kind = .u32, .trim = " \t", .first_digit = true, .forms = "a whole u32 row count (0: 2048; nonzero values clamped by the reader to 64..16384)" },
    .{ .name = "TF_CUDA_TEST_MLP128", .scope = .cuda, .kind = .flag, .blank_default = false, .forms = "0 or 1 (experimental SM120 MLP128 override; unset: automatic selection)" },
    .{ .name = "TF_NVFP4_DECODE_WARPS", .scope = .cuda, .kind = .word, .words = &.{ "1", "4" }, .blank_default = false, .forms = "1 or 4 decode warps (unset: 4)" },
    .{ .name = "TF_FLASHNEXT_PROMPT_EXPERT_TILE", .scope = .flashnext, .kind = .word, .words = &.{ "0", "1", "2" }, .blank_default = false, .forms = "0 (off), 1 (up projection), or 2 (up and down projections; unset: off)" },
    .{ .name = "TF_FLASHNEXT_LOAD_THREADS", .scope = .flashnext, .kind = .size, .min = 1, .max = 32, .blank_default = false, .forms = "a whole NVFP4 loading-thread count from 1 to 32 (unset: 8; set blank is invalid)" },
    .{ .name = "TF_FLASHNEXT_PLE_IO", .scope = .flashnext, .kind = .word, .words = &.{ "mmap", "pread" }, .blank_default = false, .forms = "mmap or pread (unset: pread on Linux, mmap elsewhere; set blank is invalid)" },
    .{ .name = "TENSORFOLD_CHECKPOINT_SHA256", .scope = .flashnext, .kind = .path, .forms = "exactly 64 hexadecimal digits for the checkpoint SHA-256 (unset: hash the checkpoint; the identity reader checks length and digits)" },
    .{ .name = "TENSORFOLD_CUDA_MEMORY_LIMIT_GB", .scope = .cuda, .kind = .positive_real, .trim = " ", .forms = "a finite positive GiB cap whose scaled byte count is at least 1 and below 2^64 (unset or blank: no cap)" },
    .{ .name = "TF_EXPERIMENTAL", .scope = .dsv41, .kind = .word, .blank_default = false, .words = &.{"deepseek"}, .forms = "deepseek to opt in to the experimental DeepSeek-V4.1 CUDA model" },
    .{ .name = "TF_DSV41_DECODE_TP_LANES", .scope = .dsv41, .kind = .u32, .blank_default = false, .forms = "a whole decode-TP lane mode (unset: 0; 1 disables drafts, 2 uses oracle drafts, 3 uses DSpark GPU)" },
    .{ .name = "TF_DSV41_DECODE_TP_PROF_ROWS", .scope = .dsv41, .kind = .path, .forms = "space-separated whole u32 row counts without control characters (unset or blank: no extra profile rows)" },
    .{ .name = "TF_DSV41_KV_CHECK_POOL_TOKENS", .scope = .dsv41, .kind = .u64, .forms = "a whole unsigned 64-bit KV-check pool-token count (unset: forward 16384)" },
    .{ .name = "TF_DSV41_KV_CHECK_TOPP", .scope = .dsv41, .kind = .signed_real, .forms = "a finite KV-check expert top-p value forwarded to TF_DSV41_EXPERT_TOPP" },
    .{ .name = "TF_INKLING2_EXL3_DIR", .scope = .cuda, .kind = .path, .forms = "a directory path without control characters holding real Inkling 2 Small EXL3 pack headers (unset or blank: skip the opt-in host test)" },
    .{ .name = "TF_QWEN27_WINDOW", .scope = .metal, .kind = .u32, .only = &.{ 16, 32, 64 }, .blank_default = false, .forms = "16, 32 or 64 verification rows (unset: 16; set blank is invalid)" },
    .{ .name = "TF_QWEN27_COPIES", .scope = .metal, .kind = .flag, .blank_default = false, .forms = "0 or 1 (unset: device/window/model policy; set blank is invalid)" },
    .{ .name = "TF_QWEN27_LANES", .scope = .metal, .kind = .flag, .blank_default = false, .forms = "0 (serial lone loop) or 1 (lane loop; unset: device/window policy; set blank is invalid)" },
};
