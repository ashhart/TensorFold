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
};
