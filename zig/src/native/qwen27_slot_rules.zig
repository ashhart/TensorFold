//! Pure stream eligibility and environment settings shared by the 27B slot owner.
const std = @import("std");
const lanes = @import("tensorfold").lanes;
const resume_rules = @import("qwen27_resume.zig");

/// Slot 0 resumes a later prompt from where its last stream left it unless TF_QWEN27_SLOT_RESUME=0 (read at start).
pub fn slotResumeEnv() !bool {
    return @import("sys").environment.read(bool, "TF_QWEN27_SLOT_RESUME", true);
}

/// Whether a lone stream's DFlash2 block adapts to its landed depth (TF_Q27_DFLASH_ADAPTIVE_BLOCK; default off).
pub fn adaptiveBlockEnv() !bool {
    return @import("sys").environment.read(bool, "TF_Q27_DFLASH_ADAPTIVE_BLOCK", false);
}

/// A stream the prompt cache would look up: no images, no prompt logprobs (they start at 0), taught rows.
pub fn plain(s: *const lanes.Stream) bool {
    return s.images.len == 0 and s.prompt_logprobs == null and !s.untaught and !s.memory_active and s.embedding == null and resume_rules.plain(s.cache_salt);
}

/// Whether the cache keeps the state at the end of `s`'s prompt (a background pass's: the cache planned that mark).
pub fn keepsEnd(s: *const lanes.Stream) bool {
    return std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(s.prompt().len)) != null;
}
