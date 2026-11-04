//! Nemotron Metal's position allocation and prefill boundary, without device dependencies.
pub const max_window = 16;
pub const max_lanes = 64;
pub const fallback = 1 << 18;

pub fn positions(window: usize) usize {
    return (if (window > 0) window else fallback) + max_lanes + 1;
}

pub fn reported(window: usize, allocated: usize) u32 {
    return @intCast(if (window > 0) window else allocated - max_window);
}

pub fn check(prompt: usize, reply: usize, allocated: usize) error{PromptTooLong}!void {
    if (prompt == 0 or prompt + reply + max_window > allocated) return error.PromptTooLong;
}
