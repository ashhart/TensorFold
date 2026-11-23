//! A window profile's timing marks (tf-cuda-test window-profile): events recorded between kernel classes.
const cuda = @import("cuda");

/// A window's kernel classes as a profile names them: each mark closes the span since the one before.
pub const Class = enum { embed, norm, mamba_in, conv_scan, gnorm, mamba_out, qkv, attention, attn_o, route, plan, experts_up, experts_down, shared_wait, head, sample };

/// Timing events recorded between kernel classes (the window profile's; null in every other forward).
pub const Marks = struct {
    ev: []cuda.Event,
    class: []Class,
    n: usize = 0,

    pub fn add(m: *Marks, s: cuda.Stream, c: Class) !void {
        if (m.n == m.ev.len) return error.TooManyMarks;
        try m.ev[m.n].record(s);
        m.class[m.n] = c;
        m.n += 1;
    }
};
