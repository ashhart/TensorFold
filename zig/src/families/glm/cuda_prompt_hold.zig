//! A sequence arena owns the prompt row until its first draft consumes it, across every admission pass.
pub fn save(e: anytype, src: u64) !void {
    try e.copy(e.cur.st.prompt_hidden, src, e.c.hidden * 2);
}
pub fn input(e: anytype) u64 {
    return e.cur.st.prompt_hidden;
}
