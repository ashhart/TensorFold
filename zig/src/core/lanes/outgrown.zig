//! Backend.grow: a stream whose next rows find no memory ends with `length` before its round; the others run on.
const be = @import("backend.zig");
const Plan = @import("windows.zig").Plan;
const Stream = @import("stream.zig").Stream;

/// Whether `s` leaves: the backend holds no memory for `w`'s rows (a backend without the hook: never).
fn cut(b: be.Backend, s: *Stream, w: be.Window) bool {
    const grow = b.vtable.grow orelse return false;
    if (grow(b.ptr, w)) return false;
    s.finished = true;
    s.reason = .length;
    s.problem = "the server had no KV memory left for this reply's next rows";
    return true;
}

/// A lone stream's verify; false when its rows found no memory and no forward ran.
pub fn one(b: be.Backend, s: *Stream, windows: []const be.Window, out: []be.Verified) !bool {
    if (cut(b, s, windows[0])) return false;
    try b.verify(windows, out);
    return true;
}

/// A shared round's verify over the windows whose rows found memory, the slices cut to them in order; false: none.
pub fn shared(b: be.Backend, plans: *[]Plan, windows: *[]be.Window, out: *[]be.Verified, total: *u64) !bool {
    var n: usize = 0;
    for (plans.*, windows.*, out.*) |p, w, o| {
        if (cut(b, p.stream, w)) {
            total.* -= w.rows();
            continue;
        }
        plans.*[n] = p;
        windows.*[n] = w;
        out.*[n] = o;
        n += 1;
    }
    plans.* = plans.*[0..n];
    windows.* = windows.*[0..n];
    out.* = out.*[0..n];
    if (n == 0) return false;
    try b.verify(windows.*, out.*);
    return true;
}
