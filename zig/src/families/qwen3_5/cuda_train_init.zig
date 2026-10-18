//! One partial constructor owner drains before freeing children, their registry and their module parents.
pub fn Partial(comptime T: type) type {
    return struct {
        t: *T,
        k: bool = false,
        gk: bool = false,
        states: usize = 0,
        mapped: usize = 0,
        pub fn deinit(p: *@This()) void {
            const t = p.t;
            const gpa = t.gpa;
            t.drain();
            inline for (.{ "host", "ys", "delta", "push", "tails" }, 0..) |field, i|
                if (i < p.mapped) @field(t, field).deinit();
            for (t.owned.items) |*buffer| buffer.free();
            t.owned.deinit(gpa);
            for (t.states[0..p.states]) |*state| state.deinit(gpa);
            gpa.free(t.full);
            gpa.free(t.light);
            gpa.free(t.w);
            gpa.free(t.resid);
            if (p.gk) t.gk.deinit();
            if (p.k) t.k.deinit();
            gpa.destroy(t);
        }
    };
}
