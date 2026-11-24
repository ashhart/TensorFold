//! A Drama request's conditional prompt for its guidance round: one chunk, the first resuming after a cached reference.
const std = @import("std");
const d3 = @import("drama3");
const reference = d3.reference_cache;
const is_cuda = !@hasDecl(d3, "gpu");

pub fn Conditional(comptime Host: type) type {
    return struct {
        const Self = @This();

        h: *Host,
        lane: usize,

        pub fn whole(c: Self) bool {
            const j = &c.h.jobs[c.lane].?;
            return j.state.started and j.state.prompt == j.request.frames.len;
        }

        pub fn chunk(c: Self) !void {
            const lane = &c.h.sessions[c.lane];
            const s = lane.*;
            const j = &c.h.jobs[c.lane].?;
            const r = j.request;
            const state = &j.state;
            if (is_cuda) {
                if (!state.started) try s.t.c.reset(s.t.cs);
                return c.h.refs.prompt(s.e, r.frames, r.reference_rows, state);
            }
            const ref_rows = reference.prefillRows(r.frames.len, r.reference_rows, s.e.o.max_rows);
            if (!state.started) {
                // Invalidate before reset, GPU work or any fallible preparation for a miss.
                if (try lane.prefix.begin(c.h.gpa, r.frames, ref_rows)) {
                    try s.e.restore(&lane.cache);
                    try s.e.beginPromptFrom(r.frames, ref_rows);
                    state.prompt = ref_rows;
                    std.log.info("drama3: reference cache hit ({d} rows restored)", .{ref_rows});
                } else {
                    try s.e.startPrompt(r.frames);
                    state.prompt = 0;
                }
                try s.t.c.reset();
                state.started = true;
            }
            var count = s.e.o.promptRows(r.frames.len - state.prompt);
            const want_snapshot = ref_rows > 0 and !lane.prefix.valid;
            if (want_snapshot and state.prompt < ref_rows) count = @min(count, ref_rows - state.prompt);
            try s.e.promptChunk(r.frames.len, state.prompt, count, null);
            state.prompt += count;
            if (want_snapshot and state.prompt == ref_rows) {
                try s.e.snapshot(&lane.cache);
                lane.prefix.saved();
                std.log.info("drama3: reference cache built ({d} rows)", .{ref_rows});
            }
        }
    };
}
