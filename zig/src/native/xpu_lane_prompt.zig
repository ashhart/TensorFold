//! Stepped and solo-chunked prompt fill for `native/xpu_lane.zig` sequence backends.
const std = @import("std");
const lanes = @import("lanes");

pub fn Prompt(comptime Self: type, comptime S: type) type {
    return struct {
        const has_solo = @hasDecl(S, "soloRows");

        pub fn open(self: *Self, s: *lanes.Stream, ids: []const u32) !*Self.Lane {
            const e = self.e;
            if (ids.len == 0 or ids.len + s.max_new + 1 > e.max_len) return error.PromptTooLong;
            const gop = try self.lane_of.getOrPut(self.gpa, s);
            if (gop.found_existing) self.dropSeq(gop.value_ptr.seq);
            const own = self.own_free;
            gop.value_ptr.* = .{ .seq = if (own) &e.own else e.newSeq() catch |err| {
                self.lane_of.removeByPtr(gop.key_ptr);
                return err;
            } };
            if (own) self.own_free = false;
            const l = gop.value_ptr;
            e.bind(l.seq);
            s.cached = 0;
            var at: usize = 0;
            if (s.reuse.saved) |saved| {
                const snap: *S.Engine.Snapshot = @ptrCast(@alignCast(saved));
                if (snap.at < ids.len) {
                    try e.snapRestore(snap);
                    at = snap.at;
                    s.cached = snap.at;
                } else s.reuse_failed = true;
            }
            if (at == 0) try e.reset();
            if (comptime @hasDecl(S.Engine, "pictures")) e.forgetPictures();
            l.filled = at;
            s.fill_at.store(@intCast(at), .release);
            return l;
        }

        pub fn step(self: *Self, s: *lanes.Stream, l: *Self.Lane, max_rows_cap: u32) !bool {
            const ids = s.prompt();
            const at = l.filled orelse return true;
            if (s.isCancelled()) {
                s.keepInterruptedPrefill(@intCast(at));
                return error.Cancelled;
            }
            if (self.stopped()) return error.ShuttingDown;
            const e = self.e;
            e.bind(l.seq);
            const busy: usize = S.promptRows(e);
            const solo: usize = if (comptime has_solo) S.soloRows(e) else busy;
            var rows = lanes.solo_chunk.passRows(&self.lane_of, s, "filled", solo, busy);
            if (max_rows_cap > 0) rows = @min(rows, max_rows_cap);
            var stop = @min(ids.len, at + @max(rows, 1));
            for (s.chunks) |c| if (c > at and c < stop) {
                stop = c;
                break;
            };
            for (s.reuse.marks) |m| if (m > at and m < stop) {
                stop = m;
                break;
            };
            // the stream's picture rows replace its image tokens' embeddings in this window
            if (comptime @hasDecl(S.Engine, "pictures")) try e.pictures(s, s.images, at);
            defer if (comptime @hasDecl(S.Engine, "pictures")) e.clearPictures();
            try e.prefillSpan(ids, at, stop, @intCast(stop - at));
            s.fill_at.store(@intCast(stop), .release);
            const last = stop == ids.len;
            l.filled = if (last) null else stop;
            if (std.mem.indexOfScalar(u32, s.reuse.marks, @intCast(stop)) != null) if (s.reuse.hook) |k| k.at(k.ptr, s, @intCast(stop));
            if (last) {
                if (s.isCancelled()) return error.Cancelled;
                var first: [1]u32 = undefined;
                try self.drawRows(0, 1, s.sampling, &.{ids.len}, &first);
                l.first_handle = self.take(first[0]);
            }
            return last;
        }

        pub fn cancel(self: *Self, s: *lanes.Stream) void {
            if (self.lane_of.getPtr(s)) |l| if (l.filled) |at| s.keepInterruptedPrefill(@intCast(at));
        }
    };
}
