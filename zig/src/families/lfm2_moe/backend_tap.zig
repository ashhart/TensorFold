//! LFM2's tapped rows: where a forward captures them, and the drafter's view of the latest ones.
const lanes = @import("lanes");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const steps = @import("prefill_steps.zig");
const backend = @import("backend.zig");
const Metal = backend.Metal;
const be = lanes.backend;

/// Where a stream's forward rows from `start` are captured, the hold it follows folded into `wait`.
pub fn tapFor(self: *Metal, stream: *lanes.Stream, start: usize, rows: usize, wait: *?fwd.Wait) !fwd.Tap {
    if (stream.head_context == false) return .none;
    const src = self.stage_source orelse return .scratch;
    if (self.scratch.taps == null) return .scratch;
    const stage = (try src.bind(src.ptr, stream, start, rows)) orelse
        return if (stream.headDrafts()) error.LfmTapsNotHeld else .none;
    if (stage.wait) |w| wait.* = if (wait.*) |x| (if (x.value >= w.value) x else w) else w;
    return .{ .stage = stage.place };
}

/// The stream's latest rows after a forward that captured them by `tap` (none: not held).
pub fn tapped(self: *const Metal, cache: *const st.Cache, rows: usize, tap: fwd.Tap, row: usize) ?st.TapRows {
    if (self.scratch.taps == null or tap == .none) return null;
    const stage: ?usize = switch (tap) {
        .stage => |p| p.off,
        else => null,
    };
    return .{ .start = cache.len - rows, .row = row, .rows = rows, .gen = self.gen, .stage = stage };
}

/// A tapped row's bytes: every tap's hidden row, bf16.
pub fn tapBytes(self: *const Metal) usize {
    return if (self.scratch.taps) |t| t.count * c.hidden * 2 else 0;
}

/// Where a features request reads its rows: the latest capture, a stepped chunk's stage, or its scratch rows.
pub const Place = enum { capture, stage, scratch };
pub const Source = struct { rows: st.TapRows, place: Place };

/// The rows [start, start + count) a features request can read now, or null when none holds them.
pub fn source(tap: ?st.TapRows, chunk: ?st.TapRows, now: u64, start: u64, count: u32) !?Source {
    if (within(tap, start, count)) {
        const held = tap.?;
        if (held.gen != 0 and held.gen != now) return error.StaleLfmTaps;
        return .{ .rows = held, .place = .capture };
    }
    // a stepped pass's latest chunk: its rows stay in the scratch or the lane's stage until the next forward (#3021)
    if (chunk) |k| if (steps.holds(.{ .start = k.start, .rows = k.rows, .gen = k.gen }, now, start, count))
        return .{ .rows = k, .place = if (k.stage != null) .stage else .scratch };
    return null;
}

/// Cache rows [start, start + count) of the stream's latest tapped rows, on the device.
pub fn featuresFn(ptr: *anyopaque, stream: *lanes.Stream, layers: []const u32, start: u64, count: u32) !be.Features {
    const self = Metal.cast(ptr);
    const t = self.scratch.taps orelse return error.UnpreparedLfmTaps;
    if (layers.len != t.count) return error.UnpreparedLfmTaps;
    const cache = try self.cacheOf(stream);
    const width = self.tapBytes();
    const held = (try source(cache.tap, cache.chunk_tap, self.gen, start, count)) orelse return error.LfmTapsNotHeld;
    const rows = held.rows;
    const shift = @as(usize, @intCast(start - rows.start));
    // rows the lane owns on the device survive other streams' forwards; the scratch and its copy do not
    const at: struct { buf: mtl.Buffer, off: usize } = switch (held.place) {
        .capture => if (rows.stage) |off| .{ .buf = try stageBuffer(self, stream), .off = off + shift * width } else .{
            .buf = if (rows.gen == 0) cache.last_tap orelse return error.LfmTapsNotHeld else t.buf,
            .off = (rows.row + shift) * width,
        },
        .stage => .{ .buf = try stageBuffer(self, stream), .off = rows.stage.? + shift * width },
        .scratch => .{ .buf = t.buf, .off = (rows.row + shift) * width },
    };
    return .{ .buffer = @intFromPtr(at.buf.id), .offset = at.off, .rows = count, .row_bytes = @intCast(width), .space = .device, .dtype = .bf16 };
}

fn stageBuffer(self: *Metal, stream: *lanes.Stream) !mtl.Buffer {
    const src = self.stage_source orelse return error.LfmTapsNotHeld;
    return src.buffer(src.ptr, stream) orelse error.LfmTapsNotHeld;
}

fn within(tap: ?st.TapRows, start: u64, count: u32) bool {
    const h = tap orelse return false;
    return count != 0 and start >= h.start and start + count <= h.start + h.rows;
}

/// The prompt's last row's taps, copied beside the stream's caches for its first draft (gen 0: kept there).
pub fn keepLastTap(self: *Metal, cache: *st.Cache, row: usize, position: usize) !void {
    const width = self.tapBytes();
    if (width == 0) return;
    // with a lane stage the scratch holds destinations, not rows: a stage tap keeps its own anchor and none is needed
    if (self.stage_source != null) return;
    if (cache.last_tap == null) cache.last_tap = try cache.memory.alloc(width);
    @memcpy(cache.last_tap.?.contents()[0..width], (self.scratch.taps.?.buf.contents() + row * width)[0..width]);
    cache.tap = .{ .start = position, .row = 0, .rows = 1, .gen = 0 };
}

test {
    _ = @import("backend_tap_test.zig");
}
