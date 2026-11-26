//! Enabled-only numeric frame receipts join actual CUDA forward buckets to the shared core's accepted paths.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const attribution = @import("cuda").round_attribution;
pub const install = attribution.Trace.install;
const platform = @import("sys");
pub const Shape = @import("cuda_shapes.zig").Shape;
pub const Trace = struct {
    const Frame = struct { first: usize, windows: usize, offered: usize, bucket: usize, ms: f64 };
    frames: [@import("cuda_round.zig").max_segs]Frame = undefined,
    len: usize = 0,
    extra: ?*attribution.Trace = null,
    legacy: bool = true,
    publishing: attribution.Scope = .{},

    pub fn enabled() bool {
        return platform.environment.read(bool, "TF_TIMING", false) catch false;
    }

    pub fn create(a: std.mem.Allocator, on: bool) !?*Trace {
        if (!on) return null;
        const trace = try a.create(Trace);
        trace.* = .{};
        return trace;
    }

    pub fn observer(t: *Trace) be.RoundObserver {
        return .{ .ptr = t, .done = done, .begin = if (t.extra != null) begin else null, .abort = if (t.extra != null) aborted else null, .committed = if (t.extra != null) committed else null, .publish = if (t.extra != null) publish else null };
    }

    pub fn reset(t: *Trace) void {
        t.len = 0;
    }

    pub fn frame(t: *Trace, first: usize, windows: usize, offered: usize, bucket: usize, ms: f64) void {
        std.debug.assert(t.len < t.frames.len);
        if (t.extra) |x| if (x.frames.items.len > 0) {
            const last = &x.frames.items[x.frames.items.len - 1];
            last.live_rows = offered;
            last.windows = windows;
        };
        t.frames[t.len] = .{ .first = first, .windows = windows, .offered = offered, .bucket = bucket, .ms = ms };
        t.len += 1;
    }

    pub fn deinit(t: *Trace, a: std.mem.Allocator) void {
        if (t.extra) |x| x.deinit();
        a.destroy(t);
    }
    fn begin(ptr: *anyopaque, streams: []const *lanes.Stream) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        const x = t.extra.?;
        x.begin();
        const a = x.round_arena.allocator();
        for (streams) |stream| {
            const stream_id = hex(a, attribution.digest(stream.id)) catch {
                x.fail("stream identity allocation");
                return;
            };
            const prompt_id = hex(a, tokenHash(stream.prompt())) catch {
                x.fail("prompt identity allocation");
                return;
            };
            const sampling = std.json.Stringify.valueAlloc(a, .{ .sampling = stream.sampling, .penalties = stream.penalties, .drafts = stream.drafts }, .{}) catch {
                x.fail("sampling identity allocation");
                return;
            };
            x.requests.append(x.a, .{ .stream = stream_id, .prompt_sha256 = prompt_id, .prompt_tokens = stream.prompt_len, .sampling_json = sampling, .cached = stream.cache_len }) catch x.fail("request metadata allocation");
        }
    }
    fn aborted(ptr: *anyopaque) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.extra) |x| if (x.active) x.finish(true);
    }
    fn publish(ptr: *anyopaque, starting: bool) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (starting) t.publishing = attribution.scope(.state_publication) else t.publishing.end();
    }
    fn committed(ptr: *anyopaque, stream: *lanes.Stream, kind: []const u8, offered: usize, path: []const u32, ids: []const u32) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        const x = t.extra.?;
        const a = x.round_arena.allocator();
        const stream_id = hex(a, attribution.digest(stream.id)) catch {
            x.fail("stream identity allocation");
            return;
        };
        const kept = a.dupe(u32, path) catch {
            x.fail("path allocation");
            return;
        };
        const got = a.dupe(u32, ids) catch {
            x.fail("commit allocation");
            return;
        };
        x.windows.append(x.a, .{ .stream = stream_id, .kind = kind, .offered_rows = offered, .proposal_nodes = if (std.mem.eql(u8, kind, "forced")) 0 else offered - 1, .cached_after = stream.cache_len, .pending_after = stream.pending, .rolled_back_rows = offered - path.len, .connected_accepted_depth = if (path.len > 0 and !std.mem.eql(u8, kind, "forced")) path.len - 1 else 0, .kept_path = kept, .committed_ids = got }) catch x.fail("window metadata allocation");
    }
    fn done(ptr: *anyopaque, paths: []const []const u32) void {
        const t: *Trace = @ptrCast(@alignCast(ptr));
        if (t.extra) |x| x.finish(false);
        if (t.legacy) for (t.frames[0..t.len]) |frame_| {
            var kept: usize = 0;
            for (paths[frame_.first..][0..frame_.windows]) |path| kept += path.len;
            std.debug.print("[timing] verify_round offered={d} kept={d} bucket={d} ms={d:.3}\n", .{ frame_.offered, kept, frame_.bucket, frame_.ms });
        };
        t.reset();
    }
};

pub fn configure(a: std.mem.Allocator, g: anytype, extra: ?*attribution.Trace) !struct { on: bool, trace: ?*Trace } {
    const on = Trace.enabled();
    const trace = try Trace.create(a, extra != null or (on and g.hidden == 5120 and g.layers == 64 and g.experts == 0));
    if (trace) |t| {
        t.extra = extra;
        t.legacy = on;
    }
    return .{ .on = on, .trace = trace };
}
fn hex(a: std.mem.Allocator, bytes: [32]u8) ![]const u8 {
    return a.dupe(u8, &std.fmt.bytesToHex(bytes, .lower));
}
fn tokenHash(tokens: []const u32) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var b: [4]u8 = undefined;
    for (tokens) |v| {
        std.mem.writeInt(u32, &b, v, .little);
        h.update(&b);
    }
    return h.finalResult();
}

pub fn keep(self: anytype, windows: []const be.Window, paths: []const []const u32) !void {
    const attribution_scope = if (comptime @import("profile").round_attribution.enabled) attribution.scope(.keep_rollback) else {};
    defer if (comptime @import("profile").round_attribution.enabled) attribution_scope.end();
    const t0 = std.Io.Clock.awake.now(self.io);
    defer @import("cuda_draft.zig").timing("keep", self.io, t0);
    for (windows, paths) |w, path| {
        const cache = try self.cacheOf(w.stream);
        if (cache.last == null) continue; // already kept: nothing to roll back
        try cache.keep(self.e, path);
    }
}
