//! The lane fake uses the production Qwen27 masked sampler and proposal filter at its device boundary.
const std = @import("std");
const lanes = @import("lanes");
const masks = @import("qwen27_masks");
const draws = @import("qwen_draw");
const be = lanes.backend;
pub const Tree = struct {
    gpa: std.mem.Allocator,
    tokens: []u32,
    parents: []i32,
    scores: []f64,
    pub fn deinit(t: *Tree) void {
        t.gpa.free(t.tokens);
        t.gpa.free(t.parents);
        t.gpa.free(t.scores);
    }
};
pub const Adapter = struct {
    gpa: std.mem.Allocator,
    target: lanes.fake.Fake = undefined,
    ready: bool = false,
    trees: std.AutoHashMapUnmanaged(*lanes.Stream, Tree) = .empty,
    first_masks: usize = 0,
    row_masks: usize = 0,
    filtered: usize = 0,
    pub fn backend(x: *Adapter) be.Backend {
        if (!x.ready) {
            x.target = .{ .gpa = x.gpa };
            x.ready = true;
        }
        return .{ .ptr = x, .vtable = &.{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .tree = tree, .release = release } };
    }
    pub fn deinit(x: *Adapter) void {
        var it = x.trees.valueIterator();
        while (it.next()) |t| t.deinit();
        x.trees.deinit(x.gpa);
        if (x.ready) x.target.deinit();
    }
    fn cast(p: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(p));
    }
    fn prefill(p: *anyopaque, s: *lanes.Stream) !void {
        return cast(p).target.backend().prefill(s);
    }
    fn draw(x: *Adapter, ideal: u32, s: *lanes.Stream, position: u64, mask: []const u32) !u32 {
        var logits: [lanes.fake.vocab]u16 = undefined;
        for (&logits, 0..) |*word, i| {
            const f: f32 = if (i == ideal) 16 else @as(f32, @floatFromInt(i % 7)) / 4;
            word.* = @intCast(@as(u32, @bitCast(f)) >> 16);
        }
        return masks.draw(draws.draw, x.gpa, &logits, s.sampling, position, mask);
    }
    fn first(p: *anyopaque, s: *lanes.Stream, position: u64) !u64 {
        return firstMasked(p, s, position, &.{});
    }
    fn firstMasked(p: *anyopaque, s: *lanes.Stream, position: u64, mask: []const u32) !u64 {
        const x = cast(p);
        const sampling = s.sampling;
        s.sampling = null;
        const handle = x.target.backend().first(s, position) catch |err| {
            s.sampling = sampling;
            return err;
        };
        s.sampling = sampling;
        const ideal = try x.target.backend().read(handle);
        x.target.drawn.items[@intCast(handle)] = try x.draw(ideal, s, position, mask);
        x.first_masks += @intFromBool(mask.len > 0);
        return handle;
    }
    fn queue(p: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) !u64 {
        return cast(p).target.backend().queue(s, feed, position);
    }
    fn read(p: *anyopaque, handle: u64) !u32 {
        return cast(p).target.backend().read(handle);
    }
    fn verify(p: *anyopaque, windows: []const be.Window, out: []be.Verified) !void {
        const x = cast(p);
        const plain = try x.gpa.dupe(be.Window, windows);
        defer x.gpa.free(plain);
        const settings = try x.gpa.alloc(?lanes.Sampling, windows.len);
        defer x.gpa.free(settings);
        for (plain, settings) |*w, *setting| {
            setting.* = w.stream.sampling;
            w.stream.sampling = null;
            w.masks = &.{};
        }
        x.target.backend().verify(plain, out) catch |err| {
            for (plain, settings) |w, setting| w.stream.sampling = setting;
            return err;
        };
        for (plain, settings) |w, setting| w.stream.sampling = setting;
        for (windows, out) |w, o| for (o.sampled, w.positions, 0..) |*token, position, row| {
            const words = (lanes.fake.vocab + 31) / 32;
            const mask = if (w.masks.len > 0) w.masks[row * words ..][0..words] else &.{};
            token.* = try x.draw(token.*, w.stream, position, mask);
            x.row_masks += @intFromBool(mask.len > 0);
        };
    }
    fn keep(p: *anyopaque, windows: []const be.Window, paths: []const []const u32) !void {
        return cast(p).target.backend().keep(windows, paths);
    }
    fn draft(p: *anyopaque, requests: []const be.DraftRequest) !void {
        const x = cast(p);
        try x.target.backend().draft(requests);
        for (requests) |r| {
            if (x.trees.fetchRemove(r.stream)) |entry| {
                var t = entry.value;
                t.deinit();
            }
            if (r.depth == 0) continue;
            const c = r.stream.grammar orelse continue;
            const forced = try x.gpa.alloc(u32, r.depth + 1);
            defer x.gpa.free(forced);
            const n = try c.rules.vtable.drafts(c.rules.ptr, forced);
            const skip: usize = @intFromBool(r.rows == null);
            const length = @min(r.depth, n -| skip);
            if (length == 0) continue;
            var t = Tree{ .gpa = x.gpa, .tokens = try x.gpa.dupe(u32, forced[skip..][0..length]), .parents = undefined, .scores = undefined };
            errdefer x.gpa.free(t.tokens);
            t.parents = try x.gpa.alloc(i32, length);
            errdefer x.gpa.free(t.parents);
            t.scores = try x.gpa.alloc(f64, length);
            errdefer x.gpa.free(t.scores);
            for (t.parents, t.scores, 0..) |*parent, *score, i| {
                parent.* = @as(i32, @intCast(i)) - 1;
                score.* = 0;
            }
            if (length > 1) {
                t.tokens[length - 1] = 96;
                t.parents[length - 1] = -1;
            }
            const pending = if (r.rows == null) switch (r.first.?) {
                .handle => |h| try x.target.backend().read(h),
                .value => |v| v,
            } else null;
            try masks.prune(x.gpa, r.stream.grammar, pending, &t);
            x.filtered += length - t.tokens.len;
            try x.trees.put(x.gpa, r.stream, t);
        }
    }
    fn tree(p: *anyopaque, s: *lanes.Stream, a: std.mem.Allocator) !?lanes.stream.Held {
        const t = cast(p).trees.get(s) orelse return .{ .count = 0, .tokens = try a.alloc(u32, 0) };
        const tokens = try a.dupe(u32, t.tokens);
        errdefer a.free(tokens);
        return .{ .count = @intCast(tokens.len), .tokens = tokens, .parents = try a.dupe(i32, t.parents) };
    }
    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const x = cast(p);
        if (x.trees.fetchRemove(s)) |entry| {
            var t = entry.value;
            t.deinit();
        }
        x.target.backend().release(s);
    }
};
