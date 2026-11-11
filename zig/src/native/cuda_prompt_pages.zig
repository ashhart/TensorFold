//! TF_KV_SHARE=n: a CUDA family on the core pool keeps prompts as its pool's pages in one prompt_radix store.
const std = @import("std");
const api = @import("engine_api");
const core = @import("core");
const snaps = @import("cuda_snaps.zig");
const radix = api.prompt_radix;
const kv = core.kv_pool;
const hooks = core.kv_prompts;
const Allocator = std.mem.Allocator;

/// Whether the family keeps prompts this way: it declares the hooks, its streams are on pages and TF_KV_SHARE is set.
pub fn wanted(comptime F: type, loaded: anytype) !bool {
    return could(F, loaded) and try hooks.sharing() > 0;
}

/// Whether TF_KV_SHARE would keep the family's prompts this way: it declares the hooks and its streams are on pages.
pub fn could(comptime F: type, loaded: anytype) bool {
    if (comptime !@hasDecl(F, "share_attach")) return false;
    return loaded.layout != null;
}

/// What the unpaged store's note on a state past its whole budget adds: the setting, where `could` holds.
pub fn hint(comptime F: type, loaded: anytype) []const u8 {
    return if (could(F, loaded)) "; TF_KV_SHARE=n keeps a prompt's rows as the pool's pages instead of copying them" else "";
}

/// The store over the family's pool: its pages in the tree, the rest of a state from the family's `snap_*` decls.
pub const Adapter = struct {
    gpa: Allocator,
    pool: *kv.Pool,
    store: radix.Store,
    detach: *const fn (*anyopaque) void,
    target: *anyopaque,

    /// The family's store, its pool's reclaim evicting the tree's idle pages; null: the family has no pool.
    pub fn open(comptime F: type, gpa: Allocator, target: anytype, page_tokens: u32, snap_limit: usize) !?*Adapter {
        const pool = F.share_pool(target) orelse return null;
        const x = try gpa.create(Adapter);
        errdefer gpa.destroy(x);
        var rules = snaps.cacheRules(F);
        rules.page = page_tokens;
        if (comptime @hasDecl(F, "share_grid")) {
            rules.planned = true;
            rules.grid = F.share_grid;
        }
        rules.min_gap = 256;
        rules.tail = true;
        const copies: radix.Snapshots = .{ .ptr = target, .vtable = &snaps.Snaps(F, null).vtable };
        x.* = .{ .gpa = gpa, .pool = pool, .store = try radix.Store.init(gpa, copies, radix.pagesOf(kv.Pool, pool, pool.page_bytes), rules), .detach = Detach(F).call, .target = target };
        errdefer x.store.deinit();
        x.store.limit(.{ .pages = pool.total(), .snaps = snap_limit });
        try pool.joinReclaim(.{ .ctx = x, .call = reclaim }); // a promise or a hold evicts the tree's idle pages first
        F.share_attach(target, .{ .ctx = x, .vtable = &vtable });
        return x;
    }

    fn Detach(comptime F: type) type {
        return struct {
            fn call(target: *anyopaque) void {
                F.share_attach(target, null);
            }
        };
    }

    /// The family stops calling, the pool's reclaim goes, the tree drops its states and releases its pages.
    pub fn close(x: *Adapter) void {
        x.detach(x.target);
        x.pool.leaveReclaim(x);
        x.store.deinit();
        x.gpa.destroy(x);
    }

    fn of(ctx: *anyopaque) *Adapter {
        return @ptrCast(@alignCast(ctx));
    }

    fn reclaim(ctx: *anyopaque, want: usize) bool {
        return of(ctx).store.reclaim(want);
    }

    fn begin(ctx: *anyopaque, q: hooks.Query, history_len: u32, shared: []const u32, owner: *anyopaque, adopt: *std.ArrayList(u32)) anyerror!hooks.Plan {
        const x = of(ctx);
        const ids = try identities(x.gpa, q.images);
        defer x.gpa.free(ids);
        const p = try x.store.begin(x.gpa, query(q, ids), history_len, shared, owner, adopt);
        return .{ .from = p.from, .marks = p.marks };
    }

    fn keep(ctx: *anyopaque, q: hooks.Query, at: u32, owner: *anyopaque, mine: []const u32, path: []u32) usize {
        const x = of(ctx);
        const ids = identities(x.gpa, q.images) catch return 0; // nothing kept: a later turn prefills it
        defer x.gpa.free(ids);
        return x.store.keep(query(q, ids), at, owner, mine, path).shared;
    }

    /// The family's query as the store takes it, its images as `ids`.
    fn query(q: hooks.Query, ids: []const radix.Identity) radix.Query {
        return .{ .key = q.key, .prompt_len = q.prompt_len, .images = ids, .frame = q.frame, .namespace = q.namespace, .starts = q.starts, .prefilled = q.prefilled };
    }

    /// The images' identities in the store's type (none allocated for text).
    fn identities(gpa: Allocator, images: []const hooks.Image) ![]radix.Identity {
        const out = try gpa.alloc(radix.Identity, images.len);
        for (images, out) |i, *o| o.* = .{ .at = i.at, .tokens = i.tokens, .gh = i.gh, .gw = i.gw, .hash = i.hash };
        return out;
    }

    fn report(ctx: *anyopaque, prompt: usize, from: u32, kept: u64) void {
        of(ctx).store.report(prompt, from, kept);
    }

    fn free(ctx: *anyopaque, marks: []const u32) void {
        of(ctx).gpa.free(marks);
    }

    const vtable: hooks.Prompts.VTable = .{ .begin = begin, .keep = keep, .report = report, .free = free };
};

const testing = std.testing;

/// A family over a real pool whose states are a byte each, its pages its streams' (a host test's stand-in).
const Fake = struct {
    pool: kv.Pool,
    prompts: ?hooks.Prompts = null,
    saved: u8 = 0,
    restored: ?*anyopaque = null,

    fn of(p: *anyopaque) *Fake {
        return @ptrCast(@alignCast(p));
    }
    pub fn share_pool(p: *anyopaque) ?*kv.Pool {
        return &of(p).pool;
    }
    pub fn share_attach(p: *anyopaque, prompts: ?hooks.Prompts) void {
        of(p).prompts = prompts;
    }
    pub fn snap_bytes(_: *anyopaque, _: u32) u64 {
        return 1;
    }
    pub fn snap_save(p: *anyopaque, _: ?*anyopaque, _: u32) anyerror!*anyopaque {
        return @ptrCast(&of(p).saved);
    }
    pub fn snap_restore(p: *anyopaque, _: ?*anyopaque, saved: *anyopaque) anyerror!void {
        of(p).restored = saved;
    }
    pub fn snap_drop(_: *anyopaque, _: *anyopaque) void {}
    pub const snap_lookahead: u32 = 0;
};

test "TF_KV_SHARE's adapter: a kept prompt's pages resume a later one, adopted; closing gives every page back" {
    var f: Fake = .{ .pool = try kv.Pool.init(testing.allocator, 8, 1) };
    defer f.pool.deinit();
    const x = (try Adapter.open(Fake, testing.allocator, &f, 4, 4)).?;
    const p = f.prompts.?;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var lease: kv.Lease = .{};
    try f.pool.promise(&lease, 3);
    try f.pool.cover(&lease, 3, null);
    var path: [2]u32 = undefined;
    var owner: u8 = 0;
    try testing.expectEqual(@as(usize, 2), p.keep(.of(&prompt), 8, &owner, lease.ids.items[0..2], &path));
    try testing.expectEqualSlices(u32, lease.ids.items[0..2], &path);
    f.pool.finish(&lease);
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try p.begin(.of(&prompt), 10, &.{}, &owner, &adopt);
    defer p.free(plan.marks);
    try testing.expectEqual(@as(u32, 8), plan.from);
    try testing.expectEqualSlices(u32, &path, adopt.items);
    try testing.expect(f.restored != null);
    for (adopt.items) |id| f.pool.release(id);
    x.close();
    try testing.expect(f.prompts == null and f.pool.idle() and f.pool.available() == 8);
}

test "TF_KV_SHARE's adapter: a picture under the same key words reaches the store, so text pages share and image pages don't" {
    var f: Fake = .{ .pool = try kv.Pool.init(testing.allocator, 8, 1) };
    defer f.pool.deinit();
    const x = (try Adapter.open(Fake, testing.allocator, &f, 4, 4)).?;
    defer x.close();
    const p = f.prompts.?;
    const prompt = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; // rows 4..8 an image's, its key words alike for any picture
    const a = [_]hooks.Image{.{ .at = 4, .tokens = 4, .gh = 2, .gw = 2, .hash = 0xa }};
    const b = [_]hooks.Image{.{ .at = 4, .tokens = 4, .gh = 2, .gw = 2, .hash = 0xb }};
    var qa: hooks.Query = .of(&prompt);
    qa.images = &a;
    var qb: hooks.Query = .of(&prompt);
    qb.images = &b;
    var la: kv.Lease = .{};
    defer f.pool.finish(&la);
    try f.pool.promise(&la, 2);
    try f.pool.cover(&la, 2, null);
    var lb: kv.Lease = .{};
    defer f.pool.finish(&lb);
    try f.pool.promise(&lb, 2);
    try f.pool.cover(&lb, 2, null);
    var path: [2]u32 = undefined;
    var owner: u8 = 0;
    try testing.expectEqual(@as(usize, 2), p.keep(qa, 8, &owner, la.ids.items, &path));
    try testing.expectEqual(@as(usize, 2), p.keep(qb, 8, &owner, lb.ids.items, &path));
    try testing.expectEqual(la.ids.items[0], path[0]); // the text page: a's
    try testing.expectEqual(lb.ids.items[1], path[1]); // the image's page: b's own
    var adopt: std.ArrayList(u32) = .empty;
    defer adopt.deinit(testing.allocator);
    const plan = try p.begin(qa, 10, &.{}, &owner, &adopt); // a again resumes a's pages
    defer p.free(plan.marks);
    try testing.expectEqualSlices(u32, la.ids.items, adopt.items);
    for (adopt.items) |id| f.pool.release(id);
}

test "TF_KV_SHARE: a state past the unpaged cache's whole budget names it where the family on pages could keep it" {
    const l: kv.Layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 4, .snap_bytes = 0 };
    const paged: struct { layout: ?kv.Layout } = .{ .layout = l };
    const windows: struct { layout: ?kv.Layout } = .{ .layout = null };
    try testing.expect(std.mem.indexOf(u8, hint(Fake, paged), "TF_KV_SHARE=n") != null);
    try testing.expectEqualStrings("", hint(Fake, windows)); // a window a stream: the store cannot share
    try testing.expectEqualStrings("", hint(struct {}, paged)); // a family without the store's hooks
}
