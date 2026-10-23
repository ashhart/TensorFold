//! The prompt cache's HIP side: pages and snapshot slots for core/prompt_radix.zig; rank 0 decides, other ranks apply.

const std = @import("std");
const hip = @import("hip");
const radix = @import("engine_api").prompt_radix;
const pages = @import("pages.zig");
const snapshots = @import("snapshots.zig");
const protocol = @import("protocol.zig");
const Allocator = std.mem.Allocator;

/// Rows from which a single rank synchronizes and polls cancellation after each prompt layer.
pub const poll_rows = 512;

/// TP ranks stop together between chunks; only a long single-rank chunk polls its layer edges.
pub fn pollsLayers(world: usize, rows: usize) bool {
    return world == 1 and rows >= poll_rows;
}

/// The cache over family side `F`: `F.Owner` is what a stream gives (`F.idOf`, `F.table`); the hooks copy its state.
pub fn Prefix(comptime F: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        f: *F,
        store: radix.Store,
        slots: snapshots.Slots,
        /// Rank 0's link to the other ranks, which mirror every page swap and snapshot slot.
        tp: *protocol.Sender,
        /// The slot the last restore copied from (reset as a pass begins): the message that begins the pass names it.
        restored: u32 = protocol.no_snapshot,

        /// The cache over `f`, holding pages of `ids` (each `page_bytes`) and snapshots on `d`, under `rules`.
        pub fn init(
            gpa: Allocator,
            f: *F,
            d: *const hip.Runtime,
            ids: *pages.Ids,
            page_bytes: u64,
            rules: radix.Rules,
            tp: *protocol.Sender,
        ) !*Self {
            const p = try gpa.create(Self);
            errdefer gpa.destroy(p);
            p.* = .{ .gpa = gpa, .f = f, .store = undefined, .slots = snapshots.Slots.init(gpa, d), .tp = tp };
            p.store = try radix.Store.init(gpa, .{ .ptr = p, .vtable = &family }, ids.pages(page_bytes), rules);
            return p;
        }

        pub fn deinit(p: *Self) void {
            p.store.deinit();
            p.slots.deinit();
            p.gpa.destroy(p);
        }

        /// The tree may hold `max_pages` pages and `snaps` snapshots (none: the slots' buffers go too).
        pub fn keepPages(p: *Self, max_pages: usize, snaps: usize) void {
            p.store.limit(.{ .pages = max_pages, .snaps = snaps });
            if (p.store.tree.snaps == 0) p.slots.clear();
        }

        /// Before a prompt pass: the longest resumable state restored into the stream, its pages in `adopt`, the marks.
        pub fn begin(
            p: *Self,
            prompt: []const u32,
            history_len: u32,
            shared: []const u32,
            o: *F.Owner,
            adopt: *std.ArrayList(u32),
        ) !radix.Plan {
            return p.store.begin(p.gpa, prompt, history_len, shared, o, adopt);
        }

        /// At `at` the tree takes the pages it lacks and the state; the stream swaps in the pages the tree had.
        pub fn keep(p: *Self, o: *F.Owner, prompt: []const u32, at: usize) !void {
            if (!p.store.on()) return;
            const n = at / pages.tokens;
            const path = try p.gpa.alloc(u32, n);
            defer p.gpa.free(path);
            const table = F.table(o);
            const kept = p.store.keep(prompt, @intCast(at), o, table[0..n], path);
            const own = path[0..kept.shared];
            if (!std.mem.eql(u32, own, table[0..kept.shared])) {
                try F.adopt(p.f, o, own);
                try p.tp.pages(F.idOf(o), 0, own);
            }
        }

        /// Slot `slot` (made if new) takes the stream's state: a save on rank 0, a mirror on the others.
        pub fn takeSlot(p: *Self, slot: u32, o: *F.Owner) !void {
            try F.save(p.f, o, try p.slots.buffer(slot, F.snapBytes(p.f)));
        }

        /// Slot `slot`'s state back into the stream.
        pub fn putSlot(p: *Self, slot: u32, o: *F.Owner) !void {
            try F.restore(p.f, o, try p.slots.get(slot));
        }

        fn slotOf(saved: radix.Saved) u32 {
            return @intCast(@intFromPtr(saved) - 1);
        }

        fn of(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn owner(ptr: ?*anyopaque) *F.Owner {
            return @ptrCast(@alignCast(ptr.?));
        }

        fn bytesFn(ptr: *anyopaque, _: u32) u64 {
            return F.snapBytes(of(ptr).f);
        }

        /// The stream's state copied into a free slot; the other ranks take their own into the same slot.
        fn save(ptr: *anyopaque, o: ?*anyopaque, _: u32) anyerror!radix.Saved {
            const p = of(ptr);
            const slot = try p.slots.claim();
            errdefer p.slots.drop(slot);
            try p.takeSlot(slot, owner(o));
            try p.tp.snap(F.idOf(owner(o)), slot);
            return @ptrFromInt(@as(usize, slot) + 1);
        }

        fn restore(ptr: *anyopaque, o: ?*anyopaque, saved: radix.Saved) anyerror!void {
            const p = of(ptr);
            try p.putSlot(slotOf(saved), owner(o));
            p.restored = slotOf(saved);
        }

        fn dropSaved(ptr: *anyopaque, saved: radix.Saved) void {
            of(ptr).slots.drop(slotOf(saved));
        }

        const family: radix.Snapshots.VTable = .{
            .bytes = bytesFn,
            .save = save,
            .restore = restore,
            .drop = dropSaved,
        };
    };
}
