//! Prompt-cache Store open during CUDA Host openWith (snap_save families).
const api = @import("engine_api");
const nemotron = @import("nemotron");
const snaps = @import("cuda_snaps.zig");
const cuda_disk = @import("cuda_disk.zig");
const kv_pages = @import("cuda_kv_pool.zig");
const Allocator = @import("std").mem.Allocator;

/// Opens h.store when the family copies prompt states. Inner errdefers clean a half-open store.
pub fn open(h: anytype, comptime F: type, gpa: Allocator, loaded: anytype, o: api.Open, a: Allocator, io: @import("std").Io, g: anytype, kernels: ?[]const u8, kept: anytype, cache: anytype, share: bool) !void {
    if (@hasDecl(F, "snap_save") and cache > 0 and !kv_pages.ownCache(loaded) and !share) {
        const pc = api.prompt_cache;
        const store = try gpa.create(pc.Store);
        errdefer gpa.destroy(store);
        // rules and budget from the family's snap_* declarations (cuda_snaps.zig)
        const image: ?*const pc.rows.Image = if (F == nemotron.native) &cuda_disk.image else if (comptime cuda_disk.keptDisk(F)) |K| &K.image else null;
        const copies: pc.Snapshots = .{ .ptr = loaded.target, .vtable = &snaps.Snaps(F, image).vtable };
        store.* = pc.Store.init(gpa, copies, snaps.passRules(F, @import("cuda_startup.zig").rows(F, loaded)), if (@hasDecl(F, "snap_slots")) cache else try snaps.cacheBudget(F, loaded.target, cache));
        errdefer {
            const disk = store.disk;
            store.deinit();
            if (disk) |rows| rows.deinit();
            h.store = null;
        }
        try kv_pages.store(F, store, loaded, kept); // its snapshot slots, before a disk probe takes one
        h.store = store;
        if (o.learn_conversations) if (o.learn) |root| {
            if (F == nemotron.native) try cuda_disk.attach(a, io, &g.ctx, o, root, kernels, loaded, store);
            if (comptime cuda_disk.keptDisk(F)) |K| try cuda_disk.attachKept(K, F.model_type, a, io, &g.ctx, o, root, kernels, loaded, store);
        };
    }
}
