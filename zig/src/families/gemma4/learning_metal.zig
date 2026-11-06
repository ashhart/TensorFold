//! A Metal learner borrows immutable projections and owns its fresh cache, scratch and ordered backward session.
const std = @import("std");
const M = @import("metal");
const core = @import("core");
const profile = @import("learning_profile.zig");
const state = @import("state.zig");
const cfg = @import("config.zig");
const K = @import("learning_metal_kernels.zig").Kernels(M, core, @import("living_sources"));
const B = K.Backend;
pub const Owner = struct {
    const Self = @This();
    pub const Runtime = @import("learning_runtime.zig").Runtime(B, K);
    pub const probability_mode: u32 = 0;
    a: std.mem.Allocator,
    model: *@import("model.zig").Model,
    commands: M.CommandSession,
    backend: B,
    r: *Runtime,
    dir: []u8,
    learned: usize,
    unified: bool,
    wrapped: bool,
    pub fn open(a: std.mem.Allocator, m: *@import("model.zig").Model, dir: []const u8, budget: u64) !*Self {
        const c = &m.config;
        try profile.check(c);
        const self = try a.create(Self);
        errdefer a.destroy(self);
        const owned = try a.dupe(u8, dir);
        errdefer a.free(owned);
        self.* = .{ .a = a, .model = m, .commands = M.CommandSession.init(a, m.queue), .backend = .{ .device = m.device, .queue = m.queue }, .r = undefined, .dir = owned, .learned = c.learned, .unified = c.geo.unified(), .wrapped = c.wrapped };
        self.backend.commands = &self.commands;
        self.r = try Runtime.init(a, &self.backend, profile.Shape.from(c), budget);
        errdefer self.r.close() catch {};
        if (c.learned != 0) {
            try self.r.pockets.load(&m.checkpoint, c);
            try self.r.kernels.ensure(&self.backend);
        }
        m.learning = self.r.listener();
        return self;
    }
    pub fn prepareClose(self: *Self) !void {
        try self.r.prepareClose();
    }
    pub fn close(self: *Self) !void {
        try self.r.close();
        try self.commands.deinit();
        self.model.learning = null;
        self.a.free(self.dir);
        self.a.destroy(self);
    }
    pub fn trainBytes(self: *Self, rows: usize) !u64 {
        return state.cacheBytes(self.model.config.geo, rows) + state.scratchBytes(self.model.config.geo, rows);
    }
    pub fn trainOpen(self: *Self, rows: usize) !Train {
        var cache = try state.Cache.init(self.model.device, self.model.config.geo, rows);
        errdefer cache.deinit();
        const scratch = try state.Scratch.init(self.model.device, self.model.config.geo, rows);
        return .{ .owner = self, .cache = cache, .scratch = scratch };
    }
    pub fn frequencies(_: *Self, l: usize, out: []f32) void {
        cfg.inverseFrequencies(l % 6 == 5, out);
    }
    pub const Train = struct {
        owner: *Self,
        cache: state.Cache,
        scratch: state.Scratch,
        pub fn deinit(t: *Train) void {
            t.scratch.deinit();
            t.cache.deinit();
        }
        pub fn forward(t: *Train, listener: Runtime.Hook, ids: []const u32) !void {
            const m = t.owner.model;
            const old = m.learning;
            defer m.learning = old;
            m.learning = listener;
            t.cache.len = 0;
            var first: usize = 0;
            while (first < ids.len) {
                const n = @min(16, ids.len - first);
                try @import("forward.zig").run(m, &t.scratch, &.{.{ .cache = &t.cache, .rows = n }}, ids[first..][0..n]);
                first += n;
            }
        }
    };
};
