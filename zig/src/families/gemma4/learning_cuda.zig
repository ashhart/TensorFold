//! A CUDA learner borrows the target's immutable weights and owns a separate fresh-sequence forward state.
const std = @import("std");
const C = @import("cuda");
const core = @import("core");
const fwd = @import("cuda_forward.zig");
const profile = @import("learning_profile.zig");
const K = @import("learning_cuda_kernels.zig").Kernels(C, core);
const B = K.Backend;
pub const Owner = struct {
    const Self = @This();
    pub const Runtime = @import("learning_runtime.zig").Runtime(B, K);
    pub const probability_mode: u32 = 2;
    a: std.mem.Allocator,
    model: *fwd.Model,
    stream: C.Stream,
    kernels: *const @import("cuda_kernels.zig").Kernels,
    backend: B,
    r: *Runtime,
    dir: []u8,
    learned: usize,
    unified: bool,
    wrapped: bool,
    pub fn open(a: std.mem.Allocator, io: std.Io, context: *const C.Context, m: *fwd.Model, stream: C.Stream, k: *const @import("cuda_kernels.zig").Kernels, dir: []const u8, budget: u64) !*Self {
        const c = &m.w.config;
        try profile.check(c);
        const self = try a.create(Self);
        errdefer a.destroy(self);
        const owned = try a.dupe(u8, dir);
        errdefer a.free(owned);
        self.* = .{ .a = a, .model = m, .stream = stream, .kernels = k, .backend = .{ .driver = context.d, .stream = stream, .context = context }, .r = undefined, .dir = owned, .learned = c.learned, .unified = c.geo.unified(), .wrapped = c.wrapped };
        self.r = try Runtime.init(a, &self.backend, profile.Shape.from(c), budget);
        errdefer self.r.close() catch {};
        if (c.learned != 0) {
            var reader = try core.Checkpoint.openModel(a, io, dir);
            defer reader.close();
            try self.r.pockets.load(&reader, c);
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
        self.model.learning = null;
        self.a.free(self.dir);
        self.a.destroy(self);
    }
    pub fn trainBytes(self: *Self, rows: usize) !u64 {
        return fwd.Model.openBytes(self.a, self.model.d, self.stream, self.model.w, rows, 1);
    }
    pub fn trainOpen(self: *Self, rows: usize) !Train {
        var m = try fwd.Model.init(self.a, self.model.d, self.stream, self.model.w, rows, 1);
        m.graphed = false;
        return .{ .owner = self, .model = m };
    }
    pub fn frequencies(_: *Self, l: usize, out: []f32) void {
        fwd.inverseFrequencies(l % 6 == 5, out);
    }
    pub const Train = struct {
        owner: *Self,
        model: fwd.Model,
        pub fn deinit(t: *Train) void {
            t.model.deinit();
        }
        pub fn forward(t: *Train, listener: Runtime.Hook, ids: []const u32) !void {
            t.model.learning = listener;
            defer t.model.learning = null;
            var first: usize = 0;
            while (first < ids.len) {
                const n = @min(16, ids.len - first);
                try t.model.forward(t.owner.kernels, t.owner.stream, &.{.{ .slot = 0, .p0 = first, .tokens = ids[first..][0..n] }}, true, 0, n);
                try t.owner.backend.drain();
                first += n;
            }
        }
    };
};
