//! Retained target features append after the canonical state/KV image, independently of its copy traversal.
const features = @import("cuda_cached_features.zig");

/// A walker queues the canonical image, then the tail, before the caller's one completion wait.
pub fn complete(e: anytype, copy: anytype, walker: anytype, tail: anytype) !void {
    try walker.transfer(e, copy);
    try tail.transfer(e, copy);
    try e.stream.synchronize();
}

pub fn bytes(e: anytype, state_bytes: u64) u64 {
    return state_bytes + features.current(e).bytes();
}

pub fn same(e: anytype, copy: anytype, state_bytes: u64) bool {
    return features.same(e, copy.features) and copy.len == state_bytes + copy.features.bytes();
}

pub const Save = struct {
    layout: features.Layout,
    source: u64,
    at: u64,

    pub fn bytes(self: Save) u64 {
        return self.at + self.layout.bytes();
    }

    /// The caller's original save wait follows every state, KV and retained-feature write.
    pub fn transfer(self: Save, e: anytype, copy: anytype) !void {
        if (self.layout.bytes() > 0) {
            if (copy.dev) |b|
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(b.ptr + self.at, self.source, self.layout.bytes(), e.stream.handle), "cuMemcpyDtoDAsync")
            else
                try e.d.check(e.d.api.cuMemcpyDtoHAsync_v2(copy.host.bytes.ptr + self.at, self.source, self.layout.bytes(), e.stream.handle), "cuMemcpyDtoHAsync");
        }
    }
};

pub fn save(e: anytype, cache: anytype, at: usize, state_bytes: u64) !Save {
    const layout = features.current(e);
    return .{ .layout = layout, .source = try features.source(e, cache, at, layout), .at = state_bytes };
}

pub const Restore = struct {
    layout: features.Layout,
    destination: u64,
    at: u64,

    /// The caller keeps its reading flag and error drain until this original restore wait succeeds.
    pub fn transfer(self: Restore, e: anytype, copy: anytype) !void {
        if (self.layout.bytes() > 0) {
            if (copy.dev) |b|
                try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(self.destination, b.ptr + self.at, self.layout.bytes(), e.stream.handle), "cuMemcpyDtoDAsync")
            else
                try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(self.destination, copy.host.bytes.ptr + self.at, self.layout.bytes(), e.stream.handle), "cuMemcpyHtoDAsync");
        }
    }

    /// Publication stays after the wait, reading-flag clear and cache metadata updates in restoreCache.
    pub fn publish(self: Restore, e: anytype, cache: anytype, at: usize) !void {
        try features.restored(e, cache, at, self.layout);
    }
};

pub fn restore(e: anytype, cache: anytype, copy: anytype, state_bytes: u64) !Restore {
    return .{ .layout = copy.features, .destination = try features.destination(e, cache, copy.features), .at = state_bytes };
}
