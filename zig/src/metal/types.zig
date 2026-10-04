//! Metal value types and enum constants, as the SDK headers define them.

pub const Size = extern struct {
    width: usize,
    height: usize = 1,
    depth: usize = 1,

    pub fn of(w: usize, h: usize, d: usize) Size {
        return .{ .width = w, .height = h, .depth = d };
    }
};

pub const Range = extern struct {
    location: usize,
    length: usize,
};

/// MTLResourceOptions: storage mode << 4, hazard tracking << 8.
pub const ResourceOptions = struct {
    pub const shared: usize = 0 << 4;
    pub const private: usize = 2 << 4;
    pub const hazard_default: usize = 0 << 8;
    pub const untracked: usize = 1 << 8;
    pub const tracked: usize = 2 << 8;
};

pub const DispatchType = enum(usize) { serial = 0, concurrent = 1 };

pub const BarrierScope = struct {
    pub const buffers: usize = 1 << 0;
};

pub const ResourceUsage = struct {
    pub const read: usize = 1 << 0;
    pub const write: usize = 1 << 1;
};

pub const CommandBufferStatus = enum(usize) {
    not_enqueued = 0,
    enqueued = 1,
    committed = 2,
    scheduled = 3,
    completed = 4,
    @"error" = 5,
    _,
};

/// MTLLanguageVersion: (major << 16) + minor.
pub fn languageVersion(major: u32, minor: u32) usize {
    return (@as(usize, major) << 16) + minor;
}

pub const MathMode = enum(isize) { safe = 0, relaxed = 1, fast = 2 };

pub const MathFunctions = enum(isize) { fast = 0, precise = 1 };

pub const IndirectCommandType = struct {
    pub const concurrent_dispatch: usize = 1 << 5;
    pub const concurrent_dispatch_threads: usize = 1 << 6;
};
