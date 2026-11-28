//! Page storage owns a mapped host allocation before mapping it or allocating its device state run.
const cuda = @import("cuda");

pub const Storage = struct {
    arena: cuda.DeviceBuffer,
    host: ?cuda.HostBuffer = null,
    base: u64,
    run_dev: ?cuda.DeviceBuffer = null,

    pub fn init(d: *const cuda.Driver, bytes: usize, run_bytes: usize, host: bool) !Storage {
        if (!host) {
            const arena = try cuda.DeviceBuffer.alloc(d, bytes);
            return .{ .arena = arena, .base = arena.ptr };
        }
        var h = cuda.HostBuffer.allocMapped(d, bytes) catch return error.QwenKvHostAllocation;
        errdefer h.free();
        const base = h.device() catch return error.QwenKvHostMapping;
        var run = cuda.DeviceBuffer.alloc(d, run_bytes) catch return error.QwenKvHostRunAllocation;
        errdefer run.free();
        return .{ .arena = .{ .d = d, .ptr = 0, .len = 0 }, .host = h, .base = base, .run_dev = run };
    }

    pub fn deinit(s: *Storage) void {
        if (s.run_dev) |*r| r.free();
        if (s.host) |*h| h.free() else s.arena.free();
    }
};
