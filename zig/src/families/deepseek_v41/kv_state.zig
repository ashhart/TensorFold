//! Shared cache ownership and exact transaction snapshots; appended rows are cursor-delimited.
const std = @import("std");
const mtl = @import("metal");
const View = @import("view.zig").View;
pub const owners = [_]u32{ 2, 8, 14, 20 };
pub const ratios = [_]u32{ 2, 2, 2, 1 };
pub const index_sources = [_]u32{ 2, 8, 14, 20, 24, 28, 32, 36 };
pub const main_stride = 288;
pub const index_stride = 68;
pub const ring_stride = 528;
pub const window = 128;
/// The backbone's layers, and the plan a drafter's three blocks add on top of them.
pub const backbone_layers = 40;
pub const drafter_layers = backbone_layers + 3;
pub const max_context = 1048576;

pub fn ownerFor(layer: u32, ratio: u32) !usize {
    if (ratio == 0) return error.NoCompressedCache;
    if (layer >= 40) return error.InvalidCacheLayer;
    if (layer < 2 or ratio != (if (layer < 20) @as(u32, 2) else 1)) return error.InvalidCacheOwner;
    var found: ?usize = null;
    for (owners, 0..) |owner, i| {
        if (owner <= layer and ratios[i] == ratio) found = i;
    }
    return found orelse error.InvalidCacheOwner;
}
pub fn isIndexSource(layer: u32) bool {
    return std.mem.indexOfScalar(u32, &index_sources, layer) != null;
}
pub const Source = struct { owner: u32, ratio: u32, rows: u32, main: View, index: View, partial: View, capture: View };
/// The compressed path's raw-row capture. The pool writes every row it consumes -- the row's own values and its
/// gate -- into one slot a position, so a rollback can re-cut the group that straddles the new position without
/// another forward (the no-re-feed port).
pub const capture_rows: u32 = 8;
pub const capture_width: u32 = 512;
/// One owner's capture: `capture_rows` slots of (values row, gate row), 512 f32 each.
pub fn captureBytes() usize {
    return @as(usize, capture_rows) * 2 * capture_width * 4;
}
pub const Plan = struct {
    capacity: u32,
    layers: u32,
    pub fn validate(p: Plan) !void {
        // Forty backbone layers, or forty-three when the drafter's three blocks hold rings of their own (they have no
        // compressor, so their packed storage is the backbone's and `packedBytes` is unchanged).
        if (p.capacity == 0 or p.capacity > max_context or (p.layers != 40 and p.layers != drafter_layers))
            return error.InvalidCacheCapacity;
    }
    pub fn packedBytes(p: Plan) usize {
        var bytes: usize = 0;
        for (ratios) |r| bytes += @as(usize, p.capacity / r) * (main_stride + index_stride);
        return bytes;
    }
    pub fn mutableBytes(p: Plan) usize {
        return @as(usize, p.layers) * window * ring_stride + 7 * 2 * 512 * 4 + owners.len * captureBytes();
    }

    /// The compressed path's partial-block scratch begins after the window rings.
    pub fn partialOffset(p: Plan) usize {
        return @as(usize, p.layers) * window * ring_stride;
    }
    pub fn partialBytes(p: Plan) usize {
        _ = p;
        return 7 * 2 * 512 * 4;
    }
    /// The raw-row capture begins after the partial scratch.
    pub fn captureOffset(p: Plan) usize {
        return p.partialOffset() + p.partialBytes();
    }
    pub fn residentBytes(p: Plan) usize {
        return p.packedBytes() + 2 * p.mutableBytes();
    }
};
pub const State = struct {
    plan: Plan,
    storage: mtl.Buffer,
    live: mtl.Buffer,
    backup: mtl.Buffer,
    position: u32 = 0,
    saved_position: ?u32 = null,

    pub fn init(device: mtl.Device, plan: Plan) !State {
        try plan.validate();
        const storage = try device.buffer(plan.packedBytes(), mtl.ResourceOptions.shared);
        errdefer storage.deinit();
        const live = try device.buffer(plan.mutableBytes(), mtl.ResourceOptions.shared);
        errdefer live.deinit();
        const backup = try device.buffer(plan.mutableBytes(), mtl.ResourceOptions.shared);
        @memset(live.contents()[0..plan.mutableBytes()], 0);
        return .{ .plan = plan, .storage = storage, .live = live, .backup = backup };
    }
    pub fn deinit(s: *State) void {
        s.storage.deinit();
        s.live.deinit();
        s.backup.deinit();
    }
    pub fn residentBytes(s: *const State) usize {
        return s.plan.residentBytes();
    }
    pub fn registerResidency(s: *const State, set: mtl.ResidencySet) void {
        set.add(s.storage);
        set.add(s.live);
        set.add(s.backup);
    }
    pub fn ring(s: *const State, layer: u32) !View {
        if (layer >= s.plan.layers) return error.InvalidCacheLayer;
        return .{ .buffer = s.live, .offset = @as(usize, layer) * window * ring_stride };
    }
    pub fn source(s: *const State, layer: u32, ratio: u32) !Source {
        if (layer >= s.plan.layers) return error.InvalidCacheLayer;
        if (s.position > s.plan.capacity) return error.CacheCapacityExceeded;
        const id = try ownerFor(layer, ratio);
        var packed_off: usize = 0;
        var partial_off: usize = @as(usize, s.plan.layers) * window * ring_stride;
        for (ratios[0..id]) |r| {
            packed_off += @as(usize, s.plan.capacity / r) * (main_stride + index_stride);
            partial_off += 2 * r * 512 * 4;
        }
        const capture_off = s.plan.captureOffset() + @as(usize, id) * captureBytes();
        return .{ .owner = owners[id], .ratio = ratio, .rows = s.position / ratio, .main = .{ .buffer = s.storage, .offset = packed_off }, .index = .{ .buffer = s.storage, .offset = packed_off + @as(usize, s.plan.capacity / ratio) * main_stride }, .partial = .{ .buffer = s.live, .offset = partial_off }, .capture = .{ .buffer = s.live, .offset = capture_off } };
    }
    /// Zero the compressed path's partial scratch: a rejected window's draft row may leave half a block's sum there,
    /// and the next forward reads it as if the block were complete. Cheaper and safer than tracking which block.
    pub fn clearPartials(s: *State) void {
        const bytes = s.live.contents();
        const off = Plan.partialOffset(s.plan);
        const len = s.live.length();
        if (off < len) @memset(bytes[off..len], 0);
    }

    /// Zero the window ring's rows [from, to) for every layer: a rejected window's draft rows stay in the ring, and
    /// this is the diagnostic that tells whether the next forward reads them. The buffer is CPU-shared.
    pub fn clearRows(s: *State, from: u64, to: u64) void {
        if (to <= from) return;
        const bytes = s.live.contents();
        var layer: usize = 0;
        while (layer < s.plan.layers) : (layer += 1) {
            const base = layer * window * ring_stride;
            var pos = from;
            while (pos < to) : (pos += 1) {
                const at = base + @as(usize, @intCast(pos % window)) * ring_stride;
                @memset(bytes[at..][0..ring_stride], 0);
            }
        }
    }

    pub fn append(s: *State, rows: u32) !void {
        if (s.position > s.plan.capacity or rows == 0 or rows > s.plan.capacity - s.position)
            return error.CacheCapacityExceeded;
        s.position += rows;
    }
    pub fn begin(s: *State) !void {
        if (s.saved_position != null) return error.CacheTransactionActive;
        s.saved_position = s.position;
    }
    pub fn commit(s: *State) !void {
        if (s.saved_position == null) return error.NoCacheTransaction;
        s.saved_position = null;
    }
    pub fn rollback(s: *State) !void {
        s.position = s.saved_position orelse return error.NoCacheTransaction;
        s.saved_position = null;
    }
};
