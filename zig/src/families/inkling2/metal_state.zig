//! Inkling 2 on Metal: a stream's state buffers, sized by state_layout.zig.
const std = @import("std");
const mtl = @import("metal");
const config = @import("config.zig");
const layout = @import("state_layout.zig");

pub const State = struct {
    gpa: std.mem.Allocator,
    geometry: layout.Geometry,
    layers: u32,
    /// Positions held: the next forward's rows start here.
    len: u32 = 0,
    /// Lowest length a `keep` may return to (see Model.keep).
    floor: u32 = 0,
    /// A forward that touched this stream failed: its rows may be half written, so only `clear` or a restore reuse it.
    failed: bool = false,
    /// Per layer: keys then values (private), and the four histories (shared, zeroed at reset).
    kv: []mtl.Buffer,
    hist: []mtl.Buffer,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, cfg: *const config.Config, g: layout.Geometry) !State {
        const n = cfg.text.executionLayers();
        if (g.capacity == 0 or g.rows == 0) return error.InvalidGeometry;
        const kv = try gpa.alloc(mtl.Buffer, n);
        errdefer gpa.free(kv);
        const h = try gpa.alloc(mtl.Buffer, n);
        errdefer gpa.free(h);
        var made: usize = 0;
        errdefer for (0..made) |l| {
            kv[l].deinit();
            h[l].deinit();
        };
        for (0..n) |l| {
            const i = cfg.text.physicalLayer(@intCast(l));
            kv[l] = try device.buffer(@max(2 * layout.kvBytes(&cfg.text, i, g), 16), mtl.ResourceOptions.private);
            errdefer kv[l].deinit();
            h[l] = try device.buffer(layout.hist(&cfg.text, i).bytes, mtl.ResourceOptions.shared);
            @memset(h[l].contents()[0..h[l].length()], 0);
            made += 1;
        }
        return .{ .gpa = gpa, .geometry = g, .layers = n, .kv = kv, .hist = h };
    }

    /// Every buffer enrolled with (or released from) a residency hook, the spark-style follow.
    pub fn setResident(self: *State, hook: ?mtl.ResidencySet.Follow) void {
        if (self.follow) |f| self.apply(f, false);
        self.follow = hook;
        if (hook) |f| self.apply(f, true);
    }

    fn apply(self: *State, f: mtl.ResidencySet.Follow, joining: bool) void {
        for (self.kv, self.hist) |k, h| {
            mtl.resident.fields.one(k, f, joining);
            mtl.resident.fields.one(h, f, joining);
        }
    }

    /// A fresh stream: no positions, zero histories (positions before 0 read as zeros).
    pub fn clear(self: *State) void {
        for (self.hist) |h| @memset(h.contents()[0..h.length()], 0);
        self.len = 0;
        self.floor = 0;
        self.failed = false;
    }

    pub fn deinit(self: *State) void {
        self.setResident(null);
        for (self.kv, self.hist) |k, h| {
            k.deinit();
            h.deinit();
        }
        self.gpa.free(self.kv);
        self.gpa.free(self.hist);
    }
};
