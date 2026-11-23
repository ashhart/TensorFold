//! Capture GLM-5.3-Flash's DFlash2 drafter taps during the target's Metal forward, opt-in.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const draft = @import("metal_dflash_config.zig");

const TapRingArgs = extern struct {
    hidden: u32,
    rows: u32,
    skip: u32,
    pos: u32,
    window: u32,
    plane_off: u32,
    ring_stride: u32,
};

/// A tap capture writes selected layer residuals into the drafter's ring-shaped pending buffer; null disables it.
pub const Taps = struct {
    layers: [5]u32 = draft.tap_ids,
    ring: mtl.Buffer,
    window: u32,
    hidden: u32,
    /// Each drafting stream's own ring and state: a shared window's seg rows capture into their stream's.
    routes: []Route = &.{},

    /// One stream's route: the target state identifying it, the ring its taps land in, and active capture switch.
    pub const Route = struct {
        s: ?*const st.State,
        ring: mtl.Buffer,
        active: bool = true,
    };

    /// A capture over `ring` (the drafter's pending buffer), one row per window slot, with `streams` per-stream routes.
    pub fn init(_: mtl.Device, ring: mtl.Buffer, window: u32, hidden: u32, gpa: std.mem.Allocator, streams: usize) !Taps {
        const routes = try gpa.alloc(Route, streams);
        @memset(routes, .{ .s = null, .ring = ring });
        return .{
            .ring = ring,
            .window = window,
            .hidden = hidden,
            .routes = routes,
        };
    }

    /// Byte offset of plane i for position `pos` in the ring, or null if `layer` is not tapped.
    pub fn offset(t: *const Taps, layer: usize, pos: u32) ?usize {
        const i = t.index(layer) orelse return null;
        const slot: usize = @intCast(pos % t.window);
        return slot * t.stride() + i * t.hidden * 2;
    }

    /// After layer `layer`, the streams' mean of rows pos..pos+rows into plane i; a shared seg to its own ring.
    pub fn after(t: *const Taps, x: *fwd.Ctx, e: mtl.ComputeEncoder, layer: usize, rows: u32, pos: u32) void {
        const i = t.index(layer) orelse return;
        if (x.segs.len == 0) {
            const ring = t.ringFor(x.s) orelse return;
            t.route(x, e, ring, x.sc.x[x.xi], rows, pos, i);
            return;
        }
        for (x.segs) |seg| {
            const ring = t.routeRing(seg.s) orelse continue;
            const src = wts.Ref{ .buf = x.sc.x[x.xi].buf, .off = x.sc.x[x.xi].off + @as(usize, seg.row0) * x.c.hidden * 8 };
            t.route(x, e, ring, src, seg.rows, seg.pos, i);
        }
    }

    /// A lone stream's ring: the shared one when no stream is routed (one drafter context), else its route's or none.
    pub fn ringFor(t: *const Taps, s: ?*const st.State) ?mtl.Buffer {
        if (t.routes.len == 0) return t.ring;
        return t.routeRing(s);
    }

    /// The stream's routed ring when active, or null when inactive or drafting no chain.
    pub fn routeRing(t: *const Taps, s: ?*const st.State) ?mtl.Buffer {
        for (t.routes) |rt| if (rt.s == s and rt.active) return rt.ring;
        return null;
    }

    /// Mean of the four HC streams from `src` scattered directly into `ring`'s plane `i` in one launch.
    fn route(t: *const Taps, x: *fwd.Ctx, e: mtl.ComputeEncoder, ring: ?mtl.Buffer, src: wts.Ref, rows: u32, pos: u32, i: usize) void {
        const D = t.hidden;
        const n = @min(rows, t.window);
        if (n == 0) return;
        const skip = rows - n;
        const dst = ring orelse t.ring;
        e.setPipeline(x.k.tap_mean_ring);
        fwd.bind(e, 0, .{ src, .{ .buf = dst, .off = 0 } });
        e.setValue(TapRingArgs{
            .hidden = D,
            .rows = n,
            .skip = skip,
            .pos = pos,
            .window = t.window,
            .plane_off = @intCast(i * D),
            .ring_stride = @intCast(t.layers.len * D),
        }, 2);
        e.dispatchThreads(fwd.size(D, n, 1), fwd.size(256, 1, 1));
    }

    pub fn stride(t: *const Taps) usize {
        return @as(usize, t.layers.len) * t.hidden * 2;
    }

    /// The stream's route into its own ring (null: no ring, its seg rows are skipped).
    pub fn setRoute(t: *Taps, slot: usize, s: *const st.State, ring: mtl.Buffer) void {
        if (slot >= t.routes.len) return;
        t.routes[slot] = .{ .s = s, .ring = ring, .active = true };
    }

    /// Toggle whether `slot`'s route writes taps during verify forwards.
    pub fn setActive(t: *Taps, slot: usize, active: bool) void {
        if (slot >= t.routes.len or t.routes[slot].s == null) return;
        t.routes[slot].active = active;
    }

    /// The stream's route gone: its rows capture into nothing.
    pub fn clearRoute(t: *Taps, s: *const st.State) void {
        for (t.routes) |*rt| {
            if (rt.s != s) continue;
            rt.s = null;
            rt.active = false;
        }
    }

    /// The routes go; the caller frees its ring.
    pub fn deinit(t: *Taps, gpa: std.mem.Allocator) void {
        gpa.free(t.routes);
        t.routes = &.{};
    }

    fn index(t: *const Taps, layer: usize) ?usize {
        const li: u32 = @intCast(layer);
        for (t.layers, 0..) |tap, i| if (tap == li) return i;
        return null;
    }
};
