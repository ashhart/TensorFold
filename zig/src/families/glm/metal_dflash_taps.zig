//! Capture GLM-5.3-Flash's DFlash2 drafter taps during the target's Metal forward, opt-in.
const std = @import("std");
const mtl = @import("metal");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const draft = @import("metal_dflash_config.zig");

/// A tap capture writes selected layer residuals into the drafter's ring-shaped pending buffer; null disables it.
pub const Taps = struct {
    layers: [5]u32 = draft.tap_ids,
    ring: mtl.Buffer,
    window: u32,
    hidden: u32,
    /// Scratch [window][hidden] bf16 for stream_mean output before scattering into the strided ring.
    stage: mtl.Buffer,
    /// Each drafting stream's own ring and state: a shared window's seg rows capture into their stream's.
    routes: []Route = &.{},

    /// One stream's route: the target state identifying it, and the ring its taps land in.
    pub const Route = struct {
        s: ?*const st.State,
        ring: mtl.Buffer,
    };

    /// A capture over `ring` (the drafter's pending buffer), one row per window slot, with `streams` per-stream routes.
    pub fn init(device: mtl.Device, ring: mtl.Buffer, window: u32, hidden: u32, gpa: std.mem.Allocator, streams: usize) !Taps {
        const stage = try device.buffer(@as(usize, window) * hidden * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer stage.deinit();
        const routes = try gpa.alloc(Route, streams);
        @memset(routes, .{ .s = null, .ring = ring });
        return .{
            .ring = ring,
            .window = window,
            .hidden = hidden,
            .stage = stage,
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
            const ring = t.ringFor(x.s) orelse return; // a stream with no route (capped, or drafting with the head)
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

    /// The stream's routed ring, or null when it drafts no chain (its seg is skipped).
    pub fn routeRing(t: *const Taps, s: ?*const st.State) ?mtl.Buffer {
        for (t.routes) |rt| if (rt.s == s) return rt.ring;
        return null;
    }

    /// The streams scratch's mean over `rows` from `pos`, scattered into `ring`'s plane `i` (null: t.ring).
    fn route(t: *const Taps, x: *fwd.Ctx, e: mtl.ComputeEncoder, ring: ?mtl.Buffer, src: wts.Ref, rows: u32, pos: u32, i: usize) void {
        const D = t.hidden;
        const n = @min(rows, t.window);
        const skip = rows - n;
        e.setPipeline(x.k.stream_mean);
        fwd.bind(e, 0, .{ src.at(@as(usize, skip) * 4 * D * 2), .{ .buf = t.stage, .off = 0 } });
        e.setValue([2]u32{ D, n }, 2);
        e.dispatchThreads(fwd.size(D, n, 1), fwd.size(256, 1, 1));
        const words = D / 2;
        e.setPipeline(x.k.copy_u32);
        const dst = ring orelse t.ring;
        for (0..n) |r| {
            const slot: u32 = @intCast((pos + skip + r) % t.window);
            const dst_off = @as(usize, slot) * t.stride() + @as(usize, i) * D * 2;
            fwd.bind(e, 0, .{ .{ .buf = t.stage, .off = r * D * 2 }, .{ .buf = dst, .off = dst_off } });
            e.setValue(words, 2);
            e.dispatchThreads(fwd.size(words, 1, 1), fwd.size(256, 1, 1));
        }
    }

    pub fn stride(t: *const Taps) usize {
        return @as(usize, t.layers.len) * t.hidden * 2;
    }

    /// The stream's route into its own ring (null: no ring, its seg rows are skipped).
    pub fn setRoute(t: *Taps, slot: usize, s: *const st.State, ring: mtl.Buffer) void {
        if (slot >= t.routes.len) return;
        t.routes[slot] = .{ .s = s, .ring = ring };
    }

    /// The stream's route gone: its rows capture into nothing.
    pub fn clearRoute(t: *Taps, s: *const st.State) void {
        for (t.routes) |*rt| {
            if (rt.s != s) continue;
            rt.s = null;
        }
    }

    /// The routes and the stage scratch go; the caller frees its ring.
    pub fn deinit(t: *Taps, gpa: std.mem.Allocator) void {
        gpa.free(t.routes);
        t.stage.deinit();
        t.routes = &.{};
    }

    fn index(t: *const Taps, layer: usize) ?usize {
        const li: u32 = @intCast(layer);
        for (t.layers, 0..) |tap, i| if (tap == li) return i;
        return null;
    }
};
