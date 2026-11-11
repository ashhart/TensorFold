//! A launch's Engram rows read ahead and joined one layer at a time, just before the command buffer that needs them.
const std = @import("std");
const host = @import("dsv41");

const eh = host.engram_host;
const Config = host.config.Config;

/// One slot's ids of a launch and the ids before them (oldest first; short: the slot's start).
pub const Piece = struct { ids: []const u32, lookback: []const u32 };

/// Most pieces (slots) in a launch.
pub const max_pieces = 16;

/// A launch's row reads: `begin` starts every Engram layer's, `join` lands one layer's, `deinit` leaves none pending.
pub const Stage = struct {
    a: std.mem.Allocator,
    host: ?*const eh.Host,
    source: ?eh.Source,
    layers: []const u32 = &.{},
    pieces: []const Piece = &.{},
    /// per Engram layer and piece: the table rows, hashed once at `begin`
    idx: [eh.max_layers][max_pieces]?[]i64 = @splat(@splat(null)),
    /// per Engram layer: its rows' destination [rows][cols * head_dim] bf16, and whether they have landed
    out: [eh.max_layers][]u16 = @splat(&.{}),
    joined: [eh.max_layers]bool = @splat(true),
    dim: usize = 0,
    cols: usize = 0,

    /// A launch without Engram layers: nothing to read.
    pub const none: Stage = .{ .a = std.heap.page_allocator, .host = null, .source = null };

    /// Hashes every layer's rows and starts reading them; `out[li]` receives layer li's rows at `join` (all pieces).
    pub fn begin(a: std.mem.Allocator, c: *const Config, h: *const eh.Host, source: ?eh.Source, pieces: []const Piece, out: []const []u16) !Stage {
        if (pieces.len > max_pieces or out.len < c.engram_layers.len) return error.BadRows;
        var s: Stage = .{ .a = a, .host = h, .source = source, .layers = c.engram_layers.items(), .pieces = pieces, .dim = c.engram_head_dim, .cols = c.hashCols() };
        errdefer s.deinit();
        for (s.layers, 0..) |layer, li| {
            s.out[li] = out[li];
            s.joined[li] = false;
            for (pieces, 0..) |pc, pi| s.idx[li][pi] = try h.shardIndex(a, pc.ids, pc.lookback, layer, 0, 1);
        }
        s.issue();
        return s;
    }

    /// Starts the reads the source can start early; a failure only means `join` reads them itself.
    fn issue(s: *const Stage) void {
        const src = s.source orelse return;
        for (s.layers, 0..) |_, li| {
            const hl = std.mem.indexOfScalar(u32, s.host.?.layer_ids, s.layers[li]) orelse continue;
            for (s.idx[li][0..s.pieces.len]) |q| src.vtable.issue(src.ptr, hl, q.?) catch {};
        }
    }

    /// Layer index `li`'s rows into its destination, pieces in row order (waits for reads still in flight).
    pub fn join(s: *Stage, li: usize) !void {
        if (s.joined[li]) return;
        s.joined[li] = true; // a failed join is not retried: its error ends the launch
        const K = s.cols * s.dim;
        var r0: usize = 0;
        for (s.pieces, s.idx[li][0..s.pieces.len]) |pc, q| {
            try s.host.?.readRows(s.source, s.layers[li], q.?, s.dim, s.out[li][r0 * K ..][0 .. pc.ids.len * K]);
            r0 += pc.ids.len;
        }
    }

    /// Joins the Engram layers whose index is in [lo, hi) of the backbone (the ones one command buffer holds).
    pub fn joinRange(s: *Stage, lo: usize, hi: usize) !void {
        for (s.layers, 0..) |layer, li| if (layer >= lo and layer < hi) try s.join(li);
    }

    pub fn joinAll(s: *Stage) !void {
        for (0..s.layers.len) |li| try s.join(li);
    }

    /// Frees the hashed rows; a layer never joined has its reads waited out first, so none outlives the launch.
    pub fn deinit(s: *Stage) void {
        for (s.layers, 0..) |layer, li| {
            for (s.idx[li][0..s.pieces.len]) |*slot| {
                const q = slot.* orelse continue;
                defer s.a.free(q);
                slot.* = null;
                if (s.joined[li]) continue;
                const buf = s.a.alloc(u16, q.len * s.dim) catch continue;
                defer s.a.free(buf);
                s.host.?.readRows(s.source, layer, q, s.dim, buf) catch {};
            }
            s.joined[li] = true;
        }
    }
};
