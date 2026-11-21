//! ZONOS3's audio decoders on the host: the default loaded with the model, another on the first request that asks.
const std = @import("std");
const cuda = @import("cuda");
const z3 = @import("zonos3");

const Decoder = z3.codec.Decoder;
const count = std.enums.values(Decoder).len;

/// The device bytes of a decoder's weights: the generator's tables and its head, measured as loaded.
pub fn weightBytes(d: Decoder) u64 {
    return switch (d) {
        .keep2_plain => 759_635_712,
        .full_mr => 4_253_052_672,
    };
}

pub const Set = struct {
    /// The decoders served, the default first, and their ids as the server lists them.
    served: []const Decoder,
    ids: []const []const u8,
    heads: [count][]const u8 = @splat(""),
    headroom_db: [count]f32 = @splat(0),
    codecs: [count]?*z3.Codec = @splat(null),
    options: z3.codec.Options,
    lanes: usize,

    /// `list`: --decoders, or null for all the release holds; the release's default stays first when listed.
    pub fn open(a: std.mem.Allocator, io: std.Io, dir: []const u8, list: ?[]const u8, options: z3.codec.Options, lanes: usize) !Set {
        var s: Set = .{ .served = &.{}, .ids = &.{}, .options = options, .lanes = lanes };
        var served: std.ArrayList(Decoder) = .empty;
        var it = std.mem.splitScalar(u8, list orelse "", ',');
        while (it.next()) |raw| {
            const id = std.mem.trim(u8, raw, " ");
            if (id.len == 0) continue;
            const d = Decoder.of(id) orelse return error.NoSuchDecoder;
            if (std.mem.indexOfScalar(Decoder, served.items, d) != null) return error.DecoderListedTwice;
            try served.append(a, d);
        }
        if (list != null and served.items.len == 0) return error.NoSuchDecoder;
        for (std.enums.values(Decoder)) |d| {
            var rel = z3.release.open(a, io, dir, @tagName(d)) catch |e| switch (e) {
                error.NoSuchDecoder => if (list == null) continue else return e,
                else => return e,
            };
            defer rel.deinit();
            // a listed decoder must be in the release; unlisted ones are skipped
            if (list != null and std.mem.indexOfScalar(Decoder, served.items, d) == null) continue;
            if (std.Io.Dir.cwd().access(io, rel.head, .{})) |_| {} else |_| {
                if (list != null) return error.DecoderHeadMissing;
                continue;
            }
            s.heads[@backingInt(d)] = try a.dupe(u8, rel.head);
            s.headroom_db[@backingInt(d)] = rel.headroom_db;
            if (list == null) try served.append(a, d);
        }
        const usual = Decoder.of(z3.release.default_decoder).?;
        if (std.mem.indexOfScalar(Decoder, served.items, usual)) |i| std.mem.swap(Decoder, &served.items[0], &served.items[i]);
        if (served.items.len == 0) return error.NoSuchDecoder;
        const ids = try a.alloc([]const u8, served.items.len);
        for (ids, served.items) |*id, d| id.* = @tagName(d);
        s.served = served.items;
        s.ids = ids;
        return s;
    }

    pub fn default(s: *const Set) Decoder {
        return s.served[0];
    }

    /// The request's decoder: `index` into `served`.
    pub fn of(s: *const Set, index: usize) !Decoder {
        if (index >= s.served.len) return error.NoSuchDecoder;
        return s.served[index];
    }

    /// Device bytes `d` takes once loaded: its weights, and a lane for every request at once.
    pub fn bytes(s: *const Set, d: Decoder) u64 {
        var o = s.options;
        o.decoder = d;
        return weightBytes(d) + z3.Codec.stateBytes(o) + (s.lanes - 1) * z3.Codec.laneBytes(o);
    }

    /// The decoder's codec, made with its lanes on its first use; `gen`: the generator checkpoint.
    pub fn load(s: *Set, gpa: std.mem.Allocator, io: std.Io, e: *z3.Engine, gen: []const u8, d: Decoder) !*z3.Codec {
        if (s.codecs[@backingInt(d)]) |c| return c;
        var o = s.options;
        o.decoder = d;
        const c = try z3.Codec.create(gpa, io, &e.driver, e.stream, gen, s.heads[@backingInt(d)], o);
        errdefer c.destroy();
        for (1..s.lanes) |_| _ = try c.addLane();
        if (c.w.bytes > weightBytes(d)) return error.DecoderOverItsAdmission;
        s.codecs[@backingInt(d)] = c;
        return c;
    }

    pub fn deinit(s: *Set) void {
        for (&s.codecs) |*c| if (c.*) |codec| {
            codec.destroy();
            c.* = null;
        };
    }
};

/// The request's decoder for host `h`, loaded on its first use when the device has room for it.
pub fn codecFor(h: anytype, index: usize) !*z3.Codec {
    const d = try h.decoders.of(index);
    if (h.decoders.codecs[@backingInt(d)]) |c| return c;
    const need = h.decoders.bytes(d);
    if (h.admit) |a| if (need > try a.room(a.ptr, &h.e.ctx)) return error.DecoderNeedsMoreDeviceMemory;
    const before = cuda.usage(false).device;
    const c = try h.decoders.load(h.gpa, h.io, h.e, h.rel.generator, d);
    const took = cuda.usage(false).device - before;
    std.log.info("zonos3: decoder {s} loaded, {d} bytes on the device ({d} admitted)", .{ @tagName(d), took, need });
    if (took > need) return error.DecoderOverItsAdmission;
    return c;
}
