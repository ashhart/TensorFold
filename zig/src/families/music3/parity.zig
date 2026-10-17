//! Parity dumps vs a ComfyUI reference run (REF) as .npy in OUT for tools/music3/parity.py.
const std = @import("std");
const m3 = @import("music3.zig");
const npy = @import("npy.zig");
const cfg = @import("config.zig");

pub const Opts = struct {
    ar: ?[]const u8 = null,
    dit: ?[]const u8 = null,
    probe: ?[]const u8 = null,
    dav: ?[]const u8 = null,
    out: []const u8,
    caption: []const u8 = "",
    lyrics: []const u8 = "",
    frames: usize = 0,
    steps: usize = 30,
    dit_cfg: f32 = 1.7,
};

fn path(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]const u8 {
    return std.fs.path.join(a, &.{ dir, name });
}

fn bf16of(x: f32) u16 {
    const u: u32 = @bitCast(x);
    const r = u + 0x7FFF + ((u >> 16) & 1);
    return @intCast(r >> 16);
}

fn bf16to(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

/// [C, L] (or [1, C, L]) -> [L, C].
fn transpose(gpa: std.mem.Allocator, src: []const f32, C: usize, L: usize) ![]f32 {
    const out = try gpa.alloc(f32, src.len);
    for (0..C) |c| for (0..L) |t| {
        out[t * C + c] = src[c * L + t];
    };
    return out;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *m3.Engine, o: Opts) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    std.Io.Dir.cwd().createDirPath(io, o.out) catch {};
    if (o.ar) |ref| {
        const pr = try e.ids(o.caption, o.lyrics);
        defer gpa.free(pr.cond);
        defer gpa.free(pr.unc);
        const js = try std.json.Stringify.valueAlloc(a, .{ .ids = pr.cond }, .{});
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try path(a, o.out, "ids.json"), .data = js });
        const codes_np = try npy.read(gpa, io, try path(a, ref, "codes.npy"));
        defer codes_np.deinit(gpa);
        const flat = try codes_np.toI32(a);
        const nf = codes_np.shape[0];
        const codes: []const [8]i32 = @as([*]const [8]i32, @ptrCast(flat.ptr))[0..nf];
        var frames = if (o.frames > 0) o.frames else nf - 1;
        frames = @min(frames, nf - 1);
        var tr: m3.ar.Trace = .{};
        defer tr.c0.deinit(gpa);
        defer tr.depth.deinit(gpa);
        const t0 = m3.ar.nowNs(io);
        const r = try e.ar.?.run(io, pr.cond, pr.unc, .{}, frames, false, 1, codes[0 .. frames + 1], &tr, null);
        defer gpa.free(r.states);
        const secs = m3.ar.ms(t0, m3.ar.nowNs(io)) / 1000;
        const nlog = tr.c0.items.len / (2 * (cfg.c0_codes + 1));
        try npy.write(gpa, io, try path(a, o.out, "c0_logits.npy"), f32, tr.c0.items, &.{ nlog, 2, cfg.c0_codes + 1 });
        try npy.write(gpa, io, try path(a, o.out, "depth_logits.npy"), f32, tr.depth.items, &.{ nlog, 7, 2, cfg.codebook_size });
        const sf = try gpa.alloc(f32, r.states.len);
        defer gpa.free(sf);
        for (sf, r.states) |*d, s| d.* = bf16to(s);
        try npy.write(gpa, io, try path(a, o.out, "states.npy"), f32, sf, &.{ r.frames, 8 * cfg.hidden });
        std.debug.print("[music3] parity ar: {d} forced frames, {d} kept, {d:.1}s\n", .{ nlog, r.frames, secs });
    }
    if (o.dit) |ref| {
        const hid = try npy.read(gpa, io, try path(a, ref, "hidden.npy"));
        defer hid.deinit(gpa);
        const hf = try hid.toF32(a);
        const frames = hid.shape[0];
        const states = try a.alloc(u16, hf.len);
        for (states, hf) |*d, x| d.* = bf16of(x);
        const nz = try npy.read(gpa, io, try path(a, ref, "noise.npy"));
        defer nz.deinit(gpa);
        const L = nz.shape[nz.rank - 1];
        if (L != cfg.framesToLatents(frames)) return error.NoiseLength;
        const noise = try transpose(a, try nz.toF32(a), 128, L);
        var d = e.dit();
        const t0 = m3.ar.nowNs(io);
        const lat = try d.denoise(states, frames, o.steps, o.dit_cfg, 0, noise, null, null);
        defer gpa.free(lat);
        const secs = m3.ar.ms(t0, m3.ar.nowNs(io)) / 1000;
        const back = try transpose(a, lat, L, 128); // [L, 128] -> [128, L]
        try npy.write(gpa, io, try path(a, o.out, "latent.npy"), f32, back, &.{ 1, 128, L });
        std.debug.print("[music3] parity dit: {d} latent rows, {d:.1}s\n", .{ L, secs });
    }
    if (o.probe) |ref| {
        const hid = try npy.read(gpa, io, try path(a, ref, "probe_hidden.npy"));
        defer hid.deinit(gpa);
        const hf = try hid.toF32(a);
        const frames = hid.shape[0];
        const states = try a.alloc(u16, hf.len);
        for (states, hf) |*d, x| d.* = bf16of(x);
        const px = try npy.read(gpa, io, try path(a, ref, "probe_x.npy"));
        defer px.deinit(gpa);
        const L = px.shape[px.rank - 1];
        if (L != cfg.framesToLatents(frames)) return error.ProbeLength;
        // the reference casts x to the DiT dtype first
        const xs = try transpose(a, try px.toF32(a), 128, L);
        for (xs) |*v| v.* = @floatCast(@as(f16, @floatCast(v.*)));
        var d = e.dit();
        for ([_]f32{ 1.0, 0.5, 0.1 }) |sigma| {
            for ([_]f32{ 1.0, 0.0 }, [_][]const u8{ "cond", "uncond" }) |g, name| {
                const out = try d.denoise(states, frames, 1, 1.0, 0, xs, .{ .cfg = g, .t = 1.0 - sigma }, null);
                defer gpa.free(out);
                const back = try transpose(a, out, L, 128);
                const fname = try std.fmt.allocPrint(a, "probe_{s}_{d:.1}.npy", .{ name, sigma });
                try npy.write(gpa, io, try path(a, o.out, fname), f32, back, &.{ 128, L });
            }
        }
        std.debug.print("[music3] parity probe: {d} latent rows\n", .{L});
    }
    if (o.dav) |ref| {
        var v = e.dav();
        const lp = try path(a, ref, "latent.npy");
        if (npy.read(gpa, io, lp)) |lat| {
            defer lat.deinit(gpa);
            const L = lat.shape[lat.rank - 1];
            const x = try transpose(a, try lat.toF32(a), 128, L);
            const t0 = m3.ar.nowNs(io);
            const audio = try v.decode(x, L);
            defer gpa.free(audio);
            const secs = m3.ar.ms(t0, m3.ar.nowNs(io)) / 1000;
            try npy.write(gpa, io, try path(a, o.out, "audio.npy"), f32, audio, &.{ 2, L * 512 });
            std.debug.print("[music3] parity dav: {d} latent rows, {d:.2}s\n", .{ L, secs });
        } else |_| {}
        if (npy.read(gpa, io, try path(a, ref, "probe_latent.npy"))) |lat| {
            defer lat.deinit(gpa);
            const L = lat.shape[lat.rank - 1];
            const x = try transpose(a, try lat.toF32(a), 128, L);
            v.o.chunk = L;
            v.o.context = 0;
            v.o.level = false;
            const audio = try v.decode(x, L);
            defer gpa.free(audio);
            try npy.write(gpa, io, try path(a, o.out, "probe_dav.npy"), f32, audio, &.{ 2, L * 512 });
        } else |_| {}
    }
    return 0;
}
