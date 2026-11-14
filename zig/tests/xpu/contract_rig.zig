//! The XPU contract's device: the card by PCI id, its images, and guarded buffers with poisoned outputs.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");
const core = @import("core");
const images = @import("xpu_contract_images");

pub const guard = 256;
const pattern = 0xa5;

/// The image a kernel lives in: the Qwen3.8-27B glue, gate 5's controls, or gates 1 to 4's.
pub const Image = enum { glue, decode, attn_upstream, controls, gate };

fn blob(i: Image) []const u8 {
    return switch (i) {
        .glue => &images.glue,
        .decode => &images.decode,
        .attn_upstream => &images.attn_upstream,
        .controls => &images.controls,
        .gate => &images.gate_controls,
    };
}

/// One card, its context and stream, and the contract's images loaded native for it.
pub const Dev = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    s: xpu.Stream,
    glue: xpu.Module,
    decode: xpu.Module,
    attn_upstream: xpu.Module,
    controls: ?xpu.Module, // the driver refuses the whole image when one control asks for more SLM than it has
    controls_err: ?anyerror = null,
    gate: xpu.Module,

    /// Opens the card `ordinal` names, else the first Arc card, and loads every image; `self` stays where it is.
    pub fn init(self: *Dev, ordinal: ?u32) !void {
        self.drv = try xpu.Driver.open();
        self.ctx = try xpu.Context.init(&self.drv, ordinal);
        self.s = try xpu.Stream.init(&self.ctx);
        self.glue = try xpu.Module.loadNative(&self.ctx, blob(.glue));
        self.decode = try xpu.Module.loadNative(&self.ctx, blob(.decode));
        self.attn_upstream = try xpu.Module.loadNative(&self.ctx, blob(.attn_upstream));
        self.controls = xpu.Module.loadNative(&self.ctx, blob(.controls)) catch |e| blk: {
            self.controls_err = e;
            break :blk null;
        };
        self.gate = try xpu.Module.loadNative(&self.ctx, blob(.gate));
    }

    pub fn deinit(self: *Dev) void {
        self.gate.unload();
        if (self.controls) |*m| m.unload();
        self.glue.unload();
        self.decode.unload();
        self.attn_upstream.unload();
        self.s.deinit();
        self.ctx.deinit();
        self.drv.close();
    }

    pub fn module(self: *Dev, i: Image) !*xpu.Module {
        return switch (i) {
            .glue => &self.glue,
            .decode => &self.decode,
            .attn_upstream => &self.attn_upstream,
            .controls => if (self.controls) |*m| m else error.ControlsNotBuilt,
            .gate => &self.gate,
        };
    }

    /// The record of the launch `symbol` names, so the kernel is made at the work-group the audit checked.
    pub fn kernelFor(self: *Dev, i: Image, symbol: []const u8, a: std.mem.Allocator) !xpu.Kernel {
        const rec = recordOf(symbol) orelse return error.NoRecord;
        const name = try a.allocSentinel(u8, rec.launch.symbol.len, 0);
        defer a.free(name); // zeKernelCreate copies the name
        @memcpy(name, rec.launch.symbol);
        return (try self.module(i)).kernel(name, rec.launch.threads);
    }

    /// A kernel by its exact name with the work-group size the caller sets.
    pub fn kernel(self: *Dev, i: Image, symbol: [:0]const u8, threads: [3]u32) !xpu.Kernel {
        return (try self.module(i)).kernel(symbol, threads);
    }

    /// A kernel by name with no work-group set: its own properties, not a refused size, are what a caller reads.
    pub fn probe(self: *Dev, i: Image, symbol: [:0]const u8) !xpu.Kernel {
        return (try self.module(i)).kernelAt(symbol);
    }

    /// The kernel's own properties as the driver reports them.
    pub fn properties(self: *Dev, k: xpu.Kernel) !ze.KernelProperties {
        var p: ze.KernelProperties = .{};
        try self.ctx.d.check(self.ctx.d.api.zeKernelGetProperties(k.handle, &p), "zeKernelGetProperties");
        return p;
    }
};

/// The launch record `symbol` names, from the XPU records the audit holds.
pub fn recordOf(symbol: []const u8) ?core.xpu_records.Record {
    for (core.xpu_records.records) |r| if (std.mem.eql(u8, r.launch.symbol, symbol)) return r;
    return null;
}

const Kind = enum { input, output, scratch };
const Slot = struct { buf: xpu.DeviceBuffer, len: usize, kind: Kind };

/// One call's device buffers on one stream: each payload its own allocation, guarded on both sides.
pub const Rig = struct {
    a: std.mem.Allocator,
    dev: *Dev,
    slots: std.ArrayList(Slot) = .empty,
    seen: std.ArrayList(u8) = .empty,
    outs: std.ArrayList(u8) = .empty,
    host: std.ArrayList(u8) = .empty,

    pub fn deinit(r: *Rig) void {
        r.reset();
        r.slots.deinit(r.a);
        r.seen.deinit(r.a);
        r.outs.deinit(r.a);
        r.host.deinit(r.a);
    }

    pub fn reset(r: *Rig) void {
        r.dev.s.synchronize() catch {};
        for (r.slots.items) |*s| s.buf.free();
        r.slots.clearRetainingCapacity();
    }

    fn slot(r: *Rig, data: ?[]const u8, len: usize, fill: u8, kind: Kind) !?*anyopaque {
        const all = try r.a.alloc(u8, len + 2 * guard);
        defer r.a.free(all);
        @memset(all, pattern);
        if (data) |d| @memcpy(all[guard..][0..len], d) else @memset(all[guard..][0..len], fill);
        var buf = try xpu.DeviceBuffer.alloc(&r.dev.ctx, all.len);
        errdefer buf.free();
        try buf.upload(r.dev.s, 0, all);
        try r.slots.append(r.a, .{ .buf = buf, .len = len, .kind = kind });
        return try buf.at(guard);
    }

    pub fn input(r: *Rig, data: []const u8) !?*anyopaque {
        return r.slot(data, data.len, 0, .input);
    }

    /// An output of `len` bytes, all 0xff: an element no launch writes reads as NaN.
    pub fn output(r: *Rig, len: usize) !?*anyopaque {
        return r.slot(null, len, 0xff, .output);
    }

    /// Working memory between a call's launches: neither an input nor an output.
    pub fn scratch(r: *Rig, len: usize) !?*anyopaque {
        return r.slot(null, len, 0, .scratch);
    }

    fn find(r: *Rig, p: ?*anyopaque) !struct { xpu.DeviceBuffer, usize } {
        const at = @intFromPtr(p);
        for (r.slots.items) |s| {
            const base = @intFromPtr(s.buf.ptr);
            if (at >= base and at < base + s.buf.len) return .{ s.buf, at - base };
        }
        return error.Invalid;
    }

    /// `len` bytes at a device address of this rig's, downloaded after the stream drains.
    pub fn bytes(r: *Rig, p: ?*anyopaque, len: usize) ![]u8 {
        try r.dev.s.synchronize();
        try r.host.resize(r.a, len);
        const b, const off = try r.find(p);
        try b.download(r.dev.s, off, r.host.items);
        return r.host.items;
    }

    /// Writes `data` at a device address of this rig's (a run restoring a state it updates in place).
    pub fn put(r: *Rig, p: ?*anyopaque, data: []const u8) !void {
        try r.dev.s.synchronize();
        const b, const off = try r.find(p);
        try b.upload(r.dev.s, off, data);
    }

    /// Sets the arguments in order and queues the kernel; the call is waited for.
    pub fn launch(r: *Rig, k: xpu.Kernel, groups: [3]u32, args: anytype) !void {
        try xpu.launch.dispatch(k, r.dev.s, groups, args);
        try r.dev.s.synchronize();
    }

    fn gather(r: *Rig, kind: Kind, out: *std.ArrayList(u8), whole: bool) ![]const u8 {
        try r.dev.s.synchronize();
        out.clearRetainingCapacity();
        for (r.slots.items) |s| if (s.kind == kind or whole) {
            const at = out.items.len;
            const n = if (whole) s.len + 2 * guard else s.len;
            try out.resize(r.a, at + n);
            try s.buf.download(r.dev.s, if (whole) 0 else guard, out.items[at..][0..n]);
        };
        return out.items;
    }

    pub fn outputBytes(r: *Rig) ![]const u8 {
        return r.gather(.output, &r.outs, false);
    }

    pub fn inputBytes(r: *Rig) ![]const u8 {
        return r.gather(.input, &r.seen, false);
    }

    /// Every guard of every buffer still holds its pattern.
    pub fn intact(r: *Rig) !bool {
        const all = try r.gather(.input, &r.seen, true);
        var at: usize = 0;
        for (r.slots.items) |s| {
            for (all[at..][0..guard]) |v| if (v != pattern) return false;
            for (all[at + guard + s.len ..][0..guard]) |v| if (v != pattern) return false;
            at += s.len + 2 * guard;
        }
        return true;
    }

    /// Each payload is its own allocation, so an overlap means one buffer bound as both input and output.
    pub fn overlap(r: *Rig) bool {
        for (r.slots.items, 0..) |s, i| for (r.slots.items[i + 1 ..]) |t| {
            if (s.buf.ptr == t.buf.ptr and s.kind != t.kind) return true;
        };
        return false;
    }
};
