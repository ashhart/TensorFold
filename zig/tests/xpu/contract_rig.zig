//! The XPU contract's device: the card by PCI id, its images, and guarded buffers with poisoned outputs.
const std = @import("std");
const xpu = @import("xpu");
const ze = @import("ze");
const core = @import("core");
const images = @import("xpu_contract_images");

const guarded = @import("contract_guarded.zig");

pub const guard = guarded.guard;

/// The image a kernel lives in: the Qwen3.8-27B glue, TP collectives, gate 5's controls, or gates 1 to 4's.
pub const Image = enum { glue, tp, decode, attn_upstream, merge_upstream, controls, gate };

fn blob(i: Image) []const u8 {
    return switch (i) {
        .glue => &images.glue,
        .tp => &images.tp,
        .decode => &images.decode,
        .attn_upstream => &images.attn_upstream,
        .merge_upstream => &images.merge_upstream,
        .controls => &images.controls,
        .gate => &images.gate_controls,
    };
}

/// One card, its context and stream, and the contract images loaded native for it.
pub const Dev = struct {
    drv: xpu.Driver,
    ctx: xpu.Context,
    s: xpu.Stream,
    glue: xpu.Module,
    tp: xpu.Module,
    decode: xpu.Module,
    attn_upstream: xpu.Module,
    merge_upstream: xpu.Module,
    controls: ?xpu.Module, // the driver refuses the whole image when one control asks for more SLM than it has
    controls_err: ?anyerror = null,
    gate: xpu.Module,

    /// Opens the card `ordinal` names, else the first Arc card, and loads every image; `self` stays where it is.
    pub fn init(self: *Dev, ordinal: ?u32) !void {
        self.drv = try xpu.Driver.open();
        self.ctx = try xpu.Context.init(&self.drv, ordinal);
        self.s = try xpu.Stream.init(&self.ctx);
        self.glue = try xpu.Module.loadNative(&self.ctx, blob(.glue));
        self.tp = try xpu.Module.loadNative(&self.ctx, blob(.tp));
        self.decode = try xpu.Module.loadNative(&self.ctx, blob(.decode));
        self.attn_upstream = try xpu.Module.loadNative(&self.ctx, blob(.attn_upstream));
        self.merge_upstream = try xpu.Module.loadNative(&self.ctx, blob(.merge_upstream));
        self.controls = xpu.Module.loadNative(&self.ctx, blob(.controls)) catch |e| blk: {
            self.controls_err = e;
            break :blk null;
        };
        self.gate = try xpu.Module.loadNative(&self.ctx, blob(.gate));
    }

    pub fn deinit(self: *Dev) void {
        self.gate.unload();
        if (self.controls) |*m| m.unload();
        self.tp.unload();
        self.glue.unload();
        self.decode.unload();
        self.attn_upstream.unload();
        self.merge_upstream.unload();
        self.s.deinit();
        self.ctx.deinit();
        self.drv.close();
    }

    pub fn module(self: *Dev, i: Image) !*xpu.Module {
        return switch (i) {
            .glue => &self.glue,
            .tp => &self.tp,
            .decode => &self.decode,
            .attn_upstream => &self.attn_upstream,
            .merge_upstream => &self.merge_upstream,
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

/// One call's device buffers on the contract's card.
pub const Rig = guarded.Rig(Dev);
