//! #1594 step 2b on the GB10: serve --mcdma-produce LINK exports handoff requests' Qwen3.8-27B states over MKVH.
const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const fabric = @import("fabric");
const lanes = @import("lanes");
const qwen35 = @import("qwen35");
const reuse = qwen35.reuse;
const hs = fabric.hybrid_state;
const hx = fabric.hybrid_export;
const producer = fabric.producer;
const pc = api.prompt_cache;
const Handoff = api.LaneHost.Export.Handoff;

/// Tokens to an attention page row: the Mac's pull (native/qwen27_remote.zig) plans the same.
const block: u32 = 16;

/// One recorded export: the family's host copy, the prompt it holds, and the export's layers and sources over it.
const Held = struct { copy: *reuse.Copy, arena: std.heap.ArenaAllocator, exp: producer.Export, name: [32]u8 };

/// The producer: a listen mailbox, its serving thread, and the exports it holds until the decoder closes them.
pub const Produce = struct {
    gpa: std.mem.Allocator,
    target: *reuse.Target,
    box: fabric.mailbox.Mailbox,
    reg: ?fabric.registration.Registration = null, // MODE poll: the listen daemon sends staged replies only to a service
    link: []const u8,
    table: producer.Table,
    serving: producer.Serving,
    thread: std.Thread = undefined,
    held: std.AutoHashMapUnmanaged([16]u8, *Held) = .empty,
    lock: std.atomic.Mutex = .unlocked, // `held`: the engine thread adds, the serving thread releases

    pub fn exporter(p: *Produce) api.LaneHost.Export.Exporter {
        return .{ .ptr = p, .snapshots = .{ .ptr = p, .vtable = &.{ .bytes = bytes, .save = save, .restore = restore, .drop = drop } }, .exported = exported, .failed = failed };
    }

    /// Stops serving and frees every export still held (after the lane host stopped: no export comes in).
    pub fn stop(p: *Produce) void {
        if (comptime builtin.os.tag != .linux) return; // `open` makes none off Linux
        p.serving.stop.store(true, .release);
        p.thread.join();
        var it = p.held.valueIterator();
        while (it.next()) |h| release(p, h.*);
        p.held.deinit(p.gpa);
        if (p.reg) |*r| r.close();
        p.serving.deinit();
        p.table.deinit();
        p.box.close();
        p.gpa.destroy(p);
    }

    fn run(p: *Produce) void {
        var done: std.ArrayList([]const u8) = .empty;
        defer done.deinit(p.gpa);
        var since: u64 = 0;
        while (!p.serving.stop.load(.acquire)) {
            p.table.expire() catch {}; // exports age out and are freed while the link is down too
            p.table.takeFinished(&done) catch {};
            for (done.items) |name| reap(p, name);
            done.clearRetainingCapacity();
            if (!p.attached(&since)) continue;
            // a new connection's HELLO zeroes the request word and its client counts from 1 again
            if (fabric.words.load(p.box.wordAt(fabric.layout.request_word)) == 0) p.serving.service.last = 0;
            if (p.serving.service.next(std.time.ns_per_ms) catch return) |req|
                p.serving.responder.handle(req.seq, req.payload) catch |err| p.serving.responder.failed(req.seq, err) catch return;
        }
    }

    /// Registered with the listen daemon; a dropped registration (the link went down) is retried each second.
    fn attached(p: *Produce, since: *u64) bool {
        if (p.reg) |*r| {
            if (r.alive()) return true;
            r.close();
            p.reg = null;
            std.log.warn("mcdma handoff: the listen daemon dropped the service on link {s}; attaching again", .{p.link});
        }
        const now = fabric.words.nowNs();
        if (now - since.* < std.time.ns_per_s) {
            fabric.words.pause(std.time.ns_per_ms);
            return false;
        }
        since.* = now;
        p.reg = fabric.registration.Registration.attach(p.link) catch return false;
        std.log.info("mcdma handoff: attached to the listen daemon on link {s}", .{p.link});
        return true;
    }

    fn reap(p: *Produce, name: []const u8) void {
        var id: [16]u8 = undefined;
        _ = std.fmt.hexToBytes(&id, name) catch return; // a failed handoff's entry holds no copy
        while (!p.lock.tryLock()) std.atomic.spinLoopHint();
        const got = p.held.fetchRemove(id);
        p.lock.unlock();
        _ = p.serving.responder.checksums.remove(id); // CLOSE removes it; an expired export's would stay
        if (got) |kv| release(p, kv.value);
    }

    fn release(p: *Produce, h: *Held) void {
        if (h.exp.frames.len > 0) p.gpa.free(h.exp.frames); // the responder planned them with this allocator at OPEN
        reuse.dropCopy(p.target.e, h.copy);
        h.arena.deinit();
        p.gpa.destroy(h);
    }

    /// A host copy whatever the device: the export reads it in place.
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoStream));
        const cache = p.target.caches.get(s) orelse return error.UnknownStream;
        return @ptrCast(try reuse.saveCache(p.target.e, cache, at, &.{}, false));
    }

    fn bytes(ptr: *anyopaque, at: u32) u64 {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        return reuse.bytes(p.target, at);
    }

    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.ExportsAreNotRestored;
    }

    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        reuse.dropCopy(p.target.e, @ptrCast(@alignCast(saved)));
    }

    fn exported(ptr: *anyopaque, h: Handoff, tokens: []const u32, saved: pc.Saved) void {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        const copy: *reuse.Copy = @ptrCast(@alignCast(saved));
        record(p, h, tokens, copy) catch |err| {
            reuse.dropCopy(p.target.e, copy);
            failed(p, h, @errorName(err));
        };
    }

    fn record(p: *Produce, h: Handoff, tokens: []const u32, copy: *reuse.Copy) !void {
        if (copy.format != .bf16) return error.Fp8KvCachesAreNotExported;
        const at: u32 = @intCast(tokens.len);
        const plan: hx.Plan = .{ .g = hs.qwen27, .s = hx.qwen27, .at = at, .block = block, .order = .cuda };
        try plan.check(); // the decoder's prefix is whole page rows
        if (copy.dev != null or copy.len != hs.qwen27.bytes(at)) return error.HybridGeometryDiffers;
        const held = try p.gpa.create(Held);
        errdefer p.gpa.destroy(held);
        held.* = .{ .copy = copy, .arena = .init(p.gpa), .exp = undefined, .name = undefined };
        errdefer held.arena.deinit();
        const a = held.arena.allocator();
        const built = try hx.build(a, plan, copy.host.bytes[0..@intCast(copy.len)]);
        held.exp = .{ .handoff = h.id, .tokens = try a.dupe(u32, tokens), .first_token = 0, .tokens_per_row = block, .layers = built.layers, .sources = built.sources };
        _ = std.fmt.bufPrint(&held.name, "{x}", .{&h.id}) catch unreachable;
        {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            defer p.lock.unlock();
            const slot = try p.held.getOrPut(p.gpa, h.id);
            if (slot.found_existing) return error.HandoffHeld; // a reused id: the export already held stays
            slot.value_ptr.* = held;
        }
        p.table.addSized(h.id, &held.name, .{ .ready = &held.exp }, copy.len) catch |err| {
            while (!p.lock.tryLock()) std.atomic.spinLoopHint();
            _ = p.held.remove(h.id); // the caller frees the copy; the arena and Held go with the errdefers
            p.lock.unlock();
            return err;
        };
    }

    fn failed(ptr: *anyopaque, h: Handoff, reason: []const u8) void {
        const p: *Produce = @ptrCast(@alignCast(ptr));
        const why = if (std.mem.eql(u8, reason, "HeldExportsFull")) "the producer already holds TF_PRODUCE_HELD_GIB of exports; it refuses more until the decoder closes some" else reason;
        std.log.warn("mcdma handoff {x}: no export ({s})", .{ &h.id, why });
        p.table.add(h.id, "failed", .{ .failed = why }) catch {};
    }
};

/// The producer for Qwen3.8-27B on CUDA, its geometry and cache format checked first; null with `problem` otherwise.
pub fn open(comptime F: type, gpa: std.mem.Allocator, loaded: anytype, link: []const u8, model: []const u8, problem: *[]const u8) !?*Produce {
    if (comptime F != qwen35.native or builtin.os.tag != .linux) {
        problem.* = if (F != qwen35.native) "--mcdma-produce exports Qwen3.8-27B (qwen3_5) states only (#1594 step 2b)" else "--mcdma-produce runs on Linux, beside mcdma-rpcd's listen daemon";
        return null;
    }
    const t: *reuse.Target = loaded.target;
    if (reuse.formatOf(t.e) != .bf16) {
        problem.* = "--mcdma-produce exports bf16 KV caches: serve without --kv-cache fp8";
        return null;
    }
    if (reuse.layoutOf(t.e, block).total() != hs.qwen27.bytes(block)) {
        problem.* = "--mcdma-produce: this checkpoint's state is not Qwen3.8-27B's hybrid geometry";
        return null;
    }
    const p = try gpa.create(Produce);
    errdefer gpa.destroy(p);
    const cap = @import("sys").environment.read(u64, "TF_PRODUCE_HELD_GIB", 32) catch {
        problem.* = "--mcdma-produce: TF_PRODUCE_HELD_GIB is malformed; the line above names its forms";
        return error.InvalidEnvironment;
    };
    p.* = .{ .gpa = gpa, .target = t, .box = undefined, .link = link, .table = .{ .gpa = gpa, .ttl_ns = 120 * std.time.ns_per_s, .clock = fabric.words.nowNs, .cap_bytes = cap << 30 }, .serving = undefined };
    p.box = fabric.mailbox.Mailbox.openListen(link) catch |err| {
        problem.* = "--mcdma-produce: no mcdma-rpcd listen daemon serves that link (start `mcdma-rpcd listen` first)";
        return err;
    };
    errdefer p.box.close();
    p.reg = fabric.registration.Registration.attach(link) catch |err| {
        problem.* = "--mcdma-produce: the listen daemon refused the service (another holds the link, or it is not running)";
        return err;
    };
    errdefer if (p.reg) |*r| r.close();
    std.log.info("mcdma handoff: attached to the listen daemon on link {s}", .{link});
    p.serving.init(gpa, &p.box, fabric.words.native, &p.table, model);
    p.thread = try std.Thread.spawn(.{}, Produce.run, .{p});
    std.log.info("mcdma handoff: exporting Qwen3.8-27B states as MKVH protocol 2 on link {s}", .{link});
    return p;
}
