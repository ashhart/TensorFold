//! Weight placement over one Metal device and its inference command queue, with tracked immutable buffers.
pub fn Backend(comptime metal: type) type {
    return struct {
        const Self = @This();
        pub const Buffer = metal.Buffer;
        device: metal.Device,
        queue: metal.Queue,
        commands: ?*metal.CommandSession = null,
        pub fn allocationBytes(n: usize) !u64 {
            return n;
        }
        fn check(self: *Self) !void {
            if (self.commands) |commands| if (commands.queue.id != self.queue.id) return error.Qwen4DeviceContext;
            if (metal.objc.msg(metal.objc.Id, self.queue.id, "device", .{}) != self.device.id)
                return error.Qwen4DeviceContext;
        }
        pub fn allocate(self: *Self, n: usize) !Buffer {
            try self.check();
            return self.device.buffer(n, metal.ResourceOptions.shared);
        }
        fn span(self: *Self, b: Buffer, offset: usize, n: usize) !void {
            try self.check();
            if (b.deviceId() != self.device.id) return error.Qwen4DeviceContext;
            if (offset > b.length() or n > b.length() - offset) return error.Qwen4StateRange;
        }
        pub fn zero(self: *Self, b: Buffer, offset: usize, n: usize) !void {
            try self.span(b, offset, n);
            if (n == 0) return;
            if (self.commands) |commands| return commands.zero(b, offset, n);
            const pool = metal.objc.Pool.push();
            defer pool.pop();
            const cb = self.queue.commandBuffer();
            const e = cb.blit();
            metal.objc.msg(void, e.id, "fillBuffer:range:value:", .{ b.id, metal.types.Range{ .location = offset, .length = n }, @as(u8, 0) });
            e.end();
            cb.commit();
        }
        pub fn copy(self: *Self, to: Buffer, to_offset: usize, from: Buffer, from_offset: usize, n: usize) !void {
            try self.span(to, to_offset, n);
            try self.span(from, from_offset, n);
            if (n == 0) return;
            if (self.commands) |commands| return commands.copy(to, to_offset, from, from_offset, n);
            const pool = metal.objc.Pool.push();
            defer pool.pop();
            const cb = self.queue.commandBuffer();
            const e = cb.blit();
            e.copy(from, from_offset, to, to_offset, n);
            e.end();
            cb.commit();
        }
        pub fn upload(self: *Self, bytes: []const u8) !Buffer {
            try self.check();
            const buffer = try self.device.buffer(bytes.len, metal.ResourceOptions.shared);
            @memcpy(buffer.contents()[0..bytes.len], bytes);
            return buffer;
        }
        /// A retained command buffer owns the immutable staging buffer after its blit has been encoded.
        pub fn write(self: *Self, to: Buffer, offset: usize, bytes: []const u8) !void {
            try self.span(to, offset, bytes.len);
            const commands = self.commands orelse return error.Qwen4CommandSessionMissing;
            if (bytes.len == 0) return;
            const staging = try self.upload(bytes);
            defer self.release(staging);
            try commands.copy(to, offset, staging, 0, bytes.len);
        }
        pub fn release(_: *Self, buffer: Buffer) void {
            buffer.deinit();
        }
        pub fn drain(self: *Self) !void {
            try self.check();
            if (self.commands) |commands| return commands.drain();
            const pool = metal.objc.Pool.push();
            defer pool.pop();
            const cb = self.queue.commandBuffer();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.Qwen4DeviceFailed;
        }
    };
}
