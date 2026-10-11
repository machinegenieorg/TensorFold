//! A committed admission handoff sends each token and row once and keeps terminal jobs owned until return.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const gpa = std.testing.allocator;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: usize = 0,
    rows: usize = 0,
    prefills: usize = 0,
    finishes: usize = 0,
    reason: ?api.Reason = null,

    fn event(ptr: *anyopaque, _: api.Id, e: *const api.Event) void {
        const x: *Box = @ptrCast(@alignCast(ptr));
        x.mutex.lockUncancelable(std.testing.io);
        defer x.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| x.tokens += t.len,
            .logprobs => |r| x.rows += r.len,
            .prefilled => x.prefills += 1,
            .finished => |done| {
                x.finishes += 1;
                x.reason = done.reason;
            },
        }
    }

    fn wait(x: *Box) !void {
        const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 5 * std.time.ns_per_s;
        while (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() < deadline) {
            x.mutex.lockUncancelable(std.testing.io);
            const done = x.reason != null;
            x.mutex.unlock(std.testing.io);
            if (done) return;
            try std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake);
        }
        return error.HostDeadline;
    }
};

const Target = struct {
    fake: lanes.fake.Fake = .{ .gpa = gpa },
    table: lanes.backend.Backend.VTable = undefined,
    boxes: [2]*Box,
    fail: bool,
    initial_calls: usize = 0,

    fn backend(x: *Target) lanes.backend.Backend {
        var b = x.fake.backend();
        x.table = b.vtable.*;
        x.table.draft = draft;
        b.vtable = &x.table;
        b.first_before_draft = true;
        return b;
    }

    fn draft(ptr: *anyopaque, requests: []const lanes.backend.DraftRequest) !void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        const x: *Target = @fieldParentPtr("fake", f);
        if (requests.len > 0 and requests[0].first != null) {
            x.initial_calls += 1;
            try std.testing.expectEqual(@as(usize, 1), requests.len);
            for (x.boxes) |box| {
                try std.testing.expectEqual(@as(usize, 1), box.tokens);
                try std.testing.expectEqual(@as(usize, 1), box.rows);
                try std.testing.expectEqual(@as(usize, 0), box.finishes);
            }
            if (x.fail) return error.InitialDraftFailed;
        }
        return x.fake.backend().draft(requests);
    }
};

test "host handoff delivers first rows once and preserves terminal success across draft failure" {
    for ([_]bool{ false, true }) |fail| {
        var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 4, .hidden_rows = true, .draft_streams = true }, 8, 4);
        defer cfg.deinit(gpa);
        var terminal: Box = .{};
        var live: Box = .{};
        var target: Target = .{ .boxes = .{ &terminal, &live }, .fail = fail };
        defer target.fake.deinit();
        var clock: lanes.fake.FixedClock = .{};
        var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
        defer core.deinit();
        var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
        const e = host.engine();
        const one: api.Request = .{ .prompt = &.{ 2, 7 }, .max_tokens = 1, .logprobs = 2 };
        const many: api.Request = .{ .prompt = &.{ 1, 8 }, .max_tokens = 4, .logprobs = 2 };
        try e.submit(1, &one, .{ .ctx = &terminal, .event = Box.event });
        try e.submit(2, &many, .{ .ctx = &live, .event = Box.event });
        try host.start();
        defer host.stop();
        try terminal.wait();
        try live.wait();
        try std.testing.expectEqual(@as(usize, 1), terminal.tokens);
        try std.testing.expectEqual(@as(usize, 1), terminal.rows);
        try std.testing.expectEqual(api.Reason.length, terminal.reason.?);
        try std.testing.expectEqual(@as(usize, if (fail) 1 else 4), live.tokens);
        try std.testing.expectEqual(live.tokens, live.rows);
        try std.testing.expectEqual(if (fail) api.Reason.failed else api.Reason.length, live.reason.?);
        try std.testing.expectEqual(@as(usize, 1), target.initial_calls);
        for ([_]*Box{ &terminal, &live }) |box| {
            try std.testing.expectEqual(@as(usize, 1), box.prefills);
            try std.testing.expectEqual(@as(usize, 1), box.finishes);
        }
        try std.testing.expectEqual(@as(usize, 0), target.fake.lanes.count());
    }
}
