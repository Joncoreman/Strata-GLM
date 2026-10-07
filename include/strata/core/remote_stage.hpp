// include/strata/core/remote_stage.hpp - a layer-split stage on ANOTHER PC (multinode branch).
//
// The layer split (docs/MULTI_GPU.md) hands one buffer per verify window from a stage to the next.  Here the
// middle stage lives in another engine process, usually on another PC: the head engine sends the window's tokens
// and its hand-off over one TCP connection, the stage engine runs its layers and sends its hand-off back.  Every
// request frame gets exactly one reply frame, so the head and the stage stay in lockstep.
//
//   frame   = magic u32 | type u32 | payload bytes u64 | payload
//   Hello   = lb i64 | le i64 | handoff floats per token i64 | max T i64      -> Ok (the stage's own four) / Err
//   Window  = T i32 | pos0 i64 | commit first i32 (0: none) | tokens i32[T] | hand-off f32[T * hf]
//                                                                         -> Ok (hand-off f32[T * hf]) / Err
//   (the head's commit of the previous window rides in the next Window: one round trip per window)
//   Commit  = n_keep i32                                                      -> Ok / Err
//   Reset   = (empty): the stage zeroes its session (a request read from token 0) -> Ok / Err
//   Quit    = (empty): the stage closes the connection, no reply
//   Err     = a message (UTF-8, no terminator)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

/// Ping = any bytes -> Ok (the same bytes): the link's own round trip, nothing computed (mn-test/ping.py)
enum class StageMsg : uint32_t { Hello = 1, Window = 2, Commit = 3, Reset = 4, Quit = 5, Ping = 6, Ok = 100, Err = 101 };

struct StageHello {
    int64_t lb = 0, le = 0, hf = 0, max_t = 0;
};

/// One blocking TCP connection carrying frames (both ends use it).
class StageSocket {
public:
    StageSocket() = default;
    ~StageSocket();
    StageSocket(const StageSocket&) = delete;
    StageSocket& operator=(const StageSocket&) = delete;

    bool connect(const std::string& host_port, std::string& err);
    bool open() const { return fd_ != kInvalid; }
    void close();
    /// One frame: the payload is `a` then `b` (either may be empty).
    bool send(StageMsg type, const void* a, size_t an, const void* b, size_t bn, std::string& err);
    bool recv(StageMsg& type, std::vector<uint8_t>& payload, std::string& err);

    uint64_t bytes_out = 0, bytes_in = 0;
    double ms_send = 0, ms_wait = 0, ms_recv = 0;   ///< time in send(), before the reply's first byte, reading it

private:
    friend class StageListener;
    static constexpr uintptr_t kInvalid = ~(uintptr_t) 0;
    uintptr_t fd_ = kInvalid;
};

/// The stage engine's listening socket: one head at a time, from the allowed addresses only.
class StageListener {
public:
    StageListener() = default;
    ~StageListener();
    StageListener(const StageListener&) = delete;
    StageListener& operator=(const StageListener&) = delete;

    /// `allow`: comma-separated IPv4 addresses a head may connect from ("any" accepts every address).
    bool listen(int port, const std::string& allow, std::string& err);
    /// Blocks for the next allowed head; a refused address is closed and logged, and the wait goes on.
    bool accept(StageSocket& out, std::string& peer, std::string& err);

private:
    uintptr_t fd_ = StageSocket::kInvalid;
    std::vector<std::string> allow_;
    bool any_ = false;
};

/// The head engine's end of the link: the request/reply calls the proxy verifier makes.
class StageLink {
public:
    /// Connects and exchanges Hello: the stage must run exactly [want.lb, want.le) with the same hand-off size.
    bool connect(const std::string& host_port, const StageHello& want, std::string& err);
    bool window(int T, int64_t pos0, const int32_t* tokens, const float* hand_in, float* hand_out, std::string& err);
    bool commit(int n_keep, std::string& err);
    bool reset(std::string& err);
    void quit();

    double ms_net = 0;     ///< wall time of the Window calls (the stage's compute included)
    int64_t windows = 0;
    double ms_commit = 0;  ///< wall time of the Commit calls (one round trip each)
    int64_t commits = 0;
    const StageSocket& socket() const { return s_; }

private:
    bool call(StageMsg type, const void* a, size_t an, const void* b, size_t bn, std::vector<uint8_t>& reply,
              std::string& err);
    StageSocket s_;
    StageHello h_;
    int32_t pending_commit_ = 0;   ///< commit() keeps it for the next Window (reset drops it)
};

}  // namespace strata::core
