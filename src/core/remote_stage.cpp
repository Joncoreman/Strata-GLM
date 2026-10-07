// src/core/remote_stage.cpp - see include/strata/core/remote_stage.hpp.
#include "strata/core/remote_stage.hpp"

#include <chrono>
#include <cstdio>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
using sock_t = SOCKET;
#define STRATA_CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using sock_t = int;
#define STRATA_CLOSESOCK ::close
#endif

namespace strata::core {
namespace {

constexpr uint32_t kMagic = 0x314E4D53u;                 // "SMN1"
constexpr uint64_t kMaxPayload = 256ull * 1024 * 1024;   // far above any window (8 tokens: ~400 KB)

bool net_init(std::string& err) {
#if defined(_WIN32)
    static const int rc = [] {
        WSADATA w;
        return WSAStartup(MAKEWORD(2, 2), &w);
    }();
    if (rc != 0) { err = "stage link: WSAStartup failed (" + std::to_string(rc) + ")"; return false; }
#else
    (void) err;
#endif
    return true;
}

std::string net_error(const char* what) {
#if defined(_WIN32)
    return std::string(what) + " (WSA error " + std::to_string(WSAGetLastError()) + ")";
#else
    return std::string(what) + " (" + std::strerror(errno) + ")";
#endif
}

void tune(sock_t s) {
    int one = 1;
    setsockopt(s, IPPROTO_TCP, TCP_NODELAY, (const char*) &one, sizeof one);
    int buf = 4 * 1024 * 1024;
    setsockopt(s, SOL_SOCKET, SO_SNDBUF, (const char*) &buf, sizeof buf);
    setsockopt(s, SOL_SOCKET, SO_RCVBUF, (const char*) &buf, sizeof buf);
}

bool send_all(sock_t s, const void* p, size_t n) {
    const char* c = (const char*) p;
    while (n > 0) {
        const int chunk = (int) (n > (1u << 30) ? (1u << 30) : n);
        const int w = ::send(s, c, chunk, 0);
        if (w <= 0) return false;
        c += w;
        n -= (size_t) w;
    }
    return true;
}

bool recv_all(sock_t s, void* p, size_t n) {
    char* c = (char*) p;
    while (n > 0) {
        const int chunk = (int) (n > (1u << 30) ? (1u << 30) : n);
        const int r = ::recv(s, c, chunk, 0);
        if (r <= 0) return false;
        c += r;
        n -= (size_t) r;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------------------------- StageSocket

StageSocket::~StageSocket() { close(); }

void StageSocket::close() {
    if (fd_ != kInvalid) {
        STRATA_CLOSESOCK((sock_t) fd_);
        fd_ = kInvalid;
    }
}

bool StageSocket::connect(const std::string& host_port, std::string& err) {
    if (!net_init(err)) return false;
    const size_t colon = host_port.rfind(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 >= host_port.size()) {
        err = "stage link: '" + host_port + "' is not HOST:PORT";
        return false;
    }
    const std::string host = host_port.substr(0, colon), port = host_port.substr(colon + 1);
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    // a LAN name can fail to resolve for a while (right after the network adapter restarts): try again for ~20 s,
    // and the mDNS spelling (HOST.local) as well
    for (int attempt = 0; attempt < 20 && res == nullptr; ++attempt) {
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res) != 0) res = nullptr;
        if (res == nullptr && host.find('.') == std::string::npos &&
            getaddrinfo((host + ".local").c_str(), port.c_str(), &hints, &res) != 0)
            res = nullptr;
        if (res == nullptr) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (res == nullptr) {
        err = "stage link: cannot resolve " + host_port;
        return false;
    }
    sock_t s = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (s == (sock_t) kInvalid) {
        freeaddrinfo(res);
        err = net_error("stage link: socket()");
        return false;
    }
    tune(s);
    const int rc = ::connect(s, res->ai_addr, (int) res->ai_addrlen);
    freeaddrinfo(res);
    if (rc != 0) {
        err = net_error(("stage link: cannot connect to " + host_port).c_str());
        STRATA_CLOSESOCK(s);
        return false;
    }
    close();
    fd_ = (uintptr_t) s;
    return true;
}

bool StageSocket::send(StageMsg type, const void* a, size_t an, const void* b, size_t bn, std::string& err) {
    if (!open()) { err = "stage link: not connected"; return false; }
    uint8_t hdr[16];
    const uint32_t m = kMagic, t = (uint32_t) type;
    const uint64_t n = (uint64_t) an + (uint64_t) bn;
    std::memcpy(hdr, &m, 4);
    std::memcpy(hdr + 4, &t, 4);
    std::memcpy(hdr + 8, &n, 8);
    const sock_t s = (sock_t) fd_;
    const auto t0 = std::chrono::steady_clock::now();
    if (!send_all(s, hdr, sizeof hdr) || (an && !send_all(s, a, an)) || (bn && !send_all(s, b, bn))) {
        err = net_error("stage link: send failed");
        close();
        return false;
    }
    bytes_out += sizeof hdr + n;
    ms_send += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

bool StageSocket::recv(StageMsg& type, std::vector<uint8_t>& payload, std::string& err) {
    if (!open()) { err = "stage link: not connected"; return false; }
    uint8_t hdr[16];
    const sock_t s = (sock_t) fd_;
    const auto t0 = std::chrono::steady_clock::now();
    // STRATA_MN_SPIN=1: poll for the reply instead of sleeping in recv() (a sleeping thread wakes late on some PCs)
    static const bool spin = [] { const char* e = std::getenv("STRATA_MN_SPIN"); return e != nullptr && e[0] == '1'; }();
    if (spin) {
        for (;;) {
            fd_set rd;
            FD_ZERO(&rd);
            FD_SET(s, &rd);
            timeval tv{0, 0};
            const int r = ::select((int) s + 1, &rd, nullptr, nullptr, &tv);
            if (r != 0) break;   // readable, or an error recv() reports below
        }
    }
    if (!recv_all(s, hdr, sizeof hdr)) {
        err = "stage link: the other end closed the connection";
        close();
        return false;
    }
    uint32_t m = 0, t = 0;
    uint64_t n = 0;
    std::memcpy(&m, hdr, 4);
    std::memcpy(&t, hdr + 4, 4);
    std::memcpy(&n, hdr + 8, 8);
    if (m != kMagic || n > kMaxPayload) {
        err = "stage link: a malformed frame (magic or size)";
        close();
        return false;
    }
    payload.resize((size_t) n);
    const auto t1 = std::chrono::steady_clock::now();
    ms_wait += std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (n && !recv_all(s, payload.data(), (size_t) n)) {
        err = "stage link: the connection broke inside a frame";
        close();
        return false;
    }
    bytes_in += sizeof hdr + n;
    ms_recv += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    type = (StageMsg) t;
    return true;
}

// -------------------------------------------------------------------------------------------- StageListener

StageListener::~StageListener() {
    if (fd_ != StageSocket::kInvalid) STRATA_CLOSESOCK((sock_t) fd_);
}

bool StageListener::listen(int port, const std::string& allow, std::string& err) {
    if (!net_init(err)) return false;
    allow_.clear();
    any_ = false;
    std::stringstream ss(allow);
    std::string a;
    while (std::getline(ss, a, ',')) {
        if (a == "any") any_ = true;
        else if (!a.empty()) allow_.push_back(a);
    }
    if (!any_ && allow_.empty()) { err = "stage listen: --stage-allow names no address"; return false; }
    sock_t s = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == (sock_t) StageSocket::kInvalid) { err = net_error("stage listen: socket()"); return false; }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char*) &one, sizeof one);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t) port);
    if (::bind(s, (const sockaddr*) &addr, sizeof addr) != 0 || ::listen(s, 1) != 0) {
        err = net_error(("stage listen: cannot listen on port " + std::to_string(port)).c_str());
        STRATA_CLOSESOCK(s);
        return false;
    }
    fd_ = (uintptr_t) s;
    return true;
}

bool StageListener::accept(StageSocket& out, std::string& peer, std::string& err) {
    for (;;) {
        sockaddr_in from{};
#if defined(_WIN32)
        int len = sizeof from;
#else
        socklen_t len = sizeof from;
#endif
        const sock_t c = ::accept((sock_t) fd_, (sockaddr*) &from, &len);
        if (c == (sock_t) StageSocket::kInvalid) { err = net_error("stage listen: accept()"); return false; }
        char ip[INET_ADDRSTRLEN] = {};
        inet_ntop(AF_INET, &from.sin_addr, ip, sizeof ip);
        peer = ip;
        bool ok = any_;
        for (const std::string& a : allow_) ok = ok || a == peer;
        if (!ok) {
            std::fprintf(stderr, "strata stage: refused a connection from %s (not in --stage-allow)\n", ip);
            STRATA_CLOSESOCK(c);
            continue;
        }
        tune(c);
        out.close();
        out.fd_ = (uintptr_t) c;
        return true;
    }
}

// ------------------------------------------------------------------------------------------------ StageLink

bool StageLink::call(StageMsg type, const void* a, size_t an, const void* b, size_t bn, std::vector<uint8_t>& reply,
                     std::string& err) {
    if (!s_.send(type, a, an, b, bn, err)) return false;
    StageMsg rt = StageMsg::Err;
    if (!s_.recv(rt, reply, err)) return false;
    if (rt == StageMsg::Err) {
        err = "the remote stage: " + std::string(reply.begin(), reply.end());
        return false;
    }
    if (rt != StageMsg::Ok) { err = "stage link: an unexpected reply"; return false; }
    return true;
}

bool StageLink::connect(const std::string& host_port, const StageHello& want, std::string& err) {
    if (!s_.connect(host_port, err)) return false;
    std::vector<uint8_t> r;
    if (!call(StageMsg::Hello, &want, sizeof want, nullptr, 0, r, err)) return false;
    StageHello got;
    if (r.size() != sizeof got) { err = "stage link: a malformed Hello reply"; return false; }
    std::memcpy(&got, r.data(), sizeof got);
    if (got.lb != want.lb || got.le != want.le || got.hf != want.hf || got.max_t < want.max_t) {
        char b[256];
        std::snprintf(b, sizeof b, "the remote stage runs layers %lld-%lld (hand-off %lld floats, T<=%lld); this engine "
                      "needs %lld-%lld (%lld floats, T<=%lld)", (long long) got.lb, (long long) got.le - 1,
                      (long long) got.hf, (long long) got.max_t, (long long) want.lb, (long long) want.le - 1,
                      (long long) want.hf, (long long) want.max_t);
        err = b;
        return false;
    }
    h_ = want;
    return true;
}

bool StageLink::window(int T, int64_t pos0, const int32_t* tokens, const float* hand_in, float* hand_out,
                       std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> head(16 + sizeof(int32_t) * (size_t) T);
    std::memcpy(head.data(), &T, 4);
    std::memcpy(head.data() + 4, &pos0, 8);
    std::memcpy(head.data() + 12, &pending_commit_, 4);   // the previous window's commit, applied first
    std::memcpy(head.data() + 16, tokens, sizeof(int32_t) * (size_t) T);
    const size_t hb = sizeof(float) * (size_t) T * (size_t) h_.hf;
    std::vector<uint8_t> r;
    if (!call(StageMsg::Window, head.data(), head.size(), hand_in, hb, r, err)) return false;
    if (r.size() != hb) { err = "stage link: a Window reply of the wrong size"; return false; }
    std::memcpy(hand_out, r.data(), hb);
    pending_commit_ = 0;
    ms_net += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++windows;
    return true;
}

bool StageLink::commit(int n_keep, std::string& err) {
    (void) err;
    pending_commit_ = n_keep;   // sent with the next Window: nothing reads the stage's state before that
    ++commits;
    return true;
}

bool StageLink::reset(std::string& err) {
    pending_commit_ = 0;   // the session starts empty: a pending commit no longer applies
    std::vector<uint8_t> r;
    return call(StageMsg::Reset, nullptr, 0, nullptr, 0, r, err);
}

void StageLink::quit() {
    std::string e;
    if (s_.open()) s_.send(StageMsg::Quit, nullptr, 0, nullptr, 0, e);
    s_.close();
}

}  // namespace strata::core
