// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// ParamStore (Phase 3): registration/finalize/offsets, lookup, aliasing,
// role-keyed init statistics, bitwise reproducibility & layout independence.
#include <cmath>
#include <cstring>
#include <vector>

#include "doctest.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/model/init.h"
#include "llmt/model/param_store.h"
#include "transfer.h"

using namespace llmt;
using namespace llmt::testing;

// The name hash is a compile-time fact: distinct names → distinct streams.
static_assert(detail::fnv1a("blk0.attn.wq") != detail::fnv1a("blk0.attn.wk"));
static_assert(detail::fnv1a("") == 0xCBF29CE484222325ull);  // FNV offset basis
// The dtype table is a compile-time value; uniform() fills every kind.
static_assert(StateDtypes::uniform(DType::FP32)[StateKind::AdamV] == DType::FP32);

namespace {

struct Moments {
    double mean, std;
};

Moments moments(const std::vector<float>& v) {
    double sum = 0, sq = 0;
    for (const float x : v) {
        sum += x;
        sq += double(x) * x;
    }
    const double mean = sum / static_cast<double>(v.size());
    return {mean, std::sqrt(sq / static_cast<double>(v.size()) - mean * mean)};
}

}  // namespace

TEST_CASE("ParamStore: registration, finalize, offsets") {
    ParamStore ps(StateDtypes::uniform(DType::FP32));
    Param& a = ps.add("a", {8, 16}, Role::Matrix);  // 128 el = 512 B
    Param& b = ps.add("b", {64}, Role::Norm);       // 64 el = 256 B

    CHECK(!ps.finalized());
    CHECK(!a.weight.valid());  // views are null until finalize
    CHECK(!b.grad.valid());
    CHECK(ps.param_numel() == 128 + 64);

    ps.finalize();
    CHECK(ps.finalized());

    // Views valid, shapes carried over, offsets fixed and aligned.
    // (Offsets are derived, not stored: pointer minus span base.)
    CHECK(a.weight.valid());
    CHECK(a.weight.shape == Shape{8, 16});
    CHECK(a.weight.data == ps.flat(StateKind::Weight).data);  // first param at offset 0
    CHECK(reinterpret_cast<uintptr_t>(a.weight.data) % Arena::kAlign == 0);
    CHECK(reinterpret_cast<uintptr_t>(b.weight.data) % Arena::kAlign == 0);

    // Parallel spans: grad sits at the same offset in its own span.
    CHECK(static_cast<char*>(b.weight.data) - static_cast<char*>(a.weight.data) == 512);
    CHECK(static_cast<char*>(b.grad.data) - static_cast<char*>(a.grad.data) == 512);
    CHECK(a.grad.data != a.weight.data);

    // Flat spans cover both params incl. padding: (512 + 256) / 4 = 192 el.
    CHECK(ps.flat(StateKind::Weight).numel() == 192);
    CHECK(ps.flat(StateKind::Grad).numel() == 192);
    CHECK(ps.flat(StateKind::Weight).data == a.weight.data);
}

TEST_CASE("ParamStore invariant: all four spans (incl. padding) start zero") {
    ParamStore ps(StateDtypes::uniform(DType::FP32));
    ps.add("a", {8, 16}, Role::Matrix);
    ps.add("b", {3}, Role::Norm);  // 12 B → 244 B of padding inside the span
    ps.finalize();
    for (int k = 0; k < kNumStateKinds; ++k) {
        const std::vector<float> h = download(ps.flat(StateKind(k)));
        for (const float x : h) REQUIRE(x == 0.0f);
    }
}

// NOTE: twin-ness holds between spans of equal dtype (see class comment);
// this test constructs the uniform-dtype config, where all four are twins.
TEST_CASE("ParamStore invariant: the four spans are layout twins") {
    ParamStore ps(StateDtypes::uniform(DType::FP32));
    ps.add("a", {8, 16}, Role::Matrix);
    ps.add("b", {64}, Role::Norm);
    ps.alias("a2", "a");
    ps.finalize();

    // One offset locates a parameter's slice in every span.
    const auto off = [&ps](const Param& p, StateKind k) {
        return static_cast<const char*>(p.state(k).data) -
               static_cast<char*>(ps.flat(k).data);
    };
    for (const Param& p : ps.params()) {
        const auto o = off(p, StateKind::Weight);
        for (int k = 1; k < kNumStateKinds; ++k) {
            REQUIRE(off(p, StateKind(k)) == o);
            REQUIRE(p.state(StateKind(k)).shape == p.weight.shape);  // M1 twins
        }
        // state(k) and the named fields are the same objects.
        REQUIRE(&p.state(StateKind::AdamM) == &p.adam_m);
        REQUIRE(&p.state(StateKind::Grad) == &p.grad);
    }
}

TEST_CASE("ParamStore: named lookup, alias identity, role iteration") {
    ParamStore ps(StateDtypes::uniform(DType::FP32));
    ps.add("embed.wte", {512, 64}, Role::Embedding);
    ps.add("blk0.norm.g", {64}, Role::Norm);
    ps.alias("lm_head.w", "embed.wte");  // weight tying (invariant 9)
    ps.finalize();

    // Both names resolve to the same memory — weight AND grad.
    CHECK(ps.at("lm_head.w").weight.data == ps.at("embed.wte").weight.data);
    CHECK(ps.at("lm_head.w").grad.data == ps.at("embed.wte").grad.data);
    CHECK(&ps.at("lm_head.w") == &ps.at("embed.wte"));

    // Aliases add no storage: iteration sees unique params only.
    CHECK(ps.params().size() == 2);

    int n_embedding = 0, n_norm = 0;
    for (const Param& p : ps.params()) {
        if (p.role == Role::Embedding) ++n_embedding;
        if (p.role == Role::Norm) ++n_norm;
    }
    CHECK(n_embedding == 1);
    CHECK(n_norm == 1);
}

TEST_CASE("init_params: role-keyed statistics") {
    Device dev(0);
    ParamStore ps(StateDtypes::uniform(DType::FP32));
    ps.add("emb", {256, 128}, Role::Embedding);      // 32768 el
    ps.add("mat", {128, 128}, Role::Matrix);         // 16384 el
    ps.add("resid", {128, 128}, Role::ResidualProj); // 16384 el
    ps.add("norm", {128}, Role::Norm);
    ps.finalize();

    constexpr int kLayers = 8;  // → resid std = 0.02 / 4 = 0.005
    init_params(ps, dev.stream(), /*seed=*/7, kLayers);
    dev.synchronize();

    const Moments emb = moments(download(ps.at("emb").weight));
    CHECK(std::abs(emb.mean) < 1e-3);
    CHECK(emb.std == doctest::Approx(0.02).epsilon(0.02));

    const Moments mat = moments(download(ps.at("mat").weight));
    CHECK(mat.std == doctest::Approx(0.02).epsilon(0.03));

    const Moments resid = moments(download(ps.at("resid").weight));
    CHECK(resid.std == doctest::Approx(0.005).epsilon(0.03));

    for (const float x : download(ps.at("norm").weight)) REQUIRE(x == 1.0f);

    // Different parameters draw from different streams: emb != mat bytes.
    const std::vector<float> he = download(ps.at("emb").weight);
    const std::vector<float> hm = download(ps.at("mat").weight);
    CHECK(std::memcmp(he.data(), hm.data(), 16384 * sizeof(float)) != 0);
}

TEST_CASE("init_params: bitwise-reproducible and layout-independent") {
    Device dev(0);
    constexpr uint64_t kSeed = 1337;
    constexpr int kLayers = 6;

    // Same registrations, DIFFERENT order → different offsets in the spans.
    ParamStore ps1(StateDtypes::uniform(DType::FP32));
    ps1.add("w1", {64, 64}, Role::Matrix);
    ps1.add("w2", {32, 32}, Role::ResidualProj);
    ps1.finalize();

    ParamStore ps2(StateDtypes::uniform(DType::FP32));
    ps2.add("w2", {32, 32}, Role::ResidualProj);  // reversed order
    ps2.add("w1", {64, 64}, Role::Matrix);
    ps2.finalize();
    // Layouts really differ: w1's byte offset (pointer minus span base).
    const auto off = [](const ParamStore& ps, const char* n) {
        return static_cast<const char*>(ps.at(n).weight.data) -
               static_cast<const char*>(ps.flat(StateKind::Weight).data);
    };
    CHECK(off(ps1, "w1") != off(ps2, "w1"));

    init_params(ps1, dev.stream(), kSeed, kLayers);
    init_params(ps2, dev.stream(), kSeed, kLayers);
    dev.synchronize();

    // Per-tensor bytes identical across layouts (stream_id = name hash).
    for (const char* name : {"w1", "w2"}) {
        const std::vector<float> h1 = download(ps1.at(name).weight);
        const std::vector<float> h2 = download(ps2.at(name).weight);
        REQUIRE(std::memcmp(h1.data(), h2.data(), h1.size() * sizeof(float)) == 0);
    }

    // Re-init of the same store is bitwise-identical too.
    const std::vector<float> before = download(ps1.at("w1").weight);
    init_params(ps1, dev.stream(), kSeed, kLayers);
    dev.synchronize();
    const std::vector<float> after = download(ps1.at("w1").weight);
    CHECK(std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0);
}
