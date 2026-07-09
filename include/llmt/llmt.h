// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
// llmt — LLM training library. Public umbrella header.
// Grows as components land (see docs/DESIGN.md §7); everything lives in namespace llmt.
#pragma once

#include "llmt/core/dtype.h"
#include "llmt/core/precision.h"
#include "llmt/core/rng.h"
#include "llmt/core/shape.h"
#include "llmt/core/tensor.h"

#ifdef LLMT_HAS_CUDA
#include "llmt/core/arena.h"
#include "llmt/core/device.h"
#include "llmt/core/error.h"
#include "llmt/core/run_ctx.h"
#include "llmt/kernels/fill.h"
#include "llmt/model/init.h"
#include "llmt/model/param_store.h"
#endif

namespace llmt {

/** Library version string ("0.1.0-dev"); defined in src/core/version.cpp. */
const char* version() noexcept;

}  // namespace llmt
