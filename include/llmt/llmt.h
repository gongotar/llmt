// llmt — LLM training library. Public umbrella header.
// Grows as components land (see DESIGN.md §7); everything lives in namespace llmt.
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
#endif

namespace llmt {

// Library version, defined in src/core/version.cpp.
const char* version() noexcept;

}  // namespace llmt
