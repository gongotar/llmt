// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Masoud Jami
#include "doctest.h"
#include "llmt/llmt.h"

#include <cstring>

TEST_CASE("library links and reports a version") {
    CHECK(std::strlen(llmt::version()) > 0);
}
