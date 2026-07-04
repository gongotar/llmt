#include "doctest.h"
#include "llmt/llmt.h"

#include <cstring>

TEST_CASE("library links and reports a version") {
    CHECK(std::strlen(llmt::version()) > 0);
}
