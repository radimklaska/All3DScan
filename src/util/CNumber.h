#pragma once

// Number parsing that ignores the process locale.
//
// std::strtod and std::strtof follow LC_NUMERIC, and Qt switches a process to
// the system locale. Under a decimal-comma locale (cs_CZ, de_DE, fr_FR, ...)
// they read "0.9393" as 0 and stop at the dot, so the scanner's factory
// calibration came out wrong without any error. These behave exactly like
// strtod/strtof in the "C" locale, whatever the process locale is.

#include <cstdlib>
#include <locale.h>
#if defined(__APPLE__)
#include <xlocale.h>
#endif

namespace revoscan::util {

inline locale_t cNumericLocale() {
    static const locale_t c = newlocale(LC_NUMERIC_MASK, "C", static_cast<locale_t>(0));
    return c;
}

inline double strtodC(const char* text, char** end) {
    return strtod_l(text, end, cNumericLocale());
}

inline float strtofC(const char* text, char** end) {
    return strtof_l(text, end, cNumericLocale());
}

} // namespace revoscan::util
