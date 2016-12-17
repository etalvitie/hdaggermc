#include "fastmath.hpp"

#include <boost/math/special_functions/log1p.hpp>

#include "jacoblog.hpp"
#include "icsilogw.hpp"
#include "PowFast.hpp"

/** Switch to disable fast math routines for debugging. */
static const bool DisableFastMath = true;


/** Allocate precomputed tables. */
static ICSILog flog(14);
static PowFast fpow(12);
static JacobianLogTable fjacoblog(1 << 12);


/** Fast, approxiate jacobian logarithm: log(1+exp(x)) */
double fast_jacoblog(double x) {

    if (DisableFastMath) return boost::math::log1p(std::exp(x));

    return fjacoblog.jacobianLog(x);
}


/** Fast, approximate natural logarithm. */
double fast_log(double x) {

    if (DisableFastMath) return std::log(x);

    return flog.log(static_cast<float>(x));
}


/** Fast, approximate exponentiation. */
double fast_exp(double x) {

    if (DisableFastMath) return std::exp(x);

    // approximation isn't accurate for these extreme ranges
    if (x <= -87.0 || x >= 88.0) return std::exp(x);

    return fpow.e(static_cast<float>(x));
}


/** Given log(x) and log(y), compute log(x+y). uses the following identity:
    log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x))) */
inline double safe_logadd(double log_x, double log_y) {

    // ensure log_y >= log_x, can save some expensive log/exp calls
    if (log_x > log_y) {
        double t = log_x; log_x = log_y; log_y = t;
    }

    double rval = log_y - log_x;

    // only replace log(1+exp(log(y)-log(x))) with log(y)-log(x)
    // if the the difference is small enough to be meaningful
    if (rval < 100.0) {
        rval = std::log(1.0 + std::exp(rval));
        rval += log_x;
        return rval;
    } else {
        return log_y;
    }
}

/* adds two numbers represented in the logarithmic domain */
double fast_logadd(double log_x, double log_y) {

    if (DisableFastMath) return safe_logadd(log_x, log_y);

    if (log_x < log_y) {
        return fast_jacoblog(log_y - log_x) + log_x;
    } else {
        return fast_jacoblog(log_x - log_y) + log_y;
    }
}
