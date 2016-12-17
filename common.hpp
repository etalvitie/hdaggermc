#ifndef __COMMON_HPP__
#define __COMMON_HPP__

/******************************
      Author: Joel Veness
        Date: 2011
******************************/

#include <vector>
#include <cmath>

// boost includes
#include <boost/cstdint.hpp>
#include <boost/dynamic_bitset.hpp>

// fixed size integer types
using boost::int8_t;
using boost::uint8_t;
using boost::int16_t;
using boost::uint16_t;
using boost::uint32_t;
using boost::int32_t;
using boost::uint64_t;
using boost::int64_t;
using boost::intmax_t;
using boost::uintmax_t;

// define a bit type
typedef uint8_t bit_t;

// stores symbol occurrence counts
typedef float count_t;

// holds context weights
typedef double weight_t;

// describe a binary context
typedef std::vector<bit_t> context_t;

// describe a binary history
typedef boost::dynamic_bitset<> history_t;
extern void zeroFill(history_t &h, size_t n);


/* given log(x) and log(y), compute log(x+y). uses the following identity:
   log(x + y) = log(x) + log(1 + y/x) = log(x) + log(1+exp(log(y)-log(x)))*/
inline double logAdd(double log_x, double log_y) {

    // ensure log_y >= log_x, can save some expensive log/exp calls
    if (log_x > log_y) {
        double t = log_x; log_x = log_y; log_y = t;
    }

    double rval = log_y - log_x;

    // only replace log(1+exp(log(y)-log(x))) with log(y)-log(x)
    // if the the difference is small enough to be meaningful
    if (rval < 100.0) rval = std::log(1.0 + std::exp(rval));

    rval += log_x;
    return rval;
}


// compressor interface
class Compressor {

    public:

        virtual ~Compressor() {}

        // the probability of seeing a particular symbol next
        virtual double prob(bit_t b) = 0;

        // the logarithm of the probability of all processed experience
        virtual double logBlockProbability() const = 0;

        // process a new piece of sensory experience
        virtual void update(bit_t b) = 0;

        // file extension
        virtual const char *fileExtension() const = 0;
};


#endif // __COMMON_HPP__

