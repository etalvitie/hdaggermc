#include "icsilog.h"

#include <cmath>


/**
@brief This method computes the icsi_log. A fast approximation of the log() function with adjustable accuracy.

@param val Should be an IEEE 753 float with a value in the interval ]0,+inf[. A value smaller or equal zero results in undefined behaviour.
@param lookup_table Requires a float* pointing to the table created by fill_icsi_log_table.
@param  n The number of bits used from the mantissa (0<=n<=23). Higher n means higher accuracy but slower execution. We found that a good value for n is 14.
@return Approximation of the natural logarithm of val.
*/
void fill_icsi_log_table(const int n, float *lookup_table)
{
    float numlog;
    int incr,i,p;
    int *const exp_ptr = ((int*)&numlog);

    int x = 0x3F800000; /*set the exponent to 0 so numlog=1.0*/
        *exp_ptr = x;
    incr = 1 << (23-n); /*amount to increase the mantissa*/
    p = 1 << n;
    for(i=0;i<p;i++)
    {
        lookup_table[i] = (float) std::log(numlog) / std::log(2.0); /*save the log of the value*/
        x += incr;
        *exp_ptr = x; /*update the float value*/
    }
}


/** ICSIlog V 2.0. */
void fill_icsi_log_table2(const unsigned precision, float* const   pTable)
{
    /* step along table elements and x-axis positions
      (start with extra half increment, so the steps intersect at their midpoints.) */
    float oneToTwo = 1.0f + (1.0f / (float)( 1 <<(precision + 1) ));
    int i;
    for(i = 0;  i < (1 << precision);  ++i )
    {
        // make y-axis value for table element
        pTable[i] = logf(oneToTwo) / 0.69314718055995f;

        oneToTwo += 1.0f / (float)( 1 << precision );
    }
}

