

#ifndef SPEEDPPR_MYRANDOM_H
#define SPEEDPPR_MYRANDOM_H


#include <random>
#include <iostream>
#include <ctime>

#include "SFMT-src-1.5.1/SFMT.h"


class SFMT64 {
private:
    static sfmt_t sfmt;
public:

    /**
    *
    * This function must be called before using other functions.
    * The function gen_rand64 should not be called after gen_rand32,
    * unless an initialization is again executed.
    */
    static void initialize() {
//        sfmt_init_gen_rand(&sfmt, clock());
        sfmt_init_gen_rand(&sfmt, 1234);
    }



    /**
     * This function generates and returns 64-bit pseudorandom number.
     * init_gen_rand or init_by_array must be called before this function.
     * The function gen_rand64 should not be called after gen_rand32,
     * unless an initialization is again executed.
     * @param sfmt SFMT internal state
     * @return 64-bit pseudorandom number
     */
//    inline static uint64_t sfmt_genrand_uint64(sfmt_t * sfmt)

    /**
    * generates a random number on [0,1) with 53-bit resolution
    * @param sfmt SFMT internal state
    * @return double on [0,1) with 53-bit resolution
    */
//    inline static double sfmt_genrand_res53(sfmt_t * sfmt)
//    {
//        return sfmt_to_res53(sfmt_genrand_uint64(sfmt));
//    }

    static inline double uniform_real() {
        return sfmt_genrand_res53(&sfmt);
    }

    /**
    *
    * @param _end
    * @return return a unifrom integer between [0, _end)
    */
    static inline uint32_t uniform_int(const uint32_t &_end) {
        return sfmt_genrand_uint64(&sfmt) % _end;
    }

    /**
     * @return a bias coin flip that comes up head with probability alpha
     */
    static inline bool bias_coin_is_head(const double &_prob) {
        return sfmt_genrand_res53(&sfmt) <= _prob;
    }

    static inline bool bias_coin_is_tail(const double &_prob) {
        return sfmt_genrand_res53(&sfmt) > _prob;
    }

};

class MinimalStandardGenerator {
    //Will be used to obtain a seed for the random number engine
    static std::random_device dev;

    /**The generator has a single value as state, which is modified by its transition
     * algorithm on each advance like x is modified in the following piece of code:
     * x = x * 48271 % 2147483647
     */
    static std::minstd_rand gen;

//    MT19937
public:

    static inline uint32_t rand_int() {
        return gen();
    }

    /**
     *
     * @return a real number between (0, 1)
     */
    static inline double uniform_real() {
        return gen() * (1.0 / (std::minstd_rand::max() + 1.0));
    }

    /**
    *
    * @param _end
    * @return return a unifrom integer between [0, _end)
    */
    static inline uint32_t uniform_int(const uint32_t &_end) {
        return uniform_real() * _end;
    }

    /**
     * @return a bias coin flip that comes up head with probability alpha
     */
    static inline bool bias_coin_is_head(const double &_alpha) {
        return gen() <= _alpha * std::minstd_rand::max();
    }

    static inline bool bias_coin_is_tail(const double &_alpha) {
        return gen() > _alpha * std::minstd_rand::max();
    }
};


#endif //SPEEDPPR_MYRANDOM_H
