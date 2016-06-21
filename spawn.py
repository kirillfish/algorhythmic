#coding: utf-8
from markov_rhythmics import Linear
from pulls_levers import (if_euclidean_factor,
                          one_one_one_factor,
                          one_zero_zero_factor,
                          kicky_beatie_factor)


__author__ = "kurtosis"


class LinearKick(Linear):

    def __init__(self, variability=0.1, density=0.5, irregularity=0.8,
                 possible_volume_bounds=(0.3, 1.), volume_serendipity=0.5,
                 volume_serendipity_tolerance=0.3, mean_volume=None,
                 mean_volume_tolerance=0.1, start='1010001010001000',
                 angas_per_avartam=4, log=None, log_level='debug', log_name='Linear Kick',
                 kickiness_level=2):

        super(LinearKick, self).__init__(variability=variability, density=density,
                     irregularity=irregularity,
                     possible_volume_bounds=possible_volume_bounds,
                     volume_serendipity=volume_serendipity,
                     volume_serendipity_tolerance=volume_serendipity_tolerance,
                     mean_volume=mean_volume,
                     mean_volume_tolerance=mean_volume_tolerance,
                     start=start,
                     angas_per_avartam=angas_per_avartam,
                     log=log, log_level=log_level, log_name=log_name)

        self.kickiness_level = kickiness_level

    def kicky_beatie_factor_wrapper(self, rhythm_bitmap, multiplier=1000):
        return kicky_beatie_factor(self, rhythm_bitmap, self.kickiness_level, multiplier)

    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearKick, self).sampling_probability(rhythm_bitmap)
        self.log_debug('Inheritance:kickiness:%.6f' % self.kicky_beatie_factor_wrapper(rhythm_bitmap))
        proba *= self.kicky_beatie_factor_wrapper(rhythm_bitmap)
        return proba


class LinearPhunk(Linear):
    pass


class LinearCarnatic(Linear):
    pass


class LinearClave(Linear):

    def __init__(self, variability=0.1, density=0.5, irregularity=0.8,
                 possible_volume_bounds=(0.3, 1.), volume_serendipity=0.5,
                 volume_serendipity_tolerance=0.3, mean_volume=None,
                 mean_volume_tolerance=0.1, start='0100100100010010',
                 angas_per_avartam=4, log=None, log_level='debug',
                 log_name='Linear Clave'):

        super(LinearClave, self).__init__(variability=variability, density=density,
                                     irregularity=irregularity,
                                     possible_volume_bounds=possible_volume_bounds,
                                     volume_serendipity=volume_serendipity,
                                     volume_serendipity_tolerance=volume_serendipity_tolerance,
                                     mean_volume=mean_volume,
                                     mean_volume_tolerance=mean_volume_tolerance,
                                     start=start,
                                     angas_per_avartam=angas_per_avartam,
                                     log=log, log_level=log_level,
                                     log_name=log_name)

    def if_euclidean_factor_wrapper(self, rhythm_bitmap, multiplier=200):
        return if_euclidean_factor(self, rhythm_bitmap, multiplier)

    def one_one_one_factor_wrapper(self, rhythm_bitmap, ones=(2,3,4), multipliers=(0.01, 0.001, 0.0001)):
        return one_one_one_factor(self, rhythm_bitmap, ones, multipliers)

    def one_zero_zero_factor_wrapper(self, rhythm_bitmap, multiplier=50):
        return one_zero_zero_factor(self, rhythm_bitmap, multiplier)

    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearClave, self).sampling_probability(rhythm_bitmap)
        self.log_debug(
            'Inheritance:euclid:%.4f\tones:%.4f\t100:%.4f\n' %
            (self.if_euclidean_factor_wrapper(rhythm_bitmap),
             self.one_one_one_factor_wrapper(rhythm_bitmap),
             self.one_zero_zero_factor_wrapper(rhythm_bitmap))
        )
        proba *= (self.if_euclidean_factor_wrapper(rhythm_bitmap) *
                  self.one_one_one_factor_wrapper(rhythm_bitmap) *
                  self.one_zero_zero_factor_wrapper(rhythm_bitmap))
        return proba


class LinearHat(Linear):

    def __init__(self, variability=0.1, density=0.5, irregularity=0.8,
                 possible_volume_bounds=(0.3, 1.), volume_serendipity=0.5,
                 volume_serendipity_tolerance=0.3, mean_volume=None,
                 mean_volume_tolerance=0.1, start='0100100100010010',
                 angas_per_avartam=4, log=None, log_level='debug',
                 log_name='Linear Hat'):

        super(LinearHat, self).__init__(variability=variability, density=density,
                                     irregularity=irregularity,
                                     possible_volume_bounds=possible_volume_bounds,
                                     volume_serendipity=volume_serendipity,
                                     volume_serendipity_tolerance=volume_serendipity_tolerance,
                                     mean_volume=mean_volume,
                                     mean_volume_tolerance=mean_volume_tolerance,
                                     start=start,
                                     angas_per_avartam=angas_per_avartam,
                                     log=log, log_level=log_level,
                                     log_name=log_name)

    def if_euclidean_factor_wrapper(self, rhythm_bitmap, multiplier=0.01):
        return if_euclidean_factor(self, rhythm_bitmap, multiplier)

    def one_one_one_factor_wrapper(self, rhythm_bitmap, ones=(3,4,5), multipliers=(20, 100, 150)):
        return one_one_one_factor(self, rhythm_bitmap, ones, multipliers)

    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearHat, self).sampling_probability(rhythm_bitmap)
        self.log_debug(
            'Inheritance:euclid:%.4f\tones:%.4f\n' %
            (self.if_euclidean_factor_wrapper(rhythm_bitmap),
             self.one_one_one_factor_wrapper(rhythm_bitmap))
        )
        proba *= (self.if_euclidean_factor_wrapper(rhythm_bitmap) *
                  self.one_one_one_factor_wrapper(rhythm_bitmap))
        return proba


class Multilinear(Linear):
    pass


class MultilinearPhunk(Multilinear):
    pass


class MultilinearCarnatic(Multilinear):
    pass


class MultilinearClave(Multilinear):
    pass