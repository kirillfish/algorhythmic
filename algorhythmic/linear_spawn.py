#coding: utf-8
from markov_rhythmics import Linear, MultilinearGeneric
from optional_params import (if_euclidean_factor,
                             one_one_one_factor,
                             one_zero_zero_factor,
                             kicky_beat_factor,
                             counterintuitive_shift_factor,
                             multi_no_overlap_factor,
                             fixed_meter_factor,
                             pattern_factor,
                             subpattern_factor)


__author__ = "kurtosis"


class LinearKick(Linear):

    def __init__(self, log_name='Linear Kick', kickiness_level=2, **kwargs):
        super(LinearKick, self).__init__(log_name=log_name, **kwargs)
        self.kickiness_level = kickiness_level

    def kicky_beatie_factor_wrapper(self, rhythm_bitmap, multiplier=3000):
        return kicky_beat_factor(self, rhythm_bitmap, self.kickiness_level, multiplier)

    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearKick, self).sampling_probability(rhythm_bitmap)
        self.log_debug('Inheritance:kickiness:%.6f' % self.kicky_beatie_factor_wrapper(rhythm_bitmap))
        proba *= (self.kicky_beatie_factor_wrapper(rhythm_bitmap))
        return proba


class LinearPhunk(Linear):
    pass


class LinearCarnatic(Linear):
    pass


class LinearClave(Linear):

    def __init__(self, log_name='Linear Clave', **kwargs):
        super(LinearClave, self).__init__(log_name=log_name, **kwargs)

    def if_euclidean_factor_wrapper(self, rhythm_bitmap, multiplier=2000):
        return if_euclidean_factor(self, rhythm_bitmap, multiplier)

    def one_one_one_factor_wrapper(self, rhythm_bitmap, ones=(2,3,4), multipliers=(10**4, 10**5, 10**6)):
        return one_one_one_factor(self, rhythm_bitmap, ones, multipliers, negative=True)

    def one_zero_zero_factor_wrapper(self, rhythm_bitmap, multiplier=50):
        return one_zero_zero_factor(self, rhythm_bitmap, multiplier)

    def counterintuitive_shift_factor_wrapper(self, rhythm_bitmap, gap=10, penalty=1000):
        return counterintuitive_shift_factor(self, rhythm_bitmap,
                                             gap=gap, penalty=penalty)

    # noinspection PyStringFormat
    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearClave, self).sampling_probability(rhythm_bitmap)

        multipliers = [self.if_euclidean_factor_wrapper(rhythm_bitmap),
                       self.one_one_one_factor_wrapper(rhythm_bitmap),
                       self.one_zero_zero_factor_wrapper(rhythm_bitmap),
                       self.counterintuitive_shift_factor_wrapper(rhythm_bitmap)]
        proba *= reduce(lambda a, b: a * b, multipliers)
        self.log_debug(
            'Inheritance:euclid:%.4f\tones:%.4f\t100:%.4f\tdullshift:%.4f\tfinal proba: %.6f\n' %
            tuple(multipliers + [proba])
        )
        return proba


class LinearHat(Linear):

    def __init__(self, log_name='Linear Hat', open=False, **kwargs):
        super(LinearHat, self).__init__(log_name=log_name, **kwargs)
        self.open = open

    def if_euclidean_factor_wrapper(self, rhythm_bitmap, multiplier=0.01):
        return if_euclidean_factor(self, rhythm_bitmap, multiplier)

    def one_one_one_factor_wrapper(self, rhythm_bitmap, ones=(3,4,5,6), multipliers=(200, 500, 1000, 1000)):
        return one_one_one_factor(self, rhythm_bitmap, ones, multipliers)

    def subpattern_factor_wrapper(self, rhythm_bitmap, subpatterns=('010',), multipliers=(10.**2,)):
        return subpattern_factor(self, rhythm_bitmap, subpatterns=subpatterns,
                                 multipliers=multipliers, negative=True)

    def counterintuitive_shift_factor_wrapper(self, rhythm_bitmap, gap=10,
                                              penalty=1000):
        return counterintuitive_shift_factor(self, rhythm_bitmap,
                                             gap=gap, penalty=penalty)

    def sampling_probability(self, rhythm_bitmap):
        proba = super(LinearHat, self).sampling_probability(rhythm_bitmap)
        multipliers = [self.if_euclidean_factor_wrapper(rhythm_bitmap),
                       self.one_one_one_factor_wrapper(rhythm_bitmap),
                       self.counterintuitive_shift_factor_wrapper(rhythm_bitmap)]
        to_log = 'Inheritance:euclid:%.4f\tones:%.4f\tdullshift:%.4f'
        if not self.open:
            multipliers.append(self.subpattern_factor_wrapper(rhythm_bitmap))
            to_log += '\t010:%.6f'
        to_log += '\tfinal proba:%.6f\n'
        proba *= reduce(lambda a, b: a * b, multipliers)

        to_log = (to_log % tuple(multipliers + [proba]))
        self.log_debug(to_log)
        return proba


class MultilinearPhunk(MultilinearGeneric):

    def __init__(self, *ordered_linears, **kwargs):
        self.kwargs = kwargs
        kwargs['overlap_penalty'] = kwargs.get('overlap_penalty', 10000)
        kwargs['hard'] = kwargs.get('hard', False)
        super(MultilinearPhunk, self).__init__(*ordered_linears, **kwargs)

    # noinspection PyUnresolvedReferences
    def all_dependencies(self, rhythm_bitmap):
        return multi_no_overlap_factor(self,
                                       rhythm_bitmap,
                                       penalty=self.overlap_penalty,
                                       hard=self.hard)


class MultilinearCarnatic(MultilinearGeneric):
    pass


class MultilinearClave(MultilinearGeneric):
    pass