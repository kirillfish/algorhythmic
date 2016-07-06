#coding: utf-8
from bitmap import BitMap
import numpy as np
from collections import OrderedDict, defaultdict, Counter
from itertools import izip_longest, chain
import re


__author__ = "kurtosis"


def _find_euclidean_rhythm(for_one='1', for_zero='0', ones=5, zeros=15, last=False):
    res = [np.array(tup, dtype='object') for tup in izip_longest([for_one]*ones, [for_zero]*zeros)]
    res = [arr[arr != np.array(None)] for arr in res]
    res = [list(chain(*tup)) for tup in res]

    patterns, indices = np.unique([tuple(arr) for arr in res], return_inverse=True)
    counts = {patterns[k]:v for k,v in Counter(indices).items()}
    if last:
        return str(''.join(chain(*chain(*res))))

    last = (min(counts.values()) <=1 or len(set(counts.values()))==1)
    return _find_euclidean_rhythm(for_one=''.join(chain(*counts.keys()[0])),
                      for_zero=''.join(chain(*counts.keys()[1])),
                      ones=counts.values()[0], zeros=counts.values()[1],
                                 last=last)

def _shifted_family(r):
    family = set()
    for pos in xrange(len(r)):
        family.add(r[pos:] + r[:pos])
    return family

def if_euclidean_factor(linear, rhythm_bitmap, multiplier=50):
    r = rhythm_bitmap.tostring()
    strokes = Counter(r)
    if len(r) % strokes['1']:
        return 1

    euclidean_family = _shifted_family(
        _find_euclidean_rhythm(ones=strokes['1'], zeros=strokes['0'])
    )
    return multiplier if r in euclidean_family else 1

def one_zero_zero_factor(linear, rhythm_bitmap, multiplier=100):
    pattern = '1001001001'
    stretched = '0'.join(pattern)
    r = rhythm_bitmap.tostring() * 2
    return multiplier if pattern in r or stretched in r else 1

def one_one_one_factor(linear, rhythm_bitmap, ones=(3,4), multipliers=(40,150), negative=False):
    multipliers_dict = dict(zip(ones, multipliers))
    r = rhythm_bitmap.tostring() * 2
    cumulative_multiplier = 0
    for num in sorted(ones, reverse=True):
        pattern = re.compile('1'*num)
        inclusion_number = len(pattern.findall(r))
        if inclusion_number == 0:
            continue
        inclusion_fraction = inclusion_number * num * 1. / Counter(r)['1'] / 2
        cumulative_multiplier += multipliers_dict[num] * inclusion_fraction
    if negative:
        return 1 if cumulative_multiplier == 0 else min(1, 1. / cumulative_multiplier)
    else:
        return max(1, cumulative_multiplier)

def kicky_beatie_factor(linear, rhythm_bitmap, level=2, multiplier=100000):
    r = rhythm_bitmap.tostring()
    kickies = map(lambda x: x[0], filter(lambda x: x[1] >= 2**-level, linear.current_anga.strengths.items()))
    all_in_place = all([r[i]=='1' for i in kickies])
    return multiplier if all_in_place else 1

def fixed_meter_factor(linear, rhythm_bitmap, jati, multiplier=10000):
    assert rhythm_bitmap.size() % jati == 0, 'The rhythm length must be divisible into jati'
    desired_onsets = np.linspace(0, rhythm_bitmap.size(), jati + 1)[:-1]
    desired_onsets = [int(rhythm_bitmap.size() - 1 - i) for i in desired_onsets]
    all_desired_onsets_present = all([rhythm_bitmap.test(i) for i in desired_onsets])
    return multiplier if all_desired_onsets_present else 1

def pattern_factor(linear, rhythm_bitmap, pattern_bitmap, multiplier=10000):
    desired_onsets = pattern_bitmap.nonzero()
    all_desired_onsets_present = all([rhythm_bitmap.test(i) for i in desired_onsets])
    return multiplier if all_desired_onsets_present else 1

def familiarity_clave_factor(linear, rhythm_bitmap):
    pass

def familiarity_arab_factor(linear, rhythm_bitmap):
    pass

def familiarity_carnatic_factor(linear, rhythm_bitmap):
    pass

def familiarity_flamenco_factor(linear, rhythm_bitmap):
    pass

def counterintuitive_shift_factor(linear, rhythm_bitmap, gap=25, penalty=1000):
    # TODO: consider moving this into main module
    """
    One of the basic components of rhythm complexity.
    One should be cautious with this. A rhythm may sound good if the 1st puls position is quite non-intuitive, BUT
    the rhythm sounds really crappy when there is a shift of it that is very-very dull.
    What is called "irregularity" in markov_rhythmics.py, is really
    only one of the irregularity components -- namely, the syncopation.
    There are at least 2 other significant components:
    -- counterintuitive shift as compared to the least-syncopated shift of the same rhythmic pattern
    -- microtiming (to be done)
    """
    shifts = _shifted_family(rhythm_bitmap.tostring())
    irregularity_as_is = linear.rhythm_irregularity(rhythm_bitmap)
    min_shifted_irregularity = min([linear.rhythm_irregularity(BitMap.fromstring(r)) for r in shifts])
    #return [linear.rhythm_irregularity(BitMap.fromstring(r)) for r in shifts], irregularity_as_is
    # TODO include also promotion of the most irregular rhythms among all the shifts if the minimum irreg shift is not too dull.
    return 1./penalty if irregularity_as_is > min_shifted_irregularity * gap else 1
