#coding: utf-8
from bitmap import BitMap
from copy import deepcopy
import numpy as np
from scipy import stats
from scipy.optimize import *
from random import shuffle
from functools import partial
from collections import OrderedDict, defaultdict, Counter
from itertools import izip
from tqdm import tqdm
import logging
import midi

import sys
sys.path.insert(1, '/Users/k.rybachuk/rhythm/midiutil/src/')

from midiutil.MidiFile import MIDIFile


__author__ = "kurtosis"


JATI_TERMS = {'tishra':3,
              'chatusra':4,
              'khanda':5,
              'mishra':7,
              'sankeerna':9}


def if_logger(func):
    def func_wrapper(self, msg):
        if self.logger is not None:
            return func(self, msg)
    return func_wrapper

print if_logger

def if_log(func):
    def func_wrapper(self, *args, **kwargs):
        if self.log is not None:
            return func(self, *args, **kwargs)
    return func_wrapper


class Nadai():
    pass


class Akshara():
    pass


class Anga(object):
    """
    Abstract "bar" structure for univocal track.
    Can be squeezed or stretched both absolutely in time
    and relatively with respect to the meter (in polyrhythms and rhythmic modulation).
    """
    def __init__(self, r='0100100100010010', r_volume=None, jati=4, gati='chatusra',
                 logger=None):
        """
        :param r: rhythm itself
        :param jati: "time"
        :param gati: the no. of pulses inside a single beat
        """
        self.logger = logger
        if r is None:
            self.pulse = jati * JATI_TERMS[gati]
            self.jati = jati
        else:
            self.pulse = len(r)
            assert self.pulse % JATI_TERMS[gati] == 0, "the rhythm's length is not divisible by gati"
            self.jati = self.pulse / JATI_TERMS[gati]
        self.r = r
        self.r_bitmap = BitMap.fromstring(self.r)
        self.nonzero = set([(self.r_bitmap.size() - 1 - index) for index in self.r_bitmap.nonzero()])
        self.log_debug("nonzero: %s" % self.nonzero)
        self.strengths = self._compute_strengths()

        if r_volume is None:
            self.r_volume = {pos: 0.7 for pos in self.nonzero}
        else:
            self.r_volume = r_volume

        self.notes_to_update_in_maps = self.nonzero.difference(set(self.r_volume.keys()))
        #print self.notes_to_update_in_maps
        self.notes_to_delete_from_maps = set(self.r_volume.keys()).difference(self.nonzero)
        for pos in self.notes_to_delete_from_maps:
            del self.r_volume[pos]

    @if_logger
    def log_debug(self, msg):
        return self.logger.debug(msg)

    @if_logger
    def log_info(self, msg):
        return self.logger.info(msg)

    @if_logger
    def log_warning(self, msg):
        return self.logger.warning(msg)

    @if_logger
    def log_error(self, msg):
        return self.logger.error(msg)

    def _compute_strengths(self):
        length = self.r_bitmap.size()
        accents = {i:[] for i in xrange(length)}
        for gati in xrange(2,length+1):
            if length % gati == 0:
                for nadai in xrange(0, length, gati):
                    accents[nadai].append(gati)
        strengths = {}
        max_on = max([len(on) for on in accents.values()])
        for nadai in accents:
            strengths[nadai] = 1./2**(max_on-len(accents[nadai]))
        return strengths

    def _flip_note(self, pos):
        # pos indexing is straight
        # bitmap indexing is reverse
        res = deepcopy(self.r_bitmap)
        res.flip(res.size()-1 - pos)
        return res

    def _shift_note(self, pos, dir='right', step=1):
        res = deepcopy(self.r_bitmap)
        if dir == 'right':
            res.set((res.size() - 1 - (pos + step)) % res.size())
        elif dir == 'left':
            res.set((res.size() - 1 - (pos - step)) % res.size())
        else:
            raise ValueError("direction parameter should take value either 'right' or 'left'")
        res.reset(res.size() - 1 - pos)
        return res

    def find_adjacent(self, admissible_steps=(1,)):
        flipped = []
        for pos in xrange(self.r_bitmap.size()):
            flipped.append(self._flip_note(pos))
        flipped = set(flipped)
        self.log_debug("flipped adjacent: %s" % len(flipped))

        shifted = []
        nonzero = [self.r_bitmap.size() - 1 - reverse_pos for reverse_pos in self.r_bitmap.nonzero()]
        for step in admissible_steps:
            for pos in nonzero:
                shifted.append(self._shift_note(pos, dir='right', step=step))
                shifted.append(self._shift_note(pos, dir='left', step=step))
        shifted = set(shifted)
        self.log_debug("shifted adjacent: %s" % len(shifted))

        adjacent = flipped.union(shifted)
        self.log_debug("shifted+flipped and the total adjacent: %s, %s" % (len(shifted) + len(flipped), len(adjacent)))
        return adjacent


class Avartam(object):

    def __init__(self, *angas):
        self.r = ''.join(anga.r for anga in angas)
        self.r_volume = self._init_volumes(*angas)

    def _init_volumes(self, *angas):
        avartam_volume = deepcopy(angas[0].r_volume)
        if len(angas) > 1:
            base = len(angas[0].r)
            for anga in angas[1:]:
                avartam_volume.update({k+base:v for k,v in anga.r_volume.items()})
                base += len(anga.r)
        return avartam_volume

    def add_tihai(self):
        pass

    def add_khali(self):
        pass

    def add_sam(self):
        pass

    def add_random_microtiming(self):
        pass


class Linear(object):

    def __init__(self, variability=0.1, density=0.5, irregularity=0.8,
                 possible_volume_bounds=(0.3, 1.),
                 volume_serendipity=0.5, volume_serendipity_tolerance=0.3,
                 mean_volume=None, mean_volume_tolerance=0.1,
                 start='0100100100010010',
                 angas_per_avartam = 4,
                 log=None, log_level='debug', log_name='Linear Logger'):
        self.logger=None
        self.log = log
        if self.log is not None:
            self.logger = self._configure_logger(log_level, log_name)

        self.start = start
        self.shift_steps = self._suggest_admissible_shift_steps()

        self.current_anga = Anga(self.start, logger=self.logger)
        self.previous_anga = Anga(self.start, logger=self.logger)

        self.angas_per_avartam = angas_per_avartam
        self.current_avartam = self.construct_avartam()

        self.poisson_lambda = self._convert_variability(variability)        # for variability

        self.triang_loc = density - 0.1     # for density
        self.triang_scale = 0.2

        self.density = density
        self.volume_tuner = VolumeTuner(self.current_anga,
                                        mean_volume=mean_volume, mean_volume_tolerance=mean_volume_tolerance,
                                        serendipity=volume_serendipity,
                                        serendipity_tolerance=volume_serendipity_tolerance,
                                        possible_volume_bounds=possible_volume_bounds,
                                        logger=self.logger)

        self.log_info("poisson lambda for variability: %s" % self.poisson_lambda)

        self.irregularity = irregularity
        self.sorted_irrs = None

    @if_logger
    def log_debug(self, msg):
        return self.logger.debug(msg)

    @if_logger
    def log_info(self, msg):
        return self.logger.info(msg)

    @if_logger
    def log_warning(self, msg):
        return self.logger.warning(msg)

    @if_logger
    def log_error(self, msg):
        return self.logger.error(msg)

    @if_log
    def _configure_logger(self, log_level='debug', log_name='Linear Logger'):
        LOGGER_FORMAT = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(name)s/%(module)s:%(lineno)d]: %(message)s"
        LOGGER_DATEFMT = "%Y-%m-%d %H:%M:%S"
        LOGFILE = self.log

        if log_level == 'debug':
            lvl = logging.DEBUG
        elif log_level == 'info':
            lvl = logging.INFO
        elif log_level == 'warning':
            lvl = logging.WARNING
        elif log_level == 'error':
            lvl = logging.ERROR
        else:
            raise ValueError("Log level must be of these: 'debug', 'info', 'warning', 'error'")
        logging.basicConfig(format=LOGGER_FORMAT,
                            datefmt=LOGGER_DATEFMT,
                            level=lvl)
        formatter = logging.Formatter(LOGGER_FORMAT, datefmt=LOGGER_DATEFMT)

        file_handler = logging.FileHandler(LOGFILE)
        file_handler.setFormatter(formatter)

        logger = logging.getLogger(log_name)
        logger.setLevel(lvl)
        logger.addHandler(file_handler)
        return logger

    def _suggest_admissible_shift_steps(self):
        return range(1, max(len(self.start) / 12, 1) + 1)

    def _convert_variability(self, variability):
        return len(self.start) * variability

    def generate_walk_length(self):
        # The number of random walk steps for making the next variation.
        # Depends on the predefined variability parameter of the track.
        return np.random.poisson(self.poisson_lambda)

    def _find_resolution_pulses(self, rhythm_bitmap, pos):
        strengths = self.current_anga.strengths
        pos_minus, dist_minus = None, None
        pos_plus, dist_plus = None, None
        for minus in xrange(pos-1, -1, -1):
            if rhythm_bitmap.test(rhythm_bitmap.size() - 1 - minus):
                break
            if strengths[pos] < strengths[minus]:
                pos_minus = minus
                dist_minus = pos - minus
                break
        for plus in xrange(pos+1, rhythm_bitmap.size()):
            if rhythm_bitmap.test(rhythm_bitmap.size() - 1 - plus):
                break
            if strengths[pos] < strengths[plus]:
                pos_plus = plus
                dist_plus = plus - pos
                break
        return pos_minus, pos_plus, dist_minus, dist_plus

    def note_irregularity(self, rhythm_bitmap, pos):
        strengths = self.current_anga.strengths
        strength_i = strengths[pos]
        pos_minus, pos_plus, dist_minus, dist_plus = self._find_resolution_pulses(rhythm_bitmap, pos)

        if pos_minus is None:
            attraction_minus = -1
        else:
            attraction_minus = strengths[pos_minus] * 0.5 / dist_minus
        if pos_plus is None:
            attraction_plus = -1
        else:
            attraction_plus = strengths[pos_plus] * 1. / dist_plus
        if attraction_minus == -1 and attraction_plus == -1:
            return 0

        def irr(pos_plusminus, attr):
            return strengths[pos_plusminus]**1.5 * 1./(strength_i)**3 / attr

        where = 'minus'
        if attraction_minus > attraction_plus:
            irregularity = irr(pos_minus, attraction_minus)
        elif attraction_minus < attraction_plus:
            where='plus'
            irregularity = irr(pos_plus, attraction_plus)
        else:
            irregularity = max(irr(pos_minus, attraction_minus),
                              irr(pos_plus, attraction_plus))
        return irregularity

    def rhythm_irregularity(self, rhythm_bitmap):
        irr = 1
        for pos in [rhythm_bitmap.size()-1-ix for ix in rhythm_bitmap.nonzero()]: #xrange(rhythm_bitmap.size()):
            irr += self.note_irregularity(rhythm_bitmap, pos)
        irr *= (len(rhythm_bitmap.nonzero())**0.2/rhythm_bitmap.size()**1.2) # previously: 0.4 and 2
        return irr

    def irregularity_factor(self, rhythm_bitmap):
        normalized_rank = self.sorted_irrs.keys().index(rhythm_bitmap) * 1. / len(self.sorted_irrs)
        proba = stats.t.pdf(normalized_rank, df=1, loc=self.irregularity, scale=0.12)
        return proba

    def density_factor(self, rhythm_bitmap):
        rhythm_density = len(rhythm_bitmap.nonzero()) * 1. / rhythm_bitmap.size()
        #proba = stats.beta.pdf(rhythm_density, self.alpha_shape, self.beta_shape)
        proba = stats.triang.pdf(rhythm_density, 0.5, loc=self.triang_loc, scale=self.triang_scale)
        return proba

    def debug_factor(self, rhythm_bitmap):
        return float(len(rhythm_bitmap.nonzero()) > 0) * float(rhythm_bitmap.tostring() != self.previous_anga.r)

    def sampling_probability(self, rhythm_bitmap):
        """
        :param rhythm_bitmap: a candidate to modify current anga
        :return: A probability (up to normalization constant) of the candidate to be chosen
                 in the next step of random walk. The probability factorizes over several parameters
                 (i.e. is just a product of distributions)
        """
        proba = (self.irregularity_factor(rhythm_bitmap)
                 * self.density_factor(rhythm_bitmap) ** 2
                 * self.debug_factor(rhythm_bitmap))
        self.log_debug("%s:\tirreg:%.6f\tdens:%.6f\tsampling probability:%.4f" % (rhythm_bitmap.tostring(),
                                           self.irregularity_factor(rhythm_bitmap),
                                           self.density_factor(rhythm_bitmap),
                                           proba))
        return proba

    def _rank_by_irregularity(self, adjacent):
        irrs = {rhythm_bitmap: self.rhythm_irregularity(rhythm_bitmap) for rhythm_bitmap in adjacent}
        sorted_irrs = OrderedDict(sorted(irrs.items(), key=lambda x: x[1]))
        return sorted_irrs

    def make_elementary_step(self):
        self.log_info("making elementary step")
        adjacent = list(self.current_anga.find_adjacent(admissible_steps=self.shift_steps))
        self.sorted_irrs = self._rank_by_irregularity(adjacent)
        self.log_debug("sorted irregularities:")
        for item in self.sorted_irrs.items():
            self.log_debug("%s\t%s" % (item[0].tostring(), item[1]))

        adj_probas = []
        for rhythm_bitmap in adjacent:
            adj_probas.append(self.sampling_probability(rhythm_bitmap))
        adj_probas = np.array(adj_probas) / sum(adj_probas)

        choice = np.random.multinomial(1, adj_probas, size=1)
        choice_index = choice.nonzero()[1]

        self.previous_anga = self.current_anga
        self.current_anga = Anga(adjacent[choice_index].tostring(),
                                 r_volume=self.current_anga.r_volume,
                                 logger=self.logger)
        assert self.current_anga.r != self.previous_anga.r
        self.log_info("the choice: %s" % self.current_anga.r)

    def construct_avartam(self):
        return Avartam(*([self.current_anga]*self.angas_per_avartam))

    def make_variation(self):
        """
        Determine how much steps of random walk to take before stopping to make it new variation
        :return: new avartam
        """
        random_walk_length = self.generate_walk_length()
        self.log_info("random_walk_length: %s" % random_walk_length)
        for i in xrange(random_walk_length):
            self.make_elementary_step()
        self.log_info("the final choice: %s" % self.current_anga.r)

        self.log_info("volume tuning ...")
        self.volume_tuner.current_anga = self.current_anga # TODO: make it nicer
        self.volume_tuner.tune_volume_batch()
        self.current_anga = self.volume_tuner.current_anga
        return self.construct_avartam()

    def test_irregularity_common_sense(self):
        self.log_debug("Testing irregularity function for common sense...")
        r16 = [
        '1001001000101000',
        '1001001000100100',
        '0100100100010010',
        '0100101001010010',
        '0101001010010100',
        '0111001110011100',
        '1010010101001010',
        '0101010010101010',
        '0101001010100101',
        '0100000000000100',
        '0100001010000100',
        '1000100010001000',
        '0010001100100011',
        '0010001000100010',
        '0101010101010101'
        ]

        r16_twice = [r*2 for r in r16]
        r16_stretch = ['0'.join(r)+'0' for r in r16]

        def irr(r):
            return self.rhythm_irregularity(BitMap.fromstring(r))
        def test_less(collection, i0, i1):
            return collection[i0] < collection[i1]

        irr16 = map(irr, r16)
        irr16_twice = map(irr, r16_twice)
        irr16_stretch = map(irr, r16_stretch)
        self.log_debug("Rhythm\tIrr-as is\tIrr-taken twice\tIrr-stretch by 2")
        for i, r in enumerate(r16):
            self.log_debug('\t'.join([r, str(irr16[i]), str(irr16_twice[i]), str(irr16_stretch[i])]))

        indices_less16 = [(0,1),
                    (1,2),
                    (1,3),
                    (4,3),
                    (5,3),
                    (6,7),
                    (7,8),
                    (10,9),
                    (10,8),
                    (11,0),
                    (12,9),
                    (12,8),
                    (13,14),
                    (14,3)]
        tests16 = [test_less(irr16, i0, i1) for i0, i1 in indices_less16]
        tests16_twice = [test_less(irr16_twice, i0, i1) for i0, i1 in indices_less16]
        tests16_stretch = [test_less(irr16_stretch, i0, i1) for i0, i1 in indices_less16]

        for i, item in enumerate(indices_less16):
            self.log_debug('%s < %s\t%s' % (r16[item[0]], r16[item[1]], tests16[i]))
            self.log_debug('%s < %s\t%s' % (r16_twice[item[0]], r16_twice[item[1]], tests16_twice[i]))
            self.log_debug('%s < %s\t%s' % (r16_stretch[item[0]], r16_stretch[item[1]], tests16_stretch[i]))

        passed = sum(tests16) + sum(tests16_twice) + sum(tests16_stretch)
        total = len(tests16) + len(tests16_twice) + len(tests16_stretch)
        self.log_info("TESTS PASSED: %s/%s" % (passed, total))

class VolumeTuner(object):

    def __init__(self, initial_anga, possible_volume_bounds=(0.3, 1.),
                 serendipity=0.5, serendipity_tolerance=0.3,
                 mean_volume=None, mean_volume_tolerance=0.05,
                 logger=None):
        """
        Here should be arguments that make sense whatever the handles you are going to use. 
        """
        self.logger = logger
        self.current_anga = initial_anga
        self.effective_size=10
        self.possible_volume_bounds = possible_volume_bounds
        self.serendipity = serendipity
        self.serendipity_tolerance=serendipity_tolerance
        if mean_volume is None:
            self.mean_volume = np.mean(self.current_anga.r_volume.values())
        else:
            self.mean_volume = mean_volume
        self.mean_volume_tolerance = mean_volume_tolerance
        self.volume_window = (
            max(self.mean_volume-self.mean_volume_tolerance, self.possible_volume_bounds[0]),
            min(self.mean_volume+self.mean_volume_tolerance, self.possible_volume_bounds[1])
                             )
        self.log_info("volume window for modifying volume: %s, %s" % self.volume_window)
        #self.mean_npvi, self.std_npvi = self.simulate_npvi_distribution()

    @if_logger
    def log_debug(self, msg):
        return self.logger.debug(msg)

    @if_logger
    def log_info(self, msg):
        return self.logger.info(msg)

    @if_logger
    def log_warning(self, msg):
        return self.logger.warning(msg)

    @if_logger
    def log_error(self, msg):
        return self.logger.error(msg)

    def volume_irregularity_factor(self, rhythm_volume_map, rhythm_length):
        # TODO
        return 1

    def npvi(self, intervals, weights=None):
        if weights is None:
            weights = [1. for i in intervals]
        npvi = 0
        for i, interval in enumerate(intervals[:-1]):
            summand = (abs(interval - intervals[i+1])
                          / max(0.05, abs((interval + intervals[i+1])/2.)) * weights[i])
            npvi += summand
        summand = (abs(intervals[-1] - intervals[0])
                      / max(0.05, abs((intervals[-1] + intervals[0])/2.)) * weights[-1])
        npvi += summand
        npvi *=  100./(len(intervals))
        return npvi

    def volume_npvi(self, rhythm_volume_map):
        ordered_volume = sorted(rhythm_volume_map.items())
        ordered_volume_pairs = list(izip(ordered_volume[:-1], ordered_volume[1:]))
        volume_values = [i[1] for i in ordered_volume]  # volumes
        volume_weights = [j[0]-i[0] for i,j in ordered_volume_pairs]  # pulse intervals between the notes
        volume_weights.append(ordered_volume[0][0] - ordered_volume[-1][0] + self.current_anga.r_bitmap.size())
        volume_npvi = self.npvi(volume_values, volume_weights)
        return volume_npvi

    def simulate_npvi_distribution(self):
        npvis = []
        means = []
        self.log_info("notes to update in volume map: %s" % self.current_anga.notes_to_update_in_maps)
        for _ in tqdm(xrange(10000)):
            new_volume = deepcopy(self.current_anga.r_volume)
            for key in self.current_anga.notes_to_update_in_maps:
                new_volume[key] = np.random.uniform(0.3, 1)
            mean = np.mean(new_volume.values())
            if mean >= self.volume_window[0] and mean <= self.volume_window[1]:
                npvi = self.volume_npvi(new_volume)
                npvis.append(npvi)
                means.append(mean)
        self.log_warning("len(npvis)=%s" % len(npvis))
        mean_npvi = np.percentile(npvis, self.serendipity*100)
        std_npvi = self.serendipity_tolerance*(np.percentile(npvis, 75) - np.percentile(npvis, 25))
        self.log_info("mean npvi as it computed from simulations (%d percentile) is %.3f" % (self.serendipity*100, mean_npvi))
        self.log_info("standard deviation of normal distribution for npvi as interquartile range: %.3f" % std_npvi)
        return mean_npvi, std_npvi
    
    def volume_serendipity_factor(self, rhythm_volume_map):
        mean = np.mean(rhythm_volume_map.values())
        if mean <= self.volume_window[0] or mean >= self.volume_window[1]:
            return 0
        current_npvi = self.volume_npvi(rhythm_volume_map)
        return stats.norm.pdf(current_npvi, self.mean_npvi, self.std_npvi)

    def goodness_volume_function(self, update_values, update_indices):
        """
        :return: how good is the resulting configuration
        How good will be volume configuration if we modify the volume of a chosen notes.
        This is only to be used inside tune_volume method (as an objective function).
        """
        rvm = deepcopy(self.current_anga.r_volume)
        rvm.update(dict(zip(update_indices, update_values)))
        return -(self.volume_serendipity_factor(rvm)
                 * self.volume_irregularity_factor(rvm, self.current_anga.r_bitmap.size()))

    def tune_volume(self):
        """
        DEPRECATED
        """
        stroke_order = list(self.current_anga.notes_to_update_in_maps)
        shuffle(stroke_order)
        print "initial volumes: ", self.current_anga.r_volume
        print "new notes (to be updated): ", self.current_anga.notes_to_update_in_maps
        print "stroke_order: ", stroke_order
        for i in range(1):
            for stroke in stroke_order:
                x0 = self.mean_volume 
                print "initial value for stroke %s: %s" % (stroke, x0)
                bounds = (0.3, 1)
                objective = partial(self.goodness_volume_function, update_index=stroke)
                optimal_volume = minimize_scalar(objective, method='bounded', bounds=bounds)
                print "optimized: ", optimal_volume.x
                self.current_anga.r_volume.update({stroke: optimal_volume.x})
                print "so, now we have volumes: ", self.current_anga.r_volume
            print "final volumes: ", self.current_anga.r_volume

    def tune_volume_batch(self):
        """
         After obtaining the next variation via markov process, you need to make spiffier and groovier accents.
         This method optimizes the volumes given the notes in already assembled anga.
         """
        self.mean_npvi, self.std_npvi = self.simulate_npvi_distribution()
        update_indices = list(self.current_anga.notes_to_update_in_maps)
        self.log_debug("update_indices for volume: %s" % update_indices)
        if len(update_indices) == 0:
            return {}
        
        mybounds = MyBounds()
        objective = partial(self.goodness_volume_function, 
                            update_indices=update_indices)
        self.log_debug("optimization with basinhopping...")
        solution = basinhopping(objective, 
                                       x0=[self.mean_volume for i in self.current_anga.notes_to_update_in_maps],
                                       niter=200, niter_success=200, accept_test=mybounds)
        self.log_info("basinhopping solution: \n%s" % solution)
        for i,res in enumerate(solution['x']):
            solution['x'][i] = min(1, res)
        optimal_volumes = dict(zip(update_indices, solution['x']))
        self.log_info("optimal volumes: %s" % optimal_volumes)
        self.current_anga.r_volume.update(optimal_volumes)
        self.log_info("all volumes including modified ones: %s" % self.current_anga.r_volume)
        self.log_debug("optimal npvi: %s" % self.volume_npvi(self.current_anga.r_volume))
        return optimal_volumes

    
class MyBounds(object):
    def __init__(self, xmax=1, xmin=0.3):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


class MidiParser(object):

    def __init__(self, jati=4, nadai=4, ticks_per_nadai=24):
        self.jati = jati
        self.nadai = nadai
        self.ticks_per_nadai = ticks_per_nadai

    def reset_meter(self, jati=4, nadai=4, ticks_per_nadai=24):
        self.jati = jati
        self.nadai = nadai
        self.ticks_per_nadai = ticks_per_nadai

    def _construct_template(self, angas=1):
        return BitMap.fromstring('0' * self.nadai * self.jati * angas)

    def load_midi_track(self, addr, track_num=0):
        pattern = midi.read_midifile(addr)
        track = pattern.get_track_by_number(track_num)
        return track

    def split_track_by_instruments(self, track):
        instruments = defaultdict(lambda: OrderedDict())
        for event in track:
            if type(event) == midi.midi.NoteOnEvent:
                instruments[event.pitch][event.tick] = {'volume': event.velocity, 'duration': None}
        return instruments

    def parse_instrument(self, track, angas=1):
        template = self._construct_template(angas)

        for onset_tick in track:
            #print onset_tick,
            if onset_tick % self.ticks_per_nadai != 0:
                raise ValueError("Onset tick doesn't match nadai")
            pos = onset_tick / self.ticks_per_nadai
            #print pos
            template.set(template.size() - 1 - pos)
        template = template.tostring()
        ticks_per_anga = len(template) / angas
        angas = [Anga(template[i*ticks_per_anga:(i+1)*ticks_per_anga]) for i in xrange(angas)]
        print [a.r for a in angas]
        return angas


class MidiWriter(object):
    """
    Writing a Multilinear solo (consisting of multiple avartams) into a single MIDI track.
    Each track (=Multilinear) corresponds to a single instrument family.
    Each channel (=Linear) corresponds to a single instrument
    """

    def __init__(self, bpm, track=0, track_name="Phunkie"):
        self.bpm = bpm
        self.track=track
        self.midi = MIDIFile(1)
        self.midi.addTrackName(self.track, 0, track_name)
        self.midi.addTempo(self.track, 0, self.bpm)
        self.ticks_total = Counter()

    def add_avartam(self, avartam, pitch, channel=0, ticks_per_nadai=32):
        # TODO: work out duration

        for key in sorted(avartam.r_volume.keys())[:-2]:
            time = (key * ticks_per_nadai + self.ticks_total[pitch]) / 128.
            duration = 1
            self.midi.addNote(self.track, channel, pitch, time, duration, volume=avartam.r_volume[key]*127)

        for key in sorted(avartam.r_volume.keys())[-2:]:
            last_possible_tick = ((len(avartam.r)) * ticks_per_nadai) / 128.
            time = (key * ticks_per_nadai + self.ticks_total[pitch]) / 128.
            if (key+8) * ticks_per_nadai / 128. > last_possible_tick:
                duration = last_possible_tick - (key * ticks_per_nadai-8) / 128.
            else:
                duration = 1

            self.midi.addNote(self.track, channel, pitch, time=time, duration=duration,
                                    volume=avartam.r_volume[key]*127)

        self.ticks_total[pitch] += len(avartam.r) * ticks_per_nadai

    def save_midi(self, addr):
        self.midi.close()
        binfile = open(addr, 'wb')
        self.midi.writeFile(binfile)
        binfile.close()
