#coding: utf-8
from bitmap import BitMap
from copy import deepcopy
import numpy as np
import scipy as sp
from scipy import stats
from scipy.optimize import minimize_scalar, basinhopping
from scipy.optimize import *
from random import shuffle
from functools import partial
from collections import OrderedDict, defaultdict
from itertools import izip

__author__ = "kurtosis"

JATI_TERMS = {'tishra':3,
              'chatusra':4,
              'khanda':5,
              'mishra':7,
              'sankeerna':9}

class Nadai():
    pass


class Akshara():
    pass


class Anga(object):
    """
    Abstract "bar" structure for univocal track.
    Could be squeezed or stretched both absolutely in time
    and relatively with respect to the meter (in polyrhythms and rhythmic modulation).
    """
    def __init__(self, r='0100100100010010', r_volume=None, jati=4, gati='chatusra'):
        """
        :param r: rhythm itself
        :param jati: "time"
        :param gati: the no. of pulses inside a single beat
        """
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
        self.strengths = self._compute_strengths()
        
        if r_volume is None:
            self.r_volume = {pos: 0.7 for pos in self.nonzero}
            #self.r_volume = {(self.r_bitmap.size() - 1 - index): 0.7 for index in self.r_bitmap.nonzero()}
        else:
            self.r_volume = r_volume

        self.notes_to_update_in_maps = self.nonzero.difference(set(self.r_volume.keys()))
        self.notes_to_delete_from_maps = set(self.r_volume.keys()).difference(self.nonzero)
        for pos in self.notes_to_delete_from_maps:
            del self.r_volume[pos]

    def _compute_strengths(self): #length=16):
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
        #print accents
        #print strengths
        return strengths

    def _flip_note(self, pos):
        # pos indexing is straight
        # bitmap indexing is reverse
        res = deepcopy(self.r_bitmap)
        res.flip(res.size()-1 - pos)
        return res

    def _shift_note(self, pos, dir='right'):
        res = deepcopy(self.r_bitmap)
        if dir == 'right':
            if pos == res.size()-1:
                res.set(res.size()-1)
            else:
                res.set(res.size() - 1 - (pos + 1))
        elif dir == 'left':
            if pos == 0:
                res.set(0)
            else:
                res.set(res.size() - 1 - (pos - 1))
        else:
            raise ValueError("direction parameter should take value either 'right' or 'left'")
        res.reset(res.size() - 1 - pos)
        return res

    def find_adjacent(self):
        flipped = []
        for pos in xrange(self.r_bitmap.size()):
            flipped.append(self._flip_note(pos))
        print len(flipped),
        flipped = set(flipped)
        print len(flipped)

        shifted = []
        nonzero = [self.r_bitmap.size()-1 - reverse_pos for reverse_pos in self.r_bitmap.nonzero()]
        for pos in nonzero:
            shifted.append(self._shift_note(pos, 'right'))
            shifted.append(self._shift_note(pos, 'left'))
        print len(shifted),
        shifted = set(shifted)
        print len(shifted)

        adjacent = flipped.union(shifted)
        print len(shifted) + len(flipped), len(adjacent)
        return adjacent


class Avartam(object):

    def __init__(self, *angas):
        self.r = ''.join(anga.r for anga in angas)
        self.r_volumes = [anga.r_volume for anga in angas]

    def add_tihai(self):
        pass


class Linear(object):

    def __init__(self, variability=0.1, density=0.5, serendipity=0.5, irregularity=0.8, 
                 possible_volume_bounds=(0.3, 1.),
                 volume_serendipity=0.5, volume_serendipity_tolerance=0.3,
                 mean_volume=None, mean_volume_tolerance=0.1,
                 start='0100100100010010',
                 angas_per_avartam = 4,
                 tracks=1):
        self.start = start
        self.current_anga = Anga(self.start)
        self.angas_per_avartam = angas_per_avartam
        self.current_avartam = self.construct_avartam()

        self.poisson_lambda = self._convert_variability(variability)        # for variability

        self.effective_size = 10                                            # for density
        self.alpha_shape = density * self.effective_size
        self.beta_shape = self.effective_size - self.alpha_shape
        self.density = density
        self.volume_tuner = VolumeTuner(self.current_anga,
                                        mean_volume=mean_volume, mean_volume_tolerance=mean_volume_tolerance,
                                       serendipity=volume_serendipity,
                                       serendipity_tolerance=volume_serendipity_tolerance,
                                       possible_volume_bounds=possible_volume_bounds)

        #self.serendipity_alpha_shape = serendipity * self.effective_size    # for serendipity
        #self.serendipity_beta_shape = self.effective_size - self.serendipity_alpha_shape
        #self.serendipity = serendipity
        print "poisson lambda for variability: ", self.poisson_lambda
        print "alpha and beta shapes in beta distribution for density: ", self.alpha_shape, self.beta_shape
        

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
                print 'oops minus'
                break
            #print strength_i, strengths[minus]
            if strengths[pos] < strengths[minus]:
                pos_minus = minus
                dist_minus = pos - minus
                break
        for plus in xrange(pos+1, rhythm_bitmap.size()):
            if rhythm_bitmap.test(rhythm_bitmap.size() - 1 - plus):
                print 'oops plus'
                break
            if strengths[pos] < strengths[plus]:
                pos_plus = plus
                dist_plus = plus - pos
                break
        return pos_minus, pos_plus, dist_minus, dist_plus

    def note_irregularity(self, rhythm_bitmap, pos):
        strengths = self.current_anga.strengths
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

        def irr(pos, attr):
            return strengths[pos]**1.5 * 1./(strengths[pos])**3 / attr

        if attraction_minus > attraction_plus:
            irregularity = irr(pos_minus, attraction_minus)
        elif attraction_minus < attraction_plus:
            irregularity = irr(pos_plus, attraction_plus)
        else:
            irregularity = max(irr(pos_minus, attraction_minus),
                              irr(pos_plus, attraction_plus))
        return irregularity
    
    
    def irregularity_factor(self, rhythm_bitmap):
        # TODO

        return 1

    def density_factor(self, rhythm_bitmap):
        #rhythm_bitmap = BitMap.fromstring(rhythm)
        rhythm_density = len(rhythm_bitmap.nonzero()) * 1. / rhythm_bitmap.size()
        proba = stats.beta.pdf(rhythm_density, self.alpha_shape, self.beta_shape)
        return proba

    def serendipity_factor(self, rhythm_bitmap):
        # TODO
        return 1

    def sampling_probability(self, rhythm_bitmap):
        """
        :param rhythm_bitmap: a candidate to modify current anga
        :return: A probability (up to normalization constant) of the candidate to be chosen
                 in the next step of random walk. The probability factorizes over several parameters
                 (i.e. is just a product of distributions)
        """
        return (self.irregularity_factor(rhythm_bitmap)
                * self.density_factor(rhythm_bitmap)
                * self.serendipity_factor(rhythm_bitmap))

    def make_elementary_step(self):
        print "making elementary step"
        adjacent = list(self.current_anga.find_adjacent())
        adj_probas = []
        for rhythm in adjacent:
            adj_probas.append(self.sampling_probability(rhythm))
        adj_probas = np.array(adj_probas) / sum(adj_probas)

        # debug
        #for i, rhythm in enumerate(adjacent):
        #    print rhythm, adj_probas[i]

        choice = np.random.multinomial(1, adj_probas, size=1)
        choice_index = choice.nonzero()[1]
        print adjacent[choice_index].tostring(), adj_probas[choice_index]
        self.current_anga = Anga(adjacent[choice_index].tostring(), r_volume=self.current_anga.r_volume)

    def construct_avartam(self):
        return Avartam(*([self.current_anga]*self.angas_per_avartam))

    def make_variation(self):
        """
        Determine how much steps of random walk to take before stopping to make it new variation
        :return: new avartam
        """
        random_walk_length = self.generate_walk_length()
        print "random_walk_length: ", random_walk_length
        for i in xrange(random_walk_length):
            self.make_elementary_step()

        print "i am in the make_variation..."
        self.volume_tuner.current_anga = self.current_anga # TODO: make it nicer
        print self.volume_tuner.current_anga.r, self.volume_tuner.current_anga.r_volume
        self.volume_tuner.tune_volume_batch()
        self.current_anga = self.volume_tuner.current_anga
        self.construct_avartam()


class VolumeTuner(object):

    def __init__(self, initial_anga, possible_volume_bounds=(0.3, 1.),
                 serendipity=0.5, serendipity_tolerance=0.3,
                 mean_volume=None, mean_volume_tolerance=0.1):
        """
        Here should be arguments that make sense whatever the handles you are going to use. 
        """
        self.current_anga = initial_anga
        self.effective_size=10
        self.possible_volume_bounds = possible_volume_bounds
        #self.serendipity_alpha_shape = serendipity * self.effective_size
        #self.serendipity_beta_shape = self.effective_size - self.serendipity_alpha_shape
        self.serendipity = serendipity
        self.serendipity_tolerance=serendipity_tolerance
        if mean_volume is None:
            self.mean_volume = np.mean(self.current_anga.r_volume.values())
        else:
            self.mean_volume = mean_volume
        self.mean_volume_tolerance = mean_volume_tolerance
        self.volume_window = (
            max(self.mean_volume*(1-self.mean_volume_tolerance), self.possible_volume_bounds[0]),
            min(self.mean_volume*(1+self.mean_volume_tolerance), self.possible_volume_bounds[1])
                             )
        print "volume window for modifying volume: ", self.volume_window
        self.mean_npvi, self.std_npvi = self.simulate_npvi_distribution()


    def volume_irregularity_factor(self, rhythm_volume_map, rhythm_length):
        # TODO
        return 1

    #@staticmethod
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
        #print 'npvi:', volume_npvi
        return volume_npvi

    def simulate_npvi_distribution(self):
        npvis = []
        means = []
        print self.current_anga.notes_to_update_in_maps
        for i in tqdm(xrange(10000)):
            new_volume = deepcopy(self.current_anga.r_volume)
            for key in self.current_anga.notes_to_update_in_maps: #new_volume:
                new_volume[key] = np.random.uniform(0.3, 1)
            mean = np.mean(new_volume.values())
            if mean >= self.volume_window[0] and mean <= self.volume_window[1]:
                npvi = self.volume_npvi(new_volume)
                npvis.append(npvi)
                means.append(mean)

        mean_npvi = np.percentile(npvis, self.serendipity*100)
        std_npvi = self.serendipity_tolerance*(np.percentile(npvis, 75) - np.percentile(npvis, 25))
        print "mean npvi as it computed from simulations (%d percentile) is %.3f" % (self.serendipity*100, mean_npvi)
        print "standard deviation of normal distribution for npvi as interquartile range: %.3f" % std_npvi
        return mean_npvi, std_npvi
    
    def volume_serendipity_factor(self, rhythm_volume_map):
        mean = np.mean(rhythm_volume_map.values())
        if mean <= self.volume_window[0] or mean >= self.volume_window[1]:
            return 0
        current_npvi = self.volume_npvi(rhythm_volume_map)
        return stats.norm.pdf(current_npvi, self.mean_npvi, self.std_npvi)

    def goodness_volume_function(self, update_values, update_indices): #update_value, update_index):
        """
        :param update_index: where to update volume
        :param update_value: how to update volume
        :return: how good is the resulting configuration
        How good will be volume configuration if we modify the volume of a chosen single note.
        This is only to be used inside tune_volume method (as an objective function).
        """
        rvm = deepcopy(self.current_anga.r_volume)
        rvm.update(dict(zip(update_indices, update_values)))
        #print rvm, self.volume_serendipity_factor(rvm)
        return -(self.volume_serendipity_factor(rvm)
                 * self.volume_irregularity_factor(rvm, self.current_anga.r_bitmap.size()))

    def tune_volume(self):
        """
        After obtaining the next variation via markov process, you need to make spiffier and groovier accents.
        This method optimizes the volumes given the notes in already assembled anga.
        :return: greedily optimized volumes
        """
        # TODO: make non-greedy batch optimization (via scipy.minimize)
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
        self.mean_npvi, self.std_npvi = self.simulate_npvi_distribution()
        update_indices = list(self.current_anga.notes_to_update_in_maps)
        print update_indices
        if len(update_indices) == 0:
            return {}
        
        mybounds = MyBounds()
        objective = partial(self.goodness_volume_function, 
                            update_indices=update_indices)
        solution = basinhopping(objective, 
                                       x0=[self.mean_volume for i in self.current_anga.notes_to_update_in_maps],
                                       niter=200, niter_success=200, accept_test=mybounds)
        print solution
        print solution['x']
        for i,res in enumerate(solution['x']):
            solution['x'][i] = min(1, res)
        print solution['x']
        optimal_volumes = dict(zip(update_indices, solution['x']))
        print "Optimal volumes: ", optimal_volumes
        self.current_anga.r_volume.update(optimal_volumes)
        print "debug: optimal npvi: ", self.volume_npvi(self.current_anga.r_volume)
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
            
    
class LinearPhunk(Linear):
    pass


class LinearCarnatic(Linear):
    pass


class LinearClave(Linear):
    pass


class Multilinear(Linear):
    pass


class MultilinearPhunk(Multilinear):
    pass


class MultilinearCarnatic(Multilinear):
    pass


class MultilinearClave(Multilinear):
    pass


class MidiParser(object):
    pass


class MidiWriter(object):
    pass