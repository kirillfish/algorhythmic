#coding: utf-8
import os
import markov_rhythmics as mr
import linear_spawn as sp


__author__ = "kurtosis"


class GenericFarm(object):

    def __init__(self, version, tags, bpm):
        self.linears = []
        self.pitches = []
        self.tpn = []
        self.starts = []
        self.version = version
        self.tags = tags
        self.bpm=bpm

    def add_linear(self, linear, pitch, ticks_per_nadai=32):
        self.linears.append(linear)
        self.pitches.append(pitch)
        self.tpn.append(ticks_per_nadai)

    def _generate_midi_loop(self, avartams_per_loop, num=None):
        w = mr.MidiWriter(bpm=self.bpm, track_name='.'.join([str(self.version), str(num)]))
        for av in xrange(avartams_per_loop):
            for i, linear in enumerate(self.linears):
                avartam = linear.make_variation()
                w.add_avartam(avartam, pitch=self.pitches[i], ticks_per_nadai=self.tpn[i])
        return w

    def generate_loops(self, how_many, avartams_per_loop=2,
                       with_initial=True,
                       output_path='/Users/k.rybachuk/rhythm/loops',
                       first_index=1):
        self.starts = []
        w = mr.MidiWriter(bpm=self.bpm, track_name=self.version)
        for i, linear in enumerate(self.linears):
            self.starts.append(linear.construct_avartam())

        if with_initial:
            for i, start in enumerate(self.starts):
                w.add_avartam(start, pitch=self.pitches[i], ticks_per_nadai=self.tpn[i])
            w.save_midi(os.path.join(output_path, '%s.0.mid' % self.version))

        for num in xrange(1, how_many+1):
            w = self._generate_midi_loop(avartams_per_loop, num=num)
            w.save_midi(os.path.join(output_path, '%s.%d.mid' % (self.version, num + first_index-1)))


# HERE START THE FARM DEFINITIONS

farm_067 = GenericFarm('0.6.7', ['latin', 'percussion', 'bonga', 'clave', '120bpm', 'algorhythmic', 'dope'], 120)
farm_082 = GenericFarm('0.8.2', ['latin', 'percussion', 'bonga', 'clave', '120bpm', 'algorhythmic'], 120)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

farm_067.add_linear(
    sp.LinearKick(variability=0.01, density=0.4, irregularity=.8,
                      log='kicklog.log', log_level='info', log_name='linear kick',
                      start = '10001011100010111000101110001011', volume_serendipity=0.8, angas_per_avartam=2,
                     kickiness_level=3),
    41, 32
)
farm_067.add_linear(
    sp.LinearClave(variability=0.01, density=0.2, irregularity=0.9,
                 log='clavelog.log', log_level='info', log_name='linear clave',
                 start = '00100000100000100000001000001000', volume_serendipity=0.2, angas_per_avartam=2),
    39, 32
)
farm_067.add_linear(
    sp.LinearHat(variability=0.3, density=0.3, irregularity=1.0,
                  log='hatlog.log', log_level='info', log_name='linear hat',
                  start = '01000010010000100011001000001001', volume_serendipity=0.2, angas_per_avartam=2),
    43, 32
)
farm_067.add_linear(
    sp.LinearHat(variability=0.1, density=0.5, irregularity=0.7,
                 log='tishralog.log', log_level='info', log_name='linear tishra',
                 start = '000001000000000001001001000000001010100001010100',
                 volume_serendipity=0.2, angas_per_avartam=2),
    45, 32*2./3
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

farm_082.add_linear(
    sp.LinearKick(variability=0.01, density=0.4, irregularity=.8,
                      log='kicklog.log', log_level='info', log_name='linear kick',
                      start = '10001011100010111000101110001011', volume_serendipity=0.8, angas_per_avartam=2,
                     kickiness_level=3),
    41, 32
)
farm_082.add_linear(
    sp.LinearClave(variability=0.01, density=0.2, irregularity=0.9,
                 log='clavelog.log', log_level='info', log_name='linear clave',
                 start = '00100000100000100000001000001000', volume_serendipity=0.2, angas_per_avartam=2),
    39, 32
)
farm_082.add_linear(
    sp.LinearHat(variability=0.3, density=0.4, irregularity=1.0,
                  log='hatlog.log', log_level='info', log_name='linear hat', mean_volume=0.9,
                  start = '0110010100100100', volume_serendipity=0.2, angas_per_avartam=2),
    43, 64
)
farm_082.add_linear(
    sp.LinearHat(variability=0.1, density=0.4, irregularity=0.9,
                 log='tishralog.log', log_level='info', log_name='linear tishra',
                 start = '010100010101000100010010',
                 volume_serendipity=0.2, angas_per_avartam=2),
    45, 32*4./3
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #