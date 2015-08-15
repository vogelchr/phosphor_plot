#!/usr/bin/python

# phosphor_plot.py (c) 2015 Christian Vogel <vogelchr@vogel.cx>
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.


import logging
from logging import debug, info, warning, error, critical

import math
import numpy as np
import scipy.signal
import scipy.misc

from functools import reduce
import operator
import itertools

# read in a filename, or some "magic" sources
def get_input(fn) :
    # magic square wave for testing
    if fn == '@square' :
        info('Generating a simple square wave.')
        data = np.zeros(100)
        for i in range(10) :
            a = i*10
            b,c = a+5, a+10
            data[a:b] = -1
            data[b:c] = 1
        return data

    if fn == '@sine' :
        info('Generating a simple sine wave.')
        return np.sin(np.linspace(0, 31.41, 100))

    if fn.lower().endswith('.wav') :
        info('Reading %s as a wav file.', fn)
        import scipy.io.wavfile
        rate, data = scipy.io.wavfile.read(fn)
        # for stereo, only use left channel

        info('%s: has a sample rate of %d Hz (ignored) and %d channel(s).',
            fn, rate, data.shape[1])
        return data[:,0]

    if fn.lower().endswith('.txt') :
        return np.loadtxt(fn)

    raise RuntimeError('Don\'t know how to read file %s.', fn)

def fract_to_int_neigh_with_weight(fract) :
    '''Convert a fractional number to their integer neighbors and a weight
    indicating the proximity. E.g. f(1.0) -> [ (1, 1.0) ],
    f(1.5) -> [ (1, 0.5), (2, 0.5)], f(1.2) = [(1, .8), (2, 0.2)].'''

    floor_idx = math.floor(fract)
    ceil_idx = math.ceil(fract)

    floor_val = ceil_idx - fract
    ceil_val  = fract - floor_idx

    if floor_idx == ceil_idx :
        return [(floor_idx, 1.0)]
    else :
        return [(floor_idx, floor_val), (ceil_idx, ceil_val)]

def add_at_fract_idx_with_weight(arr, fract_idx, increment=1.0) :
    '''Increment data in an array at a "fractional index" by spreading
    the increment around the nearest neighbors weighted by (linear) proximity.'''

    # fractional indices and weights for all dimensions, [ [(i,w),(i,w)], ..]
    neigh_idx_arr = [ fract_to_int_neigh_with_weight(fi) for fi in fract_idx ]
    for i_w_arr in itertools.product(*neigh_idx_arr) :
        # complete index for this particular neighbor
        idx = tuple([iw[0] for iw in i_w_arr])
        # weight, calculated as product of all dimensions' weights
        weight = reduce(operator.mul, [iw[1] for iw in i_w_arr], 1.0)
        # add to array
        arr[idx] += weight*increment

def scale_and_offs(amin, amax, bmin, bmax, offs_is_on_a=False) :
    '''Calculate scale factor and offset to transfer a range
      [amin, amax] to [bmin, bmax]
      if offs_is_on_a is False (default) then
         b = a * scale + offs
      if offs_is_on_a is True then
         b = (a+offs) * scale'''

    scale = (float(bmax)-float(bmin))/(float(amax)-float(amin))
    offs = bmin-amin*scale
    if offs_is_on_a :
        offs /= scale
#    debug('scale_and_offs(%f, %f, %f, %f, %s) -> (%f, %f)',
#        amin, amax, bmin, bmax, offs_is_on_a, scale, offs)
    return scale, offs

def rescale_array(arr, new_min, new_max) :
    '''Rescale array so that its values cover the range new_min to new_max.'''
    curr_min = np.amin(arr)
    curr_max = np.amax(arr)
    scale, offs = scale_and_offs(curr_min, curr_max, new_min, new_max)
    return arr*scale+offs

def add_trace_to_img(img_arr, trace_arr, color) :
    '''Add trace_arr(grayscale) to RGB image img_arr, using specified color.'''
    for i, c in enumerate(color) :
        img_arr[:,:,i] += trace_arr * c

# our favourite colours, traces cycle through them
COLORS = [ (000.0, 255.0, 255.0),  # cyan
           (255.0, 255.0, 000.0),  # yellow
           (255.0, 000.0, 000.0),  # red
           (000.0, 255.0, 000.0),  # green
         ]

def main() :
    logging.basicConfig(format='\033[0;1m%(asctime)-15s\033[0m %(message)s')

    import optparse
    parser = optparse.OptionParser(usage='%prog [options] INPUTS...')

    parser.add_option('-W', '--width', dest='width', type='int', default=800,
        metavar='N', help='Width of picture to generate. (def: 800)')
    parser.add_option('-H', '--height', dest='height', type='int', default=600,
        metavar='N', help='Height of picture to generate. (def: 600)')

    parser.add_option('-s', '--sigma', dest='sigma', type='float', default=0.0,
        metavar='PIXELS',
        help='Make trace unsharp by convolving with a gaussian of PIXELS width.'+\
         '(def: off)')
    parser.add_option('-g', '--gamma', dest='gamma', type='float', default=0.3,
        metavar='GAMMA',
        help='Apply gamma to image, to intensify dark parts of the traces. (def: 0.3)')

    parser.add_option('-R', '--resample', dest='resample', type='int', default=100,
        metavar='N',
        help='Minimal resampling (factor times the horizontal resolution) (def:100)')

    parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
        help='Be verbose. (def: not)')
    parser.add_option('-d', '--debug', dest='debug', action='store_true',
        help='Be even more verbose (for debugging) (def: not).')

    parser.add_option('-o', '--output', dest='output', default=None,
        metavar='FILENAME', help='Write out plot as image to FILENAME.')

    opts, args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if opts.debug else (
        logging.INFO if opts.verbose else logging.WARNING ))

    if not args :
        parser.error('You have to specify at least one file to plot.')
    if not opts.output :
        parser.error('You have to specify the output file using -o / --output.')

    ##################################
    ### NOW FOR THE ACTUAL DRAWING
    ##################################

    # pixels to keep empty around image, to avoid under/overflows in
    # computation of neighbors, ...
    border = 5

    # generate array representing image
    img_data = np.zeros((opts.height, opts.width, 3))
#        dtype=[('r',np.float),('g',np.float),('b',np.float)])

    img_xmin, img_xmax = border, img_data.shape[1]-border
    img_ymin, img_ymax = border, img_data.shape[0]-border

    # minimum numbers of samples in X that we need, to decide on resampling
    min_xpts = (img_xmax - img_xmin)*opts.resample

    for i, arg in enumerate(args) :
        data = get_input(arg)

        info('%s: %d samples in original data source', arg, data.shape[0])

        if data.shape[0] < min_xpts :
            data = scipy.signal.resample(data, min_xpts)
            info('Not enough datapoints, resampled to %d.', min_xpts)

        # build x and y coordinate array
        x_coords = np.linspace(img_xmin, img_xmax, len(data))
        y_coords = rescale_array(data, img_ymin, img_ymax)

        trace_img = np.zeros(img_data.shape[0:2])

        # draw actual pixels
        for x, y in np.nditer((x_coords, y_coords)) :
            add_at_fract_idx_with_weight(trace_img, (y, x))

        if opts.sigma :
            info('Making the trace unsharp, using a gaussian of width %f.',opts.sigma)
            trace_img = scipy.ndimage.filters.gaussian_filter(trace_img, opts.sigma)

        info('Applying gamma function for gamma=%f', opts.gamma)
        trace_img = np.power(rescale_array(trace_img, 0.0, 1.0), opts.gamma)

        info('Using color %s for this trace.', COLORS[i % len(COLORS)])
        add_trace_to_img(img_data, trace_img, COLORS[i % len(COLORS)])

    info('Writing out plot to file %s.', opts.output)
    scipy.misc.toimage(img_data).save(opts.output)


if __name__ == '__main__' :
    main()
