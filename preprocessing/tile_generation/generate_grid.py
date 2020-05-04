import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from build_dataset.tile_generation.tile_generator_abc import TileGeneratorABC
from build_dataset.tile_generation.utils import prep_utils
from PIL import Image
import numpy as np
from skimage import color
from skimage import morphology as skmp
import openslide
import time
import h5py
import cv2


class TileGeneratorGrid(TileGeneratorABC):
    def __init__(self, slide_loc, slide_id, tile_size, mag=5, overlap=0, check_ihc=False, verbose=False):
        """
        Create a DeepZoomGenerator wrapping an OpenSlide object.
        :param slide_loc: location for the slide
        :param slide_id: id of current slide (e.g., slide name)
        :param tile_size: the width and height of a single tile.  For best viewer
                          performance, tile_size + 2 * overlap should be a power
                          of two.
        :param mag: extract tiles at which magnification (e.g., 10x, 20x)
        :param overlap: Overlapping size (only used when tiles are extracted from the grid)
        :param check_ihc: whether to check current slide is IHC slide
        :param verbose: print logs
        """
        self.osr = openslide.OpenSlide("%s/%s.svs" % (slide_loc, slide_id))
        self.slide_id = slide_id
        self.tile_size = tile_size
        self.dw_rate = 1
        # how much overlap on the required magnification
        self.overlap = int(overlap * tile_size)
        # desired magnification
        self.mag = mag
        # whether to print logs
        self.verbose = verbose
        # original magnification
        app_mag = int(self.osr.properties['openslide.objective-power'])
        try:
            assert (self.mag <= app_mag)
        except AssertionError:
            print("Highest magnification for current slide is " + str(app_mag))
        # if the highest magnification
        if app_mag > mag:
            self.dw_rate = int(app_mag / mag)
            self.tile_size *= self.dw_rate
            self.overlap *= self.dw_rate

        self.ihc = prep_utils.check_ihc_slide(self.osr) if check_ihc else False
        self.roi = self.sep_bg_tissue()

    def is_ihc_slide(self):
        return self.ihc

    def sep_bg_tissue(self):
        """
        Generate tissue mask fot the slide
        :return:
        """
        level = self.osr.level_count
        level -= 1
        dw_samples_dim = self.osr.level_dimensions[level]
        # get sample from lowest available resolution
        dw_sample = self.osr.read_region((0, 0), level, (dw_samples_dim[0], dw_samples_dim[1]))
        # convert to RGB (originally should be RGBA)
        dw_sample = dw_sample.convert('RGB')
        # convert to HSV color space: H: hue, S: saturation, V: value
        dw_sample_hsv = color.rgb2hsv(np.asarray(dw_sample))

        # Get first ROI to remove all kinds of markers (Blue, Green, black)
        roi1 = (dw_sample_hsv[:, :, 0] <= 0.67) | (
                    (dw_sample_hsv[:, :, 1] <= 0.15) & (dw_sample_hsv[:, :, 2] <= 0.75))
        # exclude marker roi
        roi1 = ~roi1
        skmp.remove_small_holes(roi1, min_size=500, connectivity=20, in_place=True)
        skmp.remove_small_objects(roi1, min_size=300, connectivity=20, in_place=True)

        # remove background: regions with low value(black) or very low saturation (white)
        roi2 = (dw_sample_hsv[:, :, 1] >= 0.05) & (dw_sample_hsv[:, :, 2] >= 0.25)
        roi2 *= roi1

        skmp.remove_small_holes(roi2, min_size=500, connectivity=20, in_place=True)
        skmp.remove_small_objects(roi2, min_size=300, connectivity=20, in_place=True)
        return roi2

    # Generate tiles from grid, with optional overlap
    def get_all_tiles(self, thres):
        """
        Generate tiles from the grid
        :param thres: tissue threshold (i.e., tile contains tissue region greater than thres)
        :return:
        """
        # Get the lowest rate for ROI
        lowest_rate = int(self.osr.level_downsamples[self.osr.level_count - 1])
        # Calculate the tile size to be used on
        small_tile_size = int(self.tile_size / lowest_rate)
        small_overlap = int(self.overlap / lowest_rate)
        interval = small_tile_size - small_overlap
        counter = 0
        location_tracker = {}
        for i in range(0, int(self.osr.level_dimensions[self.osr.level_count - 1][1]), interval):
            for j in range(0, int(self.osr.level_dimensions[self.osr.level_count - 1][0]), interval):
                if (i * lowest_rate + self.tile_size) <= self.osr.level_dimensions[0][1] and (
                            j * lowest_rate + self.tile_size) <= self.osr.level_dimensions[0][0]:
                    size_x = small_tile_size if i + small_tile_size <= \
                                                self.osr.level_dimensions[self.osr.level_count - 1][1] \
                        else self.osr.level_dimensions[self.osr.level_count - 1][1] - i
                    size_y = small_tile_size if j + small_tile_size <= \
                                                self.osr.level_dimensions[self.osr.level_count - 1][0] \
                        else self.osr.level_dimensions[self.osr.level_count - 1][0] - j
                    small_tile = self.roi[i:i + size_x, j:j + size_y]
                    if (float(np.count_nonzero(small_tile)) / float(size_x * size_y)) >= thres:
                        # Original slide were down-sampled by lowest_rate to separate tissue from background
                        location_tracker[counter] = np.zeros((2,))
                        location_tracker[counter][0] = int(j * lowest_rate)
                        location_tracker[counter][1] = int(i * lowest_rate)
                        counter += 1
        print("Generate %d tiles in the grid" % counter)
        return counter, location_tracker

    # Generate tiles based on location at highest magnification
    def generate_tile(self, location, normalizer=None):
        orig_tile = self.osr.read_region((location[0], location[1]), 0, (self.tile_size, self.tile_size))
        orig_tile = np.asarray(orig_tile.convert('RGB'))
        _, mask = prep_utils.generate_binary_mask(orig_tile)
        norm_tile = None
        if normalizer:
            norm_method = normalizer.get_norm_method()
            try:
                if norm_method == 'reinhard':
                    norm_tile = normalizer.transform(orig_tile.astype(np.uint8), mask)
                else:
                    norm_tile = normalizer.transform(orig_tile)
            except:
                print(self.slide_id)
                orig_tile = Image.fromarray(orig_tile)
                norm_tile = orig_tile
                orig_tile.save("error_tile.png")
        return np.asarray(orig_tile), np.asarray(norm_tile), mask

    def gen_and_extract_tiles(self, normalizer, ts_thres):
        """
        generator and extract tiles from current slide
        :param normalizer: normalization method
        :param ts_thres: tissue threshold
        ( tiles need to contain at least ts_thres tissue regions to be included in the analysus)
        :return: orig_tiles, norm_tiles, locations
        """
        # compute the actual extracted tile size
        start_time = time.time()
        extracted_tile_size = int(float(self.tile_size) / float(self.dw_rate))
        counter, location_tracker = self.get_all_tiles(ts_thres)
        norm_tiles = np.zeros((counter, extracted_tile_size, extracted_tile_size, 3), dtype=np.uint8)
        orig_tiles = np.zeros((counter, extracted_tile_size, extracted_tile_size, 3), dtype=np.uint8)
        masks = np.zeros((counter, extracted_tile_size, extracted_tile_size), dtype=np.uint8)
        locations = np.zeros((counter, 2), dtype=np.int64)

        for tile_id in range(counter):
            cur_loc = location_tracker[tile_id]
            # generate normalized tiles
            orig_tile, norm_tile, mask = self.generate_tile([int(cur_loc[0]), int(cur_loc[1])], normalizer=normalizer)
            orig_tile = Image.fromarray(orig_tile)
            # resize tiles
            orig_tile = orig_tile.resize((extracted_tile_size, extracted_tile_size), Image.ANTIALIAS) \
                if extracted_tile_size < orig_tile.size[0] else orig_tile

            orig_tiles[tile_id, :, :, :] = orig_tile

            norm_tile = Image.fromarray(norm_tile)
            norm_tile = norm_tile.resize((extracted_tile_size, extracted_tile_size), Image.ANTIALIAS) \
                if extracted_tile_size < norm_tile.size[0] else norm_tile

            norm_tiles[tile_id, :, :, :] = norm_tile

            mask = mask.astype(np.float32)
            mask = cv2.resize(np.asarray(mask), None, fx=1.0 / self.dw_rate, fy=1.0 / self.dw_rate) \
                if self.dw_rate != 1 else mask

            masks[tile_id, :, :] = mask

            locations[tile_id, 0] = int(cur_loc[0])
            locations[tile_id, 1] = int(cur_loc[1])
        if self.verbose:
            print("Time to generate %d tiles from %s slide: %.2f" % (counter, str(self.slide_id), time.time()-start_time))
        return orig_tiles, norm_tiles, locations, masks

    def save_tiles_h5py(self, h5py_loc, normalizer, ts_thres, save_orig=True, save_mask=False):
        """
        save tiles into h5py format
        contains datasets for tiles
        orig_tiles_<mag>: tiles before normalization
        norm_tiles_<mag>: tiles after normalization
        locations: location for the left upper corner of each tile at highest scanning magnification
        :param h5py_loc:
        :param normalizer:
        :param ts_thres:
        :param save_orig:
        :param save_mask:
        :return:
        """
        orig_tiles, norm_tiles, locations, masks = self.gen_and_extract_tiles(normalizer, ts_thres)
        # create the h5py file
        f = h5py.File("%s/%s.hdf5" % (h5py_loc, self.slide_id), "w")
        if save_orig:
            f.create_dataset("orig_tiles_%dx" % self.mag, data=orig_tiles,
                             dtype='uint8', compression='lzf')
        if save_mask:
            f.create_dataset('masks_%dx' % self.mag, data=masks,
                             dtype='uint8', compression='lzf')
        # create dataset for normalized tiles
        f.create_dataset("norm_tiles_%dx" % self.mag, data=norm_tiles,
                         dtype='f', compression='lzf')
        # create dataset for tile locations (left up corner)
        f.create_dataset("locations", data=locations, dtype='f')

        f.close()
        return

