from skimage import color
import numpy as np
import openslide
from skimage import morphology as skmp


def check_ihc_slide(slide):
    """
    check whether the current slide is IHC stained
    :param slide:
    :return:
    """
    sample = slide.read_region((0, 0), slide.level_count - 1,
                              (slide.level_dimensions[slide.level_count - 1][0],
                               slide.level_dimensions[slide.level_count - 1][1]))
    sample = sample.convert('RGB')
    sample_hsv = color.rgb2hsv(np.asarray(sample))
    # brownish stain
    roi_ihc = (sample_hsv[:, :, 0] >= 0.056) & (sample_hsv[:, :, 0] <= 0.34) & (sample_hsv[:, :, 2] > 0.2) & (
                sample_hsv[:, :, 1] > 0.04)
    skmp.remove_small_holes(roi_ihc, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi_ihc, min_size=500, connectivity=20, in_place=True)

    is_ihc = float(np.sum(roi_ihc)) / float((roi_ihc.shape[0] * roi_ihc.shape[1])) > 0.01

    return is_ihc


def generate_binary_mask(tile):
    """
    generate binary mask for a given tile
    :param tile:
    :return:
    """
    tile_hsv = color.rgb2hsv(np.asarray(tile))
    roi1 = (tile_hsv[:, :, 0] >= 0.33) & (tile_hsv[:, :, 0] <= 0.67)
    roi1 = ~roi1

    skmp.remove_small_holes(roi1, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi1, min_size=500, connectivity=20, in_place=True)

    tile_gray = color.rgb2gray(np.asarray(tile))
    masked_sample = np.multiply(tile_gray, roi1)
    roi2 = (masked_sample <= 0.8) & (masked_sample >= 0.2)

    skmp.remove_small_holes(roi2, min_size=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi2, min_size=500, connectivity=20, in_place=True)

    return tile_hsv, roi2


def read_downsample_slide(slides_dir, slide_name):
    slide = openslide.OpenSlide(f"{slides_dir}/{slide_name}")
    level = slide.level_count
    level -= 1
    dw_samples_dim = slide.level_dimensions[level]
    # get sample from lowest available resolution
    dw_sample = slide.read_region((0, 0), level, (dw_samples_dim[0], dw_samples_dim[1]))
    return dw_sample