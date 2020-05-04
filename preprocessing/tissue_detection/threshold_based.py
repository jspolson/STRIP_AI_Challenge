import numpy as np
from skimage import color
from skimage import morphology as skmp


class tissueDetector:
    def __init__(self, slide):
        self.slide = slide

    def get_tissue_area(self):
        level = self.slide.level_count
        level -= 1
        dw_samples_dim = self.slide.level_dimensions[level]
        # get sample from lowest available resolution
        dw_sample = self.slide.read_region((0, 0), level, (dw_samples_dim[0], dw_samples_dim[1]))
        # convert to RGB (originally should be RGBA)
        dw_sample = dw_sample.convert('RGB')
        # convert to HSV color space: H: hue, S: saturation, V: value
        dw_sample_hsv = color.rgb2hsv(np.asarray(dw_sample))

        # Get first ROI to remove all kinds of markers (Blue, Green, black)
        roi1 = (dw_sample_hsv[:, :, 0] <= 0.67) | (
                (dw_sample_hsv[:, :, 1] <= 0.15) & (dw_sample_hsv[:, :, 2] <= 0.75))
        # exclude marker roi
        roi1 = ~roi1
        skmp.remove_small_holes(roi1, area_threshold=500, connectivity=20, in_place=True)
        skmp.remove_small_objects(roi1, min_size=300, connectivity=20, in_place=True)

        # remove background: regions with low value(black) or very low saturation (white)
        roi2 = (dw_sample_hsv[:, :, 1] >= 0.05) & (dw_sample_hsv[:, :, 2] >= 0.25)
        roi2 *= roi1

        skmp.remove_small_holes(roi2, area_threshold=500, connectivity=20, in_place=True)
        skmp.remove_small_objects(roi2, min_size=300, connectivity=20, in_place=True)
        return roi2