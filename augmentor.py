import imgaug as ia
from imgaug import augmenters as iaa

### augmentors by https://github.com/aleju/imgaug
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

aug_pipe = iaa.Sequential(
    [
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images

        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-90, 90),  # rotate by -90 to +90 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
            mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),

        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
                   [
                       # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                       iaa.OneOf([
                           iaa.GaussianBlur((0, 0.5)),  # blur images with a sigma between 0 and 0.5
                           # blur image using local medians with kernel sizes between 2 and 7
                       ]),
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                       # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                       # search either for all edges or for directed edges
                       # sometimes(iaa.OneOf([
                       #    iaa.EdgeDetect(alpha=(0, 0.7)),
                       #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                       # ])),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                       # add gaussian noise to images
                       iaa.OneOf([
                           iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                           # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                       ]),
                       # iaa.Invert(0.05, per_channel=True), # invert color channels
                       iaa.Add((-10, 10), per_channel=0.5),
                       # change brightness of images (by -10 to 10 of original value)
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       # change brightness of images (50-150% of original value)
                       iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                       # iaa.Grayscale(alpha=(0.0, 1.0)),
                       # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                       # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                   ],
                   random_order=True
                   )
    ],
    random_order=True
)
