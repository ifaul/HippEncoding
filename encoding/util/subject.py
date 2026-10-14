import nibabel as nib
import numpy as np
from nilearn import signal
from sklearn.base import BaseEstimator, TransformerMixin

from .path import Path


class GiftiMasker(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.init_args = kwargs

    def fit(self, gifti_imgs: Path | list[Path], **kwargs):
        self.gifti_img = gifti_imgs
        self.init_args.update(kwargs)
        return self

    def transform(self, gifti_imgs: Path | list[Path]):
        if not isinstance(gifti_imgs, list):
            gifti_imgs = [gifti_imgs]

        images = []
        for gifti_img in gifti_imgs:
            gifti = nib.load(gifti_img)
            signals = gifti.agg_data().T  # type:ignore
            images.append(signal.clean(signals, **self.init_args))

        signals = np.hstack(images)

        return signals
