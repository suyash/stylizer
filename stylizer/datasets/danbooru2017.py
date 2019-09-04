"""
A TFDS builder for the Danbooru dataset.

Currently requires manually downloading the zip files.

The "Edge Blurred" variant can be created by extracting `danbooru-images.zip` and then applying the edge
smoothing script provided in the misc folder, and then zipping it back up.

Currently not parsing metadata associated with the images, just the content, also the faces are also provided.
"""

import os

import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """
DESCRIPTION
"""

_CITATION = """
CITATION
"""


class Danbooru2017Config(tfds.core.BuilderConfig):
    def __init__(self, subset, **kwargs):
        """
        BuilderConfig for Danbooru2017
        """
        super(Danbooru2017Config,
              self).__init__(name=subset,
                             description="%s subset of Danbooru2017" % subset,
                             **kwargs)
        self.subset = subset


class Danbooru2017(tfds.core.GeneratorBasedBuilder):

    VERSION = tfds.core.Version("0.1.0")

    BUILDER_CONFIGS = [
        Danbooru2017Config(subset="danbooru-images", version="0.1.0"),
        Danbooru2017Config(subset="danbooru-images-edge-blurred",
                           version="0.1.0"),
        Danbooru2017Config(subset="moeimouto-faces", version="0.1.0"),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=(_DESCRIPTION),
            features=tfds.features.FeaturesDict({
                "image":
                tfds.features.Image(),
                "image/filename":
                tfds.features.Text(),
            }),
            urls=[
                "https://www.gwern.net/Danbooru2018",
                "https://www.kaggle.com/mylesoneill/tagged-anime-illustrations"
            ],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archive_path = os.path.join(dl_manager.manual_dir,
                                    "%s.zip" % self.builder_config.subset)
        num_shards = 250 if self.builder_config.subset == "danbooru-images" else 25
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                num_shards=num_shards,
                gen_kwargs={"archive": dl_manager.iter_archive(archive_path)})
        ]

    def _generate_examples(self, archive):
        for fname, fobj in archive:
            if fname.endswith(".jpg") or fname.endswith(".png"):
                yield {
                    "image": fobj,
                    "image/filename": fname,
                }
