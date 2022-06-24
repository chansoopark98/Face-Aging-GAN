"""imdb_dataset dataset."""

import tensorflow_datasets as tfds
import os
import glob
import natsort
import tensorflow as tf
import csv

# TODO(imdb_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(imdb_dataset): BibTeX citation
_CITATION = """
"""


class ImdbDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for imdb_dataset dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/tensorflow_datasets/'
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  
  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(cornell_grasp): Specifies the tfds.core.DatasetInfo object
    
    
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'rgb': tfds.features.Image(shape=(None, None, 3)),
            'gender': tfds.features.Tensor(shape=[], dtype=tf.int32),
            'age': tfds.features.Tensor(shape=[], dtype=tf.int32)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        # supervised_keys=('input', "depth", "box"),  # Set to `None` to disable
        supervised_keys=None,
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(cornell_grasp): Downloads the data and defines the splits
    dataset_dir = '/home/park/park/Face-Aging-GAN/preprocess_data/imdb_data'
    archive_path = dataset_dir + '/imdb_wiki.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(rgb_path=extracted_path/'imdb_crop', csv_path=extracted_path)
    }


  def _generate_examples(self, rgb_path, csv_path):
    csv_file_dir = str(csv_path)+'/imdb.csv'
    
    gender_list = []
    age_list = []
    img_path_list = []

    with open(csv_file_dir, 'r') as file:
      csv_file = csv.reader(file)
    
      for line in csv_file:
          gender, age, img_path = line
          age = int(age)

          if age >= 0 and age <10:
            age = 0
          elif age >= 10 and age <20:
            age = 1
          elif age >= 20 and age <30:
            age = 2
          elif age >= 30 and age <40:
            age = 3
          elif age >= 40 and age <50:
            age = 4
          elif age >= 50 and age <69:
            age = 5
          else:
            age = 6

          gender_list.append(gender)
          age_list.append(age)
          img_path_list.append(img_path)

    
    for idx in range(len(gender_list)):
      img_path = os.path.join(rgb_path, img_path_list[idx])
      yield idx, {
          'rgb': img_path,
          'gender' : gender_list[idx],
          'age': age_list[idx]
      }
    
    