from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE

class DatasetGenerator:
    def __init__(self, dataset_name, data_dir, image_size, batch_size, mode):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

        if mode == 'train':
            self.train_data, self.number_train = self._load_train_datasets()
        elif mode == 'all':
            self.data, self.number_all = self._load_all_datasets()
        else:
            self.valid_data, self.number_valid = self._load_valid_datasets()


    def _load_valid_datasets(self):
        valid_1 = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[90%:]', shuffle_files=True)
        valid_2 = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[90%:]', shuffle_files=True)

        valid_data = valid_1.concatenate(valid_2)

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid

    def _load_train_datasets(self):
        train_1 = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:90%]', shuffle_files=True)
        train_2 = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train[:90%]')

        train_data = train_1.concatenate(train_2)

        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", number_train)

        return train_data, number_train

    def _load_all_datasets(self):
        data = tfds.load(self.dataset_name,
                               data_dir=self.data_dir, split='train')

        number_all = data.reduce(0, lambda x, _: x + 1).numpy()
        print("전체 데이터 개수", number_all)

        return data, number_all


    def load_test(self, sample):
        img = sample['rgb']
        gender = sample['gender']
        age = sample['age']


        return (img, age, gender)


    @tf.function
    def preprocess(self, sample):
        img = tf.cast(sample['rgb'], tf.float32)
        age = tf.cast(sample['age'], tf.float32)

        if tf.random.uniform([]) > 0.5:
            img = tf.image.resize(img, size=(self.image_size[0] * 2, self.image_size[1] * 2),
                        method=tf.image.ResizeMethod.BILINEAR)

            img = tf.image.random_crop(img, [self.image_size[0], self.image_size[1], 3])
        

        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, 0.5, 1.5)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, 0.05)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, 0.5, 1.5)           
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)

        img = preprocess_input(img, mode='torch')

        return (img, age)

    @tf.function
    def preprocess_valid(self, sample):
        img = tf.cast(sample['rgb'], tf.float32)
        age = tf.cast(sample['age'], tf.float32)

        img = tf.image.resize(img, size=self.image_size,
                    method=tf.image.ResizeMethod.BILINEAR)

        img = preprocess_input(img, mode='torch')
        

        return (img, age)

    def get_trainData(self, train_data):
        train_data = train_data.shuffle(1024)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()
        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data
