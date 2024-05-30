import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from src.main import SynthesizerTrn


if __name__ == "__main__":
    datasets = tfds.load("vctk", split=["train", "test"], shuffle_files=True)

   

    print(tfds.audio.Vctk.get_metadata())

    model = SynthesizerTrn()