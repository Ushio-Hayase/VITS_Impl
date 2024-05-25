import tensorflow as tf
from tensorflow import keras

def pad_mask(x, pad_id : int):
    pad = [x!=pad_id][:, tf.newaxis, tf.newaxis, :]
    return pad

def relative_position_encoding(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    ret = 0
    n = -relative_position
    if bidirectional:
        num_buckets //= 2
        ret += (n < 0).to(tf.int64) * num_buckets
        n = tf.abs(n)
    else:
        n = tf.max(n, tf.zeros_like(n))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
            tf.math.log(n.float() / max_exact) / tf.math.log(max_distance / max_exact) * (num_buckets - max_exact)
    ).to(tf.int64)
    val_if_large = tf.min(val_if_large, tf.fill(val_if_large, num_buckets - 1))
    ret += tf.where(is_small, n, val_if_large)
    return ret


def monotonic_alignment_search(latent_representation, text_length, mel_spectrogram_length):
    """
    두 문장 간의 단어 정렬을 찾는 Monotonic Alignment Search 알고리즘을 구현합니다.

    Args:
        latent_representation: 텍스트의 잠재 표현.
        text_length: 텍스트의 길이.
        mel_spectrogram_length: Mel-스펙트rogram의 길이.

    Returns:
        단어 정렬.
    """

    # 단어 임베딩 계산
    text_embeddings = tf.layers.dense(latent_representation, text_length)
    mel_spectrogram_embeddings = tf.layers.dense(latent_representation, mel_spectrogram_length)

    # 유사도 계산
    similarity_matrix = tf.matmul(text_embeddings, mel_spectrogram_embeddings, transpose_b=True)

    # 단어 정렬 후보 생성
    candidates = tf.stack([tf.range(text_length), tf.range(mel_spectrogram_length)], axis=-1)

    # 후보 평가
    monotonic_scores = tf.reduce_sum(tf.gather(similarity_matrix, candidates), axis=-1)

    # 최고의 정렬 선택
    best_alignment = tf.argmax(monotonic_scores, axis=0)

    return best_alignment