import numpy as np

def cos_sim(v1, v2) -> float:
    """
    cos類似度を求める
    :param v1: ベクトル1
    :param v2: ベクトル2
    :return: cos類似度
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def most_similarity(vector, matrix) -> np.array:
    """
    行列でcos類似度が高い順で出力する
    :param matrix: 行列
    :param vector: 対象ベクトル
    :return: cos類似度のarray,
    """
    cos_sim_array = np.array([cos_sim(vector, v) for v in matrix])
    return cos_sim_array

