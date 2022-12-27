import pandas as pd
import numpy as np
import calculate_similarity
from glob import glob
from tqdm import tqdm


def mean_reciprocal_rank(y_true, y_pred):
    """
    MRR(Mean Reciprocal Rank)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1]
    ranking = np.where(y_true_sorted)[0].tolist()
    if ranking:
        return 1 / (ranking[0] + 1)
    return 0


def mean_average_precision(y_true, y_pred):
    """
    MAP(Mean Average Precision)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1]
    ranking = np.where(y_true_sorted)[0]
    return np.mean([(i + 1) / (rank + 1) for i, rank in enumerate(ranking)])


def precision_at_n(y_true, y_pred, n=1):
    """
    Pre@N(Precision@N)を計算する
    :param y_true: 0か1からなるラベルのリスト, example) [1, 0, 0, 1]
    :param y_pred: 降順でランキングが割り当てられる, example) [0.9, 0.3, 0.2, 0.8]
    :return: [0, 1]の範囲の値
    """
    n = min(n, len(y_true))
    sort_idx = np.argsort(y_pred)
    y_true_sorted = y_true[sort_idx][::-1][:n]
    return np.sum(y_true_sorted) / n



def metric_ensemble(encoder):
    """
    ./data/dataset 下のデータの評価を行う．各手法のコサイン類似度の平均でアンサンブルを行う
    :param encoder_list: 対象のEncoder(SWEM, BERT, SBERT)のリスト
    :return:
    """
    mrr = []
    map = []
    precision_1 = []
    precision_5 = []

    for dir in tqdm(glob("./data/dataset/*")):
        with open(f"{dir}/question.txt", "r") as f:
            text = f.read()
        data = pd.read_csv(f"{dir}/data.csv")
        vector = encoder.get_vector(text)
        matrix = encoder.get_matrix(data["question"])
        data["cos_sim"] = calculate_similarity.most_similarity(vector, matrix)
        mrr.append(mean_reciprocal_rank(data["label"], data["cos_sim"]))
        map.append(mean_average_precision(data["label"], data["cos_sim"]))
        precision_1.append(precision_at_n(data["label"], data["cos_sim"], n=1))
        precision_5.append(precision_at_n(data["label"], data["cos_sim"], n=5))
        
        

    print(f"MRR: {np.mean(mrr):.4f}, "
                  f"MAP: {np.mean(map):.4f}, "
                  f"Precision@1: {np.mean(precision_1):.4f}, "
                  f"Precision@5: {np.mean(precision_5):.4f} ")

    return np.mean(mrr), np.mean(map), np.mean(precision_1), np.mean(precision_5)



