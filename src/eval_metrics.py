# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score
#
#
# # 评估指标
# def multiclass_acc(preds, truths):
#     return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
#
#
# def weighted_accuracy(test_preds_emo, test_truth_emo):
#     true_label = (test_truth_emo > 0)
#     predicted_label = (test_preds_emo > 0)
#     tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
#     tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
#     p = float(np.sum(true_label == 1))
#     n = float(np.sum(true_label == 0))
#     return (tp * (n / p) + tn) / (2 * n)
#
#
# def test_score_model(preds, y_test, use_zero=False):
#     non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
#     preds = preds[non_zeros]
#     y_test = y_test[non_zeros]
#     mae = np.mean(np.absolute(preds - y_test))
#     corr = np.corrcoef(preds, y_test)[0][1]
#     preds = preds >= 0
#     y_test = y_test >= 0
#     f_score = f1_score(y_test, preds, average="weighted")
#     acc = accuracy_score(y_test, preds)
#     print("MAE: ", mae)
#     print("Correlation Coefficient: ", corr)
#     print("F1 score: ", f_score)
#     print("Accuracy: ", acc)
#     print("-" * 50)
#     return acc, mae, corr, f_score
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time


def multiclass_acc(preds, truths):
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))
    return (tp * (n / p) + tn) / (2 * n)


class MetricsTracker:
    def __init__(self, sequence_model_name):
        self.sequence_model_name = sequence_model_name
        self.start_time = None
        self.metrics_history = {
            'train_time': 0.0,
            'inference_time': 0.0,
            'mae': [],
            'corr': [],
            'f1': [],
            'acc': []
        }

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self, mode='train'):
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            if mode == 'train':
                self.metrics_history['train_time'] += elapsed_time
            else:
                self.metrics_history['inference_time'] += elapsed_time
            self.start_time = None
            return elapsed_time
        return 0.0


def test_score_model(preds, y_test, use_zero=False, metrics_tracker=None):
    if metrics_tracker:
        metrics_tracker.start_timer()

    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])
    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    binary_preds = preds >= 0
    binary_truth = y_test >= 0
    f_score = f1_score(binary_truth, binary_preds, average="weighted")
    acc = accuracy_score(binary_truth, binary_preds)

    if metrics_tracker:
        metrics_tracker.metrics_history['mae'].append(mae)
        metrics_tracker.metrics_history['corr'].append(corr)
        metrics_tracker.metrics_history['f1'].append(f_score)
        metrics_tracker.metrics_history['acc'].append(acc)
        inference_time = metrics_tracker.stop_timer(mode='inference')
        print(f"Inference Time: {inference_time:.2f}s")

    print(f"MAE: {mae:.4f}")
    print(f"Correlation Coefficient: {corr:.4f}")
    print(f"F1 score: {f_score:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 50)

    return acc, mae, corr, f_score


def compare_sequence_models(results_dict):
    print("\nModel Comparison:")
    print("-" * 100)
    header = "Model Name      | Accuracy | MAE     | Correlation | F1 Score | Train Time | Inference Time"
    print(header)
    print("-" * 100)

    for model_name, metrics in results_dict.items():
        acc = metrics.metrics_history['acc'][-1]
        mae = metrics.metrics_history['mae'][-1]
        corr = metrics.metrics_history['corr'][-1]
        f1 = metrics.metrics_history['f1'][-1]
        train_time = metrics.metrics_history['train_time']
        inference_time = metrics.metrics_history['inference_time']

        print(
            f"{model_name:<14} | {acc:8.4f} | {mae:7.4f} | {corr:10.4f} | {f1:8.4f} | {train_time:9.2f}s | {inference_time:13.2f}s")

    print("-" * 100)


# 使用示例
'''
# 在训练循环中使用
metrics_trackers = {
    'bilstm': MetricsTracker('BiLSTM'),
    'gru': MetricsTracker('GRU'),
    'transformer': MetricsTracker('Transformer'),
    'tcn': MetricsTracker('TCN')
}

for model_name, model in models.items():
    tracker = metrics_trackers[model_name]

    # 训练开始计时
    tracker.start_timer()

    # 训练过程
    ...

    # 训练结束记录时间
    tracker.stop_timer(mode='train')

    # 评估
    results = test_score_model(predictions, ground_truth, metrics_tracker=tracker)

# 比较不同模型
compare_sequence_models(metrics_trackers)
'''