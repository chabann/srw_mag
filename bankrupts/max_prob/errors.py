from bankrupts.max_prob.scoring import Scoring
from sklearn.metrics import roc_auc_score


class Errors:
    """
    Класс для подсчета показателей качества полученных регрессионных моделей
    """

    @staticmethod
    def pseudo_r2(ln_max_prob, ln_max_prob_trivial, example_num):
        """
        Вычисляет альтренативный коэффициент псевдо R^2
        """
        value = 1 - (1 / (1 + 2 * (ln_max_prob - ln_max_prob_trivial) / example_num))
        return value

    @staticmethod
    def mcfadden_r2(ln_l1, ln_l0):
        """
         Вычисляет альтернативный коэффициент R^2 - коэффициент R^2 МакФаддена
        """
        value = 1 - (ln_l1 / ln_l0)
        return value

    @staticmethod
    def r2_predict(y_predict, y_true):
        """
         Вычисляет коэффициент R^2
        """
        n = len(y_true)
        v_wrong_trivial = sum(y_true) / n

        v_wrong_predict = 0
        for i in range(n):
            v_wrong_predict += (y_true[i] - y_predict[i]) ** 2

        v_wrong_predict /= n

        value = 1 - (v_wrong_predict / v_wrong_trivial)
        return value

    @staticmethod
    def quality_metrics(y_predicted, y_true):
        scoring = Scoring(y_predicted, y_true, True)
        print('Precision', scoring.precision())
        print('Accuracy', scoring.accuracy())
        print('Recall', scoring.recall())
        print('F1-score', scoring.f1_score())
        print('AUC-score', roc_auc_score(y_predicted, y_true))
        print('')
