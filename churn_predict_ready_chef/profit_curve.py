from __future__ import division
from sklearn.metrics import confusion_matrix
from util import *
import matplotlib.pyplot as plt

def confusion_rates(cm):

    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    N = fp + tn
    P = tp + fn

    tpr = tp / P
    fpr = fp / P
    fnr = fn / N
    tnr = tn / N

    rates = np.array([[tpr, fpr], [fnr, tnr]])

    return rates


def profit_curve(classifiers):
    for clf_class in classifiers:
        name, clf_class = clf_class[0], clf_class[1]
        clf = clf_class()
        fit = clf.fit(X[train_index], y[train_index])
        probabilities = np.array([prob[0] for prob in fit.predict_proba(X[test_index])])
        profit = []

        indicies = np.argsort(probabilities)[::1]
        print len(indicies)
        for idx in xrange(len(indicies)):
            pred_true = indicies[:idx]
            ctr = np.arange(indicies.shape[0])
            masked_prediction = np.in1d(ctr, pred_true)
            cm = confusion_matrix(y_test.astype(int), masked_prediction.astype(int))
            rates = confusion_rates(cm)

            profit.append(np.sum(np.multiply(rates,cb)))

        plt.plot((np.arange(len(indicies)) / len(indicies) * 100), profit, label=name)
    plt.legend(loc="lower right")
    plt.title("Profits of classifiers")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.ylim(20)
    plt.show()
