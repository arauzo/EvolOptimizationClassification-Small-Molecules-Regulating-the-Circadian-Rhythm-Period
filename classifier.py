import statistics

from sklearn.metrics import accuracy_score

def evaluar_clasificador(features, cv_partitions, clf, penalty=0.0, var_penalty=False):
    n = len(features)
    selected = sum(features)
    if (selected <= 0):        # There are fitting algorithms that break if no features are selected
        return (0.0, )

    scores = []
    for partition in cv_partitions:
        X_train, X_test, y_train, y_test = partition
        clf.fit(X_train[:, features], y_train)

        y_pred = clf.predict(X_test[:, features])
        scores.append( accuracy_score(y_test, y_pred, normalize=True) )

    avg_score = statistics.mean(scores)
    if len(scores) > 1:
        var_score = statistics.variance(scores, avg_score)
    else:
        var_score = 0.0
    reduction = (n - selected) / n

    if var_penalty:
        fitness = (1 - penalty) * (avg_score - var_score) + penalty * reduction
    elif penalty:
        fitness = (1 - penalty) * avg_score + penalty * reduction
    else:
        fitness = avg_score

    return ( fitness, )

