from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_true):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

def bulk_evaluate(models, X_tests, y_trues):
    """Evaluate all models for multiple users."""
    results = []
    for user_id, model in models.items():
        precision, recall, f1 = evaluate_model(model, X_tests[user_id], y_trues[user_id])
        results.append((user_id, precision, recall, f1))
    return results
