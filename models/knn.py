import numpy as np

def predict_knn(X_train, y_train, X_query, k, p):
    D = np.sum(np.abs(X_train - X_query)**p, axis=1) ** (1/p)
    sort_idx = np.argsort(D)
    types_sorted = y_train[sort_idx][:k]
    types, counts = np.unique(types_sorted, return_counts=True)
    tied_indices = np.where(counts==counts.max())[0]

    if tied_indices.size == 1:
        return types[tied_indices[0]]
    
    else:
        tied_types = types[tied_indices]
        mask = np.isin(types_sorted, tied_types)
        type_idx = np.argmax(mask)
        return types_sorted[type_idx]