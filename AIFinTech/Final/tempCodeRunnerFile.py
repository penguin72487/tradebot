    'KNN': KNeighborsRegressor(
        n_neighbors=7,
        weights='distance',
        algorithm='auto',
        leaf_size=20,
        p=2,
        metric='minkowski'
    ),