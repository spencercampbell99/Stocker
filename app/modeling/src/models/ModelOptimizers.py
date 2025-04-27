def optimize_xgboost_params(data, features, target, objective, verbose=1):
    """
    Optimize XGBoost model parameters using GridSearchCV.

    Args:
        data (pd.DataFrame): The input data for training.
        features (list): List of feature column names.
        target (str): The target column name.
        objective (str): The objective function for the model.
        verbose (int): Verbosity level for GridSearchCV.

    Returns:
        xgb_model: The trained XGBoost model with optimized parameters.
    """
    import xgboost as xgb
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Prepare features and target for XGBoost
    X = data[features]
    y = data[target].astype(int)  # Convert to integer classes

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    xgb_model = xgb.XGBClassifier(objective=objective, random_state=42)

    # Define the parameter grid for optimization
    param_grid = {
        'max_depth': [3, 5, 7],
        'eta': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(xgb_model, param_grid, scoring='accuracy', cv=3, verbose=verbose)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Optimized XGBoost Model Accuracy: {accuracy:.4f}")
    
    return best_model, best_params

def optimize_random_forest_params(data, features, target, verbose=1):
    """
    Optimize Random Forest model parameters using GridSearchCV.

    Args:
        data (pd.DataFrame): The input data for training.
        features (list): List of feature column names.
        target (str): The target column name.
        verbose (int): Verbosity level for GridSearchCV.

    Returns:
        rf_model: The trained Random Forest model with optimized parameters.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Prepare features and target for Random Forest
    X = data[features]
    y = data[target].astype(int)  # Convert to integer classes

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    rf_model = RandomForestClassifier(random_state=42)

    # Define the parameter grid for optimization
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    # Perform GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(rf_model, param_grid, scoring='accuracy', cv=3, verbose=verbose)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Optimized Random Forest Model Accuracy: {accuracy:.4f}")
    
    return best_model, best_params