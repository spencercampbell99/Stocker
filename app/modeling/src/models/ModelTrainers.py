def train_xgboost_model(data, features, target, objective, params=None, test_data=None):
    """
    Train an XGBoost model on the provided data.
    
    Args:
        data (pd.DataFrame): The input data for training.
        features (list): List of feature column names.
        target (str): The target column name.
        objective (str): The objective function for the model.
        params (dict, optional): Parameters for the XGBoost model. Defaults to None.
        test_data (pd.DataFrame, optional): Data for testing the model. Defaults to None.
        
    Returns:
        xgb_model: The trained XGBoost model.
        scaler: The StandardScaler used for scaling the features.
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features and target for XGBoost
    X = data[features]
    y = data[target].astype(int)  # Convert to integer classes
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective=objective,
        random_state=42,
        **(params if params else {})
    )
    
    # Fit the model
    xgb_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy:.4f}")
    
    # Test against test_data if provided
    if test_data is not None:
        additional_x_test = test_data[features]
        additonal_y_test = test_data[target].astype(int)
        
        additional_x_test_scaled = scaler.transform(additional_x_test)
        additional_test_preds = xgb_model.predict(additional_x_test_scaled)
        accuracy = accuracy_score(additonal_y_test, additional_test_preds)
        
        print(f"\nXGBoost Test Accuracy for 2025: {accuracy:.4f}")
    
    return xgb_model, scaler

def train_random_forest_classifier_model(data, features, target, params=None, test_data=None):
    """
    Train a Random Forest Classifier model on the provided data.
    
    Args:
        data (pd.DataFrame): The input data for training.
        features (list): List of feature column names.
        target (str): The target column name.
        params (dict, optional): Parameters for the Random Forest model. Defaults to None.
        test_data (pd.DataFrame, optional): Data for testing the model. Defaults to None.
        
    Returns:
        rf_model: The trained Random Forest model.
        scaler: The StandardScaler used for scaling the features.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features and target for Random Forest
    X = data[features]
    y = data[target].astype(int)  # Convert to integer classes
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(
        random_state=42,
        **(params if params else {})
    )
    
    # Fit the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba = rf_model.predict_proba(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy:.4f}")
    
    # Test against test_data if provided
    if test_data is not None:
        additional_x_test = test_data[features]
        additonal_y_test = test_data[target].astype(int)
        
        additional_x_test_scaled = scaler.transform(additional_x_test)
        additional_test_preds = rf_model.predict(additional_x_test_scaled)
        accuracy = accuracy_score(additonal_y_test, additional_test_preds)
        
        print(f"\nRandom Forest Test Accuracy for 2025: {accuracy:.4f}")
    
    return rf_model, scaler

def train_tf_model(data, features, target, tf_model=None):
    """
    Train a TensorFlow model on the provided data.
    
    Args:
        data (pd.DataFrame): The input data for training.
        features (list): List of feature column names.
        target (str): The target column name.
        tf_model (tf.keras.Model, optional): Predefined TensorFlow model. Defaults to None.
        
    Returns:
        tf_model: The trained TensorFlow model.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    
    # Prepare features and target for TensorFlow
    X = data[features]
    y = data[target].astype(int)  # Convert to integer classes
    
    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the model if not provided
    if tf_model is None:
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Fit the model
    tf_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    # Make predictions
    y_pred = (tf_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"TensorFlow Model Accuracy: {accuracy:.4f}")
    
    return tf_model, scaler