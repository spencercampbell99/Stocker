#!/usr/bin/env python
"""
UpDownTfModelEval.py

Script for evaluating the TfUpDownModel_v0 model on testing data.
This script:
1. Loads the saved model and its metadata
2. Fetches testing pre-market data
3. Generates predictions and evaluates performance
4. Simulates trading performance using the model's predictions
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to the path for imports
sys.path.append(str(Path(__file__).parents[3]))

# Import the data handler
from models.DataHandler import get_up_down_percent_model_data


def load_model_and_metadata(model_version="v0"):
    """
    Load the saved TensorFlow model and its metadata
    
    Args:
        model_version: Version tag of the model to load
        
    Returns:
        tuple: (model, metadata_dict)
    """
    print(f"Loading TfUpDownModel_{model_version} and metadata...")
    
    # Define paths to model and metadata files
    model_dir = Path(__file__).parents[3] / "saved_models"
    model_path = model_dir / f"TfUpDownModel_{model_version}.h5"
    metadata_path = model_dir / f"TfUpDownModel_{model_version}_metadata.pkl"
    
    # Check if files exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    # Load the TensorFlow model
    model = tf.keras.models.load_model(str(model_path))
    
    # Load the metadata
    metadata = joblib.load(str(metadata_path))
    
    return model, metadata


def get_testing_data(metadata):
    """
    Get the data for model evaluation using parameters from metadata
    
    Args:
        metadata: Dictionary containing model metadata including thresholds
        
    Returns:
        pd.DataFrame: data for evaluation
    """
    print("Fetching testing data...")
    
    # Extract parameters from metadata
    ticker = metadata['metadata']['ticker']
    up_threshold = metadata['thresholds']['up_threshold']
    down_threshold = metadata['thresholds']['down_threshold']
    
    # Get data using the DataHandler
    data = get_up_down_percent_model_data(
        start_date="2024-01-01",
        ticker=ticker,
        up_threshold=up_threshold,
        down_threshold=down_threshold
    )
    
    return data


def evaluate_model(model, data, features, scaler):
    """
    Evaluate the model on the given data
    
    Args:
        model: Trained TensorFlow model
        data: DataFrame containing features and target
        features: List of feature names
        scaler: Fitted scaler for preprocessing features
        
    Returns:
        dict: Evaluation metrics and predictions
    """
    print("Evaluating model on testing data...")
    
    # Prepare features and scale them
    X = data[features]
    y_true = data['move_status']
    
    # Apply scaling
    X_scaled = scaler.transform(X)
    
    # Get model predictions
    y_pred_proba = model.predict(X_scaled, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(data.head())
    
    # Create a copy of data with date as a column if it's in the index
    eval_data = data.copy()
    if 'date' not in eval_data.columns and eval_data.index.name == 'date':
        eval_data = eval_data.reset_index()
    
    # Return results
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'extra_eval_data': eval_data[['date', 'high', 'low', 'open']],
    }


def simulate_trading(results, initial_balance=1000.0, daily_risk=1):
    """
    Simulate trading performance based on model predictions
    
    Trading rules:
    - Start with initial_balance
    - Risk daily_risk% of current balance on each trade
    - Fixed 20% stop loss on trade amount
    - Fixed 20% profit target when prediction is correct (for up/down, not neutral)
    - For neutral outcomes, calculate based on day's high/low price action
    
    Args:
        results: Dictionary containing evaluation metrics and predictions
        initial_balance: Starting balance for the simulation
        daily_risk: Percentage of account to risk per trade as decimal (e.g., 0.25 for 25%)
        
    Returns:
        dict: Trading simulation results
    """
    print("\nSimulating trading performance...")
    
    # Extract predictions and actual outcomes
    y_pred = results['y_pred']
    y_true = results['y_true']
    dates = results['extra_eval_data']['date']
    
    # Initialize tracking variables
    balance = initial_balance
    balances = [balance]
    trades = []
    win_count = 0
    loss_count = 0
    neutral_count = 0
    
    # Define trading rules
    profit_pct = 0.25
    loss_pct = -0.25
    
    def calculate_neutral_loss_pct(day_high_pct, day_low_pct):
        """
        If actual result is flat/neutral (1), calculate gain/loss percentage based on day's high/low
        
        Args:
            day_high_pct: High price of the day, pct of open
            day_low_pct: Low price of the day, pct of open
        """
        if day_high_pct > 1.0035:
            return 0.15
        elif day_high_pct > 1.002:
            return 0.1
        
        if day_low_pct < 0.9975:
            return -0.15
        elif day_low_pct < 0.998:
            return -0.1
        return 0.0
    
    # Simulate trading for each day
    for i in range(len(y_pred)):
        date = dates[i]
        prediction = int(y_pred[i])  # 0: Down, 1: Flat, 2: Up
        actual = int(y_true.iloc[i])
        
        # Calculate trade outcome percentage on the trade amount
        trade_pnl_pct = 0
        trade_result = "NEUTRAL"
        
        if prediction == 1:
            # If prediction is neutral, skip the trade
            continue
        
        # If actual is neutral (1), apply neutral outcome calculation
        if actual == 1:
            day_high_pct = results['extra_eval_data']['high'].iloc[i] / results['extra_eval_data']['open'].iloc[i]
            day_low_pct = results['extra_eval_data']['low'].iloc[i] / results['extra_eval_data']['open'].iloc[i]
            trade_pnl_pct = calculate_neutral_loss_pct(day_high_pct, day_low_pct)
            if trade_pnl_pct > 0:
                trade_result = "WIN"
                win_count += 1
            elif trade_pnl_pct < 0:
                trade_result = "LOSS"
                loss_count += 1
            neutral_count += 1
        else:
            # If predicted correctly (both up or both down)
            if prediction == actual:
                trade_pnl_pct = profit_pct
                trade_result = "WIN"
                win_count += 1
            else:
                trade_pnl_pct = loss_pct
                trade_result = "LOSS"
                loss_count += 1
        
        # Calculate the amount placed on the trade
        trade_amount = balance * daily_risk
        
        # Cut stop loss in half if prediction is neutral
        if prediction == 1 and (actual == 0 or actual == 2):
            trade_pnl_pct /= 2
        
        # Calculate P&L in dollars based on the trade amount
        trade_pnl = trade_amount * trade_pnl_pct
        
        # Update balance
        balance += trade_pnl
        
        # Record trade
        trades.append({
            'date': date,
            'prediction': prediction,
            'actual': actual,
            'result': trade_result,
            'trade_amount': trade_amount,
            'pnl_percent': trade_pnl_pct * 100,
            'pnl_dollar': trade_pnl,
            'balance': balance
        })
        
        # Record balance history
        balances.append(balance)
    
    # Calculate overall performance metrics
    total_trades = len(trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    return {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': (balance / initial_balance - 1) * 100,
        'trades': trades,
        'balances': balances,
        'win_count': win_count,
        'loss_count': loss_count,
        'neutral_count': neutral_count,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'daily_risk': daily_risk
    }


def plot_trading_results(trading_results, save_path=None):
    """
    Plot the trading simulation results
    
    Args:
        trading_results: Dictionary containing trading simulation results
        save_path: Path to save the plot (optional)
    """
    balances = trading_results['balances']
    trades = trading_results['trades']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot balance over time
    ax1.plot(range(len(balances)), balances, 'b-', linewidth=2)
    ax1.set_title('Trading Performance Simulation', fontsize=16)
    ax1.set_ylabel('Account Balance ($)', fontsize=12)
    ax1.set_xlim(0, len(balances)-1)
    ax1.grid(True)
    
    # Annotate start and end balance
    ax1.annotate(f"Start: ${trading_results['initial_balance']:.2f}", 
                xy=(0, trading_results['initial_balance']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='green')
    
    ax1.annotate(f"End: ${trading_results['final_balance']:.2f}", 
                xy=(len(balances)-1, trading_results['final_balance']),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color='blue')
    
    # Add horizontal line at initial balance
    ax1.axhline(y=trading_results['initial_balance'], color='r', linestyle='--', alpha=0.5)
    
    # Plot individual trade results
    trade_dates = range(len(trades))
    pnl_values = [trade['pnl_dollar'] for trade in trades]
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
    
    ax2.bar(trade_dates, pnl_values, color=colors, width=0.8)
    ax2.set_xlabel('Trade Number', fontsize=12)
    ax2.set_ylabel('Trade P&L ($)', fontsize=12)
    ax2.set_xlim(-0.5, len(trades)-0.5)
    ax2.grid(True, axis='y')
    
    # Add overall stats to the plot
    stats_text = (
        f"Total Return: {trading_results['total_return']:.2f}%\n"
        f"Win Rate: {trading_results['win_rate']:.2%}\n"
        f"Total Trades: {trading_results['total_trades']}\n"
        f"Wins: {trading_results['win_count']}\n"
        f"Losses: {trading_results['loss_count']}\n"
        f"Neutral: {trading_results['neutral_count']}"
    )
    
    # Position the text box in figure coords
    plt.figtext(0.15, 0.01, stats_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def display_results(results, trading_results=None):
    """
    Display the evaluation results
    
    Args:
        results: Dictionary containing evaluation metrics
        trading_results: Dictionary containing trading simulation results (optional)
    """
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS FOR TESTING DATA")
    print("=" * 50)
    
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    
    print("\nClassification Report:")
    report = results['classification_report']
    for cls in sorted(report.keys()):
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Class {cls} (Support: {report[cls]['support']})")
            print(f"  Precision: {report[cls]['precision']:.4f}")
            print(f"  Recall:    {report[cls]['recall']:.4f}")
            print(f"  F1-Score:  {report[cls]['f1-score']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Display sample predictions
    print("\nSample Predictions (First 10 days):")
    print("Date       | Predicted | Actual | Confidence")
    print("-" * 45)
    
    class_names = {
        0: "Down ",
        1: "Flat ",
        2: "Up   "
    }
    
    for i in range(min(10, len(results['y_pred']))):
        date = results['extra_eval_data']['date'][i]
        pred = results['y_pred'][i]
        true = results['y_true'].iloc[i]
        conf = results['y_pred_proba'][i][pred]
        
        print(f"{date} | {class_names[pred]} | {class_names[true]} | {conf:.4f}")
    
    # Summary of model performance
    correct = sum(results['y_pred'] == results['y_true'])
    total = len(results['y_true'])
    
    print("\nSummary:")
    print(f"Correctly predicted {correct} out of {total} days in testing data")
    print(f"Accuracy: {correct/total:.2%}")
    
    # Display trading results if available
    if trading_results:
        print("\n" + "=" * 50)
        print(f"TRADING SIMULATION RESULTS")
        print("=" * 50)
        
        print(f"\nInitial Balance: ${trading_results['initial_balance']:.2f}")
        print(f"Final Balance: ${trading_results['final_balance']:.2f}")
        print(f"Total Return: {trading_results['total_return']:.2f}%")
        print(f"Total Trades: {trading_results['total_trades']}")
        print(f"Win Rate: {trading_results['win_rate']:.2%}")
        print(f"Wins: {trading_results['win_count']}")
        print(f"Losses: {trading_results['loss_count']}")
        print(f"Neutral Trades: {trading_results['neutral_count']}")
        
        # Show sample of trades
        print("\nSample of Recent Trades:")
        print("Date       | Prediction | Actual | Result  | P&L %   | P&L $    | Balance")
        print("-" * 85)
        
        for trade in trading_results['trades'][-10:]:  # Show the last 10 trades
            print(f"{trade['date']} | {class_names[trade['prediction']]} | {class_names[trade['actual']]} | {trade['result']:<7} | {trade['pnl_percent']:7.2f}% | ${trade['pnl_dollar']:8.2f} | ${trade['balance']:8.2f}")

def save_trading_results_to_table(trading_results, table_name="trading_results"):
    """
    Save the trading results to a postgres table
    
    Args:
        trading_results: Dictionary containing trading simulation results
        table_name: Name of the table to save results to
    """
    from db.database import get_session
    from sqlalchemy import text
    session = get_session()
    
    # Create table if it doesn't exist
    session.execute(text(f"""DROP TABLE IF EXISTS {table_name}"""))
    create_table_query = f"""
    CREATE TABLE {table_name} (
        date DATE,
        prediction INT,
        actual INT,
        result VARCHAR(10),
        trade_amount FLOAT,
        pnl_percent FLOAT,
        pnl_dollar FLOAT,
        balance FLOAT
    )
    """
    session.execute(text(create_table_query))
    
    # insert data into the table with pandas
    import pandas as pd
    
    trades_df = pd.DataFrame(trading_results['trades'])
    trades_df['date'] = pd.to_datetime(trades_df['date'])
    trades_df.to_sql(table_name, session.connection(), if_exists='append', index=False)
    
    session.connection().commit()
    session.connection().close()
    

def main():
    """Main function"""
    try:
        # Load model and metadata
        model, metadata = load_model_and_metadata(model_version="v0.1")
        
        # Extract key information from metadata
        features = metadata['features']
        scaler = metadata['scaler']
        ticker = metadata['metadata']['ticker']
        model_version = metadata['metadata']['model_version']
        up_threshold = metadata['thresholds']['up_threshold']
        down_threshold = metadata['thresholds']['down_threshold']
        
        print(f"\nModel information:")
        print(f"- Version: {model_version}")
        print(f"- Ticker: {ticker}")
        print(f"- Threshold settings: Up > {up_threshold:.4f}, Down < {down_threshold:.4f}")
        print(f"- Features: {', '.join(features)}")
        
        # Get testing data
        testing_data = get_testing_data(metadata)
        print(f"Loaded {len(testing_data)} days of testing data")
        
        # Evaluate model
        results = evaluate_model(model, testing_data, features, scaler)
        
        # Simulate trading
        initial_balance = 1000.0
        trading_results = simulate_trading(results, initial_balance, daily_risk=0.25)
        
        # Display results
        display_results(results, trading_results)
        
        # Plot trading results
        plot_path = Path(__file__).parent / "trading_simulation.png"
        plot_trading_results(trading_results, save_path=str(plot_path))
        print(f"\nTrading simulation plot saved to: {plot_path}")
        
        # If user wants to save to table, have them enter table name
        save_to_table = input("Do you want to save the trading results to a table? (y/n): ").strip().lower()
        if save_to_table == 'y':
            table_name = input("Enter table name (default: trading_results): ").strip() or "trading_results"
            save_trading_results_to_table(trading_results, table_name=table_name)
            print(f"Trading results saved to table: {table_name}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()