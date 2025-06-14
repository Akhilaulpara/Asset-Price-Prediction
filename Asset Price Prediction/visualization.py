import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_historical_data(historical_data, future_dates, predictions, lower_bound, upper_bound):
    """
    Plot historical data with predictions and confidence intervals
    
    Args:
        historical_data (pandas.DataFrame): DataFrame with historical price data
        future_dates (list): List of future dates for predictions
        predictions (numpy.ndarray): Array of predicted prices
        lower_bound (numpy.ndarray): Lower bound of confidence interval
        upper_bound (numpy.ndarray): Upper bound of confidence interval
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create figure with secondary y-axis for volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add historical price trace
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['close'],
            mode='lines',
            name='Historical Price',
            line=dict(color='blue')
        ),
        secondary_y=False
    )
    
    # Add volume as bar chart on secondary y-axis
    if 'volume' in historical_data.columns:
        fig.add_trace(
            go.Bar(
                x=historical_data.index,
                y=historical_data['volume'],
                name='Volume',
                marker=dict(color='rgba(200, 200, 200, 0.5)')
            ),
            secondary_y=True
        )
    
    # Add prediction trace
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        ),
        secondary_y=False
    )
    
    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Confidence Interval'
        ),
        secondary_y=False
    )
    
    # Update layout
    fig.update_layout(
        title='Price History & Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white'
    )
    
    # Update secondary y-axis
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    fig.update_yaxes(title_text="Price", secondary_y=False)
    
    return fig

def plot_risk_metrics_gauge(value, title, max_value=100, suffix="%"):
    """
    Create a gauge chart for risk metrics
    
    Args:
        value (float): Value to display
        title (str): Title of the gauge
        max_value (float): Maximum value for the gauge
        suffix (str): Suffix to add to the value (e.g., "%")
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Determine colors based on value
    if title == "Sharpe Ratio":
        # For Sharpe Ratio, higher is better
        if value < 1:
            color = "red"
        elif value < 2:
            color = "orange"
        else:
            color = "green"
    else:
        # For volatility, VaR, and drawdown, lower is better
        if value < max_value * 0.3:
            color = "green"
        elif value < max_value * 0.7:
            color = "orange"
        else:
            color = "red"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        number={"suffix": suffix, "font": {"size": 24}},
        gauge={
            "axis": {"range": [None, max_value]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, max_value * 0.3], "color": "lightgreen"},
                {"range": [max_value * 0.3, max_value * 0.7], "color": "lightyellow"},
                {"range": [max_value * 0.7, max_value], "color": "lightcoral"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    
    return fig


def plot_predictions_with_confidence(historical_data, future_dates, predictions, lower_bound, upper_bound):
    """
    Plot predictions with confidence intervals
    
    Args:
        historical_data (pandas.DataFrame): DataFrame with historical price data
        future_dates (list): List of future dates for predictions
        predictions (numpy.ndarray): Array of predicted prices
        lower_bound (numpy.ndarray): Lower bound of confidence interval
        upper_bound (numpy.ndarray): Upper bound of confidence interval
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure
    """
    # Create figure
    fig = go.Figure()
    
    # Add prediction trace
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Add confidence interval as a filled area
    fig.add_trace(
        go.Scatter(
            x=future_dates + future_dates[::-1],
            y=list(upper_bound) + list(lower_bound)[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.1)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            name='Confidence Interval'
        )
    )
    
    # Update layout
    fig.update_layout(
        title='Price Predictions with Confidence Intervals',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white'
    )
    
    return fig
