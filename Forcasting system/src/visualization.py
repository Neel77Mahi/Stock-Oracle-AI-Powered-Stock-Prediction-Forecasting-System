import plotly.graph_objects as go
import pandas as pd
import numpy as np

def get_chart_types():
    """
    Return available chart types for display
    
    Note: We're only using line charts now as requested
    """
    return {
        "Line Chart": "line"
    }

def get_sentiment_emoji(sentiment):
    """
    Map sentiment labels to appropriate emojis
    """
    emoji_map = {
        'anxious': '😰',
        'confident': '😊',
        'disappointed': '😞',
        'excited': '😃',
        'indifferent': '😐',
        'optimistic': '😀',
        'uncertain': '🤔',
        'worried': '😟'
    }
    return emoji_map.get(sentiment, '❓')

def display_stock_chart(data, chart_type='line', ticker='', is_forecast=False):
    """
    Create an interactive stock chart based on selected chart type
    """
    # Define common layout settings
    layout = go.Layout(
        title=f"{ticker} {'Forecast' if is_forecast else 'Historical'} Prices",
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
        hovermode='closest',
        height=500
    )
    
    # Create appropriate chart based on type - using only line chart as requested
    fig = go.Figure(layout=layout)
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=data['date'],
            y=data['close'],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
        )
    )
    
    # Add clickable points for sentiment info - with custom hover behavior
    if not is_forecast and 'emo_label' in data.columns:
        # We'll make the entire line clickable to show sentiment
        # But hide specific sentiment markers until hover/click
        
        # Create emoji list mapped to sentiment
        emojis = [get_sentiment_emoji(sentiment) for sentiment in data['emo_label']]
        sentiments = data['emo_label'].tolist()
        
        # Add invisible markers layer that only show on hover/click
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='rgba(0, 0, 0, 0)',  # Transparent markers
                    symbol='circle',
                    line=dict(width=0)  # No outline
                ),
                hoverinfo='x+y+text',
                name='Click for Sentiment',
                text=[f"{emoji} {sentiment}" for emoji, sentiment in zip(emojis, sentiments)],
                hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<br>Sentiment: %{text}<extra></extra>'
            )
        )
        
        # Add custom interaction instructions
        fig.update_layout(
            annotations=[
                dict(
                    x=0.5,
                    y=1.05,
                    xref="paper",
                    yref="paper",
                    text="Hover on any point to see sentiment information",
                    showarrow=False,
                    font=dict(size=10, color="gray")
                )
            ]
        )
                
    elif chart_type == 'candlestick':
        # For a proper candlestick, we need OHLC data
        # If we only have close prices, we'll estimate the others
        if 'open' not in data.columns:
            # Estimate OHLC from available data
            data['open'] = data['close'].shift(1)
            data['high'] = data['close'] * 1.01  # Estimate 1% higher
            data['low'] = data['close'] * 0.99   # Estimate 1% lower
            # Fill first row's open with close
            data['open'].iloc[0] = data['close'].iloc[0]
        
        fig = go.Figure(
            go.Candlestick(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                hovertemplate='%{x}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>'
            ),
            layout=layout
        )
        
    elif chart_type == 'ohlc':
        # Similar to candlestick, estimate OHLC if needed
        if 'open' not in data.columns:
            data['open'] = data['close'].shift(1)
            data['high'] = data['close'] * 1.01
            data['low'] = data['close'] * 0.99
            data['open'].iloc[0] = data['close'].iloc[0]
        
        fig = go.Figure(
            go.Ohlc(
                x=data['date'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                increasing_line_color='green',
                decreasing_line_color='red',
                hovertemplate='%{x}<br>Open: $%{open:.2f}<br>High: $%{high:.2f}<br>Low: $%{low:.2f}<br>Close: $%{close:.2f}<extra></extra>'
            ),
            layout=layout
        )
        
    elif chart_type == 'area':
        fig = go.Figure(layout=layout)
        
        fig.add_trace(
            go.Scatter(
                x=data['date'],
                y=data['close'],
                mode='lines',
                fill='tozeroy',
                name='Price',
                line=dict(color='blue', width=2),
                fillcolor='rgba(0, 100, 255, 0.2)',
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            )
        )
        
    elif chart_type == 'bar':
        fig = go.Figure(layout=layout)
        
        fig.add_trace(
            go.Bar(
                x=data['date'],
                y=data['close'],
                marker_color='blue',
                name='Price',
                hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
            )
        )
    
    # Add forecast indication if this is a forecast chart
    if is_forecast:
        # Find where historical data ends and forecast begins
        if 'forecast' in data.columns:
            historical = data[data['forecast'] == False]
            forecast = data[data['forecast'] == True]
            
            # Add vertical line to indicate forecast start
            if not historical.empty and not forecast.empty:
                # Get the first date of the forecast
                forecast_start_date = forecast['date'].min()
                
                # Convert to string format to avoid timestamp error
                forecast_start_str = forecast_start_date.strftime('%Y-%m-%d')
                
                # Add shapes to indicate forecast start instead of using add_vline
                fig.update_layout(
                    shapes=[
                        # Vertical line
                        dict(
                            type="line",
                            xref="x",
                            yref="paper",
                            x0=forecast_start_str,
                            y0=0,
                            x1=forecast_start_str,
                            y1=1,
                            line=dict(
                                color="red",
                                width=2,
                                dash="dash",
                            )
                        )
                    ],
                    annotations=[
                        dict(
                            x=forecast_start_str,
                            y=1.0,
                            xref="x",
                            yref="paper",
                            text="Forecast Start",
                            showarrow=False,
                            xanchor="left",
                            font=dict(color="red")
                        )
                    ]
                )
    
    # Configure the legend and other chart properties
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig
