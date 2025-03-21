import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import re
from sklearn.linear_model import LinearRegression
import numpy as np
from prophet import Prophet

# Load the CSV file
df = pd.read_csv("refined_textual_data.csv")

def extract_price(text):
    match = re.search(r'at (\d+\.\d+)', text)
    return float(match.group(1)) if match else None

df['closing_price'] = df['original'].apply(extract_price)
df = df.dropna(subset=['closing_price'])
df['date'] = pd.to_datetime(df['date'])

# Sentiment emojis mapping
sentiment_emojis = {
    "optimistic": "🥰", "worried": "😨", "anxious": "😱",
    "excited": "🚀", "disappointed": "😢", "confident": "😎",
    "indifferent": "😐", "uncertain": "🤔", "calm": "😌"
}

def predict_stock_trend(ticker):
    company_df = df[df['ticker'] == ticker].copy()
    company_df = company_df.sort_values(by='date')
    
    if len(company_df) < 30:
        return "Not enough data"
    
    company_df['days'] = (company_df['date'] - company_df['date'].min()).dt.days
    X = company_df[['days']]
    y = company_df['closing_price']
    
    model = LinearRegression()
    model.fit(X, y)
    future_price = model.predict([[X.max()[0] + 30]])[0]
    
    latest_price = y.iloc[-1]
    if future_price > latest_price * 1.05:
        return "BUY ✅"
    elif future_price < latest_price * 0.95:
        return "SELL ❌"
    else:
        return "HOLD ⏳"

def forecast_stock(ticker, period, unit):
    company_df = df[df['ticker'] == ticker][['date', 'closing_price']].rename(columns={'date': 'forecast_date', 'closing_price': 'y'})
    
    if unit == "Years":
        period *= 365
    elif unit == "Months":
        period *= 30
    
    model = Prophet()
    model.fit(company_df.rename(columns={'forecast_date': 'ds'}))
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'forecast_date', 'yhat': 'predicted_price'})

app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Dropdown(
        id='company-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in df['ticker'].unique()],
        value=df['ticker'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='stock-trend-graph', config={'scrollZoom': True}),
    html.Div(id='selected-point-info'),
    
    html.Div([
        dcc.Dropdown(id='forecast-unit', options=[
            {'label': 'Years', 'value': 'Years'},
            {'label': 'Months', 'value': 'Months'},
            {'label': 'Days', 'value': 'Days'}
        ], value='Days', clearable=False),
        dcc.Input(id='forecast-period', type='number', placeholder='Enter period', min=1),
        html.Button('Show Forecast', id='forecast-btn')
    ]),
    dcc.Graph(id='forecast-graph'),
    html.Div(id='forecast-recommendation', style={'fontSize': '20px', 'marginTop': '20px', 'fontWeight': 'bold'})
])

@app.callback(
    Output('stock-trend-graph', 'figure'),
    Input('company-dropdown', 'value')
)
def update_graph(selected_ticker):
    filtered_df = df[df['ticker'] == selected_ticker]
    fig = px.line(filtered_df, x='date', y='closing_price', title=f'Stock Trend for {selected_ticker}')
    return fig

@app.callback(
    [Output('forecast-graph', 'figure'), Output('forecast-recommendation', 'children')],
    [Input('forecast-btn', 'n_clicks')],
    [dash.State('company-dropdown', 'value'), dash.State('forecast-period', 'value'), dash.State('forecast-unit', 'value')]
)
def show_forecast(n_clicks, selected_ticker, period, unit):
    if not n_clicks or not period:
        return px.scatter(title='No Forecast Yet'), ""
    
    forecast_df = forecast_stock(selected_ticker, period, unit)
    fig = px.line(forecast_df, x='forecast_date', y='predicted_price', title=f'Forecast for {selected_ticker}')
    
    fig.add_scatter(x=forecast_df['forecast_date'], y=forecast_df['predicted_price'], mode='lines', name='Predicted Price', line=dict(color='blue'))
    fig.add_scatter(x=forecast_df['forecast_date'], y=forecast_df['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='green', dash='dot'))
    fig.add_scatter(x=forecast_df['forecast_date'], y=forecast_df['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='red', dash='dot'))
    
    latest_price = df[df['ticker'] == selected_ticker]['closing_price'].iloc[-1]
    forecast_price = forecast_df['predicted_price'].iloc[-1]
    recommendation = ""
    if forecast_price > latest_price * 1.05:
        recommendation = "BUY ✅"
    elif forecast_price < latest_price * 0.95:
        recommendation = "SELL ❌"
    else:
        recommendation = "HOLD ⏳"
    
    return fig, f"Recommendation: {recommendation}"

@app.callback(
    Output('selected-point-info', 'children'),
    Input('stock-trend-graph', 'clickData')
)
def display_sentiment(clickData):
    if clickData is None:
        return "Click on a point to see sentiment info."
    
    clicked_date = clickData['points'][0]['x']
    clicked_price = clickData['points'][0]['y']
    
    row = df[(df['date'] == clicked_date) & (df['closing_price'] == clicked_price)]
    if not row.empty:
        sentiment = row.iloc[0]['emo_label']
        emoji = sentiment_emojis.get(sentiment, "❓")
        return html.Div(f"Sentiment: {sentiment} {emoji}", style={'fontSize': '20px'})
    
    return "No sentiment found for this point."

if __name__ == '__main__':
    app.run_server(debug=True)