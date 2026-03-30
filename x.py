import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np

# ── App Setup ──
app = dash.Dash(__name__)
app.title = "Quant Research Dashboard"

# ── Layout ──
app.layout = html.Div([

    # Header
    html.Div([
        html.H1("📈 Quant Research Dashboard",
                style={'color': '#00d4ff',
                       'fontFamily': 'monospace',
                       'marginBottom': '4px'}),
        html.P("Real-time stock analysis · Screener · Portfolio Optimizer",
               style={'color': '#64748b',
                      'fontFamily': 'monospace',
                      'fontSize': '13px'})
    ], style={'marginBottom': '24px'}),

    # Ticker input
    html.Div([
        html.Label("Stock Ticker",
                   style={'color': '#94a3b8',
                          'fontFamily': 'monospace',
                          'fontSize': '12px',
                          'letterSpacing': '0.1em'}),
        dcc.Input(
            id='ticker-input',
            value='AAPL',
            type='text',
            debounce=True,
            style={
                'background': '#0d1117',
                'color': 'white',
                'border': '1px solid #00d4ff',
                'padding': '10px 16px',
                'fontSize': '16px',
                'fontFamily': 'monospace',
                'marginLeft': '12px',
                'outline': 'none',
                'width': '120px'
            }
        ),
    ], style={'marginBottom': '24px'}),

    # Stats cards row
    html.Div(id='stats-cards',
             style={'display': 'flex',
                    'gap': '12px',
                    'marginBottom': '24px',
                    'flexWrap': 'wrap'}),

    # Price chart
    dcc.Graph(id='price-chart'),

    # Returns chart
    dcc.Graph(id='returns-chart'),

], style={
    'background': '#050810',
    'minHeight': '100vh',
    'padding': '40px',
    'boxSizing': 'border-box'
})


# ── Helper: Stats Card ─────────────────────────────────────────
def make_card(title, value, color='#00d4ff'):
    return html.Div([
        html.Div(title,
                 style={'fontSize': '10px',
                        'color': '#64748b',
                        'letterSpacing': '0.2em',
                        'textTransform': 'uppercase',
                        'fontFamily': 'monospace',
                        'marginBottom': '6px'}),
        html.Div(value,
                 style={'fontSize': '22px',
                        'color': color,
                        'fontFamily': 'monospace',
                        'fontWeight': 'bold'})
    ], style={
        'background': '#0d1117',
        'border': f'1px solid {color}22',
        'borderTop': f'2px solid {color}',
        'padding': '16px 20px',
        'minWidth': '140px'
    })


# ── Callback ──
@app.callback(
    Output('stats-cards', 'children'),
    Output('price-chart', 'figure'),
    Output('returns-chart', 'figure'),
    Input('ticker-input', 'value')
)
def update_dashboard(ticker):
    if not ticker:
        ticker = 'AAPL'

    ticker = ticker.upper().strip()

    # Download 1 year of data
    df = yf.download(ticker, period='1y', auto_adjust=True)

    if df.empty:
        empty = go.Figure()
        return [], empty, empty

    # Flatten columns if MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df['Close'].squeeze()
    returns = close.pct_change().dropna()

    # ── Calculate Stats ───────────────────────────────────────
    annual_return = returns.mean() * 252
    annual_vol    = returns.std() * np.sqrt(252)
    sharpe        = annual_return / annual_vol
    max_dd        = ((close - close.cummax()) / close.cummax()).min()
    total_return  = (close.iloc[-1] / close.iloc[0]) - 1

    # ── Stats Cards ───────────────────────────────────────────
    cards = [
        make_card("Total Return",  f"{total_return:+.1%}",
                  '#10b981' if total_return > 0 else '#ef4444'),
        make_card("Annual Return", f"{annual_return:+.1%}",
                  '#10b981' if annual_return > 0 else '#ef4444'),
        make_card("Volatility",    f"{annual_vol:.1%}",    '#f59e0b'),
        make_card("Sharpe Ratio",  f"{sharpe:.2f}",
                  '#10b981' if sharpe > 1 else '#f59e0b'),
        make_card("Max Drawdown",  f"{max_dd:.1%}",        '#ef4444'),
    ]

    # ── Price Chart ───────────────────────────────────────────
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    price_fig = go.Figure()

    price_fig.add_trace(go.Scatter(
        x=close.index, y=close,
        name='Price',
        line=dict(color='#00d4ff', width=2)
    ))
    price_fig.add_trace(go.Scatter(
        x=ma50.index, y=ma50,
        name='MA 50',
        line=dict(color='#f59e0b', width=1, dash='dash')
    ))
    price_fig.add_trace(go.Scatter(
        x=ma200.index, y=ma200,
        name='MA 200',
        line=dict(color='#ec4899', width=1, dash='dot')
    ))

    price_fig.update_layout(
        title=f'{ticker} — Price & Moving Averages',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='white', family='monospace'),
        xaxis=dict(gridcolor='#1a2332', showgrid=True),
        yaxis=dict(gridcolor='#1a2332', showgrid=True),
        legend=dict(bgcolor='#0d1117'),
        hovermode='x unified'
    )

    # ── Returns Chart ─────────────────────────────────────────
    colors = ['#10b981' if r > 0 else '#ef4444'
              for r in returns]

    returns_fig = go.Figure()
    returns_fig.add_trace(go.Bar(
        x=returns.index,
        y=returns,
        marker_color=colors,
        name='Daily Returns'
    ))

    returns_fig.update_layout(
        title=f'{ticker} — Daily Returns',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        font=dict(color='white', family='monospace'),
        xaxis=dict(gridcolor='#1a2332'),
        yaxis=dict(gridcolor='#1a2332',
                   tickformat='.1%'),
        hovermode='x unified'
    )

    return cards, price_fig, returns_fig


# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)