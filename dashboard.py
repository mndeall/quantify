import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as optimization
import os

# ── App Setup ─────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Quantify"

# ── Colors ────────────────────────────────────────────────────
C = {
    'bg':      '#050810',
    'surface': '#0d1117',
    'border':  '#1a2332',
    'blue':    '#00d4ff',
    'green':   '#10b981',
    'red':     '#ef4444',
    'amber':   '#f59e0b',
    'purple':  '#7c3aed',
    'muted':   '#64748b',
    'text':    '#e2e8f0',
}

# ── Top 30 S&P 500 stocks used by the screener ────────────────
SP500_TOP30 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL',
    'META', 'TSLA', 'JPM', 'V', 'WMT',
    'XOM', 'UNH', 'MA', 'JNJ', 'PG',
    'HD', 'COST', 'ABBV', 'MRK', 'CVX',
    'BAC', 'KO', 'PEP', 'AVGO', 'LLY',
    'TMO', 'ORCL', 'ACN', 'MCD', 'NKE'
]

# ── Main Layout ───────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),

    # Sidebar is dynamically rebuilt by callback to reflect active page
    html.Div(id='sidebar', style={
        'width': '220px',
        'minHeight': '100vh',
        'background': C['surface'],
        'borderRight': f'1px solid {C["border"]}',
        'padding': '32px 20px',
        'position': 'fixed',
        'top': 0,
        'left': 0,
    }),

    # Main content area — populated by the router callback
    html.Div(
        id='page-content',
        style={
            'marginLeft': '220px',
            'padding': '40px',
            'background': C['bg'],
            'minHeight': '100vh',
        }
    )
], style={'background': C['bg']})


# ── Helpers ───────────────────────────────────────────────────
def make_card(title, value, color=None):
    """
    Returns a styled metric card with a title label and a value.
    Color of the top border and value text reflects the metric status
    (green = good, red = bad, amber = neutral).
    """
    color = color or C['blue']
    return html.Div([
        html.Div(title, style={
            'fontSize': '9px',
            'color': C['muted'],
            'letterSpacing': '0.2em',
            'textTransform': 'uppercase',
            'fontFamily': 'monospace',
            'marginBottom': '8px'
        }),
        html.Div(value, style={
            'fontSize': '24px',
            'color': color,
            'fontFamily': 'monospace',
            'fontWeight': 'bold'
        })
    ], style={
        'background': C['surface'],
        'border': f'1px solid {C["border"]}',
        'borderTop': f'2px solid {color}',
        'padding': '16px 20px',
        'minWidth': '140px',
        'flex': '1'
    })


def dark_layout(title=''):
    """
    Returns a shared Plotly layout dictionary with dark theme styling.
    Applied to all charts for visual consistency across the dashboard.
    """
    return dict(
        title=title,
        paper_bgcolor=C['surface'],
        plot_bgcolor=C['surface'],
        font=dict(color=C['text'], family='monospace'),
        xaxis=dict(gridcolor=C['border'], showgrid=True),
        yaxis=dict(gridcolor=C['border'], showgrid=True),
        hovermode='x unified',
        margin=dict(t=60, b=40, l=60, r=20)
    )


def gap():
    """Returns an empty div used as vertical spacing between charts."""
    return html.Div(style={'marginBottom': '32px'})


def nav_link(label, number, href, active):
    """
    Returns a styled sidebar navigation link.
    Highlights with a blue left border and background when active.
    """
    return dcc.Link([
        html.Span(f"{number} ", style={
            'color': C['blue'] if active else C['muted'],
            'fontFamily': 'monospace',
            'fontSize': '10px'
        }),
        html.Span(label, style={
            'color': C['text'] if active else C['muted'],
            'fontFamily': 'monospace',
            'fontSize': '13px'
        })
    ], href=href, style={
        'display': 'block',
        'textDecoration': 'none',
        'padding': '10px 12px',
        'marginBottom': '4px',
        'borderLeft': f'2px solid {C["blue"] if active else "transparent"}',
        'background': f'{C["blue"]}11' if active else 'transparent',
        'transition': 'all 0.2s'
    })


# ══════════════════════════════════════════════════════════════
# CALLBACK: SIDEBAR
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('sidebar', 'children'),
    Input('url', 'pathname')
)
def update_sidebar(pathname):
    """
    Rebuilds the sidebar on every page navigation.
    Reads the current URL pathname to determine which link is active
    and applies the active highlight style accordingly.
    """
    return [
        html.Div([
            html.Div("📈", style={'fontSize': '28px'}),
            html.Div("QUANTIFY", style={
                'color': C['blue'],
                'fontFamily': 'monospace',
                'fontWeight': 'bold',
                'fontSize': '16px',
                'letterSpacing': '0.2em'
            }),
            html.Div("RESEARCH DASHBOARD", style={
                'color': C['muted'],
                'fontFamily': 'monospace',
                'fontSize': '9px',
                'letterSpacing': '0.3em'
            }),
        ], style={'marginBottom': '40px', 'textAlign': 'center'}),

        html.Div("NAVIGATION", style={
            'color': C['muted'],
            'fontSize': '9px',
            'letterSpacing': '0.3em',
            'fontFamily': 'monospace',
            'marginBottom': '12px'
        }),

        nav_link("Stock Analyzer",      "01", '/',
                 pathname == '/' or pathname == ''),
        nav_link("Stock Screener",      "02", '/screener',
                 pathname == '/screener'),
        nav_link("Portfolio Optimizer", "03", '/optimizer',
                 pathname == '/optimizer'),
    ]


# ══════════════════════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════════════════════
def page_analyzer():
    """
    Layout for Page 1 — Stock Analyzer.
    Contains a ticker input, metric cards, price chart,
    RSI indicator chart, and daily returns bar chart.
    All charts are populated by the update_analyzer callback.
    """
    return html.Div([
        html.Div("01 / STOCK ANALYZER", style={
            'color': C['muted'], 'fontFamily': 'monospace',
            'fontSize': '11px', 'letterSpacing': '0.2em',
            'marginBottom': '8px'
        }),
        html.H2("Analyze Any Stock", style={
            'color': C['text'], 'fontFamily': 'monospace',
            'fontWeight': '800', 'margin': '0 0 24px 0'
        }),

        html.Div([
            html.Label("TICKER", style={
                'color': C['muted'], 'fontFamily': 'monospace',
                'fontSize': '10px', 'letterSpacing': '0.2em',
                'marginRight': '12px'
            }),
            dcc.Input(
                id='ticker-input', value='AAPL',
                type='text', debounce=True,
                style={
                    'background': C['bg'], 'color': 'white',
                    'border': f'1px solid {C["blue"]}',
                    'padding': '10px 16px', 'fontSize': '16px',
                    'fontFamily': 'monospace', 'outline': 'none',
                    'width': '120px'
                }
            ),
            html.Div(id='signal-badge', style={'marginLeft': '16px'})
        ], style={'display': 'flex', 'alignItems': 'center',
                  'marginBottom': '24px'}),

        html.Div(id='stats-cards', style={
            'display': 'flex', 'gap': '8px',
            'marginBottom': '40px', 'flexWrap': 'wrap'
        }),

        dcc.Graph(id='price-chart'),
        gap(),
        dcc.Graph(id='rsi-chart'),
        gap(),
        dcc.Graph(id='returns-chart'),
        gap(),
    ])


def page_screener():
    """
    Layout for Page 2 — Stock Screener.
    Auto-loads on mount using a one-shot dcc.Interval trigger.
    Results are populated by the update_screener callback.
    """
    return html.Div([
        html.Div("02 / STOCK SCREENER", style={
            'color': C['muted'], 'fontFamily': 'monospace',
            'fontSize': '11px', 'letterSpacing': '0.2em',
            'marginBottom': '8px'
        }),
        html.H2("Top S&P 500 Stocks", style={
            'color': C['text'], 'fontFamily': 'monospace',
            'fontWeight': '800', 'margin': '0 0 8px 0'
        }),
        html.P("Top 30 S&P 500 stocks ranked by Sharpe ratio",
               style={
                   'color': C['muted'], 'fontFamily': 'monospace',
                   'fontSize': '12px', 'marginBottom': '32px'
               }),

        html.Div(id='screener-results'),
        dcc.Interval(id='screener-trigger', interval=500, max_intervals=1)
    ])


def page_optimizer():
    """
    Layout for Page 3 — Portfolio Optimizer.
    User enters comma-separated tickers.
    Results are populated by the update_optimizer callback which runs
    Monte Carlo simulation and Markowitz optimization.
    """
    return html.Div([
        html.Div("03 / PORTFOLIO OPTIMIZER", style={
            'color': C['muted'], 'fontFamily': 'monospace',
            'fontSize': '11px', 'letterSpacing': '0.2em',
            'marginBottom': '8px'
        }),
        html.H2("Markowitz Optimizer", style={
            'color': C['text'], 'fontFamily': 'monospace',
            'fontWeight': '800', 'margin': '0 0 8px 0'
        }),
        html.P("Find optimal portfolio weights to maximize Sharpe ratio",
               style={
                   'color': C['muted'], 'fontFamily': 'monospace',
                   'fontSize': '12px', 'marginBottom': '24px'
               }),

        html.Div([
            html.Label("ENTER STOCKS (comma separated)", style={
                'color': C['muted'], 'fontFamily': 'monospace',
                'fontSize': '10px', 'letterSpacing': '0.2em',
                'display': 'block', 'marginBottom': '8px'
            }),
            dcc.Input(
                id='optimizer-input',
                value='AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA',
                type='text', debounce=True,
                style={
                    'background': C['bg'], 'color': 'white',
                    'border': f'1px solid {C["purple"]}',
                    'padding': '10px 16px', 'fontSize': '13px',
                    'fontFamily': 'monospace', 'outline': 'none',
                    'width': '400px'
                }
            ),
        ], style={'marginBottom': '24px'}),

        html.Div(id='optimizer-stats', style={
            'display': 'flex', 'gap': '8px',
            'marginBottom': '40px', 'flexWrap': 'wrap'
        }),

        dcc.Graph(id='frontier-chart'),
        gap(),
        dcc.Graph(id='weights-chart'),
        gap(),
    ])


# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """
    Routes the user to the correct page based on the URL pathname.
    / → Stock Analyzer
    /screener → Stock Screener
    /optimizer → Portfolio Optimizer
    """
    if pathname == '/screener':
        return page_screener()
    elif pathname == '/optimizer':
        return page_optimizer()
    else:
        return page_analyzer()


# ══════════════════════════════════════════════════════════════
# CALLBACK 1 — STOCK ANALYZER
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('stats-cards',   'children'),
    Output('price-chart',   'figure'),
    Output('rsi-chart',     'figure'),
    Output('returns-chart', 'figure'),
    Output('signal-badge',  'children'),
    Input('ticker-input',   'value')
)
def update_analyzer(ticker):
    """
    Triggered when the user types a ticker symbol.
    Downloads 1 year of daily OHLCV data from Yahoo Finance.

    Calculates:
    - Total return, annualized return, annualized volatility
    - Sharpe ratio (return / volatility)
    - Maximum drawdown (worst peak-to-trough decline)
    - RSI (14-day Relative Strength Index)
    - 50-day and 200-day moving averages
    - Buy/Hold/Sell signal based on RSI + price vs MA50

    Returns metric cards, price chart, RSI chart,
    returns bar chart, and a signal badge.
    """
    if not ticker:
        ticker = 'AAPL'
    ticker = ticker.upper().strip()

    try:
        df = yf.download(ticker, period='1y', auto_adjust=True)
    except Exception:
        df = pd.DataFrame()

    if df.empty:
        empty = go.Figure()
        empty.update_layout(**dark_layout(f'No data for {ticker}'))
        return [], empty, empty, empty, ""

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close   = df['Close'].squeeze()
    returns = close.pct_change().dropna()

    # ── Metrics ───────────────────────────────────────────────
    annual_return = returns.mean() * 252
    annual_vol    = returns.std() * np.sqrt(252)
    sharpe        = annual_return / annual_vol
    max_dd        = ((close - close.cummax()) / close.cummax()).min()
    total_return  = (close.iloc[-1] / close.iloc[0]) - 1

    # ── RSI (14-day) ──────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / loss))

    # ── Moving Averages + Signal ──────────────────────────────
    ma50       = close.rolling(50).mean()
    ma200      = close.rolling(200).mean()
    last_rsi   = rsi.iloc[-1]
    last_price = close.iloc[-1]
    last_ma50  = ma50.iloc[-1]

    # Signal logic: BUY if oversold + above MA50, SELL if overbought
    # + below MA50, otherwise HOLD
    if last_rsi < 35 and last_price > last_ma50:
        sig = ("BUY",  C['green'])
    elif last_rsi > 65 and last_price < last_ma50:
        sig = ("SELL", C['red'])
    else:
        sig = ("HOLD", C['amber'])

    signal_badge = html.Div(sig[0], style={
        'background': f'{sig[1]}22',
        'color': sig[1],
        'border': f'1px solid {sig[1]}',
        'padding': '6px 16px',
        'fontFamily': 'monospace',
        'fontSize': '13px',
        'fontWeight': 'bold',
        'letterSpacing': '0.1em'
    })

    # ── Metric Cards ──────────────────────────────────────────
    cards = [
        make_card("Total Return",
                  f"{total_return:+.1%}",
                  C['green'] if total_return > 0 else C['red']),
        make_card("Annual Return",
                  f"{annual_return:+.1%}",
                  C['green'] if annual_return > 0 else C['red']),
        make_card("Volatility",   f"{annual_vol:.1%}",  C['amber']),
        make_card("Sharpe Ratio",
                  f"{sharpe:.2f}",
                  C['green'] if sharpe > 1 else C['amber']),
        make_card("Max Drawdown", f"{max_dd:.1%}",      C['red']),
        make_card("RSI (14)",
                  f"{last_rsi:.1f}",
                  C['red']   if last_rsi > 70
                  else C['green'] if last_rsi < 30
                  else C['amber']),
    ]

    # ── Price Chart with Moving Averages ──────────────────────
    price_fig = go.Figure()
    price_fig.add_trace(go.Scatter(
        x=close.index, y=close, name='Price',
        line=dict(color=C['blue'], width=2)
    ))
    price_fig.add_trace(go.Scatter(
        x=ma50.index, y=ma50, name='MA 50',
        line=dict(color=C['amber'], width=1, dash='dash')
    ))
    price_fig.add_trace(go.Scatter(
        x=ma200.index, y=ma200, name='MA 200',
        line=dict(color='#ec4899', width=1, dash='dot')
    ))
    price_fig.update_layout(
        **dark_layout(f'{ticker} — Price & Moving Averages'),
        legend=dict(
            bgcolor=C['surface'],
            x=0.01, y=0.99,
            xanchor='left', yanchor='top'
        )
    )

    # ── RSI Chart with Overbought/Oversold Lines ──────────────
    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(
        x=rsi.index, y=rsi, name='RSI',
        line=dict(color=C['purple'], width=2)
    ))
    rsi_fig.add_hline(y=70, line_dash='dash', line_color=C['red'],
                      annotation_text='Overbought')
    rsi_fig.add_hline(y=30, line_dash='dash', line_color=C['green'],
                      annotation_text='Oversold')
    rsi_fig.update_layout(
        **dark_layout(f'{ticker} — RSI (14)'),
        showlegend=False
    )
    rsi_fig.update_yaxes(range=[0, 100])

    # ── Daily Returns Bar Chart ───────────────────────────────
    colors = [C['green'] if r > 0 else C['red'] for r in returns]
    returns_fig = go.Figure()
    returns_fig.add_trace(go.Bar(
        x=returns.index, y=returns,
        marker_color=colors, name='Returns'
    ))
    returns_fig.update_layout(
        **dark_layout(f'{ticker} — Daily Returns'),
        showlegend=False
    )
    returns_fig.update_yaxes(tickformat='.1%')

    return cards, price_fig, rsi_fig, returns_fig, signal_badge


# ══════════════════════════════════════════════════════════════
# CALLBACK 2 — STOCK SCREENER
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('screener-results', 'children'),
    Input('screener-trigger',  'n_intervals')
)
def update_screener(n):
    """
    Triggered once automatically when the screener page loads
    via a dcc.Interval with max_intervals=1.

    Downloads 1 year of data for all 30 S&P 500 stocks,
    calculates key metrics for each, sorts by Sharpe ratio,
    and renders a scrollable ranked table.

    Metrics per stock:
    - Annualized return and volatility
    - Sharpe ratio (primary sort key)
    - 20-day price momentum
    - Maximum drawdown
    """
    if n is None:
        return html.P("Loading...",
                      style={'color': C['muted'],
                             'fontFamily': 'monospace'})

    results = []
    for ticker in SP500_TOP30:
        try:
            df = yf.download(ticker, period='1y',
                             auto_adjust=True, progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close   = df['Close'].squeeze()
            returns = close.pct_change().dropna()
            if len(returns) < 20:
                continue

            annual_return = returns.mean() * 252
            annual_vol    = returns.std() * np.sqrt(252)
            sharpe        = annual_return / annual_vol
            momentum      = (close.iloc[-1] / close.iloc[-20]) - 1
            max_dd        = ((close - close.cummax()) /
                             close.cummax()).min()

            results.append({
                'Ticker':         ticker,
                'Annual Return':  f"{annual_return:+.1%}",
                'Volatility':     f"{annual_vol:.1%}",
                'Sharpe':         round(sharpe, 2),
                'Momentum (20d)': f"{momentum:+.1%}",
                'Max Drawdown':   f"{max_dd:.1%}",
                '_sharpe':        sharpe
            })
        except Exception:
            continue

    if not results:
        return html.P("Failed to load.",
                      style={'color': C['muted'],
                             'fontFamily': 'monospace'})

    results = sorted(results, key=lambda x: x['_sharpe'], reverse=True)

    headers = ['Rank', 'Ticker', 'Annual Return',
               'Volatility', 'Sharpe', 'Momentum (20d)', 'Max Drawdown']

    header_row = html.Tr([
        html.Th(h, style={
            'color': C['muted'],
            'fontFamily': 'monospace',
            'fontSize': '10px',
            'letterSpacing': '0.15em',
            'textTransform': 'uppercase',
            'padding': '12px 16px',
            'borderBottom': f'1px solid {C["border"]}',
            'textAlign': 'left',
            'whiteSpace': 'nowrap'
        }) for h in headers
    ])

    rows = []
    for i, r in enumerate(results):
        rank_color = (C['amber'] if i == 0
                      else C['text'] if i < 3
                      else C['muted'])
        row = html.Tr([
            html.Td(f"#{i+1}", style={
                'color': rank_color,
                'fontFamily': 'monospace',
                'padding': '12px 16px',
                'borderBottom': f'1px solid {C["border"]}22',
                'whiteSpace': 'nowrap'
            }),
            html.Td(r['Ticker'], style={
                'color': C['blue'],
                'fontFamily': 'monospace',
                'fontSize': '14px',
                'fontWeight': 'bold',
                'padding': '12px 16px',
                'borderBottom': f'1px solid {C["border"]}22',
                'whiteSpace': 'nowrap'
            }),
            *[html.Td(r[h], style={
                'color': C['text'],
                'fontFamily': 'monospace',
                'fontSize': '13px',
                'padding': '12px 16px',
                'borderBottom': f'1px solid {C["border"]}22',
                'whiteSpace': 'nowrap'
            }) for h in ['Annual Return', 'Volatility', 'Sharpe',
                         'Momentum (20d)', 'Max Drawdown']]
        ], style={
            'background': C['surface'] if i % 2 == 0 else C['bg']
        })
        rows.append(row)

    return html.Div([
        html.P(
            f"{len(results)} stocks screened · Sorted by Sharpe Ratio",
            style={
                'color': C['muted'],
                'fontFamily': 'monospace',
                'fontSize': '11px',
                'marginBottom': '16px'
            }),
        html.Div(
            html.Table(
                [html.Thead(header_row), html.Tbody(rows)],
                style={
                    'width': '100%',
                    'borderCollapse': 'collapse',
                    'border': f'1px solid {C["border"]}'
                }
            ),
            style={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'maxHeight': '70vh',
            }
        )
    ])


# ══════════════════════════════════════════════════════════════
# CALLBACK 3 — PORTFOLIO OPTIMIZER
# ══════════════════════════════════════════════════════════════
@app.callback(
    Output('optimizer-stats', 'children'),
    Output('frontier-chart',  'figure'),
    Output('weights-chart',   'figure'),
    Input('optimizer-input',  'value')
)
def update_optimizer(tickers_str):
    """
    Triggered when the user enters a comma-separated list of tickers.
    Downloads 2 years of data and runs Markowitz portfolio optimization.

    Steps:
    1. Download historical closing prices for all tickers
    2. Calculate log returns and covariance matrix
    3. Run Monte Carlo simulation (5,000 random portfolios)
       to approximate the efficient frontier
    4. Use scipy SLSQP optimization to find the portfolio
       that maximizes the Sharpe ratio
    5. Return metric cards, efficient frontier scatter plot,
       and optimal weights bar chart

    Constraints: weights sum to 1, no short selling (0 <= w <= 1)
    """
    empty = go.Figure()
    empty.update_layout(**dark_layout())

    if not tickers_str:
        return [], empty, empty

    tickers = [t.strip().upper() for t in tickers_str.split(',')]
    if len(tickers) < 2:
        return [], empty, empty

    try:
        data = yf.download(tickers, period='2y',
                           auto_adjust=True,
                           progress=False)['Close']
    except Exception:
        return [], empty, empty

    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()

    if data.empty or len(data) < 50:
        return [], empty, empty

    returns = np.log(data / data.shift(1)).dropna()
    n = len(tickers)

    def port_stats(w):
        """Calculate annualized return, volatility, and Sharpe for weights w."""
        ret = np.sum(returns.mean() * w) * 252
        vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        return ret, vol, ret / vol

    # ── Monte Carlo Simulation ────────────────────────────────
    mc_ret, mc_vol, mc_sharpe = [], [], []
    for _ in range(5000):
        w = np.random.random(n)
        w /= w.sum()
        r, v, s = port_stats(w)
        mc_ret.append(r)
        mc_vol.append(v)
        mc_sharpe.append(s)

    # ── Markowitz Optimization ────────────────────────────────
    result = optimization.minimize(
        fun=lambda w: -port_stats(w)[2],   # minimize negative Sharpe
        x0=np.ones(n) / n,                 # start from equal weights
        method='SLSQP',
        bounds=tuple((0, 1) for _ in range(n)),
        constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )

    opt_ret, opt_vol, opt_sharpe = port_stats(result.x)

    # ── Metric Cards ──────────────────────────────────────────
    cards = [
        make_card("Expected Return",
                  f"{opt_ret:+.1%}",
                  C['green'] if opt_ret > 0 else C['red']),
        make_card("Volatility",   f"{opt_vol:.1%}",  C['amber']),
        make_card("Sharpe Ratio",
                  f"{opt_sharpe:.2f}",
                  C['green'] if opt_sharpe > 1 else C['amber']),
    ]

    # ── Efficient Frontier Chart ──────────────────────────────
    frontier_fig = go.Figure()
    frontier_fig.add_trace(go.Scatter(
        x=mc_vol, y=mc_ret, mode='markers',
        marker=dict(
            color=mc_sharpe,
            colorscale='Viridis',
            size=3, opacity=0.5,
            colorbar=dict(
                title='Sharpe',
                x=0.85,
                thickness=15
            )
        ),
        name='Random Portfolios'
    ))
    frontier_fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret], mode='markers',
        marker=dict(color=C['green'], size=16, symbol='star'),
        name='Optimal'
    ))
    frontier_fig.update_layout(
        **dark_layout('Efficient Frontier — 5,000 Random Portfolios'),
        legend=dict(
            bgcolor=C['surface'],
            x=0.01, y=0.99,
            xanchor='left', yanchor='top'
        )
    )
    frontier_fig.update_layout(
        margin=dict(t=60, b=40, l=60, r=120)
    )

    # ── Optimal Weights Bar Chart ─────────────────────────────
    weights_fig = go.Figure()
    weights_fig.add_trace(go.Bar(
        x=tickers,
        y=result.x,
        marker_color=C['purple'],
        text=[f"{w:.1%}" for w in result.x],
        textposition='outside',
        textfont=dict(color=C['text'], family='monospace')
    ))
    weights_fig.update_layout(
        **dark_layout('Optimal Portfolio Weights'),
        showlegend=False
    )
    weights_fig.update_yaxes(tickformat='.0%')

    return cards, frontier_fig, weights_fig


# ── Run ───────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 8050)),
        debug=False
    )
