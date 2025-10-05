import plotly.graph_objects as go


def plot_candlestick(df, title="Candlestick Chart", width=900, height=500):
    """
    Plots candlestick chart for a given DataFrame.

    Parameters:
        df (pd.DataFrame): Must contain columns ['open_time', 'open', 'high', 'low', 'close'].
        title (str): Chart title.
        width (int): Figure width.
        height (int): Figure height.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["open_time"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="green",
                decreasing_line_color="red",
                name="",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        width=width,
        height=height,
    )

    fig.show()


def plot_volume(df, title="Volume Chart", width=900, height=300):
    """
    Plots volume bar chart for a given DataFrame.

    Parameters:
        df (pd.DataFrame): Must contain columns ['open_time', 'volume'].
        title (str): Chart title.
        width (int): Figure width.
        height (int): Figure height.
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=df["open_time"],
                y=df["volume"],
                marker_color="blue",
                name="Volume",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Volume",
        width=width,
        height=height,
    )

    fig.show()
