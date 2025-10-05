import plotly.graph_objects as go
import plotly.express as px


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


def plot_bar_fractions(curr, df):
    number_of_bars = {
        "Monday": 1440,
        "Tuesday": 1440,
        "Wednesday": 1440,
        "Thursday": 1440,
        "Friday": 1320,
        "Saturday": 0,
        "Sunday": 120,
    }
    df_curr = df[df["currency"] == curr].copy()
    df_curr = df_curr.assign(
        date=df_curr["open_time"].dt.date,
        day_of_week=df_curr["open_time"].dt.day_name(),
    )

    bars_per_day = (
        df_curr.groupby(["date", "day_of_week"], as_index=False)
        .size()
        .rename(columns={"size": "bars_count"})
    )

    bars_per_day["expected_bars"] = bars_per_day["day_of_week"].map(number_of_bars)
    bars_per_day["fraction"] = (
        bars_per_day["bars_count"] / bars_per_day["expected_bars"]
    )

    fig = px.bar(
        bars_per_day,
        x="date",
        y="fraction",
        color="day_of_week",
        labels={
            "fraction": "Fraction of expected bars",
            "date": "Date",
            "day_of_week": "Day of Week",
        },
        title=f"Fraction of expected bars per day for {curr}",
    )
    fig.update_layout(xaxis_tickangle=-45, width=1200, height=400)
    fig.show()
