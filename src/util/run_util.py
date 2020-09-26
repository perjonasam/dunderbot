import pandas as pd

def create_timedelta_and_plot(*, model, logy=True):
    """
    From model timer attribute, plots time cost of each step, mostly to detect increases over time.
    """
    times = model.env.get_attr('timer')
    times = pd.DataFrame(data=times[0], columns=['timestamp'])
    times['timestamp'] = pd.to_datetime(times['timestamp'])
    times['timestamp_1'] = times['timestamp'].shift(-1)
    times['timedelta'] = times['timestamp'].shift(-1)-times['timestamp']

    times = times.drop(times.loc[times['timedelta']<pd.to_timedelta(0)].index)
    times = times.drop(times.loc[times['timedelta']>pd.to_timedelta('1h')].index)

    times = times.dropna().reset_index(drop=True)

    times['timedelta'].plot(logy=logy)
    #return times