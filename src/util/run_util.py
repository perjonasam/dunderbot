import pandas as pd
import os

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


def retrieve_model_dir(*, first):
    """ Retrieving save dir for model and metadata. Everything is unique only locally.
    Structure is folder in ./data/models where name is incremental number. 
    Makes no checks of correct structure.

    Also being called from in rendering methods.
    """

    base_dir = './data/models/'
    dirlist = os.listdir(base_dir)
    dirlist = [int(content) for content in dirlist if os.path.isdir(base_dir + content)]
    dirlist.sort()
    highest_existing_increment = dirlist[-1] if len(dirlist) > 0 else '0'
    if first:
        model_dir = base_dir + str(int(highest_existing_increment)+1) + '/'
        os.mkdir(model_dir)
    else:
        model_dir = base_dir + str(highest_existing_increment) + '/'
    return model_dir
