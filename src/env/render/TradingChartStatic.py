import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import style
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

from src.util.config import get_config
config = get_config()

style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33


class TradingChartStatic:
    """An OHLCV trading visualization using matplotlib made to render gym environments"""

    def __init__(self, df):
        self.df = df


    def _render_net_worth(self, step_range, times, current_step, net_worths, benchmarks):
        # Clear the frame rendered last step
        #self.net_worth_ax.clear()
        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2, colspan=1)
        # Plot net worths
        self.net_worth_ax.plot(times, net_worths[step_range], label='Net Worth', color="g")

        #self._render_benchmarks(step_range, times, benchmarks)

        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

        # Add space above and below min/max net worth
        #self.net_worth_ax.set_ylim(min(net_worths) / 1.25, max(net_worths) * 1.25)


    def _render_benchmarks(self, step_range, times, benchmarks):
        colors = ['orange', 'cyan', 'purple', 'blue',
                  'magenta', 'yellow', 'black', 'red', 'green']

        for i, benchmark in enumerate(benchmarks):
            self.net_worth_ax.plot(times, benchmark['values'][step_range],
                                   label=benchmark['label'], color=colors[i % len(colors)], alpha=0.3)


    def _render_price(self, step_range, times, current_step):
        #self.price_ax.clear()
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=8, colspan=1, sharex=self.net_worth_ax)
        
        # Plot assetprice
        self.price_ax.plot(times, self.df['Close'].values[step_range], color="black")

        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * VOLUME_CHART_HEIGHT, ylim[1])


    def _render_volume(self, step_range, times):
        #self.volume_ax.clear()
        self.volume_ax = self.price_ax.twinx()
        volume = np.array(self.df['VolumeUSD'].values[step_range])

        self.volume_ax.plot(times, volume,  color='blue')
        self.volume_ax.fill_between(times, volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])


    def _render_trades(self, step_range, trades):
        for trade in trades:
            if trade['step'] in range(sys.maxsize)[step_range]:
                date = self.df['Timestamp'].values[trade['step']]
                close = self.df['Close'].values[trade['step']]

                if trade['type'] == 'buy':
                    color = 'g'
                else:
                    color = 'r'

                self.price_ax.annotate(' ', (date, close),
                                       xytext=(date, close),
                                       size="large",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))


    def _render_title(self, net_worths):
        net_worth = round(net_worths[-1], 2)
        initial_net_worth = round(net_worths[0], 2)
        profit_percent = round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        self.fig.suptitle('Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')


    def render(self, current_step, net_worths, benchmarks, trades, window_size=200, figwidth=15):
        window_start = max(current_step - window_size, 0)
        step_range = slice(window_start, current_step + 1)
        times = self.df['Timestamp'].values[step_range]

        self.data_n_timesteps = int(config.data_n_timesteps)

        # Create a figure on screen and set the title
        self.fig = plt.figure(figsize=(figwidth, figwidth*0.7))

        # Create top subplot for net worth axis
        self._render_net_worth(step_range, times, current_step, net_worths, benchmarks)

        # Create bottom subplot for shared price/volume axis
        self._render_price(step_range, times, current_step)
        
        # Create a new axis for volume which shares its x-axis with price
        self._render_volume(step_range, times)
        
        # Create informative title
        self._render_title(net_worths)

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.90, wspace=0.2, hspace=0)
        
        # Improve x axis annotation
        date_col = pd.to_datetime(self.df['Timestamp'], unit='s').dt.strftime('%m/%d/%Y %H:%M')
        date_labels = date_col.values[step_range]
        self.price_ax.set_xticklabels(date_labels, rotation=45, horizontalalignment='right')

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)
        
        # Show the graph without blocking the rest of the program
        plt.show()

    def close(self):
        plt.close()