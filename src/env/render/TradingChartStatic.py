import sys
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib import style
from datetime import datetime
from pandas.plotting import register_matplotlib_converters

from src.env.trade.Benchmarks import buy_and_hold, rsi_divergence, sma_crossover
from src.util.run_util import retrieve_model_dir
from src.util.config import get_config
config = get_config()

style.use('ggplot')
register_matplotlib_converters()

VOLUME_CHART_HEIGHT = 0.33


class TradingChartStatic:
    """An OHLCV trading visualization using matplotlib made to render gym environments"""

    def __init__(self, df, start_step, end_step):
        self.df = df
        self.start_step = start_step
        self.end_step = end_step
        self.step_range = slice(start_step, end_step)
        self.benchmarks = config.benchmarks

    def _render_net_worth(self, times, net_worths):
        # Clear the frame rendered last step
        self.net_worth_ax = plt.subplot2grid((6, 1), (0, 0), rowspan=1, colspan=1)
        # Plot net worths
        self.net_worth_ax.plot(mdates.date2num(times), net_worths.loc[self.step_range], label='dunderbot', color="g", linewidth=3)

        #self._render_benchmarks(times, benchmarks)
        # Show legend, which uses the label we defined for the plot above
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

    def _render_benchmarks(self, times):
        colors = ['black', 'red', 'green', 'purple',
                  'magenta', 'yellow', 'cyan', 'orange']

        initial_account_balance = config.trading_params.initial_account_balance
        commission = config.trading_params.commission
        for i, benchmark_name in enumerate(self.benchmarks.strategies):
            benchmark_fn = eval(benchmark_name)
            
            benchmark_values = benchmark_fn(self.df.loc[self.step_range].reset_index()['Close'], initial_account_balance, commission)
            self.net_worth_ax.plot(mdates.date2num(times), benchmark_values,
                                   label=benchmark_name, color=colors[i % len(colors)], alpha=0.5)
        self.net_worth_ax.legend()
        legend = self.net_worth_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

    def _render_assets_held(self, times, assets_held_hist):
        self.assets_ax = plt.subplot2grid((6, 1), (1, 0), rowspan=1, colspan=1, sharex=self.net_worth_ax)

        # Plot assetprice
        self.assets_ax.plot(mdates.date2num(times), assets_held_hist.loc[self.step_range], label='Assets held', color="red")

        # Shift price axis up to give volume chart space
        ylim = self.assets_ax.get_ylim()
        self.assets_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * VOLUME_CHART_HEIGHT, ylim[1])

        self.assets_ax.legend()
        legend = self.assets_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

    def _render_price(self, times):
        self.price_ax = plt.subplot2grid((6, 1), (2, 0), rowspan=4, colspan=1, sharex=self.net_worth_ax)

        # Plot assetprice
        self.price_ax.plot(mdates.date2num(times), self.df['Close'].loc[self.step_range], label='Price', color="black")
        
        # Shift price axis up to give volume chart space
        ylim = self.price_ax.get_ylim()
        self.price_ax.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * VOLUME_CHART_HEIGHT, ylim[1])

        self.price_ax.legend()
        legend = self.price_ax.legend(loc=2, ncol=2, prop={'size': 8})
        legend.get_frame().set_alpha(0.4)

    def _render_volume(self, times):
        self.volume_ax = self.price_ax.twinx()
        volume = np.array(self.df['VolumeBTC'].loc[self.step_range])

        self.volume_ax.plot(mdates.date2num(times), volume,  color='blue')
        self.volume_ax.fill_between(mdates.date2num(times), volume, color='blue', alpha=0.5)

        self.volume_ax.set_ylim(0, max(volume) / VOLUME_CHART_HEIGHT)
        self.volume_ax.yaxis.set_ticks([])

    def _render_trades(self, trades):
        """ NOTE: the number of trades is commonly lower than number of timesteps,
        especially for models trained very short. This is due to the model trying to sell,
        despite not having any assets, and these events are not shown. """

        for trade in trades:
            if trade['step'] in range(sys.maxsize)[self.step_range]:
                time = self.df['Timestamp'].loc[trade['step']]
                close = self.df['Close'].loc[trade['step']]

                if trade['type'] == 'buy':
                    cmap = matplotlib.cm.get_cmap('Greens')
                    color = cmap(trade['action_amount'] * 2)
                elif trade['type'] == 'sell':
                    cmap = matplotlib.cm.get_cmap('Reds')
                    color = cmap(trade['action_amount'] * 2)
                elif trade['type'] == 'hold':
                    color = 'lightgray'

                self.price_ax.annotate(' ', (mdates.date2num(time), close),
                                       xytext=(mdates.date2num(time), close),
                                       size="small",
                                       arrowprops=dict(arrowstyle='simple', facecolor=color))

    def _render_title(self, net_worths):
        net_worth = round(net_worths.iloc[-1], 2)
        initial_net_worth = round(net_worths.iloc[0], 2)
        profit_percent = round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        self.fig.suptitle('Net worth: $' + str(net_worth) + ' | Profit: ' + str(profit_percent) + '%')

    def render(self, net_worths, trades, account_history, save_dir, figwidth=15):
        # Displace index to slice everything consistently
        assets_held_hist = account_history['asset_held']
        assets_held_hist.index = assets_held_hist.index + self.start_step

        # Convert lists to Series with index following (start_step, end_step)
        net_worths = pd.Series(net_worths, index=range(self.start_step, self.end_step+1))

        times = self.df['Timestamp'].loc[self.step_range]

        self.data_n_timesteps = int(config.data_params.data_n_timesteps)

        # Create a figure on screen and set the title
        self.fig = plt.figure(figsize=(figwidth, figwidth*0.8))

        # Add padding to make graph easier to view
        plt.subplots_adjust(left=0.11, bottom=0.24, right=0.90, top=0.95, wspace=0.2, hspace=0.05)

        # Render subplots which share x-axis (price)
        self._render_net_worth(times, net_worths)
        if len(config.benchmarks.strategies) > 0:
            self._render_benchmarks(times)
        self._render_assets_held(times, assets_held_hist)
        self._render_price(times)
        self._render_volume(times)
        self._render_title(net_worths)

        # Render trades, if they are not too many (when they become too hard to visually distinguish)
        trade_threshold = 5000
        if len(trades) <= trade_threshold:
            self._render_trades(trades)
        else:
            print(f'Not rendering trades since they are too many to distinguish in plot ({len(trades)}>{trade_threshold})')

        # Improve x axis annotation (use either DateFormatter or ConciseDateFormatter)
        locator = mdates.AutoDateLocator()
        self.price_ax.xaxis.set_major_locator(locator)
        # self.price_ax.xaxis.set_major_formatter(mdate s.ConciseDateFormatter(locator))
        self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M')) 
        self.price_ax.set_xlim([mdates.date2num(times.iloc[0]), mdates.date2num(times.iloc[-1])])
        self.fig.autofmt_xdate()

        # Hide duplicate net worth date labels
        plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

        # Save to disk
        nowtime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        plt.savefig(os.path.join(save_dir, f'TradingChartStatic_{nowtime}.pdf'))
        plt.show()

    def close(self):
        plt.close()
