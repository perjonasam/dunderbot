# Resources
## RL concepts/intros
* https://spinningup.openai.com/en/latest/spinningup/rl_intro.html: OPenAI intro to RL
* Super informative post on the problem setting of ML trading, incl. order book intro, and implementation overview, as well as a case for RL: http://www.wildml.com/2018/02/introduction-to-learning-to-trade-with-reinforcement-learning/
... follow-up post a few years later. Also very informative: https://dennybritz.com/blog/ai-trading/

## Resource collections
* https://github.com/firmai/financial-machine-learning

## Exchanges and data
* Stock value API: https://www.alphavantage.co/#page-top
* Crypto exchange, allegedly the “best”: https://www.binance.com/en
* Exchange and Value data source: https://pro.coinbase.com

## Crypto bots
* Faux, but interesting, review of common crypto bots, includes lists of exchanges and some notable trading strategies: https://3commas.io/blog/best-crypto-trading-bot ***NOTE: check out Zenbot.***


# Implementations
## RL implementations
* https://towardsdatascience.com/trade-smarter-w-reinforcement-learning-a5e91163f315: medium intro article on the tensortrade package, using RL
* Example implementation, simple RL NN PG with interesting details: https://launchpad.ai/blog/trading-bitcoin
* Example implementation: DQN, features, rewards, epsilon, and more: https://gradienttrader.github.io
* Example implementation: RL,  detailed feature ideas, benchmark ideas: https://link.medium.com/72jvcfKpU7
* Simple implementation example, A2C, LSTM with concat, a couple of neat details on data and reward: https://link.medium.com/BswAJUQ9W7
* Custom gym implementation for BTC trading: https://link.medium.com/sWKC4wBDX7

## Supervised implementations
* https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md: uses GAN and packed with different techs for features (ARIMA; stacked autoencoders using NN, anomaly detection using NN etc.)
* Interesting implementation, Supervised, Grad Boost, features, seemingly good theoretical return: https://github.com/cbyn/bitpredict & fork https://github.com/AdeelMufti/CryptoBot
* Story about a supervised model with some list of problems and solutions: https://www.softkraft.co/applying-machine-learning-to-cryptocurrency-trading/

# Model details
## GAN
* https://github.com/soumith/ganhacks
* OK intro-artikel: https://pathmind.com/wiki/generative-adversarial-network-gan

## RL
### Learning
* Q-learning and policy gradient, simple and Nice: https://flyyufelix.github.io/2017/10/12/dqn-vs-pg.html

### Custom env and gym
* Stable baselines docs on building a custom environment: https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html
* Simple guides how to setup custom gym: https://link.medium.com/19YAM5jcZ7 & https://link.medium.com/7hR4BUcdZ7
* Custom gym implementation on toy problem (simple maze competition): https://link.medium.com/ywRxqfEmY7
* Custom OpenAI Gym for trading: https://link.medium.com/dp7Yvt0bZ7

### Evaluation
* RL Evaluation: https://arxiv.org/abs/1709.06560

### Visualisation
* (Poor) visualization example in OpenAI Gy for trading: https://link.medium.com/Z6vfy7SbZ7

### General tips
* Very handy RL tips: https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html


# Idéer
## Features
* Tekniska indikatorer (gyllene kors och sånt). (7 and 21 days moving average, exponential moving average, momentum, Bollinger bands, MACD, etc. etc.)
* Värden med tidslagg och trender
* datetime (beteenden förändras sannolikt över året och under dagen)
* handelsvolymer
* korrelerade tillgångar/priser/index/etc. (VIX)
* fouriertransform för cykler (n olika komponenter)
* anomalier

## Förprocessering
* Kolla efter heteroskedasticity, multicollinearity, or serial correlation







