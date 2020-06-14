# Ideas
## Features
* Tekniska indikatorer (gyllene kors och sånt). (7 and 21 days moving average, exponential moving average, momentum, Bollinger bands, MACD.)
* Värden med tidslagg och trender
* datetime (beteenden förändras sannolikt över året och under dagen)
* handelsvolymer
* korrelerade tillgångar/priser/index/etc. (VIX, trading currency)
* fouriertransform för cykler (n olika komponenter)
* anomalier
* Prophetvärden

## Förprocessering
* Kolla efter heteroskedasticity, multicollinearity, or serial correlation


## Random
* En modul som bedömmer om det kommer gå upp eller ner, och en annan som agerar på den informationen (köper/säjer). Ska man hantera det som ett regressions- eller klassificeringsproblem? Båda kan funka. 
* Ingen newshantering el. dyl. För komplext och osäkert att agera på. NLP och sånt. Nae. Tänker också att det är irrelevant för tidsskalan jag tänker mig (typ minuter) och eftersom tanken är att man bara ligger lite före det förutsägbara kollektivet.
* Kan använda feature importance med annan modelltyp än vad vi kör

## Kolla in
* GAN
* stacked autoencoders

# Resources
## Collections
* https://github.com/firmai/financial-machine-learning
## GANs
* OK intro-artikel: https://pathmind.com/wiki/generative-adversarial-network-gan

## Approaches
* https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md: uses GAN and packed with different techs for features (ARIMA; stacked autoencoders using NN, anomaly detection using NN etc.)
## Models

## GAN
* https://github.com/soumith/ganhacks
