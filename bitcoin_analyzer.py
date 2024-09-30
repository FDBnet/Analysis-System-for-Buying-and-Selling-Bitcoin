# Instale as dependências necessárias: 
    # pip install yfinance pandas numpy requests aiohttp scikit-learn ta pytrends joblib statsmodels textblob ccxt tweepy

# Substitua as seguintes chaves de API com suas próprias chaves:
    # YOUR_BINANCE_API_KEY e YOUR_BINANCE_SECRET_KEY
    # YOUR_GLASSNODE_API_KEY
    # YOUR_CRYPTOPANIC_API_KEY
    # TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from ta import add_all_ta_features
from pytrends.request import TrendReq
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
import ccxt
import tweepy
import math

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class AdvancedBitcoinAnalyzer:
    def __init__(self):
        self.symbol = "BTC/USDT"
        self.timeframe = "5m"
        self.buy_threshold = 90
        self.sell_threshold = 40
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self.load_or_train_model()
        self.session = None
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.exchange = ccxt.binance({
            'apiKey': 'YOUR_BINANCE_API_KEY',
            'secret': 'YOUR_BINANCE_SECRET_KEY',
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future'
            }
        })
        self.glassnode_api_key = "YOUR_GLASSNODE_API_KEY"
        self.cryptopanic_api_key = "YOUR_CRYPTOPANIC_API_KEY"
        self.twitter_api = self.setup_twitter_api()

    def setup_twitter_api(self):
        auth = tweepy.OAuthHandler("TWITTER_CONSUMER_KEY", "TWITTER_CONSUMER_SECRET")
        auth.set_access_token("TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET")
        return tweepy.API(auth)

    async def initialize(self):
        self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    def load_or_train_model(self):
        try:
            model = joblib.load('bitcoin_model.joblib')
            logger.info("Modelo carregado do arquivo.")
        except FileNotFoundError:
            logger.info("Treinando novo modelo...")
            model = self.train_model()
            joblib.dump(model, 'bitcoin_model.joblib')
        return model

    def train_model(self):
        df = self.get_historical_data()
        df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
        df = df.dropna()
        
        X = df.drop(['Close'], axis=1)
        y = df['Close']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def get_historical_data(self):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    async def get_current_data(self):
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=1)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df.iloc[0]

    async def get_on_chain_data(self):
        async with self.session.get(
            "https://api.glassnode.com/v1/metrics/indicators/sopr",
            params={"api_key": self.glassnode_api_key, "a": "BTC", "i": "24h"}
        ) as response:
            if response.status == 200:
                sopr_data = await response.json()
                sopr = sopr_data[-1]['v'] if sopr_data else None
            else:
                logger.error("Falha ao obter dados SOPR")
                sopr = None

        nvt = await self.get_nvt_ratio()
        mvrv = await self.get_mvrv_ratio()

        return {
            'sopr': sopr,
            'nvt': nvt,
            'mvrv': mvrv
        }

    async def get_nvt_ratio(self):
        async with self.session.get(
            "https://api.glassnode.com/v1/metrics/indicators/nvt",
            params={"api_key": self.glassnode_api_key, "a": "BTC", "i": "24h"}
        ) as response:
            if response.status == 200:
                nvt_data = await response.json()
                return nvt_data[-1]['v'] if nvt_data else None
            else:
                logger.error("Falha ao obter dados NVT")
                return None

    async def get_mvrv_ratio(self):
        async with self.session.get(
            "https://api.glassnode.com/v1/metrics/market/mvrv",
            params={"api_key": self.glassnode_api_key, "a": "BTC", "i": "24h"}
        ) as response:
            if response.status == 200:
                mvrv_data = await response.json()
                return mvrv_data[-1]['v'] if mvrv_data else None
            else:
                logger.error("Falha ao obter dados MVRV")
                return None

    async def get_market_sentiment(self):
        # Google Trends
        self.pytrends.build_payload(["bitcoin", "crypto", "blockchain"], timeframe='now 1-H')
        trends_data = self.pytrends.interest_over_time()
        trends_score = trends_data.iloc[-1].mean()

        # Análise de notícias
        news_sentiment = await self.get_news_sentiment()

        # Análise de sentimento do Twitter
        twitter_sentiment = self.get_twitter_sentiment()

        # Combinar os diferentes sentimentos
        combined_sentiment = (trends_score + news_sentiment + twitter_sentiment) / 3

        return combined_sentiment

    async def get_news_sentiment(self):
        async with self.session.get(f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_api_key}&currencies=BTC") as response:
            if response.status == 200:
                news_data = await response.json()
                sentiments = []
                for item in news_data['results']:
                    sentiment = TextBlob(item['title']).sentiment.polarity
                    if item['sentiment'] == 'positive':
                        sentiment = max(sentiment, 0.1)
                    elif item['sentiment'] == 'negative':
                        sentiment = min(sentiment, -0.1)
                    sentiments.append(sentiment)
                return np.mean(sentiments) if sentiments else 0
            else:
                logger.error("Falha ao obter dados de notícias")
                return 0

    def get_twitter_sentiment(self):
        tweets = self.twitter_api.search_tweets(q="bitcoin", count=100, lang="en", tweet_mode="extended")
        sentiments = [TextBlob(tweet.full_text).sentiment.polarity for tweet in tweets]
        return np.mean(sentiments)

    def predict_price_rf(self, data):
        features = add_all_ta_features(
            data, open="Open", high="High", low="Low", close="Close", volume="Volume"
        ).dropna()
        prediction = self.model.predict(features.iloc[-1].to_frame().T)
        return prediction[0]

    def predict_price_arima(self, data):
        model = ARIMA(data['Close'], order=(5,1,0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]

    async def get_futures_data(self):
        try:
            futures_data = self.exchange.fetch_ohlcv(f'{self.symbol}:USDT', '1h', limit=24)
            df = pd.DataFrame(futures_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Erro ao obter dados de futuros: {e}")
            return None

    def calculate_implied_volatility(self, data):
        log_returns = np.log(data['close'] / data['close'].shift(1))
        return np.sqrt(252) * log_returns.std()

    def calculate_put_call_ratio(self, futures_data):
        volume = futures_data['volume']
        price_change = futures_data['close'].pct_change()
        put_volume = volume[price_change < 0].sum()
        call_volume = volume[price_change > 0].sum()
        return put_volume / call_volume if call_volume != 0 else 1

    async def analyze(self):
        current_data = await self.get_current_data()
        historical_data = self.get_historical_data()
        on_chain_data = await self.get_on_chain_data()
        sentiment = await self.get_market_sentiment()
        futures_data = await self.get_futures_data()

        rf_prediction = self.predict_price_rf(current_data.to_frame().T)
        arima_prediction = self.predict_price_arima(historical_data)
        predicted_price = (rf_prediction + arima_prediction) / 2

        current_price = current_data['Close']

        technical_score = self.calculate_technical_score(historical_data)
        on_chain_score = self.calculate_on_chain_score(on_chain_data)
        sentiment_score = self.calculate_sentiment_score(sentiment)

        price_change_prediction = (predicted_price - current_price) / current_price * 100
        prediction_score = self.calculate_prediction_score(price_change_prediction)

        risk_score = self.calculate_risk_score(historical_data)

        if futures_data is not None:
            implied_volatility = self.calculate_implied_volatility(futures_data)
            put_call_ratio = self.calculate_put_call_ratio(futures_data)
            
            volatility_score = self.calculate_volatility_score(implied_volatility)
            market_sentiment_score = self.calculate_market_sentiment_score(put_call_ratio)
        else:
            volatility_score = 0
            market_sentiment_score = 0

        total_score = (
            technical_score + 
            on_chain_score + 
            sentiment_score + 
            prediction_score - 
            risk_score +
            volatility_score +
            market_sentiment_score
        )

        logger.info(f"Preço atual do Bitcoin: ${current_price:.2f}")
        logger.info(f"Preço previsto: ${predicted_price:.2f} ({price_change_prediction:.2f}%)")
        logger.info(f"Pontuação Técnica: {technical_score}/40")
        logger.info(f"Pontuação On-Chain: {on_chain_score}/30")
        logger.info(f"Pontuação de Sentimento: {sentiment_score}/15")
        logger.info(f"Pontuação de Previsão: {prediction_score}/15")
        logger.info(f"Pontuação de Risco: -{risk_score}/20")
        logger.info(f"Pontuação de Volatilidade: {volatility_score}/10")
        logger.info(f"Pontuação de Sentimento de Mercado: {market_sentiment_score}/10")
        logger.info(f"Pontuação Total: {total_score}/120")

        recommendation = self.get_recommendation(total_score)
        logger.info(f"Recomendação: {recommendation}")

        return recommendation, current_price, predicted_price, total_score

    def calculate_technical_score(self, data):
        score = 0
        
        # Tendência
        if data['Close'].iloc[-1] > data['Close'].rolling(window=50).mean().iloc[-1]:
            score += 10
        
        # Momentum
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]
        if 40 < rsi < 60:
            score += 10
        
        # Volatilidade
        bbands = ta.volatility.BollingerBands(data['Close'])
        if bbands.bollinger_lband().iloc[-1] < data['Close'].iloc[-1] < bbands.bollinger_hband().iloc[-1]:
            score += 10
        
        # Volume
        if data['Volume'].iloc[-1] > data['Volume'].rolling(window=20).mean().iloc[-1]:
            score += 10
        
        return score

    def calculate_on_chain_score(self, data):
        score = 0
        if data['sopr'] is not None:
            if data['sopr'] > 1:
                score += 10
        if data['nvt'] is not None:
            if data['nvt'] < 65:
                score += 10
        if data['mvrv'] is not None:
            if 1 < data['mvrv'] < 3.5:
                score += 10
        return score

    def calculate_sentiment_score(self, sentiment):
        return int((sentiment + 1) / 2 * 15)  # Normaliza de -1 a 1 para 0-15

    def calculate_prediction_score(self, price_change_prediction):
        if price_change_prediction > 5:
            return 15
        elif price_change_prediction > 2:
            return 10
        elif price_change_prediction > 0:
            return 5
        return 0

    def calculate_risk_score(self, data):
        returns = data['Close'].pct_change().dropna()
        var = np.percentile(returns, 5)
        max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
        
        risk_score = int(-var * 100) + int(-max_drawdown * 100)
        return min(risk_score, 20)  # Limita a 20 pontos

    def calculate_volatility_score(self, implied_volatility):
        if implied_volatility > 0.1:
            return 0
        elif implied_volatility > 0.05:
            return 5
        else:
            return 10

    def calculate_market_sentiment_score(self, put_call_ratio):
        if 0.8 <= put_call_ratio <= 1.2:
            return 5
        elif put_call_ratio > 1.5:
            return 0
        elif put_call_ratio < 0.5:
            return 10
        else:
            return 3

    def get_recommendation(self, total_score):
        if total_score >= self.buy_threshold:
            return "COMPRAR"
        elif total_score <= self.sell_threshold:
            return "VENDER"
        else:
            return "MANTER"

    async def run(self):
        await self.initialize()
        logger.info("Iniciando análise avançada do Bitcoin...")
        try:
            while True:
                recommendation, current_price, predicted_price, score = await self.analyze()
                logger.info(f"Recomendação: {recommendation} - Preço Atual: ${current_price:.2f} - Preço Previsto: ${predicted_price:.2f} - Pontuação: {score}/120")
                await asyncio.sleep(600)  # Analisa a cada 10 minutos
        except Exception as e:
            logger.error(f"Erro durante a análise: {e}")
        finally:
            await self.close()

if __name__ == "__main__":
    analyzer = AdvancedBitcoinAnalyzer()
    asyncio.run(analyzer.run())
