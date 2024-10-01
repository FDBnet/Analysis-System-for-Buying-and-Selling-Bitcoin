import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import ta
from ta import add_all_ta_features
from pytrends.request import TrendReq
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
import math
import ssl
import socket
import json
from scipy import stats
from statsmodels.api import OLS

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_ssl_connection(host, port=443):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port)) as sock:
            with context.wrap_socket(sock, server_hostname=host) as secure_sock:
                cipher = secure_sock.cipher()
                logger.info(f"Conexão SSL bem-sucedida para {host}:{porta}")
                logger.info(f"Versão SSL: {secure_sock.version()}")
                # logger.info(f"Cipher: {cipher}")
        return True
    except ssl.SSLError as e:
        logger.error(f"Erro SSL ao conectar-se a {host}:{port}: {e}")
    except socket.error as e:
        logger.error(f"Erro de soquete ao conectar {host}:{port}: {e}")
    except Exception as e:
        logger.error(f"Erro inesperado ao conectar-se a {host}:{port}: {e}")
    return False

def rate_limited_api_call(url, max_retries=5, initial_wait=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Too Many Requests
                tempo_de_espera = initial_wait * (2 ** attempt)  # Espera exponencial
                logger.warning(f"Limite de taxa atingido. Aguardando {tempo_de_espera} segundos antes de tentar novamente.")
                time.sleep(tempo_de_espera)
            else:
                logger.error(f"Falha na chamada da API: {e}")
                return None
    logger.error("Máximo de tentativas atingido. Não é possível buscar dados.")
    return None

def almon_poly(x, p):
    return np.array([x**i for i in range(p+1)]).T

def midas_weights(j, theta, p):
    almon = almon_poly(j, p)
    w = np.exp(np.dot(almon, theta))
    return w / np.sum(w)

def midas_regressao(y, x, lags, p):
    T = len(y)
    X = np.zeros((T, p+1))
    
    for t in range(T):
        if t < lags:
            continue
        x_lagged = x[t-lags:t]
        weights = midas_weights(np.arange(lags), np.ones(p+1), p)
        X[t] = np.dot(almon_poly(np.arange(lags), p).T, weights * x_lagged)
    
    model = OLS(y[lags:], X[lags:])
    results = model.fit()
    return results

class EnhancedBitcoinAnalyzer:
    def __init__(self):
        self.symbol = "bitcoin"
        self.vs_currency = "usd"
        self.timeframe = "5m"
        self.buy_threshold = 75
        self.sell_threshold = 35
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.session = None
        self.pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25), retries=2, backoff_factor=0.1)
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.alternative_api_url = "https://api.alternative.me/v2/ticker/bitcoin/"
        self.github_base_url = "https://api.github.com"
        self.model = self.load_or_train_model()
        self.time_horizons = {
            '10m': 10,
            '1h': 60,
            '6h': 360,
            '1d': 1440
        }
        self.horizonte_selecionado = '1h'  # Padrão para 1 hora
        self.midas_model = None

    def check_internet_connection(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except OSError as e:
            logger.error(f"Erro na conexão com Internet: {e}")
            return False

    async def initialize(self):
        connector = aiohttp.TCPConnector(family=4)
        self.session = aiohttp.ClientSession(connector=connector)

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
        df = self.add_technical_indicators(df)
        df = df.dropna()
        
        X = df.drop(['Close'], axis=1)
        y = df['Close']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    def add_technical_indicators(self, df):
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                df[col] = df['Close']  # Use Close as a fallback
            df[col] = df[col].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Formato de dados históricos: {df.shape}")
        logger.info(f"Intervalo de datas: {df.index.min()} a {df.index.max()}")
        
        if len(df) < 14:  # A maioria dos indicadores requer pelo menos 14 pontos de dados
            logger.warning(f"Não há pontos de dados suficientes para indicadores técnicos: {len(df)}")
            return df
        
        try:
            return add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
            return df

    def get_historical_data(self):
        try:
            days_needed = 2
            url = f"{self.coingecko_base_url}/coins/{self.symbol}/market_chart?vs_currency={self.vs_currency}&days={days_needed}"
            data = rate_limited_api_call(url)
            if not data or 'prices' not in data:
                raise Exception("Falha ao recuperar dados do CoinGecko")
            
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Adicione outros dados OHLCV se disponíveis
            for col in ['total_volumes', 'market_caps']:
                if col in data:
                    df[col.rstrip('s').capitalize()] = [x[1] for x in data[col]]
            
            # Preencher dados OHLC ausentes
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close']
            df['Low'] = df['Close']
            
            logger.info(f"Dados históricos obtidos do CoinGecko: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao obter dados históricos da CoinGecko: {e}")
            return self.get_alternative_historical_data()
        
    def get_alternative_historical_data(self):
        try:
            url = f"{self.alternative_api_url}?format=json"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or '1' not in data['data']:
                raise ValueError("Formato de dados inesperado da API alternativa")
            
            bitcoin_data = data['data']['1']
            df = pd.DataFrame({
                'Close': float(bitcoin_data['price_usd']),
                'Volume': float(bitcoin_data['volume24']),
                'Market_Cap': float(bitcoin_data['market_cap_usd'])
            }, index=[pd.Timestamp.now()])
            
            # Preencher dados OHLC ausentes
            df['Open'] = df['Close']
            df['High'] = df['Close']
            df['Low'] = df['Close']
            
            logger.info(f"Dados históricos obtidos da API alternativa: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao obter dados históricos da API alternativa: {e}")
            return pd.DataFrame()  # Retornar DataFrame vazio se falhar

    def get_current_data_sync(self):
        if not self.check_internet_connection():
            logger.error("Sem conexão com a internet")
            return pd.Series()

        url = f"{self.coingecko_base_url}/simple/price?ids={self.symbol}&vs_currencies={self.vs_currency}&include_24hr_vol=true&include_24hr_change=true&include_last_updated_at=true"
        for _ in range(3):  # Tenta 3 vezes
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Dados obtidos do CoinGecko: {data}")
                current_price = data[self.symbol][self.vs_currency]
                volume = data[self.symbol][f"{self.vs_currency}_24h_vol"]
                change_24h = data[self.symbol][f"{self.vs_currency}_24h_change"]
                last_updated = data[self.symbol]['last_updated_at']
                
                return pd.Series({
                    'Close': current_price,
                    'Volume': volume,
                    'Change_24h': change_24h,
                    'Last_Updated': pd.to_datetime(last_updated, unit='s')
                })
            except Exception as e:
                logger.error(f"Erro ao obter dados do CoinGecko: {e}")
                time.sleep(5)  # Espera 5 segundos antes de tentar novamente
        
        # Se falhar, tenta a API alternativa
        try:
            response = requests.get(self.alternative_api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            #logger.debug(f"Dados obtidos da API alternativa: {data}")
            bitcoin_data = data['data']['1']  # '1' é o ID do Bitcoin nesta API
            return pd.Series({
                'Close': float(bitcoin_data['price_usd']),
                'Volume': float(bitcoin_data['volume24']),
                'Change_24h': float(bitcoin_data['percent_change_24h']),
                'Last_Updated': pd.to_datetime(int(bitcoin_data['last_updated']), unit='s')
            })
        except Exception as e:
            logger.error(f"Erro ao obter dados da API alternativa: {e}")
        
        return pd.Series()  # Retorna uma série vazia se todas as tentativas falharem

    def get_market_data(self):
        url = f"{self.coingecko_base_url}/coins/{self.symbol}"
        for attempt in range(5):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 429:
                    tempo_de_espera = int(response.headers.get('Retry-After', 60))
                    print(f"Limite de taxa atingido. Aguardando {tempo_de_espera} segundos antes de tentar novamente...")
                    time.sleep(tempo_de_espera)
                    continue
                response.raise_for_status()
                data = response.json()
                return data['market_data']
            except requests.exceptions.RequestException as e:
                print(f"Erro ao buscar dados de mercado: {e}")
                time.sleep(5)
        
        print("Falha ao buscar dados do CoinGecko. Tentando fonte alternativa...")
        return self.get_alternative_market_data()
    
    def get_alternative_market_data(self):
        try:
            response = requests.get(self.alternative_api_url, timeout=30)
            response.raise_for_status()
            data = response.json()
            if 'data' not in data or '1' not in data['data']:
                raise ValueError("Formato de dados inesperado da API alternativa")
            bitcoin_data = data['data']['1']
            return {
                'current_price': {'usd': float(bitcoin_data['price_usd'])},
                'market_cap_rank': int(bitcoin_data['rank']),
                'price_change_percentage_24h': float(bitcoin_data['percent_change_24h']),
                'market_cap_change_percentage_24h': float(bitcoin_data['percent_change_24h'])  # Usando o mesmo valor da mudança de preço
            }
        except Exception as e:
            logger.error(f"Erro ao buscar dados de mercado alternativos: {e}")
            return None

    def get_github_data(self):
        search_query = "bitcoin OR cryptocurrency"
        url = f"{self.github_base_url}/search/repositories?q={search_query}&sort=stars&order=desc"
        
        headers = {
            "Accept": "application/vnd.github.v3+json"
        }

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return {
                'total_count': data['total_count'],
                'top_repos': data['items'][:5]  # Obtemos os 5 principais repositórios
            }
        except Exception as e:
            logger.error(f"Falha ao obter dados do GitHub: {e}")
            return None
        
    def analyze_github_data(self, github_data):
        if not github_data:
            return 0

        score = 0
        total_count = github_data['total_count']

        # Pontuação baseada no número total de repositórios
        if total_count > 100000:
            score += 10
        elif total_count > 50000:
            score += 7
        elif total_count > 10000:
            score += 5
        elif total_count > 1000:
            score += 3

        # Análise dos principais repositórios
        for repo in github_data['top_repos']:
            if repo['stargazers_count'] > 10000:
                score += 2
            elif repo['stargazers_count'] > 5000:
                score += 1

            # Verificar se houve atualizações recentes
            last_update = datetime.strptime(repo['updated_at'], "%Y-%m-%dT%H:%M:%SZ")
            if (datetime.now() - last_update).days < 7:
                score += 1

        return min(score, 20)  # Limite máximo de 20 pontos

    def get_google_trends_data(self):
        keywords = ["bitcoin", "crypto", "blockchain", "btc price", "bitcoin trading"]
        url = "https://trends.google.com/trends/api/dailytrends"
        params = {
            "hl": "en-US",
            "tz": "-180",
            "geo": "US",
            "ns": "15"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.text[5:]  # Remove ")]}',\n" at the beginning
            trends_data = json.loads(data)
            
            #logger.info(f"Dados do Google Trends: {json.dumps(trends_data, indent=2)}")
            
            score = 0
            for trend in trends_data['default']['trendingSearchesDays'][0]['trendingSearches']:
                title = trend['title']['query'].lower()
                #logger.info(f"Analisando trend: {title}")
                if any(keyword in title for keyword in keywords):
                    traffic = int(trend['formattedTraffic'].replace('+', '').replace('K', '000'))
                    #logger.info(f"Tendência relevante encontrada: {title} (Tráfego: {traffic})")
                    score += traffic
                    if any(pos in title for pos in ['bull', 'rise', 'surge', 'green', 'up']):
                        score += traffic * 0.5
                    elif any(neg in title for neg in ['bear', 'crash', 'fall', 'drop', 'down']):
                        score -= traffic * 0.5
            
            normalized_score = min(score / 1000, 100)  # Normalizar para 0-100
            logger.info(f"Pontuação de sentimento do Google Trends: {normalized_score}")
            return normalized_score
        except Exception as e:
            logger.error(f"Erro ao buscar dados do Google Trends: {e}")
            return 50  # Retorna uma pontuação neutra em caso de erro

    def predict_price_rf(self, data):
        features = self.add_technical_indicators(data.to_frame().T)
        features = features.dropna()
        
        # Garantir que usamos apenas recursos que estavam presentes durante o treinamento
        model_features = self.model.feature_names_in_
        features = features.reindex(columns=model_features, fill_value=0)
        
        prediction = self.model.predict(features)
        return prediction[0]

    def predict_price_arima(self, data):
        if len(data) < 5:
            logger.warning("Não há pontos de dados suficientes para a previsão ARIMA")
            return data['Close'].iloc[-1]  # Retorne o último preço conhecido
        
        try:
            model = ARIMA(data['Close'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast[0]
        except Exception as e:
            logger.error(f"Erro na previsão ARIMA: {e}")
            return data['Close'].iloc[-1]  # Retorna o último preço conhecido como fallback

    def calculate_technical_score(self, data):
        if len(data) < 50:
            logger.warning("Dados insuficientes para cálculo de score técnico completo. Usando cálculo simplificado.")
            return 20  # Retorna uma pontuação neutra

        score = 0
        
        # Tendência
        if data['Close'].iloc[-1] > data['Close'].rolling(window=min(50, len(data))).mean().iloc[-1]:
            score += 10
        
        # Momento
        rsi = ta.momentum.RSIIndicator(data['Close']).rsi().iloc[-1]
        if 40 < rsi < 60:
            score += 10
        
        # Volatilidade
        bbands = ta.volatility.BollingerBands(data['Close'])
        if bbands.bollinger_lband().iloc[-1] < data['Close'].iloc[-1] < bbands.bollinger_hband().iloc[-1]:
            score += 10
        
        # Volume
        if 'Volume' in data.columns and len(data) >= 20:
            if data['Volume'].iloc[-1] > data['Volume'].rolling(window=20).mean().iloc[-1]:
                score += 10
        else:
            score += 5  # Pontuação neutra para volume se os dados não estiverem disponíveis
        
        return score

    def calculate_market_score(self, market_data):
        score = 0
        
        # Classificação de capitalização de mercado
        if market_data['market_cap_rank'] <= 1:
            score += 10
        elif market_data['market_cap_rank'] <= 5:
            score += 5
        
        # Porcentagem de mudança de preço
        if market_data['price_change_percentage_24h'] > 0:
            score += 10
        elif market_data['price_change_percentage_24h'] > -5:
            score += 5
        
        # Porcentagem de mudança de capitalização de mercado
        if market_data['market_cap_change_percentage_24h'] > 0:
            score += 10
        elif market_data['market_cap_change_percentage_24h'] > -5:
            score += 5
        
        return score

    def calculate_sentiment_score(self, sentiment):
        return int(sentiment / 100 * 15)  # Normaliza de 0-100 para 0-15

    def calculate_prediction_score(self, price_change_prediction):
        if price_change_prediction > 5:
            return 15
        elif price_change_prediction > 2:
            return 12
        elif price_change_prediction > 0:
            return 8
        elif price_change_prediction > -2:
            return 5
        elif price_change_prediction > -5:
            return 2
        return 0

    def calculate_risk_score(self, data):
        returns = data['Close'].pct_change().dropna()
        var = np.percentile(returns, 5)
        max_drawdown = (data['Close'] / data['Close'].cummax() - 1).min()
        
        # Calcular médias históricas
        historical_var = np.mean([np.percentile(returns[:i], 5) for i in range(30, len(returns))])
        historical_max_drawdown = np.mean([(data['Close'][:i] / data['Close'][:i].cummax() - 1).min() for i in range(30, len(data))])
        
        var_score = int(abs(var / historical_var) * 100)
        drawdown_score = int(abs(max_drawdown / historical_max_drawdown) * 100)
        
        risk_score = (var_score + drawdown_score) // 2
        
        #logger.info(f"Value at Risk (5%): {var:.2%} (Historical avg: {historical_var:.2%})")
        #logger.info(f"Max Drawdown: {max_drawdown:.2%} (Historical avg: {historical_max_drawdown:.2%})")
        #logger.info(f"VaR Score: {var_score}")
        #logger.info(f"Drawdown Score: {drawdown_score}")
        #logger.info(f"Total Risk Score: {risk_score}")
        
        return min(risk_score, 20)  # Ainda limitado a 20 pontos

    def get_recommendation(self, total_score):
        if total_score >= self.buy_threshold:
            return "COMPRAR"
        elif total_score <= self.sell_threshold:
            return "VENDER"
        else:
            return "MANTER"

    def train_midas_model(self, daily_data, intraday_data, lags=22, p=2):
        y = daily_data['Close'].pct_change().dropna()
        x = intraday_data['Close'].pct_change().dropna().resample('D').last()
        
        self.midas_model = midas_regressao(y, x, lags, p)

    def predict_midas(self, intraday_data, horizon='1d'):
        if self.midas_model is None:
            raise ValueError("O modelo MIDAS ainda não foi treinado.")
        
        lags = 22  # Deve ser o mesmo usado no treinamento
        p = 2  # Deve ser o mesmo usado no treinamento
        
        x_pred = intraday_data['Close'].pct_change().dropna().tail(lags)
        
        X_pred = np.zeros((1, p+1))
        weights = midas_weights(np.arange(lags), self.midas_model.params, p)
        X_pred[0] = np.dot(almon_poly(np.arange(lags), p).T, weights * x_pred)
        
        predicted_return = self.midas_model.predict(X_pred)[0]
        last_price = intraday_data['Close'].iloc[-1]
        predicted_price = last_price * (1 + predicted_return)
        
        return predicted_price

    def analyze(self):
        try:
            print(f"\nAnalisando condições de mercado do Bitcoin para as próximas {self.horizonte_selecionado}...")
            
            try:
                current_data = self.get_current_data_sync()
                print(f"Dados atuais obtidos: {current_data}")
            except Exception as e:
                logger.error(f"Erro ao obter dados atuais: {e}")
                raise

            try:
                historical_data = self.get_historical_data()
                print(f"Dados históricos obtidos: {historical_data.head()}")
            except Exception as e:
                logger.error(f"Erro ao obter dados históricos: {e}")
                raise

            try:
                market_data = self.get_market_data()
                print(f"Dados de mercado obtidos: {market_data}")
            except Exception as e:
                logger.error(f"Erro ao obter dados de mercado: {e}")
                raise

            try:
                sentiment = self.get_google_trends_data()
                print(f"Dados de sentimento obtidos: {sentiment}")
            except Exception as e:
                logger.error(f"Erro ao obter dados de sentimento: {e}")
                raise

            try:
                github_data = self.get_github_data()
                print(f"Dados do GitHub obtidos: {github_data}")
            except Exception as e:
                logger.error(f"Erro ao obter dados do GitHub: {e}")
                raise

            current_price = current_data['Close']
            
            # Previsões
            try:
                arima_prediction = self.predict_price_arima(historical_data)
                ets_prediction = self.predict_ets(historical_data)
                lstm_prediction = self.predict_lstm(historical_data)
                print(f"Previsões: ARIMA={arima_prediction}, ETS={ets_prediction}, LSTM={lstm_prediction}")
            except Exception as e:
                logger.error(f"Erro ao fazer previsões: {e}")
                raise
            
            # Média das previsões
            predicted_price = np.mean([arima_prediction, ets_prediction, lstm_prediction])

            try:
                scores = {
                    'Técnica': self.calculate_technical_score(historical_data),
                    'Mercado': self.calculate_market_score(market_data) if market_data else 0,
                    'Sentimento': self.calculate_sentiment_score(sentiment),
                    'GitHub': self.analyze_github_data(github_data),
                    'Risco': -self.calculate_risk_score(historical_data)
                }
                print(f"Pontuações calculadas: {scores}")
            except Exception as e:
                logger.error(f"Erro ao calcular pontuações: {e}")
                raise

            price_change_prediction = (predicted_price - current_price) / current_price * 100
            scores['Previsão'] = self.calculate_prediction_score(price_change_prediction)

            total_score = sum(scores.values())

            print("\n--- Análise de Mercado do Bitcoin ---")
            print(f"Preço Atual: R${current_price:.2f}")
            print(f"Preço Previsto: R${predicted_price:.2f} ({price_change_prediction:.2f}%)")
            print(f"\nPrevisões individuais:")
            print(f"ARIMA: R${arima_prediction:.2f}")
            print(f"ETS: R${ets_prediction:.2f}")
            print(f"LSTM: R${lstm_prediction:.2f}")
            print(f"\nPontuações (total de 120):")
            for category, score in scores.items():
                print(f"{category}: {score}")
            print(f"\nPontuação Total: {total_score}/120")

            recommendation = self.get_recommendation(total_score)
            print(f"\nRecomendação: {recommendation}")

            return recommendation, current_price, predicted_price, total_score

        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            print(f"Erro detalhado na análise: {str(e)}")
            return None, None, None, None

    def predict_price_arima(self, data):
        try:
            model = ARIMA(data['Close'], order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=1)
            return forecast[0]
        except Exception as e:
            logger.error(f"Erro na previsão ARIMA: {e}")
            return data['Close'].iloc[-1]  # Return the last known price as a fallback

    def predict_ets(self, data):
        try:
            model = ExponentialSmoothing(data['Close'])
            results = model.fit()
            forecast = results.forecast(steps=1)
            return forecast[0]
        except Exception as e:
            logger.error(f"Erro na previsão ETS: {e}")
            return data['Close'].iloc[-1]  # Return the last known price as a fallback

    def predict_lstm(self, data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
        
        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=1, batch_size=1, verbose=0)
        
        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        
        return pred_price[0][0]

    def set_time_horizon(self, horizon):
        if horizon in self.time_horizons:
            self.horizonte_selecionado = horizon
        else:
            print(f"Horizonte de tempo inválido. Opções válidas: {', '.join(self.time_horizons.keys())}")

    def run(self):
        print("Bem-vindo ao Analisador de Mercado do Bitcoin")
        while True:
            print("\nO que você gostaria de fazer?")
            print("1. Analisar mercado do Bitcoin")
            print("2. Ver última análise")
            print("3. Definir horizonte temporal")
            print("4. Sair")
            
            choice = input("Digite sua escolha (1-4): ")
            
            if choice == '1':
                if not self.check_internet_connection():
                    print("Erro: Sem conexão com a internet. Por favor, verifique sua conexão e tente novamente.")
                    continue

                result = self.analyze()
                if result[0] is not None:
                    recommendation, current_price, predicted_price, score = result
                    self.last_analysis = result
                    print("\nAnálise completa. Você pode ver os resultados selecionando a opção 2.")
                else:
                    print("A análise falhou. Por favor, verifique os logs para mais detalhes.")
            elif choice == '2':
                if hasattr(self, 'last_analysis'):
                    recommendation, current_price, predicted_price, score = self.last_analysis
                    print(f"\nResultados da Última Análise:")
                    print(f"Recomendação: {recommendation}")
                    print(f"Preço Atual: R${current_price:.2f}")
                    print(f"Preço Previsto: R${predicted_price:.2f}")
                    print(f"Pontuação Geral: {score}/120")
                else:
                    print("Nenhuma análise prévia disponível. Por favor, execute uma análise primeiro.")
            elif choice == '3':
                print(f"Horizontes disponíveis: {', '.join(self.time_horizons.keys())}")
                new_horizon = input("Digite o horizonte desejado: ")
                self.set_time_horizon(new_horizon)
            elif choice == '4':
                print("Obrigado por usar o Analisador de Mercado do Bitcoin. Até logo!")
                break
            else:
                print("Escolha inválida. Por favor, digite 1, 2, 3 ou 4.")

            print("\n" + "-"*40)

def main():
    analyzer = EnhancedBitcoinAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
