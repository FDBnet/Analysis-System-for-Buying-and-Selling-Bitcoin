# Por Rodrigo Magalhães, da FDB (https://github.com/FDBnet)
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
import ta
from ta import add_all_ta_features
from pytrends.request import TrendReq
import joblib
import warnings
from statsmodels.tsa.arima.model import ARIMA
import math
import ssl
import socket
import json

# Configuração específica para Windows
if asyncio.get_event_loop_policy().__class__.__name__ == 'WindowsProactorEventLoopPolicy':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def test_ssl_connection(host, port=443):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, port)) as sock:
            with context.wrap_socket(sock, server_hostname=host) as secure_sock:
                cipher = secure_sock.cipher()
                logger.info(f"Successful SSL connection to {host}:{port}")
                logger.info(f"SSL version: {secure_sock.version()}")
                logger.info(f"Cipher: {cipher}")
        return True
    except ssl.SSLError as e:
        logger.error(f"SSL error connecting to {host}:{port}: {e}")
    except socket.error as e:
        logger.error(f"Socket error connecting to {host}:{port}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error connecting to {host}:{port}: {e}")
    return False

# Call this method before making API calls
test_ssl_connection('api.coingecko.com')
test_ssl_connection('api.github.com')

def rate_limited_api_call(url, max_retries=5, initial_wait=1):
    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if response.status_code == 429:  # Too Many Requests
                wait_time = initial_wait * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retrying.")
                time.sleep(wait_time)
            else:
                logger.error(f"API call failed: {e}")
                return None
    logger.error("Max retries reached. Unable to fetch data.")
    return None

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
        logging.getLogger().setLevel(logging.ERROR)

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
        
        logger.info(f"Historical data shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        if len(df) < 14:  # Most indicators require at least 14 data points
            logger.warning(f"Not enough data points for technical indicators: {len(df)}")
            return df
        
        try:
            return add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df

    def get_historical_data(self):
        try:
            url = f"{self.coingecko_base_url}/coins/{self.symbol}/ohlc?vs_currency={self.vs_currency}&days=180"  # Increased to 180 days
            data = rate_limited_api_call(url)
            if not data:
                raise Exception("Falha ao recuperar dados do CoinGecko")
            
            df = pd.DataFrame(data, columns=['timestamp', 'Open', 'High', 'Low', 'Close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            volume_url = f"{self.coingecko_base_url}/coins/{self.symbol}/market_chart?vs_currency={self.vs_currency}&days=180&interval=daily"
            volume_data = rate_limited_api_call(volume_url)
            if not volume_data:
                raise Exception("Falha ao recuperar dados de volume do CoinGecko")
            
            volume_df = pd.DataFrame(volume_data['total_volumes'], columns=['timestamp', 'Volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            
            df = df.join(volume_df)
            
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos da CoinGecko: {e}")
            logger.info("Tentando obter dados históricos da API alternativa...")
            
            try:
                url = f"{self.alternative_api_url}?format=chart&timeframe=1-year&roller=24-hours"
                response = requests.get(url)
                response.raise_for_status()
                data = response.json()
                
                df = pd.DataFrame(data['data'], columns=['timestamp', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                df.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
                
                # Inferir OHLC a partir do preço de fechamento
                df['Open'] = df['Close'].shift(1)
                df['High'] = df['Close']
                df['Low'] = df['Close']
                
                df.dropna(inplace=True)  # Remover primeira linha que terá NaN devido ao deslocamento
                
            except Exception as e:
                logger.error(f"Erro ao obter dados históricos da API alternativa: {e}")
                return pd.DataFrame()

        return df

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
            logger.debug(f"Dados obtidos da API alternativa: {data}")
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
                    wait_time = int(response.headers.get('Retry-After', 60))
                    print(f"Limite de taxa atingido. Aguardando {wait_time} segundos antes de tentar novamente...")
                    time.sleep(wait_time)
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
            bitcoin_data = data['data']['1']  # '1' is the ID for Bitcoin in this API
            return {
                'current_price': {'usd': float(bitcoin_data['price_usd'])},
                'market_cap_rank': int(bitcoin_data['rank']),
                'price_change_percentage_24h': float(bitcoin_data['percent_change_24h']),
                'market_cap_change_percentage_24h': float(bitcoin_data['percent_change_24h'])  # Using same value as price change
            }
        except Exception as e:
            print(f"Error fetching alternative market data: {e}")
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
                'top_repos': data['items'][:5]  # We get the top 5 repositories
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
            
            logger.info(f"Google Trends data: {json.dumps(trends_data, indent=2)}")
            
            score = 0
            for trend in trends_data['default']['trendingSearchesDays'][0]['trendingSearches']:
                title = trend['title']['query'].lower()
                logger.info(f"Analyzing trend: {title}")
                if any(keyword in title for keyword in keywords):
                    traffic = int(trend['formattedTraffic'].replace('+', '').replace('K', '000'))
                    logger.info(f"Relevant trend found: {title} (Traffic: {traffic})")
                    score += traffic
                    if any(pos in title for pos in ['bull', 'rise', 'surge', 'green', 'up']):
                        score += traffic * 0.5
                    elif any(neg in title for neg in ['bear', 'crash', 'fall', 'drop', 'down']):
                        score -= traffic * 0.5
            
            normalized_score = min(score / 1000, 100)  # Normalize to 0-100
            logger.info(f"Google Trends sentiment score: {normalized_score}")
            return normalized_score
        except Exception as e:
            logger.error(f"Error fetching Google Trends data: {e}")
            return 50  # Return a neutral score on error

    def predict_price_rf(self, data):
        features = self.add_technical_indicators(data.to_frame().T)
        features = features.dropna()
        
        # Ensure we only use features that were present during training
        model_features = self.model.feature_names_in_
        features = features.reindex(columns=model_features, fill_value=0)
        
        prediction = self.model.predict(features)
        return prediction[0]

    def predict_price_arima(self, data):
        if len(data) < 5:
            logger.warning("Not enough data points for ARIMA prediction")
            return data['Close'].iloc[-1]  # Return the last known price
        
        try:
            model = ARIMA(data['Close'], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast[0]
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            return data['Close'].iloc[-1]  # Return the last known price as a fallback

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

    def calculate_market_score(self, market_data):
        score = 0
        
        # Market Cap Rank
        if market_data['market_cap_rank'] <= 1:
            score += 10
        elif market_data['market_cap_rank'] <= 5:
            score += 5
        
        # Price Change Percentage
        if market_data['price_change_percentage_24h'] > 0:
            score += 10
        elif market_data['price_change_percentage_24h'] > -5:
            score += 5
        
        # Market Cap Change Percentage
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
        
        # Calculate historical averages
        historical_var = np.mean([np.percentile(returns[:i], 5) for i in range(30, len(returns))])
        historical_max_drawdown = np.mean([(data['Close'][:i] / data['Close'][:i].cummax() - 1).min() for i in range(30, len(data))])
        
        var_score = int(abs(var / historical_var) * 100)
        drawdown_score = int(abs(max_drawdown / historical_max_drawdown) * 100)
        
        risk_score = (var_score + drawdown_score) // 2
        
        logger.info(f"Value at Risk (5%): {var:.2%} (Historical avg: {historical_var:.2%})")
        logger.info(f"Max Drawdown: {max_drawdown:.2%} (Historical avg: {historical_max_drawdown:.2%})")
        logger.info(f"VaR Score: {var_score}")
        logger.info(f"Drawdown Score: {drawdown_score}")
        logger.info(f"Total Risk Score: {risk_score}")
        
        return min(risk_score, 20)  # Still capped at 20 points

    def get_recommendation(self, total_score):
        if total_score >= self.buy_threshold:
            return "COMPRAR"
        elif total_score <= self.sell_threshold:
            return "VENDER"
        else:
            return "MANTER"

    def analyze(self):
        try:
            print("Analisando condições de mercado do Bitcoin...")
            
            current_data = self.get_current_data_sync()
            if current_data.empty:
                print("Erro: Não foi possível obter dados atuais de mercado.")
                return None, None, None, None

            historical_data = self.get_historical_data()
            if historical_data.empty:
                print("Erro: Não foi possível obter dados históricos.")
                return None, None, None, None

            market_data = self.get_market_data()
            if not market_data:
                print("Aviso: Não foi possível obter dados de mercado. Algumas pontuações podem ser afetadas.")

            sentiment = self.get_google_trends_data()
            github_data = self.get_github_data()

            rf_prediction = self.predict_price_rf(current_data)
            arima_prediction = self.predict_price_arima(historical_data)
            predicted_price = (rf_prediction + arima_prediction) / 2

            current_price = current_data['Close']

            technical_score = self.calculate_technical_score(historical_data)
            market_score = self.calculate_market_score(market_data) if market_data else 0
            sentiment_score = self.calculate_sentiment_score(sentiment)
            github_score = self.analyze_github_data(github_data)

            price_change_prediction = (predicted_price - current_price) / current_price * 100
            prediction_score = self.calculate_prediction_score(price_change_prediction)

            risk_score = self.calculate_risk_score(historical_data)

            total_score = (
                technical_score + 
                market_score + 
                sentiment_score + 
                prediction_score +
                github_score -
                risk_score
            )

            print("\n--- Análise de Mercado do Bitcoin ---")
            print(f"Preço Atual: R${current_price:.2f}")
            print(f"Preço Previsto: R${predicted_price:.2f} ({price_change_prediction:.2f}%)")
            print(f"\nPontuações (total de 120):")
            print(f"Técnica: {technical_score}/40")
            print(f"Mercado: {market_score}/30")
            print(f"Sentimento: {sentiment_score}/15")
            print(f"Previsão: {prediction_score}/15")
            print(f"Atividade no GitHub: {github_score}/20")
            print(f"Risco: -{risk_score}/20")
            print(f"\nPontuação Total: {total_score}/120")

            recommendation = self.get_recommendation(total_score)
            print(f"\nRecomendação: {recommendation}")

            return recommendation, current_price, predicted_price, total_score
        except Exception as e:
            print(f"Ocorreu um erro durante a análise: {e}")
            return None, None, None, None

    def run(self):
        print("Bem-vindo ao Analisador de Mercado do Bitcoin")
        while True:
            print("\nO que você gostaria de fazer?")
            print("1. Analisar mercado do Bitcoin")
            print("2. Ver última análise")
            print("3. Sair")
            
            choice = input("Digite sua escolha (1-3): ")
            
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
                    print("A análise falhou. Por favor, tente novamente mais tarde.")
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
                print("Obrigado por usar o Analisador de Mercado do Bitcoin. Até logo!")
                break
            else:
                print("Escolha inválida. Por favor, digite 1, 2 ou 3.")

            print("\n" + "-"*40)

def main():
    analyzer = EnhancedBitcoinAnalyzer()
    analyzer.run()

if __name__ == "__main__":
    main()
