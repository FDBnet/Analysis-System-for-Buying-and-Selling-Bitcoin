# Analisador de Mercado do Bitcoin Aprimorado (atualizado) 
Versão em PT-BR (Português do Brasil)

## Visão Geral 
Este projeto é um sistema aprimorado de análise de mercado do Bitcoin que utiliza dados de preço, indicadores técnicos, taxas de financiamento, sentimento do Google Trends, dados on-chain e análise macroeconômica para fornecer uma avaliação abrangente do mercado. O sistema recolhe dados de diversas fontes, realiza análises e fornece recomendações de compra, venda ou manutenção com base num modelo de classificação ponderada.

## Funcionalidades 
- Coleta de dados de preço e volume do Bitcoin através da API CoinGecko 
- Comparação do preço atual com a média móvel de 200 dias 
- Estimativa da taxa de financiamento futuro com base em dados históricos 
- Obtenção das taxas de financiamento atuais das bolsas Binance, Bybit e OKX 
- Análise de sentimento do mercado usando dados do Google Trends 
- Cálculo de indicadores técnicos como RSI e MACD 
- Análise da distribuição da oferta de Bitcoin 
- Estimativa do EPR (Estimated Price Ratio), uma aproximação do SOPR 
- Cálculo do NVT Ratio (Network Value to Transactions Ratio) 
- Análise da taxa de hash da rede Bitcoin _[necessita de correção]_ 
- Incorporação de dados macroeconômicos (taxas de juros globais) _[necessita de correção]_ 
- Sistema de avaliação ponderada considerando todos os indicadores 
- Recomendações de compra, venda ou manutenção com base na pontuação final 
- Interface de linha de comando interativa com opções para análise, limpeza de cache e saída 
- Registro de atividades e erros usando o módulo de logging do Python

## Requisitos 
- Python 3.6+ 
- Bibliotecas Python (veja `requisitos.txt`)

## Instalação 
1. Clone a pasta ```pt-br``` do repositório: _FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin_ 
2. Instale as dependências: ``` pip install -r requisitos.txt ```

## Uso Execute o script principal: 
``` python analisador_bitcoin.py ``` 

O sistema será iniciado e apresentará um menu interativo. Você pode escolher realizar uma nova análise, limpar o cache de dados ou sair do programa.

## Contribuição
Contribuições são bem-vindas! 

Se você quiser contribuir, por favor: 
1. Faça um fork do repositório
2. Crie uma branch para seu feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas alterações (`git commit -m 'Add some AmazingFeature'`)
4. Push para um branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Aviso Legal 
Este projeto é apenas para fins educacionais. Não é aconselhamento financeiro e não deve ser usado como base única para decisões de investimento. Investir em criptomoedas envolve riscos significativos. Sempre faça sua própria pesquisa antes de investir.

## Licença 
Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato
[Rodrigo S. Magalhães]

### Threads
*[@rodrigomagalhaes.8](https://www.threads.net/@rodrigomagalhaes.8)*

### E-mail
*falecom@fortalezadigital.net* *digocatu@hotmail.com*

### Link do Projeto:
*[https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin](https://github.com/FDBnet /Sistema-de-Análise-para-Compra-e-Venda-de-Bitcoin)*

## Apoie o Projeto

### *Pix:*
digocatu@hotmail.com

### *Bitcoin:*
bc1q63mezfs72jss00xvqhhjzhld33jzm322wn95x3
