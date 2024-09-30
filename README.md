# Sistema de Análise para Compra e Venda de Bitcoin

## Visão Geral

Este projeto é um sistema avançado de análise de Bitcoin que combina análise técnica, on-chain, sentimento de mercado e elementos inspirados nas fórmulas de Midas para fornecer recomendações de trading. O sistema utiliza múltiplas fontes de dados e técnicas de machine learning para oferecer uma análise abrangente do mercado de Bitcoin.

**Nota: Este projeto está em desenvolvimento ativo e busca colaboradores. Se você tem experiência em criptomoedas, análise de dados ou machine learning, sua contribuição seria muito bem-vinda!**

## Funcionalidades

- Análise técnica usando indicadores da biblioteca `ta`
- Análise on-chain com dados da Glassnode (SOPR, NVT, MVRV)
- Análise de sentimento usando Google Trends, CryptoPanic e Twitter
- Previsão de preços usando Random Forest e ARIMA
- Elementos inspirados nas fórmulas de Midas (volatilidade implícita e razão put/call adaptada)
- Sistema de pontuação abrangente
- Recomendações de compra, venda ou manutenção baseadas na pontuação total

## Requisitos

- Python 3.7+
- Bibliotecas Python (veja `requirements.txt`)
- Chaves de API para: Binance, Glassnode, CryptoPanic, Twitter

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/seu-usuario/bitcoin-analysis-system.git
   cd bitcoin-analysis-system
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Configure as chaves de API:
   Edite o arquivo `config.py` e insira suas chaves de API para Binance, Glassnode, CryptoPanic e Twitter.

## Uso

Execute o script principal:

```
python bitcoin_analyzer.py
```

O sistema iniciará a análise e fornecerá recomendações a cada 10 minutos.

## Contribuição

Estamos ativamente buscando colaboradores para este projeto. Se você tem interesse em contribuir, aqui estão algumas áreas onde precisamos de ajuda:

1. Refinamento dos algoritmos de análise
2. Otimização de performance
3. Adição de novas fontes de dados
4. Melhoria da interface do usuário
5. Testes e validação do sistema

Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Aviso Legal

Este projeto é apenas para fins educacionais e de pesquisa. Não é aconselhamento financeiro e não deve ser usado como base única para decisões de investimento. Investir em criptomoedas envolve riscos significativos. Sempre faça sua própria pesquisa e consulte um profissional financeiro antes de investir.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## Contato

[Seu Nome] - [@seu_twitter](https://twitter.com/seu_twitter) - email@example.com

Link do Projeto: [https://github.com/seu-usuario/bitcoin-analysis-system](https://github.com/seu-usuario/bitcoin-analysis-system)

---

Estou buscando colaboração responsável para fazer este projeto funcionar em sua totalidade. Se você tem experiência em análise de criptomoedas, machine learning, ou desenvolvimento de sistemas de trading, e está interessado em contribuir, por favor, entre em contato. Estou procurando alguém que possa ajudar a refinar os algoritmos, melhorar a precisão das análises e potencialmente expandir o sistema para outras criptomoedas. Toda ajuda é bem-vinda!
