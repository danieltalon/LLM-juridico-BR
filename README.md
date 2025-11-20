# LLM-juridico-BR
Preparação de dados das provas da OAB e treinamento de LLM para questões jurídicas do Brasil.
O arquivo estavel que funciona com os exames 43 e 44 é o extracao_PDF_geracao_JSON.py
Após a extração, utiliza-se processa_json.py para gerar um JSON mais enxuto apenas com o titulo e as alternativas das questões.
Então criam-se os arquivos PKL (matriz, vetor e IDs) com vetoriza_questoes.py.
Por fim, utiliza-se clusterizacao.py para gerar as analises e comparar os algoritmos K-means e GMM.