#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para inspecionar rapidamente os arquivos PKL gerados.

Autor......: Daniel Talon (adaptado por Gemini)
Criação....: 2025-11-19
Atualização: NA
Descrição..: NA

pip install scipy

"""

import pickle
import os

# Caminho para a pasta
PASTA_DADOS = 'dados_processados'

# 1. Inspecionar a matriz TF-IDF
with open(os.path.join(PASTA_DADOS, 'matriz_tfidf.pkl'), 'rb') as f:
    matriz = pickle.load(f)

print(f"Tipo da matriz: {type(matriz)}")
# <class 'scipy.sparse._csr.csr_matrix'> -> É uma matriz esparsa, eficiente para muitos zeros.

print(f"Formato da matriz: {matriz.shape}")
# (158, 1661)

# Para ver uma pequena parte dela (convertendo para uma matriz densa)
print("Amostra da matriz (primeiras 5 linhas, 10 colunas):")
print(matriz.toarray()[:5, :10])
# Você verá uma matriz de números (muitos zeros e alguns valores TF-IDF).

# 2. Inspecionar o vetorizador
with open(os.path.join(PASTA_DADOS, 'vectorizer.pkl'), 'rb') as f:
    vetorizador = pickle.load(f)

print(f"\nTipo do vetorizador: {type(vetorizador)}")
# <class 'sklearn.feature_extraction.text.TfidfVectorizer'>

# Para ver as palavras (features) que ele aprendeu
features = vetorizador.get_feature_names_out()
print(f"Número de features: {len(features)}")
# 1661

print("As 10 primeiras features (palavras) são:")
print(features[:10])
# ['abandono', 'abatimento', 'abc', 'aberto', ...]

# 3. Inspecionar os IDs
with open(os.path.join(PASTA_DADOS, 'ids_questoes.pkl'), 'rb') as f:
    ids = pickle.load(f)

print(f"\nTipo dos IDs: {type(ids)}")
# <class 'list'>

print(f"Total de IDs: {len(ids)}")
# 158

print("Os 5 primeiros IDs são:")
print(ids[:5])
# ['P44_T1_Q1', 'P44_T1_Q2', ...]