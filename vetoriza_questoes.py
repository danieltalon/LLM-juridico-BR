#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-processador e Vetorizador de Questões da OAB para Machine Learning
Autor......: Daniel Talon (adaptado por Gemini)
Criação....: 2025-11-17
Atualização: 2025-11-18 (Adicionada verificação e download automático de pacotes NLTK)
Descrição..: Lê um arquivo JSON contendo questões da OAB, realiza um
             pré-processamento completo do texto e o converte em uma matriz
             vetorial TF-IDF, salvando os resultados para uso em algoritmos de
             aprendizado de máquina.

Principais pontos do script

1) Constantes configuráveis para caminhos de arquivos de entrada e saída.
2) Carregamento de modelos de linguagem (spaCy, NLTK) para processamento em português.
3) Função de pré-processamento que combina enunciado e alternativas para maior riqueza de contexto.
4) Pipeline de limpeza robusto: minúsculas, remoção de pontuação/números, stopwords e lematização.
5) Vetorização do texto limpo em uma matriz TF-IDF usando Scikit-learn.
6) Salvamento dos artefatos processados (matriz, vetorizador, IDs) com `pickle` para uso em etapas futuras.

Instalação de dependências:
pip install scikit-learn nltk spacy tqdm

Download de modelos de linguagem (executar uma vez):
python -m nltk.downloader punkt stopwords
python -m spacy download pt_core_news_sm

Execução do script:
python vetoriza_questoes.py

"""

import json
import os
import re
import pickle
from tqdm import tqdm
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# --- FUNÇÃO DE VERIFICAÇÃO DE DADOS ---
def verificar_e_baixar_nltk_data():
    """Verifica se os pacotes NLTK necessários existem, caso contrário, faz o download."""
    pacotes_necessarios = ['punkt', 'stopwords']
    for pacote in pacotes_necessarios:
        try:
            # Tenta encontrar o pacote. Se não encontrar, lança um LookupError.
            nltk.data.find(f'tokenizers/{pacote}' if pacote == 'punkt' else f'corpora/{pacote}')
            print(f"Pacote NLTK '{pacote}' já está disponível.")
        except LookupError:
            print(f"Pacote NLTK '{pacote}' não encontrado. Fazendo download...")
            nltk.download(pacote)

# --- CONFIGURAÇÃO ---
PASTA_JSON = 'json'
ARQUIVO_ENTRADA = os.path.join(PASTA_JSON, 'OAB_questoes_versao_curta_para_classificacao.json')
PASTA_SAIDA = 'dados_processados'
os.makedirs(PASTA_SAIDA, exist_ok=True)

# --- INICIALIZAÇÃO E VERIFICAÇÃO ---
verificar_e_baixar_nltk_data() # Garante que os dados NLTK estão presentes

print("\nCarregando modelos de linguagem (pode levar um momento)...")
try:
    nlp_spacy = spacy.load('pt_core_news_sm')
except IOError:
    print("Modelo 'pt_core_news_sm' do spaCy não encontrado.")
    print("Para instalar, execute no terminal: python -m spacy download pt_core_news_sm")
    exit()

stopwords_pt = set(stopwords.words('portuguese'))
print("Modelos carregados com sucesso.")

def preprocessar_texto(texto: str) -> str:
    """Executa um pipeline completo de limpeza e normalização em um texto."""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', ' ', texto)
    texto = re.sub(r'\d+', ' ', texto)
    tokens = word_tokenize(texto, language='portuguese')
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords_pt and len(palavra) > 2]
    texto_filtrado = " ".join(tokens_filtrados)
    doc_spacy = nlp_spacy(texto_filtrado)
    lemmas = [token.lemma_ for token in doc_spacy]
    return " ".join(lemmas)

def main():
    """Orquestra o processo de carregamento, pré-processamento e vetorização."""
    print(f"\nCarregando dados de '{ARQUIVO_ENTRADA}'...")
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
        questoes = json.load(f)
    print(f"{len(questoes)} questões carregadas.")

    textos_completos = []
    ids_questoes = []
    
    print("\nCombinando textos das questões...")
    for q in questoes:
        enunciado = q.get('enunciado', '')
        texto_extra = ''
        if q.get('alternativas') and isinstance(q.get('alternativas'), dict):
            texto_extra = " ".join(q.get('alternativas', {}).values())
        else:
            texto_extra = q.get('resposta_correta_definitiva_texto', '')
        texto_completo = f"{enunciado} {texto_extra}"
        textos_completos.append(texto_completo)
        ids_questoes.append(q.get('id_questao'))

    print("\nIniciando pré-processamento dos textos...")
    textos_processados = [preprocessar_texto(texto) for texto in tqdm(textos_completos, desc="Processando Questões")]

    print("\nIniciando vetorização com TF-IDF...")
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=5000)
    matriz_tfidf = vectorizer.fit_transform(textos_processados)
    print("Vetorização concluída.")
    print(f"Formato da matriz TF-IDF: {matriz_tfidf.shape}")

    print(f"\nSalvando resultados na pasta '{PASTA_SAIDA}'...")
    with open(os.path.join(PASTA_SAIDA, 'matriz_tfidf.pkl'), 'wb') as f:
        pickle.dump(matriz_tfidf, f)
    with open(os.path.join(PASTA_SAIDA, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(PASTA_SAIDA, 'ids_questoes.pkl'), 'wb') as f:
        pickle.dump(ids_questoes, f)
    print("Resultados salvos com sucesso!")
    print("\nPróximo passo: Usar os arquivos .pkl para aplicar os algoritmos de clustering.")

if __name__ == "__main__":
    main()