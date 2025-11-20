#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para imprimir as 10 features de maior peso para uma questão específica.

Autor......: Daniel Talon (adaptado por Gemini)
Criação....: 2025-11-20
Atualização: NA
Descrição..: NA

1.Escolha uma questão (pelo seu índice, de 0 a 157).
2.Pegue a linha correspondente da matriz TF-IDF.
3.Pegue os nomes de todas as features (palavras) do vetorizador.
4.Junte essas duas informações e ordene os pesos do maior para o menor.
5.Mostre as 10 primeiras palavras e seus respectivos pesos.

"""

import pickle
import os
import numpy as np

# --- CONFIGURAÇÃO ---
PASTA_DADOS = 'dados_processados'

# --- CARREGAMENTO DOS DADOS ---
# (Mesma lógica do seu script de verificação)
try:
    with open(os.path.join(PASTA_DADOS, 'matriz_tfidf.pkl'), 'rb') as f:
        matriz_tfidf = pickle.load(f)

    with open(os.path.join(PASTA_DADOS, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(PASTA_DADOS, 'ids_questoes.pkl'), 'rb') as f:
        ids_questoes = pickle.load(f)

except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Certifique-se de que a pasta '{PASTA_DADOS}' existe e contém os arquivos .pkl.")
    print(f"Detalhe do erro: {e}")
    exit()

# --- FUNÇÃO DE ANÁLISE ---
def exibir_top_features_por_questao(indice_questao: int, top_n: int = 10):
    """
    Exibe as 'top_n' palavras (features) com maior peso TF-IDF para uma
    determinada questão.
    
    Args:
        indice_questao (int): O índice da questão a ser analisada (de 0 a N-1).
        top_n (int): O número de palavras-chave a serem exibidas.
    """
    if not (0 <= indice_questao < matriz_tfidf.shape[0]):
        print(f"Erro: Índice da questão inválido. Por favor, escolha um número entre 0 e {matriz_tfidf.shape[0] - 1}.")
        return

    # Pega o ID da questão para referência
    id_questao = ids_questoes[indice_questao]
    print(f"--- Análise da Questão ID: {id_questao} (Índice: {indice_questao}) ---")

    # 1. Pega a linha correspondente à questão na matriz esparsa
    # A matriz é esparsa, então convertemos a linha para um array denso (numpy)
    vetor_questao = matriz_tfidf[indice_questao].toarray().flatten()

    # 2. Pega os nomes de todas as features (palavras)
    nomes_features = vectorizer.get_feature_names_out()

    # 3. Ordena os índices das features pelos seus pesos (do maior para o menor)
    # np.argsort() retorna os índices que ordenariam o array
    indices_ordenados = np.argsort(vetor_questao)[::-1]

    # 4. Exibe as top_n features e seus pesos
    print(f"\nAs {top_n} palavras mais importantes (features) para esta questão são:\n")
    print(f"{'Palavra (Feature)':<20} | {'Peso TF-IDF':<20}")
    print("-" * 43)

    for i in range(top_n):
        indice_feature = indices_ordenados[i]
        
        # Pega o nome da palavra e seu peso
        palavra = nomes_features[indice_feature]
        peso = vetor_questao[indice_feature]

        # Se o peso for 0, significa que já listamos todas as palavras presentes
        if peso == 0:
            break
            
        print(f"{palavra:<20} | {peso:<20.4f}")


# --- EXECUÇÃO DO EXEMPLO ---
if __name__ == "__main__":
    
    # Define o número de questões que queremos analisar
    numero_de_questoes_para_analisar = 10
    
    print(f"=== EXIBINDO AS PALAVRAS MAIS IMPORTANTES PARA AS PRIMEIRAS {numero_de_questoes_para_analisar} QUESTÕES ===")

    # Cria um loop que vai do índice 0 até o índice 9
    # A função range(N) gera números de 0 a N-1
    for i in range(numero_de_questoes_para_analisar):
        exibir_top_features_por_questao(indice_questao=i, top_n=10)
        
        # Adiciona um separador entre as análises de cada questão
        print("\n" + "="*50 + "\n")