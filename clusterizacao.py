#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clusterização e Análise de Questões da OAB
Autor......: Daniel Talon (adaptado por Gemini)
Criação....: 2025-11-18
Atualização: 2025-11-20 (Adicionado DBSCAN para detecção de outliers)
Descrição..: Carrega os dados vetorizados da OAB e aplica um conjunto de
             algoritmos de clustering (K-Means, GMM, Hierárquico, DBSCAN)
             para agrupar as questões por similaridade, analisar a estrutura
             dos tópicos e identificar questões atípicas (outliers).

Principais pontos do script

1) Carregamento dos dados pré-processados (arquivos .pkl).
2) Funções para determinar o número ótimo de clusters (k) para K-Means/GMM.
3) Treinamento cronometrado e avaliação comparativa de K-Means e GMM.
4) Análise de ambiguidade do GMM para identificar questões interdisciplinares.
5) Geração de dendrograma com Clustering Hierárquico para visualizar a relação entre tópicos.
6) Aplicação do DBSCAN para descobrir clusters baseados em densidade e identificar outliers.
7) Salvamento de todos os resultados (gráficos, relatórios, nuvens de palavras) em uma pasta.

Execução do script:

pip install matplotlib seaborn wordcloud pandas scikit-learn scipy
python clusterizacao.py

"""

import os
import pickle
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import WordCloud

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors


# --- CONSTANTES DE CONFIGURAÇÃO ---
PASTA_DADOS = 'dados_processados'
PASTA_JSON = 'json'
PASTA_RESULTADOS = 'resultados_clusterizacao'
os.makedirs(PASTA_RESULTADOS, exist_ok=True)
ARQUIVO_JSON_ORIGINAL = os.path.join(PASTA_JSON, 'OAB_questoes_versao_curta_para_classificacao.json')
# Parâmetros para a busca de k
# Aumente o range se suspeitar que há mais de 15 áreas do direito
RANGE_K = range(2, 16) 
# Defina o k escolhido após analisar os gráficos.
# Um bom ponto de partida, sabendo que a prova da OAB tem várias disciplinas.
K_ESCOLHIDO = 8

def carregar_dados():
    """Carrega a matriz TF-IDF, o vetorizador e os IDs dos arquivos .pkl."""
    print("Carregando dados pré-processados...")
    with open(os.path.join(PASTA_DADOS, 'matriz_tfidf.pkl'), 'rb') as f:
        matriz_tfidf = pickle.load(f)
    with open(os.path.join(PASTA_DADOS, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(PASTA_DADOS, 'ids_questoes.pkl'), 'rb') as f:
        ids_questoes = pickle.load(f)
    with open(ARQUIVO_JSON_ORIGINAL, 'r', encoding='utf-8') as f:
        questoes_originais = json.load(f)
        # Cria um dicionário para busca rápida da questão pelo ID
        mapa_questoes = {q['id_questao']: q for q in questoes_originais}

    print(f"Dados carregados. Formato da matriz: {matriz_tfidf.shape}")
    return matriz_tfidf, vectorizer, ids_questoes, mapa_questoes

def encontrar_k_otimo(matriz_tfidf):
    """Calcula e plota o Método do Cotovelo e a Análise de Silhueta."""
    print("\nIniciando busca pelo número ideal de clusters (k)...")
    inercia = []
    silhueta_scores = []

    for k in RANGE_K:
        print(f"  Testando k={k}...")
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(matriz_tfidf)
        inercia.append(kmeans.inertia_)
        silhueta_scores.append(silhouette_score(matriz_tfidf, kmeans.labels_))

    # Plotando o Método do Cotovelo
    plt.figure(figsize=(10, 5))
    plt.plot(RANGE_K, inercia, 'bo-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inércia (WCSS)')
    plt.title('Método do Cotovelo para Determinar o k Ideal')
    plt.grid(True)
    caminho_cotovelo = os.path.join(PASTA_RESULTADOS, 'metodo_cotovelo.png')
    plt.savefig(caminho_cotovelo)
    plt.close()
    print(f"Gráfico do Método do Cotovelo salvo em: {caminho_cotovelo}")
    
    # Plotando a Análise de Silhueta
    plt.figure(figsize=(10, 5))
    plt.plot(RANGE_K, silhueta_scores, 'ro-')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Coeficiente de Silhueta')
    plt.title('Análise de Silhueta para Determinar o k Ideal')
    plt.grid(True)
    caminho_silhueta = os.path.join(PASTA_RESULTADOS, 'analise_silhueta.png')
    plt.savefig(caminho_silhueta)
    plt.close()
    print(f"Gráfico da Análise de Silhueta salvo em: {caminho_silhueta}")

def treinar_e_avaliar_modelos(matriz_tfidf, k):
    """Treina K-Means e GMM, cronometra e avalia os resultados."""
    print(f"\nTreinando e avaliando modelos com k={k}...")
    resultados = {}

    # --- K-Means ---
    print("  Treinando K-Means...")
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    start_time = time.time()
    kmeans_labels = kmeans.fit_predict(matriz_tfidf)
    end_time = time.time()
    
    resultados['K-Means'] = {
        'labels': kmeans_labels,
        'model': kmeans,
        'tempo_execucao': end_time - start_time,
        'silhouette': silhouette_score(matriz_tfidf, kmeans_labels),
        'calinski_harabasz': calinski_harabasz_score(matriz_tfidf.toarray(), kmeans_labels),
        'davies_bouldin': davies_bouldin_score(matriz_tfidf.toarray(), kmeans_labels),
    }
    print("  K-Means concluído.")

    # --- Gaussian Mixture Model (GMM) ---
    print("  Treinando GMM...")
    gmm = GaussianMixture(n_components=k, random_state=42)
    start_time = time.time()
    gmm_labels = gmm.fit_predict(matriz_tfidf.toarray())
    end_time = time.time()

    resultados['GMM'] = {
        'labels': gmm_labels,
        'model': gmm,
        'tempo_execucao': end_time - start_time,
        'silhouette': silhouette_score(matriz_tfidf, gmm_labels),
        'calinski_harabasz': calinski_harabasz_score(matriz_tfidf.toarray(), gmm_labels),
        'davies_bouldin': davies_bouldin_score(matriz_tfidf.toarray(), gmm_labels),
    }
    print("  GMM concluído.")
    
    return resultados

def analisar_e_salvar_clusters(resultados, vectorizer, ids_questoes, mapa_questoes):
    """Gera relatórios de texto e nuvens de palavras para cada cluster."""
    print("\nGerando análise qualitativa dos clusters...")
    
    termos = vectorizer.get_feature_names_out()

    for nome_modelo, data in resultados.items():
        print(f"  Analisando clusters do modelo: {nome_modelo}")
        caminho_relatorio = os.path.join(PASTA_RESULTADOS, f'analise_{nome_modelo.lower()}.txt')
        
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            f.write(f"--- ANÁLISE DOS CLUSTERS - MODELO: {nome_modelo} ---\n\n")
            
            labels = data['labels']
            model = data['model']
            
            for i in range(model.n_clusters if nome_modelo == 'K-Means' else model.n_components):
                f.write(f"====================\n")
                f.write(f"   CLUSTER {i}\n")
                f.write(f"====================\n")

                # Extrair palavras-chave
                if nome_modelo == 'K-Means':
                    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
                    palavras_chave = [termos[ind] for ind in order_centroids[i, :15]]
                else: # GMM
                    # Para GMM, pegamos os termos com maior probabilidade média no cluster
                    component_probs = model.means_[i]
                    top_indices = component_probs.argsort()[::-1][:15]
                    palavras_chave = [termos[idx] for idx in top_indices]

                f.write(f"Palavras-chave: {', '.join(palavras_chave)}\n\n")

                # Amostra de questões do cluster
                ids_no_cluster = [ids_questoes[j] for j, label in enumerate(labels) if label == i]
                f.write(f"Número de questões no cluster: {len(ids_no_cluster)}\n")
                f.write("Amostra de 5 enunciados:\n")
                for q_id in ids_no_cluster[:5]:
                    enunciado_curto = mapa_questoes[q_id]['enunciado'][:200] + '...'
                    f.write(f"  - ({q_id}): {enunciado_curto}\n")
                f.write("\n")
                
                # Gerar e salvar nuvem de palavras
                wordcloud = WordCloud(
                    background_color='white',
                    width=800,
                    height=400,
                    max_words=50
                ).generate(" ".join(palavras_chave))
                
                caminho_nuvem = os.path.join(PASTA_RESULTADOS, f'{nome_modelo.lower()}_cluster_{i}.png')
                wordcloud.to_file(caminho_nuvem)

        print(f"  Relatório e nuvens de palavras salvos para {nome_modelo}.")

def analisar_ambiguidade_gmm(gmm_model, matriz_densa, ids_questoes, mapa_questoes, top_n=10):
    """
    Analisa as probabilidades do GMM para identificar as questões mais ambíguas.
    """
    print("\n--- Iniciando Análise de Ambiguidade do GMM ---")
    
    # 1. Obter a matriz de probabilidades
    probabilidades = gmm_model.predict_proba(matriz_densa)
    
    # 2. Calcular a confiança (probabilidade máxima) para cada questão
    confianca_por_questao = np.max(probabilidades, axis=1)
    
    # 3. Calcular a confiança média do modelo
    confianca_media_geral = np.mean(confianca_por_questao)
    
    # 4. Encontrar os índices das 'top_n' questões mais ambíguas (menor confiança)
    indices_ambiguos = np.argsort(confianca_por_questao)[:top_n]
    
    # 5. Gerar e salvar o relatório
    caminho_relatorio = os.path.join(PASTA_RESULTADOS, 'analise_ambiguidade_gmm.txt')
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("--- ANÁLISE DE AMBIGUIDADE E CONFIANÇA DO MODELO GMM ---\n\n")
        f.write(f"Confiança média do modelo em suas atribuições: {confianca_media_geral:.2%}\n")
        f.write("(Este valor representa a probabilidade média da atribuição ao cluster mais provável)\n\n")
        
        f.write(f"--- As {top_n} Questões Mais Ambíguas (Potencialmente Interdisciplinares) ---\n")
        f.write("(Questões com a menor confiança na atribuição a um único cluster)\n\n")
        
        for idx in indices_ambiguos:
            id_questao = ids_questoes[idx]
            confianca = confianca_por_questao[idx]
            enunciado = mapa_questoes.get(id_questao, {}).get('enunciado', 'N/A')
            
            # Pega as probabilidades para esta questão e ordena
            probs_questao = probabilidades[idx]
            indices_prob_ordenados = np.argsort(probs_questao)[::-1]
            
            f.write(f"ID da Questão: {id_questao}\n")
            f.write(f"Confiança no Cluster Principal: {confianca:.2%}\n")
            f.write("Distribuição de Probabilidades (Top 3 Clusters):\n")
            for i in range(min(3, len(probs_questao))):
                cluster_idx = indices_prob_ordenados[i]
                prob = probs_questao[cluster_idx]
                f.write(f"  - Cluster {cluster_idx}: {prob:.2%}\n")
            
            f.write(f"Enunciado: {enunciado[:300]}...\n")
            f.write("-" * 50 + "\n\n")
            
    print(f"Relatório de ambiguidade salvo em: {caminho_relatorio}")

def executar_hierarquico_e_plotar_dendrograma(matriz_densa):
    """
    Executa o clustering hierárquico e gera um dendrograma para visualização.
    """
    print("\n--- Executando Clustering Hierárquico ---")
    
    # 1. Calcular a matriz de ligação (linkage matrix)
    # O método 'ward' minimiza a variância das fusões de clusters. É uma escolha robusta.
    start_time = time.time()
    matriz_ligacao = linkage(matriz_densa, method='ward')
    end_time = time.time()
    
    print(f"Matriz de ligação calculada em {end_time - start_time:.2f} segundos.")
    
    # 2. Plotar o dendrograma
    plt.figure(figsize=(15, 7))
    plt.title('Dendrograma do Clustering Hierárquico')
    plt.xlabel('Questões (índice)')
    plt.ylabel('Distância (Ward)')
    
    # O dendrograma completo com 158 folhas seria ilegível.
    # 'truncate_mode' e 'p' mostram apenas as últimas 'p-1' fusões,
    # o que revela a estrutura de alto nível dos clusters.
    dendrogram(
        matriz_ligacao,
        truncate_mode='lastp',  # Mostra apenas os últimos p clusters fundidos
        p=12,                   # O número de clusters finais a serem mostrados (ajuste se necessário)
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,   # Para representar um cluster por um único nó
    )
    
    plt.grid(axis='y')
    caminho_dendrograma = os.path.join(PASTA_RESULTADOS, 'dendrograma_hierarquico.png')
    plt.savefig(caminho_dendrograma)
    plt.close()
    
    print(f"Dendrograma salvo em: {caminho_dendrograma}")

def executar_dbscan_e_analisar_outliers(matriz_densa, ids_questoes, mapa_questoes):
    """
    Executa o DBSCAN para encontrar clusters baseados em densidade e identificar outliers.
    """
    print("\n--- Executando DBSCAN para Detecção de Outliers ---")
    
    # Heurística para encontrar um 'eps' inicial:
    # Calcula a distância para os k-vizinhos mais próximos (k=min_samples)
    # e plota. O "cotovelo" nesse gráfico é um bom candidato para 'eps'.
    # min_samples geralmente é 2 * n_dimensões, mas para texto podemos usar um valor menor.
    min_samples = 5
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(matriz_densa)
    distances, indices = nbrs.kneighbors(matriz_densa)
    
    # Pega a distância do vizinho mais distante e ordena
    k_distances = np.sort(distances[:, -1])
    
    # Plot do "K-distance graph" para ajudar a escolher o eps
    plt.figure(figsize=(10, 5))
    plt.plot(k_distances)
    plt.title('Gráfico de Distância K-NN (ajuda a escolher o "eps" do DBSCAN)')
    plt.xlabel("Pontos ordenados pela distância ao k-ésimo vizinho")
    plt.ylabel(f"Distância ao {min_samples}º vizinho")
    plt.grid(True)
    plt.savefig(os.path.join(PASTA_RESULTADOS, 'dbscan_eps_finder.png'))
    plt.close()
    
    # Com base no gráfico, um "cotovelo" pode aparecer. Vamos chutar um valor inicial.
    # Se a curva for suave, pode precisar de ajuste manual.
    #eps_estimado = 1.3 # VALOR PARA AJUSTAR APÓS OLHAR O GRÁFICO
    eps_estimado = 1.28

    print(f"Gráfico para escolha do 'eps' salvo. Estimando um valor inicial de eps={eps_estimado}")

    # Executar o DBSCAN
    dbscan = DBSCAN(eps=eps_estimado, min_samples=min_samples)
    start_time = time.time()
    dbscan_labels = dbscan.fit_predict(matriz_densa)
    end_time = time.time()
    
    # Análise dos resultados
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_outliers = np.sum(dbscan_labels == -1)
    
    caminho_relatorio = os.path.join(PASTA_RESULTADOS, 'analise_outliers_dbscan.txt')
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write("--- ANÁLISE DE OUTLIERS COM DBSCAN ---\n\n")
        f.write(f"Parâmetros usados: eps={eps_estimado}, min_samples={min_samples}\n")
        f.write(f"Tempo de execução: {end_time - start_time:.4f} segundos\n\n")
        f.write(f"Número de clusters encontrados: {n_clusters}\n")
        f.write(f"Número de outliers (ruído) identificados: {n_outliers}\n\n")
        
        if n_outliers > 0:
            f.write("--- Enunciados das Questões Identificadas como Outliers ---\n\n")
            outlier_indices = np.where(dbscan_labels == -1)[0]
            for idx in outlier_indices:
                id_questao = ids_questoes[idx]
                enunciado = mapa_questoes.get(id_questao, {}).get('enunciado', 'N/A')
                f.write(f"ID da Questão: {id_questao}\n")
                f.write(f"Enunciado: {enunciado}\n")
                f.write("-" * 50 + "\n\n")

    print(f"Análise do DBSCAN concluída. Relatório de outliers salvo.")

def main():
    """Função principal que orquestra todo o processo."""
    matriz_tfidf, vectorizer, ids_questoes, mapa_questoes = carregar_dados()
    matriz_densa = matriz_tfidf.toarray() # Convertendo para densa para GMM e outras análises
    
    # Etapa 1: Encontrar o k ideal (gera gráficos para análise manual)
    encontrar_k_otimo(matriz_tfidf)
    print(f"\nGráficos gerados. Analise-os na pasta '{PASTA_RESULTADOS}' para escolher o melhor 'k'.")
    print(f"Continuando com o valor pré-definido K_ESCOLHIDO = {K_ESCOLHIDO}")
    
    # Etapa 2: Treinar, cronometrar e avaliar os modelos com o k escolhido
    resultados = treinar_e_avaliar_modelos(matriz_tfidf, K_ESCOLHIDO)
    
    # Etapa 3: Exibir tabela comparativa dos resultados
    df_resultados = pd.DataFrame(resultados).T
    print("\n--- Tabela Comparativa de Resultados ---\n")
    # Removendo colunas que não são métricas para a tabela
    print(df_resultados[['tempo_execucao', 'silhouette', 'calinski_harabasz', 'davies_bouldin']].round(4))
    
    # Etapa 4: Análise qualitativa (palavras-chave, exemplos, nuvens de palavras)
    analisar_e_salvar_clusters(resultados, vectorizer, ids_questoes, mapa_questoes)
    
    # Executa a nova análise de ambiguidade para o modelo GMM
    analisar_ambiguidade_gmm(
        gmm_model=resultados['GMM']['model'],
        matriz_densa=matriz_densa,
        ids_questoes=ids_questoes,
        mapa_questoes=mapa_questoes,
        top_n=10  # Número de questões ambíguas a serem listadas
    )

    executar_hierarquico_e_plotar_dendrograma(matriz_densa)

    executar_dbscan_e_analisar_outliers(matriz_densa, ids_questoes, mapa_questoes)

    print(f"\nProcesso de clusterização concluído!")
    print(f"Todos os resultados foram salvos na pasta: '{PASTA_RESULTADOS}'")

if __name__ == "__main__":
    main()