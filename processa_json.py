#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera JSON enxuto para agrupamento
 
Autor......: Daniel Talon
Criação....: 2025-11-15
Atualização: 2025-11-17 
Descrição..: Utiliza o JSON gerado por extracao_PDF_geracao_JSON.py para criar uma versao mais enxuta de JSON contendo somente os campos especificados em CAMPOS_DESEJADOS e que atendam os criterios de FILTROS_DE_CONTEUDO

"""

import json
import os

# --- CONFIGURAÇÃO ---

# 1. CONSTANTE DE ESTRUTURA: Define QUAIS CAMPOS estarão no arquivo de saída.
# Mude para 'True' para incluir o campo ou 'False' para removê-lo.
CAMPOS_DESEJADOS = {
    "id_questao": True,
    "tipo_prova_fonte": False,
    "exame_id": False,
    "ano": False, 
    "data_prova_1a_fase": False,
    "disciplina": False,
    "enunciado": True,
    "alternativas": True,
    "resposta_correta_preliminar_letra": False,
    "resposta_correta_preliminar_texto": False,
    "resposta_correta_definitiva_letra": True,
    "resposta_correta_definitiva_texto": True,
    "status_gabarito": False,
    "status_questao": True # Deixei como True para vermos o filtro de status funcionando
}

# 2. CONSTANTE DE FILTRO: Define QUAIS QUESTÕES serão incluídas no arquivo de saída.
# Deixe o dicionário VAZIO `{}` para não aplicar nenhum filtro.
# Adicione pares "campo": "valor" para filtrar as questões que atendem a TODOS os critérios.
# Exemplos:
# FILTROS_DE_CONTEUDO = {"status_questao": "MANTIDA"} -> Apenas questões mantidas
# FILTROS_DE_CONTEUDO = {"exame_id": 44, "status_questao": "MANTIDA"} -> Apenas do exame 44 E mantidas
# FILTROS_DE_CONTEUDO = {} -> Inclui todas as questões
FILTROS_DE_CONTEUDO = {
    #"ano": 2025,
    "status_questao": "MANTIDA" # questoes que nao foram anuladas e constam como validas no gabarito definitivo
}


# Pasta para ler e salvar os arquivos JSON
PASTA_JSON = 'json'

# Nomes dos arquivos de entrada e saída
ARQUIVO_ENTRADA = os.path.join(PASTA_JSON, 'oab_questoes-44-43.json')
ARQUIVO_SAIDA = os.path.join(PASTA_JSON, 'OAB_questoes_versao_curta_para_classificacao.json')


def atende_aos_filtros(questao, filtros):
    """
    Verifica se uma determinada questão atende a todos os critérios de filtro.
    Retorna True se a questão deve ser incluída, False caso contrário.
    """
    if not filtros:  # Se o dicionário de filtros está vazio, não filtra nada
        return True

    for campo_filtro, valor_esperado in filtros.items():
        # Usa .get() para evitar erro se o campo não existir na questão.
        # Se o campo não existir, a comparação será com None, resultando em False (corretamente).
        if questao.get(campo_filtro) != valor_esperado:
            return False  # Se qualquer um dos filtros falhar, a questão é rejeitada

    return True  # Se passou por todos os filtros, a questão é aceita


def processar_json(arquivo_entrada, arquivo_saida, campos_estrutura, filtros_conteudo):
    """
    Lê um arquivo JSON, filtra as questões com base em seu conteúdo,
    depois filtra os campos de cada questão e salva o resultado.
    """
    try:
        # Garantir que o diretório de dados exista
        if not os.path.exists(PASTA_JSON):
            print(f"Erro: A pasta '{PASTA_JSON}' não foi encontrada.")
            return

        # Ler o arquivo JSON de entrada
        with open(arquivo_entrada, 'r', encoding='utf-8') as f:
            dados_originais = json.load(f)
        total_inicial = len(dados_originais)
        print(f"Arquivo '{arquivo_entrada}' lido com sucesso. Total de {total_inicial} questões.")

        questoes_processadas = []

        # Iterar sobre cada questão original
        for questao_original in dados_originais:
            # Etapa 1: Aplicar o FILTRO DE CONTEÚDO
            if atende_aos_filtros(questao_original, filtros_conteudo):
                # Etapa 2: Se passou no filtro, criar a versão enxuta com os CAMPOS DESEJADOS
                nova_questao = {}
                for campo, manter in campos_estrutura.items():
                    if manter and campo in questao_original:
                        nova_questao[campo] = questao_original[campo]
                questoes_processadas.append(nova_questao)

        total_final = len(questoes_processadas)
        print(f"\nAplicando filtros: {filtros_conteudo if filtros_conteudo else 'Nenhum filtro aplicado.'}")
        print(f"{total_inicial - total_final} questões foram removidas pelo filtro de conteúdo.")
        print(f"{total_final} questões foram incluídas no arquivo de saída.")

        # Salvar a lista de questões processadas no novo arquivo JSON
        with open(arquivo_saida, 'w', encoding='utf-8') as f:
            json.dump(questoes_processadas, f, indent=4, ensure_ascii=False)

        print(f"\nProcesso concluído com sucesso!")
        print(f"O arquivo enxuto foi salvo em: '{arquivo_saida}'")

    except FileNotFoundError:
        print(f"Erro: O arquivo de entrada '{arquivo_entrada}' não foi encontrado.")
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{arquivo_entrada}' não é um JSON válido.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- EXECUÇÃO DO SCRIPT ---
if __name__ == "__main__":
    processar_json(ARQUIVO_ENTRADA, ARQUIVO_SAIDA, CAMPOS_DESEJADOS, FILTROS_DE_CONTEUDO)