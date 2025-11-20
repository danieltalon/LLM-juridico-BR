#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de JSON com as questões do Exame de Ordem
Autor......: Daniel Talon
Criação....: 2025-11-09
Atualização: 2025-11-13 (parser de PDF melhorado)
Descrição..: Lê cadernos de questões (Tipo 1) e respectivos gabaritos
             preliminar e definitivo, produzindo arquivo JSON estruturado.

Principais pontos do script

TO-DO
1) Constantes configuráveis (caminhos, exames a processar, etc.).
2) Logging: todas as mensagens são enviadas ao log e, opcionalmente, à tela.

OK
3) Descoberta automática dos trios de arquivos na pasta.
4) Leitura de gabaritos e questões com `pdfplumber` para maior precisão.
5) Parser de questões robusto que lida com layout de 2 colunas.
6) Geração do JSON com controle de anuladas e divergências.
7) Relatório final de execução.

pip install pdfplumber
python extracao_PDF_geracao_JSON.py

"""

import pdfplumber
import os
import re
import json
from datetime import datetime

# --- CONFIGURAÇÃO PRINCIPAL ---
EXAMES_A_PROCESSAR = [44, 43] # Adicione os números dos exames aqui

# --- Funções Auxiliares ---

def extrair_metadados_questoes(caminho_pdf):
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            texto_pagina_1 = pdf.pages[0].extract_text()
            exame_id_match = re.search(r'(\d+)[ºo\.] EXAME', texto_pagina_1, re.IGNORECASE)
            tipo_prova_match = re.search(r'(TIPO \d+)', texto_pagina_1)
            exame_id = int(exame_id_match.group(1)) if exame_id_match else None
            tipo_prova_fonte = tipo_prova_match.group(1) if tipo_prova_match else None
            tipo_prova_num = int(re.search(r'\d+', tipo_prova_fonte).group(0)) if tipo_prova_fonte else None
            return {"exame_id": exame_id, "tipo_prova_num": tipo_prova_num, "tipo_prova_fonte": tipo_prova_fonte}
    except Exception as e:
        print(f"AVISO: Falha ao extrair metadados de '{os.path.basename(caminho_pdf)}'. Erro: {e}")
        return {}

def extrair_metadados_gabarito(caminho_pdf):
    if not os.path.exists(caminho_pdf):
        print(f"AVISO: Arquivo de gabarito '{os.path.basename(caminho_pdf)}' não encontrado.")
        return {}
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            texto_pagina_1 = pdf.pages[0].extract_text()
            # <<< CORREÇÃO DA DATA APLICADA AQUI >>>
            # Altera de \d{2} para \d{1,2} para aceitar meses/dias com um ou dois dígitos.
            data_match = re.search(r'(\d{1,2}/\d{1,2}/\d{4})', texto_pagina_1)
            if data_match:
                data_str = data_match.group(1)
                data_obj = datetime.strptime(data_str, '%d/%m/%Y')
                return {"data_prova_1a_fase": data_obj.strftime('%Y-%m-%d'), "ano": data_obj.year}
    except Exception as e:
        print(f"AVISO: Falha ao extrair metadados de '{os.path.basename(caminho_pdf)}'. Erro: {e}")
    return {}

def parse_gabarito_pdf(caminho_pdf):
    """
    # LÓGICA FINAL E MAIS ROBUSTA: Divide o texto em blocos e depois processa o bloco do Tipo 1.
    Lê um PDF de gabarito e retorna um dicionário de respostas para a Prova Tipo 1.
    """
    respostas = {}
    if not os.path.exists(caminho_pdf): return None
    
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            texto_pagina = pdf.pages[0].extract_text(x_tolerance=2, y_tolerance=2) or ""
            
            # 1. Usa um "lookahead" para dividir o texto em blocos, um para cada TIPO de prova.
            # O `re.split` vai quebrar o texto ANTES de cada ocorrência de "PROVA TIPO".
            blocos = re.split(r'(?=.*PROVA TIPO \d)', texto_pagina)
            
            bloco_tipo_1_texto = ""
            for bloco in blocos:
                # 2. Encontra o bloco que pertence ao Tipo 1.
                # Usamos .strip() para remover espaços/linhas em branco no início.
                if bloco.strip().startswith("PROVA TIPO 1") or re.search(r'PROVA TIPO 1', bloco):
                    bloco_tipo_1_texto = bloco
                    break
            
            if not bloco_tipo_1_texto:
                print(f"AVISO CRÍTICO: Bloco 'PROVA TIPO 1' não foi isolado em {os.path.basename(caminho_pdf)}.")
                return {}

            # 3. Remove a linha do cabeçalho para não interferir na leitura das linhas de resposta.
            conteudo_tabela = re.sub(r'.*PROVA TIPO 1\n', '', bloco_tipo_1_texto, count=1).strip()
            
            linhas = conteudo_tabela.split('\n')
            
            # 4. Processa as linhas da tabela isolada.
            for i in range(0, len(linhas) - 1, 2):
                # Limpa qualquer texto residual (como o número do exame) no início da linha de números
                linha_numeros_limpa = re.sub(r'^\D*\s*', '', linhas[i].strip())
                
                numeros_linha = re.split(r'\s+', linha_numeros_limpa)
                letras_linha = re.split(r'\s+', linhas[i+1].strip())
                
                if len(numeros_linha) == len(letras_linha):
                    for num, letra in zip(numeros_linha, letras_linha):
                        if num.isdigit() and (letra in ['A', 'B', 'C', 'D', '*']):
                            respostas[num] = letra
                
    except Exception as e:
        print(f"ERRO INESPERADO ao analisar gabarito '{os.path.basename(caminho_pdf)}'. Erro: {e}")
        return {}
        
    return respostas

# --- Função de Processamento para um Único Exame ---
def processar_uma_prova(numero_exame, pasta_pdf="OAB_PDF"):
    print(f"\n{'='*20} INICIANDO PROCESSAMENTO DO EXAME Nº {numero_exame} {'='*20}")
    arquivo_questoes = os.path.join(pasta_pdf, f"OAB{numero_exame} Questoes Tipo 1.pdf")
    arquivo_gabarito_prel = os.path.join(pasta_pdf, f"OAB{numero_exame} Gabarito preliminar.pdf")
    arquivo_gabarito_def = os.path.join(pasta_pdf, f"OAB{numero_exame} Gabarito definitivo.pdf")

    if not os.path.exists(arquivo_questoes):
        print(f"AVISO: Arquivo de questões '{os.path.basename(arquivo_questoes)}' não encontrado. Pulando exame {numero_exame}.")
        return []

    meta_questoes = extrair_metadados_questoes(arquivo_questoes)
    meta_gabarito = extrair_metadados_gabarito(arquivo_gabarito_prel)
    gabarito_preliminar = parse_gabarito_pdf(arquivo_gabarito_prel) or {}
    gabarito_definitivo = parse_gabarito_pdf(arquivo_gabarito_def)
    status_gabarito = "DEFINITIVO" if gabarito_definitivo is not None else "PRELIMINAR"
    if gabarito_definitivo is None: gabarito_definitivo = {}

    texto_completo_prova = ""
    with pdfplumber.open(arquivo_questoes) as pdf:
        paginas_a_ler = pdf.pages[2:21]
        for page in paginas_a_ler:
            meio = page.width / 2
            texto_esq = page.crop((0, 0, meio, page.height)).extract_text(x_tolerance=2)
            texto_dir = page.crop((meio, 0, page.width, page.height)).extract_text(x_tolerance=2)
            if texto_esq: texto_completo_prova += texto_esq + "\n"
            if texto_dir: texto_completo_prova += texto_dir + "\n"

    lista_de_questoes_do_exame = []
    padrao_questoes = r'(\d{1,2})\n(.*?)(?=\n\d{1,2}\n|$)'
    questoes_encontradas = re.findall(padrao_questoes, texto_completo_prova, re.DOTALL)
    
    for num_str, texto_bruto in questoes_encontradas:
        partes = re.split(r'\((A|B|C|D)\)', texto_bruto)
        if len(partes) < 9: continue
        enunciado = ' '.join(partes[0].strip().split())
        alternativas = {partes[i]: ' '.join(partes[i+1].strip().split()) for i in range(1, len(partes), 2)}
        letra_prel = gabarito_preliminar.get(num_str, "")
        letra_def = gabarito_definitivo.get(num_str, letra_prel)
        status_questao = "ANULADA" if letra_def == '*' else "MANTIDA"
        texto_prel = alternativas.get(letra_prel, "")
        texto_def = "Questão Anulada" if status_questao == "ANULADA" else alternativas.get(letra_def, "")
        questao_json = {
            "id_questao": f"P{meta_questoes.get('exame_id', numero_exame)}_T{meta_questoes.get('tipo_prova_num', 1)}_Q{num_str}",
            "tipo_prova_fonte": meta_questoes.get("tipo_prova_fonte", ""), "exame_id": meta_questoes.get("exame_id", numero_exame),
            "ano": meta_gabarito.get("ano", None), "data_prova_1a_fase": meta_gabarito.get("data_prova_1a_fase", None),
            "disciplina": "A ser preenchido", "enunciado": enunciado, "alternativas": alternativas,
            "resposta_correta_preliminar_letra": letra_prel, "resposta_correta_preliminar_texto": texto_prel,
            "resposta_correta_definitiva_letra": letra_def, "resposta_correta_definitiva_texto": texto_def,
            "status_gabarito": status_gabarito, "status_questao": status_questao
        }
        lista_de_questoes_do_exame.append(questao_json)
    
    print(f"Exame {numero_exame}: {len(lista_de_questoes_do_exame)} questões processadas.")
    return lista_de_questoes_do_exame

# --- Função Principal de Orquestração ---
def processar_todas_as_provas():
    todas_as_questoes_consolidadas = []
    for numero_exame in EXAMES_A_PROCESSAR:
        questoes_do_exame = processar_uma_prova(numero_exame)
        if questoes_do_exame:
            todas_as_questoes_consolidadas.extend(questoes_do_exame)
    if todas_as_questoes_consolidadas:
        pasta_saida_json = "json"
        nome_arquivo_json = "oab_questoes.json"
        os.makedirs(pasta_saida_json, exist_ok=True)
        caminho_json_saida = os.path.join(pasta_saida_json, nome_arquivo_json)
        with open(caminho_json_saida, 'w', encoding='utf-8') as f:
            json.dump(todas_as_questoes_consolidadas, f, ensure_ascii=False, indent=4)
        print("\n" + "="*50)
        print("PROCESSO GERAL CONCLUÍDO COM SUCESSO!")
        print(f"Arquivo JSON consolidado salvo em: '{caminho_json_saida}'")
        print(f"Total de exames processados: {len(EXAMES_A_PROCESSAR)}")
        print(f"Total de questões salvas no arquivo: {len(todas_as_questoes_consolidadas)}")
        print("="*50)
    else:
        print("\nNenhuma questão foi processada. Verifique os nomes dos arquivos e a lista de exames.")

if __name__ == "__main__":
    processar_todas_as_provas()