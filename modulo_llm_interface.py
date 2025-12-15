# Este módulo faz a integração entre o sistema e a API da OpenAI usando o modelo GPT-4o-mini.
# Ele é responsável por conversar com o paciente e tentar preencher os valores das variáveis necessárias
# para acionar o módulo da rede neural.
# O código está todo comentado para que todos os integrantes do grupo entendam a lógica e todas as alterações.
# Para que tudo funcione, deve-se definir a chave da OpenAI como variável de ambiente (mais seguro do que colocar a chave direto no código):
# - No Windows (PowerShell): $env:OPENAI_API_KEY="sua-chave-aqui"
# - No CMD: set OPENAI_API_KEY=sua_chave_aqui
# - No Linux/Mac: export OPENAI_API_KEY=sua_chave_aqui

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Só para não esquecer de setar a chave
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("A variável de ambiente OPENAI_API_KEY não foi definida!")

# Nome do modelo que será usado pelo chatbot (escolhemos esse devido ao bom desempenho aliado ao menor custo)
MODELO_PADRAO = "gpt-4o-mini"

def perguntar_llm(
    mensagem_usuario: str, # Texto enviado pelo usuário
    instrucao_sistema: str | None = None, # Regras de comportamento do modelo (persona)
    modelo: str = MODELO_PADRAO,
) -> str: # Texto da resposta gerada pelo modelo

    # Montamos a lista de mensagens no formato da API da OpenAI.
    # A API exige este formato:
    # [
    #   {"role": "system", "content": "..."},
    #   {"role": "user", "content": "..."}
    # ]
    
    mensagens = []

    if instrucao_sistema:
        mensagens.append({"role": "system", "content": instrucao_sistema})

    mensagens.append({"role": "user", "content": mensagem_usuario})

    # Chamando o modelito
    resposta = client.chat.completions.create(
        model=modelo,
        messages=mensagens,
        temperature=0.2,        # Respostas mais consistentes
        max_tokens=300,         # Limite da resposta (para economizar, pois ainda não ganhamos na mega sena)
    )

    # Resposta do modelo
    texto = resposta.choices[0].message.content
    return texto

# A ideia aqui é enviar uma mensagem ao LLM e obrigar o modelo a responder com JSON válido.
# Retorna um dicionário Python (resultado de json.loads sobre a resposta).
def perguntar_llm_json(
    mensagem_usuario: str,
    instrucao_sistema: str | None = None,
    modelo: str = MODELO_PADRAO,
) -> dict:
    mensagens = []

    if instrucao_sistema:
        mensagens.append({"role": "system", "content": instrucao_sistema})

    mensagens.append({"role": "user", "content": mensagem_usuario})

    resposta = client.chat.completions.create(
        model=modelo,
        messages=mensagens,
        temperature=0.2,
        max_tokens=800,
        response_format={"type": "json_object"}, 
    )

    conteudo = resposta.choices[0].message.content

    try:
        dados = json.loads(conteudo)
        return dados
    except json.JSONDecodeError as e:
        # Se, por algum motivo, não conseguir interpretar o JSON.
        raise RuntimeError(
            f"Falha ao interpretar JSON retornado pelo modelo: {e}\nResposta bruta: {conteudo}"
        )

# Teste basicão só para testar antes de prosseguir
if __name__ == "__main__":
    print("\nTestando integração com GPT-4o-mini (texto livre)...\n")
    texto = perguntar_llm(
        mensagem_usuario="Explique o que são redes neurais em 3 linhas.",
        instrucao_sistema="Explique como professor universitário, em PT-BR.",
    )
    print(texto)

    print("\nTestando integração com GPT-4o-mini (JSON)...\n")
    dados = perguntar_llm_json(
        mensagem_usuario=(
            "Crie um JSON com dois campos: "
            "'mensagem' (string em PT-BR) e 'valor' (número 42)."
        ),
        instrucao_sistema="Responda apenas em JSON.",
    )
    print(dados)
