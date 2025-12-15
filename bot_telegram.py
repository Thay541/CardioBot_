# Bot do Telegram usando polling + integração com:
# SessaoDiagnosticoTextoLivre (orquestrador)
# GPT-4o-mini (OpenAI API) via modulo_llm_interface.py
# Rede neural (modulo_nn.py)

# Cada usuário (chat_id) tem sua própria sessão.

import os
import telebot  # pip install pyTelegramBotAPI
from orquestrador import SessaoDiagnosticoTextoLivre

# Não vamos colocar o token aqui por segurança
# O token será setado na variável de ambiente
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN não definido nas variáveis de ambiente.")

bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Dicionário que guarda uma sessão por usuário
user_sessions = {}

# Cria uma nova sessão
def iniciar_sessao(chat_id: int):
    sessao = SessaoDiagnosticoTextoLivre()
    user_sessions[chat_id] = sessao

    mensagem_inicial = (
        "Oi! Eu sou o CardioBot! :)\n"
        "Eu posso te ajudar a ter uma noção bem geral do seu risco cardiovascular nos próximos anos!\n"
        "Lembrando que eu sou apenas um assistente e que não posso substituir uma consulta médica real.\n"
        "Pra começar, sem pressa: como você tem se sentido ultimamente? O que te trouxe aqui hoje?"
    )

    bot.send_message(chat_id, mensagem_inicial, parse_mode="Markdown")

# Handler para o comando /start
@bot.message_handler(commands=["start"])
def cmd_start(message):
    chat_id = message.chat.id
    iniciar_sessao(chat_id)

@bot.message_handler(commands=["reiniciar", "reset"])
def cmd_reset(message):
    chat_id = message.chat.id
    if chat_id in user_sessions:
        del user_sessions[chat_id]
    iniciar_sessao(chat_id)

# Handler para qualquer mensagem de texto do usuário
@bot.message_handler(func=lambda message: True)
def tratar_mensagem(message):
    chat_id = message.chat.id
    texto = message.text.strip()

    # Se o usuário não tem sessão ainda, cria uma nova
    if chat_id not in user_sessions:
        iniciar_sessao(chat_id)

    sessao = user_sessions[chat_id]

    # Envia para o orquestrador (LLM + rede neural)
    resposta = sessao.processar_mensagem_usuario(texto)

    # Responde ao usuário
    bot.send_message(chat_id, resposta, parse_mode="Markdown")


# Inicia o polling
print("🤖 Bot Telegram (CardioBot) iniciado com sucesso!")
bot.infinity_polling()
