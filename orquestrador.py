# Orquestrador de diálogo do CardioBot (Framingham). Coordena a conversa em texto livre
# (LLM como interface/extrator) e aciona um modelo supervisionado (MLP) para estimar
# risco cardiovascular em 10 anos quando houver dados suficientes.

# Regras de conversa (UX):
# - Nas 3 primeiras interações: sem perguntas diretas; apenas acolhimento e convite aberto.
# - A partir da 4ª: no máximo 1 pergunta por turno (uma única feature), com explicação curta.

# Regras de inferência:
# - Rodar rede neural quando TODAS as features prioritárias estiverem preenchidas OU quando >= 8 features no total estiverem preenchidas.
# - Se não atingir >= 8 e todas as features faltantes atingirem limite de tentativas (2), encerra sem inferência.

# Regra específica de tabagismo:
# - Só perguntar cigsPerDay se a pessoa fuma OU já fumou anteriormente.
# - Se "nunca fumou", setar cigsPerDay=0 automaticamente.
# - Se "não fuma atualmente", perguntar primeiro se já fumou antes; só perguntar cigsPerDay se já fumou.

from typing import Dict, List, Optional, Tuple
import re

from modulo_nn import carregar_modelo, predicao_prob_risco
from modulo_llm_interface import perguntar_llm, perguntar_llm_json


class SessaoDiagnosticoTextoLivre:
    def __init__(self) -> None:
        _, feature_names = carregar_modelo()
        self.feature_names: List[str] = feature_names

        # Mantém valores numéricos para o modelo
        self.features: Dict[str, float] = {f: 0.0 for f in self.feature_names}
        self.preenchidas: set[str] = set()

        self.historico: List[Dict[str, str]] = []
        self.finalizada: bool = False

        self.ultimo_risco_prob: Optional[float] = None
        self.ultima_categoria_risco: Optional[str] = None
        self.ultima_explicacao: Optional[str] = None

        self.qtd_mensagens: int = 0
        self.fase_direcionamento: int = 0

        # Controle de repetição por feature
        self.perguntas_por_feature: Dict[str, int] = {f: 0 for f in self.feature_names}

        # Tópico interno (não é feature do Framingham): "já fumou antes?"
        # Serve para evitar perguntar cigsPerDay quando não faz sentido
        self.perguntas_por_feature["smokingHistory"] = 0

        # Estratégia de coleta
        self.min_features_para_inferir: int = 8

        # Features prioritárias
        prioritarias = [
            "currentSmoker",
            "cigsPerDay",
            "totChol",
            "diabetes",
            "sysBP",
            "age",
            "prevalentHyp",
            "glucose",
            "male",
        ]
        self.features_prioritarias: List[str] = [f for f in prioritarias if f in self.feature_names]

        # Heurísticas (para não depender 100% do LLM)
        self._keywords_para_feature: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\bcolesterol\b|\bcolest\.", re.I), "totChol"),
            (re.compile(r"\bpress[aã]o\b|\bpa\b", re.I), "sysBP"),  # conta também diaBP
            (re.compile(r"\bglicose\b|\ba[cç][uú]car no sangue\b", re.I), "glucose"),
            (re.compile(r"\bcigarro(s)?\b|\btabag(ismo|ista)\b", re.I), "currentSmoker"),
            (re.compile(r"\bdiabet(es|e)\b", re.I), "diabetes"),
            (re.compile(r"\bhipertens[aã]o\b|\bpress[aã]o alta\b", re.I), "prevalentHyp"),
            (re.compile(r"\bidade\b|\banos\b", re.I), "age"),
        ]

        self._mapa_niveis = {
            "pressao": {
                "alta": (150.0, 95.0),
                "normal": (120.0, 80.0),
                "baixa": (100.0, 60.0),
            },
            "colesterol": {"alto": 260.0, "normal": 200.0, "baixo": 160.0},
            "glicose": {"alta": 150.0, "normal": 100.0, "baixa": 70.0},
        }

        # Guarda qual feature/tópico foi perguntado no turno anterior (para interpretar respostas curtas tipo "sim"/"não")
        self.ultima_feature_perguntada: Optional[str] = None

        # Estado interno sobre tabagismo anterior:
        # None = desconhecido / 0 = nunca fumou / 1 = já fumou (ex-fumante ou fumante atual)
        self.ever_smoked: Optional[int] = None

    # --------------------------------------------
    # Controle de "faltante"
    # --------------------------------------------

    def _marcar_preenchida(self, feature: str) -> None:
        if feature in self.feature_names:
            self.preenchidas.add(feature)

    def _ta_faltando(self, feature: str) -> bool:
        return feature in self.feature_names and feature not in self.preenchidas

    def _count_preenchidas(self) -> int:
        return len(self.preenchidas)

    def _prioritarias_preenchidas(self) -> bool:
        # Regra especial: cigsPerDay só é obrigatória se ever_smoked=1 (fuma ou já fumou)
        # Se ever_smoked=0 (nunca fumou), cigsPerDay é preenchida automaticamente como 0

        for f in self.features_prioritarias:
            if f == "cigsPerDay":
                if self.ever_smoked == 0:
                    # se nunca fumou, garantimos que cigsPerDay está resolvido
                    if "cigsPerDay" in self.feature_names and self._ta_faltando("cigsPerDay"):
                        self.features["cigsPerDay"] = 0.0
                        self._marcar_preenchida("cigsPerDay")
                    continue
                # se ever_smoked desconhecido e currentSmoker=0, a prioridade vira smokingHistory antes
                if self.ever_smoked is None:
                    # não consideramos "faltando" aqui; a coleta vai resolver via smokingHistory (para não ficar travando)
                    continue

            if self._ta_faltando(f):
                return False
        return True

    def _limites_estourados(self) -> bool:
        # Considera estourado quando:
        # - ainda não atingiu meta
        # - e todas as features faltantes já foram perguntadas 2x
        
        if self._prioritarias_preenchidas() or self._count_preenchidas() >= self.min_features_para_inferir:
            return False

        faltantes = [f for f in self.feature_names if self._ta_faltando(f)]
        if not faltantes:
            return False

        for f in faltantes:
            if self.perguntas_por_feature.get(f, 0) < 2:
                return False

        # e também o tópico smokingHistory, se ele for necessário
        if (
            "currentSmoker" in self.feature_names
            and "currentSmoker" in self.preenchidas
            and self.features.get("currentSmoker", 0.0) == 0.0
            and self.ever_smoked is None
        ):
            if self.perguntas_por_feature.get("smokingHistory", 0) < 2:
                return False

        return True

    # -----------------------------------------------
    # Heurísticas do usuário
    # -----------------------------------------------

    def _aplicar_heuristicas_do_usuario(self, texto_usuario: str) -> None:
        t = (texto_usuario or "").strip().lower()

        # Pressão numérica: "12 por 8", "120/80", "12x8"
        m = re.search(r"(\d{2,3})\s*(?:/|x|por)\s*(\d{2,3})", t)
        if m:
            s = float(m.group(1))
            d = float(m.group(2))
            if s < 40:
                s *= 10.0
            if d < 40:
                d *= 10.0
            if "sysBP" in self.feature_names and self._ta_faltando("sysBP"):
                self.features["sysBP"] = s
                self._marcar_preenchida("sysBP")
            if "diaBP" in self.feature_names and self._ta_faltando("diaBP"):
                self.features["diaBP"] = d
                self._marcar_preenchida("diaBP")

        # Pressão por categoria
        if "press" in t or re.search(r"\bpa\b", t):
            for nivel in ["alta", "normal", "baixa"]:
                if re.search(rf"\b{nivel}\b", t):
                    s, d = self._mapa_niveis["pressao"][nivel]
                    if "sysBP" in self.feature_names and self._ta_faltando("sysBP"):
                        self.features["sysBP"] = s
                        self._marcar_preenchida("sysBP")
                    if "diaBP" in self.feature_names and self._ta_faltando("diaBP"):
                        self.features["diaBP"] = d
                        self._marcar_preenchida("diaBP")
                    break

        # Colesterol por categoria
        if "colesterol" in t:
            for nivel in ["alto", "normal", "baixo"]:
                if re.search(rf"\b{nivel}\b", t):
                    if "totChol" in self.feature_names and self._ta_faltando("totChol"):
                        self.features["totChol"] = self._mapa_niveis["colesterol"][nivel]
                        self._marcar_preenchida("totChol")
                    break

        # Glicose por categoria
        if "glicose" in t or "açucar" in t or "acucar" in t:
            for nivel in ["alta", "normal", "baixa"]:
                if re.search(rf"\b{nivel}\b", t):
                    if "glucose" in self.feature_names and self._ta_faltando("glucose"):
                        self.features["glucose"] = self._mapa_niveis["glicose"][nivel]
                        self._marcar_preenchida("glucose")
                    break

        # -------------------------
        # Tabagismo
        # -------------------------

        # "nunca fumei" (nunca fumou)
        if re.search(r"\bnunca fumei\b|\bnunca fumo\b|\bnunca fui fumante\b", t):
            if "currentSmoker" in self.feature_names and self._ta_faltando("currentSmoker"):
                self.features["currentSmoker"] = 0.0
                self._marcar_preenchida("currentSmoker")
            self.ever_smoked = 0
            if "cigsPerDay" in self.feature_names and self._ta_faltando("cigsPerDay"):
                self.features["cigsPerDay"] = 0.0
                self._marcar_preenchida("cigsPerDay")

        # "parei de fumar" / "já fumei" (ex-fumante)
        if re.search(r"\bparei de fumar\b|\bj[aá]\s*fumei\b|\bfui fumante\b|\bex[-\s]?fumante\b", t):
            if "currentSmoker" in self.feature_names and self._ta_faltando("currentSmoker"):
                self.features["currentSmoker"] = 0.0
                self._marcar_preenchida("currentSmoker")
            self.ever_smoked = 1  # já fumou antes (ex-fumante)

        # "sou fumante" / "fumo" (fumante atual)
        if re.search(r"\bsou fumante\b|\beu fumo\b|\bfumo\b", t):
            if "currentSmoker" in self.feature_names and self._ta_faltando("currentSmoker"):
                self.features["currentSmoker"] = 1.0
                self._marcar_preenchida("currentSmoker")
            self.ever_smoked = 1

        # "não fumo" / "não fumo atualmente" (não fumante atual, mas pode ser ex)
        if re.search(r"\bn[aã]o fumo\b|\bn[aã]o fumo atualmente\b|\bhoje eu n[aã]o fumo\b", t):
            if "currentSmoker" in self.feature_names and self._ta_faltando("currentSmoker"):
                self.features["currentSmoker"] = 0.0
                self._marcar_preenchida("currentSmoker")
            # Aqui não setamos ever_smoked automaticamente, pois pode ser ex-fumante
            # A regra pede perguntar "já fumou anteriormente?" se ever_smoked ainda for None

        # Cigarros por dia (heurística simples)
        # Ex.: "fumo 10 por dia", "10 cigarros"
        m_cigs = re.search(r"\b(\d{1,2})\s*(cigarros|cigarro)\b", t)
        if m_cigs and "cigsPerDay" in self.feature_names and self._ta_faltando("cigsPerDay"):
            n = float(m_cigs.group(1))
            if 0 <= n <= 80:
                self.features["cigsPerDay"] = n
                self._marcar_preenchida("cigsPerDay")
                # Se informou quantidade, então necessariamente já fumou
                self.ever_smoked = 1

        # Hipertensão (prevalentHyp)
        if re.search(r"\bhipertens[aã]o\b|\bpress[aã]o alta\b", t):
            if "prevalentHyp" in self.feature_names and self._ta_faltando("prevalentHyp"):
                self.features["prevalentHyp"] = 1.0
                self._marcar_preenchida("prevalentHyp")

        if re.search(r"\bn[aã]o tenho hipertens[aã]o\b|\bminha press[aã]o é normal\b", t):
            if "prevalentHyp" in self.feature_names and self._ta_faltando("prevalentHyp"):
                self.features["prevalentHyp"] = 0.0
                self._marcar_preenchida("prevalentHyp")

        # Idade: "tenho 52 anos"
        m_age = re.search(r"\btenho\s+(\d{1,3})\s+anos\b", t)
        if m_age and "age" in self.feature_names and self._ta_faltando("age"):
            age = float(m_age.group(1))
            if 10 <= age <= 120:
                self.features["age"] = age
                self._marcar_preenchida("age")

        # Sexo (male)
        if "male" in self.feature_names and self._ta_faltando("male"):
            if re.search(r"\bsou homem\b|\bmasculin", t):
                self.features["male"] = 1.0
                self._marcar_preenchida("male")
            elif re.search(r"\bsou mulher\b|\bfeminin", t):
                self.features["male"] = 0.0
                self._marcar_preenchida("male")

        # Diabetes (heurística por texto)
        if re.search(r"\btenho diabetes\b|\bsou diab[eé]tic", t):
            if "diabetes" in self.feature_names and self._ta_faltando("diabetes"):
                self.features["diabetes"] = 1.0
                self._marcar_preenchida("diabetes")

        if re.search(r"\bn[aã]o tenho diabetes\b|\bnunca tive diabetes\b", t):
            if "diabetes" in self.feature_names and self._ta_faltando("diabetes"):
                self.features["diabetes"] = 0.0
                self._marcar_preenchida("diabetes")

    # -------------------------------------------------------------
    # Respostas curtas (sim/não) aplicadas à última feature/tópico
    # -------------------------------------------------------------

    def _aplicar_resposta_curta_para_ultima_feature(self, texto_usuario: str) -> None:
        if not self.ultima_feature_perguntada:
            return

        t = (texto_usuario or "").strip().lower()
        if t == "":
            return

        sim = {"sim", "s", "yes", "y", "claro", "uhum", "aham"}
        nao = {"não", "nao", "n", "no", "negativo", "nunca"}

        feat = self.ultima_feature_perguntada

        # Tópico interno: "já fumou anteriormente?"
        if feat == "smokingHistory":
            if t in sim:
                self.ever_smoked = 1
                return
            if t in nao:
                self.ever_smoked = 0
                # Se nunca fumou, cigsPerDay deve virar 0 e nunca mais ser perguntado
                if "cigsPerDay" in self.feature_names and self._ta_faltando("cigsPerDay"):
                    self.features["cigsPerDay"] = 0.0
                    self._marcar_preenchida("cigsPerDay")
                return

        # Binárias do modelo
        if feat in {"currentSmoker", "diabetes", "prevalentHyp", "male", "BPMeds", "prevalentStroke"}:
            if t in sim:
                if feat in self.feature_names:
                    self.features[feat] = 1.0
                    self._marcar_preenchida(feat)
                if feat == "currentSmoker":
                    self.ever_smoked = 1
                return
            if t in nao:
                if feat in self.feature_names:
                    self.features[feat] = 0.0
                    self._marcar_preenchida(feat)
                if feat == "currentSmoker":
                    # não fumante atual; histórico ainda pode ser desconhecido (regra manda perguntar depois)
                    pass
                return

        # Numéricas simples
        if re.fullmatch(r"\d{1,3}", t):
            val = float(t)
            if feat == "cigsPerDay" and "cigsPerDay" in self.feature_names:
                self.features["cigsPerDay"] = val
                self._marcar_preenchida("cigsPerDay")
                self.ever_smoked = 1 if val > 0 else (self.ever_smoked or 0)
                # Se respondeu 0 cigarros/dia, isso também implica não fumante atual
                if val == 0 and "currentSmoker" in self.feature_names and self._ta_faltando("currentSmoker"):
                    self.features["currentSmoker"] = 0.0
                    self._marcar_preenchida("currentSmoker")
                return

    # -----------------------------------------
    # Perguntas: detectar/filtrar/contabilizar
    # -----------------------------------------

    def _detectar_features_perguntadas_no_texto(self, resposta_assistente: str) -> List[str]:
        texto = resposta_assistente or ""
        achadas: List[str] = []

        for pattern, feat in self._keywords_para_feature:
            if pattern.search(texto):
                if feat == "sysBP":
                    if "sysBP" in self.feature_names:
                        achadas.append("sysBP")
                    if "diaBP" in self.feature_names:
                        achadas.append("diaBP")
                else:
                    if feat in self.feature_names:
                        achadas.append(feat)

        out: List[str] = []
        for f in achadas:
            if f not in out:
                out.append(f)
        return out

    def _filtrar_features_ainda_faltantes(self, feats: List[str]) -> List[str]:
        out: List[str] = []
        for f in feats:
            if f in self.feature_names and self._ta_faltando(f) and f not in out:
                out.append(f)
        return out

    def _registrar_perguntas_features(self, feats: List[str]) -> None:
        # Registra tentativas para features do modelo    
        for f in feats:
            if f in self.feature_names and self._ta_faltando(f) and self.perguntas_por_feature.get(f, 0) < 2:
                self.perguntas_por_feature[f] = self.perguntas_por_feature.get(f, 0) + 1

    # ----------------------------
    # Atualizações do LLM
    # ----------------------------

    def _aplicar_atualizacoes_features(self, atualizacoes: Dict[str, float]) -> None:
        for nome, valor in (atualizacoes or {}).items():
            if nome not in self.feature_names:
                continue
            try:
                valor_numerico = float(valor)
            except (TypeError, ValueError):
                continue
            self.features[nome] = valor_numerico
            self._marcar_preenchida(nome)

            # Se o LLM atualiza currentSmoker, ajusta ever_smoked coerentemente
            if nome == "currentSmoker":
                if valor_numerico >= 1:
                    self.ever_smoked = 1
                elif self.ever_smoked is None:
                    # não fumante atual, mas histórico ainda pode ser desconhecido
                    pass

            # Se o LLM atualiza cigsPerDay, isso implica ever_smoked=1 (se >0)
            if nome == "cigsPerDay":
                if valor_numerico > 0:
                    self.ever_smoked = 1
                elif self.ever_smoked is None:
                    self.ever_smoked = 0

    # ----------------------------------------
    # Seleção do "alvo" (1 pergunta por turno)
    # ----------------------------------------

    def _selecionar_proximo_alvo_prioritario(self) -> Optional[str]:
        # Decide qual feature perguntar em seguida (apenas 1), respeitando:
        # - não repetir além de 2x
        # - regra de tabagismo

        # Regra de tabagismo: se currentSmoker=0 e não sabemos se já fumou antes, perguntar isso primeiro
        if (
            "currentSmoker" in self.feature_names
            and "currentSmoker" in self.preenchidas
            and self.features.get("currentSmoker", 0.0) == 0.0
            and self.ever_smoked is None
            and self.perguntas_por_feature.get("smokingHistory", 0) < 2
        ):
            return "smokingHistory"

        for f in self.features_prioritarias:
            # Pula se já estourou limite
            if self.perguntas_por_feature.get(f, 0) >= 2:
                continue

            # Regra cigsPerDay: só perguntar se fuma ou já fumou
            if f == "cigsPerDay":
                if self.ever_smoked == 0:
                    # nunca fumou -> fixa como 0 e não pergunta
                    if "cigsPerDay" in self.feature_names and self._ta_faltando("cigsPerDay"):
                        self.features["cigsPerDay"] = 0.0
                        self._marcar_preenchida("cigsPerDay")
                    continue
                if self.ever_smoked is None:
                    # histórico ainda indefinido -> não perguntar quantidade ainda
                    continue

            if f in self.feature_names and self._ta_faltando(f):
                return f

        return None

    # ----------------------------
    # Loop principal
    # ----------------------------

    def processar_mensagem_usuario(self, mensagem: str) -> str:
        self.qtd_mensagens += 1
        self.fase_direcionamento = self.qtd_mensagens

        self.historico.append({"role": "user", "content": mensagem})

        # Se o usuário respondeu algo curto (sim/não/0) para a pergunta anterior, aplica na feature correta
        self._aplicar_resposta_curta_para_ultima_feature(mensagem)

        # Heurísticas antes do LLM (evita loops em "alto/normal/baixo", binárias etc.)
        self._aplicar_heuristicas_do_usuario(mensagem)

        prompt = self._montar_prompt_para_llm(mensagem)

        instrucao_coletor = (
            "Você é um assistente médico educacional (PT-BR), acolhedor, humano e não técnico.\n\n"
            "Objetivo oculto: coletar variáveis do Framingham, mas SEM parecer questionário.\n"
            "NÃO pergunte sobre histórico familiar, estresse/relaxamento, ansiedade, dieta detalhada ou exercícios "
            "como requisito de coleta, pois isso NÃO preenche o vetor do modelo.\n\n"
            "REGRA DE TABAGISMO (muito importante):\n"
            "- Só pergunte 'quantos cigarros por dia' (cigsPerDay) se a pessoa fuma OU já fumou anteriormente.\n"
            "- Se a pessoa disser 'nunca fumou', NÃO pergunte quantidade.\n"
            "- Se a pessoa disser 'não fuma atualmente', pergunte primeiro se ela já fumou antes.\n\n"
            "REGRAS DE RITMO:\n"
            f"- Interação do paciente: {self.fase_direcionamento}\n"
            "- Se for interação 1, 2 ou 3:\n"
            "  * NÃO faça perguntas diretas sobre variáveis clínicas.\n"
            "  * Seja empático e agradável e convide o paciente a falar livremente sobre saúde/rotina.\n"
            "  * Se o paciente mencionar dados espontaneamente, você pode atualizar.\n"
            "  * perguntas_features deve ser [].\n"
            "- A partir da interação 4:\n"
            "  * Faça APENAS UMA pergunta por resposta (uma única feature).\n"
            "  * A pergunta não pode ser direta: inclua 1–2 frases curtas explicando por que isso importa para o coração,\n"
            "    e só então pergunte.\n"
            "  * Nunca pergunte novamente uma variável já preenchida.\n\n"
            "FORMATO:\n"
            "- Responda sempre em JSON válido no formato solicitado.\n"
        )

        try:
            dados = perguntar_llm_json(mensagem_usuario=prompt, instrucao_sistema=instrucao_coletor)
        except RuntimeError as e:
            print("[ERRO LLM/JSON]", e)

            if (not self.finalizada) and (self._prioritarias_preenchidas() or self._count_preenchidas() >= self.min_features_para_inferir):
                return self._rodar_inferencia_e_finalizar()

            if not self.finalizada and self._limites_estourados():
                self.finalizada = True
                msg = self._gerar_mensagem_insuficiente()
                self.historico.append({"role": "assistant", "content": msg})
                return msg

            resposta = (
                "Desculpa — acho que não entendi direitinho sua última mensagem. "
                "Você pode me contar de novo, com suas palavras, como você tem se sentido e o que te preocupa mais?"
            )
            self.historico.append({"role": "assistant", "content": resposta})
            return resposta

        resposta_paciente = (dados.get("resposta_ao_paciente", "") or "").strip()
        atualizacoes = dados.get("atualizacoes", {}) or {}
        perguntas_features = dados.get("perguntas_features", []) or []

        # Aplica updates do LLM
        self._aplicar_atualizacoes_features(atualizacoes)

        # Fallback para detectar perguntas pelo texto
        perguntadas_no_texto = self._detectar_features_perguntadas_no_texto(resposta_paciente)

        # Filtra só as ainda faltantes
        perguntas_features = self._filtrar_features_ainda_faltantes(perguntas_features)
        perguntadas_no_texto = self._filtrar_features_ainda_faltantes(perguntadas_no_texto)

        # Unifica sem duplicar
        todas: List[str] = []
        for f in (perguntas_features + perguntadas_no_texto):
            if f not in todas:
                todas.append(f)

        # ---------------------------
        # Regras das 3 primeiras interações
        # ---------------------------
        if self.fase_direcionamento <= 3:
            if "?" in resposta_paciente or len(todas) > 0:
                repair_prompt = f"""
Reescreva a resposta abaixo para cumprir as regras:

REGRAS:
- Estamos nas 3 primeiras interações: NÃO faça nenhuma pergunta (zero '?').
- Seja empático, agradável e convide o paciente a falar livremente sobre saúde/rotina.
- Não peça idade, colesterol, pressão, diabetes etc.
- Não mencione histórico familiar nem estresse/relaxamento como requisito.
- Retorne APENAS o texto final (sem JSON).

RESPOSTA ORIGINAL:
\"\"\"{resposta_paciente}\"\"\"
"""
                resposta_paciente = perguntar_llm(
                    mensagem_usuario=repair_prompt,
                    instrucao_sistema="Você reescreve mensagens para cumprir restrições de diálogo com tom empático.",
                ).strip()

            perguntas_features = []
            self.ultima_feature_perguntada = None

        # ---------------------------
        # A partir da 4ª: garantir 1 pergunta (1 feature/tópico)
        # ---------------------------
        if self.fase_direcionamento >= 4:
            # Nosso alvo final é decidido pelo orquestrador (para respeitar regra do tabagismo)
            alvo = self._selecionar_proximo_alvo_prioritario()

            # Se o LLM perguntou algo diferente, fazemos repair para perguntar apenas o alvo
            muitos_interrog = (resposta_paciente.count("?") >= 2)

            # Detecta se a mensagem atual do assistente não está alinhada com o alvo
            desalinhado = False
            if alvo == "smokingHistory":
                # Esperamos uma pergunta do tipo "já fumou antes?"
                # Se ele perguntou "cigsPerDay" ou algo com "cigarros por dia", desalinhou
                if re.search(r"\bquantos\b.*\bcigar", resposta_paciente.lower()):
                    desalinhado = True
            elif alvo == "cigsPerDay" and self.ever_smoked in (0, None):
                # Não deveria perguntar quantidade ainda
                desalinhado = True
            elif alvo is not None and len(todas) > 0:
                # Se ele perguntou outra feature, desalinhado
                if alvo not in todas:
                    desalinhado = True

            if muitos_interrog or desalinhado or (len(todas) > 1):
                # Monta repair com alvo escolhido
                if alvo is None:
                    # Se não temos alvo, removemos perguntas (melhor do que perguntar errado)
                    repair_prompt = f"""
Reescreva a resposta abaixo para ficar empática e sem perguntas (zero '?').

REGRAS:
- Mantenha tom acolhedor.
- NÃO faça perguntas.
- NÃO fale de histórico familiar nem estresse/relaxamento como requisito.
- Retorne APENAS o texto final.

RESPOSTA ORIGINAL:
\"\"\"{resposta_paciente}\"\"\"
"""
                    resposta_paciente = perguntar_llm(
                        mensagem_usuario=repair_prompt,
                        instrucao_sistema="Você reescreve mensagens para cumprir restrições de diálogo.",
                    ).strip()
                    perguntas_features = []
                    self.ultima_feature_perguntada = None
                else:
                    if alvo == "smokingHistory":
                        pergunta_alvo = "smokingHistory"
                        descricao_alvo = (
                            "verificar se a pessoa já fumou anteriormente (ex-fumante), pois isso muda a interpretação de risco"
                        )
                    else:
                        pergunta_alvo = alvo
                        descricao_alvo = f"coletar a variável clínica '{alvo}' para estimativa de risco cardiovascular"

                    repair_prompt = f"""
Reescreva a resposta abaixo para cumprir as regras:

REGRAS:
- Mantenha o tom empático e natural.
- Faça APENAS UMA pergunta (um único '?').
- A pergunta deve ser SOMENTE sobre: {pergunta_alvo}.
- A pergunta não pode ser direta: inclua 1–2 frases curtas explicando por que essa informação importa para o coração,
  e só então pergunte (estilo conversacional, "enrolado").
- NÃO pergunte sobre histórico familiar, estresse/relaxamento.
- NÃO pergunte variáveis já preenchidas.
- Não invente valores.
- Retorne APENAS o texto final (sem JSON).

CONTEXTO DA COLETA:
- Precisamos {descricao_alvo}.

RESPOSTA ORIGINAL:
\"\"\"{resposta_paciente}\"\"\"
"""
                    resposta_paciente = perguntar_llm(
                        mensagem_usuario=repair_prompt,
                        instrucao_sistema="Você reescreve mensagens para ficar natural e cumprir restrições de diálogo.",
                    ).strip()

                    # Registra qual pergunta foi feita
                    if alvo == "smokingHistory":
                        perguntas_features = []  # tópico interno; não entra no vetor
                        self.ultima_feature_perguntada = "smokingHistory"
                        if self.perguntas_por_feature.get("smokingHistory", 0) < 2:
                            self.perguntas_por_feature["smokingHistory"] += 1
                    else:
                        perguntas_features = [alvo]
                        self.ultima_feature_perguntada = alvo
                        self._registrar_perguntas_features(perguntas_features)
            else:
                # Se está ok, registra a única pergunta detectada (ou nenhuma)
                if alvo == "smokingHistory":
                    self.ultima_feature_perguntada = "smokingHistory"
                    if self.perguntas_por_feature.get("smokingHistory", 0) < 2:
                        self.perguntas_por_feature["smokingHistory"] += 1
                    perguntas_features = []
                elif alvo is not None:
                    perguntas_features = [alvo]
                    self.ultima_feature_perguntada = alvo
                    self._registrar_perguntas_features(perguntas_features)
                else:
                    perguntas_features = []
                    self.ultima_feature_perguntada = None

        # salva resposta no histórico
        self.historico.append({"role": "assistant", "content": resposta_paciente})

        # ---------------------------
        # Critério de parada
        # ---------------------------
        if not self.finalizada and (self._prioritarias_preenchidas() or self._count_preenchidas() >= self.min_features_para_inferir):
            return self._rodar_inferencia_e_finalizar()

        if not self.finalizada and self._limites_estourados():
            self.finalizada = True
            msg = self._gerar_mensagem_insuficiente()
            self.historico.append({"role": "assistant", "content": msg})
            return msg

        return resposta_paciente

    # ----------------------------
    # Prompt do LLM
    # ----------------------------

    def _montar_prompt_para_llm(self, mensagem_atual: str) -> str:
        preenchidas_list = [f"{f}={self.features[f]}" for f in sorted(self.preenchidas)]
        texto_preenchidas = ", ".join(preenchidas_list) if preenchidas_list else "nenhuma ainda"

        historico_texto = ""
        for msg in self.historico:
            if msg["role"] == "user":
                historico_texto += f"Paciente: {msg['content']}\n"
            else:
                historico_texto += f"Assistente: {msg['content']}\n"

        lista_features_str = ", ".join(self.feature_names)
        pri_str = ", ".join(self.features_prioritarias) if self.features_prioritarias else "(nenhuma detectada)"
        pri_faltantes = [f for f in self.features_prioritarias if self._ta_faltando(f)]
        pri_faltantes_str = ", ".join(pri_faltantes) if pri_faltantes else "(nenhuma)"

        vars_limite = [f for f, c in self.perguntas_por_feature.items() if f in self.feature_names and c >= 2 and self._ta_faltando(f)]
        vars_limite_str = ", ".join(vars_limite) if vars_limite else "nenhuma"

        tabagismo_status = (
            f"currentSmoker={'preenchido' if ('currentSmoker' in self.preenchidas) else 'faltando'}, "
            f"everSmoked={'desconhecido' if self.ever_smoked is None else self.ever_smoked}, "
            f"cigsPerDay={'preenchido' if ('cigsPerDay' in self.preenchidas) else 'faltando'}"
        )

        if self.fase_direcionamento <= 3:
            regra_fase = (
                "FASE: acolhimento (interações 1 a 3).\n"
                "- NÃO faça perguntas diretas sobre variáveis clínicas.\n"
                "- Apenas acolha e convide o paciente a falar livremente.\n"
                "- Se ele mencionar dados espontaneamente, preencha em 'atualizacoes'.\n"
                "- 'perguntas_features' deve ser [].\n"
            )
        else:
            regra_fase = (
                "FASE: direcionamento suave (a partir da 4ª interação).\n"
                "- Faça APENAS 1 pergunta por resposta (uma única feature).\n"
                "- A pergunta deve vir 'enrolada' com 1–2 frases curtas explicando por que isso importa para o coração.\n"
                "- Use exatamente UMA interrogação '?' no texto.\n"
                "- Pergunte primeiro uma feature PRIORITÁRIA ainda faltante.\n"
            )

        prompt = f"""
Você está ajudando a preencher um vetor de variáveis clínicas do estudo de Framingham.

{regra_fase}

REGRAS IMPORTANTES:
- NÃO pergunte sobre histórico familiar nem estresse/relaxamento.
- NÃO pergunte variáveis já preenchidas.
- Não faça interrogatório; mantenha tom natural.
- Limite por variável: no máximo 2 tentativas.

REGRA DE TABAGISMO:
- Só pergunte cigsPerDay (quantos cigarros/dia) se a pessoa fuma OU já fumou anteriormente.
- Se a pessoa disser "nunca fumou", NÃO pergunte cigsPerDay.
- Se a pessoa disser "não fuma atualmente", pergunte primeiro se ela já fumou antes.
- Tabagismo atual no sistema: {tabagismo_status}

FEATURES PRIORITÁRIAS:
{pri_str}

PRIORITÁRIAS AINDA FALTANTES:
{pri_faltantes_str}

Nomes exatos de todas as features disponíveis:
{lista_features_str}

Estado atual:
- Preenchidas: {texto_preenchidas}
- Total preenchidas: {self._count_preenchidas()} (meta mínima: {self.min_features_para_inferir})
- Limite atingido (evite insistir): {vars_limite_str}

Histórico:
{historico_texto}

Nova mensagem do paciente:
\"\"\"{mensagem_atual}\"\"\"

Conversões quando o paciente não souber números:
- Pressão: alta -> sysBP=150,diaBP=95 | normal -> 120/80 | baixa -> 100/60
- Colesterol: alto -> 260 | normal -> 200 | baixo -> 160
- Glicose: alta -> 150 | normal -> 100 | baixa -> 70

FORMATO (APENAS JSON):
{{
  "resposta_ao_paciente": "texto",
  "atualizacoes": {{"feature": valor}},
  "terminou_coleta": false,
  "perguntas_features": ["feature_que_foi_perguntada"]
}}

Observação:
- Se estiver nas 3 primeiras interações, "perguntas_features" deve ser [].
- A partir da 4ª, "perguntas_features" deve ter no máximo 1 item.
"""
        return prompt

    # ----------------------------
    # Inferência / mensagens finais
    # ----------------------------

    def _rodar_inferencia_e_finalizar(self) -> str:
        prob_risco = predicao_prob_risco(self.features)
        categoria = self._classificar_risco(prob_risco)

        self.ultimo_risco_prob = prob_risco
        self.ultima_categoria_risco = categoria

        explicacao = self._gerar_explicacao_risco(prob_risco, categoria)
        self.ultima_explicacao = explicacao

        self.finalizada = True

        resposta_final = (
            "Categoria de risco estimada (modelo Framingham): "
            f"*{categoria.upper()}*\n\n"
            f"{explicacao}"
        )
        self.historico.append({"role": "assistant", "content": resposta_final})
        return resposta_final

    def _classificar_risco(self, prob_risco: float) -> str:
        if prob_risco < 0.10:
            return "baixo"
        elif prob_risco < 0.20:
            return "moderado"
        else:
            return "alto"

    def _gerar_explicacao_risco(self, prob_risco: float, categoria: str) -> str:
        if categoria == "alto":
            resumo = "O modelo sugere um risco alto em cerca de 10 anos."
        elif categoria == "moderado":
            resumo = "O modelo sugere um risco moderado em cerca de 10 anos."
        else:
            resumo = "O modelo sugere um risco baixo (não nulo) em cerca de 10 anos."

        prompt = (
            "Explique o resultado para um paciente leigo, com cuidado.\n"
            f"Categoria: {categoria}\n"
            f"Probabilidade aproximada: {prob_risco:.2%}\n"
            f"Resumo: {resumo}\n"
            "Não faça diagnóstico. Sugira acompanhamento médico e hábitos saudáveis.\n"
        )

        return perguntar_llm(
            mensagem_usuario=prompt,
            instrucao_sistema="Você é um assistente médico educacional. Seja claro, empático e responsável.",
        )

    def _gerar_mensagem_insuficiente(self) -> str:
        prompt = (
            "Escreva uma mensagem curta, acolhedora e responsável informando que não foi possível estimar o risco "
            "por falta de informações suficientes. Não faça predição."
        )
        return perguntar_llm(
            mensagem_usuario=prompt,
            instrucao_sistema="Você é um assistente médico educacional. Seja empático e responsável.",
        )


# ----------------------------
# Teste no terminal
# ----------------------------

def fluxo_teste_terminal() -> None:
    print("=== Teste de sessão em TEXTO LIVRE (Risco Cardiovascular / Framingham) ===\n")
    sessao = SessaoDiagnosticoTextoLivre()

    mensagem_inicial = (
        "Oi! Eu sou o CardioBot! :)\n"
        "Eu posso te ajudar a ter uma noção bem geral do seu risco cardiovascular nos próximos anos!\n"
        "Lembrando que eu sou apenas um assistente e que não posso substituir uma consulta médica real.\n"
        "Pra começar, sem pressa: como você tem se sentido ultimamente? O que te trouxe aqui hoje?"
    )
    print(mensagem_inicial)

    while not sessao.finalizada:
        msg = input("Você (paciente): ")
        resp = sessao.processar_mensagem_usuario(msg)
        print("\nAssistente:\n", resp, "\n")

    print("=== Fim da sessão ===")


if __name__ == "__main__":
    fluxo_teste_terminal()
