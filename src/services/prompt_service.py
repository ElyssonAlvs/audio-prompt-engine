"""Prompt Engineering Engine using Strategy Pattern for the 6 core techniques."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.domain.enums import PromptTechnique
from src.domain.schemas import PromptTechniqueInfo


class IPromptStrategy(ABC):
    """Abstract Strategy interface for prompt generation."""

    @abstractmethod
    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        """Synthesize the prompt from processed speech."""
        pass

    @abstractmethod
    def get_info(self) -> PromptTechniqueInfo:
        """Return educational metadata about this technique."""
        pass


class ZeroShotStrategy(IPromptStrategy):
    """Direct, instruction-driven zero-shot prompt without prior examples."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        context = custom_context or "Você é um assistente de inteligência artificial de alta precisão técnica."
        return (
            f"# DIRETRIZ DE EXECUÇÃO (ZERO-SHOT)\n\n"
            f"## [Contexto Operacional]\n"
            f"{context}\n\n"
            f"## [Instrução Principal]\n"
            f"Analise a solicitação extraída diretamente da fala do usuário e execute-a com máximo rigor técnico, clareza e objetividade.\n\n"
            f"## [Fala do Usuário Transcrita]\n"
            f"<user_speech>\n"
            f"{processed_text}\n"
            f"</user_speech>\n\n"
            f"## [Formato da Resposta]\n"
            f"- Forneça a solução pronta para uso imediato, sem introduções protocolares.\n"
            f"- Adicione blocos de código ou passos práticos devidamente comentados quando aplicável."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.ZERO_SHOT,
            name="Zero-Shot Prompting",
            description="Estrutura a solicitação falada em uma instrução direta e imperativa com seções de Contexto, Tarefa e Formato sem exemplos prévios.",
            when_to_use="Demandas diretas, consultas pontuais, refatorações simples e respostas objetivas.",
            example_structure="[Contexto] -> [Instrução] -> <user_speech> -> [Formato de Saída]",
        )


class FewShotStrategy(IPromptStrategy):
    """Few-shot prompt injecting high-quality input/output pairs to steer LLM style."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        context = custom_context or "Você é um especialista em transformar demandas verbais em código e soluções estruturadas."
        return (
            f"# DIRETRIZ DE EXECUÇÃO (FEW-SHOT)\n\n"
            f"## [Contexto Operacional]\n"
            f"{context}\n\n"
            f"## [Exemplos de Calibração (Padrão Entrada -> Saída)]\n"
            f"### Exemplo 1:\n"
            f"- **Entrada:** 'Preciso de um script Python que leia um CSV e remova duplicados por email.'\n"
            f"- **Saída:**\n"
            f"```python\n"
            f"import pandas as pd\n\n"
            f"def deduplicate_emails(csv_path: str, output_path: str) -> None:\n"
            f"    df = pd.read_csv(csv_path)\n"
            f"    cleaned_df = df.drop_duplicates(subset=['email'], keep='first')\n"
            f"    cleaned_df.to_csv(output_path, index=False)\n"
            f"```\n\n"
            f"### Exemplo 2:\n"
            f"- **Entrada:** 'Crie uma query SQL para encontrar clientes sem compras nos últimos 90 dias.'\n"
            f"- **Saída:**\n"
            f"```sql\n"
            f"SELECT c.id, c.name, c.email\n"
            f"FROM customers c\n"
            f"LEFT JOIN orders o ON c.id = o.customer_id AND o.created_at >= CURRENT_DATE - INTERVAL '90 days'\n"
            f"WHERE o.id IS NULL;\n"
            f"```\n\n"
            f"## [Fala do Usuário Transcrita]\n"
            f"<user_speech>\n"
            f"{processed_text}\n"
            f"</user_speech>\n\n"
            f"## [Tarefa]\n"
            f"Reproduza o mesmo padrão de concisão, tipagem estrita e rigor dos exemplos acima para atender à demanda de <user_speech>."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.FEW_SHOT,
            name="Few-Shot Prompting",
            description="Injeta pares padronizados de Entrada -> Saída Esperada para calibrar o tom, formato de dados (JSON/SQL/Código) e densidade técnica da resposta.",
            when_to_use="Tarefas onde o formato de saída deve seguir um contrato exato, extração de dados e conversões sintáticas.",
            example_structure="[Contexto] -> [Exemplos 1, 2...] -> <user_speech> -> [Instrução de Calibração]",
        )


class ChainOfThoughtStrategy(IPromptStrategy):
    """Mandates step-by-step reasoning before providing the final deliverable."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        context = custom_context or "Você é um arquiteto analítico especializado em resolução de problemas lógicos e arquiteturais de alta complexidade."
        return (
            f"# DIRETRIZ DE RACIOCÍNIO EXPLÍCITO (CHAIN-OF-THOUGHT)\n\n"
            f"## [Contexto Operacional]\n"
            f"{context}\n\n"
            f"## [Fala do Usuário Transcrita]\n"
            f"<user_speech>\n"
            f"{processed_text}\n"
            f"</user_speech>\n\n"
            f"## [Instruções Obrigatórias de Raciocínio (Passo a Passo)]\n"
            f"Você DEVE estruturar sua resposta nas 4 etapas analíticas a seguir antes de qualquer conclusão:\n\n"
            f"1. **Decomposição do Problema:** Isole os requisitos funcionais, restrições não funcionais e potenciais ambiguidades da fala do usuário.\n"
            f"2. **Identificação de Riscos & Edge Cases:** Analise pontos de falha, concorrência, limites de escala e brechas de segurança.\n"
            f"3. **Raciocínio Dedutivo:** Compare as opções arquiteturais possíveis e justifique a escolha técnica ótima.\n"
            f"4. **Síntese & Entregável Final:** Apresente a solução definitiva, documentada e pronta para implementação."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            name="Chain of Thought (CoT)",
            description="Força o modelo a explicitar sua cadeia de raciocínio lógico em etapas sequenciais antes de emitir a resposta definitiva.",
            when_to_use="Resolução de bugs complexos, decisões de arquitetura, matemática, modelagem de dados e análise de segurança.",
            example_structure="[Contexto] -> <user_speech> -> [1. Decomposição -> 2. Riscos -> 3. Dedução -> 4. Síntese]",
        )


class RolePromptingStrategy(IPromptStrategy):
    """Assigns an authoritative expert persona with rigorous domain constraints."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        context = custom_context or "Sua conduta profissional deve priorizar manutenibilidade, tolerância a falhas, segurança defensiva e aderência a padrões industriais."
        return (
            f"# DIRETRIZ DE PERSONA ESPECIALISTA (ROLE PROMPTING)\n\n"
            f"## [Atribuição de Papel & Senioridade]\n"
            f"Você é um **Staff Software Engineer & Arquiteto de Sistemas Distribuídos** com mais de 20 anos de experiência em sistemas de missão crítica.\n\n"
            f"## [Contexto Operacional e Postura]\n"
            f"{context}\n\n"
            f"## [Fala do Usuário Transcrita]\n"
            f"<user_speech>\n"
            f"{processed_text}\n"
            f"</user_speech>\n\n"
            f"## [Diretrizes de Resposta]\n"
            f"- Adote uma postura técnica sênior, propositiva e incisiva.\n"
            f"- Se a solicitação contiver premissas equivocadas ou práticas frágeis, corrija-as formalmente antes de prosseguir.\n"
            f"- Entregue arquiteturas resilientes com tipagem estrita, logs e considerações de observabilidade."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.ROLE_PROMPTING,
            name="Role Prompting",
            description="Atribui ao LLM uma persona de altíssimo nível técnico com postura crítica e restrições rigorosas de domínio.",
            when_to_use="Auditorias de código, revisões arquiteturais, consultorias estratégicas e redação de padrões técnicos.",
            example_structure="[Atribuição de Papel Sênior] -> [Postura & Restrições] -> <user_speech> -> [Diretrizes]",
        )


class CoStarStrategy(IPromptStrategy):
    """Corporate CO-STAR framework: Context, Objective, Style, Tone, Audience, Response."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        ctx = custom_context or "Demanda técnica em ambiente de engenharia de software corporativo."
        aud = target_audience or "Engenheiros de Software, Tech Leads e Arquitetos de Sistemas."
        return (
            f"# FRAMEWORK ESTRATÉGICO DE PROMPT (CO-STAR)\n\n"
            f"- **CONTEXT (Contexto):**\n"
            f"  {ctx}\n\n"
            f"- **OBJECTIVE (Objetivo Central):**\n"
            f"  Executar rigorosamente a demanda expressa na solicitação transcrita:\n"
            f"  <user_speech>\n"
            f"  {processed_text}\n"
            f"  </user_speech>\n\n"
            f"- **STYLE (Estilo):**\n"
            f"  Direto, altamente técnico, estruturado em tópicos e sem divagações.\n\n"
            f"- **TONE (Tom):**\n"
            f"  Assertivo, consultivo, pragmático e focado em excelência de engenharia.\n\n"
            f"- **AUDIENCE (Público-Alvo):**\n"
            f"  {aud}\n\n"
            f"- **RESPONSE FORMAT (Formato de Saída):**\n"
            f"  Documento técnico formatado em Markdown contendo: 1. Resumo da Abordagem, 2. Implementação Completa, 3. Matriz de Riscos & Validações."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.CO_STAR,
            name="CO-STAR Framework",
            description="Estrutura a demanda corporativa nos 6 pilares estratégicos: Context, Objective, Style, Tone, Audience e Response format.",
            when_to_use="Prompts de produção corporativa de alta precisão, especificações técnicas formais e relatórios executivos.",
            example_structure="[Context -> Objective -> Style -> Tone -> Audience -> Response Format]",
        )


class ReActStrategy(IPromptStrategy):
    """ReAct (Reasoning + Acting) autonomous agent prompt pattern."""

    def generate(
        self,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        context = custom_context or "Você é um agente autônomo operando sob o ciclo ReAct para planejar e executar ações passo a passo."
        return (
            f"# DIRETRIZ DE AGENTE AUTÔNOMO (ReAct: Reason + Act)\n\n"
            f"## [Contexto Operacional]\n"
            f"{context}\n\n"
            f"## [Objetivo da Missão]\n"
            f"Atender à demanda transcrita pelo usuário decompondo-a em ciclos de raciocínio e ação:\n"
            f"<user_speech>\n"
            f"{processed_text}\n"
            f"</user_speech>\n\n"
            f"## [Protocolo de Execução ReAct]\n"
            f"Para cada etapa necessária, emita estritamente o seguinte bloco:\n"
            f"- **Thought:** Análise cognitiva sobre o estado atual e a necessidade imediata.\n"
            f"- **Action:** Ação concreta a ser realizada (ex: consulta conceitual, cálculo, planejamento de código).\n"
            f"- **Observation:** Resultado esperado ou validado dessa ação.\n"
            f"*(Repita o ciclo até consolidar todos os requisitos)*\n\n"
            f"- **Final Answer:** A resposta definitiva e completa entregue ao usuário."
        )

    def get_info(self) -> PromptTechniqueInfo:
        return PromptTechniqueInfo(
            technique=PromptTechnique.REACT,
            name="ReAct (Reason + Act)",
            description="Framework de agentes autônomos que combina raciocínio dedutivo com ações explícitas (Thought -> Action -> Observation -> Final Answer).",
            when_to_use="Construção de fluxos para agentes autônomos, integrações com ferramentas externas e raciocínio multi-etapa.",
            example_structure="<user_speech> -> [Loop: Thought -> Action -> Observation] -> Final Answer",
        )


class PromptEngineeringService:
    """Orchestrates prompt generation across the 6 supported techniques."""

    def __init__(self):
        self._strategies: Dict[PromptTechnique, IPromptStrategy] = {
            PromptTechnique.ZERO_SHOT: ZeroShotStrategy(),
            PromptTechnique.FEW_SHOT: FewShotStrategy(),
            PromptTechnique.CHAIN_OF_THOUGHT: ChainOfThoughtStrategy(),
            PromptTechnique.ROLE_PROMPTING: RolePromptingStrategy(),
            PromptTechnique.CO_STAR: CoStarStrategy(),
            PromptTechnique.REACT: ReActStrategy(),
        }

    def generate_prompt(
        self,
        technique: PromptTechnique,
        processed_text: str,
        custom_context: Optional[str] = None,
        target_audience: Optional[str] = None,
    ) -> str:
        strategy = self._strategies.get(technique)
        if not strategy:
            strategy = self._strategies[PromptTechnique.ZERO_SHOT]
        return strategy.generate(
            processed_text=processed_text,
            custom_context=custom_context,
            target_audience=target_audience,
        )

    def get_catalog(self) -> List[PromptTechniqueInfo]:
        """Return the educational metadata catalog for all techniques."""
        return [strategy.get_info() for strategy in self._strategies.values()]


# Global instance
_prompt_service_instance: Optional[PromptEngineeringService] = None


def get_prompt_service() -> PromptEngineeringService:
    global _prompt_service_instance
    if _prompt_service_instance is None:
        _prompt_service_instance = PromptEngineeringService()
    return _prompt_service_instance
