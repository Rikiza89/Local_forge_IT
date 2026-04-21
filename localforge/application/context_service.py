"""
Context service — sole responsibility for assembling LLM prompts and managing token budgets.
All LLM prompt construction must go through this service.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from localforge.domain.exceptions import TokenBudgetExceededWarning
from localforge.domain.models import FileChunk, GenerationPlan, Message, ProjectIndex

logger = logging.getLogger(__name__)

_WORDS_TO_TOKENS = 1.3
_DEFAULT_TOKEN_LIMIT = 6000


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _WORDS_TO_TOKENS)


class ContextService:
    """
    Assembles LLM context (prompts) for all modes.
    Centralises all prompt-building logic and enforces token budgets.
    """

    def __init__(self, token_limit: int = _DEFAULT_TOKEN_LIMIT) -> None:
        self._token_limit = token_limit

    def update_token_limit(self, limit: int) -> None:
        self._token_limit = limit

    def _guard_budget(self, prompt: str, label: str = "") -> str:
        estimated = _estimate_tokens(prompt)
        if estimated > self._token_limit:
            logger.warning("Token budget exceeded [%s]: estimated=%d, limit=%d", label, estimated, self._token_limit)
        return prompt

    # ------------------------------------------------------------------
    # Generate mode
    # ------------------------------------------------------------------

    def build_plan_prompt(
        self,
        user_prompt: str,
        folder_name: str,
        file_tree_text: str,
        context_md: str,
        git_log: str,
    ) -> str:
        parts = [f"Nome progetto: {folder_name}"]
        if file_tree_text.strip():
            parts.append(f"Struttura file attuale:\n{file_tree_text}")
        if context_md.strip():
            parts.append(f"Contesto progetto:\n{context_md}")
        if git_log.strip():
            parts.append(f"Commit git recenti:\n{git_log}")

        parts.append(
            f"\nRichiesta dell'utente:\n{user_prompt}\n\n"
            "Genera un piano di struttura file per soddisfare la richiesta sopra, nel seguente formato JSON:\n"
            "```json\n"
            "{\n"
            '  "project_name": "nome del progetto",\n'
            '  "description": "descrizione generale del progetto",\n'
            '  "files": [\n'
            '    {\n'
            '      "path": "percorso/relativo/file",\n'
            '      "description": "ruolo e contenuto di questo file",\n'
            '      "dependencies": ["lista dei file da cui dipende"]\n'
            '    }\n'
            '  ]\n'
            "}\n"
            "```\n"
            "Restituisci solo il JSON, senza altro testo."
        )
        return self._guard_budget("\n\n".join(parts), "generate_plan")

    def build_file_generation_prompt(
        self,
        target_file: str,
        target_description: str,
        context_md: str,
        plan_json: str,
        dependency_contents: List[tuple[str, str]],
    ) -> str:
        parts = [
            f"File da generare: {target_file}",
            f"Ruolo e contenuto: {target_description}",
        ]
        if context_md.strip():
            parts.append(f"Contesto progetto:\n{context_md}")
        if plan_json.strip():
            parts.append(f"Piano completo del progetto:\n{plan_json}")

        dep_parts: List[str] = []
        for dep_path, dep_content in dependency_contents:
            dep_parts.append(f"--- {dep_path} ---\n{dep_content}")

        available = self._token_limit - _estimate_tokens("\n\n".join(parts))
        for dep_text in dep_parts:
            if _estimate_tokens(dep_text) < available:
                parts.append(f"File dipendente:\n{dep_text}")
                available -= _estimate_tokens(dep_text)
            else:
                logger.warning("Token budget insufficient, skipping dependency: %s", dep_text[:100])
                break

        parts.append(
            f"\nRestituisci solo il codice sorgente completo di {target_file}."
            " Non usare blocchi markdown (```). Restituisci solo il contenuto del file."
        )
        return self._guard_budget("\n\n".join(parts), f"generate_file:{target_file}")

    def build_context_update_prompt(
        self,
        previous_context: str,
        new_file_path: str,
        new_file_first_200_lines: str,
    ) -> str:
        parts = []
        if previous_context.strip():
            parts.append(f"Contesto attuale:\n{previous_context}")
        parts.append(
            f"Nuovo file generato: {new_file_path}\n"
            f"Contenuto (prime 200 righe):\n{new_file_first_200_lines}"
        )
        parts.append(
            "Aggiorna il memo di contesto del progetto in formato markdown tenendo conto del nuovo file."
            " Includi cambiamenti, aggiunte e dipendenze in modo sintetico."
            " Restituisci solo il testo aggiornato del contesto."
        )
        return self._guard_budget("\n\n".join(parts), "context_update")

    # ------------------------------------------------------------------
    # Explain mode
    # ------------------------------------------------------------------

    def build_batch_file_summary_prompt(
        self,
        file_chunks: List["FileChunk"],
        content_limit: int = 400,
    ) -> str:
        sections = []
        for chunk in file_chunks:
            excerpt = chunk.content[:content_limit]
            sections.append(f"FILE: {chunk.path}\n{excerpt}")

        prompt = (
            "Riassumi in una frase il ruolo di ciascun file.\n"
            "Formato output: FILE: <percorso>\\nSUMMARY: <riassunto>\n\n"
            + "\n\n".join(sections)
        )
        return self._guard_budget(prompt, "batch_file_summary")

    def build_file_summary_prompt(self, file_path: str, content: str, extension: str) -> str:
        prompt = (
            f"File: {file_path} (estensione: {extension})\n\n"
            f"{content}\n\n"
            "Riassumi in 3-5 frasi il ruolo di questo file, le classi/funzioni principali, le esportazioni e le dipendenze."
            " Restituisci solo il riassunto."
        )
        return self._guard_budget(prompt, f"file_summary:{file_path}")

    def build_project_index_prompt(
        self,
        file_summaries: List[tuple[str, str]],
        folder_tree: str,
        root_configs: str,
    ) -> str:
        summaries_text = "\n".join(f"- {path}: {summary}" for path, summary in file_summaries)
        if _estimate_tokens(summaries_text) > self._token_limit * 0.6:
            truncated = file_summaries[: int(len(file_summaries) * 0.6)]
            summaries_text = "\n".join(f"- {path}: {summary}" for path, summary in truncated)
            logger.warning("ProjectIndex prompt: summary list truncated")

        parts = [f"Struttura directory:\n{folder_tree}"]
        if root_configs.strip():
            parts.append(f"File di configurazione principali:\n{root_configs}")
        parts.append(f"Riassunti dei file:\n{summaries_text}")
        parts.append(
            "Scrivi una panoramica del progetto in 3-5 frasi che descriva lo scopo, i componenti principali e lo stack tecnologico."
            " Restituisci solo la panoramica."
        )
        return self._guard_budget("\n\n".join(parts), "project_index")

    def build_report_section_prompt(
        self,
        section_name: str,
        project_index_json: str,
        relevant_summaries: List[tuple[str, str]],
    ) -> str:
        summaries_text = "\n".join(f"- {path}: {summary}" for path, summary in relevant_summaries)
        prompt = (
            f"Panoramica progetto:\n{project_index_json}\n\n"
            f"Riassunti file rilevanti:\n{summaries_text}\n\n"
            f"Scrivi un'analisi dettagliata della seguente sezione in italiano:\n"
            f"Sezione: {section_name}\n\n"
            "Restituisci solo il contenuto di questa sezione in formato markdown."
        )
        return self._guard_budget(prompt, f"report_section:{section_name}")

    def build_qa_prompt(
        self,
        question: str,
        project_index_json: str,
        top_summaries: List[tuple[str, str]],
        full_contents: List[tuple[str, str]],
        conversation_history: List[Message],
    ) -> str:
        parts = [f"Panoramica progetto:\n{project_index_json}"]
        summaries_text = "\n".join(f"- {p}: {s}" for p, s in top_summaries)
        if summaries_text:
            parts.append(f"Riassunti file rilevanti:\n{summaries_text}")

        available = self._token_limit - _estimate_tokens("\n\n".join(parts))
        for fc_path, fc_content in full_contents:
            fc_text = f"--- {fc_path} ---\n{fc_content}"
            if _estimate_tokens(fc_text) < available:
                parts.append(f"Contenuto file:\n{fc_text}")
                available -= _estimate_tokens(fc_text)
            else:
                break

        if conversation_history:
            history_text = "\n".join(
                f"{'Utente' if m.role == 'user' else 'Assistente'}: {m.content}"
                for m in conversation_history[-10:]
            )
            parts.append(f"Cronologia conversazione:\n{history_text}")

        parts.append(f"Domanda: {question}\n\nRispondi in italiano basandoti sul codebase descritto sopra.")
        return self._guard_budget("\n\n".join(parts), "qa")

    # ------------------------------------------------------------------
    # Resume mode
    # ------------------------------------------------------------------

    def build_resume_continue_prompt(self, target_file, target_description, context_md, plan_json, completed_contents):
        return self.build_file_generation_prompt(
            target_file=target_file,
            target_description=target_description,
            context_md=context_md,
            plan_json=plan_json,
            dependency_contents=completed_contents,
        )

    def build_foreign_resume_qa_prompt(self, question, project_index_json, top_summaries, conversation_history):
        return self.build_qa_prompt(
            question=question,
            project_index_json=project_index_json,
            top_summaries=top_summaries,
            full_contents=[],
            conversation_history=conversation_history,
        )
