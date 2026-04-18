"""Document Intelligence Toolkit — Windows Desktop App.

Built with customtkinter, styled after Anthropic/Claude's design language.
Runs entirely offline using rule-based extraction, extractive summarisation,
and text-diff comparison — no API key or internet connection required.

Run with:
    python ui/desktop_app.py
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any

import customtkinter as ctk

from src.document_parsers.docx_parser import load_docx
from src.document_parsers.pdf_parser import load_pdf
from src.intelligence.comparator import compare_documents
from src.intelligence.extractor import extract_structured_data
from src.intelligence.summarizer import summarize_documents
from src.utils.llm_factory import has_api_key

# ---------------------------------------------------------------------------
# Anthropic / Claude brand palette
# ---------------------------------------------------------------------------

_ORANGE      = "#CC785C"   # Claude's warm coral-orange
_ORANGE_DARK = "#B5613F"   # hover state
_BG          = "#1A1A1A"   # near-black background
_CARD        = "#242424"   # raised card / panel
_CARD2       = "#2C2C2C"   # secondary card
_BORDER      = "#3A3A3A"   # subtle divider
_TEXT        = "#EBEBEB"   # primary text
_MUTED       = "#8A8A8A"   # secondary / muted text
_WHITE       = "#FFFFFF"
_GREEN       = "#4CAF82"   # success accent
_YELLOW      = "#E8A838"   # warning accent

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_file(path: str) -> list[Any]:
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return load_pdf(p)
    return load_docx(p)


def _pick_file(label_var: tk.StringVar) -> str | None:
    path = filedialog.askopenfilename(
        filetypes=[("Documents", "*.pdf *.docx"), ("PDF", "*.pdf"), ("Word", "*.docx")]
    )
    if path:
        label_var.set(Path(path).name)
    return path or None


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------


class _SectionHeader(ctk.CTkLabel):
    def __init__(self, parent: Any, text: str, **kwargs: Any) -> None:
        super().__init__(
            parent,
            text=text,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color=_MUTED,
            **kwargs,
        )


class _OrangeButton(ctk.CTkButton):
    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(
            parent,
            fg_color=_ORANGE,
            hover_color=_ORANGE_DARK,
            text_color=_WHITE,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            height=38,
            **kwargs,
        )


class _GhostButton(ctk.CTkButton):
    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(
            parent,
            fg_color=_CARD2,
            hover_color=_BORDER,
            text_color=_TEXT,
            border_color=_BORDER,
            border_width=1,
            corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            height=38,
            **kwargs,
        )


class _ResultBox(ctk.CTkTextbox):
    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(
            parent,
            fg_color=_CARD,
            text_color=_TEXT,
            border_color=_BORDER,
            border_width=1,
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
            **kwargs,
        )

    def set_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("1.0", text)
        self.configure(state="disabled")


# ---------------------------------------------------------------------------
# Analyse Tab
# ---------------------------------------------------------------------------


class AnalyseTab(ctk.CTkFrame):
    def __init__(self, parent: Any) -> None:
        super().__init__(parent, fg_color="transparent")
        self._file_path: str | None = None
        self._docs: list[Any] = []
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)

        # --- File picker row ---
        file_row = ctk.CTkFrame(self, fg_color=_CARD, corner_radius=10)
        file_row.grid(row=0, column=0, sticky="ew", padx=0, pady=(0, 12))
        file_row.grid_columnconfigure(1, weight=1)

        _GhostButton(file_row, text="📂  Open Document", width=160,
                     command=self._pick).grid(row=0, column=0, padx=12, pady=12)

        self._file_label = ctk.CTkLabel(
            file_row, text="No file selected",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=_MUTED, anchor="w",
        )
        self._file_label.grid(row=0, column=1, padx=8, pady=12, sticky="w")

        # --- Controls row ---
        ctrl = ctk.CTkFrame(self, fg_color=_CARD, corner_radius=10)
        ctrl.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 12))
        ctrl.grid_columnconfigure((0, 1, 2, 3), weight=1)

        _SectionHeader(ctrl, text="STYLE").grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        self._style_var = tk.StringVar(value="concise")
        ctk.CTkOptionMenu(
            ctrl, values=["concise", "detailed", "bullet"],
            variable=self._style_var,
            fg_color=_CARD2, button_color=_ORANGE, button_hover_color=_ORANGE_DARK,
            text_color=_TEXT, font=ctk.CTkFont(family="Segoe UI", size=13),
        ).grid(row=1, column=0, padx=12, pady=(0, 12), sticky="ew")

        _OrangeButton(ctrl, text="Summarise", command=self._summarise).grid(
            row=1, column=1, padx=6, pady=(0, 12), sticky="ew")
        _OrangeButton(ctrl, text="Extract Data", command=self._extract).grid(
            row=1, column=2, padx=6, pady=(0, 12), sticky="ew")
        _GhostButton(ctrl, text="Clear", command=self._clear).grid(
            row=1, column=3, padx=12, pady=(0, 12), sticky="ew")

        # --- Results ---
        results = ctk.CTkFrame(self, fg_color="transparent")
        results.grid(row=2, column=0, sticky="nsew", padx=0)
        results.grid_columnconfigure((0, 1), weight=1)
        results.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        _SectionHeader(results, text="SUMMARY").grid(row=0, column=0, sticky="w", pady=(0, 4))
        _SectionHeader(results, text="EXTRACTED DATA").grid(row=0, column=1, sticky="w", pady=(0, 4), padx=(12, 0))

        self._summary_box = _ResultBox(results)
        self._summary_box.grid(row=1, column=0, sticky="nsew", pady=(0, 0))
        self._summary_box.set_text("Summary will appear here…")

        self._extract_box = _ResultBox(results)
        self._extract_box.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self._extract_box.set_text("Extracted fields will appear here…")

        # --- Status bar ---
        self._status = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=_MUTED, anchor="w",
        )
        self._status.grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _pick(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf *.docx"), ("PDF", "*.pdf"), ("Word", "*.docx")]
        )
        if path:
            self._file_path = path
            self._file_label.configure(text=Path(path).name, text_color=_TEXT)
            self._set_status(f"Loaded: {Path(path).name}")

    def _set_status(self, msg: str, color: str = _MUTED) -> None:
        self._status.configure(text=msg, text_color=color)

    def _run_in_thread(self, fn: Any) -> None:
        threading.Thread(target=fn, daemon=True).start()

    def _summarise(self) -> None:
        if not self._file_path:
            self._set_status("Please open a document first.", _YELLOW)
            return
        self._summary_box.set_text("Summarising…")
        self._set_status("Running…", _ORANGE)

        def _work() -> None:
            try:
                docs = _load_file(self._file_path)  # type: ignore[arg-type]
                result = summarize_documents(docs, style=self._style_var.get())
                self._summary_box.set_text(result)
                self._set_status("Summary complete.", _GREEN)
            except Exception as exc:
                self._summary_box.set_text(f"Error: {exc}")
                self._set_status("Failed.", _ORANGE_DARK)

        self._run_in_thread(_work)

    def _extract(self) -> None:
        if not self._file_path:
            self._set_status("Please open a document first.", _YELLOW)
            return
        self._extract_box.set_text("Extracting…")
        self._set_status("Running…", _ORANGE)

        def _work() -> None:
            try:
                docs = _load_file(self._file_path)  # type: ignore[arg-type]
                result = extract_structured_data(docs)
                lines = [
                    f"Title:         {result.title}",
                    f"Type:          {result.document_type}",
                    f"Parties:       {', '.join(result.parties) or '—'}",
                    f"Dates:         {', '.join(result.dates) or '—'}",
                    "",
                    "Key Topics:",
                    *[f"  • {t}" for t in result.key_topics],
                    "",
                    "Key Clauses:",
                    *([f"  • {c}" for c in result.key_clauses] or ["  —"]),
                    "",
                    "Summary:",
                    result.summary,
                ]
                self._extract_box.set_text("\n".join(lines))
                self._set_status("Extraction complete.", _GREEN)
            except Exception as exc:
                self._extract_box.set_text(f"Error: {exc}")
                self._set_status("Failed.", _ORANGE_DARK)

        self._run_in_thread(_work)

    def _clear(self) -> None:
        self._summary_box.set_text("Summary will appear here…")
        self._extract_box.set_text("Extracted fields will appear here…")
        self._file_path = None
        self._file_label.configure(text="No file selected", text_color=_MUTED)
        self._set_status("")


# ---------------------------------------------------------------------------
# Compare Tab
# ---------------------------------------------------------------------------


class CompareTab(ctk.CTkFrame):
    def __init__(self, parent: Any) -> None:
        super().__init__(parent, fg_color="transparent")
        self._path_a: str | None = None
        self._path_b: str | None = None
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- File pickers ---
        for col, label, attr in [(0, "Document A  (original)", "_label_a"),
                                  (1, "Document B  (revised)", "_label_b")]:
            card = ctk.CTkFrame(self, fg_color=_CARD, corner_radius=10)
            card.grid(row=0, column=col, sticky="ew", padx=(0, 6) if col == 0 else (6, 0), pady=(0, 12))
            card.grid_columnconfigure(1, weight=1)

            _SectionHeader(card, text=label).grid(row=0, column=0, columnspan=2, padx=12, pady=(10, 4), sticky="w")
            _GhostButton(card, text="📂  Open", width=100,
                         command=lambda c=col: self._pick(c)).grid(row=1, column=0, padx=12, pady=(0, 12))
            lbl = ctk.CTkLabel(card, text="No file selected",
                               font=ctk.CTkFont(family="Segoe UI", size=12),
                               text_color=_MUTED, anchor="w")
            lbl.grid(row=1, column=1, padx=8, pady=(0, 12), sticky="w")
            setattr(self, attr, lbl)

        # --- Run button ---
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        btn_row.grid_columnconfigure(0, weight=1)

        _OrangeButton(btn_row, text="Run Comparison", command=self._compare).grid(
            row=0, column=0, sticky="ew")

        # --- Results ---
        results = ctk.CTkFrame(self, fg_color="transparent")
        results.grid(row=2, column=0, columnspan=2, sticky="nsew")
        results.grid_columnconfigure(0, weight=1)
        results.grid_rowconfigure(1, weight=1)

        # Metrics row (filled after comparison)
        self._metrics_frame = ctk.CTkFrame(results, fg_color=_CARD, corner_radius=10)
        self._metrics_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self._metrics_labels: list[ctk.CTkLabel] = []

        _SectionHeader(results, text="COMPARISON REPORT").grid(row=1, column=0, sticky="w", pady=(0, 4))
        self._result_box = _ResultBox(results)
        self._result_box.grid(row=2, column=0, sticky="nsew")
        self._result_box.set_text("Comparison output will appear here…")

        # Status
        self._status = ctk.CTkLabel(
            self, text="", font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=_MUTED, anchor="w",
        )
        self._status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _pick(self, col: int) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Documents", "*.pdf *.docx"), ("PDF", "*.pdf"), ("Word", "*.docx")]
        )
        if path:
            if col == 0:
                self._path_a = path
                self._label_a.configure(text=Path(path).name, text_color=_TEXT)
            else:
                self._path_b = path
                self._label_b.configure(text=Path(path).name, text_color=_TEXT)

    def _compare(self) -> None:
        if not self._path_a or not self._path_b:
            self._status.configure(text="Please open both documents first.", text_color=_YELLOW)
            return
        self._result_box.set_text("Comparing…")
        self._status.configure(text="Running…", text_color=_ORANGE)

        def _work() -> None:
            try:
                docs_a = _load_file(self._path_a)  # type: ignore[arg-type]
                docs_b = _load_file(self._path_b)  # type: ignore[arg-type]
                result = compare_documents(docs_a, docs_b)

                self._render_metrics(result)

                report = "\n".join([
                    f"Document A:  {Path(self._path_a).name}  ({result.doc_a_chars:,} chars)",
                    f"Document B:  {Path(self._path_b).name}  ({result.doc_b_chars:,} chars)",
                    "",
                    f"Similarity:  {result.similarity_assessment.upper()}",
                    "",
                    "─" * 60,
                    "",
                    result.semantic_summary,
                ])
                self._result_box.set_text(report)
                self._status.configure(text="Comparison complete.", text_color=_GREEN)
            except Exception as exc:
                self._result_box.set_text(f"Error: {exc}")
                self._status.configure(text="Failed.", text_color=_ORANGE_DARK)

        threading.Thread(target=_work, daemon=True).start()

    def _render_metrics(self, result: Any) -> None:
        for w in self._metrics_frame.winfo_children():
            w.destroy()

        stats = [
            ("Total Changes", str(result.exact_diff.total_changes)),
            ("Lines Added",   str(result.exact_diff.iterable_item_added)),
            ("Lines Removed", str(result.exact_diff.iterable_item_removed)),
            ("Lines Modified", str(result.exact_diff.values_changed)),
        ]
        for col, (label, value) in enumerate(stats):
            self._metrics_frame.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(
                self._metrics_frame, text=value,
                font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
                text_color=_ORANGE,
            ).grid(row=0, column=col, padx=16, pady=(14, 2))
            ctk.CTkLabel(
                self._metrics_frame, text=label,
                font=ctk.CTkFont(family="Segoe UI", size=11),
                text_color=_MUTED,
            ).grid(row=1, column=col, padx=16, pady=(0, 14))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Document Intelligence Toolkit")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(fg_color=_BG)
        self._build()

    def _build(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- Header ---
        header = ctk.CTkFrame(self, fg_color=_CARD, corner_radius=0, height=64)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        # Orange accent stripe
        ctk.CTkFrame(header, fg_color=_ORANGE, width=4, corner_radius=0).grid(
            row=0, column=0, sticky="ns", padx=(0, 16))

        ctk.CTkLabel(
            header, text="Document Intelligence Toolkit",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
            text_color=_TEXT,
        ).grid(row=0, column=1, sticky="w")

        # Mode badge
        mode_text = "⚡ Rule-based mode" if not has_api_key() else "✦ LLM mode"
        mode_color = _YELLOW if not has_api_key() else _GREEN
        ctk.CTkLabel(
            header, text=mode_text,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color=mode_color,
        ).grid(row=0, column=2, padx=20, sticky="e")

        # --- Offline notice ---
        if not has_api_key():
            notice = ctk.CTkFrame(self, fg_color="#2A2210", corner_radius=0)
            notice.grid(row=1, column=0, sticky="ew")
            ctk.CTkLabel(
                notice,
                text="  ⚠  No API key detected — running fully offline with rule-based "
                     "extraction, extractive summarisation, and text-diff comparison.",
                font=ctk.CTkFont(family="Segoe UI", size=12),
                text_color=_YELLOW,
                anchor="w",
            ).grid(row=0, column=0, padx=12, pady=7, sticky="w")

        # --- Tab view ---
        tabs = ctk.CTkTabview(
            self,
            fg_color=_BG,
            segmented_button_fg_color=_CARD,
            segmented_button_selected_color=_ORANGE,
            segmented_button_selected_hover_color=_ORANGE_DARK,
            segmented_button_unselected_color=_CARD,
            segmented_button_unselected_hover_color=_CARD2,
            text_color=_TEXT,
            text_color_disabled=_MUTED,
        )
        tabs.grid(row=2, column=0, sticky="nsew", padx=20, pady=16)
        tabs.add("  Analyse  ")
        tabs.add("  Compare  ")

        AnalyseTab(tabs.tab("  Analyse  ")).pack(fill="both", expand=True, padx=8, pady=8)
        CompareTab(tabs.tab("  Compare  ")).pack(fill="both", expand=True, padx=8, pady=8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
