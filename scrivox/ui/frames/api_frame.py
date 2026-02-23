"""API key entry fields with show/hide toggle and test button."""

import os
import tkinter as tk
from tkinter import ttk
import threading

from ..theme import COLORS, FONTS


class ApiFrame(ttk.LabelFrame):
    """API key fields for HuggingFace and OpenRouter."""

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, text="API KEYS", **kwargs)
        self.config_manager = config_manager

        self.hf_token_var = tk.StringVar()
        self.openrouter_key_var = tk.StringVar()
        self._show_keys = False

        self._build()
        self._load_from_config()

    def _build(self):
        # HuggingFace Token
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="HF Token:").pack(side=tk.LEFT)
        self._hf_entry = ttk.Entry(row, textvariable=self.hf_token_var, show="*")
        self._hf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))

        # OpenRouter Key
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(row, text="OR Key:   ").pack(side=tk.LEFT)
        self._or_entry = ttk.Entry(row, textvariable=self.openrouter_key_var, show="*")
        self._or_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))

        # Buttons row
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._show_btn = ttk.Button(btn_row, text="Show", command=self._toggle_show, width=6)
        self._show_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._test_btn = ttk.Button(btn_row, text="Test Keys", command=self._test_keys)
        self._test_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._status_label = ttk.Label(btn_row, text="", style="Dim.TLabel")
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _load_from_config(self):
        """Load keys from config, falling back to env vars."""
        hf = ""
        or_key = ""
        if self.config_manager:
            hf, or_key = self.config_manager.get_credentials()
        if not hf:
            hf = os.environ.get("HF_TOKEN", "")
        if not or_key:
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.hf_token_var.set(hf)
        self.openrouter_key_var.set(or_key)

    def _toggle_show(self):
        self._show_keys = not self._show_keys
        show_char = "" if self._show_keys else "*"
        self._hf_entry.configure(show=show_char)
        self._or_entry.configure(show=show_char)
        self._show_btn.configure(text="Hide" if self._show_keys else "Show")

    def _test_keys(self):
        """Test API keys in a background thread."""
        self._status_label.configure(text="Testing...", style="Dim.TLabel")
        self._test_btn.configure(state=tk.DISABLED)

        def _do_test():
            results = []
            hf = self.hf_token_var.get().strip()
            or_key = self.openrouter_key_var.get().strip()

            if hf:
                try:
                    import requests
                    resp = requests.get(
                        "https://huggingface.co/api/whoami-v2",
                        headers={"Authorization": f"Bearer {hf}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        results.append("HF: OK")
                    else:
                        results.append(f"HF: {resp.status_code}")
                except Exception as e:
                    results.append(f"HF: {type(e).__name__}")
            else:
                results.append("HF: not set")

            if or_key:
                try:
                    import requests
                    resp = requests.get(
                        "https://openrouter.ai/api/v1/models",
                        headers={"Authorization": f"Bearer {or_key}"},
                        timeout=10,
                    )
                    if resp.status_code == 200:
                        results.append("OR: OK")
                    else:
                        results.append(f"OR: {resp.status_code}")
                except Exception as e:
                    results.append(f"OR: {type(e).__name__}")
            else:
                results.append("OR: not set")

            status_text = " | ".join(results)
            all_ok = all("OK" in r for r in results if "not set" not in r)

            try:
                self.after(0, lambda: self._update_test_status(status_text, all_ok))
            except Exception:
                pass

        threading.Thread(target=_do_test, daemon=True).start()

    def _update_test_status(self, text, all_ok):
        style = "Success.TLabel" if all_ok else "Error.TLabel"
        self._status_label.configure(text=text, style=style)
        self._test_btn.configure(state=tk.NORMAL)

    def save_to_config(self):
        """Save current keys to config manager."""
        if self.config_manager:
            self.config_manager.set_credentials(
                hf_token=self.hf_token_var.get().strip(),
                openrouter_key=self.openrouter_key_var.get().strip(),
            )

    def get_hf_token(self):
        return self.hf_token_var.get().strip()

    def get_openrouter_key(self):
        return self.openrouter_key_var.get().strip()
