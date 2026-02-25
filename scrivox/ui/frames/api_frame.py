"""API key entry fields with provider selection, show/hide toggle, and test button."""

import os
import tkinter as tk
from tkinter import ttk
import threading

from ...core.constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
from ...core.diarizer import _get_bundled_models_dir
from ..theme import COLORS, FONTS


class ApiFrame(ttk.LabelFrame):
    """API key fields for HuggingFace and LLM providers."""

    def __init__(self, parent, config_manager=None, **kwargs):
        super().__init__(parent, text="API KEYS", **kwargs)
        self.config_manager = config_manager

        self.hf_token_var = tk.StringVar()
        self.openrouter_key_var = tk.StringVar()
        self.anthropic_key_var = tk.StringVar()
        self.provider_var = tk.StringVar(value=DEFAULT_LLM_PROVIDER)
        self.custom_base_var = tk.StringVar()
        self._show_keys = False
        self._has_bundled = _get_bundled_models_dir() is not None

        self._build()
        self._load_from_config()

    def _build(self):
        # HuggingFace Token — always shown
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Label(row, text="HF Token:").pack(side=tk.LEFT)
        self._hf_entry = ttk.Entry(row, textvariable=self.hf_token_var, show="*")
        self._hf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))

        if self._has_bundled:
            ttk.Label(self, text="Optional \u2014 bundled models found",
                      style="Dim.TLabel").pack(padx=8, pady=(0, 2), anchor=tk.W)
        else:
            ttk.Label(self, text="Required to download diarization models",
                      style="Dim.TLabel").pack(padx=8, pady=(0, 2), anchor=tk.W)

        # LLM Provider
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(row, text="Provider:").pack(side=tk.LEFT)
        providers = list(LLM_PROVIDERS.keys()) + ["Custom"]
        provider_combo = ttk.Combobox(row, textvariable=self.provider_var,
                                       values=providers, state="readonly", width=16)
        provider_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # API Key (for OpenRouter/OpenAI/Ollama/Custom)
        self._api_key_frame = ttk.Frame(self)
        self._api_key_frame.pack(fill=tk.X, padx=8, pady=(0, 4))
        ttk.Label(self._api_key_frame, text="API Key:  ").pack(side=tk.LEFT)
        self._or_entry = ttk.Entry(self._api_key_frame, textvariable=self.openrouter_key_var, show="*")
        self._or_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))

        # Anthropic API Key (shown when Anthropic selected)
        self._anthropic_key_frame = ttk.Frame(self)
        ttk.Label(self._anthropic_key_frame, text="API Key:  ").pack(side=tk.LEFT)
        self._anthropic_entry = ttk.Entry(self._anthropic_key_frame,
                                           textvariable=self.anthropic_key_var, show="*")
        self._anthropic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 4))

        # Anthropic hint
        self._anthropic_hint = ttk.Label(
            self, text="Get your key at console.anthropic.com/settings/keys",
            style="Dim.TLabel")

        # Custom base URL (hidden by default)
        self._custom_frame = ttk.Frame(self)
        ttk.Label(self._custom_frame, text="Base URL:").pack(side=tk.LEFT)
        ttk.Entry(self._custom_frame, textvariable=self.custom_base_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))

        # Buttons row
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))

        self._show_btn = ttk.Button(btn_row, text="Show", command=self._toggle_show, width=6)
        self._show_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._test_btn = ttk.Button(btn_row, text="Test Keys", command=self._test_keys)
        self._test_btn.pack(side=tk.LEFT, padx=(0, 4))

        self._status_label = ttk.Label(btn_row, text="", style="Dim.TLabel")
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _on_provider_change(self, event=None):
        provider = self.provider_var.get()

        # Show/hide Anthropic key vs standard API key
        if provider == "Anthropic":
            self._api_key_frame.pack_forget()
            self._anthropic_key_frame.pack(fill=tk.X, padx=8, pady=(0, 4),
                                            before=self._show_btn.master)
            self._anthropic_hint.pack(padx=8, pady=(0, 4), anchor=tk.W,
                                       before=self._show_btn.master)
            self._custom_frame.pack_forget()
        else:
            self._anthropic_key_frame.pack_forget()
            self._anthropic_hint.pack_forget()
            self._api_key_frame.pack(fill=tk.X, padx=8, pady=(0, 4),
                                      before=self._show_btn.master)
            if provider == "Custom":
                self._custom_frame.pack(fill=tk.X, padx=8, pady=(0, 4),
                                         before=self._show_btn.master)
            else:
                self._custom_frame.pack_forget()

    def _load_from_config(self):
        """Load keys from config, falling back to env vars."""
        hf = ""
        or_key = ""
        ant_key = ""
        if self.config_manager:
            hf, or_key, ant_key = self.config_manager.get_credentials()
        if not hf:
            hf = os.environ.get("HF_TOKEN", "")
        if not or_key:
            or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not ant_key:
            ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.hf_token_var.set(hf)
        self.openrouter_key_var.set(or_key)
        self.anthropic_key_var.set(ant_key)

        # Load provider setting
        if self.config_manager:
            provider = self.config_manager.get("api", "provider", DEFAULT_LLM_PROVIDER)
            self.provider_var.set(provider)
            custom_base = self.config_manager.get("api", "custom_base", "")
            self.custom_base_var.set(custom_base)
            self._on_provider_change()

    def _toggle_show(self):
        self._show_keys = not self._show_keys
        show_char = "" if self._show_keys else "*"
        self._hf_entry.configure(show=show_char)
        self._or_entry.configure(show=show_char)
        self._anthropic_entry.configure(show=show_char)
        self._show_btn.configure(text="Hide" if self._show_keys else "Show")

    def _test_keys(self):
        """Test API keys in a background thread."""
        self._status_label.configure(text="Testing...", style="Dim.TLabel")
        self._test_btn.configure(state=tk.DISABLED)

        def _do_test():
            results = []
            hf = self.hf_token_var.get().strip()
            or_key = self.openrouter_key_var.get().strip()
            ant_key = self.anthropic_key_var.get().strip()
            provider = self.provider_var.get()

            # Only test HF token if it's set — skip entirely on Full builds
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
                    elif resp.status_code == 401:
                        results.append("HF: Invalid (401)")
                    else:
                        results.append(f"HF: Error ({resp.status_code})")
                except Exception as e:
                    results.append(f"HF: {type(e).__name__}")

            # Test the active provider's key
            if provider == "Anthropic":
                if ant_key:
                    try:
                        import requests
                        resp = requests.post(
                            "https://api.anthropic.com/v1/messages",
                            headers={
                                "x-api-key": ant_key,
                                "anthropic-version": "2023-06-01",
                                "content-type": "application/json",
                            },
                            json={
                                "model": "claude-haiku-4-5-20251001",
                                "max_tokens": 1,
                                "messages": [{"role": "user", "content": "Hi"}],
                            },
                            timeout=15,
                        )
                        if resp.status_code == 200:
                            results.append("Anthropic: OK")
                        elif resp.status_code == 401:
                            results.append("Anthropic: Invalid (401)")
                        elif resp.status_code == 403:
                            results.append("Anthropic: Forbidden")
                        else:
                            results.append(f"Anthropic: Error ({resp.status_code})")
                    except Exception as e:
                        results.append(f"Anthropic: {type(e).__name__}")
                else:
                    results.append("Anthropic: not set")
            else:
                if or_key:
                    try:
                        import requests
                        base_url = self.get_api_base()
                        test_url = base_url.replace("/chat/completions", "/models")
                        resp = requests.get(
                            test_url,
                            headers={"Authorization": f"Bearer {or_key}"},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            results.append("API: OK")
                        elif resp.status_code == 401:
                            results.append("API: Invalid (401)")
                        else:
                            results.append(f"API: Error ({resp.status_code})")
                    except Exception as e:
                        results.append(f"API: {type(e).__name__}")
                else:
                    results.append("API: not set")

            if not results:
                results.append("No keys to test")

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
        """Save current keys and provider to config manager."""
        if self.config_manager:
            self.config_manager.set_credentials(
                hf_token=self.hf_token_var.get().strip(),
                openrouter_key=self.openrouter_key_var.get().strip(),
                anthropic_key=self.anthropic_key_var.get().strip(),
            )
            self.config_manager.set("api", "provider", self.provider_var.get())
            self.config_manager.set("api", "custom_base", self.custom_base_var.get().strip())

    def get_hf_token(self):
        return self.hf_token_var.get().strip()

    def get_openrouter_key(self):
        """Get the active LLM API key for the selected provider."""
        if self.provider_var.get() == "Anthropic":
            return self.anthropic_key_var.get().strip()
        return self.openrouter_key_var.get().strip()

    def get_anthropic_key(self):
        """Get the Anthropic API key specifically."""
        return self.anthropic_key_var.get().strip()

    def get_api_base(self):
        """Get the resolved API base URL for the selected provider."""
        provider = self.provider_var.get()
        if provider == "Custom":
            return self.custom_base_var.get().strip()
        return LLM_PROVIDERS.get(provider, LLM_PROVIDERS[DEFAULT_LLM_PROVIDER])
