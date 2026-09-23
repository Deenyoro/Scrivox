"""API key entry fields with provider selection, show/hide toggle, and test button."""

import os
import tkinter as tk
from tkinter import ttk
import threading

from ...core.constants import LLM_PROVIDERS, DEFAULT_LLM_PROVIDER
from ...core.diarizer import _get_bundled_models_dir
from ..theme import SP_L, SP_S, SP_XS, px
from ..widgets import WrappingLabel, call_in_ui


class ApiFrame(ttk.Frame):
    """API key fields for Hugging Face and AI (LLM) providers.

    Lives on the "AI services" tab of the Settings dialog.
    """

    def __init__(self, parent, config_manager=None, **kwargs):
        kwargs.setdefault("padding", (px(16), px(12)))
        super().__init__(parent, **kwargs)
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
        ttk.Label(self, text="Speaker identification", style="Header.TLabel").pack(
            anchor=tk.W, pady=(0, SP_XS))
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, SP_XS))
        ttk.Label(row, text="Hugging Face token").pack(side=tk.LEFT)
        self._hf_entry = ttk.Entry(row, textvariable=self.hf_token_var, show="\u2022")
        self._hf_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_S, 0))

        if self._has_bundled:
            hf_hint = "Optional: this download already includes the speaker models."
        else:
            hf_hint = ("Free. Create a \"Read\" token at huggingface.co/settings/tokens and "
                       "accept the terms of pyannote/speaker-diarization-community-1.")
        WrappingLabel(self, text=hf_hint, style="Dim.TLabel", justify=tk.LEFT).pack(
            fill=tk.X, pady=(0, SP_L))

        ttk.Label(self, text="AI service (summaries, translation, on-screen content)",
                  style="Header.TLabel").pack(anchor=tk.W, pady=(0, SP_XS))
        # LLM Provider
        row = ttk.Frame(self)
        row.pack(fill=tk.X, pady=(0, SP_XS))
        ttk.Label(row, text="Provider").pack(side=tk.LEFT)
        providers = list(LLM_PROVIDERS.keys()) + ["Custom"]
        provider_combo = ttk.Combobox(row, textvariable=self.provider_var,
                                       values=providers, state="readonly", width=16)
        provider_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(SP_S, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)

        # API Key (for OpenRouter/OpenAI/Ollama/Custom)
        self._api_key_frame = ttk.Frame(self)
        self._api_key_frame.pack(fill=tk.X, pady=(0, SP_XS))
        ttk.Label(self._api_key_frame, text="API key").pack(side=tk.LEFT)
        self._or_entry = ttk.Entry(self._api_key_frame, textvariable=self.openrouter_key_var,
                                   show="\u2022")
        self._or_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_S, 0))

        # Anthropic API Key (shown when Anthropic selected)
        self._anthropic_key_frame = ttk.Frame(self)
        ttk.Label(self._anthropic_key_frame, text="API key").pack(side=tk.LEFT)
        self._anthropic_entry = ttk.Entry(self._anthropic_key_frame,
                                           textvariable=self.anthropic_key_var, show="\u2022")
        self._anthropic_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_S, 0))

        # Per-provider key hint (text set in _on_provider_change)
        self._key_hint = WrappingLabel(self, text="", style="Dim.TLabel", justify=tk.LEFT)

        # Custom base URL (hidden by default)
        self._custom_frame = ttk.Frame(self)
        ttk.Label(self._custom_frame, text="Server URL").pack(side=tk.LEFT)
        ttk.Entry(self._custom_frame, textvariable=self.custom_base_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(SP_S, 0))

        # Buttons row
        btn_row = ttk.Frame(self)
        btn_row.pack(fill=tk.X, pady=(SP_XS, SP_XS))

        self._show_btn = ttk.Button(btn_row, text="Show keys", style="Small.TButton",
                                     command=self._toggle_show)
        self._show_btn.pack(side=tk.LEFT, padx=(0, SP_XS))

        self._test_btn = ttk.Button(btn_row, text="Test keys", style="Small.TButton",
                                     command=self._test_keys)
        self._test_btn.pack(side=tk.LEFT, padx=(0, SP_XS))

        # Test status on its own full-width row so long results wrap instead
        # of being clipped next to the buttons
        self._status_label = WrappingLabel(self, text="", style="Dim.TLabel", justify=tk.LEFT)
        self._status_label.pack(fill=tk.X, pady=(0, SP_S), anchor=tk.W)

        where = self.config_manager.path if self.config_manager else "scrivox_config.json"
        self._storage_note = WrappingLabel(
            self, text=f"Keys are saved on this computer in {where}. "
                       "Keys in a .env file next to Scrivox also work.",
            style="Dim.TLabel", justify=tk.LEFT)
        self._storage_note.pack(fill=tk.X, pady=(SP_S, 0))

        self._on_provider_change()

    _KEY_HINTS = {
        "OpenRouter": "Get a key at openrouter.ai/keys (one key, many models).",
        "OpenAI": "Get a key at platform.openai.com/api-keys.",
        "Anthropic": "Get a key at console.anthropic.com/settings/keys.",
        "Ollama (local)": "Runs on this computer. No key needed; Ollama must be running.",
        "Custom": "Any OpenAI-compatible server. A key is needed unless the URL is local.",
    }

    def _on_provider_change(self, event=None):
        provider = self.provider_var.get()

        # Show/hide Anthropic key vs standard API key
        if provider == "Anthropic":
            self._api_key_frame.pack_forget()
            self._anthropic_key_frame.pack(fill=tk.X, pady=(0, SP_XS),
                                            before=self._show_btn.master)
            self._custom_frame.pack_forget()
        else:
            self._anthropic_key_frame.pack_forget()
            self._api_key_frame.pack(fill=tk.X, pady=(0, SP_XS),
                                      before=self._show_btn.master)
            if provider == "Custom":
                self._custom_frame.pack(fill=tk.X, pady=(0, SP_XS),
                                         before=self._show_btn.master)
            else:
                self._custom_frame.pack_forget()

        # Per-provider key hint, always just above the buttons
        self._key_hint.configure(
            text=self._KEY_HINTS.get(provider, "An API key may be required."))
        self._key_hint.pack_forget()
        self._key_hint.pack(fill=tk.X, pady=(0, SP_XS), anchor=tk.W,
                            before=self._show_btn.master)

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
        show_char = "" if self._show_keys else "\u2022"
        self._hf_entry.configure(show=show_char)
        self._or_entry.configure(show=show_char)
        self._anthropic_entry.configure(show=show_char)
        self._show_btn.configure(text="Hide keys" if self._show_keys else "Show keys")

    def _test_keys(self):
        """Test API keys in a background thread."""
        self._status_label.configure(text="Testing\u2026", style="Dim.TLabel")
        self._test_btn.configure(state=tk.DISABLED)

        # Read Tk variables on the main thread before the worker starts
        hf = self.hf_token_var.get().strip()
        or_key = self.openrouter_key_var.get().strip()
        ant_key = self.anthropic_key_var.get().strip()
        provider = self.provider_var.get()
        base_url = self.get_api_base()

        def _do_test():
            results = []

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

            # Success requires at least one key actually tested OK — an empty
            # generator would make all() return True for "not set" results
            tested = [r for r in results if "not set" not in r and "No keys" not in r]
            all_ok = bool(tested) and all("OK" in r for r in tested)
            return " | ".join(results), all_ok

        def _worker():
            # Always report back so the Test button never stays disabled
            try:
                status_text, all_ok = _do_test()
            except Exception as e:
                status_text, all_ok = f"Test failed: {type(e).__name__}", False
            try:
                call_in_ui(self, self._update_test_status, status_text, all_ok)
            except Exception:
                pass  # window destroyed

        threading.Thread(target=_worker, daemon=True).start()

    def _update_test_status(self, text, all_ok):
        if not self.winfo_exists():
            return
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

    def first_empty_field(self):
        """The key field a user most likely came here to fill in."""
        if not self.hf_token_var.get().strip() and not self._has_bundled:
            return self._hf_entry
        if self.provider_var.get() == "Anthropic":
            return self._anthropic_entry
        return self._or_entry

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
