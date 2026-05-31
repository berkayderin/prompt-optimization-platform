"""LLM saglayicilarina tek bir arayuz uzerinden erisim.

DeepSeek, GLM, OpenAI ve yerel Ollama hepsi OpenAI-uyumlu bir API sundugu icin
tek bir istemci ile kullanilir; yalnizca base_url ve api_key degisir. API anahtari
gerektirmeden gelistirme yapabilmek icin ayrica "mock" saglayici eklenmistir.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

# Her saglayicinin OpenAI-uyumlu uc noktasi ve anahtar/model icin kullanilacak
# ortam degiskenlerinin adlari.
PROVIDERS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-chat",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "GLM_API_KEY",
        "model_env": "GLM_MODEL",
        "default_model": "glm-4-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "key_env": None,  # yerel calisir, anahtar gerekmez
        "model_env": "OLLAMA_MODEL",
        "default_model": "llama3",
    },
}


@dataclass
class LLMResponse:
    """Bir model cagrisinin sonucu ve olcum verileri."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Secilen saglayiciya gore model cagrisi yapan ince sarmalayici."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "mock")).lower()
        self.model = model
        self._client = None

        if self.provider == "mock":
            return

        cfg = PROVIDERS.get(self.provider)
        if cfg is None:
            raise ValueError(f"Bilinmeyen saglayici: {self.provider}")

        api_key = os.getenv(cfg["key_env"]) if cfg["key_env"] else "ollama"
        if not api_key:
            raise RuntimeError(
                f"{self.provider} icin API anahtari bulunamadi ({cfg['key_env']}). "
                ".env dosyasini kontrol edin."
            )

        # OpenAI SDK yalnizca gercek saglayici secildiginde gerekir; mock modunda
        # bagimliligi zorunlu kilmamak icin tembel (lazy) import yapilir.
        from openai import OpenAI

        self.model = model or os.getenv(cfg["model_env"], cfg["default_model"])
        self._client = OpenAI(api_key=api_key, base_url=cfg["base_url"])

    def complete(self, system: str, user: str, temperature: float = 0.7) -> LLMResponse:
        """system + user mesajiyla modeli cagirir ve sonucu olcerek dondurur."""
        if self.provider == "mock":
            return self._mock_complete(system, user)

        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_s=latency,
            model=self.model,
        )

    def _mock_complete(self, system: str, user: str) -> LLMResponse:
        """Anahtar olmadan tum akisin test edilebilmesi icin sahte yanit.

        Gercek bir cikarim yapmaz; girdiyle orantili sabit bir metin ve makul
        token/gecikme degerleri uretir. Anlamli dogruluk metrikleri icin gercek
        bir saglayici (DeepSeek/GLM) ayarlanmalidir.
        """
        time.sleep(0.05)  # gercekci kucuk bir gecikme
        text = (
            "[MOCK] Bu temsili bir yanittir. Gercek sonuclar icin .env dosyasinda "
            "bir saglayici (DeepSeek/GLM) ayarlayin."
        )
        prompt_tokens = len((system + " " + user).split())
        completion_tokens = len(text.split())
        return LLMResponse(text, prompt_tokens, completion_tokens, 0.05, "mock")
