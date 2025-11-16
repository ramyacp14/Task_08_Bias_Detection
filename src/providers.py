from __future__ import annotations
import os, time, random
from typing import Dict, Any, Optional

class LLMClient:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic()
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.client = genai
        else:
            raise ValueError("Unknown provider")

    def completions(self, prompt: str, temperature: float, top_p: float, max_tokens: int, seed: Optional[int]) -> Dict[str, Any]:
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=seed
            )
            msg = resp.choices[0].message.content
            usage = getattr(resp, "usage", None)
            return {"text": msg, "usage": usage.model_dump() if usage else {}}

        if self.provider == "anthropic":
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                messages=[{"role":"user","content":prompt}],
                extra_headers={"anthropic-beta":"prompt-caching-2024-07-31"}  # why: stability
            )
            return {"text": resp.content[0].text, "usage": {"input_tokens":resp.usage.input_tokens, "output_tokens":resp.usage.output_tokens}}

        if self.provider == "gemini":
            model = self.client.GenerativeModel(self.model)
            gen = model.generate_content(prompt, generation_config={"temperature": temperature, "top_p": top_p, "max_output_tokens": max_tokens})
            return {"text": gen.text, "usage": {}}

        raise RuntimeError("Unsupported provider")
