"""Connection-test stubs. They never invent policy decisions."""


class FastLLM:
    def __init__(self, name: str = "fake", usd_per_1k_tokens: float = 0):
        self.name = name
        self.usd_per_1k = usd_per_1k_tokens


class QualityLLM(FastLLM):
    pass
