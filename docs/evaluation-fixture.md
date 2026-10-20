# 실측 평가 결과

**가상 테스트 문서 기반 결과이며 실제 규정 서비스 정확도가 아닙니다.**

데이터: tests/fixtures/cases.json

LLM: ollama / qwen3.5:2b; 임베딩: ollama / qwen3-embedding:0.6b

```json
{
  "baseline": {
    "retrieval_hit_at_3": 0.9,
    "eligibility_accuracy": 0.4,
    "citation_accuracy": 0.2,
    "document_procedure_completeness": 0.4,
    "follow_up_appropriateness": 0.7,
    "average_latency_seconds": 5.260300000000001,
    "total_tokens": 10249,
    "estimated_cost_usd": 0.051245
  },
  "agentic": {
    "retrieval_hit_at_3": 0.9,
    "eligibility_accuracy": 0.5,
    "citation_accuracy": 0.8,
    "document_procedure_completeness": 0.2,
    "follow_up_appropriateness": 0.7,
    "average_latency_seconds": 9.3306,
    "total_tokens": 26389,
    "estimated_cost_usd": 0.13194499999999998
  }
}
```
