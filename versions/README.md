# Versioned artifacts

공통 입력 데이터와 노트북은 프로젝트 루트에 두고, LLM 추출 결과와 비교 분석 산출물만 버전별로 보관합니다.

## Directory layout

```text
versions/
├─ v2/
│  ├─ extraction/              # v2 JSONL 및 CSV 결과
│  └─ analysis/                # 최초 관계 분석 워크북과 이미지
├─ v3/
│  ├─ script/                  # explicit-only 추출 스크립트
│  ├─ extraction/              # v3 JSONL 및 CSV 결과
│  └─ analysis/comparison/     # v3 비교 워크북과 검증 결과
├─ v4/
│  ├─ script/                  # direct 판독 + Synovitis←Effusion 정책
│  ├─ extraction/              # v4 JSONL 및 CSV 결과
│  ├─ analysis/comparison/     # v4 비교 워크북과 검증 결과
│  └─ review/                  # 수동 불일치 검토 자료
└─ v5/
   ├─ script/                  # direct-evidence-only 추출 및 v4 재처리 스크립트
   ├─ extraction/              # v5 JSONL 및 CSV 결과
   └─ analysis/comparison/     # v5 비교 워크북과 검증 결과
```

## 버전별 목적과 정책 변화

모든 버전은 같은 58개 report와 같은 `gemma4:e4b` 모델을 사용합니다. 결과가 다른 주된 이유는 모델이나 원본 데이터가 바뀌어서가 아니라, **타깃 정의·직접 음성 판정 기준·관련 소견 후처리 정책이 버전마다 달라졌기 때문**입니다.

### v2 — 초기 탐색 버전

- 최초 12개 타깃 추출 결과로, 직접 판독과 `uncertain_or_indirect`를 함께 탐색했습니다.
- `related_findings` 필드에 해부학적 위치나 간접 소견이 포함될 수 있어 직접 근거와 관련 소견의 경계가 이후 버전보다 느슨합니다.
- 원문 근거 인용 검증도 이후 버전보다 덜 엄격하여 일부 `invalid_evidence_rows`가 남아 있습니다.
- 타깃 간 관계와 LLM-원본 라벨 차이를 처음 살펴보는 기준선 역할을 합니다.
- 당시 실행 스크립트는 현재 보관되어 있지 않으며, 결과 파일의 `prompt_version=v2.0` 메타데이터와 산출물만 남아 있습니다.

### v3 — explicit-only 버전

- report에 타깃 자체가 직접 언급된 경우만 `P`(양성), `N`(음성), `U`(불확실)로 판독합니다.
- 관련 소견만 있거나 타깃이 직접 언급되지 않으면 `M`(not mentioned)으로 유지합니다.
- 예를 들어 Effusion만으로 Synovitis를 추론하거나, marrow edema만으로 Contusion을 추론하지 않습니다.
- 모든 `P/N/U`에는 report에서 복사한 정확한 원문 근거가 필요하고, 관련 소견 필드는 비활성화했습니다.
- 따라서 v2에서 간접·불확실 소견으로 분류되던 사례가 v3에서는 직접 판독 또는 `M`으로 재분류되어 결과 분포가 달라졌습니다.

### v4 — 직접 음성 강화 + Synovitis←Effusion 보정

- v3의 직접 근거 우선 원칙을 유지하면서 `normal`, `intact`, `preserved`, `no`, `without` 등의 **해부학적으로 범위가 명확한 정상·부정 표현**을 직접 음성으로 더 명확하게 판독하도록 프롬프트를 보강했습니다.
- ACL/MCL, 반월상연골, 구획별 OA, Effusion, Synovitis, Contusion, Fracture의 음성 판정 범위를 구체적으로 정의했습니다.
- LLM의 직접 판독(`direct_status`)과 후처리 결과(`final_status`)를 분리해 저장합니다.
- Synovitis가 직접 언급되지 않은 경우에만, 직접 판독된 Effusion을 관련 소견으로 사용합니다.
  - Effusion 직접 양성 → `related_positive`
  - Effusion 직접 음성 → `related_negative`
  - Effusion도 미언급 → Synovitis를 `not_mentioned`로 유지
- 직접 Synovitis 판독이 있으면 항상 그것을 우선하며, 다른 타깃에는 아직 관련 소견 보정을 적용하지 않습니다.

### v5 — direct-evidence-only 기준 버전

- v4에서 실험한 `Synovitis←Effusion` 관련 소견 보정을 제거하고, 12개 타깃 모두 **report에 직접 기록된 근거만으로** 판독합니다.
- 각 타깃의 `final_status`와 `final_binary`는 항상 `direct_status`와 `direct_binary`와 같으며, 다른 타깃이나 간접 소견으로 값을 보충하지 않습니다.
- `P/N/U`는 report 원문에서 복사한 정확한 인용문을 요구하고, 타깃이 직접 언급되지 않으면 `M`으로 유지합니다.
- 직접 음성은 해부학적·문법적 범위가 명확한 `normal`, `intact`, `preserved`, `no`, `without` 등의 표현에만 적용합니다.
- 타깃 정의를 더 명시적으로 고정했습니다. 예를 들어 Contusion은 bone contusion/bruise만 인정하고, OA는 구획이 일치하는 명시적 OA/arthrosis 진단을 요구하며, Effusion과 Synovitis는 서로를 대신해 추론하지 않습니다.
- `extract_58_gemma4_v5.py`는 v5 프롬프트로 58건을 새로 추출합니다. `reprocess_v4_direct_only_reference.py`는 모델을 다시 호출하지 않고 기존 v4 직접 판독에서 관련 소견 보정만 제거하는 참조용 스크립트입니다.

## 결과가 달라지는 구체적인 이유

1. **타깃 정의가 더 엄격해졌습니다.** 예를 들어 muscle contusion은 Contusion 타깃에서 제외하고 bone contusion만 인정합니다. 단순 cartilage abnormality도 명시적 OA 진단과 구분합니다.
2. **음성 판정 범위가 달라졌습니다.** v4와 v5는 구조물이 정상·온전하다는 문장이 어느 타깃까지 적용되는지 문법적·해부학적 범위를 더 구체적으로 지정합니다.
3. **미언급과 간접 소견의 처리 방식이 다릅니다.** v3는 모두 미언급으로 남기고, v4는 Synovitis에 한해 Effusion 근거를 결정론적으로 보충하며, v5는 다시 모든 타깃에서 직접 근거만 사용합니다.
4. **비교 대상의 분모가 달라집니다.** `P/N`으로 판독된 경우만 0/1 일치율 계산에 포함되므로, 판독 가능 건수가 늘면 일치율도 함께 변할 수 있습니다.
5. **원본 라벨과 report가 항상 일치하지 않습니다.** 아래 일치율은 기존 58개 라벨을 기준으로 한 값이며, 불일치가 반드시 LLM 오류를 의미하지는 않습니다. 명시적 report 소견과 원본 라벨이 충돌하는 사례는 별도 검토가 필요합니다.

## 버전별 결과 요약

전체 비교 단위는 `58 reports × 12 targets = 696건`입니다. `0/1 판독 가능 건수`는 직접 `P/N` 또는 v4 관련 소견 후처리로 이진값이 만들어진 건수이고, 일치율은 그 건수 안에서 기존 라벨과 같은 비율입니다. v5에서는 후처리 보정이 없으므로 direct와 final 결과가 같습니다.

| 결과 | 0/1 판독 가능 건수 | 기존 라벨 일치율 | 해석 |
|---|---:|---:|---|
| v2 | 377 / 696 | 80.6% | 초기 탐색 기준선; 간접·불확실 소견 경계가 비교적 느슨함 |
| v3 direct | 399 / 696 | 81.2% | 직접 근거만 사용하여 감사 가능성을 높임 |
| v4 direct | 411 / 696 | 82.7% | 직접 음성 표현을 더 명확히 판독하여 coverage와 일치율이 증가 |
| v4 final | 448 / 696 | 79.7% | Effusion→Synovitis 보정으로 coverage는 증가하지만 전체 일치율은 감소 |
| v5 direct/final | 425 / 696 | 81.9% | 모든 관련 소견 보정을 제거하고 직접 원문 근거만 사용하는 기준 결과 |

### v4 Synovitis 결과 해석

- 직접 Synovitis 판독만 사용하면 18건을 0/1로 비교할 수 있고, 그중 14건이 기존 라벨과 일치합니다(`77.8%`).
- Effusion 보정을 적용하면 비교 가능 건수가 55건으로 증가하지만, 일치는 31건입니다(`56.4%`).
- 즉 Effusion은 Synovitis의 **검토 우선순위 또는 확률 조정용 관련 소견**으로는 유용할 수 있지만, 현재 58건 결과에서는 Synovitis의 확정 0/1 라벨로 치환하면 오탐이 크게 늘어납니다.
- 따라서 v4의 `related_positive/related_negative`는 직접 판독과 구분해서 사용해야 합니다.

### v5 결과 해석

- 696개 비교 단위 중 `P/N` 이진 판독은 425건이며, 348건이 기존 라벨과 일치합니다(`81.9%`).
- 상태 분포는 `P` 255건, `N` 170건, `U` 25건, `M` 246건입니다.
- 모든 `P/N/U` 직접 근거는 report 원문에서 확인되었고, `invalid_direct_evidence`와 `invalid_related_evidence`는 모두 0건입니다.
- v4 final보다 판독 가능 건수는 448건에서 425건으로 감소하지만, 관련 소견을 확정 라벨로 치환하지 않아 일치율은 `79.7%`에서 `81.9%`로 회복됩니다.
- v4 direct와 비교하면 coverage는 411건에서 425건으로 늘고 일치율은 `82.7%`에서 `81.9%`로 소폭 낮아졌습니다. 이는 v5 타깃 정의와 직접 음성 규칙을 반영해 새로 추출한 결과이므로, 단순히 v4 후처리만 제거한 결과와는 구분해야 합니다.

## Run

프로젝트 루트에서 실행합니다. 기본 입력은 루트의 `train.csv`이며 결과는 해당 버전의 `extraction` 폴더에 저장됩니다.

```powershell
python versions/v3/script/extract_58_gemma4.py
python versions/v4/script/extract_58_gemma4_v4.py
python versions/v5/script/extract_58_gemma4_v5.py
```

기존 v4 직접 판독을 유지한 채 관련 소견 보정만 제거한 참조 결과가 필요하면 다음 스크립트를 사용합니다.

```powershell
python versions/v5/script/reprocess_v4_direct_only_reference.py
```

`train.csv`, `train_with_targets.csv`, `train_report_only.csv`, `data_check.ipynb`, `참고/`는 모든 버전이 공유하는 자료이므로 루트에 유지합니다.
