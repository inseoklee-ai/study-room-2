# 패턴 1 · 로짓 마스킹 — 실습 기록 (노트북 1/4)

> 📘 **책 본문 개념 정리는 → [ch02-pattern01-logit-masking.md](ch02-pattern01-logit-masking.md)**
> 이 문서는 예제 노트북을 **실제로 돌린 기록**이다.
>
> 예제: `examples/01_logits_masking/0_logits_masking.ipynb` ✅ **완주**
> 실습 환경: Windows 11 · **CPU 전용** · Python 3.12 · transformers 5.16.1

> ⚠️ **후속 정정** — 이 실습 당시 "GPU 없음"으로 판단했으나, 실제로는 **GTX 1650(4GB)이 있는데 CPU 전용 torch(`2.13.0+cpu`)가 설치돼 있어 못 쓰고 있었다.** [패턴 2 실습](ch02-pattern02-grammar-lab.md)에서 CUDA 빌드로 전환했고, 문장 생성이 5분 16초 → 수십 초로 줄었다. **아래 CPU 속도 측정치는 그 전환 이전 값이다.**

## 진행 상태 — 4개 중 1개

| 노트북 | 주제 | 상태 |
|---|---|---|
| `0_logits_masking.ipynb` | 브랜드 금칙어 마스킹 | ✅ **완주 — 이 문서** |
| `1_sequence_selection.ipynb` | 두운(alliteration) | ⏳ 예정 |
| `2_sequence_regeneration.ipynb` | logprobs로 삼행시 | ⏳ 예정 |
| `3_autocomplete.ipynb` | 문구 자동완성 | ⏳ 예정 |

---

## 1. 겪은 문제 6가지

CPU 전용 윈도우 환경에서 책 예제를 돌리며 실제로 막혔던 지점들. 순서대로 적는다.

### ① `AssertionError` — 토큰이 아니라 안내 문구가 들어 있었다

```
AssertionError: Please sign up for access to the specific Llama model...
```

**결정적 단서는 이게 `KeyError`가 아니라 `AssertionError`라는 점.**

| 만약 | 나왔을 에러 |
|---|---|
| `HF_TOKEN`이 아예 없다 | `KeyError` |
| 있는데 값이 "hf"로 시작 안 함 | **`AssertionError`** ← 실제 상황 |

`keys.env`가 책 저장소의 템플릿 그대로였다.

```
HF_TOKEN=#See https://huggingface.co/settings/tokens
```

`.env`에서 `#`이 주석이 되려면 **앞에 공백이 필요**한데, `=` 뒤에 바로 붙어 있어 값의 일부로 읽혔다.

> 📌 `load_dotenv()`는 기본이 `override=False`. 이미 `os.environ`에 값이 있으면 **파일을 고쳐도 덮어쓰지 않는다.** 반드시 커널 재시작 또는 `override=True`.

> 🔒 `keys.env`가 `.gitignore`에 없고 git이 추적 중이다. 토큰을 채우기 전에 `git rm --cached` 할 것.

### ② 커널이 `.venv`가 아니었다

VS Code가 `cpython-3.12.13-windows-x86_64-none`을 선택하고 있었다. **uv가 받아둔 원본 파이썬**이지 프로젝트 `.venv`가 아니다. transformers가 없다.

| | 경로 | transformers |
|---|---|---|
| 🔴 잘못 잡힌 것 | `AppData\Roaming\uv\python\cpython-3.12.13-...` | 없음 |
| 🟢 써야 할 것 | `프로젝트\.venv\Scripts\python.exe` | 5.16.1 |

- 커널 목록에 **`.venv`라는 이름으로 안 보인다.** `pyvenv.cfg`의 `prompt` 값(= 프로젝트 폴더명)으로 표시됨
- **버전(3.12.13)이 같아도 다른 환경.** 경로를 보고 판단할 것
- 확인법: `import sys; print(sys.executable)` → 경로에 `\.venv\Scripts\`가 있어야 함

### ③ 코드 셀이 접혀 있었다

VS Code 노트북에서 마크다운 제목 아래 셀들이 `2 cells hidden` 처럼 접혀 있어 셀을 못 찾았다. 제목 왼쪽 `>` 화살표로 펼쳐야 한다.

### ④ `UnicodeDecodeError` — 한국어 윈도우의 cp949

```
UnicodeDecodeError: 'cp949' codec can't decode byte 0xe2 in position 148
```

`banned_phrases.txt`의 **`Alzheimer’s`** — 곧은 따옴표(`'`)가 아닌 **곡선 따옴표(`’`, U+2019)**. 파일은 UTF-8인데 한국어 윈도우의 파이썬은 `open()` 기본 인코딩을 cp949로 잡는다.

```python
# 책 원본 (맥/리눅스에서는 동작)
with open("banned_phrases.txt") as ifp:

# 수정
with open("banned_phrases.txt", encoding="utf-8") as ifp:
```

> 📌 한국어 윈도우에서 해외 예제를 돌릴 때 **가장 자주 만나는 에러**. `open()`에는 항상 `encoding="utf-8"`을 붙이는 습관을 들일 것.

### ⑤ 노트북 상태 함정 — 이미 줄어든 목록을 또 줄였다

속도를 위해 단어 목록을 줄이는 셀을 넣었는데, 그 셀을 두 번 실행하니 이렇게 됐다.

```python
banned_phrases = set(sorted(banned_phrases)[:120])
#                            ↑ 이미 50개로 줄어든 상태
#                            50개에서 120개는 못 뽑음 → 그대로 50개
```

**셀을 고쳐도 그 셀이 쓰는 변수가 이미 오염돼 있으면 결과가 이상해진다.** ①의 `HF_TOKEN` 문제와 정확히 같은 종류.

> 📌 노트북은 "코드"가 아니라 **"메모리 상태"**로 돌아간다.
> 셀 하나를 고쳤으면 **그 위에서 값을 만드는 셀들도 같이 다시 실행**할 것. 꼬였으면 Restart 후 Run All.

### ⑥ `set` 순서가 실행마다 달라진다

```python
banned_phrases = set(list(banned_phrases)[:50])   # ❌ 매번 다른 50개
```

파이썬은 실행할 때마다 문자열 해시를 다르게 잡는다(해시 랜덤화). 3번 돌려본 결과:

```
실행1: best mover, eco-friendly, endorsed by amazon, parkinson, ...
실행2: detoxifying, has shipped, lasting quality, matchless, ...
실행3: dementia, foremost, leading seller, mildew, ...
```

**커널을 재시작할 때마다 채점 기준이 바뀐다** → 마스킹 전후를 비교할 수 없다.

```python
banned_phrases = set(sorted(banned_phrases)[:120])   # ✅ 항상 동일
```

> 📌 **측정이 흔들리면 개선을 주장할 수 없다.** 6장 심판형 LLM(패턴 17)에서 온도를 0으로 두는 것과 같은 이유.

---

## 2. 모델 교체 — Llama → Qwen

Llama-3.2-3B는 ① 메타의 **게이트 모델**이라 접근 승인이 필요하고 ② 6.5GB이며 ③ CPU에서 너무 느렸다. Qwen으로 교체.

### 후보 비교

| 모델 | 크기 | 성격 | 판단 |
|---|---|---|---|
| **Qwen3.5-0.8B** | 1.75GB | 멀티모달(`image-text-to-text`) | ✅ **채택** |
| Qwen2.5-1.5B-Instruct | 3.1GB | 순수 텍스트 | 무난한 대안 |
| Qwen3-0.6B / 1.7B | 1.4/3.4GB | 사고 모드 기본 켜짐 | ⚠️ `<think>` 덤프 |

`Qwen/Qwen3.5-0.8B` 확인 결과:

| 항목 | 결과 |
|---|---|
| 게이트 | **없음** — HF 토큰 불필요 |
| 라이선스 | Apache-2.0 |
| 아키텍처 | `Qwen3_5ForConditionalGeneration` (멀티모달) |
| `text-generation` 매핑 | ✅ `qwen3_5` → `Qwen3_5ForCausalLM` |
| **사고(thinking) 모드** | ✅ **기본 꺼짐** — 템플릿이 빈 `<think></think>`를 미리 채움 |
| 토크나이저 | `Qwen2Tokenizer` (Llama와 다름) |

### ⚠️ CPU에서 dtype 선택은 직관과 반대

| dtype | 속도 |
|---|---|
| **float32** | **1.8 tok/s** ✅ |
| bfloat16 | 0.4 tok/s ❌ **4.5배 느림** |

가중치가 bf16으로 저장돼 있어도, **CPU에는 bf16 전용 연산기가 없어** 매번 float32로 변환한다. 그 비용이 메모리 절약분을 잡아먹는다. **GPU였다면 정반대.**

`torch.set_num_threads()`도 챙길 것 — 기본값이 4인데 이 PC는 8코어였다.

---

## 3. 실측 시간 (CPU 8코어, float32)

| 설정 | 시간 |
|---|---|
| 모델 로딩 | 14~20초 |
| 원본 프롬프트(6줄) · 80토큰 · 마스킹 없음 | **581초** (9분 41초) |
| 짧은 프롬프트(1줄) · 60토큰 · 마스킹 없음 | **316초** (5분 16초) |
| 짧은 프롬프트 · 40토큰 · **마스킹 + 4빔** | **149초** (2분 29초) |

> 📌 **토큰 수보다 프롬프트 길이가 시간을 더 좌우한다.** 매 토큰마다 앞 문맥 전체를 다시 훑기 때문. 시스템 프롬프트를 6줄 → 1줄로 줄인 것이 가장 효과가 컸다.

원본 설정(`max_new_tokens=512`, `num_beams=10`)은 CPU에서 수십 분 이상. 실습용 권장값:

| 파라미터 | 원본 | 권장 |
|---|---|---|
| 시스템 프롬프트 | 6줄 | **1줄** |
| `max_new_tokens` | 512 | **40~60** |
| `num_beams` | 10 | **4** (1은 불가 — 고를 후보가 없어짐) |
| 금지어 목록 | 412개 | **120개** |

---

## 4. 결과 — 패턴은 작동했다

**프롬프트는 한 글자도 바꾸지 않았다.** 마스킹 여부만 다르다.

| | 점수 | 걸린 희망어 | 걸린 금지어 |
|---|---|---|---|
| **v1** (마스킹 없음) | **−1** | 없음 | `complete` |
| **v2** (마스킹 적용) | **+2** | `calorie`, `protein shake` | **없음** |

**v1**
> "...delivers 25g of **complete** protein per scoop..."
> → `complete`가 아마존 제한 키워드에 걸림(과장 광고류)

**v2**
> "...our premium, zero-**calorie** **protein shake** designed to fuel your workouts..."
> → 금지어 0개, SEO 키워드 2개를 자연스럽게 포함

빔 4개가 각자 문장을 써 내려가는 동안, **매 토큰마다** 4개를 채점해 최고점이 아닌 빔을 `-10000`으로 짓눌렀다. 죽은 빔은 살아나지 못하고 살아남은 빔만 이어진다.

---

---

## 5. 정직하게 볼 한계 (실습에서 드러난 것)

### ① 한 번의 결과일 뿐이다

v2는 `do_sample=True, temperature=0.8`이라 **돌릴 때마다 다른 문장**이 나온다. 이번엔 +2였지만 다음엔 0일 수도 있다. 제대로 검증하려면 여러 번 돌려 평균을 봐야 한다.

1장의 그 이야기 — **"같은 입력에 다른 답."**

### ② 평가 기준을 우리가 느슨하게 만들었다

금지어를 412개 → 120개로 줄였다. **전체 목록으로 채점하면 v2에도 `quality`가 걸린다**(`high-quality`). 점수는 +2가 아니라 +1.

여전히 −1보다 낫지만, **기준을 느슨하게 하면 결과가 좋아 보인다**는 걸 우리 손으로 확인한 셈. 실무에서 자주 나오는 함정이다.

### 느리다

`__call__`이 매 토큰마다 **전체 시퀀스를 디코딩하고 금지어 목록 전부와 대조**한다. 빔 4개면 그게 4배. 40토큰 생성에 디코딩 160회 + 문자열 검색 수만 회.

> 기법 자체의 구조적 한계(토큰 조각 문제 · 상용 API 제약)는 [개념 정리](ch02-pattern01-logit-masking.md#3-이-패턴의-구조적-한계) 참고.

---

## 6. 재현용 — 최종 셀 코드

<details>
<summary><b>펼치기</b></summary>

**① `# CHANGE this to the Llama model...` 셀 (통째 교체)**

```python
MODEL_ID = "Qwen/Qwen3.5-0.8B"

# Qwen은 게이트 모델이 아니라 HF 토큰이 필요 없습니다
import os, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.set_num_threads(os.cpu_count())
print("모델:", MODEL_ID, "| 스레드:", torch.get_num_threads())
```

**② `# From https://channelkey.com/...` 셀**

```python
with open("banned_phrases.txt", encoding="utf-8") as ifp:   # encoding 추가
    banned_phrases = [line.strip().lower() for line in ifp.readlines()]
banned_phrases[100:105]
```

**③ `# Based on https://marketkeep.com/...` 셀**

```python
with open("desired_phrases.txt", encoding="utf-8") as ifp:  # encoding 추가
    desired_phrases = [line.strip().lower() for line in ifp.readlines()]
desired_phrases[:5]
```

**④ `# makes unique` 셀**

```python
banned_phrases = set(banned_phrases)
desired_phrases = set(desired_phrases)

# 실행마다 동일하게 뽑히도록 정렬 후 자릅니다
banned_phrases  = set(sorted(banned_phrases)[:120])
desired_phrases = set(sorted(desired_phrases))
print("금지어", len(banned_phrases), "/ 희망어", len(desired_phrases))
```

**⑤ `from transformers import pipeline` 셀 (통째 교체)**

```python
from transformers import pipeline
import torch

pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    use_fast=True,
    device_map="cpu",
    model_kwargs={"dtype": torch.float32},   # bfloat16은 CPU에서 4.5배 느림
)
print("로드 완료:", pipe.model.__class__.__name__)
```

**⑥ `def generate_product_description` 셀 (통째 교체)**

```python
import time

def generate_product_description(item: str) -> str:
    system_prompt = "You are a marketer for nutrition supplements. Write 3 sentences. No preamble."
    user_prompt = f"Write a product description for a {item}."
    input_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    results = pipe(input_message, max_new_tokens=60)
    return results[0]['generated_text'][-1]['content'].strip()

t0 = time.time()
prod = generate_product_description("protein drink")
print(f"[{time.time()-t0:.0f}초 걸림]\n")
print(prod)
```

**⑦ `def generate_product_description_v2` 셀 (통째 교체)**

```python
def generate_product_description_v2(item: str) -> str:
    system_prompt = "You are a marketer for nutrition supplements. Write 3 sentences. No preamble."
    user_prompt = f"Write a product description for a {item}."
    input_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    brand_processor = BrandLogitsProcessor(pipe.tokenizer, desired_phrases, banned_phrases)
    results = pipe(input_message,
                   max_new_tokens=40,
                   do_sample=True,
                   temperature=0.8,
                   num_beams=4,
                   use_cache=True,
                   logits_processor=[brand_processor])
    return results[0]['generated_text'][-1]['content'].strip()

import time
t0 = time.time()
prod_v2 = generate_product_description_v2("protein drink")   # prod 아님 — v1을 보존
print(f"[{time.time()-t0:.0f}초 걸림]\n")
print(prod_v2)
```

**⑧ 마지막 비교 셀 (통째 교체)**

```python
print("=== v1 (마스킹 없음) ===")
evaluate_verbose(prod, desired_phrases, banned_phrases)
print()
print("=== v2 (마스킹 적용) ===")
evaluate_verbose(prod_v2, desired_phrases, banned_phrases)
```

</details>

### 무시해도 되는 경고들

| 경고 | 뜻 |
|---|---|
| `generation_config ... is deprecated` | transformers 5.x의 사전 안내 |
| `Both max_new_tokens(=40) and max_length(=20)` | `max_new_tokens`가 이김. 정상 |
| `clean_up_tokenization_spaces for BPE` | Qwen 토크나이저에 안 맞는 옵션 안내 |

`hf_logging.set_verbosity_error()` 로 끌 수 있다.

---

---

## 이 실습이 남긴 것

| 배운 것 | 내용 |
|---|---|
| CPU의 현실 | 문장 하나에 **5분 16초**. bf16이 fp32보다 **4.5배 느리다** |
| 노트북의 함정 | 코드가 아니라 **메모리 상태**로 돌아간다 |
| 윈도우의 함정 | `open()`에 **`encoding="utf-8"`**을 항상 |
| 평가의 함정 | 기준을 느슨하게 하면 **결과가 좋아 보인다** |
| 결과 | 마스킹 전 **−1** → 후 **+2** (프롬프트는 그대로) |
