# 패턴 1 · 로짓 마스킹 — 개념과 실습 기록

> 『에이전트 시대의 AI 시스템 설계』 2장 패턴 1
> 실습 환경: Windows 11 · **CPU 전용** · Python 3.12 · transformers 5.16.1
>
> ⚠️ **후속 정정** — 이 실습 당시 "GPU 없음"으로 판단했으나, 실제로는 **GTX 1650(4GB)이 있는데 CPU 전용 torch(`2.13.0+cpu`)가 설치돼 있어 못 쓰고 있었다.** [패턴 2 실습](ch02-pattern02-grammar-lab.md)에서 CUDA 빌드로 전환했고, 문장 생성이 5분 16초 → 수십 초로 줄었다. 아래 CPU 속도 측정치는 그 전환 이전 값이다.

## ⚠️ 진행 상태 — 4개 중 1개 완료

`examples/01_logits_masking/` 에는 노트북이 **4개** 있고, 이 문서는 그중 **첫 번째만** 다룬다.

| 노트북 | 주제 | 상태 |
|---|---|---|
| `0_logits_masking.ipynb` | 생성 목표에 안 맞는 후속을 마스킹 (브랜드 금칙어) | ✅ **완료 — 이 문서** |
| `1_sequence_selection.ipynb` | 원하는 두운(alliteration)에 맞춰 로짓 변경 | ⏳ 예정 |
| `2_sequence_regeneration.ipynb` | logprobs로 삼행시(acrostic) 보장 | ⏳ 예정 |
| `3_autocomplete.ipynb` | 로짓 기반 문구 자동완성 | ⏳ 예정 |

---

## 한 줄로

> **프롬프트가 "부탁"이라면, 로짓 마스킹은 "물리적 차단"이다.**

---

## 1. 개념

### 어디서 개입하는가

1장에서 본 생성 흐름에 그대로 얹힌다.

```
마지막 층 → 로짓(원시 점수) → [여기] → 소프트맥스 → 확률 → 표집
```

로짓과 소프트맥스 **사이**에 손을 넣는다. 핵심은 소프트맥스 공식이다.

```
P(token_i) = exp(logit_i) / Σ_j exp(logit_j)
```

`exp(-∞) = 0` 이므로, 금지 토큰의 로짓을 `-inf`로 만들면 **확률이 정확히 0**이 된다. 0.001%가 아니라 진짜 0이다.

1장에서 *"언어 모델은 어떤 후속 단어라도 확률이 0이 되지 않는 표집 전략을 쓴다"*고 했는데, 로짓 마스킹은 **그 원칙을 의도적으로 깨는 것**이다.

### 왜 프롬프트로는 안 되는가

| | 프롬프트 | 로짓 마스킹 |
|---|---|---|
| 성격 | **부탁** — "경쟁사 이름은 쓰지 마세요" | **차단** — 그 토큰을 뽑을 수 없음 |
| 보장 | 확률적. 대부분 지키지만 **가끔 뚫림** | **구조적으로 불가능** |
| 실패 시 | 왜 어겼는지 모름 | 실패라는 게 성립 안 함 |

규정 준수·법무가 걸린 출력에서 "대부분 지킴"은 쓸모없다. 그래서 이 패턴이 2장 첫 번째에 온다 — **가장 낮은 층위에서 가장 확실하게** 개입한다.

### 두 가지 방향, 두 가지 강도

| 방식 | 하는 일 | 쓰는 곳 |
|---|---|---|
| **블랙리스트** | 특정 토큰만 막음 | 금칙어, 비속어, 경쟁사명 |
| **화이트리스트** | **허용 목록 외 전부** 막음 | 형식 강제, 분류 라벨, 제한 어휘 |

| 강도 | 코드 | 효과 |
|---|---|---|
| **소프트 (기울이기)** | `logits[ids] -= 5` | 잘 안 나오게. 꼭 필요하면 **여전히 나올 수 있음** |
| **하드 (차단)** | `logits[ids] = -float('inf')` | 확률 **정확히 0**. 절대 안 나옴 |

화이트리스트를 문장 구조 전체로 확장한 것이 **패턴 2 「문법」**이다. JSON 생성 시 "지금 위치에 올 수 있는 토큰"만 남기면 깨진 JSON이 나올 수 없다. 패턴 1과 2는 형제.

### 사후 필터링과 뭐가 다른가

| | 사후 필터링 | 로짓 마스킹 |
|---|---|---|
| 언제 | 문장이 다 나온 뒤 | **매 토큰 생성 시점** |
| 걸리면 | 통째로 버리고 **재생성** | 애초에 안 나옴 |
| 결과물 | 잘라내면 문장이 깨짐 | **막힌 상태에서 자연스럽게 이어서** 씀 |

---

## 2. 책 예제 코드 해설

```python
class BrandLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, positives, negatives):
        self.tokenizer = tokenizer
        self.positives = positives
        self.negatives = negatives

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids, input_logits):
        output_logits = input_logits.clone()

        num_matches = [0] * len(input_ids)
        for idx, seq in enumerate(input_ids):
            decoded = self.tokenizer.decode(seq)
            num_matches[idx] = evaluate(decoded, self.positives, self.negatives)
        max_matches = np.max(num_matches)

        # 최고점이 아닌 빔을 짓눌러 탈락시킨다
        for idx in range(len(input_ids)):
            if num_matches[idx] != max_matches:
                output_logits[idx] = -10000

        return output_logits
```

### 인자 두 개의 정체

| 인자 | 모양(shape) | 내용 |
|---|---|---|
| `input_ids` | `(빔 개수, 지금까지_길이)` | **여태 만들어진 토큰들**. "직전에 뭘 썼나"를 보고 판단하라고 주는 것 |
| `input_logits` | `(빔 개수, 어휘_크기)` | **다음 토큰 하나**에 대한 전 어휘 점수 (Qwen은 약 15만 개) |

> ⚠️ `input_logits`는 문장 전체가 아니라 **바로 다음 한 토큰**의 점수판이다. 허깅페이스 공식 문서에서는 보통 `scores`라고 부른다.

### 알아둘 점

- **`__call__`은 매 토큰마다 호출된다.** 40토큰 생성이면 40번 실행
- `.clone()` — 원본 텐서를 제자리에서 고치면 빔 탐색 내부 점수 관리가 꼬인다. **안전한 관례**
- `-10000`을 쓰는 이유 — 주석대로 *"torch doesn't like it to be -np.inf"*. 실질적으로 −∞와 같은 효과
- `add_start_docstrings` / `LOGITS_PROCESSOR_INPUTS_DOCSTRING` — **순수 문서용.** 지워도 동작은 같다
- **이 구현은 토큰이 아니라 빔을 마스킹한다.** 후보 문장 여러 개를 굴리다가 점수 낮은 것을 죽이는 방식 → **`num_beams=1`이면 무의미**

### 붙여 쓰는 법

```python
results = pipe(input_message,
               num_beams=4,
               logits_processor=[brand_processor])
```

---

## 3. 실습 기록 — 겪은 문제 6가지

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

## 4. 모델 교체 — Llama → Qwen

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

## 5. 실측 시간 (CPU 8코어, float32)

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

## 6. 결과 — 패턴은 작동했다

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

## 7. 정직하게 볼 한계

### ① 한 번의 결과일 뿐이다

v2는 `do_sample=True, temperature=0.8`이라 **돌릴 때마다 다른 문장**이 나온다. 이번엔 +2였지만 다음엔 0일 수도 있다. 제대로 검증하려면 여러 번 돌려 평균을 봐야 한다.

1장의 그 이야기 — **"같은 입력에 다른 답."**

### ② 평가 기준을 우리가 느슨하게 만들었다

금지어를 412개 → 120개로 줄였다. **전체 목록으로 채점하면 v2에도 `quality`가 걸린다**(`high-quality`). 점수는 +2가 아니라 +1.

여전히 −1보다 낫지만, **기준을 느슨하게 하면 결과가 좋아 보인다**는 걸 우리 손으로 확인한 셈. 실무에서 자주 나오는 함정이다.

### ③ 토큰 조각 문제 — 의미는 못 막는다

모델을 바꾸면 토크나이저가 바뀌고, 같은 단어도 쪼개지는 방식이 달라진다.

- `"green"`과 `" green"`(앞 공백)은 **다른 토큰**. 문장 중간 단어는 거의 항상 공백 붙은 쪽
- 여러 토큰으로 쪼개지는 단어를 막으면 **"그 단어"가 아니라 "그 조각들"**을 막는 것
- `Apple`을 막으면 과일 사과도 못 쓴다 — **문맥 구분 불가**
- `바 보`처럼 띄어 쓰면 우회된다

> **로짓 마스킹은 정확한 토큰 차단에는 완벽하지만, 의미 차단은 못 한다.**

확인용 코드:

```python
tok = pipe.tokenizer
for w in ["protein", " protein", "miracle", " miracle"]:
    ids = tok.encode(w, add_special_tokens=False)
    print("%-12r 토큰수=%d  %s" % (w, len(ids), tok.convert_ids_to_tokens(ids)))
```

### ④ 느리다

`__call__`이 매 토큰마다 **전체 시퀀스를 디코딩하고 금지어 목록 전부와 대조**한다. 빔 4개면 그게 4배. 40토큰 생성에 디코딩 160회 + 문자열 검색 수만 회.

### ⑤ 대부분의 상용 API에서는 못 쓴다

로짓에 손대려면 `model.generate()`를 직접 돌려야 한다.

| | 가능 여부 |
|---|---|
| 로컬 / 가중치 공개 모델 | ✅ 자유 |
| OpenAI | △ `logit_bias` (토큰당 −100~100) |
| **앤트로픽** | ❌ 해당 파라미터 없음 |

> 1장 "모델 지형도"에서 *"가중치 공개 모델은 자체 최적화가 가능하다"*고 한 게 추상적인 말이 아니라 **바로 이런 걸 할 수 있다**는 뜻이었다.

---

## 8. 재현용 — 최종 셀 코드

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

## 9. 언제 무엇을 쓰나

| 요구사항의 성격 | 맞는 도구 |
|---|---|
| "가급적 ~하게" (선호) | 프롬프트 / 퓨샷 |
| **"절대 ~하면 안 됨"** (하드 제약) | **로짓 마스킹 (패턴 1)** |
| 출력 구조 전체가 정해짐 (JSON·스키마) | 문법 (패턴 2) |
| 톤·문체를 바꾸고 싶음 | 스타일 전이 (패턴 3) |
| 나가기 전 최종 검사 | 가드레일 (패턴 32) |

**판단 기준 한 줄** — *"이게 한 번이라도 뚫리면 사고인가?"* 예라면 프롬프트로 끝내면 안 된다.

---

## 10. 우리 프로젝트와 연결

[모니터링 에이전트](https://github.com/inseoklee-ai/hackathon1-monitoring-agent)는 LLM으로 문장을 생성하지 않으니 직접 쓸 일은 없다. 다만 발상이 겹친다.

> 화면 문구를 **Jinja2 템플릿 + 결정론적 치환**으로 만드는 건, 로짓 마스킹을 **극단까지 밀어붙인 형태**다.
> 토큰 몇 개를 막는 게 아니라 **모델에게 토큰 선택권을 아예 주지 않은** 것.

같은 축 위에 있다 — **「부탁 → 일부 차단 → 전면 차단」**. 정확성 요구가 극단이라 맨 오른쪽을 택했을 뿐이다. 나중에 다이제스트에 LLM을 조금 쓰게 되면, 이 축에서 **어느 지점까지 물러설지**가 설계 결정이 된다.

---

## 이 실습이 남긴 것

| 배운 것 | 내용 |
|---|---|
| 로짓 마스킹의 정체 | 프롬프트가 아니라 **생성 경로**를 바꾼다 |
| 빔이 필요한 이유 | 후보가 하나면 **고를 게 없다** |
| CPU의 현실 | 문장 하나에 **2~10분**. bf16이 fp32보다 느리다 |
| 노트북의 함정 | 코드가 아니라 **메모리 상태**로 돌아간다 |
| 윈도우의 함정 | `open()`에 **`encoding="utf-8"`**을 항상 |
| 평가의 함정 | 기준을 느슨하게 하면 **결과가 좋아 보인다** |
| 실무 제약 | **가중치 공개 모델에서만** 가능 |
