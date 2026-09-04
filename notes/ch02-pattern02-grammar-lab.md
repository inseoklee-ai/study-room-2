# 패턴 2 · 문법 — 실습 기록 (노트북 1/3)

> 📘 **책 본문 개념 정리는 → [ch02-pattern02-grammar.md](ch02-pattern02-grammar.md)**
> 이 문서는 예제 노트북을 **실제로 돌린 기록**이다.
>
> 예제: `examples/02_grammar/1_applying_grammar.ipynb` ✅ **완주**
> 실습 환경: Windows 11 · **GTX 1650 4GB (실습 도중 CPU→GPU 전환)** · Python 3.12 · transformers 5.16.1

## 진행 상태

| 노트북 | 책의 방법 | 상태 |
|---|---|---|
| `1_applying_grammar.ipynb` | 방법 1 — BNF 문법 + 로짓 프로세서 | ✅ **완주 — 이 문서** |
| `2a_json_mode.ipynb` | 방법 2 — JSON 모드 | ⏸ `OPENAI_API_KEY` 필요(유료) |
| `2b_structured_data.ipynb` | 방법 3 — 구조화 출력 | ⏸ `GEMINI_API_KEY` 필요 |

---

## 한 줄로

> **문법은 파싱 실패를 0으로 만든다. 그러나 내용의 정확성은 손대지 않는다.**

---

## 1. 개념 — 패턴 1의 직계 확장

[패턴 1 로짓 마스킹](ch02-pattern01-logit-masking.md)은 금지어 몇 개를 막았다. 문법은 같은 자리에서 훨씬 강하게 개입한다.

```
마지막 층 → 로짓 → [여기서 개입] → 소프트맥스 → 확률 → 표집
```

| | 로짓 마스킹 (패턴 1) | 문법 (패턴 2) |
|---|---|---|
| 막는 것 | 지정한 **몇 개 토큰** | **문법이 허용하지 않는 전부** |
| 방식 | 블랙리스트 | **화이트리스트** |
| 판단 기준 | 고정 목록 | **지금 문법 상태에서 올 수 있는가** (매 토큰 갱신) |

Qwen 어휘가 248,077개인데, 매 스텝 그중 **문법상 유효한 것만 남기고 나머지를 전부 마스킹**한다. 그래서 문법적으로 깨진 출력이 **나올 수가 없다.**

---

## 2. 실습 결과 — 세 가지 문법

### 실험 A. 산술 표현식 강제

```
root  ::= (expr "=" ws term "\n")+
expr  ::= term ([-+*/] term)*
term  ::= ident | num | "(" ws expr ")" ws
ident ::= [a-z] [a-z0-9_]* ws
num   ::= [0-9]+ ws
ws    ::= [ \t\n]*
```

질문: *"Bill has 3 apples and 2 oranges. Mae has 2 apples and 4 oranges. How many total apples?"*

**문법 없이** (35초)

> To find the total number of apples Bill and Mae have, we need to add the number of apples from each person.
> 1. **Bill's apples**: 3
> 2. **Mae's apples**: 2
> $$3 + 2 = 5$$

**문법 적용** (63초)

```
3 +2 = 5
5 +2 = 7
7 +2 = 9
...
```

✅ **영어 산문이 한 글자도 없다.** 프롬프트로 "수식만 답해"라고 부탁한 게 아니라, 영어 문장을 만들 토큰 자체를 못 고르게 했다.

⚠️ **그런데 멈추지 않는다.** `root ::= (...)+` 의 `+`가 "한 번 이상 반복"이라 문법이 계속 줄을 허용한다.

> 📌 **문법은 "형식"을 강제하지 "언제 멈출지"는 안 알려준다.**
> 멈추게 하려면 문법 자체에서 반복을 빼야 한다 (`(...)+` → `(...)`).

### 실험 B. 문법이 답을 표현할 수 없을 때

질문을 바꿨다: *"Do Bill and Mae have **more apples than oranges**?"* — 답하려면 `>` 비교가 필요한데, 문법에는 `+ - * /`와 `=`뿐이다.

**결과**

```
yes

*3 +2 = 5
2 +4 = 6
5 +6 = 11
11 +11 = 22
...
88 +88 = 17
```

#### `yes`는 문법 위반이 아니다 — 헐거운 문법이 통과시킨 것

```
ident ::= [a-z] [a-z0-9_]* ws     ← "소문자로 시작하는 아무 단어"
ws    ::= [ \t\n]*                ← 줄바꿈을 공백으로 흡수
```

파서가 실제로 읽은 것:

```
yes ⏎⏎ * 3 ⏎+ 2  =  5
└ident┘   └────expr────┘  └term┘
```

**`yes * 3 + 2 = 5`라는 하나의 수식.** 화면에는 세 줄로 보이지만 문법상 한 줄이다. 변수명을 허용하려고 넣은 `ident` 규칙이 `yes`·`no`·`maybe`를 전부 통과시킨다.

숫자도 질문과 무관한 배수 놀이(5+6=11, 11+11=22, 22+22=44…)를 한다.

> 📌 **두 가지 교훈**
> ① 문법이 정답을 표현할 수 없으면, 모델은 **"못 하겠다"고 말할 방법이 없다.** 거절이라는 선택지 자체가 막혀 있다.
> ② **헐거운 문법은 헐거운 대로 뚫린다.** 촘촘하지 않은 제약은 제약답지 못하다.

### 실험 C. 구조화 추출 — 실무형

```
record    ::= author separator title separator year
author    ::= [a-zA-Z ]* | unk
title     ::= [a-zA-Z ]* | unk
year      ::= [1-2][0-9][0-9][0-9] | unk
unk       ::= "NULL"
separator ::= "|"
```

실험 A의 문법과 결정적으로 다른 점:

| | 산술 문법 | 책 정보 문법 |
|---|---|---|
| 반복 | `+` 있음 → 끝없이 생성 | 없음 → **레코드 하나로 종료** |
| 연도 | 아무 숫자 | **`[1-2][0-9][0-9][0-9]`** |
| 정보 없을 때 | 표현 불가 | **`NULL`** 탈출구 |

**C-1. 정보가 다 있는 경우** — 성공

입력: *Love in the Time of Cholera ... by Gabriel García Márquez and published in 1985.*

```
'Gabriel Garcia Marquez | Love in the Time of Cholera |1985'
```

✅ 저자의 결과와 완전히 일치. 파싱 100% 성공.

관찰 두 가지:
- **악센트가 사라졌다** (`García Márquez` → `Garcia Marquez`). `[a-zA-Z ]*`가 영문자만 허용하므로 `í`,`á`를 쓸 방법이 없다. **한국어 데이터에 이 문법을 쓰면 저자명이 통째로 빈다.**
- **`Cholera |1985`** — 구분자 뒤에 공백이 없다. 문법이 `separator ::= "|"` 하나만 정의했기 때문. **명세에 적힌 그대로가 출력의 모양이다.**

**C-2. 🔴 정보가 문법에 안 맞는 경우** — 여기서 갈렸다

입력: 타밀 고전 『티루쿠랄』. 연도 정보가 *"300 BCE to 5th century CE"*, *"450 to 500 CE"* — **전부 3자리이거나 숫자가 아니다.**

| | 결과 |
|---|---|
| 저자(Phi-3 3.8B) | `'Valluvar \| The Tirukkural \|NULL'` ✅ |
| **우리(Qwen 0.8B)** | **`'Valluvar \| Tirukkural \|1000'`** ❌ |

**`1000`은 입력 문단 어디에도 없다. 모델이 지어냈다.**

문법은 두 갈래를 다 열어뒀다 — 네 자리 숫자 **또는** `NULL`. 모델은 앞쪽을 골랐고, **문법은 전혀 위반되지 않았다.**

> 📌 **탈출구는 필요조건이지 충분조건이 아니다.**
> 문법은 "NULL을 쓸 **수 있게**" 할 뿐, "NULL을 **쓰게**" 만들지 못한다.
> 같은 문법, 다른 모델, 다른 정직성.

---

## 3. outlines — 같은 일을 정규식으로

노트북 후반부는 `outlines` 라이브러리로 같은 추출을 한다. 문법 대신 **정규식**으로 명세한다.

```python
generator = outlines.generate.regex(
    model,
    r"([a-zA-Z ]+|NULL) \| ([a-zA-Z ]+|NULL) \| ([1-2][0-9][0-9][0-9]|NULL)",
    sampler=outlines.samplers.greedy(),
)
```

**결과**

```
transformers-cfg → 'Gabriel Garcia Marquez | Love in the Time of Cholera |1985'
outlines         → ' Gabriel | Love in the Time of Cholera | 1985'
```

| 차이 | 원인 |
|---|---|
| 구분자 양쪽 공백 | 정규식에 ` \| `로 **명시**됨 (문법은 `"|"`만) |
| 맨 앞 공백 | `[a-zA-Z ]+`가 공백도 허용 + 프롬프트 조립 코드의 들여쓰기 |
| 🔴 **저자가 `Gabriel`로 잘림** | `[a-zA-Z ]+`는 **한 글자 이상이면 언제든 끝낼 수 있다** |

**`'Gabriel'`은 형식상 완벽하고 내용상 틀렸다.** 정규식 검증을 100% 통과한다.

### 두 도구 비교 (실측)

| | transformers-cfg | outlines |
|---|---|---|
| 명세 방식 | 문법(GBNF) — `root ::= ...` | **정규식** |
| 준비 시간 | 문법 객체 **20초** | FSM 인덱스 **16초** |
| 생성 (짧은 출력) | 30~63초 | 59초 |
| 표현력 | 재귀 구조 가능(중첩 JSON) | 정규식 한계 내 |
| 배우기 | 문법 문법을 익혀야 함 | **정규식이면 됨** |
| 오늘 필요한 손질 | `byte_encoder` **패치** | **인자를 넘기면 안 됨** |

> 준비 시간(16~20초)은 어휘 248k에 대해 "각 토큰이 이 상태에서 허용되는가"를 미리 계산하는 비용이다. **함수 안에서 만들면 호출마다 반복**되므로 실무에서는 밖으로 빼야 한다.

---

## 4. 🔴 이 실습이 계속 가리킨 한 곳

| 실험 | 형식 | 내용 |
|---|---|---|
| B. 사과 비교 | ✅ 문법 통과 | ❌ `yes *3 +2 = 5` 헛소리 |
| C-2. 타밀 연도 | ✅ 문법 통과 | ❌ `1000` 지어냄 |
| outlines | ✅ 정규식 통과 | ⚠️ `Gabriel` 잘림 |

**세 번 다 검증은 통과하고 내용은 어긋났다.** 도구를 바꿔도 이 성질은 그대로였다.

> ### 패턴 2의 결론
> 문법·정규식은 **파싱 실패를 0으로** 만든다. 그러나 **내용의 정확성은 손대지 않는다.**
> **"구조화 출력을 썼으니 안전하다"가 가장 위험한 착각이다.**

실무에서 가장 무서운 조합이 이것이다 — **형식이 완벽해서 검증을 통과하는데 값이 거짓인 경우.** `'Valluvar | Tirukkural |1000'`은 파서를 통과하고, DB에 들어가고, 아무도 눈치채지 못한다.

그래서 형식 계층과 내용 계층은 별개다:

| 계층 | 담당 |
|---|---|
| 형식 | 패턴 1 로짓 마스킹 · **패턴 2 문법** |
| 내용 | 9장 **자체점검(31)** — 토큰 확률로 환각 탐지 · **가드레일(32)** — 출력 전 검사 |

### 우리 프로젝트와 연결

모니터링 에이전트에서 **"DOI·특허번호를 추론해 채우지 말 것"**을 규칙으로 박아둔 이유가 정확히 이것이다. **형식이 맞는 가짜 식별자는 빈칸보다 훨씬 해롭다.** 정규식 검증을 통과하고, 링크가 걸리고, 연구자가 클릭한 뒤에야 없는 문헌임이 드러난다.

「확인 불가」와 「존재하지 않음」을 구분하는 것도 같은 이야기 — **빈칸을 표현할 수단을 시스템에 마련해 두는 것**이 신뢰성의 출발점이다.

---

## 5. 환경 문제 — 오늘 막힌 것들

### ① `transformers-cfg`가 transformers 5.x에서 깨진다

```
AttributeError: GPT2Tokenizer has no attribute byte_encoder
```

`transformers-cfg 0.2.7`은 토크나이저의 내부 속성 `byte_encoder`/`byte_decoder`를 쓰는데, **transformers 5.x에서 제거**됐다. (`requirements.txt`가 `transformers==4.51.0`으로 핀돼 있는 이유)

**해결 — GPT-2의 표준 바이트 매핑을 직접 만들어 클래스에 붙인다**

```python
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
_BE = _bytes_to_unicode()
GPT2Tokenizer.byte_encoder = _BE
GPT2Tokenizer.byte_decoder = {v: k for k, v in _BE.items()}
```

인스턴스가 아니라 **클래스 속성**으로 붙이는 게 요령이다. `transformers-cfg` 내부가 `AutoTokenizer.from_pretrained("gpt2", use_fast=False)`로 토크나이저를 새로 만들기 때문에, 우리가 쥔 객체에 붙여봐야 소용없다.

### ② `outlines`에 인자를 넘기면 **세그폴트**

```python
# ❌ 프로세스가 통째로 죽는다 (예외가 아니라 Segmentation fault)
model = outlines.models.transformers(MODEL_ID, device="cuda",
                                     model_kwargs={"dtype": "float16"})

# ✅ 인자 없이
model = outlines.models.transformers(MODEL_ID)
```

`outlines 0.2.3`도 transformers 4.51 기준이라 인자 규약이 어긋난다. 네이티브 레벨에서 터져서 traceback도 안 남는다.

### ③ 📌 책 예제의 버그 — `return` 누락

```python
def convert_message_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"""..."""
    prompt += """
    <|im_start|>assistant
    """
    # ← return prompt 가 없다! None을 반환한다
```

`generator(convert_message_to_prompt(...), ...)` 에 **`None`이 들어간다.** 노트북에 이 셀의 저장된 출력이 없는 걸 보면 저자도 완주하지 못한 듯하다.

### ④ 메모리 부족 → GPU 전환

문법 트라이(0.91GB) + 모델(fp32 3.2GB)이 가용 RAM 3.0GB를 넘겨 **세그폴트**가 났다. 그 과정에서 발견한 것:

> **"GPU가 없다"와 "GPU를 못 쓰고 있다"는 다르다.**

| | |
|---|---|
| 실제 하드웨어 | GTX 1650 Max-Q **4GB** ✅ |
| 설치돼 있던 torch | `2.13.0+**cpu**` — **CPU 전용 빌드** |
| 결과 | GPU가 있는데 안 쓰고 있었음 |

`torch.cuda.is_available()`이 `False`라고 GPU가 없는 게 아니다. **하드웨어는 `nvidia-smi`로, 소프트웨어는 `torch.__version__`의 `+cpu`/`+cu126` 꼬리표로 따로 확인**해야 한다.

**전환 과정에서 막힌 것**

```
No module named pip
```

이 `.venv`는 **uv로 만들어져서 pip이 없다.** uv를 써야 한다.

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.13.0+cu126"
```

버전은 `2.13.0` 그대로 두고 꼬리표만 `+cpu` → `+cu126`으로 바꾸는 게 가장 안전하다. `torchvision`/`torchaudio`가 없어 버전 맞출 것도 없었다.

**전환 직후 첫 import가 5분 걸린다** — CUDA DLL 초기화 + 윈도우 디펜더가 새 DLL 수천 개를 스캔하기 때문. 이때 `ModuleNotFoundError: Could not import module 'pipeline'` 이라는 **엉뚱한 에러**가 뜬다. transformers의 `_LazyModule`이 진짜 원인을 감추는 것이니, **실패가 아니라 느린 것**이다. 두 번째부터는 정상 속도.

### ⑤ dtype 선택은 하드웨어마다 뒤집힌다

| 환경 | 최적 dtype | 이유 |
|---|---|---|
| CPU | **float32** | bf16은 전용 연산기가 없어 매번 변환 → **4.5배 느림** |
| GTX 1650 (Turing, compute 7.5) | **float16** | bf16 하드웨어 지원 없음 |
| Ampere 이상 | bfloat16 | 수치 안정성 유리 |

> `torch.cuda.is_bf16_supported()`가 `True`로 나와도 Turing에서는 **소프트웨어 에뮬레이션**이다. 지원과 빠름은 다르다.

---

## 6. 성능 (GTX 1650, fp16)

| 작업 | 시간 |
|---|---|
| 모델 로드 (GPU) | **9~16초** · VRAM **1.5GB** |
| 문법 객체 생성 (어휘 248k) | 20초 |
| 문법 제약 생성 (64토큰) | 63초 |
| 문법 없이 생성 (64토큰) | 35초 |
| outlines FSM 인덱스 | 16초 |
| outlines 생성 | 59초 |

> ⚠️ **문법을 쓰면 오히려 느려진다** (35초 → 63초). 매 토큰마다 "25만 개 중 무엇이 허용되는가"를 계산하기 때문. **후보를 줄이는 대가로 계산을 더 한다.** 형식이 반드시 맞아야 하는 곳에서만 값을 하는 기법이다.

참고로 CPU 시절(패턴 1 실습)에는 문장 하나에 **5분 16초**가 걸렸다. GPU 전환으로 상황이 완전히 달라졌다.

---

## 7. 재현용 — 최종 셀 코드

<details>
<summary><b>펼치기</b></summary>

**셀 2 (`MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"`) — 통째 교체**

```python
MODEL_ID = "Qwen/Qwen3.5-0.8B"      # Phi-3-mini(3.8B, 7.6GB)는 4GB VRAM에 무리

import os, torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
torch.set_num_threads(os.cpu_count())

# ── transformers 5.x 호환 패치 ─────────────────────────────
def _bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
_BE = _bytes_to_unicode()
GPT2Tokenizer.byte_encoder = _BE
GPT2Tokenizer.byte_decoder = {v: k for k, v in _BE.items()}
# ──────────────────────────────────────────────────────────

print("모델:", MODEL_ID, "| 호환 패치 적용")
```

**셀 3 (`from transformers import pipeline`) — 통째 교체**

```python
from transformers import pipeline
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor
import torch

print("CUDA 사용가능:", torch.cuda.is_available())

pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    device_map="cuda:0",
    model_kwargs={"dtype": torch.float16},   # Turing은 bf16 미지원 → fp16
)
print("로드 완료:", pipe.model.__class__.__name__, "|", pipe.model.device, "|", pipe.model.dtype)
```

**셀 4·7 — `max_new_tokens`만 조정** (256 → 64). 문법은 원본 그대로.

**셀 9 (outlines 모델) — 통째 교체**

```python
import outlines

# ⚠️ device/model_kwargs를 넘기면 세그폴트. 인자 없이 로드할 것
model = outlines.models.transformers(MODEL_ID)
print("outlines 모델 로드 완료")
```

**셀 10 — `return prompt` 추가**

```python
def convert_message_to_prompt(messages):
    prompt = ""
    for message in messages:
        prompt += f"""
<|im_start|>{message['role']}
{message['content']}
<|im_end|>
        """
    prompt += """
    <|im_start|>assistant
    """
    return prompt          # ← 책에 빠져 있던 줄
```

</details>

### 무시해도 되는 경고

| 경고 | 뜻 |
|---|---|
| `generation_config ... is deprecated` | transformers 5.x 사전 안내 |
| `Both max_new_tokens and max_length` | `max_new_tokens`가 이김. 정상 |
| `clean_up_tokenization_spaces for BPE` | Qwen 토크나이저에 안 맞는 옵션 안내 |
| `generation flags are not valid: ['temperature']` | greedy 샘플러에 온도가 딸려 들어감 |
| `TqdmWarning: IProgress not found` | 진행바 위젯 미설치 |

---

## 8. 언제 무엇을 쓰나

| 요구사항 | 도구 |
|---|---|
| "가급적 ~하게" | 프롬프트 / 퓨샷 |
| "절대 이 단어는 안 됨" | 로짓 마스킹 (패턴 1) |
| **"출력 구조가 반드시 이래야 함"** | **문법 / 정규식 (패턴 2)** |
| 중첩 JSON, 재귀 구조 | 문법(GBNF) 또는 pydantic 스키마 |
| 단순 형식 (구분자, 날짜, 라벨) | **정규식 (outlines)** |
| **값이 사실인지** | ❌ 문법으로 불가 → **9장 자체점검(31)·가드레일(32)** |

**판단 한 줄** — *"형식이 깨지면 시스템이 멈추는가?"* 예라면 문법. *"값이 틀리면 사람이 다치는가?"* 예라면 문법만으로는 부족하다.

---

## 이 실습이 남긴 것

| 배운 것 | 내용 |
|---|---|
| 문법의 정체 | 매 토큰 **화이트리스트** — 25만 개 중 유효한 것만 남김 |
| 문법이 못 하는 것 | 언제 멈출지, 값이 맞는지 |
| 헐거운 문법 | `ident`를 열어두면 `yes`가 통과한다 |
| 탈출구 | `NULL`은 **필요조건이지 충분조건이 아니다** |
| 명세 = 출력 모양 | 구분자 공백 하나까지 그대로 |
| 문자 집합 주의 | `[a-zA-Z ]`는 **악센트도 한글도 못 담는다** |
| 비용 | 문법을 쓰면 **약 2배 느려진다** |
| 환경 | 라이브러리 셋이 서로 다른 시기를 가리킨다 — 손질 없이는 안 돌아감 |
| 하드웨어 | **"GPU가 없다"와 "GPU를 못 쓴다"는 다르다** |
