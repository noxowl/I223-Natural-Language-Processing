# 2022 I223 NLP Final exam Summary

## Language Resource(언어자원)
Knowledge or data used for NLP
* Dictionary
* Theasaurus
* Corpus
* Word Embedding

### Dictionary
* 단어에 관한 다양한 정보를 기재한 지식 데이터베이스
    - 형태소분석, 구문분석, 의미분석에 필요한 정보
    - 강의에서의 예: word dictionary, case frame dictionary
* Morphological information(형태소정보)
    - Pronunciation(발음), POS(품사), inflection(어형변화)(EN)
    - Pronunciation(발음), POS(품사), Connectivity(L/R), inflection(어형변화)(JA), Canonical form
    - Notational Variant(表記のゆれ, 일본어의 경우)
        - canonical form(대표표기)을 병기한다
* Syntactic information(구문정보)
    - Surface case frame (JA)
    - Subcategorizaton frame (EN)
	    - SUBCAT(abbreviation)
* Semantic information(의미정보)
    - Case frame dict.
    - Semantic primitive(意味素)
* Bilingual dict.(대역사전, parallel corpus)
* 사전의 예
    - EDR Japanses word dict.
    - IPAL dict.

### Thesaurus
Database compiling hypernym-hyponym relations between words(단어간의 상위-하위관계를 정의하는 데이터베이스)
* Semantic Class
    - 시소러스의 내부 노드를 나타내는 단어의 집합. 자연언어 또는 기호로 나타낸다.
* Example of semantic relation
    - Hypernym, Hyponym (IS-A relation, 상위-하위관계)
	<br>: (living thing)-(animal), (fruit)-(apple)
    - Synonym(동의관계)
	<br>: (car)-(automobile), (movie)-(cinema)
    - Antonym(반의관계)
	<br>: (big)-(small), (hot)-(cold) : 형용사 부사 등 방향성을 가진 단어
* Tree(계층구조)로 정의되는 경우가 많으나, 그래프구조로 나타낼수도 있음.
* 시소러스의 종류
    - Classification Thesaurus 분류시소러스: 단어는 시소러스의 잎에만 존재
    - Hypernym/Hyponym Thesaurus 상위하위시소러스: 단어는 시소러스의 내부 노드에도 존재
* 시소러스의 상위 노드는 해석시점에 따라 달라질 수 있다.
### Corpus
collection of sentences = 예문집
* Raw corpus(plain text)
* Annotated corpus
	- POS tagged corpus
	- Syntactically annotated corpus
	- Sense tagged corpus
	- Parallel corpus
* 코퍼스의 활용
    * 파라미터 추정
    * Example-based NLP
    * 코퍼스에서 지식획득
    * NLP 시스템의 평가

#### Parameter Estmiation
NLP에서의 해의 우선순위(또는 확률모델)의 파라미터를 추정함
* 코스트최소법
* 은닉마르코프모델의 확률
* 확률문맥자유문법의 규칙의 확률

#### Evaluation of NLP Systems
* 형태소분석 시스템이나 구문분석시스템의 정확도를 평가
    - 코퍼스에 붙어있는 정보를 제거해 해석
    - 시스템의 해석결과와 코퍼스의 정보(정답)이 얼마나 일치하고 있는가를 판단
    - 시스템의 개량결과를 바로 알 수 있음
    - 같은 코퍼스로 평가하면 다른 시스템에도 공평한 평가
* 자주 사용되는 코퍼스
    - Penn Treebank
    - 京都コーパス

### Word Embedding
分散表現, 단어의 의미를 벡터화한 것.
* Nueral Network(NN)
    - 입력층, 은닉층, 출력층
    - 유닛
        - 입력 $\text{in}_i = \sum_j w_{ji}a_j$
        - 출력 $a_i = g(\text{in}_i)$
        - 활성화함수 $g$
            - Step, Sigmoid, Softmax...
    - 학습
        - Unit $j$와$i$간의 weight $w_{ji}$의 반복학습
        - 오차역전파학습(Back propagation)
            - 출력과 정답의 차(오차)를 최소화할 수 있도록 Weight를 수정
            - 오차를 출력층에서 입력층쪽으로 전달
* Skip-gram(Neural Network method)
    * 문맥에 출현하는 단어의 유사도를 학습(유사 벡터를 정의)
    * 유사 벡터를 가상에 차원에 모으는 것으로 원본보다 차원이 압축되므로 입력단어의 총 수와 다를 수 있다.
    * Vector로 학습하므로 Annotated corpus가 필요 없다.
    * A : B :: C : D (A,B의 관계는 C, D의 관계와 같다)
        - ex. Athens : Greece :: Oslo : Norway (수도-나라)
        - ex. apparent : apparently :: rapid : rapidly (형용사-부사)
    * A, B, C 가 주어졌을 때 D를 추정한다(벡터계산에 의한 추정)
        - $\vec{X} = \vec{B} - \vec{A} + \vec{C}$ 라면 $\vec{X}$와 가장 유사도가 높은 벡터를 가진 단어를 검출
        - $\vec{Queen} = \vec{King} - \vec{Man} + \vec{Woman}$

## Probabilistic langauge model
* Probability distribution of sequences of characters/words
    * $P(\text{Sue}, \text{swallowed}, \text{the}, \text{large}, \text{green}, \text{pill})$
* 용도
    - Spell checking
    - Charactor recognition
    - Speech recognition
    - Machine translation  
    - sequence나 words가 얼마나 자연스러운지 평가하는데 사용된다
    - 작용
        - 인식 결과의 여러 후보 중 올바른 값을 선택한다.
        - 언어학적 정보를 사용해 인식 정확도를 높일 수 있음.

### n-gram model
$\begin{matrix}
P(w_1\cdots w_m) &=& P(w_1)\times P(w_2|w_1)\times P(w_3|w_1w_2)\\
&=& \prod_{i=1}^m P(w_i|w_1w_2\cdots w_{i-1})\\
&\simeq& \prod_{i=1}^m P(w_i|w_{i-n+1}\cdots w_{i-1})
\end{matrix}$

* Example sentence: "two tall men play golf"
* Uni-gram($n=1$)
    * $\prod_{i=1}^m P(w_i)$
    * $P(\text{two})P(\text{tall})P(\text{men})P(\text{play})P(\text{golf})$
* Bi-gram($n=2$)
    * $\prod_{i=1}^m P(w_i|w_{i-1})$
    * $P(\text{two}|\theta)P(\text{tall}|\text{two})P(\text{men}|\text{tall})P(\text{play}|\text{men})P(\text{golf}|\text{play})$
* Tri-gram($n=3$)
    * $\prod_{i=1}^m P(w_i|w_{i-2}w_{i-1})$
    * $P(\text{two}|\theta_1\ \theta_2)P(\text{tall}|\theta_2\ \text{two})P(\text{men}|\text{two}\ \text{tall})P(\text{play}|\text{tall}\ \text{men})P(\text{golf}|\text{men}\ \text{play})$

### Estimating n-gram model
* Maximum likelihood estimation(MLE)
* $O(w)$: frequency of $w$ in corpus
* $\begin{matrix}
P(\text{white}|\text{black and})
= \frac{O(\text{black and white})}{O(\text{black and})}
\end{matrix}$

### Smoothing / Discounting (평탄화)
MLE에 의한 Data sparseness problem(데이터 부족 문제)의 대처 방법
* 이럴 때 심화된다
    - 훈련코퍼스의 사이즈가 작을 때
    - n-gram의 n이 클 때
* Data sparseness 문제에 대처하는 방법
    - 이미 알고 있는 event의 확률을 줄인다
    - 아직 모르는 event의 확률에 낮은 값을 할당한다
- 확률분포를 부드럽게 한다

#### Additive method
* Simple한것이 특징
* Laplace's law / adding one
    - 모든 event의 frequencies에 1을 더함
* Lidstone's law
    - 모든 envet의 frequenceis에 $\lambda$를 더함
* Jeffreys-Perks law
    - Lidstone's law when $\lambda$=0.5
    - Expected likelihood estimation (ELE)
* 모든 미지의 event에 같은 확률을 더하는 것은 반드시 적절하지 않음
    - 예) $P(bagpipe|the)$는 $P(the|the)$보다 확률이 높다고 생각할 수 있지만, 가산법에서는 동일한 확률이 더해짐
    - 예) $P(glass|the)$는 $P(goblet|the)$ 보다 확률이 높다고 생각할 수 있지만, 서로 같은 확률이 주어짐

#### Deleted interpolation(삭제보간법)
* Heldout interpolation에서는 코퍼스의 반만 n-gram모델의 추정에 사용된다
* 교차검정에 의해 n-gram model의 추정에 사용되는 데이터양을 늘린다
* 코퍼스를 잘게 쪼개서 반복해서 검증한다

#### Good-Turing Estimation
* Adjusting frequencies of events

#### Backoff smoothing
- Good-Turing estimation의 개량

#### Katz's backoff smoothing
* Problem on Good-Turing and backoff
- r^* is not relable at all when r is large
* Katz's approach

## Information Retrieval(IR, 정보검색)
정보원에서 유저가 가진 문제(정보요구)를 해결할 수 있는 정보를 찾아내는 것

좁은 의미로는 텍스트검색. 문서집합 안에서 유저의 쿼리에 적합한 문서를 찾아내는 것.

$\text{Relevant documents} \gets \text{Matching} =
\begin{cases}
\text{Instention of Searching}\to \text{Query} \\
\text{Document collection}\to \text{Indexing}\to \text{Collection of index terms}
\end{cases}$

* Query
    - By index terms directly (logical forms)
    - By natural language ("how can I ~")
* Indexing
    - Indexing should be done automatically (텍스트검색의 대상문서수가 너무 많기 때문)
    - 형태소분석등의 처리가 필요
    - Unit of index terms
        - Word単語 (e.g. cake, recipe, ingredient)
        - Phrase句 (e.g. recipe for cake, ingredient of cake)
* Stop word
* Matching

### Matching model
* **Inverted Index** (전치인덱스)
    - bit 단위 작업으로 빠르게 검색 (Vector bit)
    - cannot rank the documents
* **Vector Space Model** (VSM)
    - Query와 Document set 안의 모든 document의 유사성을 계산
    - 문서와 Query를 벡터로 표현: 문서벡터 = $\vec{D_i}$, Query벡터 = $\vec{Q}$
    - 벡터간의 유사도를 계산(최대의 유사도를 가진 문서 $\vec{D_i}$ 를 추출)
    * $\vec{D_i} =
    \begin{pmatrix} W_1^i \\ \vdots \\ W_j^i \\ \vdots \\ W_n^i \end{pmatrix}
    $
    - $W_j^i$는 Index term의 Weght

그러므로 둘을 조합해서...
1. inverted index로 검색키워드를 포함한 문서를 추출
2. 추출한 문서에 대해 vector space model로 순위를 결정


### Weighting Index Terms
* Simple indexing
- 문서에 존재하면 1, otherwise 0 (weighting for a query vector $\vec{Q}$)
* **TF/IDF**
    - TF(Term Frequency)
	    - $tf_j^i$: 문서 i에 대한 Index term j 의 빈도
	    - 같은 문서에 몇번이고 등장하는 단어일수록, 검색에 유력한 후보가 됨
    - IDF(Inverse Document Frequency)
	- $idf_j = \log\frac{N}{df_j}$
	- $df_j$ = 문서빈도(index term $j$를 포함한 문서수)
	- $N$: 문서의 총 수
	- 다양한 문서에 나타나는 단어는 검색에 유력한 후보가 되지 않음

### Similarity between 2 Vectors
* Similarity: $sim(\vec{D_i}, \vec{Q})$
- 유사도가 큰 상위 n개의 문서를 추출
* 유사도의 예
- 벡터의 내적
- 코사인유사도 (벡터간의 사이)
    - (설명에서는 2차원벡터인데 일반적으로는 다차원벡터임)
    - 두 벡터 사이의 세타에 대한 코사인을 구함(코사인함수: $\theta$가 $0$에 가까워질수록 커지고 $\theta$가 커질수록 작아짐)
    - 그러므로 두 벡터 사이의 $\theta$가 0에 가깝다면 두 벡터 유사도는 크다!

### IR의 평가(Vector Space Model)
* Precision 과 Recall
* 둘 사이의 trade-off
    - 시스템이 많은 문서를 추출한다면... precision은 작아지고, recall은 커진다
- Precision (적합률, 정밀도)
    - $\frac{\text{시스템이 찾은 적합문서수}}{\text{시스템이 찾은 문서수}}$
    - Precision이 중시될 때: 유저에게 적합문서만 제공하고싶을 때(ex. 웹 검색 엔진)
- Recall (재현률)
    - $\frac{\text{시스템이 찾은 적합문서수}}{\text{문서집합의 진짜 적합문서수}}$
    - Recall이 중시될 때: 검색누락(漏れ)을 적게 하고 싶을 때(ex. 특허문서의 검색)
- F-measure
    - Precision과 Recall 모두 평가할 때
    - $F = \frac{2PR}{P+R}$ ($P$=precision, $R$=recall)

## Inprovement in Text Retrieval
* Relevance feedback
* Query expansion

### Relevance Feedback(in SVM)
* 한번의 검색으로 좋은 결과를 얻는 일은 드물다.
    - 유저와 인터랙티브 검색
* Procedure
    - 시스템이 텍스트 검색을 행한다(n개의 문서를 유저에게 제공함)
    - 유저는 각각의 문서가 적합문서인지 여부를 판정한다
    - Query Vector $\vec{Q}$를 수정한다.
* Effectiveness of relevance feedback(관련피드백의 효과)
    - 적합문서Vector와 닮은 문서가 새로이 검색된다
    - 비적합문서Vector와 닮은 문서는 검색되지 않는다
    - precision, recall의 향상이 기대 가능
* Pseudo Relevance feedback
    - 인간에 의한 적합문서의 판정은 하지 않는다
    - 검색결과의 상위 문서를 적합문서로 치고 관련피드백을 행한다
    - 완전한 자동처리

### Query Expansion
* Query에 관련 단어를 자동으로 추가하는 기술
    - 이표현, 동의어, 상의어, 하의어
    - 단어사전, 시소러스를 이용
* **단어사전(word dictionary)의 예**
<pre>
WORD  | SYNONYM
grow  | farm, raise
store | keep, save
carry | convey
</pre>
* 단어사전을 이용해 <code>grain AND store</code>를 Query expansion 한다면...
    * "store"의 synonym(동,유의어)인 "keep" 과 "save"가 추가 가능
    * <code>grain AND (store OR keep OR save)</code>
* **시소러스의 예**
<pre>
   (plant)
   /
  (grass)
  /
 (grain)
 /
(wheat, corn, rice)
</pre>
* 시소러스를 이용해 <code>grain AND store</code>를 Query expansion 한다면...
    * "wheat", "corn", "rice"뿐만 아니라 hypernym(상의어)인 "grass"까지 추가 가능
    * <code>(grain OR grass OR wheat OR corn OR rice) AND store</code>
* 둘을 합하면...
    * <code>(grain OR grass OR wheat OR corn OR rice) AND (store OR keep OR save)</code>


## Text Processing
### Information Extraction (IE)
* 텍스트로부터 알고 싶은 정보를 직접 추출하는 것
* (차이)Information Retrieval(정보검색)
    - 알고 싶은 정보가 써있는 문서를 추출하는 것
* Method using a frame(프레임형 정보추출)
    - 추출해야 하는 정보를 미리 프레임으로 지정
    - 패턴매칭을 이용해 $\text{<maker>}$, $\text{<product name>}$ 등에서 추출한다
    - 패턴은 수동, 코퍼스에서 자동 학습
    - 텍스트 이외로 '오늘' 등의 글자에서 날짜를 특정도 가능
* Relation extraction(관계추출)
    - 정보추출의 한 종류
    - 텍트스에서 특정 의미적 관계가 성립되는 집합을 추출
    - 예: is-a관계(wheat,crop)(Picasso,painter)
    - 예: reaction관계(magnesium,oxygen)(hydrazine,water)
    - Pattern matching
        - 인스턴스와 패턴의 신뢰도를 계산한 후 신뢰도가 높은 인스턴스와 패턴을 추출한다.

### Question Answering(QA)
질문문의 해답을 문서집합에서 검색, 추출하여 답하는 시스템
* 정보검색과의 차이
    - IR: 해답을 포함한 문서를 출력
    - QA: 해답 그 자체를 출력
* 정보추출과의 차이
    - IE: 제한된 정보만을 추출한다고 가정(프레임), 답이 되는 문서는 미리 준비되어있음
    - QA: 추출되는 정보에 제한 없음, 답이 되는 문서를 찾아야 함
1. Question Analysis: 질문 유형을 결정하고 질문문에서 키워드 추출
2. Text Retrieval: 문서집합에서 해답을 포함하는 문서를 추출
3. Answer Extraction: 문서에서 해답을 추출
    - Named Entity Extraction(고유명사추출)
    - 텍스트 안에서 고유명사를 추출하고 지명, 인명, 조직명, 상품명 등의 분류를 함.
    - 고유명사사전은 사용되지 않음(쓸 수 없음)
    - 주변 단어를 실마리로 추정(인명이면 Mr.가 등장, 지명의 주면에는 hold가 등장)
    - 실마리 단어는 코퍼스에서 학습

### Text Categorization; Classfication
Text Categorization은 document를 토픽이나 테마에 맞게 미리 정해진 Category로 분류하는 작업이다.
* 대량의 텍스트를 정리하는데 사용(전형적인 분류문제).
* 비지도학습이 주류
    - Naive-Bayse
    - Dicision Tree
    - K-nearest Neighbor
    - SVM
* SVM
    - 이진분류문제
    - 마진 최대화
* 문서의 표현형식
    - Feature를 Vector화
    - Stop word를 제외
    - Feature의 가중치
        - Feature가 문서에 출현하면 1 otherwise 0
        - 출현빈도, Relative 빈도
        - TF/IDF
    - Feature Selection
        - Feature 중에서도 유효한 Feature만을 자동선택
        - 출현빈도로 자동선택
        - Pointwise Mutual Information으로 선택

## Machine Translation(MT)
* Interlingua method
* Transfer method
* EBMT(Example Based Machine Translation)
* SMT(Statistical Machine Translation)
* (Nueral machine translation)

### Interlingua Method
* Interlingua: sentence 의미의 표현형식. 모든 언어에 대해 공통
* 처리의 흐름
    - $\text{src} \to \text{analysis} \to \text{interlingua} \to \text{generation} \to \text{dst}$ 
* 다언어번역 대응을 하기 좋다: 언어와 interlingua의 상호변환만 되면 되므로
* 그러나 interlingua의 설계는 매우 어렵다.

### Transfer Method
* 처리의 흐름
    - $\text{src} \to \text{analysis} \to \text{result(analysis)} \to \text{transfer} \to \text{result(analysis)} \to \text{generation} \to \text{dst}$ 
* 

### EBMT(Example Based Machine Translation, 예제기반 기계번역)
* 번역사례(Corpus)+번역패턴 = 번역메모리가 필요하다.
* Parallel corpus는 주로 번역할 대상 문장과 유사한 문장을 검색하는데 사용된다.
* 자연스러운 번역문(의역)이 만들어지기 쉽다(하지만 비슷한 번역예가 필요)
* 번역메모리의 양이 늘어나면
    - 번역의 정확도가 높아진다
    - 처리시간이 증대한다
* 흐름
    - 번역메모리를 준비
    - 입력문과 번역사례와의 유사도를 계산해, 가장 비슷한 사례를 검색(유사도는 시소러스등으로 계산)
    - 번역패턴에 따라 입력문을 번역

### SMT(Statistical Machine Translation, 통계적 기계번역)
확률모델에 기반해 최선의 번역문을 선택 ($P(S|T)P(T)$를 최대화하는 $T$를 선택하는 method)
* (패러랠코퍼스)확률 $P(S|T)$는 번역된 dest가 source의 의미를 얼마나 유지하고 있는가 평가함
* 통계적기계번역에서는 번역된 문장이 얼마나 자연스러운지 까지는 고려하지 않음  
* 확률 $P(S|T)$의 추정에는 Parallel corpus가 필요함 ($P(T)$에는 필요 없음)
* 언어모델 $P(T)$ = 문장의 생성확률, n-gram모델을 사용함
* 하지만... 모든 $P(S|T)P(T)$를 구하는 것은 불가능
    * Decoder
        * 최대확률 T를 효율적으로 구하는 프로그램(검색공간을 축소)
        * A*, Beam search

#### Word alignment
* Parallel Corpus에 대한 단어의 대응관계를 결정
* SMT의 번역모델 $P(S|T)$의 추정에 필요

#### IBM Model 1
* Parallel corpus 안의 Word의 Alignment를 추정하는 통계적 알고리즘(EM 알고리즘)
* 모든 언어 쌍에 적용된다
* 비지도학습이다
* alignment는 반복학습으로 추정한다
* 1대다 또는 다대다 Alignment는 고려하지 않는다.
* P(J|T)의 근사가 부적절할 수 있다
* P(A|T,J)의 근사가 부적절할 수 있다
* "verde"와 "green"이 스페인어-영어 대역관계인 sentence에서 자주 나타난다면 확률 $P(\text{verde}|\text{green})$은 높게 추정된다
* "glove"와 "グラブ"가 영어-일본어 Parallel corpus의 번역 쌍으로 자주 나타날 경우, 확률 $P(\text{glove}|\text{グラブ})$는 높다 (스페인어-영어와 같은 말)
* 흐름
    1. Soruce 문장의 길이를 선택
    2. Alignment A를 선택
    3. 각각의 Dest 언어의 단어에 대해, 그것에 대응하는 Src의 단어를 선택
* EM 알고리즘의 흐름
    1. 파라미터 초기화
    2. E-Step: 대역관계인 word를 count
    3. M-Step: 파라미터 재추정
    4. 계속될때까지 2와 3을 반복
* 그외 Model 2~5, HMM based alignment도 있다.

## Dialog system
* 인간과 대화하는 소프트웨어. Machine Interface의 한 종류.
* 음성입력과 키보드 입력 등을 받을 수 있다.
* Task-oriented dialog system
    - 유저는 명확한 대화의 목적이 있다.
    - 유저는 원하는 목적을 달성하면 대화를 종료한다.
    - 도메인(topic)이 제한됨: 호텔 예약, 비행기 예약, 네비게이션 등
* Non-task-oriented dialog system
    - 유저는 명확한 대화의 목적이 없다.
    - 유저는 자기가 대화하고 싶을 때까지 대화한다.
    - 도메인 제한이 없으므로 free conversation system, chat system 이라고도 함.
### Task-oriented dialog system(TD)
1. Speech recognition
2. Understanding of User's utterance
    - Morphological(형태소) analysis
    - Syntactic(구문) analysis
    - Semantic(의미) analysis
3. Planning
    - 의도를 Goal로써, 그것을 달성하기 위한 Plan또는 Plan Tree를 구축
    - Human's action: Desire→Intention→Action
    - Planning을 위한 지식: 인과관계의 규칙의 집합
    - 도메인이 한정될 필요가 있다. (추론의 범위가 너무 넓어지기 때문)
4. Intention recognition(의도이해)
    - 말하는 이의 의도를 이해해 적절한 응답을 제공
    - 예: "XX교수의 전화번호를 알려주세요" 라는 질문을 받았을 때, 시스템은 "전화번호를 알려달라" 라는 질문 의도를 이해하고, XX교수가 출장으로 부재중이라면 전화번호를 알려주는 것이 아닌 "XX교수는 현재 출장중으로 연락이 불가능합니다" 라는 대답을 하는 것.
    - Plan recognition
        - 발화에서 상대의 의도, 행동계획을 추정
        - Planning의 역조작
        - 지식베이스를 기반으로 추정
5. Dialog act estimation(대화행위추정)
    - Rule-based approach
        - 대화행위를 결정하는 룰을 수동으로 준비
        - 발화가 "안녕하세요"를 포함 -> 인사
        - 발화가 "인가요?" 를 포함 -> 질문
    - Machine learning approach
        - 훈련데이터(올바른 대화행위가 붙은 발화의 집합)에서 대화행위를 추정하는 모델을 학습
        - ML의 Features
            - Content word, 문말표현, 키워드, 이전 발화의 대화행위
    - 무엇을 말하는 것으로 무엇을 하려 하는가? 무엇을 하고 있는가 라는 관점에서의 발화의 분류
    - 대화행위의 예
        - 자신의 취향이나 감정을 밝힘
        - 정보제공
        - 질문
        - 확인: 상대가 말한 것을 확인함
        - 재촉: 흠, 그래서?
        - 인사: 대화의 시작과 끝
        - Filler: 발화의 사이를 메우는 의미없는 발화(아 그러니까, 잠깐만)
6. Discourse structure analysis(담화구조인식)
    - 화제는 추이한다
    - 같은 Topic을 가진 발화의 집합 -> 담화구조
    - Domain 지식, Cue word, Cue phrase, Prosody(운율)가 검출될 수 있음.

#### Planning vs. Plan recognition
* 둘의 유사점
    - Plan Tree를 생성한칙
    - 인과관계의 규칙 $C(\text{ause}) \to E(\text{ffect})$의 집합을 사용한다.
* Planning
    - 대화의 내용을 결정하는게 목적
    - 시스템이 Plan Tree를 생성
    - Top-Down
    - 인과관계의 규칙은 Effect에서 Cause로
* Plan recognition
    - 유저의 의도를 추정하는게 목적
    - 유저가 Plan Tree를 생성
    - Bottom-Up
    - 인과관계의 규칙은 Cause에서 Effect로

### Non-task-oriented dialog system(NTD, FCS)
* Free Cnversation System; FCS 이기도 함.
* 사람과 채팅하는 시스템은 FCS의 한 예이다.
* 대화 그 자체가 목적이며, 태스크지향 시스템을 개선하는데 사용될 수 있다.
* FCS를 개발하는데 필요한 지식정보는 때때로 Web에서 자동으로 획득된다.
* 사용자와 FCS간의 대화가 얼마나 오래 지속되는지를 평가하여 FCS를 평가하는 것이 적절하다.
* Planning을 쓰지 않는다(수행할 태스크, 목표가 없으므로).

1. Rule-based Method
    - 발화에서 응답을 생성하는 룰을 수동으로 작성
    - 자연스러운 응답문을 생성할 수 있다
    - 여러 토픽을 망라한 룰을 준비하는 것은 어렵다
2. Extraction-based Method
    - 발화와 응답의 페어를 웹 등에서 추출
    - 입력발화와 비슷한 발화를 데이터베이스에서 검색해 돌려준다.
    - 광범위한 토픽을 커버할 수 있다
    - 응답이 부자연스러워지기 쉽다
3. Understanding-and-Generation Method
    - 발화를 이해하고, 그것에 대한 응답을 생성
    - 광범위한 토픽을 커버할 수 있다
    - 시스템에서의(체계적) 개선이 가능
* 종합
    - 응답의 질: Rule > U&G > Extraction
    - 개발코스트: Rule > U&G > Extraction

## Deep learning on NLP
DL은 고도로 추상화된 Feature 표현을 학습하는 수법이다.
* NL에 기반한 방법이 주류이다.
* Neural Network는 기본적으로 Data가 아니라 Signal만 Node로 전파한다.
* 입력: 입력데이터를 표현하는 Vector
* 출력: 각 출력후보 $y$의 스코어를 나열한 Vector
* 은닉층: 고도로 추상화된 Feature
* 각 층에 대한 신호전달
    - $h_{i+1} = f(W_ih_i+b_i)$
    - $W_i$는 Weight Vector
    - $b_i$는 Bias
    - $f()$는 Activation function
* NLP에서의 Input/Ouput
    - Input: 1-hot Vector또는 Word Embedding
    - Output
        - 출력층의 노드=분류클래스(분류문제인 경우)
        - 1-hot Vector(단어를 출력할 때, 실제로는 스코어분포가 출력되어 가장 높은 스코어의 노드를 선택)
    - sentence를 입력할 때, 단어를 시계열로써 다룸

### CNN
* 화상처리 수법으로써 제안
* Convolution과 Pooling을 반복함으로써 추상적 Feature를 얻을 수 있음
* Convolution Layer
    - 작은 Filter를 위치를 빗겨가며 차원을 압축함
* Pooling Layer(Sub-sampling layer)
    - 작은 범위 내의 최대치만을 선택하여 차원을 압축함
* Fully Connected Layer
    - 추상화된 Feature를 입력, 분류카테고리를 출력으로하는 NN

문장 스타일의 분류 문제(Classficaion problem of sentence style)에 적용할 수 있다.

### RNN
* 시계열을 입력으로 사용하는 모델 (재귀형, recurrent)
    - 하나 앞 시점의 숨김층을 입력으로 추가
    - 문맥의 정보가 학습가능
* Classficaion problem of sentence style에 적용할 수 있다.

#### Bi-RNN
* 순방향과 역방향의 RNN 조합
* $y_i = ( \overleftarrow{y_i} ; \overrightarrow{y_i} )$
* 앞의 문맥만이 아니라 뒤의 문맥도 학습 가능

### LSTM
* RNN의 숨김층을 LSTM유닛으로 바꾼 모델
* 인간이 가진 장기기억, 단기기억을 흉내
* (RNN에서는 쓸 수 없는)장거리의 의존성을 학습가능
* LSTM유닛
    1. Memory Cell: 정보를 장기적으로 보관
    2. Input gate: 메모리셀에 추가할 정보를 결정
    3. Forgot gate: 메모리셀에서 삭제할 정보를 결정
    4. Output gate: LSTM유닛에서 숨김층의 값을 결정
* CNN(Encoder)+LSTM(Decoder)로 이미지 캡션 자동생성이 가능하다.

LSTM, Bi-RNN, RNN 모두 **시계열 입력**을 받으므로, sequence를 입력으로 받는다.

### Sequence-to-Sequence model (Seq2Seq)
* sequence를 another sequence로 변환
- Machine Translation: 영어 문장을 입력해 일본어 문장을 생성
- Dialog system: User's utterance -> System's response
- Summarization: document -> summary

### Encoder-Decoder
- 입력 sequence를 추상적 벡터표현으로 변환(encoder)
- 벡터표현에서 출력sequence를 생성(decoder)
- encoder/decoder를 RNN, LSTM 등으로 학습
- decoder는 하나 앞 출력도 입력으로 추가함 (하나 앞의 단어에서 다음 단어를 예상)
* Classficaion problem of sentence style에 적용할 수 없다.
* Attention
    - LSTM cannot memorize all past information<br>(e.g. 기계번역의 경우, 장문일수록 번역정밀도가 떨어짐)
    - 과거의 sequence 중에서, 핀포인트로 참조하고 싶은 부분이 있다면<br>(e.g. 기계번역의 경우, 목표언어의 단어를 생성할 때 출발언어의 단어를 참조하고 싶다)
    - 참조하고 싶은 부분을 "직접적directly"으로 이용
* Procedure of Attention
    - encoder의 각시점에는 숨김층$\overline{h_s}$의 값을 모두 기록
    - 현시점에서의 decoder의 숨김층$h_t$의 내적을 계산 ($\overline{h_s}$와 $h_t$가 얼마나 비슷한가??)
    - alignment weight vector $a_t(s)$를 계산
    - $a_t(s)$를 weight로써 encoder의 숨김층 값의 선형합을 context vector $c_t$ 로 사용
    - decoder의 숨김층의 출력$h_t$에 $c_t$의 정보를 더해 갱신함(갱신후의 숨김층의 출력 = $\widetilde{h_t}$)
- Google 번역, 자동요약 등에 사용

### Transformer
* Encoder-Decoder 모델의 일종
* Self-attention
	- Encoder(또는 decoder)에 대한 다른 단어와의 관련성을 학습
	(cf. 통상의 attention에서는 encoder와 decoder 사이의 단어의 관련성을 학습)
	- 특히 장거리의 의존성을 학습
- RNN과는 달리 하나 앞의 스텝의 출력을 입력으로 사용하지 않음(병렬계산에 의한 고속화가 쉬움)
- Applied for Machine translation at first(그 후 다른 태스크에도 응용됨)

*Encoder-Decoder* 모델의 "Attention" algorithm과 *Transformer*의 "Self-attention"은 유사한 이름이지만 둘은 전혀 다른 개념을 의미한다.
* *Encoder-Decoder* 모델의 "Attention"
    - 과거의 sequence에서 참조할 부분을 직접적으로 이용
* *Transformer*의 "Self-attention"
    - encoder와 decoder 사이의 단어의 관련성(의존성)을 학습

### BERT(Bidirectional Encoder Representation from Transformer)
* Transformer의 Encoder 부분만을 겹쳐 사용하므로 Encoder/Decoder 모델의 일종이 아니다.
* pre-training: 양방향의 transformer에 의한 sentence 벡터표현의 전후학습(대량의 텍스트를 이용해 일반적 표현을 학습)
* fine-tuning: sentence vector 표현에서 출력벡터를 예상하는 모델의 학습 & sentence vector 표현의 재학습
	- 비교적 소량의 annotated corpus를 사용
	- 학습시간이 짧음
* pre-training은 fine-tuning보다 더 대량의 corpus를 필요로 한다.

### GPT
* Pre-training of language model(확률언어모델)을 사용
    - Transformer의 Decoder만을 사용
* Fine-tuning for down-streaming task
    - 입력 sequence의 가장 마지막 symbol의 출력=입력의 추상표현
    - 이것을 입력으로 한 선형출력층을 겹쳐 fine-tuning
    - (BERT와 비슷하지만 언어모델을 사용하고 다음 단어를 예상한다는 것이 다름)
* 0.12억 파라미터

#### GPT-2
- no fine tuning (zero-shot learning, one-shot learning)
- pre-trained model만으로 다양한 task를 해결
- 태스크에 응해 입력단어열을 주고, 언어모델의 예측단어열을 출력
	- 요약
        - original document TL;DR -> summary
	- Machine Translation
	    - En sentencel = Fr sentencel / En sentence2 = -> Fr sentece2
* 15억 파라미터
* 비지도학습의 수법으로써는 우수하지만, state-of-the-art(SOTA) mothod보다는 떨어짐(대부분 지도학습)

### GPT-3
* few-shot learning
	- Task description & 소량(10~100개)의 정답이 입력으로써 주어짐
* 1750억 파라미터 
* Larger corpus is used for pre-training
	- 5000억 단어(570GB 이상)의 웹 텍스트(GPT-2는 40GB)
* Performance
	- SOTA보다 떨어지나, 태스크에 따라서는 비슷한 정도까지 향상됨
	- Title/Subtitle에서 신문기사를 생성하는 태스크에 관해, 인간이 쓴 기사와의 차이를 찾는게 어려웠었다 (52% accuracy)