# kakao-arena-brunch-rec

## Requirements
- python 3.6
- numpy
- gensim
- tqdm

## 파일 구조
- 아래 데이터만 사용합니다.
   - users.json
   - metadata.json
   - predict.tar
   - read.tar
- 데이터 설명 및 다운로드 : https://arena.kakao.com/c/2/data

~~~
├── inference.py
├── util.py
├── config.py
├── res
│   ├── users.json
│   ├── metadata.json
│   ├── predict
│   └── read
└── tmp
~~~

## 모델
- 모든 과정에서 매우 심플한 모델을 구성하였습니다.
- 모델은 1.candidate 선정 / 2.ranking 순서로 이루어져 있습니다.
   1. Candidate 문서 선정 
      - a. [테스트 기간 몇 일 전부터 ~ 테스트 기간] 동안 작성된 새로운 문서들 중 **유저가 팔로우한 문서**
      - b. [테스트 기간 몇 일 전부터 ~ 테스트 기간] 동안 작성된 새로운 문서들 중 **유저가 과거에 많이 본 에디터의 문서**
      - c. 유저가 최근에 본 문서 히스토리와 유사한 **doc2vec top-k문서** (모든 기간의 문서)
      - a + b + c 를 모두 candidate 문서로 이용
   2. Ranking
      - **기본적으로 a + b + c 순서를 유지**하되 a, b, c 각각의 내용을 아래 우선순위에 따라 re-rank 함 
         - 우선순위 1. 유저가 과거에 본 에디터의 글 횟수 -> 많을수록 우선
         - 우선순위 2. 해당 날짜에 만들어진 문서가 클릭될 확률 순위 -> rank 작을수록 우선
         - 우선순위 3. 문서가 만들어진 최신순


## 실행 방법
1. 아래와 같이 실행합니다.

	~~~
	$ python inference.py
	~~~

2. 기본적으로 2 개의 user 인풋으로부터 각각의 결과 파일을 만듭니다. **최종 제출 파일은 "./res/predict/test.recommend.txt"** 입니다.
   1. dev 제출용
      - user_path: ./res/predict/dev.users
      - output_path: ./res/predict/dev.recommend.txt
   2. ** **test 제출용**
      - user_path: ./res/predict/test.users
      - **output_path: ./res/predict/test.recommend.txt**
