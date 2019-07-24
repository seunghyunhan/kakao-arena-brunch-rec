# kakao-arena-brunch-rec

## Requirements
- python 3.6
- numpy
- gensim
- tqdm

## 파일 구조
- 데이터 중 아래 데이터 만을 사용합니다
   - users.json
   - metadata.json
   - predict.tar
   - read.tar
- 데이터 다운로드 : https://arena.kakao.com/c/2/data

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


## 실행 방법
1. 아래와 같이 실행합니다.

	~~~
	$ python inference.py
	~~~


2. 기본적으로 3개의 user 인풋으로부터 각각의 결과 파일을 만듭니다. **최종 제출 파일은 "./res/predict/test.recommend.txt"** 입니다.
   1. 자체 테스트용
      - user_path: ./tmp/dev.users
      - output_path: ./tmp/dev.users.recommend
   2. dev 제출용
      - user_path: ./res/predict/dev.users
      - output_path: ./res/predict/dev.recommend.txt
   3. ** **test 제출용**
      - user_path: ./res/predict/test.users
      - **output_path: ./res/predict/test.recommend.txt**
