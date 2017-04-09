# Naive-Bayes
classifier

1. 배경
Naive-Bayes는 확률이론인 Bayes Therem을기반으로 하는 분류자이다.
가지고 있는 Training data set에서 임의의 tuple을 X라 하고,
Class lable을 C1,C2, ... ,Cm 이라 하자.
이러한 경우, P(H|X) X라는 data가 주어졌을때 해당 data의 class lable을 묻는 문제이다. 
계산 과정을 단순화하기 위해서 Naive Bayes는 각 data의 feature가 서로 independent하다고 가정한다.
이에대한 솔루션으로 Naive Bayes classifier는 다음과 같은 방법을 제시한다.

2. 이론
Dataset을 이메일메시지에따른 스팸여부라고 해보자.
즉, Data의 feature X1,X2,X3,...,Xn은 이메일에 쓰여진 각 단어를 의미하고
Class lable은 스팸인지 아닌지, 스팸 : 1, 햄 : 0이라고 하자. (C1: 1, C2:0)

Classifier에서 예상하는 class lable을 H(hypothesis)라 정의하면
P(H|X) = P(X|H)*P(H) / P(X) 이다.
Bayes classifier는 dataset이 가진 모든 class lable(C1,C2,C3,...,Cm)에 대해
P(X|Ci)*P(Ci) / P(X) 값을 최대로 갖게 하는 class lable로 H를 정의하고 있다.
즉, test-data,X가 주어졌을때 가장 가능성이 높은 class lable로 분류하는 것이다.

3. 구현
P(X)는 모든 class lables에 대해 상수이므로 P(X|Ci) * P(Ci)를 최대로하는 class lable을 구하면 된다.
P(H|X)를 구해낼때 사전확률(Prior Probability)로써 P(Ci),P(X|Ci)가 요구된다.
- P(Ci)는 |Dci| / |D| 로써 전체 데이터베이스 크기 중에 해당 class lable을 가진 튜플의 수로 구해낼수 있다.
- P(X|Ci)
X를 임의의 tuple이라 하면 X가 갖는 feature는 x1,x2,x3,...,xn이 존재한다.
Naive Bayes에서는 각 tuple이 서로 independent하다고 가정하였으므로(독립) 다음과 같이 P(X|Ci)를 구한다.
P(X|Ci) = P(x1 = X.x1 | Ci) * P(x2 = X.x2 | Ci) * ... * P(xn = X.xn | Ci)

결과적으로 위로부터 구한값으로 P(X|Ci)*P(Ci) 최댓값을 만족하는 class label을 구해내어
test-data,X에 대한 class lable을 예측한다.

4. 문제점1 : P(xi = X.xi | Ci) == 0 인 경우
위 수식에 따르면 P(X|Ci)는 각 feature들이 가진 확률의 "곱셈"이므로 어떤 임의의 확률값이 0인 경우 나머지 확률 term이 높은 확률값을 갖더라도 최종값으로 0을 갖는 문제점이 존재한다.
위 문제에 대한 직접적인 예시는 다음과 같다. 스팸여부를 나타내는 class lable, S : 스팸 이라 하자
"이마트 세일 정보" 라는 이메일에 속한 단어 이마트,세일,정보에 대해서 P(xi = 이마트|S) 에 대한 확률과 P(Xi = 세일|S)는 높은 확률값을 지녔지만 training data set에서 스팸메일중 "정보"라는 단어를 가진 데이터가 없었다고 하면,
P(Xi= 정보|S) = 0 이 되어 전체적인 P(X|Ci) = 0 이 되어버리는 잘못된 결과를 반환한다.

5. 해결책1 : Smoothing
따라서, 위 문제에 대하 해결책으로 가짜 빈도수(pseudocount),k 를 사용한다.
위 예시를 다시 사용하면 스팸메일에 '정보' 단어를 지닌 데이터를 k개 추가하고 스팸메일에 '정보' 단어를 지니지 않은 데이터를 k개 추가함으로써 P(xi = 정보|S)의 수식을 다음과 같이 수정하는 것이다.
P(xi = 정보|S) = (k + 스팸메일중 '정보'단어를 가진 데이터수)  / (2k + 전체 스팸메일수)
이를통해 각 feature에 대한 확률값이 0이 되는일을 방지 할 수 있다.
smoothing을 통한 오차값은 데이터베이스의 크기가 커질수록 점차 줄어든다.

6. 문제점2 : underflow
P(X|Ci)는 각 feature가 가진 확률값의 곱이므로 결과값이 0에 수렴하게 되어 컴퓨터가 0에 가까운 floating point를 잘 처리해 주지 못해 결과값이 0이 구해지는 경우가 있다. 결과적으로 underflow가 발생하게 되면 classifier가 정상적인 예측을 하지 못하게 된다.

7. 해결책2 : log scale
X = A * B이면
log(X) = log(A) + log(B)이고 대소관계는 변하지 않으므로 프로그램 구현과정중 P(X|Ci)에 대한 처리를 로그 스케일로 하여 underflow 문제를 방지한다.

8. 주요 소스코드
<pre><code>
def classify(dataSet,lables,testVec,k=0.1):
    m = dataSet.shape[0]
    unique_lable = np.unique(lables)
    lable_num = unique_lable.shape[0]
    
    prob_per_classes = []
    Idx2Lable = {}
    index = 0
    for lable in unique_lable:
        Idx2Lable[index] = lable
        bool_arr = lables == lable
        selected_sentences = dataSet[bool_arr]
        classNumber = selected_sentences.shape[0]
        prob_Ci = selected_sentences.shape[0] / m
        total_cond_prob = 0 # log scale
        for attributeIdx in range(dataSet.shape[1]):
            value = testVec[attributeIdx]
            matched = selected_sentences[:,attributeIdx] # return column vector
            selector = matched == value
            matched = matched[selector]
            count = matched.shape[0]
            cond_prob = (k + count) / (2*k + classNumber)
            total_cond_prob += -np.log(cond_prob) # log scale
        total_cond_prob *= prob_Ci
        prob_per_classes.append(total_cond_prob)
        index+=1
    idx = np.argmax(prob_per_classes)
    idx = np.argmin(prob_per_classes)
    print(prob_per_classes)
    prediction = Idx2Lable[idx]
    return prediction
</code></pre>

### Bayes Network
Naive Bayes모델은 비교적 높은 accuracy를 보장하나 치명적인문제점으로 각 feature들이 independent하다는 것을 가정이 있다. 실제 데이터들은 각 feature간 서로 dependency 관계를 가진경우가 많으므로 위 모델이 이러한 데이터특징에 대해 적합하지 않은 모델이 될 수 있다. 이에대한 해결책으로 위 Naive Bayes 모델을 확장하여 Acyclic Directed Graph(DAG)와 Conditional Probability Table을 갖는 Bayes Network 모델이 있다.


