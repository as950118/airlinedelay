from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt

d_model = 512
num_layer = 6
num_heads = 8
d_ff = 2048

'''
인코딩
rnn은 데이터의 위치에 따라 순차적으로 입력받아 처리하기떄문에 각 데이터의 위치정보가 필요.
transformer는 데이터를 순차적으로 받는 방식이 아니므로 위치정보를 다른방식으로 알려줘야함.
그래서 각 데이터의 embedding vector 위치 정보를 더하여 모델의 입력으로 사용.
이를 포지셔널 인코딩이라고 함.
'''

def get_angels(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates
def positional_encoding(position, d_model):
    angle_rads = get_angels(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    #배열의 짝수 인덱스에는 sin 함수
    sines = tf.math.sin(angle_rads[:, 0::2])
    #배열의 홀수 인덱스에는 cos 함수
    cosines = tf.math.cos(angle_rads[:, 1::2])
    #pos_encoding = tf.concat([sines, cosines], axis = -1)
    pos_encoding = np.concatenate([sines, cosines], axis = -1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype = tf.float32)

#test
print("*****Encoding******")
pos_encoding = positional_encoding(50, 512)
print(pos_encoding.shape)
plt.pcolormesh(pos_encoding[0], cmap="RdBu")
plt.xlabel("Depth")
plt.xlim((0,512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
'''
인코더
transformer는 기본적으로 hyperparmeter인 num_layer와 같은 n개의 인코더를 가짐.
인코더는 2개의 sublayer로 나누어진 layer임.
'''
'''
인코더 멀티-헤드 어텐션
1)어텐션
어텐션 함수는 주어진 쿼리(q)에 대한 키(k)와의 유사도를 구함.
그리고 이 유사도를 가중치로 하여 키와 맵핑되어있는 각각의 값(v)을 구함.
그리고 각각의 값을 가중합하여 반환함.

2)q,k,v
q = t-1시점의 디코더 셀에서의 은닉상태
k = 모든 시점의 인코더 셀의 은닉상태
v = 모든 시점의 인코더 셀의 은닉상태
하지만 q는 계속 변화하면서 반복적으로 수행하므로
q = 모든 시점의 디코더 셀에서의 은닉상태
라고 볼수도있다

3)셀프 어텐션
그러나 셀프 어텐션에서는
q = 입력 데이터들의 모든 데이터 벡터들
k = 입력 데이터들의 모든 데이터 벡터들
v = 입력 데이터들의 모든 데이터 벡터들
셀프 어텐션은 데이터들 내에서 유사도들을 구함.

4)q, k,v 벡터
셀프 어텐션은 우선 각 데이터 벡터들로부터 q,k,v 벡터들을 얻음.
이때 초기입력인 d_model의 차원을 가지는 데이터 벡터들보다 더 작은 차원을 가짐.
q,k,v 벡터의 크기는 hyperparameter인 num_heads로 결정됨.

각 가중치 행렬은 d_model * (d_model/num_heads)의 크기를 가짐.
예를들어 d_model=512, num_heads=8라면 각 벡터에 3개의 서로 다른 가중치 행렬을 곱하고 64의 크기를 가지는 q,k,v 벡터를 얻어냄.

5)스케일드 닷-프로덕트 어텐션
qkv를 얻은후엔 기존 어텐션 매커니즘과 동일.
q는 모든 k에 대해서 어텐션 스코어를 구하여 어텐션 분포를 구함.
그리고 이를 활용해 모든 v를 가중합하여 어텐션 값 혹은 컨텍스트 벡터를 구함.
이것은 모든 q에 대해 반복함.
그리고 여기에 특정값으로 나누어주는것이 스케일드 닷-프로덕트 어텐션임.
socre(q,k)=q*k/root(n) 임.

이제 이 어텐션 스코어에 소프트맥스를 적용하여 어텐션 분포를 구함.
각 v와 가중합하여 어텐션 값을 구함.
이것을 데이터에 대한 어텐션 값 또는 컨텍스트 벡터라고 함.
하지만 굳이 모든 q에대해 따로 연산해야할 필요없음.
행렬 연산으로 일괄처리가 가능함.
'''
def scaled_dot_procudt_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(dk)

    #mask가 있을경우
    if mask is not None:
        logits += (mask * -1e9)

    #어텐션 분포
    attention_weights = tf.nn.softmax(logits, axis = -1)

    #어텐션 값
    output = tf.matmul(attention_weights, value)

    return output, attention_weights

#test
print("*****Attention******")
np.set_printoptions(suppress=True)
temp_k = tf.constant([[10,0,0],
                     [0,10,0],
                     [0,0,10],
                     [0,0,10]], dtype = tf.float32) #(4,3)
temp_v = tf.constant([[  1,0],
                     [  10,0],
                     [ 100,5],
                     [1000,6]], dtype = tf.float32) #(4,2)
temp_q = tf.constant([[0,10,0]], dtype = tf.float32) #(1,3)

temp_out, temp_attn = scaled_dot_procudt_attention(temp_q, temp_k, temp_v, None)
print("1. temp_attn:",temp_attn," ///// temp_out",temp_out)

temp_q = tf.constant([[0,0,10]], dtype=tf.float32)
temp_out, temp_attn = scaled_dot_procudt_attention(temp_q, temp_k, temp_v, None)
print("2. temp_attn:",temp_attn," ///// temp_out",temp_out)

temp_q = tf.constant([[0, 0, 10], [0, 10, 0], [10, 10, 0]], dtype=tf.float32)
temp_out, temp_attn = scaled_dot_procudt_attention(temp_q, temp_k, temp_v, None)
print("3. temp_attn:",temp_attn," ///// temp_out",temp_out)

'''
6)멀티헤드 어텐션
첫번째 sublayers
앞의 어텐션에서는 d_model의 차원을 가진 단어 벡터를 num_heads로 나눈 차원을 가지는 qkv벡터로 바꾸어 어텐션은 수행.
왜냐면 한번의 어텐션을 하는 것보다 여러번의 어텐션을 병렬로 사용하는것이 더 효과적이기때문.
그래서 d_model/num_heads의 차원을 가지는 qkv에 대해서 num_heads의 개수만큼 병렬 어텐션 수행
이때 가중치 행렬 Wq, Wk, Wv의 값은 어텐션 헤드마다 전부 다름.
병렬 어텐션을 모두 실행한 후 모든 어텐션 헤드를 concat함.
어텐션 행렬의 크기는 (seq_len, d_model)임.
어텐션 헤드를 모두 연결한 행렬은 또 다른 가중치 Wo를 곰함.
이렇게 나온 결과 행렬이 멀티헤드 어텐션의 최종 결과물.
'''
'''
멀티헤드어텐션은 크게 5가지 파트로 구성됨.
1.Wq,Wk,Wv에 해당하는 밀집층.
2.num_heads만큼 split.
3.스케일드 닷 프로덕트 어텐션
4.나눠진 헤드들을 concat.
5.Wo에 해당하는 밀집층.
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % self.num_heads == 0

        self. depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model) # Wq
        self.key_dense = tf.keras.layers.Dense(units=d_model) # Wk
        self.value_dense = tf.keras.layers.Dense(units=d_model) # Wv

        self.dense = tf.keras.layers.Dense(units=d_model) # Wo

    def split_heads(self, inputs, batch_size): #아래 call함수의 heads를 나누기 위해
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        #1. Wq, Wk, Wv에 해당하는 밀집층
        #(bathc_size, sqe_len, d_model)
        query = self.query_dense(query)
        key = self.query_dense(key)
        value = self.query_dense(value)

        #2. heads split
        #(batch_size, num_heads, seq_len, d_model/num_heads)
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        #3. 스케일드 닷 프로덕트 어텐션
        scaled_attention = scaled_dot_procudt_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm = [0, 2, 1, 3])

        #4. heads concat
        concat_attention = tf.reshpae(scaled_attention, (batch_size, -1, self.d_model))

        #5. Wo에 해당하는 밀집층
        outputs = self.dense(concat_attention)

        return outputs
'''
7)포지션-와이즈 피드 포워드 신경망
두번째 sublayer
인코더와 디코더에서 공통적으로 존재
FFNN(x) = MAX(0, x*W1 + b1) * W2 + b2
x -> F1 = x*W1+b1 -> Relu==F2 = max(0, F1) -> F3 = F2*W2 + b2
'''
def point_wise_feed_forward_network(d_model, dff):
    outputs = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation="relu"), #relu는 첫번째 layer에만
        tf.keras.layers.Dense(d_model)
    ])
    return outputs
'''
8)잔차연결
sublayer의 출력에 입력값을 더하는것
H(x) = x + F(x)

9)층 정규화

'''