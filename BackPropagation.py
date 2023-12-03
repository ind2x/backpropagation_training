import numpy as np

# Usage: python .\BackPropagation_2019270109.py
# 테스트 데이터는 직접 입력해줘야 함

# 소수점 6자리까지만 표시
np.set_printoptions(precision=6, suppress=True)

# 순전파 단계 계산 (제곱오차 제외)
def calcValues(image_pattern_data, WeightInputToHidden, BiasInputToHidden, 
               WeightHiddenToOutput, BiasHiddenToOutput):
    # z2_i 계산
    hidden_layer_input = np.dot(image_pattern_data, WeightInputToHidden) + BiasInputToHidden
    # a2_i 계산 --> 64x3 형태의 행렬
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # z3_i 계산
    output_layer_input = np.dot(hidden_layer_output, WeightHiddenToOutput) + BiasHiddenToOutput
    # a3_i 계산 --> 64x2 형태의 행렬
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

# activation function (시그모이드 함수 사용)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 제곱오차 계산
# t1 = '0'의 정답 데이터 변수, 이미지가 0인 경우 1
# t2 = '1'의 정답 데이터 변수, 이미지가 0인 경우 0
def SquaredError(output_layer_output):
    # 1 ~ 32번 데이터는 이미지가 0이므로 t1 = 1, t2 = 0
    # 33 ~ 64번은 이미지가 1이므로 t1 = 0, t2 = 1

    # 1. 64x2 행렬을 32x2, 32x2 행렬 두 개로 나눔
    zero = output_layer_output[:32]     # 0 ~ 31
    one = output_layer_output[32:]      # 32 ~ 63

    # 2. (t1 - a3_1)^2, (t2 - a3_2)^2 계산
    zero = np.square([1,0]-zero)
    one = np.square([0,1]-one)

    # 3. 2에서 계산한 값에서 1열과 2열의 값 더한 후 나누기 2 진행 
    zero = (zero[:,0] + zero[:,1])/2
    one = (one[:,0] + one[:,1])/2
    
    # 64x1 행렬로 변경 후 합치기
    zero = zero.reshape(32,1)
    one = one.reshape(32,1)
    squared_error_output = np.concatenate((zero, one), axis=0)
    
    return squared_error_output

# 시그모이드 함수의 미분값 계산
def DerivativeOfSigmoid(a):
    return a*(1-a)

# 출력층의 유닛 오차 계산
def calcErrorOfUnitOfOutputLayer(output_layer_output):
    # 1. 64x2 행렬을 32x2, 32x2 행렬 두 개로 나눔
    zero = output_layer_output[:32]     # 0 ~ 31 -> t1 = 1, t2 = 0
    one = output_layer_output[32:]      # 32 ~ 63 -> t1 = 0, t2 = 1
    
    # 2. 계산 및 합치기
    zero = (zero-[1,0])*DerivativeOfSigmoid(zero)
    one = (one-[0,1])*DerivativeOfSigmoid(one)
    EoU3 = np.concatenate((zero, one), axis=0)
    
    return EoU3
    
# 은닉층의 유닛 오차 계산
def calcErrorOfUnitOfHiddenLayer(EoU3, WeightHiddenToOutput, hidden_layer_output):
    # 3x2 --> 2x3
    w3_i = WeightHiddenToOutput.T
    
    # 64x3 형태의 행렬
    EoU2 = np.dot(EoU3, w3_i) * DerivativeOfSigmoid(hidden_layer_output)
    
    return EoU2

# 제곱오차 C의 가중치에 관한 편미분 계산
def DerivativeOfSquaredError(EoU, ai_j):
    dataEoU = EoU.shape[1]      # 유닛의 오차의 데이터 개수(은닉층=3, 출력층=2)
    dataAi_j = ai_j.shape[1]    # 각각의 데이터의 원소의 개수
    
    total_w = []
    total_b = []

    for i in range(0, 64):
        res_w = []
        res_b = []
        
        for j in range(dataEoU):
            res_w.append(ai_j[i]*EoU[i][j])
            res_b.append(EoU[i][j])
        total_w.append(res_w)
        total_b.append(res_b)
    
    dCdW = np.array(total_w)    # 64개의 데이터에 관하여 은닉층은 3x12 형태
                                # 출력층은 2x3 형태 -> (64, 2, 3)
    
    dCdB = np.array(total_b)    # 64개의 데이터에 관해 은닉층은 64x3 형태
                                # 출력층은 64x2 형태 -> (64, 2)
    
    return dCdW, dCdB

# 가중치, 편향 기울기 계산
def DerivativeOfCostFunction(dCdW, dCdB):
    # 은닉층은 가중치는 3x12, 편향은 1x3 형태
    # 출력층은 가중치는 2x3, 편향은 1x2 형태

    dCtdW_H = dCdW[0]
    dCtdB_H = dCdB[0]

    for i in range(1,64):
        dCtdW_H += dCdW[i]
        dCtdB_H += dCdB[i]
    
    return dCtdW_H, dCtdB_H

# 오차역전파 계산
def BackPropagation(image_pattern_data, learning_rate, epochs):
    global iWeightInputToHidden, iWeightHiddenToOutput
    global iBiasInputToHidden, iBiasHiddenToOutput

    # 가중치, 편향
    WeightInputToHidden = iWeightInputToHidden.T
    WeightHiddenToOutput = iWeightHiddenToOutput.T
    BiasInputToHidden = iBiasInputToHidden
    BiasHiddenToOutput = iBiasHiddenToOutput

    for epoch in range(epochs):
        # 1. forward
        hidden_layer_output, output_layer_output = calcValues(image_pattern_data, WeightInputToHidden, BiasInputToHidden, WeightHiddenToOutput, BiasHiddenToOutput)

        # 제곱오차 계산
        squared_error_output = SquaredError(output_layer_output)

        # 2. backward
        # 오차역전파법으로 유닛의 오차 계산
        # 출력층의 유닛의 오차 계산 --> 64x2 형태의 행렬
        EoU3_i = calcErrorOfUnitOfOutputLayer(output_layer_output)

        # 은닉층의 유닛의 오차 계산 --> 64x3 형태의 행렬
        EoU2_i = calcErrorOfUnitOfHiddenLayer(EoU3_i, WeightHiddenToOutput, hidden_layer_output)

        # 유닛의 오차에서 제곱오차 C의 편미분 계산
        # 제곱오차 C 편미분 (은닉층)
        # dCdW_H = (64, 3, 12), dCdB_H = (64, 3)
        dCdW_H, dCdB_H = DerivativeOfSquaredError(EoU2_i, image_pattern_data)
        
        # 제곱오차 C 편미분 (출력층)
        # dCdW_O = (64, 2, 3), dCdB_O = (64, 2)
        dCdW_O, dCdB_O = DerivativeOfSquaredError(EoU3_i, hidden_layer_output)

        # 3. 비용함수 C_t와 기울기 C_t 계산
        sumOfC = np.sum(squared_error_output)
        dCtdW_H, dCtdB_H = DerivativeOfCostFunction(dCdW_H, dCdB_H)
        dCtdW_O, dCtdB_O = DerivativeOfCostFunction(dCdW_O, dCdB_O)
        
        # 4. 가중치, 편향 업데이트
        # 입력층 -> 은닉층
        WeightInputToHidden -= (learning_rate * dCtdW_H).T
        BiasInputToHidden -= (learning_rate * dCtdB_H).T
        # 은닉층 -> 출력층
        WeightHiddenToOutput -= (learning_rate * dCtdW_O).T
        BiasHiddenToOutput -= (learning_rate * dCtdB_O).T
        
        print(f"[*] epoch = {epoch+1}, error = {sumOfC}")
    
    return WeightInputToHidden, BiasInputToHidden, WeightHiddenToOutput, BiasHiddenToOutput

# 1. 학습 데이터 준비 (이미지 패턴, 정답), 총 64개의 데이터
# 64x12 형태의 행렬
image_pattern_data = np.array([
    [1,1,1,1,0,1,1,0,1,1,1,1], # 1
    [0,1,1,1,0,1,1,0,1,1,1,1], # 2
    [1,1,0,1,0,1,1,0,1,1,1,1], # 3
    [1,1,1,1,0,1,1,0,1,1,1,0], # 4
    [1,1,1,1,0,1,1,0,1,0,1,1], # 5
    [0,0,0,1,1,1,1,0,1,1,1,1], # 6
    [0,0,0,0,1,1,1,0,1,1,1,1], # 7
    [0,0,0,1,1,0,1,0,1,1,1,1], # 8
    [0,0,0,1,1,1,1,0,1,1,1,0], # 9
    [0,0,0,1,1,1,1,0,1,0,1,1], # 10
    [1,1,1,1,0,1,1,1,1,0,0,0], # 11
    [0,1,1,1,0,1,1,1,1,0,0,0], # 12
    [1,1,0,1,0,1,1,1,1,0,0,0], # 13
    [1,1,1,1,0,1,1,1,0,0,0,0], # 14
    [1,1,1,1,0,1,0,1,1,0,0,0], # 15
    [1,0,1,1,0,1,1,0,1,1,1,1], # 16
    [1,1,1,1,0,0,1,0,1,1,1,1], # 17
    [1,1,1,1,0,1,1,0,0,1,1,1], # 18
    [1,1,1,1,0,1,1,0,1,1,0,1], # 19
    [1,1,1,1,0,1,0,0,1,1,1,1], # 20
    [1,1,1,0,0,1,1,0,1,1,1,1], # 21
    [0,0,1,1,0,1,1,0,1,1,1,1], # 22
    [0,1,1,1,0,0,1,0,1,1,1,1], # 23
    [0,1,1,1,0,1,1,0,0,1,1,1], # 24
    [0,1,1,1,0,1,1,0,1,1,0,1], # 25
    [0,1,1,1,0,1,0,0,1,1,1,1], # 26
    [0,1,1,0,0,1,1,0,1,1,1,1], # 27
    [1,1,0,1,0,0,1,0,1,1,1,1], # 28
    [1,1,0,1,0,1,1,0,0,1,1,1], # 29
    [1,1,0,1,0,1,1,0,1,1,0,1], # 30
    [1,1,0,1,0,1,0,0,1,1,1,1], # 31
    [1,1,0,0,0,1,1,0,1,1,1,1], # 32
    [0,1,0,0,1,0,0,1,0,0,1,0], # 33
    [1,1,0,0,1,0,0,1,0,0,1,0], # 34
    [0,1,0,0,1,0,0,1,0,0,1,0], # 35
    [0,1,0,0,1,0,0,1,0,1,1,0], # 36
    [0,1,0,0,1,0,0,1,0,0,1,1], # 37
    [1,1,0,0,1,0,0,1,0,1,1,0], # 38
    [1,1,0,0,1,0,0,1,0,0,1,1], # 39
    [1,1,0,0,1,0,0,1,0,1,1,1], # 40
    [0,1,0,0,1,1,0,1,0,0,1,0], # 41
    [0,1,0,0,1,0,0,1,1,0,1,0], # 42
    [1,1,0,0,1,1,0,1,0,0,1,0], # 43
    [1,1,0,0,1,0,0,1,1,0,1,0], # 44
    [0,1,0,0,1,1,0,1,0,1,1,0], # 45
    [0,1,0,0,1,0,0,1,1,1,1,0], # 46
    [0,1,0,0,1,0,0,1,0,1,1,1], # 47
    [1,1,0,0,1,1,0,1,1,0,1,1], # 48
    [1,1,0,0,1,0,0,1,0,0,1,0], # 49
    [0,1,1,0,1,1,0,1,1,0,1,1], # 50
    [1,1,0,1,1,0,0,1,0,0,1,0], # 51
    [1,1,0,0,1,0,1,1,0,0,1,0], # 52
    [1,1,0,1,1,0,1,1,0,1,1,0], # 53
    [1,1,0,0,1,0,0,0,0,0,1,0], # 54
    [0,1,0,0,1,0,0,1,0,1,0,0], # 55
    [1,0,0,0,1,0,0,1,0,0,1,0], # 56
    [1,0,0,0,1,0,0,1,0,0,0,1], # 57
    [0,1,0,0,0,0,0,1,0,1,1,0], # 58
    [0,1,0,0,1,0,0,0,0,1,1,0], # 59
    [0,0,0,0,1,0,0,1,0,1,1,0], # 60
    [0,0,0,0,1,0,0,1,0,0,1,0], # 61
    [0,1,0,0,1,0,0,1,0,0,0,0], # 62
    [0,1,0,0,0,1,0,0,1,0,1,0], # 63
    [0,1,0,1,0,0,1,0,0,0,1,0], # 64
])

# 64x1 형태의 행렬
image_pattern_answer = np.array([
    [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
    [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]
])

# 2. 각 유닛의 가중치와 편향의 초기값 설정
# 정규분포 난수를 이용, 학습률은 적절한 작은 값을 갖는 양의 상수로 설정
iWeightInputToHidden = np.array([     
    # 초기 가중치 값 (입력층 -> 은닉층), 3x12 형태의 행렬
    [0.679, 0.089, -0.667, -0.742, -1.534, -1.14, -0.332, -1.994, -0.75, -0.336, -0.174, -0.267],
    [-0.309, -0.505, 0.126, 1.746, 0.507, 1.819, -0.065, 0.861, -0.482, -0.966, -0.682, 0.993],
    [2.042, 0.074, -0.041, -0.51 , 1.069, 0.436, 0.532, -0.275, 0.676, 0.382, 1.301, 0.115]
])

iBiasInputToHidden = np.array([0.032, -0.662, -0.296]) # 초기 편향 값 (입력층 -> 은닉층)

iWeightHiddenToOutput = np.array([   
    # 초기 가중치 값 (은닉층 -> 출력층)
    # 2x3 형태로 작성
    [-2.402, 0.908, -1.415],
    [1.499, 0.474, -1.456],
])
iBiasHiddenToOutput = np.array([0.185, 1.676])  # 초기 편향 값 (은닉층 -> 출력층)

learning_rate = 0.2
epochs = 100

# training 및 test
wH, bH, wO, bO = BackPropagation(image_pattern_data, learning_rate, epochs)
# Test
image_test_data = np.array([
    [0,1,0,1,0,1,1,0,1,1,0,1],
    [0,1,1,0,1,0,0,1,0,0,1,0],
])

print('\n\n')

for data in image_test_data:
    a2i, a3i = calcValues(data, wH, bH, wO, bO)
    print("---------- TEST ------------")
    print(data[0:3])
    print(data[3:6])
    print(data[6:9])
    print(data[9:12])
    print("Result:", a3i)

    if a3i[0] > a3i[1]:
        print("Predict:", 0, '\n')
    else:
        print("Predict:", 1, '\n')
