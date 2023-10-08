#"https://qiita.com/Uta10969/items/a5dc0d37ebfc9ac6400b"を参考に自作数字認識を作ってみる
#出力した学習済みモデルを読み込んで推論の実施を行ってみた

#パッケージのimport
import torch #pytorchのリストを使うため
import torch.nn as nn #ネットワークの構築
import torch.nn.functional as F #様々な関数の使用
import torch.optim as optim #最適化アルゴリズムの使用
import torchvision #画像処理に関係する処理の使用
import torchvision.transforms as transforms #画像変換機能の使用

import os
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#予測関数の定義
class Net(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, output_size)

    def forward(self,x): #x:入力
        z1 = F.relu(self.fc1(x))
        z2 = F.relu(self.fc2(z1))
        y = self.fc3(z2)
        return y
            
#予測関数のインスタンス化
input_size = 28 * 28 #データセットに用いる画像の大きさ
hidden1_size = 1024 #中間層1の大きさ 中間層の大きさは任意
hidden2_size = 512  #中間層2の大きさ
output_size = 10 #分類数 今回は0-9の10個

#学習済みモデルのファイルパス
path = "./weight/practice01.pth"

#学習済みモデルのロード
model_weight = torch.load(path)
print('学習済みモデルのパラメータ：',model_weight)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device : ", device)
model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device) #Networkもdevice指定が必要
model.load_state_dict(model_weight)

#MNISTから検証データ取得
test_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = False,
                                           transform = transforms.ToTensor(),
                                           download = True)

#学習済みモデルにデータを投入
plt.figure(figsize = (20, 10))
for i in range(10):
    image, label = test_dataset[i]
    image = image.view(-1, 28*28).to(device)

    #推論
    prediction_label = torch.argmax(model(image))

    ax = plt.subplot(1, 10, i+1)

    plt.imshow(image.detach().to('cpu').numpy().reshape(28, 28), cmap = 'gray')
    ax.axis('off')
    ax.set_title('label : {}\n Prediction : {}'.format(label, prediction_label), fontsize = 15)
plt.show()