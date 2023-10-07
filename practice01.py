#"https://qiita.com/Uta10969/items/a5dc0d37ebfc9ac6400b"を参考に自作数字認識を作ってみる
#python 3.9.12
#torch 1.8.1
#torchvision 0.9.1
#最初にインストールしたtorchvisionは0.15.2でこれだとダメだった
#いろいろ入れたので上記環境ではない anacondaの仮想環境pytorchを使用

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

#MNISTから訓練データ取得
train_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = True,
                                           transform = transforms.ToTensor(),
                                           download = True)
#MNISTから検証データ取得
test_dataset = torchvision.datasets.MNIST(root = './data',
                                           train = False,
                                           transform = transforms.ToTensor(),
                                           download = True)

print("train_dataset\n",train_dataset)
print("\ntest_dataset\n",test_dataset)

#ミニバッチ処理
batch_size = 256

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

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

#gpuを有効にできず困っていた
#いろいろ試したが下記URLにあった3つのpipを実行することで解決できた
#cpu-onlyのバージョンが残っていたのかも
#"https://github-com.translate.goog/pytorch/pytorch/issues/30664?_x_tr_sl=en&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=op,sc"
#pip uninstall torch
#pip cache purge
#pip install torch -f https://download.pytorch.org/whl/torch_stable.html
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("device : ", device)
model = Net(input_size, hidden1_size, hidden2_size, output_size).to(device) #Networkもdevice指定が必要
print(model)

#学習済みモデル保存パス設定
weight_dir = "./weight/"
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)
path = weight_dir + "practice01.pth"

#損失関数 criterion：基準
#CrossEntropyLoss：交差エントロピー誤差関数
criterion = nn.CrossEntropyLoss()

#最適化法の指定 optimizer：最適化
#SGD：確率的勾配降下法
optimizer = optim.SGD(model.parameters(), lr = 0.01)

#学習1エポック分を関数化
def train_model(model, train_loader, criterion, optimizer, device = 'cuda:0'):
    train_loss = 0.0 #train損失用変数を定義
    num_train = 0    #学習回数記録用変数を定義

    #モデルを学習モードに変換
    model.train()

    #データの分割数分繰り返す
    #バッチサイズ分のデータで1回パラメータを修正する
    for i, (images, labels) in enumerate(train_loader):

        #batch数をカウント
        num_train += len(labels)

        images, labels = images.view(-1, 28*28).to(device), labels.to(device)

        #勾配を初期化
        optimizer.zero_grad()

        #1 推論（順伝搬）
        outputs = model(images)

        #2 損失の算出
        loss = criterion(outputs, labels)

        #3 勾配計算
        loss.backward()

        #4 パラメータの更新
        optimizer.step()

        #lossを加算
        train_loss += loss.item()

    #lossの平均値をとる
    train_loss = train_loss / num_train

    return train_loss

#検証データによるモデル評価を行う関数の定義
def test_model(model, test_loader, criterion, optimizer, device = 'cuda:0'):
    test_loss = 0.0
    num_test = 0

    #modelを評価モードに変更
    model.eval()

    with torch.no_grad(): #勾配計算の無効化
        for i, (images, labels) in enumerate(test_loader):
            num_test += len(labels)
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        #lossの平均値をとる
        test_loss = test_loss / num_test
    return test_loss

#モデル学習を行う関数の定義
def learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device = 'cuda:0'):
    train_loss_list = []
    test_loss_list = []

    #epoch数分繰り返す
    for epoch in range(1, num_epochs + 1, 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device = device)
        test_loss = test_model(model, test_loader, criterion, optimizer, device = device)

        print("epoch : {}, train_loss : {:.5f}, test_loss : {:.5f}" .format(epoch, train_loss, test_loss))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list

#学習
num_epochs = 10
train_loss_list, test_loss_list = learning(model, train_loader, test_loader, criterion, optimizer, num_epochs, device = device)

#学習済みモデルの保存
torch.save(model.state_dict(),path)

#学習推移の確認
plt.plot(range(len(train_loss_list)), train_loss_list, c='b', label='train loss')
plt.plot(range(len(test_loss_list)), test_loss_list, c='r', label='test loss')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid()
plt.show()

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

"""
これからやりたいこと（2023/10/04）
1 学習済みモデルの外部ファイル化（？）・外部ファイル出力（？）
2 別のテストモデルでの学習（英数字での学習や日本語の学習等）
3 このソースではなく学習済みの本をベースに画像認識
"""