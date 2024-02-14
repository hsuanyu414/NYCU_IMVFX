環境資訊
-	硬體參數
	CPU：13th Gen Intel(R) Core(TM) i5-13600K   3.50 GHz
	RAM： 32.0 GB 
	GPU：NVIDIA GeForce RTX 4070
-	軟體參數
	系統類型	64 位元作業系統，x64 型處理器
	版本	Windows 11 專業教育版
	版本	22H2
	OS 組建	22621.2428
	Cuda 版本：11.8


本作業使用基於 pytorch，請事先安裝 Cuda 11.8 
並使用 Anaconda 重建環境，請先安裝 Anaconda 並使用 Anaconda 建立環境，安裝套件。

作業放置資料夾以下簡稱 {hw_root}，請自行替換成本作業的根目錄。

1. 重建環境
$cd {hw_root}
$conda create -n imvfx python=3.8

2. 進入環境&安裝套件
$conda activate imvfx
$pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
$pip3 install opencv-python
$pip3 install matplotlib
$pip3 install tqdm
$pip3 install imageio
$pip3 install numpy
$pip3 install einops

3. 下載資料集(助教提供之連結)
- MNIST dataset: https://drive.google.com/file/d/16j8CG2FxVIUGpn4ZmocCl_wNj9VGv6FM/view?usp=drive_link
- Anime Face dataset: https://drive.google.com/file/d/1UOJ-C_TQxsTdN0SaH2uhQ4rEOh43qnTJ/view?usp=drive_link

4. 解壓縮
將剛剛下載的資料集下載後解壓縮，並將資料夾放到本作業的根目錄下
並重新命名為 mnist_dataset 與 anime_face_dataset，資料夾結構如下：
{hw_root}/
    mnist_dataset/
        data/
            0/
            ├── 1.jpg
            ├── 21.jpg
            ├── 34.jpg
            ├── ...
            ...
    anime_face_dataset/
        data/
        ├── 1.png
        ├── 2.png
        ├── 3.png
        ├── ...

5. 執行
$cd {hw_root}

5.1 MNIST dataset
$python 312551080_hw2_2_mnist.py

5.2 Anime Face dataset
$python 312551080_hw2_2_anime_face.py

6. 結果
執行完後，執行過程會分別存在 {hw_root}/mnist_log/ 與 {hw_root}/anime_face_log/ 中，
兩 model 權重 mnist.pt 與 anime_face.pt 與作業要求的相關檔案會直接存在 {hw_root} 中
結果會儲存在 {hw_root} 中
