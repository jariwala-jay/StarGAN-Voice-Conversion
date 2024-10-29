#StarGAN-Voice-Conversion
This repository contains code for StarGAN-VC, a model designed for non-parallel many-to-many voice conversion. The model modifies mel-spectrograms based on a target speaker index and generates waveforms using neural vocoders like HiFi-GAN or Parallel WaveGAN.
##Table of Contents
1. Overview
2. Installation
3. Dependencies
4. Dataset
5. Preprocessing
6. Training
7. Running the Model
8. Results
9. Acknowledgments
##Overview
This project aims to convert voices from one speaker to another by altering the mel-spectrograms and using neural vocoders for waveform synthesis. Due to limited resources, we are training this model in increments, and currently, only 10,000 iterations have been completed. Further training is ongoing.
##Installation
Clone the repository and navigate into the project directory:
bash
git clone https://github.com/your-username/StarGAN-Voice-Conversion
cd StarGAN-Voice-Conversion


##Creating the Conda Environment
Ensure you have Anaconda installed, and then create a Conda environment with the specific dependencies as follows:
conda create -n stargan_env python=3.6
conda activate stargan_env


##Dependencies
Below are the main libraries required for this project, along with their specific versions:
* Python: 3.6 (or 3.5)
* PyTorch: 0.4.0
* PyWorld
* TQDM
* Librosa: 0.6.0
* TensorBoardX and TensorBoard
Other libraries and their specific versions based on our environment setup are as follows:
* Numpy: 1.19.2
* Scipy: 1.5.4
* Scikit-learn: 0.24.1
* Matplotlib: 3.3.2
* Pillow: 8.0.1
* TensorFlow: 2.3.0
* Protobuf: 3.15.8
* Soundfile: 0.10.3.post1
* Google-auth: 1.22.1
* Decorator: 5.1.1
* Joblib: 1.2.0
* Cython: 0.29.22
##To install the required libraries, use the following commands:
bash
conda install numpy=1.19.2 scipy=1.5.4 scikit-learn=0.24.1 matplotlib=3.3.2 pillow=8.0.1
conda install pytorch=0.4.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install absl-py==0.14.0 audioread==3.0.0 librosa==0.8.0 tensorflow==2.3.0 protobuf==3.15.8 soundfile==0.10.3.post1 requests==2.24.0 google-auth==1.22.1 decorator==5.1.1 joblib==1.2.0 Cython==0.29.22


##Dataset
We use the CSTR VCTK Corpus, a large dataset containing speech from 110 English speakers with various accents. Each speaker reads around 400 sentences, drawn from newspapers, the rainbow passage, and an elicitation paragraph. Recordings were made at 96kHz/24-bit in a controlled environment and later downsampled to 48kHz/16-bit. The dataset provides a rich diversity of accents and phonetic contexts, making it ideal for voice conversion tasks.
Training Note
Due to limited resources, only 10,000 training iterations have been completed so far, with plans for further training to improve model performance.

https://www.kaggle.com/datasets/pratt3000/vctk-corpus
Preprocessing
To prepare the dataset for training, run the preprocess.py script:
python preprocess.py 


This will convert audio files into mel-spectrograms and organize them for training. Ensure that the dataset is structured correctly to avoid any preprocessing issues.
Training
To start training the model, use the following command:


python train.py


 Adjust parameters like batch size, learning rate, and the number of epochs based on your requirements and computational resources.
Training Notes
Due to limited resources, we are currently training the model in increments, with 10,000 iterations completed. Further training is planned to enhance the model’s performance.
Running the Model
Once training is complete, you can run main.py to test the model’s voice conversion capabilities:


python main.py --config config.yaml --checkpoint <path_to_checkpoint>


Replace <path_to_checkpoint> with the path to your trained model checkpoint file. This will perform voice conversion based on the input and target speaker indices specified in the configuration.
Results
Results, including converted samples and logs, are saved in the samples and logs directories, respectively. You can monitor the training and conversion progress by checking these directories.
Acknowledgments
This project utilizes the StarGAN-VC framework for voice conversion. Special thanks to the developers of the libraries and tools used, such as PyTorch, Librosa, and TensorBoard.
________________
