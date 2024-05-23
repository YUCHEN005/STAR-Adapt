# Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models

[[Paper]]() 

<p align="center">  <img src="https://github.com/YUCHEN005/STAR-Adapt/blob/master/star.png" height ="180"> </p>

This work proposes a source-free unsupervised domain adaptation approach for speech foundation models.

## Conda Environment Configuration

Our conda environment is provided via the file `requirements.txt`, please run the command below to install necessary packages:
```bash
pip install -r requirements.txt
```

## Data Preparation

Our code requires two kaldi-format data files: `wav.scp` and `text`.

- `wav.scp` contains a list of audio files, each line includes sample ID and absolute audio path:
    
  ```
  utt_1  /your-data-path/1.wav
  utt_2  /your-data-path/2.wav
  ```

- `text` contains a list of ground-truth transcriptions, each line includes sample ID and transcription:
    
  ```
  utt_1  i feel good
  utt_2  he is coming back
  ```
  
**NOTE:** each line in above two files should be paired.


## Training
Please refer to our training script `finetune.sh` and specify some settings:
- `dataset`: training data name;
- `model_size`: whisper model size;
- `train_data`: training data directory that contains files `wav.scp` and `text`;
- `dev_data`: development data directory that contains files `wav.scp` and `text`;

Then, please run command `bash finetune.sh` to start training. The model weights will be saved at `runs/{dataset}_{model_size}`.


## Inference
Please refer to our inference script `inference.sh` and specify some settings:
- `dataset`: training data name;
- `model_size`: whisper model size;
- `checkpoint`: path of the trained model checkpoint (`.pth` file);
- `test_data`: test data directory that contains files `wav.scp` and `text`;

Please run command `bash inference.sh` for inference. WER results would be printed in the log.


## References

We kindly hope you can cite our paper in your publication when using our research or code:
```bib
@article{hu2024self,
  title={Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models},
  author={Hu, Yuchen and Chen, Chen and Yang, Chao-Han Huck and Qin, Chengwei and Chen, Pin-Yu and Chng, Eng Siong and Zhang, Chao},
  journal={arXiv preprint arXiv:2405},
  year={2024}
}
```