[![Python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2407.21030-B31B1B.svg)](https://arxiv.org/abs/2407.21030)
[![Conference](http://img.shields.io/badge/ISMIR-2024-4b44ce.svg)](https://ismir2024.ismir.net/papers)

**This repo is still under development, it will be finalized shortly** 

# Piano Staff and Voice Separation (Piano_SVSep)

Code for the paper "Cluster and Separate: A GNN Approach to Voice and Staff Prediction for Score Engraving".

_Nominated as Best Paper at ISMIR 2024!!!_

Read the full paper [here](https://arxiv.org/abs/2407.21030)

This work approaches the problem of separating the notes from a quantised symbolic music piece (e.g., a MIDI file) into multiple voices and staves. This is a fundamental part of the larger task of music score engraving (or score type-setting), which aims to produce readable musical scores for human performers. We focus on piano music and support homophonic voices, i.e., voices that can contain chords, and cross-staff voices, which are notably difficult tasks that have often been overlooked in previous research. We propose an end-to-end system based on graph neural networks that clusters notes that belong to the same chord and connects them with edges if they are part of a voice. Our results show clear and consistent improvements over a previous approach on two datasets of different styles. To aid the qualitative analysis of our results, we support the export in symbolic music formats and provide a direct visualisation of our outputs graph over the musical score. 


## Installation

To install Piano_SVSep you first need to install the Pytorch version suitable for your system.
You can find the instructions [here](https://pytorch.org/get-started/locally/).

Then clone the repo:
```shell
  git clone https://github.com/cpjku/piano_svsep.git
  cd piano_svsep
```

To install using pip simply navigate to the root of the project and run:
```shell
  pip install .
```

## Usage

### Prediction

To predict the voices and staves of a MIDI or musicXML file you can use the following command:
```shell
  python launch_scripts/predict.py --input_score path/to/midi --output_path path/to/output
```

#### Example

To predict the voice assignment for a given score using a pre-trained model, you can use the following command:
```shell
  python launch_scripts/predict.py --model_path pretrained_models/model.ckpt --score_path artifacts/test_score.musicxml --save_path artifacts/test_score_pred.mei
```

#### Visualization

To visualize the voice assignment use our tool [MusGViz](https://github.com/fosfrancesco/musgviz/)

With this tool you can visualize the input, output and ground truth of the graphs used by the model.
Furthermore, you can also visualize the exported MEI score.



### Train your own model

To train or test the voice separation model using the provided script, you can use the following command:

```bash
python launch_scripts/train.py --gpus 0 --n_layers 3 --n_hidden 128 --n_epochs 50 --dropout 0.3 --lr 0.0005 --weight_decay 0.0001 --num_workers 8 --collection musescore_pop --model SageConv --batch_size 32 --method vocsep --use_wandb --tags "experiment1,voice_separation"
```

This command will:
- Use GPU 0 for training.
- Set the number of layers to 3 and the number of hidden units to 128.
- Train the model for 50 epochs with a dropout rate of 0.3.
- Use a learning rate of 0.0005 and a weight decay of 0.0001.
- Use 8 workers for data loading.
- Use the `musescore_pop` collection and the `SageConv` model.
- Set the batch size to 32.
- Use the `vocsep` method for training.
- Enable logging with WandB and add the tags "experiment1" and "voice_separation".

For more details on the available command-line arguments, refer to the script's documentation.


## Cite Us

```bibtex
@inproceedings{piano_SVSep_2024,
  title={Cluster and Separate: a GNN Approach to Voice and Staff Prediction for Score Engraving},
  author={Foscarin, Francesco and Karystinaios, Emmanouil and Nakamura, Eita and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```

## Acknowledgments

This project receives funding from the European Research Council (ERC) under 
the European Union's Horizon 2020 research and innovation programme under grant 
agreement No 101019375 ["Whither Music?"](https://www.jku.at/en/institute-of-computational-perception/research/projects/whither-music/).

