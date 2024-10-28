[![Python](https://img.shields.io/badge/-Python_3.8+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Paper](http://img.shields.io/badge/paper-arxiv.2407.21030-B31B1B.svg)](https://arxiv.org/abs/2407.21030)
[![Conference](http://img.shields.io/badge/ISMIR-2024-4b44ce.svg)](https://ismir2024.ismir.net/papers)

**This repo is still under development, it will be finalized shortly** 

# Piano Staff and Voice Separation (Piano_SVSep)

Code for the paper "Cluster and Separate: A GNN Approach to Voice and Staff Prediction for Score Engraving, by Francesco Foscarin and Emmanouil Karystinaios, Eita Nakamura, and Gerhard Widmer, at ISMIR 2014".

_Nominated as Best Paper at ISMIR 2024!_

Read the full paper [here](https://arxiv.org/abs/2407.21030).

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

Our system is supposed to be run in the final stage of a transcription or generation process, where a musical score is *almost* ready, but it is still missing voice and staff information.

The best way to simulate such a situation is to input a musical score (with potentially random voice and staff assignments). The system will then return a MEI version of the same musical score with newly produced voice and staff information.

For this, you can use the following command:
```shell
  python launch_scripts/predict.py --model_path path/to/model  --input_score path/to/input_score --output_path path/to/output_score
```

You can try it out on the example score we provide.
```shell
  python launch_scripts/predict.py --model_path pretrained_models/model.ckpt --score_path artifacts/test_score.musicxml --save_path artifacts/test_score_pred.mei
```
This uses the pretrained model we use for the evaluation in our paper.

#### Visualization

We provide the [MusGViz](https://github.com/fosfrancesco/musgviz/) tool for visualization, to directly inspect the input, output, and ground truth graphs.
 
Visit the repo and follow the instructions there. You can load the example files in `artifacts/` or produce new files.



### Train your own model

Unfortunately, the `jpop` dataset we used in our paper is not public, therefore the results of the paper are not entirely reproducible.
We still make all the code available, including the code to handle the jpop dataset in case you should obtain access to it.

We also changed the default parameters in our training script, to train only with the `dcml` dataset.
Specifically, the `train_datasets` parameter is not set to `dcml` only, while in the paper we use both `dcml` and `jpop` datasets.
is now set to `dcml` only.
To train, you can use the following command:

```bash
python launch_scripts/train.py 
```

The default parameters (excluding the `train_dataset` parameter) are the ones we used in the paper, and can be inspected with
```bash
python launch_scripts/train.py --help
```


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

