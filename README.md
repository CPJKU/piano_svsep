# Piano Staff and Voice Separation (Piano_SVSep)
Code for the paper Cluster and Separate: A GNN Approach to Voice and Staff Prediction for Score Engraving.

This work approaches the problem of separating the notes from a quantised symbolic music piece (e.g., a MIDI file) into multiple voices and staves. This is a fundamental part of the larger task of music score engraving (or score type-setting), which aims to produce readable musical scores for human performers. We focus on piano music and support homophonic voices, i.e., voices that can contain chords, and cross-staff voices, which are notably difficult tasks that have often been overlooked in previous research. We propose an end-to-end system based on graph neural networks that clusters notes that belong to the same chord and connects them with edges if they are part of a voice. Our results show clear and consistent improvements over a previous approach on two datasets of different styles. To aid the qualitative analysis of our results, we support the export in symbolic music formats and provide a direct visualisation of our outputs graph over the musical score. 


## Installation

To install Piano_SVSep you first need to install the Pytorch version suitable for your system.
You can find the instructions [here](https://pytorch.org/get-started/locally/).

You also need to install Pytorch Geometric. You can find the instructions [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
We recommend to use conda:
```shell
conda install conda install pyg -c pyg
```

You can install the rest of the dependencies using pip:
```
pip install -r requirements.txt
```

## Usage


## Cite Us

```bibtex
@inproceedings{piano_SVSep_2024,
  title={Cluster and Separate: a GNN Approach to Voice and Staff Prediction for Score Engraving},
  author={Foscarin, Francesco and Karystinaios, Emmanouil and Nakamura, Eita and Widmer, Gerhard},
  booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR)},
  year={2024}
}
```
