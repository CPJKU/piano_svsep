import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from piano_svsep.models.pl_models import PLPianoSVSep, AlgorithmicVoiceSeparationModel
from piano_svsep.data.mix_vs import GraphPolyphonicVoiceSeparationDataModule
import argparse
from pytorch_lightning import Trainer, seed_everything

# for repeatability
seed_everything(0, workers=True)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default="-1",
                        help="GPUs to use, for multiple separate by comma, i.e. 0,1,2. Use -1 for CPU. (Default: -1)")
    parser.add_argument('--n_layers', type=int, default=3,
                        help="Number of layers on the Graph Convolutional Encoder Network")
    parser.add_argument('--n_hidden', type=int, default=256, help="Number of hidden units")
    parser.add_argument('--n_epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--dropout', type=float, default=0.5, help="Dropout")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of workers")
    parser.add_argument("--load_from_checkpoint", action="store_true", help="Load model from WANDB checkpoint")
    parser.add_argument("--linear_assignment", action="store_true",
                        help="Use linear assignment Hungarian algorithm for val and test predictions")
    parser.add_argument("--force_reload", action="store_true", help="Force reload of the data")
    parser.add_argument("--collection", type=str, choices=["musescore_pop", "dcml"], default="musescore_pop",
                        help="Collection to use")
    parser.add_argument("--model", type=str, default="SageConv", help="Block Convolution Model to use",
                        choices=["SageConv"])
    parser.add_argument("--tags", type=str, default="", help="Tags to add to the WandB run api")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--compile", action="store_true", help="Compile the model")
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb")
    parser.add_argument("--method", type=str, default="vocsep", choices=["vocsep", "baseline"], help="Method to use")
    parser.add_argument("--subgraph_size", type=int, default=1000, help="Subgraph size")
    parser.add_argument("--no_pos_weight", action="store_true", help="Disable positive weight for the loss")
    parser.add_argument("--chord_pooling_mode", type=str, default="cos_similarity",
                        choices=["mlp", "dot_product", "dot_product_d", "cos_similarity", "cos_similarity_d", "none"],
                        help="Chord pooling mode between 'mlp', 'dot_product' and 'none'")
    parser.add_argument("--staff_feedback", action="store_true",
                        help="Feed staff logits to the Voice prediction decoder")
    parser.add_argument("--feat_norm_scale", type=float, default=0.1,
                        help="Scale factor for the feature normalization loss")
    return parser


def main():
    parser = get_parser()

    args = parser.parse_args()

    # Determine the devices and accelerator to use based on the provided GPU argument
    if args.gpus == "-1":
        devices = 1
        accelerator = "cpu"
        use_ddp = False
    else:
        devices = [eval(gpu) for gpu in args.gpus.split(",")]
        accelerator = "auto"
        use_ddp = len(devices) > 1

    # Initialize the data module
    datamodule = GraphPolyphonicVoiceSeparationDataModule(batch_size=args.batch_size, subgraph_size=args.subgraph_size,
                                                          num_workers=args.num_workers, force_reload=args.force_reload,
                                                          test_collections=args.collection)
    datamodule.setup()

    # Set positive weights for the loss function if not disabled
    if not args.no_pos_weight:
        pos_weight = int(datamodule.pot_real_ratio)
        pos_weight_chord = int(datamodule.pot_real_ratio_chord)
        pos_weights = {"voice": pos_weight, "chord": pos_weight_chord}
        print(f"Using pos_weights: voice {pos_weight}, chord {pos_weight_chord}")
    else:
        pos_weights = {"voice": 1, "chord": 1}

    # Initialize the model based on the specified method
    if args.method == "vocsep":
        model = PLPianoSVSep(datamodule.features, args.n_hidden, args.n_layers, dropout=args.dropout,
                                               lr=args.lr, weight_decay=args.weight_decay, pos_weights=pos_weights,
                                               conv_type=args.model, chord_pooling_mode=args.chord_pooling_mode,
                                               feat_norm_scale=args.feat_norm_scale, staff_feedback=args.staff_feedback)
        model_name = f"{args.model}_{args.n_layers}x{args.n_hidden}-dropout={args.dropout}-lr={args.lr}-wd={args.weight_decay}-pos_weights={pos_weights['voice']},{pos_weights['chord']}-chord_pooling={args.chord_pooling_mode}-staff_feedback={args.staff_feedback}-fns={args.feat_norm_scale}"
        job_type = f"{args.model}-StaffFDB={args.staff_feedback}-ChordPool={args.chord_pooling_mode}"
    elif args.method == "baseline":
        model = AlgorithmicVoiceSeparationModel()
        model_name = "Baseline-Algorithm"
        job_type = "Baseline"
    else:
        raise ValueError(f"Method {args.method} not supported")

    # Compile the model if the compile flag is set and PyTorch version is >= 2.0.0
    if args.compile and torch.__version__ >= "2.0.0":
        model = torch.compile(model, dynamic=True)

    # Initialize WandB logger if enabled
    if args.use_wandb:
        wandb_logger = WandbLogger(
            project="Polyphonic-Voice-Separation",
            entity="vocsep",
            group=f"{args.collection}",
            job_type=job_type,
            name=model_name,
            tags=args.tags.split(",") if args.tags != "" else None,
            log_model=True,
        )
        wandb_logger.log_hyperparams(args)

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss", mode="min")

    # Initialize the trainer
    trainer = Trainer(
        max_epochs=args.n_epochs, accelerator=accelerator, devices=devices,
        num_sanity_val_steps=2,
        logger=wandb_logger if args.use_wandb else None,
        callbacks=[checkpoint_callback],
        # deterministic=False, # this is not yet supported by some functions in pytorch geometric
        reload_dataloaders_every_n_epochs=5,
        log_every_n_steps=10,
    )

    # Train and test the model based on the specified method
    if args.method == "vocsep":
        # train the model
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule, ckpt_path=checkpoint_callback.best_model_path)
    elif args.method == "baseline":
        trainer.test(model, datamodule)


if __name__ == "__main__":
    main()
