
import sys
import argparse

import torch
import data, train
import numpy as np
from neural_painter import NeuralPaintStroke
import torchvision

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Neural Painter')
    subparsers = parser.add_subparsers(dest='command')

    gen_stroke = subparsers.add_parser('gen-stroke')
    gen_stroke.add_argument(
        'number',
        help='number of samples',
        type=int
    )
    gen_stroke.add_argument(
        '--name', '-n',
        required=False,
        default='data',
        help='filename for data',
        type=str
    )

    train_stroke = subparsers.add_parser('train-stroke')

    train_stroke.add_argument(
        'epochs',
        help='number of epochs',
        type=int
    )
    train_stroke.add_argument(
        'epoch_size',
        help="epoch size",
        type=int
    )
    train_stroke.add_argument(
        '--refresh', "-r",
        help="refresh batch each n epoch",
        type=int,
        default=-1
    )
    train_stroke.add_argument(
        '--load', '-l',
        required=False,
        default=None,
        help='load parameters from file',
        type=str
    )
    train_stroke.add_argument(
        '--save', '-s',
        required=False,
        default=1,
        help='save checkpoint every n epochs',
        type=str
    )
    train_stroke.add_argument(
        '--name', '-n',
        required=False,
        default="model/stroke_model",
        type=str
    )
    train_stroke.add_argument(
        '--learning-rate', '-lr',
        required=False,
        default=None,
        type=float
    )
    train_stroke.add_argument(
        '--draw', '-d',
        required=False,
        default=1,
        help="draw preview image every n epochs",
        type=int
    )

    test_stroke = subparsers.add_parser('test-stroke')
    test_stroke.add_argument(
        'model',
        help='model parameter name',
        type=str
    )

    args = parser.parse_args(sys.argv[1:])

    if args.command == "gen-stroke":
        print(f"Generating {args.number} samples...")
        x, y = data.generate(args.number)
        print(f"Saving as '{args.name}'...")
        np.save(args.name+"_x", x)
        np.save(args.name+"_y", y)
        print("Done!")

    if args.command == "train-stroke":
        print(f"Training stroke model for {args.epochs} epochs with size {args.epoch_size}.")
        print(f" load parameters: {args.load}")
        print(f" save as '{args.name}' every {args.save} epochs")
        print(f" saving preview draw every '{args.draw}' epochs")

        model = NeuralPaintStroke(8)

        if args.load is not None:
            model.load_state_dict(torch.load(args.load))

        train.train_stroke(
            model,
            args.epoch_size,
            args.refresh,
            batch_size=32,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            save=args.save,
            name=args.name,
            draw=args.draw
       )

    if args.command == "test-stroke":
        model = NeuralPaintStroke(8)
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

        x, y = data.generate(16, verbose=False)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        p = model.forward(x)

        torchvision.utils.save_image(y, "test_y.png")
        torchvision.utils.save_image(p, "test_p.png")



