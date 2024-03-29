
import sys
import argparse

import torch, cv2
import data, train
import numpy as np
from neural_painter import NeuralPaintStroke, NeuralUpscale
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
        type=int
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

    paint = subparsers.add_parser('paint')
    paint.add_argument(
        'model',
        help='model parameter name',
        type=str
    )
    paint.add_argument(
        'strokes',
        help='number of strokes',
        type=int
    )
    paint.add_argument(
        '--epochs', '-e',
        required=False,
        default=100,
        type=int
    )
    paint.add_argument(
        '--learning-rate', '-lr',
        required=False,
        default=None,
        type=float
    )
    paint.add_argument(
        '--simultaneous', '-s',
        required=False,
        default=1,
        type=int
    )
    paint.add_argument(
        '--background', '-b',
        required=False,
        default=None,
        type=float
    )
    paint.add_argument(
        '--target', '-t',
        required=False,
        default="data/target_col.png",
        type=str
    )

    chunk = subparsers.add_parser('chunk')
    chunk.add_argument(
        'model',
        help='model parameter name',
        type=str
    )
    chunk.add_argument(
        'target',
        help='target file name',
        type=str
    )
    chunk.add_argument(
        '--chunks', '-c',
        help='number of chunks (calculated from width)',
        required=False,
        default=8,
        type=int
    )

    upscale = subparsers.add_parser('upscale')

    args = parser.parse_args(sys.argv[1:])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.command == "gen-stroke":
        print(f"Generating {args.number} samples...")
        x, y = data.generate(args.number)
        torchvision.utils.save_image(torch.tensor(y, dtype=torch.float), "strokes.png")

    if args.command == "train-stroke":
        print(f"Training stroke model for {args.epochs} epochs with size {args.epoch_size}.")
        print(f" load parameters: {args.load}")
        print(f" save as '{args.name}' every {args.save} epochs")
        print(f" saving preview draw every '{args.draw}' epochs")

        model = NeuralPaintStroke(5)

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
        model = NeuralPaintStroke(5)
        model.load_state_dict(torch.load(args.model, map_location=device))

        x, _ = data.generate(32)
        acts = torch.tensor(x, dtype=torch.float)
        cols = torch.ones(32, 1)

        for i in range(30):
            acts = torch.clip(acts, 0, 1)
            p = model.forward(acts)
            torchvision.utils.save_image(p, "test/test_{:05d}.png".format(i))
            acts += torch.rand(32, 5, dtype=torch.float) * 0.2 - 0.1

    if args.command == "paint":
        model = NeuralPaintStroke(5)
        model.load_state_dict(torch.load(args.model, map_location=device))

        target = cv2.imread(args.target, cv2.IMREAD_COLOR)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = torch.tensor([target / 255.0], dtype=torch.float)
        target = target.permute(0, 3, 1, 2)

        train.train_painting(
            target,
            model,
            epochs=args.epochs,
            strokes=args.strokes,
            simultaneous=args.simultaneous,
            background=args.background,
            learning_rate=args.learning_rate
        )

    if args.command == "chunk":
        model = NeuralPaintStroke(5)
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))

        train.paint_chunked(
            args.target,
            model,
            chunks=args.chunks
        )

    if args.command == "upscale":
        model = NeuralPaintStroke(5)
        model.load_state_dict(torch.load("goodmodel/clean32", map_location=device))
        upscale = NeuralUpscale()
        upscale.load_state_dict(torch.load("goodmodel/upscale256", map_location=device))
        train.train_upscale(model, upscale)
