import argparse


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True, required=True,
                        help='True: Load trained model  False: Train model default: True')

    parser.print_help()

    return parser.parse_args()
