"""Starter script to evaluate the pipeline model (placeholder).
"""
import argparse


def main(args):
    print('Placeholder eval script. Implement evaluation code that logs metrics to MLflow.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    main(args)
