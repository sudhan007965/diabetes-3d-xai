# -*- coding: utf-8 -*-
"""
Inside Diabetes — Neural Network Training & XAI Pipeline
Model: Multilayer Perceptron (Artificial Neural Network: 8 -> 32 -> 16 -> 1)
Dataset: Pima Indians Diabetes Database (NIDDK)
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train_and_evaluate_all import run_experiment

if __name__ == '__main__':
    run_experiment()
