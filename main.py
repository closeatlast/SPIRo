import sys
import argparse

if __name__ == '__main__':
    frame_size = 48
    crop_size = 40

    parser = argparse.ArgumentParser(description='Binary PTSD vs. Non-PTSD Classification')

    # 1. Experiment Setting
    # 1.1. Server
    parser.add_argument('-gpu', default=2, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=1, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=1, type=int,
                        help='On HPC or not? If set to 1, GPU/CPU settings will be ignored, e.g. on Colab or NSCC.')

    # 1.2. Paths
    parser.add_argument('-dataset_path', default='/misc/scratch11/dataset_path', type=str,
                        help='Root directory of the preprocessed dataset.')
    parser.add_argument('-load_path', default='/misc/scratch11/pretrained_model', type=str,
                        help='Path to load pretrained backbones (e.g., res50).')
    parser.add_argument('-save_path', default='/misc/scratch11/save', type=str,
                        help='Path to save trained models and logs.')
    parser.add_argument('-python_package_path', default='/misc/scratch11/RCMA_new', type=str,
                        help='Path to the entire repository/codebase.')

    # 1.3. Experiment name and stamp
    parser.add_argument('-experiment_name', default="MyPTSDExp", help='Experiment name.')
    parser.add_argument('-stamp', default='PTSD_binary_setup', type=str,
                        help='Extra label to differentiate experiments.')

    # 1.4. Load checkpoint or not?
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint? 1=yes, 0=no')

    # 1.5. Debug or not?
    parser.add_argument('-debug', default=0, type=int,
                        help='If >0, load only a few samples for debugging.')

    # 1.6. Modality
    # Example: we keep only 'video' (visual frames) and 'logmel' (audio) for a binary classification pipeline.
    parser.add_argument('-modality', default=['video', 'logmel'], nargs="*",
                        help='Which modalities to use? (audio+visual)')

    # Calculate mean and std for each modality?
    parser.add_argument('-calc_mean_std', default=0, type=int,
                        help='Whether to compute and save mean/std for each modality (0=no, 1=yes)')

    # 1.7. We remove the 'emotion' argument and any references to valence/arousal.
    # If you do need an "emotion" placeholder, you can keep it but it won't be used for continuous tasks.

    # 1.8. Whether to save the models?
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save models? 1=yes')

    # 2. Training settings.
    parser.add_argument('-num_heads', default=2, type=int, help='Number of transformer heads (if used).')
    parser.add_argument('-modal_dim', default=32, type=int, help='Dim of modal transformer embedding.')
    parser.add_argument('-tcn_kernel_size', default=5, type=int, help='Kernel size for TCN.')

    # 2.1. Overall settings
    parser.add_argument('-model_name', default="RCMA", help='Choose the model: RCMA or CAN, etc.')
    parser.add_argument('-cross_validation', default=1, type=int,
                        help='Whether to use cross-validation (1=yes).')
    parser.add_argument('-num_folds', default=6, type=int, help='Number of CV folds.')
    parser.add_argument('-folds_to_run', default=[1], nargs="+", type=int,
                        help='Which fold(s) to run? e.g. [1], [1,2], etc.')

    # 2.2. Epochs and data
    parser.add_argument('-num_epochs', default=100, type=int, help='Max epochs to train.')
    parser.add_argument('-min_num_epochs', default=5, type=int, help='Minimum epoch count before early stopping.')
    parser.add_argument('-early_stopping', default=50, type=int,
                        help='Stop if no improvement after X epochs.')
    parser.add_argument('-window_length', default=300, type=int, help='Window length for data slicing.')
    parser.add_argument('-hop_length', default=200, type=int, help='Stride for window slicing.')
    parser.add_argument('-batch_size', default=12, type=int, help='Batch size.')

    # 2.3. Scheduler / Param Control
    parser.add_argument('-seed', default=3407, type=int)
    parser.add_argument('-scheduler', default='plateau', type=str, help='plateau or cosine.')
    parser.add_argument('-learning_rate', default=1e-5, type=float, help='Initial LR.')
    parser.add_argument('-min_learning_rate', default=1e-8, type=float, help='Min LR.')
    parser.add_argument('-patience', default=5, type=int, help='Patience for LR changes.')
    parser.add_argument('-factor', default=0.1, type=float, help='Factor for LR decrease.')
    parser.add_argument('-gradual_release', default=1, type=int,
                        help='Gradual layer unfreezing? 1=yes.')
    parser.add_argument('-release_count', default=3, type=int,
                        help='How many layer groups to unfreeze.')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int,
                        help='Epochs to unfreeze or do special actions.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int,
                        help='Reload best model state each epoch? 1=yes.')

    # 2.4. Label alignment or time delay
    parser.add_argument('-time_delay', default=0, type=float,
                        help='Shift labels if needed. Probably 0 for binary classification.')

    # 2.5. Metrics for evaluation
    # For binary classification, you might use accuracy, f1, etc.
    parser.add_argument('-metrics', default=["accuracy", "f1", "kappa"], nargs="*",
                        help='Evaluation metrics for binary classification.')

    parser.add_argument('-save_plot', default=0, type=int,
                        help='Whether to plot outputs/targets (mostly for regression). 0=no.')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    # Import your custom Experiment class
    from experiment import Experiment

    exp = Experiment(args)
    exp.prepare()
    exp.run()
