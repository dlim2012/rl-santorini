import os
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser


def read_data(log_dir):
    from tensorflow.python.summary.summary_iterator import summary_iterator
    algo_dirs = os.listdir(log_dir)
    data = []
    for dir_ in algo_dirs:
        splits = dir_.split('_')
        algo_name = splits[0]
        mode = splits[1]
        run_dir = os.path.join(log_dir, dir_)
        runs = os.listdir(run_dir)
        for run in runs:
            lr, idx = run.split('_')
            lr = float(lr[2:])
            idx = int(idx[3:])
            event_files = os.listdir(os.path.join(run_dir, run))
            for event_file in event_files:
                tf_file_path = os.path.join(run_dir, run, event_file)
                for e in summary_iterator(tf_file_path):
                    for v in e.summary.value:
                        data_item = {'algorithm': algo_name, 'train_mode': mode, 'lr':lr, 'run_idx':idx, 'wall_time':e.wall_time, 'step':e.step, 'tag':v.tag, 'value':v.simple_value}
                        data.append(data_item)

    df = pd.DataFrame(data)
    df.to_csv("logs_data.csv")
    return df


def main(args):
    if args.data_file is not None:
        df = pd.read_csv(args.data_file)
    else:
        log_dir = args.log_dir
        df = read_data(log_dir)
    grid = sns.FacetGrid(df[(df['train_mode']=='learn') & ~(df['tag'].str.contains("invalid_action"))], col='algorithm', row='tag', hue='lr', height=5)
    grid.map(sns.lineplot, "step", "value")
    grid.set_titles(row_template="Opponent:{row_name}")
    grid.add_legend()
    grid.set_ylabels("reward")
    grid.tight_layout()
    grid.savefig('learning_curves_learn.svg')

    grid = sns.FacetGrid(df[(df['train_mode']=='self') & ~(df['tag'].str.contains("invalid_action"))], col='algorithm', row='tag', hue='lr', height=5)
    grid.map(sns.lineplot, "step", "value")
    grid.set_titles(row_template="Opponent:{row_name}")
    grid.add_legend()
    grid.set_ylabels("reward")
    grid.tight_layout()
    grid.savefig('learning_curves_self.svg')

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--log-dir', '-ld', help="Path to tensorboard event files", required=False)
    parser.add_argument('--data-file', '-df', help="Extracted data in csv file", required=False)
    args = parser.parse_args()
    if args.log_dir is None and args.data_file is None:
        print("One of log-dir or data-file must be specified")
        exit(1)
    main(args)