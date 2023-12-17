r"""Hyperparameter search for training baseline models on tile dataset."""

from collections.abc import Sequence
import json
import time
import os
from absl import app
from absl import flags
import apache_beam as beam
from tpu_graphs.baselines.tiles import train_args
from tpu_graphs.baselines.tiles import train_lib



_DEBUG = flags.DEFINE_bool(
    'debug', False, 'If set, only 2 jobs will be ran')

_BASE_ARGS = {
    'out_dir': '~/out/tpugraphs_tiles/beam',
    'test_mode': 'predictions',
    'validate_batches': 50,
    'eval_every': 2,
    'early_stop': 10,  # Stops after 10 epochs if val OPA does not improve.
    'epochs': 500,
    'configs': 8,
}


def get_parameter_sweep_1() -> list[train_args.TrainArgs]:
  """Hyperparameter sweep for main paper submission."""
  batch_sizes = [10]
  learning_rates = [1e-2]
  clip_norms = [1e-3]
  run_ids = [0]
  losses = ['ListMLELoss:1']
  gnn_models = ('EarlyJoinSAGE', 'EarlyJoinResGCN')
  gnn_args = [
      {'num_gnns': 2, 'op_embed_dim': 64, 'hidden_dim': 64},
      # {'num_gnns': 3, 'op_embed_dim': 64, 'hidden_dim': 64},
      {'num_gnns': 3, 'op_embed_dim': 128, 'hidden_dim': 128},
      # {'num_gnns': 3, 'op_embed_dim': 256, 'hidden_dim': 128},
      {'num_gnns': 4, 'op_embed_dim': 256, 'hidden_dim': 128},
  ]
  model_and_args_list = [
      # MLP
      ('MLP', {'op_embed_dim': 64, 'hidden_dim': 64, 'mlp_layers': 2}),
      ('MLP', {'op_embed_dim': 64, 'hidden_dim': 64, 'mlp_layers': 2}),
      # ('MLP', {'op_embed_dim': 128, 'hidden_dim': 128, 'mlp_layers': 2}),
      # ('MLP', {'op_embed_dim': 128, 'hidden_dim': 128, 'mlp_layers': 3}),
      # ('MLP', {'op_embed_dim': 256, 'hidden_dim': 128, 'mlp_layers': 3}),
  ]

  for gnn_model in gnn_models:
    for gnn_arg in gnn_args:
      model_and_args_list.append((gnn_model, gnn_arg))

  all_args: list[train_args.TrainArgs] = []
  for model, model_args in model_and_args_list:
    for loss in losses:
      for run_id in run_ids:
        for clip_norm in clip_norms:
          for lr in learning_rates:
            for batch in batch_sizes:
              args = dict(_BASE_ARGS)  # copy.
              args['results_csv'] = os.path.join(os.path.expanduser(args['out_dir']), f'results_{int(time.time() * 1000)}.csv')
              args['model'] = model
              args['model_kwargs_json'] = json.dumps(model_args)
              args['losses'] = loss
              args['clip_norm'] = clip_norm
              args['learning_rate'] = lr
              args['run_id'] = run_id
              args['batch_size'] = batch
              all_args.append(train_args.TrainArgs(**args))
  print(len(all_args))
  return all_args


def _run_args(args):
  train_lib.train(train_args.TrainArgs(**args))


def pipeline(root) -> beam.pvalue.PCollection:
  train_arg_list = get_parameter_sweep_1()
  if _DEBUG.value:
    train_arg_list = train_arg_list[:2]
  train_arg_list = [args._asdict() for args in train_arg_list]
  return (
      root
      | beam.Create(train_arg_list)
      | beam.Reshuffle()
      | beam.Map(_run_args))


def main(unused_argv: Sequence[str]) -> None:
  p = beam.Pipeline(); pipeline(p); p.run()


if __name__ == '__main__':
  app.run(main)
