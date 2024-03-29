import time
import numpy as np

from research_paper.methods.mace import normalizedDistance

from random import seed
RANDOM_SEED = 1122334455
seed(RANDOM_SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(RANDOM_SEED)


def findClosestObservableSample(model, potential_observable_samples, dataset_obj, factual_sample, norm_type):

  closest_observable_sample = {'sample': {}, 'distance': np.infty, 'norm_type': None} # in case no observables are found.

  for idx, observable_sample in potential_observable_samples[model.predict(potential_observable_samples) != factual_sample['y']].iterrows():

    observable_sample = observable_sample.to_dict()
    observable_sample['y'] = (factual_sample['y'] + 1) % 2

    # Only compare against those observable samples that DO NOT differ with the
    # factual sample in non-actionable features!
    violating_actionable_attributes = False
    for attr_name_kurz in dataset_obj.getInputAttributeNames('kurz'):
      attr_obj = dataset_obj.attributes_kurz[attr_name_kurz]
      if attr_obj.actionability == 'none' and factual_sample[attr_name_kurz] != observable_sample[attr_name_kurz]:
        violating_actionable_attributes = True
      elif attr_obj.actionability == 'same-or-increase' and factual_sample[attr_name_kurz] > observable_sample[attr_name_kurz]:
        violating_actionable_attributes = True
      elif attr_obj.actionability == 'same-or-decrease' and factual_sample[attr_name_kurz] < observable_sample[attr_name_kurz]:
        violating_actionable_attributes = True

    observable_distance = normalizedDistance.getDistanceBetweenSamples(factual_sample, observable_sample, norm_type, dataset_obj)

    if violating_actionable_attributes:
      continue

    if observable_distance < closest_observable_sample['distance']:
      closest_observable_sample = {'sample': observable_sample, 'distance': observable_distance, 'norm_type': norm_type}

  return closest_observable_sample


def getPrettyStringForSampleDictionary(sample, dataset_obj):

  if len(sample.keys()) == 0 :
    return 'No sample found.'

  key_value_pairs_with_x_in_key = {}
  key_value_pairs_with_y_in_key = {}
  for key, value in sample.items():
    if key in dataset_obj.getInputAttributeNames('kurz'):
      key_value_pairs_with_x_in_key[key] = value
    elif key in dataset_obj.getOutputAttributeNames('kurz'):
      key_value_pairs_with_y_in_key[key] = value
    else:
      raise Exception('Sample keys may only be `x` or `y`.')

  assert \
    len(key_value_pairs_with_y_in_key.keys()) == 1, \
    f'expecting only 1 output variables, got {len(key_value_pairs_with_y_in_key.keys())}'

  all_key_value_pairs = []
  for key, value in sorted(key_value_pairs_with_x_in_key.items(), key = lambda x: int(x[0][1:].split('_')[0])):
    all_key_value_pairs.append(f'{key} : {value}')
  all_key_value_pairs.append(f"{'y'}: {key_value_pairs_with_y_in_key['y']}")

  return f"{{{', '.join(all_key_value_pairs)}}}"


def genExp(
  explanation_file,
  model_trained,
  dataset_obj,
  factual_sample,
  potential_observable_samples,
  norm_type):

  start_time = time.time()

  log_file = explanation_file

  factual_sample_dict = factual_sample.to_dict()
  factual_sample_dict['y'] = model_trained.predict([factual_sample])[0]

  closest_observable_sample = findClosestObservableSample(
    model_trained,
    potential_observable_samples,
    dataset_obj,
    factual_sample_dict,
    norm_type
  )

  print('\n', file=log_file)
  print(f"Factual sample: \t\t {getPrettyStringForSampleDictionary(factual_sample_dict, dataset_obj)}", file=log_file)
  print(f"Best observable sample: \t {getPrettyStringForSampleDictionary(closest_observable_sample['sample'], dataset_obj)} (verified)", file=log_file)
  print('', file=log_file)
  print(f"Minimum observable distance (by searching the dataset):\t {closest_observable_sample['distance']:.6f}", file=log_file)

  end_time = time.time()

  return {
    'factual_sample': factual_sample_dict,
    'counterfactual_sample': closest_observable_sample['sample'],
    'counterfactual_found': True,
    'counterfactual_plausible': True,
    'counterfactual_distance': closest_observable_sample['distance'],
    'counterfactual_time': end_time - start_time,
  }





