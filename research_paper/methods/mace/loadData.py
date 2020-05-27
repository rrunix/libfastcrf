import os
import sys
import copy
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint

sys.path.insert(0, '_data_main')
from research_paper.methods.mace._data_main.fair_adult_data import *
# from _data_main.fair_compas_data import *
# from _data_main.process_credit_data import *

# import sys
# sys.path.insert(0, '_data_main')
# from _data_main import loadData
# import loadData
# loadData.loadDataset('adult', False, False).printDataset()
# loadData.loadDataset('adult', True, False).printDataset()
# loadData.loadDataset('compass', False, False).printDataset()
# loadData.loadDataset('compass', True, False).printDataset()
# loadData.loadDataset('credit', False, False).printDataset()
# loadData.loadDataset('credit', True, False).printDataset()


VALID_ATTRIBUTE_TYPES = { \
  'numeric-int', \
  'numeric-real', \
  'binary', \
  'categorical', \
  'sub-categorical', \
  'ordinal', \
  'sub-ordinal'}
VALID_IS_INPUT_TYPES = {True, False}
VALID_ACTIONABILITY_TYPES = { \
  'none', \
  'any', \
  'same-or-increase', \
  'same-or-decrease', \
}
# attributes

class Dataset(object):

  # TODO: getOneHotEquivalent can be a class method, and this object can store
  # both one-hot and non-hot versions!

  def __init__(self, data_frame, attributes, is_one_hot):

    self.is_one_hot = is_one_hot

    attributes_long = attributes
    data_frame_long = data_frame
    self.data_frame_long = data_frame_long # i.e., data_frame is indexed by attr_name_long
    self.attributes_long = attributes_long # i.e., attributes is indexed by attr_name_long
    attributes_kurz = dict((attributes[key].attr_name_kurz, value) for (key, value) in attributes_long.items())
    data_frame_kurz = copy.deepcopy(data_frame_long)
    data_frame_kurz.columns = self.getInputOutputAttributeNames('kurz')
    self.data_frame_kurz = data_frame_kurz # i.e., data_frame is indexed by attr_name_kurz
    self.attributes_kurz = attributes_kurz # i.e., attributes is indexed by attr_name_kurz
    self.long_kurz_mapping = {long_name: value.attr_name_kurz for long_name, value in attributes.items()}

    # assert that data_frame and attributes match on variable names (long)
    assert len(np.setdiff1d(
      data_frame.columns.values,
      np.array(self.getInputOutputAttributeNames('long'))
    )) == 0

    # assert attributes and is_one_hot agree on one-hot-ness
    tmp_is_one_hot = False
    for attr_name in attributes.keys():
      attr_obj = attributes[attr_name]
      if attr_obj.parent_name_long != -1 or attr_obj.parent_name_kurz != -1 :
        tmp_is_one_hot = True
    assert is_one_hot == tmp_is_one_hot, "Dataset object and actual attributes don't agree on one-hot"

    self.assertSiblingsShareAttributes('long')
    self.assertSiblingsShareAttributes('kurz')

  def getInputOutputAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check is_input and attr_name,
    # doesn't matter if we loop through self.attributes_long or
    # self.attributes_kurz as they share the same values.
    for attr_name in self.data_frame_long.columns:
      attr_obj = self.attributes_long[attr_name]
      if long_or_kurz == 'long':
        names.append(attr_obj.attr_name_long)
      elif long_or_kurz == 'kurz':
        names.append(attr_obj.attr_name_kurz)
      else:
        raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getInputAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_name, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name in self.attributes_long.keys():
      attr_obj = self.attributes_long[attr_name]
      if attr_obj.is_input == False:
        continue
      if long_or_kurz == 'long':
        names.append(attr_obj.attr_name_long)
      elif long_or_kurz == 'kurz':
        names.append(attr_obj.attr_name_kurz)
      else:
        raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getOutputAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputOutputAttributeNames(long_or_kurz)
    b = self.getInputAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getBinaryAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check binary, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.is_input and attr_obj.attr_type == 'binary':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getDiscreteAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check actionability, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.is_input and attr_obj.attr_type in {'binary', 'categorical'}:
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getOrdinalAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check actionability, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.is_input and attr_obj.attr_type == 'ordinal':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)


  def getActionableAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check actionability, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.is_input and attr_obj.actionability != 'none':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getNonActionableAttributeNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getActionableAttributeNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def getIntegerBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-int':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def getRealBasedAttributeNames(self, long_or_kurz = 'kurz'):
    names = []
    # We must loop through all attributes and check attr_type, doesn't matter
    # if we loop through self.attributes_long or self.attributes_kurz as they
    # share the same values.
    for attr_name_long in self.getInputAttributeNames('long'):
      attr_obj = self.attributes_long[attr_name_long]
      if attr_obj.attr_type == 'numeric-real':
        if long_or_kurz == 'long':
          names.append(attr_obj.attr_name_long)
        elif long_or_kurz == 'kurz':
          names.append(attr_obj.attr_name_kurz)
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))
    return np.array(names)

  def assertSiblingsShareAttributes(self, long_or_kurz = 'kurz'):
    # assert elems of dictOfSiblings share attr_type, parent, is_input-ness, and actionable-ness
    dict_of_siblings = self.getDictOfSiblings(long_or_kurz)
    for parent_name in dict_of_siblings['cat'].keys():
      siblings = dict_of_siblings['cat'][parent_name]
      assert len(siblings) > 1
      for sibling in siblings:
        if long_or_kurz == 'long':
          self.attributes_long[sibling].attr_type = self.attributes_long[siblings[0]].attr_type
          self.attributes_long[sibling].is_input = self.attributes_long[siblings[0]].is_input
          self.attributes_long[sibling].actionability = self.attributes_long[siblings[0]].actionability
          self.attributes_long[sibling].parent_name_long = self.attributes_long[siblings[0]].parent_name_long
          self.attributes_long[sibling].parent_name_kurz = self.attributes_long[siblings[0]].parent_name_kurz
        elif long_or_kurz == 'kurz':
          self.attributes_kurz[sibling].attr_type = self.attributes_kurz[siblings[0]].attr_type
          self.attributes_kurz[sibling].is_input = self.attributes_kurz[siblings[0]].is_input
          self.attributes_kurz[sibling].actionability = self.attributes_kurz[siblings[0]].actionability
          self.attributes_kurz[sibling].parent_name_long = self.attributes_kurz[siblings[0]].parent_name_long
          self.attributes_kurz[sibling].parent_name_kurz = self.attributes_kurz[siblings[0]].parent_name_kurz
        else:
          raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))

  def getSiblingsFor(self, attr_name_long_or_kurz):
    # If attr_name_long is given, we will return siblings_long (the same length)
    # but not siblings_kurz. Same for the opposite direction.
    assert \
      'cat' in attr_name_long_or_kurz or 'ord' in attr_name_long_or_kurz, \
      'attr_name must include either `cat` or `ord`.'
    if attr_name_long_or_kurz in self.getInputOutputAttributeNames('long'):
      attr_name_long = attr_name_long_or_kurz
      dict_of_siblings_long = self.getDictOfSiblings('long')
      for parent_name_long in dict_of_siblings_long['cat']:
        siblings_long = dict_of_siblings_long['cat'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
      for parent_name_long in dict_of_siblings_long['ord']:
        siblings_long = dict_of_siblings_long['ord'][parent_name_long]
        if attr_name_long_or_kurz in siblings_long:
          return siblings_long
    elif attr_name_long_or_kurz in self.getInputOutputAttributeNames('kurz'):
      attr_name_kurz = attr_name_long_or_kurz
      dict_of_siblings_kurz = self.getDictOfSiblings('kurz')
      for parent_name_kurz in dict_of_siblings_kurz['cat']:
        siblings_kurz = dict_of_siblings_kurz['cat'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
      for parent_name_kurz in dict_of_siblings_kurz['ord']:
        siblings_kurz = dict_of_siblings_kurz['ord'][parent_name_kurz]
        if attr_name_long_or_kurz in siblings_kurz:
          return siblings_kurz
    else:
      raise Exception('{attr_name_long_or_kurz} not recognized as a valid `attr_name_long_or_kurz`.'.format(attr_name_long_or_kurz=attr_name_long_or_kurz))

  def getDictOfSiblings(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':

      dict_of_siblings_long = {}
      dict_of_siblings_long['cat'] = {}
      dict_of_siblings_long['ord'] = {}

      for attr_name_long in self.getInputAttributeNames('long'):
        attr_obj = self.attributes_long[attr_name_long]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_long not in dict_of_siblings_long['cat'].keys():
            dict_of_siblings_long['cat'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['cat'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_long not in dict_of_siblings_long['ord'].keys():
            dict_of_siblings_long['ord'][attr_obj.parent_name_long] = [] # initiate key-value pair
          dict_of_siblings_long['ord'][attr_obj.parent_name_long].append(attr_obj.attr_name_long)

      # sort sub-arrays
      for key in dict_of_siblings_long['cat'].keys():
        dict_of_siblings_long['cat'][key] = sorted(dict_of_siblings_long['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_long['ord'].keys():
        dict_of_siblings_long['ord'][key] = sorted(dict_of_siblings_long['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_long

    elif long_or_kurz == 'kurz':

      dict_of_siblings_kurz = {}
      dict_of_siblings_kurz['cat'] = {}
      dict_of_siblings_kurz['ord'] = {}

      for attr_name_kurz in self.getInputAttributeNames('kurz'):
        attr_obj = self.attributes_kurz[attr_name_kurz]
        if attr_obj.attr_type == 'sub-categorical':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['cat'].keys():
            dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['cat'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)
        elif attr_obj.attr_type == 'sub-ordinal':
          if attr_obj.parent_name_kurz not in dict_of_siblings_kurz['ord'].keys():
            dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz] = [] # initiate key-value pair
          dict_of_siblings_kurz['ord'][attr_obj.parent_name_kurz].append(attr_obj.attr_name_kurz)

      # sort sub-arrays
      for key in dict_of_siblings_kurz['cat'].keys():
        dict_of_siblings_kurz['cat'][key] = sorted(dict_of_siblings_kurz['cat'][key], key = lambda x : int(x.split('_')[-1]))

      for key in dict_of_siblings_kurz['ord'].keys():
        dict_of_siblings_kurz['ord'][key] = sorted(dict_of_siblings_kurz['ord'][key], key = lambda x : int(x.split('_')[-1]))

      return dict_of_siblings_kurz

    else:

      raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))

  def getOneHotAttributesNames(self, long_or_kurz = 'kurz'):
    tmp = self.getDictOfSiblings(long_or_kurz)
    names = []
    for key1 in tmp.keys():
      for key2 in tmp[key1].keys():
        names.extend(tmp[key1][key2])
    return np.array(names)

  def getNonHotAttributesNames(self, long_or_kurz = 'kurz'):
    a = self.getInputAttributeNames(long_or_kurz)
    b = self.getOneHotAttributesNames(long_or_kurz)
    return np.setdiff1d(a,b)

  def printDataset(self, long_or_kurz = 'kurz'):
    if long_or_kurz == 'long':
      for attr_name_long in self.attributes_long:
        print(self.attributes_long[attr_name_long].__dict__)
    elif long_or_kurz == 'kurz':
      for attr_name_kurz in self.attributes_kurz:
        print(self.attributes_kurz[attr_name_kurz].__dict__)
    else:
      raise Exception('{long_or_kurz} not recognized as a valid `long_or_kurz`.'.format(long_or_kurz=long_or_kurz))


class DatasetAttribute(object):

  def __init__(
    self,
    attr_name_long,
    attr_name_kurz,
    attr_type,
    is_input,
    actionability,
    parent_name_long,
    parent_name_kurz,
    lower_bound,
    upper_bound):

    if attr_type not in VALID_ATTRIBUTE_TYPES:
      raise Exception("`attr_type` must be one of %r." % VALID_ATTRIBUTE_TYPES)

    if is_input not in VALID_IS_INPUT_TYPES:
      raise Exception("`is_input` must be one of %r." % VALID_IS_INPUT_TYPES)

    if actionability not in VALID_ACTIONABILITY_TYPES:
      raise Exception("`actionability` must be one of %r." % VALID_ACTIONABILITY_TYPES)

    if lower_bound > upper_bound:
      raise Exception("`lower_bound` must be <= `upper_bound`")

    if attr_type in {'sub-categorical', 'sub-ordinal'}:
      assert parent_name_long != -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz != -1, 'Parent ID set for non-hot attribute.'
      if attr_type == 'sub-categorical':
        assert lower_bound == 0
        assert upper_bound == 1
      if attr_type == 'sub-ordinal':
        # the first elem in thermometer is always on, but the rest may be on or off
        assert lower_bound == 0 or lower_bound == 1
        assert upper_bound == 1
    else:
      assert parent_name_long == -1, 'Parent ID set for non-hot attribute.'
      assert parent_name_kurz == -1, 'Parent ID set for non-hot attribute.'

    if attr_type in {'categorical', 'ordinal'}:
      assert lower_bound == 1 # setOneHotValue & setThermoValue assume this in their logic

    if attr_type in {'binary', 'categorical', 'sub-categorical'}: # not 'ordinal' or 'sub-ordinal'
      # IMPORTANT: surprisingly, it is OK if all sub-ordinal variables share actionability
      #            think about it, if each sub- variable is same-or-increase, along with
      #            the constraints that x0_ord_1 >= x0_ord_2, all variables can only stay
      #            the same or increase. It works :)
      assert actionability in {'none', 'any'}, "{attr_type}'s actionability can only be in {'none', 'any'}, not `{actionability}`.".format(attr_type=attr_type, actionability=actionability)

    if not is_input:
      assert actionability == 'none', 'Output attribute is not actionable.'

    if parent_name_long == -1 or parent_name_kurz == -1:
      assert parent_name_long == parent_name_kurz == -1

    self.attr_name_long = attr_name_long
    self.attr_name_kurz = attr_name_kurz
    self.attr_type = attr_type
    self.is_input = is_input
    self.actionability = actionability
    self.parent_name_long = parent_name_long
    self.parent_name_kurz = parent_name_kurz
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound


def getInputOutputColumns(data_frame):
  all_data_frame_cols = data_frame.columns.values
  input_cols = [x for x in all_data_frame_cols if 'label' not in x.lower()]
  output_cols = [x for x in all_data_frame_cols if 'label' in x.lower()]
  assert len(output_cols) == 1
  return input_cols, output_cols[0]


def loadDataset(dataset_name, return_one_hot, load_from_cache = True, debug_flag = False):

  one_hot_string = 'one_hot' if return_one_hot else 'non_hot'
  save_file_path = os.path.join(
    os.path.dirname(__file__),
    '_data_main/_cached/{dataset_name}_{one_hot_string}'.format(dataset_name=dataset_name, one_hot_string=one_hot_string)
  )

  # load from cache
  # if load_from_cache:
  #   #if debug_flag: print('[INFO] Attempting to load saved dataset (`{dataset_name}`) from cache...\t'.format(dataset_name=dataset_name), end = '')
  #   try:
  #     tmp = pickle.load(open(save_file_path, 'rb'))
  #     if debug_flag: print('done.')
  #     return tmp
  #   except:
  #     if debug_flag: print('failed. Re-creating dataset...')

  if dataset_name == 'adult':

    data_frame_non_hot = load_adult_data_new()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}


    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      is_input = False,
      actionability = 'none',
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'Sex':
        attr_type = 'binary'
        actionability = 'any' # 'none'
      elif col_name == 'Age':
        attr_type = 'numeric-int'
        actionability = 'any' # 'none'
      elif col_name == 'NativeCountry': #~ RACE
        attr_type = 'binary'
        actionability = 'any' # 'none'
      elif col_name == 'WorkClass':
        attr_type = 'categorical'
        actionability = 'any'
      elif col_name == 'EducationNumber':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'EducationLevel':
        attr_type = 'ordinal'
        actionability = 'any'
      elif col_name == 'MaritalStatus':
        attr_type = 'categorical'
        actionability = 'any'
      elif col_name == 'Occupation':
        attr_type = 'categorical'
        actionability = 'any'
      elif col_name == 'Relationship':
        attr_type = 'categorical'
        actionability = 'any'
      elif col_name == 'CapitalGain':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'CapitalLoss':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'HoursPerWeek':
        attr_type = 'numeric-int'
        actionability = 'any'

      print("ADULT ", 'x{col_idx}'.format(col_idx=col_idx), col_name)
      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = 'x{col_idx}'.format(col_idx=col_idx),
        attr_type = attr_type,
        is_input = True,
        actionability = actionability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())
      

  elif dataset_name == 'credit':

    data_frame_non_hot = load_credit_data()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      is_input = False,
      actionability = 'none',
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'isMale':
        attr_type = 'binary'
        actionability = 'any' # 'none'
      elif col_name == 'isMarried':
        attr_type = 'binary'
        actionability = 'any'
      elif col_name == 'AgeGroup':
        attr_type = 'ordinal'
        actionability = 'any' # 'none'
      elif col_name == 'EducationLevel':
        attr_type = 'ordinal'
        actionability = 'any'
      elif col_name == 'MaxBillAmountOverLast6Months':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'MaxPaymentAmountOverLast6Months':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'MonthsWithZeroBalanceOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'MonthsWithLowSpendingOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'MonthsWithHighSpendingOverLast6Months':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'MostRecentBillAmount':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'MostRecentPaymentAmount':
        attr_type = 'numeric-real'
        actionability = 'any'
      elif col_name == 'TotalOverdueCounts':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'TotalMonthsOverdue':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'HasHistoryOfOverduePayments':
        attr_type = 'binary'
        actionability = 'any'

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = 'x{col_idx}'.format(col_idx=col_idx),
        attr_type = attr_type,
        is_input = True,
        actionability = actionability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  elif dataset_name == 'compass':

    data_frame_non_hot = load_compas_data_new()
    data_frame_non_hot = data_frame_non_hot.reset_index(drop=True)
    attributes_non_hot = {}

    input_cols, output_col = getInputOutputColumns(data_frame_non_hot)

    col_name = output_col
    attributes_non_hot[col_name] = DatasetAttribute(
      attr_name_long = col_name,
      attr_name_kurz = 'y',
      attr_type = 'binary',
      is_input = False,
      actionability = 'none',
      parent_name_long = -1,
      parent_name_kurz = -1,
      lower_bound = data_frame_non_hot[col_name].min(),
      upper_bound = data_frame_non_hot[col_name].max())

    for col_idx, col_name in enumerate(input_cols):

      if col_name == 'AgeGroup':
        attr_type = 'ordinal'
        actionability = 'any' # 'none'
      elif col_name == 'Race':
        attr_type = 'binary'
        actionability = 'any' # 'none'
      elif col_name == 'Sex':
        attr_type = 'binary'
        actionability = 'any' # 'none'
      elif col_name == 'PriorsCount':
        attr_type = 'numeric-int'
        actionability = 'any'
      elif col_name == 'ChargeDegree':
        attr_type = 'binary'
        actionability = 'any'

      attributes_non_hot[col_name] = DatasetAttribute(
        attr_name_long = col_name,
        attr_name_kurz = 'x{col_idx}'.format(col_idx=col_idx),
        attr_type = attr_type,
        is_input = True,
        actionability = actionability,
        parent_name_long = -1,
        parent_name_kurz = -1,
        lower_bound = data_frame_non_hot[col_name].min(),
        upper_bound = data_frame_non_hot[col_name].max())

  else:

    raise Exception('{dataset_name} not recognized as a valid dataset.'.format(dataset_name=dataset_name))

  if return_one_hot:
    data_frame, attributes = getOneHotEquivalent(data_frame_non_hot, attributes_non_hot)
  else:
    data_frame, attributes = data_frame_non_hot, attributes_non_hot
  print(data_frame.shape)

  # save then return
  dataset_obj = Dataset(data_frame, attributes, return_one_hot)
  # if not loading from cache, we always overwrite the cache
  pickle.dump(dataset_obj, open(save_file_path, 'wb'))
  return dataset_obj


# TODO: consider moving into Dataset class with getOneHot and getNonHot methods
def getOneHotEquivalent(data_frame_non_hot, attributes_non_hot):

  # TODO: see how we can switch between feature_names = col names for kurz and long (also maybe ordered)

  data_frame = copy.deepcopy(data_frame_non_hot)
  attributes = copy.deepcopy(attributes_non_hot)

  def setOneHotValue(val):
    return np.append(np.append(
      np.zeros(val - 1),
      np.ones(1)),
      np.zeros(num_unique_values - val)
    )

  def setThermoValue(val):
    return np.append(
      np.ones(val),
      np.zeros(num_unique_values - val)
    )

  for col_name in data_frame.columns.values:

    if attributes[col_name].attr_type not in {'categorical', 'ordinal'}:
      continue

    old_col_name_long = col_name
    new_col_names_long = []
    new_col_names_kurz = []

    old_attr_name_long = attributes[old_col_name_long].attr_name_long
    old_attr_name_kurz = attributes[old_col_name_long].attr_name_kurz
    old_attr_type = attributes[old_col_name_long].attr_type
    old_is_input = attributes[old_col_name_long].is_input
    old_actionability = attributes[old_col_name_long].actionability
    old_lower_bound = attributes[old_col_name_long].lower_bound
    old_upper_bound = attributes[old_col_name_long].upper_bound

    num_unique_values = int(old_upper_bound - old_lower_bound + 1)

    assert old_col_name_long == old_attr_name_long

    new_attr_type = 'sub-' + old_attr_type
    new_is_input = old_is_input
    new_actionability = old_actionability
    new_parent_name_long = old_attr_name_long
    new_parent_name_kurz = old_attr_name_kurz


    if attributes[col_name].attr_type == 'categorical': # do not do this for 'binary'!

      new_col_names_long = ['{old_attr_name_long}_cat_{i}'.format(old_attr_name_long=old_attr_name_long, i=i) for i in range(num_unique_values)]
      new_col_names_kurz = ['{old_attr_name_kurz}_cat_{i}'.format(old_attr_name_kurz=old_attr_name_kurz, i=i) for i in range(num_unique_values)]
      print('Replacing column {col_name} with {c}'.format(col_name=col_name, c=", ".join(new_col_names_long)))
      tmp = np.array(list(map(setOneHotValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    elif attributes[col_name].attr_type == 'ordinal':

      new_col_names_long = ['{old_attr_name_long}_ord_{i}'.format(old_attr_name_long=old_attr_name_long, i=i) for i in range(num_unique_values)]
      new_col_names_kurz = ['{old_attr_name_kurz}_ord_{i}'.format(old_attr_name_kurz=old_attr_name_kurz, i=i) for i in range(num_unique_values)]
      print('Replacing column {col_name} with {c}'.format(col_name=col_name, c=", ".join(new_col_names_long)))
      tmp = np.array(list(map(setThermoValue, list(data_frame[col_name].astype(int).values))))
      data_frame_dummies = pd.DataFrame(data=tmp, columns=new_col_names_long)

    # Update data_frame
    data_frame = pd.concat([data_frame.drop(columns = old_col_name_long), data_frame_dummies], axis=1)

    # Update attributes
    del attributes[old_col_name_long]
    for col_idx in range(len(new_col_names_long)):
      new_col_name_long = new_col_names_long[col_idx]
      new_col_name_kurz = new_col_names_kurz[col_idx]
      attributes[new_col_name_long] = DatasetAttribute(
        attr_name_long = new_col_name_long,
        attr_name_kurz = new_col_name_kurz,
        attr_type = new_attr_type,
        is_input = new_is_input,
        actionability = new_actionability,
        parent_name_long = new_parent_name_long,
        parent_name_kurz = new_parent_name_kurz,
        lower_bound = data_frame[new_col_name_long].min(),
        upper_bound = data_frame[new_col_name_long].max())

  return data_frame, attributes









