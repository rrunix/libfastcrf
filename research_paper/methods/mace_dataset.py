from research_paper.methods.mace import loadData


def convert_to_mace_dataset(dataset_info, X, y):
    attributes = {}
    output_col = 'y'
    attributes[output_col] = loadData.DatasetAttribute(
        attr_name_long=output_col,
        attr_name_kurz='y',
        attr_type='binary',
        is_input=False,
        actionability='none',
        parent_name_long=-1,
        parent_name_kurz=-1,
        lower_bound=y.min(),
        upper_bound=y.max())

    one_hot = False

    X['y'] = y

    for attr_info in dataset_info.dataset_description.values():
        attr = attr_info['name']
        if attr_info['type'] == 1:
            col_idx = str(attr_info['current_position'])
            attributes[attr] = loadData.DatasetAttribute(
                attr_name_long=attr,
                attr_name_kurz='x' + col_idx,
                attr_type='numeric-real',
                is_input=True,
                actionability='any',
                parent_name_long=-1,
                parent_name_kurz=-1,
                lower_bound=attr_info['lower_bound'],
                upper_bound=attr_info['upper_bound'])
        elif attr_info['type'] in (2, 3):
            col_idx = str(attr_info['current_position'])
            attributes[attr] = loadData.DatasetAttribute(
                attr_name_long=attr,
                attr_name_kurz='x' + col_idx,
                attr_type='numeric-int',
                is_input=True,
                actionability='any',
                parent_name_long=-1,
                parent_name_kurz=-1,
                lower_bound=attr_info['lower_bound'],
                upper_bound=attr_info['upper_bound'])
        else:
            col_idx = str(attr_info['categories_original_position'][0])
            attr_name_long = attr,
            attr_name_kurz = 'x' + col_idx
            one_hot = True

            for new_col_name_kurz, new_col_name_long in zip(attr_info['categories_original_position'],
                                                            attr_info['category_names']):
                attributes[new_col_name_long] = loadData.DatasetAttribute(
                    attr_name_long=new_col_name_long,
                    attr_name_kurz=new_col_name_kurz,
                    attr_type='categorical',
                    is_input=True,
                    actionability='any',
                    parent_name_long=attr_name_long,
                    parent_name_kurz=attr_name_kurz,
                    lower_bound=0,
                    upper_bound=1)

    return loadData.Dataset(X, attributes, one_hot)

