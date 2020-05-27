# Experiments
The counterfactuals extracted for each dataset and methods are located in research_paper/experiments/{dataset}/{cv}/{method}/sample_{observation_id}

Setup:
  * Paper enviroment: research_paper.yml
  * LORE environment: lore.yml  (python 2)
  * Activate paper-env and compile rf-ocse: python setup.py build_ext --inplace

Add new datasets for in experiments:
  * Add the dataset in research_paper/dataset_reader
  * Create the folder structure for the dataset with: python -m research_paper.create_experiments
  * Launch the experiments
  
Add new methods:
  * Add the method in research_paper/launch_experiments (method run)

Launch the experiments:
  * Add the desired methods in research_paper/launch_experiments
  * Notice that LORE uses a distinct environment than the other methods
  * python -m research_paper.launch_experiments

To reproduce the tables in the paper (using the provided environment):
  * Table 1: python -m research_paper.create_paper_datasets_table
  * Table 2: python -m research_paper.create_paper_counterfactual_distances_table
  * Table 3: python -m research_paper.create_paper_counterfactual_example_table
  * Table 5: python -m research_paper.create_paper_counterfactual_sets_table
  * Table 6: python -m research_paper.create_paper_counterfactual_sets_example_table
  * Table 7: python -m research_paper.create_paper_rf_ocse_pseudo_counterfactuals_table
