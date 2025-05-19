#!/bin/bash
echo 'This is a CLI script for /home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Evolutionary_Learning_Under_Stress_Using_GFP_Feedback'


# Basic usage examples:
python src/main.py --mode single --preset quick_test
python src/main.py --mode compare --preset standard
python src/main.py --mode sweep --sweep-param mutation_rate --sweep-min 0.1 --sweep-max 5.0
python src/main.py --mode demo  # Quick visual demo



bsub -q gsla-cpu -R rusage[mem=2GB] python src/main.py --mode compare --preset standard
bsub -q gsla-cpu -R rusage[mem=2GB] python src/main.py --mode sweep --sweep-param mutation_rate --sweep-min 0.1 --sweep-max 5.0
bsub -q gsla-cpu -R rusage[mem=2GB] 


(UKbiobank) 16:00:42 🖤 barc@login5:~ > python '/home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Evolutionary_Learning_Under_Stress_Using_GFP_Feedback/src/evolutionary_learning_simulation.py' --experiment quick_test --help
usage: evolutionary_learning_simulation.py [-h]
                                           [--experiment {quick_test,standard_experiment,large_scale}]
                                           [--wells WELLS]
                                           [--generations GENERATIONS]
                                           [--mutation-rate MUTATION_RATE]
                                           [--genome-length GENOME_LENGTH]
                                           [--cells-per-well CELLS_PER_WELL]
                                           [--temperature-feedback TEMPERATURE_FEEDBACK]
                                           [--output OUTPUT] [--seed SEED]
                                           [--no-plots] [--compare-evolution]
                                           [--parameter-sweep PARAMETER_SWEEP]
                                           [--sweep-min SWEEP_MIN]
                                           [--sweep-max SWEEP_MAX]
                                           [--sweep-steps SWEEP_STEPS]

🧪 Yeast Evolutionary Learning Simulation

options:
  -h, --help            show this help message and exit
  --experiment {quick_test,standard_experiment,large_scale}
                        Experimental preset (default: quick_test)
  --wells WELLS         Number of wells (overrides preset)
  --generations GENERATIONS
                        Number of generations (overrides preset)
  --mutation-rate MUTATION_RATE
                        Mutations per genome per generation
  --genome-length GENOME_LENGTH
                        Length of DNA sequence
  --cells-per-well CELLS_PER_WELL
                        Maximum cells per well
  --temperature-feedback TEMPERATURE_FEEDBACK
                        Temperature feedback strength
  --output OUTPUT       Output directory name
  --seed SEED           Random seed (default: 42)
  --no-plots            Skip plot generation
  --compare-evolution   Compare evolution with vs without feedback
  --parameter-sweep PARAMETER_SWEEP
                        Parameter to sweep (wells, mutation_rate, etc.)
  --sweep-min SWEEP_MIN
                        Minimum sweep value
  --sweep-max SWEEP_MAX
                        Maximum sweep value
  --sweep-steps SWEEP_STEPS
                        Number of sweep steps

    🧬 Biological Context:
    This simulation models evolutionary learning in yeast with GFP-sugar metabolism
    construct under shared temperature feedback. Only one well is randomly selected
    each generation to influence global temperature, but all wells experience the
    same environmental conditions.

    Examples:
    python corrected_simulation.py --experiment quick_test
    python corrected_simulation.py --experiment standard_experiment --output my_results
    python corrected_simulation.py --wells 96 --generations 800 --mutation-rate 1.0
        
(UKbiobank) 16:03:33 🖤 barc@login5:~ > 


python '/home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Evolutionary_Learning_Under_Stress_Using_GFP_Feedback/src/evolutionary_learning_simulation.py' --experiment quick_test 
python '/home/labs/pilpel/barc/Evolutionthroughprogramming_2025/Evolutionary_Learning_Under_Stress_Using_GFP_Feedback/src/evolutionary_learning_simulation.py' --experiment  standard_experiment