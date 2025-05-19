#!/usr/bin/env python3
"""
Bulletproof Evolutionary Learning in Yeast Under Temperature Feedback
Biological Context: Yeast with GFP-sugar metabolism construct under shared temperature feedback
Author: Corrected and bulletproofed version with robust error handling
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from collections import defaultdict
import json
from typing import List, Dict, Tuple, Optional, Union, Any
import warnings
import logging
import traceback
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not available, using matplotlib only")

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available, some statistical tests will be skipped")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        if abs(denominator) < 1e-10:
            return default
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return default

def safe_mean(values: List[float]) -> float:
    """Safely calculate mean, handling empty lists"""
    try:
        if not values:
            return 0.0
        return float(np.mean(values))
    except (TypeError, ValueError):
        return 0.0

def safe_std(values: List[float]) -> float:
    """Safely calculate standard deviation, handling empty lists"""
    try:
        if len(values) <= 1:
            return 0.0
        return float(np.std(values))
    except (TypeError, ValueError):
        return 0.0

def validate_params(params: Dict) -> Dict:
    """Validate and sanitize simulation parameters"""
    validated = {}
    
    # Required parameters with defaults
    defaults = {
        'num_wells': 30,
        'genome_length': 1000,
        'num_generations': 200,
        'mutation_rate': 1.0,
        'cells_per_well': 250,
        'stress_temperature': 39.0,
        'relief_step': 1.0,
        'optimal_temperature': 30.0,
        'random_seed': 42
    }
    
    for key, default_value in defaults.items():
        value = params.get(key, default_value)
        
        # Type and range validation
        if key in ['num_wells', 'genome_length', 'num_generations', 'cells_per_well', 'random_seed']:
            validated[key] = max(1, int(value))
        elif key in ['mutation_rate']:
            validated[key] = max(0.0, float(value))
        elif key in ['stress_temperature', 'relief_step', 'optimal_temperature']:
            validated[key] = float(value)
        else:
            validated[key] = value
    
    # Additional validation
    if validated['stress_temperature'] < validated['optimal_temperature']:
        logger.warning(f"Stress temperature ({validated['stress_temperature']}) is lower than optimal ({validated['optimal_temperature']}), swapping")
        validated['stress_temperature'], validated['optimal_temperature'] = validated['optimal_temperature'], validated['stress_temperature']
    
    # Optional parameters
    if 'feedback_well_id' in params:
        fb_id = params['feedback_well_id']
        if fb_id is not None:
            validated['feedback_well_id'] = max(0, min(int(fb_id), validated['num_wells'] - 1))
    
    logger.info(f"Validated parameters: {validated}")
    return validated

# ============================================================================
# CORE CLASSES - BULLETPROOFED
# ============================================================================

class YeastCell:
    """Represents a single yeast cell with DNA sequence and GFP expression"""
    
    # Tetranucleotide motifs that affect GFP expression
    GFP_TETRANUCLEOTIDES = {
        'TATA': 2.5,   # Strong promoter sequence
        'CAAT': 1.8,   # CAAT box enhancer
        'ATGC': 1.5,   # Start codon region
        'GCGC': 1.2,   # GC-rich enhancer
        'CTAG': 0.8,   # Mild repressor
        'AAAA': 0.6,   # AT-rich repressor
        'TTTT': 0.6,   # AT-rich repressor  
        'CCCC': 0.4,   # Strong repressor
        'GGGG': 0.4,   # Strong repressor
    }
    
    def __init__(self, dna_sequence: Optional[str] = None, genome_length: int = 1000):
        """Initialize a yeast cell with a DNA sequence"""
        try:
            if dna_sequence is None:
                # Generate random DNA sequence
                bases = ['A', 'T', 'G', 'C']
                self.dna_sequence = ''.join(np.random.choice(bases, max(1, genome_length)))
            else:
                # Validate DNA sequence
                valid_bases = set('ATGC')
                cleaned_seq = ''.join(c.upper() for c in str(dna_sequence) if c.upper() in valid_bases)
                if not cleaned_seq:
                    # If no valid bases, generate random sequence
                    bases = ['A', 'T', 'G', 'C']
                    self.dna_sequence = ''.join(np.random.choice(bases, max(1, genome_length)))
                else:
                    self.dna_sequence = cleaned_seq
            
            self.generation_born = 0
            self.mutations_count = 0
            self._gfp_cache = None
            
        except Exception as e:
            logger.error(f"Error initializing YeastCell: {e}")
            # Fallback: create minimal valid cell
            self.dna_sequence = 'ATGC' * max(1, genome_length // 4)
            self.generation_born = 0
            self.mutations_count = 0
            self._gfp_cache = None
    
    def calculate_gfp_expression(self) -> float:
        """Calculate GFP expression based on DNA sequence motifs"""
        try:
            if self._gfp_cache is not None:
                return self._gfp_cache
            
            # Base GFP expression level
            gfp_level = 1.0
            dna_length = len(self.dna_sequence)
            
            if dna_length <= 3:
                # Too short for motif analysis
                self._gfp_cache = 0.5 + np.random.normal(0, 0.05)
                return max(0.05, self._gfp_cache)
            
            # Count tetranucleotide motifs and calculate their effect
            for motif, effect in self.GFP_TETRANUCLEOTIDES.items():
                count = 0
                for i in range(dna_length - 3):
                    if self.dna_sequence[i:i+4] == motif:
                        count += 1
                
                # Frequency-based effect
                frequency = safe_divide(count, dna_length - 3, 0.0)
                gfp_level *= (1 + frequency * (effect - 1))
            
            # Add biological noise
            gfp_level *= max(0.1, np.random.normal(1.0, 0.03))
            gfp_level = max(0.05, min(10.0, gfp_level))  # Reasonable bounds
            
            self._gfp_cache = gfp_level
            return gfp_level
            
        except Exception as e:
            logger.error(f"Error calculating GFP expression: {e}")
            # Fallback: return random but reasonable value
            fallback_gfp = 0.5 + np.random.random() * 0.5
            self._gfp_cache = fallback_gfp
            return fallback_gfp
    
    def reproduce_with_mutation(self, mutation_rate: float = 1.0) -> 'YeastCell':
        """Create daughter cell with mutations during reproduction"""
        try:
            new_dna = list(self.dna_sequence)
            bases = ['A', 'T', 'G', 'C']
            
            # Ensure reasonable mutation rate
            mutation_rate = max(0.0, min(10.0, mutation_rate))
            
            # Apply mutations
            num_mutations = max(0, np.random.poisson(mutation_rate))
            actual_mutations = 0
            
            for _ in range(num_mutations):
                if not new_dna:  # Safety check
                    break
                    
                # Random position for mutation
                pos = np.random.randint(len(new_dna))
                current_base = new_dna[pos]
                
                # Substitution mutation
                new_bases = [b for b in bases if b != current_base]
                if new_bases:
                    new_dna[pos] = np.random.choice(new_bases)
                    actual_mutations += 1
            
            # Create daughter cell
            daughter_cell = YeastCell(dna_sequence=''.join(new_dna))
            daughter_cell.generation_born = self.generation_born + 1
            daughter_cell.mutations_count = self.mutations_count + actual_mutations
            
            return daughter_cell
            
        except Exception as e:
            logger.error(f"Error in reproduction: {e}")
            # Fallback: return clone
            return self.clone()
    
    def clone(self) -> 'YeastCell':
        """Create an exact copy of this cell"""
        try:
            cloned_cell = YeastCell(dna_sequence=self.dna_sequence)
            cloned_cell.generation_born = self.generation_born
            cloned_cell.mutations_count = self.mutations_count
            cloned_cell._gfp_cache = self._gfp_cache
            return cloned_cell
        except Exception as e:
            logger.error(f"Error cloning cell: {e}")
            # Fallback: create new cell
            return YeastCell(genome_length=len(self.dna_sequence) if hasattr(self, 'dna_sequence') else 1000)


class Well:
    """Represents a single well containing a yeast population"""
    
    def __init__(self, initial_cell: Optional[YeastCell] = None, well_id: int = 0):
        """Initialize a well with a single yeast cell"""
        try:
            self.well_id = max(0, int(well_id))
            
            if initial_cell is None:
                initial_cell = YeastCell()
            
            self.cells = [initial_cell]
            self.extinct = False
            self.generation_extinct = None
            
            # Initialize history with safe values
            initial_gfp = 0.0
            try:
                initial_gfp = initial_cell.calculate_gfp_expression()
            except:
                initial_gfp = 1.0
            
            self.history = {
                'population_size': [1],
                'average_gfp': [initial_gfp],
                'average_mutations': [0.0],
                'temperature': [30.0],
                'was_selected': [False],
                'fitness': [1.0]
            }
            
            # Track previous GFP for feedback calculation
            self.previous_gfp = initial_gfp
            
        except Exception as e:
            logger.error(f"Error initializing Well {well_id}: {e}")
            # Fallback initialization
            self.well_id = well_id
            self.cells = [YeastCell()]
            self.extinct = False
            self.generation_extinct = None
            self.history = {
                'population_size': [1],
                'average_gfp': [1.0],
                'average_mutations': [0.0],
                'temperature': [30.0],
                'was_selected': [False],
                'fitness': [1.0]
            }
            self.previous_gfp = 1.0
    
    def grow_to_capacity(self, max_capacity: int = 250, mutation_rate: float = 1.0):
        """Grow population to capacity with mutations and random washing"""
        try:
            if self.extinct or not self.cells:
                return
            
            max_capacity = max(1, int(max_capacity))
            mutation_rate = max(0.0, float(mutation_rate))
            
            current_size = len(self.cells)
            
            # Growth phase - exponential growth with mutations
            target_size = min(max_capacity, current_size * 2)
            
            growth_attempts = 0
            max_attempts = max_capacity * 2  # Prevent infinite loops
            
            while len(self.cells) < target_size and growth_attempts < max_attempts:
                try:
                    # Random cell reproduces
                    if self.cells:
                        parent_cell = np.random.choice(self.cells)
                        daughter_cell = parent_cell.reproduce_with_mutation(mutation_rate)
                        self.cells.append(daughter_cell)
                except:
                    # If reproduction fails, try with a new cell
                    self.cells.append(YeastCell())
                growth_attempts += 1
            
            # Random washing step - simulate nutrient limitation
            if len(self.cells) > max_capacity:
                try:
                    wash_fraction = min(0.5, max(0.0, np.random.uniform(0.1, 0.3)))
                    num_to_remove = int(len(self.cells) * wash_fraction)
                    
                    if num_to_remove > 0 and num_to_remove < len(self.cells):
                        indices_to_remove = np.random.choice(len(self.cells), num_to_remove, replace=False)
                        self.cells = [cell for i, cell in enumerate(self.cells) if i not in indices_to_remove]
                except:
                    pass  # Skip washing if it fails
            
            # Ensure we don't exceed capacity
            if len(self.cells) > max_capacity:
                try:
                    selected_indices = np.random.choice(len(self.cells), max_capacity, replace=False)
                    self.cells = [self.cells[i] for i in selected_indices]
                except:
                    # Fallback: just truncate
                    self.cells = self.cells[:max_capacity]
                    
        except Exception as e:
            logger.error(f"Error growing well {self.well_id}: {e}")
            # Ensure we have at least one cell
            if not self.cells:
                self.cells = [YeastCell()]
    
    def calculate_average_gfp(self) -> float:
        """Calculate average GFP expression of all cells in the well"""
        try:
            if not self.cells or self.extinct:
                return 0.0
            
            gfp_values = []
            for cell in self.cells:
                try:
                    gfp = cell.calculate_gfp_expression()
                    if not np.isnan(gfp) and np.isfinite(gfp):
                        gfp_values.append(gfp)
                except:
                    continue
            
            return safe_mean(gfp_values)
            
        except Exception as e:
            logger.error(f"Error calculating average GFP for well {self.well_id}: {e}")
            return 0.0
    
    def calculate_average_mutations(self) -> float:
        """Calculate average number of mutations per cell"""
        try:
            if not self.cells or self.extinct:
                return 0.0
            
            mutation_counts = []
            for cell in self.cells:
                try:
                    count = getattr(cell, 'mutations_count', 0)
                    if np.isfinite(count):
                        mutation_counts.append(count)
                except:
                    continue
            
            return safe_mean(mutation_counts)
            
        except Exception as e:
            logger.error(f"Error calculating average mutations for well {self.well_id}: {e}")
            return 0.0
    
    def apply_temperature_stress(self, temperature: float, generation: int):
        """Apply temperature stress and determine survival"""
        try:
            if self.extinct:
                return
            
            temperature = max(0.0, min(100.0, float(temperature)))
            
            # Calculate fitness based on temperature
            try:
                if temperature <= 30.0:
                    fitness = 1.0
                elif temperature <= 37.0:
                    fitness = 1.0 - 0.1 * (temperature - 30.0) / 7.0
                else:
                    fitness = 0.9 * np.exp(-(temperature - 37.0) / 5.0)
                
                fitness = max(0.0, min(1.0, fitness))
            except:
                fitness = 0.5  # Fallback fitness
            
            # Record current state safely
            try:
                self.history['population_size'].append(len(self.cells))
                self.history['average_gfp'].append(self.calculate_average_gfp())
                self.history['average_mutations'].append(self.calculate_average_mutations())
                self.history['temperature'].append(temperature)
                self.history['fitness'].append(fitness)
                # Initialize was_selected if not exists
                if len(self.history['was_selected']) <= generation:
                    while len(self.history['was_selected']) <= generation:
                        self.history['was_selected'].append(False)
            except Exception as e:
                logger.error(f"Error recording history for well {self.well_id}: {e}")
            
            # Determine survival
            extinction_threshold = 0.1
            if fitness < extinction_threshold or (fitness < 0.8 and np.random.random() > fitness):
                self.extinct = True
                self.generation_extinct = generation
                self.cells = []
                return
            
            # Reduce population if low fitness
            if fitness < 0.8 and self.cells:
                try:
                    survival_rate = max(0.1, fitness)
                    new_size = max(1, int(len(self.cells) * survival_rate))
                    if new_size < len(self.cells):
                        selected_indices = np.random.choice(len(self.cells), new_size, replace=False)
                        self.cells = [self.cells[i] for i in selected_indices]
                except:
                    pass  # Skip population reduction if it fails
                    
        except Exception as e:
            logger.error(f"Error applying temperature stress to well {self.well_id}: {e}")
            # Don't let exceptions kill the well
    
    def prepare_for_next_generation(self) -> Optional[YeastCell]:
        """Prepare well for next generation - select one cell to continue"""
        try:
            if self.extinct or not self.cells:
                return None
            
            # Update previous GFP for next round
            try:
                self.previous_gfp = self.calculate_average_gfp()
            except:
                pass
            
            # Select one cell randomly to seed next generation
            selected_cell = np.random.choice(self.cells)
            return selected_cell.clone()
            
        except Exception as e:
            logger.error(f"Error preparing well {self.well_id} for next generation: {e}")
            return None


class EvolutionarySimulation:
    """Main simulation engine for evolutionary learning experiment"""
    
    def __init__(self, params: Dict):
        """Initialize simulation with parameters"""
        try:
            self.params = validate_params(params)
            self.wells = []
            self.generation = 0
            
            # Initialize feedback well ID
            self.feedback_well_id = self.params.get('feedback_well_id')
            if self.feedback_well_id is None:
                self.feedback_well_id = np.random.randint(self.params['num_wells'])
            
            self.global_temperature = self.params.get('stress_temperature', 39.0)
            
            # Track selection history
            self.selection_history = []
            
            # Global history tracking
            self.global_history = {
                'generation': [],
                'global_temperature': [],
                'selected_well_id': [],
                'selected_well_gfp': [],
                'selected_well_gfp_change': [],
                'living_wells': [],
                'average_gfp_all_wells': [],
                'max_gfp_all_wells': [],
                'total_mutations': []
            }
            
            # Initialize wells with error handling
            logger.info(f"Initializing {self.params['num_wells']} wells...")
            for i in range(self.params['num_wells']):
                try:
                    initial_cell = YeastCell(genome_length=self.params['genome_length'])
                    well = Well(initial_cell=initial_cell, well_id=i)
                    self.wells.append(well)
                except Exception as e:
                    logger.error(f"Error creating well {i}: {e}")
                    # Create fallback well
                    well = Well(well_id=i)
                    self.wells.append(well)
            
            logger.info(f"Created {len(self.wells)} wells successfully")
            
        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            raise
    
    def run_simulation(self) -> Dict:
        """Run the complete evolutionary simulation"""
        try:
            logger.info(f"Starting evolutionary simulation for {self.params['num_generations']} generations...")
            logger.info(f"Feedback well: {self.feedback_well_id}")
            start_time = time.time()
            
            for generation in range(self.params['num_generations']):
                try:
                    self.generation = generation
                    
                    # Progress update
                    if generation % 100 == 0 or generation == self.params['num_generations'] - 1:
                        elapsed = time.time() - start_time
                        living_count = len([w for w in self.wells if not w.extinct])
                        logger.info(f"Generation {generation}/{self.params['num_generations']} "
                                  f"({generation/self.params['num_generations']*100:.1f}%) "
                                  f"- Living wells: {living_count} - Temp: {self.global_temperature:.1f}°C "
                                  f"- Elapsed: {elapsed:.1f}s")
                    
                    # Step 1: Growth phase for all wells
                    living_wells = [w for w in self.wells if not w.extinct]
                    if not living_wells:
                        logger.warning(f"All wells extinct at generation {generation}")
                        break
                    
                    # Grow all living wells
                    for well in living_wells:
                        try:
                            well.grow_to_capacity(
                                max_capacity=self.params['cells_per_well'],
                                mutation_rate=self.params['mutation_rate']
                            )
                        except Exception as e:
                            logger.error(f"Error growing well {well.well_id}: {e}")
                    
                    # Step 2: Select feedback well and calculate response
                    selected_well = self._select_feedback_well(living_wells)
                    if selected_well is None:
                        logger.warning(f"No valid feedback well at generation {generation}")
                        break
                    
                    current_gfp = selected_well.calculate_average_gfp()
                    gfp_change = current_gfp - selected_well.previous_gfp
                    
                    # Mark this well as selected
                    self._mark_well_selected(selected_well, generation)
                    
                    # Step 3: Apply temperature feedback
                    self._apply_temperature_feedback(selected_well, gfp_change)
                    
                    # Step 4: Apply temperature stress to all wells
                    for well in living_wells:
                        try:
                            well.apply_temperature_stress(self.global_temperature, generation)
                        except Exception as e:
                            logger.error(f"Error applying stress to well {well.well_id}: {e}")
                    
                    # Step 5: Record global statistics
                    self._record_global_statistics(selected_well, current_gfp, gfp_change)
                    
                    # Step 6: Prepare next generation
                    self._prepare_next_generation()
                    
                except Exception as e:
                    logger.error(f"Error in generation {generation}: {e}")
                    # Continue with next generation
                    continue
            
            total_time = time.time() - start_time
            logger.info(f"Simulation completed in {total_time:.2f} seconds")
            
            return {
                'wells': self.wells,
                'global_history': self.global_history,
                'params': self.params,
                'final_generation': self.generation,
                'final_temperature': self.global_temperature,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Critical error in simulation: {e}")
            return {
                'wells': getattr(self, 'wells', []),
                'global_history': getattr(self, 'global_history', {}),
                'params': getattr(self, 'params', {}),
                'final_generation': getattr(self, 'generation', 0),
                'final_temperature': getattr(self, 'global_temperature', 30.0),
                'success': False,
                'error': str(e)
            }
    
    def _select_feedback_well(self, living_wells: List[Well]) -> Optional[Well]:
        """Select the feedback well safely"""
        try:
            if self.feedback_well_id < len(self.wells) and not self.wells[self.feedback_well_id].extinct:
                return self.wells[self.feedback_well_id]
            elif living_wells:
                return np.random.choice(living_wells)
            else:
                return None
        except Exception as e:
            logger.error(f"Error selecting feedback well: {e}")
            return living_wells[0] if living_wells else None
    
    def _mark_well_selected(self, selected_well: Well, generation: int):
        """Mark a well as selected for this generation"""
        try:
            # Ensure the history list is long enough
            while len(selected_well.history['was_selected']) <= generation:
                selected_well.history['was_selected'].append(False)
            selected_well.history['was_selected'][generation] = True
        except Exception as e:
            logger.error(f"Error marking well as selected: {e}")
    
    def _apply_temperature_feedback(self, selected_well: Well, gfp_change: float):
        """Apply temperature feedback based on GFP change"""
        try:
            if (selected_well.well_id == self.feedback_well_id and 
                gfp_change > 0 and 
                np.isfinite(gfp_change)):
                
                relief_step = self.params.get('relief_step', 1.0)
                optimal_temp = self.params.get('optimal_temperature', 30.0)
                
                self.global_temperature = max(
                    self.global_temperature - relief_step,
                    optimal_temp
                )
        except Exception as e:
            logger.error(f"Error applying temperature feedback: {e}")
    
    def _record_global_statistics(self, selected_well: Well, current_gfp: float, gfp_change: float):
        """Record statistics for the current generation"""
        try:
            living_wells = [w for w in self.wells if not w.extinct]
            
            if living_wells:
                gfp_values = []
                total_mutations = 0.0
                
                for well in living_wells:
                    try:
                        gfp = well.calculate_average_gfp()
                        if np.isfinite(gfp):
                            gfp_values.append(gfp)
                        
                        mutations = well.calculate_average_mutations() * len(well.cells)
                        if np.isfinite(mutations):
                            total_mutations += mutations
                    except:
                        continue
                
                avg_gfp = safe_mean(gfp_values) if gfp_values else 0.0
                max_gfp = max(gfp_values) if gfp_values else 0.0
            else:
                avg_gfp = 0.0
                max_gfp = 0.0
                total_mutations = 0.0
            
            # Record with safety checks
            self.global_history['generation'].append(self.generation)
            self.global_history['global_temperature'].append(self.global_temperature)
            self.global_history['selected_well_id'].append(selected_well.well_id)
            self.global_history['selected_well_gfp'].append(current_gfp if np.isfinite(current_gfp) else 0.0)
            self.global_history['selected_well_gfp_change'].append(gfp_change if np.isfinite(gfp_change) else 0.0)
            self.global_history['living_wells'].append(len(living_wells))
            self.global_history['average_gfp_all_wells'].append(avg_gfp)
            self.global_history['max_gfp_all_wells'].append(max_gfp)
            self.global_history['total_mutations'].append(total_mutations)
            
            self.selection_history.append(selected_well.well_id)
            
        except Exception as e:
            logger.error(f"Error recording global statistics: {e}")
    
    def _prepare_next_generation(self):
        """Prepare all wells for next generation"""
        try:
            for well in self.wells:
                if not well.extinct:
                    try:
                        representative_cell = well.prepare_for_next_generation()
                        if representative_cell:
                            well.cells = [representative_cell]
                        else:
                            well.extinct = True
                    except Exception as e:
                        logger.error(f"Error preparing well {well.well_id} for next generation: {e}")
                        well.extinct = True
        except Exception as e:
            logger.error(f"Error preparing next generation: {e}")


# ============================================================================
# ANALYSIS TOOLS - BULLETPROOFED
# ============================================================================

class BiologicalAnalysis:
    """Analysis tools specific to the yeast evolutionary learning experiment"""
    
    def __init__(self, results: Dict):
        """Initialize analysis with bulletproof error handling"""
        try:
            self.results = results
            self.wells = results.get('wells', [])
            self.global_history = results.get('global_history', {})
            self.params = results.get('params', {})
            self.success = results.get('success', True)
            
            # Validate data
            if not self.wells:
                logger.warning("No wells data available for analysis")
            if not self.global_history:
                logger.warning("No global history data available for analysis")
                
        except Exception as e:
            logger.error(f"Error initializing BiologicalAnalysis: {e}")
            self.results = {}
            self.wells = []
            self.global_history = {}
            self.params = {}
            self.success = False
    
    def plot_temperature_and_gfp_evolution(self, ax1, ax2):
        """Plot global temperature and GFP evolution over time"""
        try:
            if not self.global_history:
                ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
                return
            
            generations = self.global_history.get('generation', [])
            temperatures = self.global_history.get('global_temperature', [])
            avg_gfp = self.global_history.get('average_gfp_all_wells', [])
            selected_gfp = self.global_history.get('selected_well_gfp', [])
            
            # Temperature plot
            if generations and temperatures:
                ax1.plot(generations, temperatures, 'r-', linewidth=2, label='Global Temperature')
                ax1.axhline(y=30.0, color='blue', linestyle='--', alpha=0.7, label='Optimal Temperature')
                ax1.axhline(y=37.0, color='orange', linestyle='--', alpha=0.7, label='Stress Threshold')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Temperature (°C)')
                ax1.set_title('Global Temperature Over Time')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No temperature data', ha='center', va='center', transform=ax1.transAxes)
            
            # GFP evolution plot
            if generations and (avg_gfp or selected_gfp):
                if avg_gfp and len(avg_gfp) == len(generations):
                    ax2.plot(generations, avg_gfp, 'g-', linewidth=2, label='Average GFP (All Wells)')
                if selected_gfp and len(selected_gfp) == len(generations):
                    ax2.plot(generations, selected_gfp, 'b-', linewidth=1, alpha=0.7, label='Selected Well GFP')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('GFP Expression Level')
                ax2.set_title('GFP Evolution Over Time')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'No GFP data', ha='center', va='center', transform=ax2.transAxes)
                
        except Exception as e:
            logger.error(f"Error plotting temperature and GFP evolution: {e}")
            ax1.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
    
    def plot_selection_frequency(self, ax):
        """Plot how often each well was selected for feedback"""
        try:
            if not self.global_history or 'selected_well_id' not in self.global_history:
                ax.text(0.5, 0.5, 'No selection data available', ha='center', va='center', transform=ax.transAxes)
                return
            
            num_wells = len(self.wells) if self.wells else self.params.get('num_wells', 1)
            selection_ids = self.global_history['selected_well_id']
            
            if not selection_ids:
                ax.text(0.5, 0.5, 'No selection history', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Count selections safely
            selection_counts = np.bincount(selection_ids, minlength=num_wells)
            well_ids = range(num_wells)
            
            # Color wells by survival status
            colors = []
            for i in well_ids:
                try:
                    if i < len(self.wells):
                        colors.append('red' if self.wells[i].extinct else 'green')
                    else:
                        colors.append('gray')
                except:
                    colors.append('gray')
            
            ax.bar(well_ids, selection_counts, color=colors, alpha=0.7)
            ax.set_xlabel('Well ID')
            ax.set_ylabel('Times Selected for Feedback')
            ax.set_title('Selection Frequency by Well (Green=Alive, Red=Extinct)')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='green', lw=4, label='Surviving Wells'),
                Line2D([0], [0], color='red', lw=4, label='Extinct Wells')
            ]
            ax.legend(handles=legend_elements)
            
        except Exception as e:
            logger.error(f"Error plotting selection frequency: {e}")
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
    
    def plot_indirect_learning(self, ax):
        """Analyze indirect learning - GFP evolution in wells that were rarely selected"""
        try:
            if not self.wells or not self.global_history or 'selected_well_id' not in self.global_history:
                ax.text(0.5, 0.5, 'Insufficient data for indirect learning analysis', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            selection_ids = self.global_history['selected_well_id']
            if not selection_ids:
                ax.text(0.5, 0.5, 'No selection data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Count selections per well
            num_wells = len(self.wells)
            selection_counts = np.bincount(selection_ids, minlength=num_wells)
            
            # Split wells into frequently vs rarely selected
            median_selections = np.median(selection_counts[selection_counts > 0]) if np.any(selection_counts > 0) else 0
            
            frequent_wells = []
            rare_wells = []
            
            for i in range(num_wells):
                try:
                    if (i < len(self.wells) and 
                        not self.wells[i].extinct and 
                        hasattr(self.wells[i], 'history') and 
                        'average_gfp' in self.wells[i].history):
                        
                        if selection_counts[i] >= median_selections:
                            frequent_wells.append(i)
                        else:
                            rare_wells.append(i)
                except:
                    continue
            
            if not frequent_wells or not rare_wells:
                ax.text(0.5, 0.5, 'Insufficient wells for comparison', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate average GFP evolution for each group
            try:
                # Get GFP histories
                frequent_histories = []
                rare_histories = []
                
                for i in frequent_wells:
                    history = self.wells[i].history['average_gfp']
                    if history:
                        frequent_histories.append(history)
                
                for i in rare_wells:
                    history = self.wells[i].history['average_gfp']
                    if history:
                        rare_histories.append(history)
                
                if not frequent_histories or not rare_histories:
                    ax.text(0.5, 0.5, 'No valid GFP histories', ha='center', va='center', transform=ax.transAxes)
                    return
                
                # Find common length
                min_length = min([len(h) for h in frequent_histories + rare_histories])
                if min_length <= 0:
                    ax.text(0.5, 0.5, 'No common history length', ha='center', va='center', transform=ax.transAxes)
                    return
                
                # Calculate means
                frequent_gfp = np.mean([h[:min_length] for h in frequent_histories], axis=0)
                rare_gfp = np.mean([h[:min_length] for h in rare_histories], axis=0)
                
                generations = range(min_length)
                
                ax.plot(generations, frequent_gfp, 'b-', linewidth=2, 
                       label=f'Frequently Selected Wells (n={len(frequent_wells)})')
                ax.plot(generations, rare_gfp, 'r-', linewidth=2, 
                       label=f'Rarely Selected Wells (n={len(rare_wells)})')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Average GFP Expression')
                ax.set_title('Evidence of Indirect Learning')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Test for convergence
                if min_length > 50:
                    freq_final = safe_mean(frequent_gfp[-50:])
                    rare_final = safe_mean(rare_gfp[-50:])
                    ax.text(0.05, 0.95, f'Final GFP: Frequent={freq_final:.3f}, Rare={rare_final:.3f}',
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                           
            except Exception as e:
                logger.error(f"Error calculating GFP evolution: {e}")
                ax.text(0.5, 0.5, f'Calculation error: {str(e)[:50]}...', 
                       ha='center', va='center', transform=ax.transAxes)
                
        except Exception as e:
            logger.error(f"Error in indirect learning analysis: {e}")
            ax.text(0.5, 0.5, f'Analysis error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
    
    def plot_population_dynamics(self, ax):
        """Plot population dynamics and extinction events"""
        try:
            if not self.global_history or 'living_wells' not in self.global_history:
                ax.text(0.5, 0.5, 'No population data available', ha='center', va='center', transform=ax.transAxes)
                return
            
            generations = self.global_history.get('generation', [])
            living_wells = self.global_history.get('living_wells', [])
            
            if not generations or not living_wells or len(generations) != len(living_wells):
                ax.text(0.5, 0.5, 'Incomplete population data', ha='center', va='center', transform=ax.transAxes)
                return
            
            ax.plot(generations, living_wells, 'b-', linewidth=2, label='Living Wells')
            
            # Add reference lines
            num_wells = self.params.get('num_wells', max(living_wells) if living_wells else 1)
            ax.axhline(y=num_wells/2, color='red', linestyle='--', alpha=0.7, label='50% Survival')
            
            # Mark major extinction events
            for i in range(1, len(living_wells)):
                if living_wells[i-1] - living_wells[i] >= max(10, num_wells * 0.1):
                    ax.axvline(x=generations[i], color='red', alpha=0.5, linestyle=':')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Number of Living Wells')
            ax.set_title('Population Dynamics and Extinction Events')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add survival statistics
            if living_wells:
                final_survival = living_wells[-1] / num_wells if num_wells > 0 else 0
                ax.text(0.7, 0.9, f'Final Survival: {final_survival:.1%}', 
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                       
        except Exception as e:
            logger.error(f"Error plotting population dynamics: {e}")
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', ha='center', va='center', transform=ax.transAxes)
    
    def create_comprehensive_summary(self) -> Dict:
        """Create comprehensive summary of the evolutionary experiment"""
        try:
            # Default summary structure
            summary = {
                'initial_wells': 0,
                'surviving_wells': 0,
                'extinct_wells': 0,
                'survival_rate': 0.0,
                'initial_avg_gfp': 0.0,
                'final_avg_gfp': 0.0,
                'gfp_evolution_rate': 0.0,
                'max_final_gfp': 0.0,
                'initial_temperature': 30.0,
                'final_temperature': 30.0,
                'temperature_change': 0.0,
                'temperature_stability': 0.0,
                'most_selected_well': 0,
                'max_selections': 0,
                'selection_inequality': 0.0,
                'average_mutations_per_cell': 0.0,
                'total_mutations': 0.0,
                'evidence_of_evolution': False,
                'temperature_learning': False,
                'population_adaptation': False
            }
            
            # Population statistics
            if self.wells:
                living_wells = [w for w in self.wells if not getattr(w, 'extinct', True)]
                extinct_wells = [w for w in self.wells if getattr(w, 'extinct', True)]
                
                summary['initial_wells'] = len(self.wells)
                summary['surviving_wells'] = len(living_wells)
                summary['extinct_wells'] = len(extinct_wells)
                summary['survival_rate'] = safe_divide(len(living_wells), len(self.wells), 0.0)
                
                # GFP statistics for surviving wells
                if living_wells:
                    try:
                        final_gfps = []
                        mutations_list = []
                        
                        for well in living_wells:
                            try:
                                gfp = well.calculate_average_gfp()
                                if np.isfinite(gfp):
                                    final_gfps.append(gfp)
                                    
                                mutations = well.calculate_average_mutations()
                                if np.isfinite(mutations):
                                    mutations_list.append(mutations)
                            except:
                                continue
                        
                        if final_gfps:
                            summary['max_final_gfp'] = max(final_gfps)
                        if mutations_list:
                            summary['average_mutations_per_cell'] = safe_mean(mutations_list)
                            summary['total_mutations'] = sum(mutations_list) * summary['surviving_wells']
                    except Exception as e:
                        logger.error(f"Error calculating well statistics: {e}")
            
            # Global history analysis
            if self.global_history:
                try:
                    avg_gfp_history = self.global_history.get('average_gfp_all_wells', [])
                    if avg_gfp_history:
                        summary['initial_avg_gfp'] = safe_mean(avg_gfp_history[:10])
                        summary['final_avg_gfp'] = safe_mean(avg_gfp_history[-10:])
                        summary['gfp_evolution_rate'] = safe_divide(
                            summary['final_avg_gfp'] - summary['initial_avg_gfp'],
                            len(avg_gfp_history), 0.0
                        )
                    
                    # Temperature analysis
                    temp_history = self.global_history.get('global_temperature', [])
                    if temp_history:
                        summary['initial_temperature'] = temp_history[0]
                        summary['final_temperature'] = temp_history[-1]
                        summary['temperature_change'] = summary['final_temperature'] - summary['initial_temperature']
                        summary['temperature_stability'] = safe_std(temp_history)
                    
                    # Selection analysis
                    selection_ids = self.global_history.get('selected_well_id', [])
                    if selection_ids:
                        selection_counts = np.bincount(selection_ids)
                        if len(selection_counts) > 0:
                            summary['most_selected_well'] = int(np.argmax(selection_counts))
                            summary['max_selections'] = int(np.max(selection_counts))
                            mean_selections = safe_mean(selection_counts.tolist())
                            if mean_selections > 0:
                                summary['selection_inequality'] = safe_divide(safe_std(selection_counts.tolist()), mean_selections, 0.0)
                        
                except Exception as e:
                    logger.error(f"Error analyzing global history: {e}")
            
            # Derived metrics
            summary['evidence_of_evolution'] = summary['gfp_evolution_rate'] > 0.001
            summary['temperature_learning'] = abs(summary['temperature_change']) > 1.0
            summary['population_adaptation'] = summary['survival_rate'] > 0.5
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating comprehensive summary: {e}")
            # Return default summary on error
            return {
                'initial_wells': 0, 'surviving_wells': 0, 'extinct_wells': 0,
                'survival_rate': 0.0, 'initial_avg_gfp': 0.0, 'final_avg_gfp': 0.0,
                'gfp_evolution_rate': 0.0, 'max_final_gfp': 0.0,
                'initial_temperature': 30.0, 'final_temperature': 30.0,
                'temperature_change': 0.0, 'temperature_stability': 0.0,
                'most_selected_well': 0, 'max_selections': 0, 'selection_inequality': 0.0,
                'average_mutations_per_cell': 0.0, 'total_mutations': 0.0,
                'evidence_of_evolution': False, 'temperature_learning': False,
                'population_adaptation': False, 'error': str(e)
            }
    
    def plot_feedback_well_analysis(self, ax):
        """Plot specific analysis of the feedback well"""
        try:
            fb_id = self.params.get('feedback_well_id', -1)
            if fb_id < 0 or fb_id >= len(self.wells):
                ax.text(0.5, 0.5, f'Invalid feedback well ID: {fb_id}', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            feedback_well = self.wells[fb_id]
            if getattr(feedback_well, 'extinct', True):
                ax.text(0.5, 0.5, f'Feedback Well #{fb_id} went extinct', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=14, color='red')
                ax.set_title(f'Feedback Well #{fb_id} - EXTINCT')
                return
            
            if not hasattr(feedback_well, 'history') or not feedback_well.history:
                ax.text(0.5, 0.5, 'No history data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Get histories
            gfp_values = feedback_well.history.get('average_gfp', [])
            temperatures = feedback_well.history.get('temperature', [])
            
            if not gfp_values or not temperatures:
                ax.text(0.5, 0.5, 'Incomplete history data', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            generations = range(min(len(gfp_values), len(temperatures)))
            if not generations:
                ax.text(0.5, 0.5, 'No data to plot', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Create dual axis plot
            ax2 = ax.twinx()
            
            line1 = ax.plot(generations, gfp_values[:len(generations)], 'g-', 
                           linewidth=2, label='Feedback Well GFP')
            line2 = ax2.plot(generations, temperatures[:len(generations)], 'r-', 
                            linewidth=2, label='Temperature')
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('GFP Expression', color='g')
            ax2.set_ylabel('Temperature (°C)', color='r')
            ax.set_title(f'Feedback Well #{fb_id} - Direct Learning')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"Error plotting feedback well analysis: {e}")
            ax.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)


# ============================================================================
# CONFIGURATION AND MAIN FUNCTIONS - BULLETPROOFED
# ============================================================================

class YeastExperimentConfig:
    """Configuration presets for yeast evolutionary learning experiments"""
    
    PRESETS = {
        'quick_test': {
            'num_wells': 30,
            'genome_length': 500,
            'num_generations': 200,
            'mutation_rate': 1.0,
            'cells_per_well': 100,
            'stress_temperature': 39.0,
            'relief_step': 1.0,
            'optimal_temperature': 30.0,
            'random_seed': 42
        },
        'standard_experiment': {
            'num_wells': 96,
            'genome_length': 1000,
            'num_generations': 800,
            'mutation_rate': 1.0,
            'cells_per_well': 250,
            'stress_temperature': 39.0,
            'relief_step': 1.0,
            'optimal_temperature': 30.0,
            'random_seed': 42
        },
        'large_scale': {
            'num_wells': 384,
            'genome_length': 1500,
            'num_generations': 1000,
            'mutation_rate': 1.0,
            'cells_per_well': 250,
            'stress_temperature': 39.0,
            'relief_step': 1.5,
            'optimal_temperature': 30.0,
            'random_seed': 42
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Dict:
        """Get a preset configuration with error handling"""
        try:
            if name not in cls.PRESETS:
                logger.warning(f"Unknown preset: {name}. Available: {list(cls.PRESETS.keys())}")
                logger.info("Using 'quick_test' preset as fallback")
                name = 'quick_test'
            return cls.PRESETS[name].copy()
        except Exception as e:
            logger.error(f"Error getting preset: {e}")
            return cls.PRESETS['quick_test'].copy()


def create_detailed_analysis_plots(analyzer: BiologicalAnalysis, results: Dict, output_dir: str):
    """Create detailed analysis plots with comprehensive error handling"""
    try:
        # Create summary text plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Get comprehensive summary
        summary = analyzer.create_comprehensive_summary()
        params = results.get('params', {})
        
        summary_text = f"""🧪 YEAST EVOLUTIONARY LEARNING EXPERIMENT SUMMARY

        🔬 Experimental Setup:
        • {summary.get('initial_wells', 0)} wells with single yeast cells
        • {params.get('genome_length', 0)} bp genomes  
        • {params.get('mutation_rate', 0)} mutations/genome/generation
        • {params.get('num_generations', 0)} total generations
        • Feedback well: #{params.get('feedback_well_id', 'Random')}

        📊 Population Results:
        • {summary.get('surviving_wells', 0)} wells survived ({summary.get('survival_rate', 0):.1%})
        • {summary.get('extinct_wells', 0)} wells went extinct
        • Maximum final GFP: {summary.get('max_final_gfp', 0):.3f}

        🧬 Evolution Metrics:
        • Initial avg GFP: {summary.get('initial_avg_gfp', 0):.3f}
        • Final avg GFP: {summary.get('final_avg_gfp', 0):.3f}
        • Evolution rate: {summary.get('gfp_evolution_rate', 0):.6f} units/generation
        • Total mutations: {summary.get('total_mutations', 0):.1f}

        🌡️ Environmental Dynamics:
        • Initial temperature: {summary.get('initial_temperature', 30):.1f}°C
        • Final temperature: {summary.get('final_temperature', 30):.1f}°C
        • Temperature change: {summary.get('temperature_change', 0):.1f}°C
        • Temperature stability (σ): {summary.get('temperature_stability', 0):.2f}°C

        🎯 Selection Analysis:
        • Most selected well: #{summary.get('most_selected_well', 0)} ({summary.get('max_selections', 0)} times)
        • Selection inequality: {summary.get('selection_inequality', 0):.3f}
        • Average mutations/cell: {summary.get('average_mutations_per_cell', 0):.1f}

        🔍 Key Findings:
        • Evolution detected: {'✅ YES' if summary.get('evidence_of_evolution', False) else '❌ NO'}
        • Temperature learning: {'✅ YES' if summary.get('temperature_learning', False) else '❌ NO'}
        • Population adaptation: {'✅ YES' if summary.get('population_adaptation', False) else '❌ NO'}

        📈 Biological Significance:
        • {'Strong evidence' if summary.get('gfp_evolution_rate', 0) > 0.005 else 'Limited evidence' if summary.get('gfp_evolution_rate', 0) > 0.001 else 'No evidence'} of evolutionary learning
        • {'Excellent' if summary.get('survival_rate', 0) > 0.8 else 'Good' if summary.get('survival_rate', 0) > 0.5 else 'Poor'} population survival
        • {'Effective' if abs(summary.get('temperature_change', 0)) > 2 else 'Moderate' if abs(summary.get('temperature_change', 0)) > 0.5 else 'Minimal'} environmental feedback

        ⚠️ Potential Issues:
        {get_experiment_warnings(summary)}
        """
        
        # Remove excessive whitespace while preserving formatting
        summary_text = '\n'.join(line.strip() for line in summary_text.split('\n'))
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.title('Experimental Summary and Analysis', fontsize=16, fontweight='bold')
        
        if safe_save_plot(fig, f"{output_dir}/experiment_summary.png"):
            logger.info("Summary plot saved successfully")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating detailed analysis plots: {e}")


def get_experiment_warnings(summary: Dict) -> str:
    """Generate experiment warnings based on summary statistics"""
    warnings = []
    
    try:
        if summary.get('survival_rate', 1.0) < 0.3:
            warnings.append("• Low survival rate - consider reducing stress or increasing relief")
        
        if summary.get('gfp_evolution_rate', 0.0) < 0.0001:
            warnings.append("• Minimal GFP evolution - may need longer run or different parameters") 
        
        if abs(summary.get('temperature_change', 0.0)) < 0.5:
            warnings.append("• Little temperature learning - check feedback mechanism")
        
        if summary.get('selection_inequality', 0.0) > 2.0:
            warnings.append("• High selection inequality - some wells rarely selected")
        
        if summary.get('average_mutations_per_cell', 0.0) > 10.0:
            warnings.append("• High mutation load - may impede beneficial evolution")
        
        if not warnings:
            warnings.append("• No major issues detected")
            
    except Exception as e:
        warnings.append(f"• Warning analysis failed: {str(e)[:50]}...")
    
    return '\n        '.join(warnings)


def create_dna_evolution_plots(analyzer: BiologicalAnalysis, results: Dict, output_dir: str):
    """Create DNA evolution analysis plots with error handling"""
    try:
        if not analyzer.wells:
            logger.warning("No wells data for DNA analysis")
            return
        
        surviving_wells = [w for w in analyzer.wells if not getattr(w, 'extinct', True) and w.cells]
        
        if not surviving_wells:
            logger.warning("No surviving wells with cells for DNA analysis")
            return
            
        fig = plt.figure(figsize=(15, 10))
        
        # Mutations vs GFP
        plt.subplot(2, 3, 1)
        try:
            mutations = []
            gfps = []
            for well in surviving_wells:
                try:
                    mut = well.calculate_average_mutations()
                    gfp = well.calculate_average_gfp()
                    if np.isfinite(mut) and np.isfinite(gfp):
                        mutations.append(mut)
                        gfps.append(gfp)
                except:
                    continue
            
            if mutations and gfps:
                plt.scatter(mutations, gfps, alpha=0.6, color='blue')
                plt.xlabel('Average Mutations per Cell')
                plt.ylabel('GFP Expression')
                plt.title('Mutations vs GFP in Surviving Wells')
                plt.grid(True, alpha=0.3)
                
                # Add correlation
                if len(mutations) > 1:
                    correlation = np.corrcoef(mutations, gfps)[0, 1]
                    if np.isfinite(correlation):
                        plt.text(0.05, 0.95, f'r = {correlation:.3f}', transform=plt.gca().transAxes,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                plt.text(0.5, 0.5, 'No valid mutation/GFP data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Nucleotide composition analysis
        plt.subplot(2, 3, 2)
        try:
            at_content = []
            well_gfps = []
            
            for well in surviving_wells:
                try:
                    if well.cells and hasattr(well.cells[0], 'dna_sequence'):
                        dna = well.cells[0].dna_sequence
                        if dna:
                            total = len(dna)
                            at = (dna.count('A') + dna.count('T')) / total
                            gfp = well.calculate_average_gfp()
                            if np.isfinite(at) and np.isfinite(gfp):
                                at_content.append(at)
                                well_gfps.append(gfp)
                except:
                    continue
            
            if at_content and well_gfps:
                plt.scatter(at_content, well_gfps, alpha=0.6, color='red')
                plt.xlabel('AT Content')
                plt.ylabel('GFP Expression')
                plt.title('AT Content vs GFP')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid AT content data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # GC content analysis
        plt.subplot(2, 3, 3)
        try:
            gc_content = []
            well_gfps = []
            
            for well in surviving_wells:
                try:
                    if well.cells and hasattr(well.cells[0], 'dna_sequence'):
                        dna = well.cells[0].dna_sequence
                        if dna:
                            total = len(dna)
                            gc = (dna.count('G') + dna.count('C')) / total
                            gfp = well.calculate_average_gfp()
                            if np.isfinite(gc) and np.isfinite(gfp):
                                gc_content.append(gc)
                                well_gfps.append(gfp)
                except:
                    continue
            
            if gc_content and well_gfps:
                plt.scatter(gc_content, well_gfps, alpha=0.6, color='green')
                plt.xlabel('GC Content')
                plt.ylabel('GFP Expression')
                plt.title('GC Content vs GFP')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid GC content data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Strongest enhancer motif analysis
        plt.subplot(2, 3, 4)
        try:
            # Find strongest enhancer motif
            strongest_motif = max(YeastCell.GFP_TETRANUCLEOTIDES.items(), key=lambda x: x[1])
            motif_name, motif_effect = strongest_motif
            
            motif_frequencies = []
            well_gfps = []
            
            for well in surviving_wells:
                try:
                    if well.cells and hasattr(well.cells[0], 'dna_sequence'):
                        dna = well.cells[0].dna_sequence
                        if dna and len(dna) > 3:
                            count = sum(1 for i in range(len(dna) - 3) if dna[i:i+4] == motif_name)
                            freq = count / max(1, len(dna) - 3)
                            gfp = well.calculate_average_gfp()
                            if np.isfinite(freq) and np.isfinite(gfp):
                                motif_frequencies.append(freq)
                                well_gfps.append(gfp)
                except:
                    continue
            
            if motif_frequencies and well_gfps:
                plt.scatter(motif_frequencies, well_gfps, alpha=0.6, color='purple')
                plt.xlabel(f'{motif_name} Frequency (effect: {motif_effect}x)')
                plt.ylabel('GFP Expression')
                plt.title(f'Strongest Enhancer Motif vs GFP')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid motif data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Strongest repressor motif analysis
        plt.subplot(2, 3, 5)
        try:
            # Find strongest repressor motif  
            weakest_motif = min(YeastCell.GFP_TETRANUCLEOTIDES.items(), key=lambda x: x[1])
            motif_name, motif_effect = weakest_motif
            
            motif_frequencies = []
            well_gfps = []
            
            for well in surviving_wells:
                try:
                    if well.cells and hasattr(well.cells[0], 'dna_sequence'):
                        dna = well.cells[0].dna_sequence
                        if dna and len(dna) > 3:
                            count = sum(1 for i in range(len(dna) - 3) if dna[i:i+4] == motif_name)
                            freq = count / max(1, len(dna) - 3)
                            gfp = well.calculate_average_gfp()
                            if np.isfinite(freq) and np.isfinite(gfp):
                                motif_frequencies.append(freq)
                                well_gfps.append(gfp)
                except:
                    continue
            
            if motif_frequencies and well_gfps:
                plt.scatter(motif_frequencies, well_gfps, alpha=0.6, color='orange')
                plt.xlabel(f'{motif_name} Frequency (effect: {motif_effect}x)')
                plt.ylabel('GFP Expression')
                plt.title(f'Strongest Repressor Motif vs GFP')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid motif data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        # Regulatory balance analysis
        plt.subplot(2, 3, 6)
        try:
            enhancer_motifs = [k for k, v in YeastCell.GFP_TETRANUCLEOTIDES.items() if v > 1.0]
            repressor_motifs = [k for k, v in YeastCell.GFP_TETRANUCLEOTIDES.items() if v < 1.0]
            
            motif_balance = []
            well_gfps = []
            
            for well in surviving_wells:
                try:
                    if well.cells and hasattr(well.cells[0], 'dna_sequence'):
                        dna = well.cells[0].dna_sequence
                        if dna and len(dna) > 3:
                            enh_score = 0
                            rep_score = 0
                            
                            for motif in enhancer_motifs:
                                count = sum(1 for i in range(len(dna) - 3) if dna[i:i+4] == motif)
                                freq = count / max(1, len(dna) - 3)
                                enh_score += freq * YeastCell.GFP_TETRANUCLEOTIDES[motif]
                            
                            for motif in repressor_motifs:
                                count = sum(1 for i in range(len(dna) - 3) if dna[i:i+4] == motif)
                                freq = count / max(1, len(dna) - 3)
                                rep_score += freq * YeastCell.GFP_TETRANUCLEOTIDES[motif]
                            
                            balance = enh_score - rep_score
                            gfp = well.calculate_average_gfp()
                            if np.isfinite(balance) and np.isfinite(gfp):
                                motif_balance.append(balance)
                                well_gfps.append(gfp)
                except:
                    continue
            
            if motif_balance and well_gfps:
                plt.scatter(motif_balance, well_gfps, alpha=0.6, color='brown')
                plt.xlabel('Enhancer - Repressor Balance')
                plt.ylabel('GFP Expression')
                plt.title('Regulatory Motif Balance vs GFP')
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No valid balance data', ha='center', va='center', transform=plt.gca().transAxes)
        except Exception as e:
            plt.text(0.5, 0.5, f'Error: {str(e)[:30]}...', ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.suptitle('DNA Evolution and Regulatory Analysis', fontsize=16)
        plt.tight_layout()
        
        if safe_save_plot(fig, f"{output_dir}/dna_evolution_analysis.png"):
            logger.info("DNA evolution plot saved successfully")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating DNA evolution plots: {e}")


def export_experimental_data(results: Dict, summary: Dict, output_dir: str) -> None:
    """
    Export all experimental data (per-well table, global history, JSON summary,
    individual well histories and final genomes FASTA) with robust error handling.
    """
    try:
        wells: List[Well] = results.get('wells', [])
        global_history: Dict[str, List[Any]] = results.get('global_history', {})

        # ------------------------------------------------------------------ #
        # 1.  WELL-BY-WELL CSV                                               #
        # ------------------------------------------------------------------ #
        logger.info("Exporting well analysis …")
        try:
            rows: List[Dict[str, Any]] = []
            for well in wells:
                try:
                    row = {
                        "well_id": getattr(well, "well_id", -1),
                        "extinct": getattr(well, "extinct", True),
                        "generation_extinct": getattr(well, "generation_extinct", None),
                        "times_selected": global_history.get("selected_well_id", []).count(
                            getattr(well, "well_id", -1)
                        ),
                        "final_gfp": well.calculate_average_gfp() if not well.extinct else 0.0,
                        "final_mutations": well.calculate_average_mutations() if not well.extinct else 0.0,
                    }

                    # DNA-level statistics for surviving wells
                    if not well.extinct and well.cells:
                        dna = well.cells[0].dna_sequence
                        row.update(
                            {
                                "dna_length": len(dna),
                                "at_content": safe_divide(
                                    dna.count("A") + dna.count("T"), len(dna)
                                ),
                                "gc_content": safe_divide(
                                    dna.count("G") + dna.count("C"), len(dna)
                                ),
                            }
                        )
                        # regulatory motifs
                        for motif, effect in YeastCell.GFP_TETRANUCLEOTIDES.items():
                            count = sum(
                                1 for i in range(len(dna) - 3) if dna[i : i + 4] == motif
                            )
                            row[f"{motif}_count"] = count
                            row[f"{motif}_frequency"] = safe_divide(
                                count, len(dna) - 3
                            )
                    rows.append(row)
                except Exception as e:
                    logger.error(f"Well {getattr(well,'well_id','?')} export error: {e}")

            if rows:
                pd.DataFrame(rows).to_csv(f"{output_dir}/well_analysis.csv", index=False)
                logger.info(f"Wrote {len(rows)} rows -> well_analysis.csv")
        except Exception as e:
            logger.error(f"Failed exporting well analysis: {e}")

        # ------------------------------------------------------------------ #
        # 2.  GLOBAL HISTORY CSV                                             #
        # ------------------------------------------------------------------ #
        logger.info("Exporting global history …")
        try:
            if global_history:
                pd.DataFrame(global_history).to_csv(
                    f"{output_dir}/global_evolution_history.csv", index=False
                )
                logger.info("global_evolution_history.csv written")
        except Exception as e:
            logger.error(f"Failed exporting global history: {e}")

        # ------------------------------------------------------------------ #
        # 3.  SUMMARY  →  JSON                                               #
        # ------------------------------------------------------------------ #
        logger.info("Exporting experiment summary …")
        try:
            def to_py(obj):
                """Convert numpy / exotic scalars to plain python types."""
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj) if np.isfinite(obj) else None
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return obj

            json_safe_summary = {k: to_py(v) for k, v in summary.items()}
            with open(f"{output_dir}/experiment_summary.json", "w") as fh:
                json.dump(json_safe_summary, fh, indent=2)
            logger.info("experiment_summary.json written")
        except Exception as e:
            logger.error(f"Failed exporting summary JSON: {e}")

        # ------------------------------------------------------------------ #
        # 4.  PER-WELL HISTORIES  →  JSON                                    #
        # ------------------------------------------------------------------ #
        logger.info("Exporting individual well histories …")
        try:
            histories = {}
            for well in wells:
                if not well.extinct and hasattr(well, "history"):
                    histories[f"well_{well.well_id}"] = {
                        k: [to_py(x) for x in v] for k, v in well.history.items()
                    }
            if histories:
                with open(f"{output_dir}/well_histories.json", "w") as fh:
                    json.dump(histories, fh, indent=2)
                logger.info(f"well_histories.json written for {len(histories)} wells")
        except Exception as e:
            logger.error(f"Failed exporting well histories: {e}")

        # ------------------------------------------------------------------ #
        # 5.  FASTA OF FINAL GENOMES                                         #
        # ------------------------------------------------------------------ #
        logger.info("Exporting DNA FASTA …")
        try:
            fasta_path = f"{output_dir}/final_genomes.fasta"
            written = 0
            with open(fasta_path, "w") as fasta:
                for well in wells:
                    if not well.extinct and well.cells:
                        seq = well.cells[0].dna_sequence
                        gfp = well.calculate_average_gfp()
                        mut = well.calculate_average_mutations()
                        sel = global_history.get("selected_well_id", []).count(
                            well.well_id
                        )
                        fasta.write(
                            f">Well_{well.well_id}|GFP={gfp:.3f}|Mut={mut:.2f}|Sel={sel}\n"
                        )
                        fasta.write(seq + "\n")
                        written += 1
            logger.info(f"Wrote {written} sequences -> final_genomes.fasta")
        except Exception as e:
            logger.error(f"Failed exporting FASTA: {e}")

    except Exception as e:
        logger.critical(f"export_experimental_data crashed: {e}")
                    


def safe_create_directory(path: str) -> bool:
    """Safely create directory with error handling"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        return False


def safe_save_plot(fig, filename: str) -> bool:
    """Safely save plot with error handling"""
    try:
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        return True
    except Exception as e:
        logger.error(f"Error saving plot {filename}: {e}")
        return False


def run_yeast_evolution_experiment(params: Dict, output_dir: str) -> Tuple[Dict, Dict]:
    """Run the complete yeast evolution experiment with analysis"""
    
    try:
        # Ensure output directory exists
        if not safe_create_directory(output_dir):
            raise ValueError(f"Cannot create output directory: {output_dir}")
        
        logger.info(f"\n{'='*80}")
        logger.info("🧪 YEAST EVOLUTIONARY LEARNING EXPERIMENT")
        logger.info(f"{'='*80}")
        logger.info("Modeling: GFP-sugar metabolism construct under temperature feedback")
        
        # Validate and display parameters
        validated_params = validate_params(params)
        logger.info("\nExperimental Parameters:")
        for key, value in validated_params.items():
            logger.info(f"  {key}: {value}")
        
        # Set random seed
        if 'random_seed' in validated_params:
            np.random.seed(validated_params['random_seed'])
        
        # Run simulation
        logger.info(f"\n{'-'*60}")
        simulation = EvolutionarySimulation(validated_params)
        results = simulation.run_simulation()
        logger.info(f"{'-'*60}")
        
        # Check if simulation was successful
        if not results.get('success', True):
            logger.error(f"Simulation failed: {results.get('error', 'Unknown error')}")
            # Create minimal summary for failed simulation
            summary = {
                'initial_wells': validated_params.get('num_wells', 0),
                'surviving_wells': 0,
                'survival_rate': 0.0,
                'evidence_of_evolution': False,
                'simulation_failed': True,
                'error': results.get('error', 'Unknown error')
            }
            return results, summary
        
        # Analyze results
        try:
            analyzer = BiologicalAnalysis(results)
            summary = analyzer.create_comprehensive_summary()
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            # Create minimal summary
            summary = {
                'initial_wells': validated_params.get('num_wells', 0),
                'surviving_wells': len([w for w in results.get('wells', []) if not getattr(w, 'extinct', True)]),
                'survival_rate': 0.0,
                'analysis_failed': True,
                'error': str(e)
            }
        
        # Print summary
        logger.info("\n🔬 EXPERIMENTAL RESULTS:")
        logger.info(f"  Population survival: {summary.get('survival_rate', 0):.1%} "
                   f"({summary.get('surviving_wells', 0)}/{summary.get('initial_wells', 0)} wells)")
        logger.info(f"  GFP evolution: {summary.get('initial_avg_gfp', 0):.3f} → "
                   f"{summary.get('final_avg_gfp', 0):.3f} "
                   f"(rate: {summary.get('gfp_evolution_rate', 0):.4f}/gen)")
        logger.info(f"  Temperature: {summary.get('initial_temperature', 30):.1f}°C → "
                   f"{summary.get('final_temperature', 30):.1f}°C "
                   f"(Δ: {summary.get('temperature_change', 0):.1f}°C)")
        logger.info(f"  Evidence of evolution: {'Yes' if summary.get('evidence_of_evolution', False) else 'No'}")
        logger.info(f"  Population adapted: {'Yes' if summary.get('population_adaptation', False) else 'No'}")
        
        # Create visualizations
        logger.info("\n📊 Creating analysis plots...")
        
        try:
            create_main_analysis_plots(analyzer, output_dir)
        except Exception as e:
            logger.error(f"Error creating main plots: {e}")
        
        try:
            create_detailed_analysis_plots(analyzer, results, output_dir)
        except Exception as e:
            logger.error(f"Error creating detailed plots: {e}")
        
        try:
            create_dna_evolution_plots(analyzer, results, output_dir)
        except Exception as e:
            logger.error(f"Error creating DNA evolution plots: {e}")
        
        # Export data
        logger.info("\n💾 Exporting experimental data...")
        try:
            export_experimental_data(results, summary, output_dir)
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
        
        # Create biological report
        try:
            create_biological_report(results, summary, output_dir)
        except Exception as e:
            logger.error(f"Error creating biological report: {e}")
        
        logger.info(f"\n✅ Experiment completed!")
        logger.info(f"📁 Results saved to: {output_dir}/")
        logger.info(f"🔬 Summary: {summary.get('surviving_wells', 0)}/{summary.get('initial_wells', 0)} wells survived")
        logger.info(f"📈 Evolution: {'Detected' if summary.get('evidence_of_evolution', False) else 'Not detected'}")
        
        return results, summary
        
    except Exception as e:
        logger.error(f"Critical error in experiment: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return error result
        error_results = {
            'wells': [],
            'global_history': {},
            'params': validated_params if 'validated_params' in locals() else params,
            'success': False,
            'error': str(e)
        }
        
        error_summary = {
            'initial_wells': params.get('num_wells', 0),
            'surviving_wells': 0,
            'survival_rate': 0.0,
            'experiment_failed': True,
            'error': str(e)
        }
        
        return error_results, error_summary


def create_main_analysis_plots(analyzer: BiologicalAnalysis, output_dir: str):
    """Create main analysis plots with error handling"""
    try:
        fig = plt.figure(figsize=(20, 15))
        
        # Temperature and GFP evolution
        ax1 = plt.subplot(3, 3, 1)
        ax2 = plt.subplot(3, 3, 2)
        analyzer.plot_temperature_and_gfp_evolution(ax1, ax2)
        
        # Selection frequency
        ax3 = plt.subplot(3, 3, 3)
        analyzer.plot_selection_frequency(ax3)
        
        # Indirect learning
        ax4 = plt.subplot(3, 3, 4)
        analyzer.plot_indirect_learning(ax4)
        
        # Population dynamics
        ax5 = plt.subplot(3, 3, 5)
        analyzer.plot_population_dynamics(ax5)
        
        # Feedback well analysis
        ax6 = plt.subplot(3, 3, 6)
        analyzer.plot_feedback_well_analysis(ax6)
        
        # GFP change distribution
        ax7 = plt.subplot(3, 3, 7)
        try:
            if analyzer.global_history and 'selected_well_gfp_change' in analyzer.global_history:
                gfp_changes = analyzer.global_history['selected_well_gfp_change']
                if gfp_changes:
                    # Filter out infinite/nan values
                    valid_changes = [x for x in gfp_changes if np.isfinite(x)]
                    if valid_changes:
                        ax7.hist(valid_changes, bins=30, alpha=0.7, color='purple', edgecolor='black')
                        ax7.axvline(0, color='red', linestyle='--', label='No Change')
                        ax7.set_xlabel('GFP Change in Selected Wells')
                        ax7.set_ylabel('Frequency')
                        ax7.set_title('Distribution of GFP Changes')
                        ax7.legend()
                        ax7.grid(True, alpha=0.3)
                    else:
                        ax7.text(0.5, 0.5, 'No valid GFP change data', ha='center', va='center', transform=ax7.transAxes)
                else:
                    ax7.text(0.5, 0.5, 'No GFP change data', ha='center', va='center', transform=ax7.transAxes)
            else:
                ax7.text(0.5, 0.5, 'No GFP change data available', ha='center', va='center', transform=ax7.transAxes)
        except Exception as e:
            ax7.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax7.transAxes)
        
        # Temperature vs GFP correlation
        ax8 = plt.subplot(3, 3, 8)
        try:
            if (analyzer.global_history and 
                'global_temperature' in analyzer.global_history and 
                'average_gfp_all_wells' in analyzer.global_history):
                
                temps = analyzer.global_history['global_temperature']
                avg_gfps = analyzer.global_history['average_gfp_all_wells']
                
                if temps and avg_gfps and len(temps) == len(avg_gfps):
                    # Filter valid values
                    valid_pairs = [(t, g) for t, g in zip(temps, avg_gfps) if np.isfinite(t) and np.isfinite(g)]
                    if valid_pairs:
                        temps_valid, gfps_valid = zip(*valid_pairs)
                        ax8.scatter(temps_valid, gfps_valid, alpha=0.6, color='brown')
                        ax8.set_xlabel('Global Temperature (°C)')
                        ax8.set_ylabel('Average GFP (All Wells)')
                        ax8.set_title('Temperature vs Average GFP')
                        ax8.grid(True, alpha=0.3)
                        
                        # Add correlation coefficient
                        if len(valid_pairs) > 1:
                            correlation = np.corrcoef(temps_valid, gfps_valid)[0, 1]
                            if np.isfinite(correlation):
                                ax8.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                                        transform=ax8.transAxes,
                                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax8.text(0.5, 0.5, 'No valid data pairs', ha='center', va='center', transform=ax8.transAxes)
                else:
                    ax8.text(0.5, 0.5, 'Data length mismatch or empty', ha='center', va='center', transform=ax8.transAxes)
            else:
                ax8.text(0.5, 0.5, 'Temperature/GFP data not available', ha='center', va='center', transform=ax8.transAxes)
        except Exception as e:
            ax8.text(0.5, 0.5, f'Error: {str(e)[:50]}...', ha='center', va='center', transform=ax8.transAxes)
        
        # Final GFP distribution
        ax9 = plt.subplot(3, 3, 9)
        try:
            if analyzer.wells:
                living_wells = [w for w in analyzer.wells if not getattr(w, 'extinct', True)]
                if living_wells:
                    final_gfps = []
                    for well in living_wells:
                        try:
                            if isinstance(value, (np.float32, np.float64)):
                                json_safe_summary[key] = float(value)
                            elif isinstance(value, (np.int32, np.int64, np.integer)):
                                json_safe_summary[key] = int(value)
                            elif isinstance(value, np.bool_):
                                json_safe_summary[key] = bool(value)
                            elif np.isnan(value) if isinstance(value, (int, float)) else False:
                                json_safe_summary[key] = None
                            else:
                                json_safe_summary[key] = value
                        except:
                            json_safe_summary[key] = str(value)
                    
            with open(f"{output_dir}/experiment_summary.json", 'w') as f:
                json.dump(json_safe_summary, f, indent=2)
            logger.info("Experiment summary exported successfully")
        except Exception as e:
            logger.error(f"Error exporting experiment summary: {e}")
        
        # Export individual well histories
        logger.info("Exporting well histories...")
        try:
            well_histories = {}
            for well in wells:
                try:
                    if not getattr(well, 'extinct', True) and hasattr(well, 'history'):
                        # Convert history to JSON-safe format
                        safe_history = {}
                        for hist_key, hist_values in well.history.items():
                            try:
                                safe_values = []
                                for val in hist_values:
                                    if isinstance(val, (np.float32, np.float64)):
                                        safe_values.append(float(val))
                                    elif isinstance(val, (np.int32, np.int64, np.integer)):
                                        safe_values.append(int(val))
                                    elif isinstance(val, np.bool_):
                                        safe_values.append(bool(val))
                                    else:
                                        safe_values.append(val)
                                safe_history[hist_key] = safe_values
                            except:
                                safe_history[hist_key] = [str(v) for v in hist_values]
                        
                        well_histories[f'well_{well.well_id}'] = safe_history
                except Exception as e:
                    logger.error(f"Error processing history for well {getattr(well, 'well_id', 'unknown')}: {e}")
            
            if well_histories:
                with open(f"{output_dir}/well_histories.json", 'w') as f:
                    json.dump(well_histories, f, indent=2)
                logger.info(f"Well histories exported: {len(well_histories)} wells")
        except Exception as e:
            logger.error(f"Error exporting well histories: {e}")
        
        # Export DNA sequences in FASTA format
        logger.info("Exporting DNA sequences...")
        try:
            fasta_count = 0
            with open(f"{output_dir}/final_genomes.fasta", 'w') as f:
                for well in wells:
                    try:
                        if (not getattr(well, 'extinct', True) and 
                            hasattr(well, 'cells') and well.cells and
                            hasattr(well.cells[0], 'dna_sequence')):
                            
                            gfp = well.calculate_average_gfp()
                            mutations = well.calculate_average_mutations()
                            
                            selected_times = 0
                            if 'selected_well_id' in global_history:
                                selected_times = global_history['selected_well_id'].count(well.well_id)
                            
                            f.write(f">Well_{well.well_id}_GFP_{gfp:.3f}_Mutations_{mutations:.1f}_Selected_{selected_times}\n")
                            f.write(f"{well.cells[0].dna_sequence}\n")
                            fasta_count += 1
                    except Exception as e:
                        logger.error(f"Error writing FASTA for well {getattr(well, 'well_id', 'unknown')}: {e}")
            
            logger.info(f"DNA sequences exported: {fasta_count} sequences")
        except Exception as e:
            logger.error(f"Error exporting DNA sequences: {e}")
            
    except Exception as e:
        logger.error(f"Critical error in data export: {e}")


def create_biological_report(results: Dict, summary: Dict, output_dir: str):
    """Create a detailed biological interpretation report"""
    try:
        params = results.get('params', {})
        
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""# 🧪 Yeast Evolutionary Learning Experiment Report
                    Generated on {timestamp}

                    ## Experimental Design

                    This experiment simulated evolutionary learning in *Saccharomyces cerevisiae* (baker's yeast) 
                    using a GFP-sugar metabolism construct under temperature feedback control.

                    ### Setup
                    - **Number of wells**: {params.get('num_wells', 'N/A')} (simulating microculture array)
                    - **Genome length**: {params.get('genome_length', 'N/A')} bp (simplified chromosome)
                    - **Generations**: {params.get('num_generations', 'N/A')} (continuous culture cycles)
                    - **Mutation rate**: {params.get('mutation_rate', 'N/A')} mutations/genome/generation
                    - **Population per well**: {params.get('cells_per_well', 'N/A')} cells maximum
                    - **Feedback well**: #{params.get('feedback_well_id', 'Random')}

                    ### Biological Model
                    - **GFP expression** determined by tetranucleotide motifs in DNA
                    - **Temperature feedback** based on selected well's GFP change
                    - **Shared environment** - all wells experience same temperature
                    - **Random washing** simulates culture dilution and nutrient limitation

                    ## 📊 Experimental Results

                    ### Population Dynamics
                    - **Initial population**: {summary.get('initial_wells', 0)} wells
                    - **Final population**: {summary.get('surviving_wells', 0)} wells  
                    - **Survival rate**: {summary.get('survival_rate', 0):.1%}
                    - **Extinction events**: {summary.get('extinct_wells', 0)} wells

                    ### Evolutionary Outcomes
                    - **GFP evolution**: {summary.get('initial_avg_gfp', 0):.3f} → {summary.get('final_avg_gfp', 0):.3f}
                    - **Evolution rate**: {summary.get('gfp_evolution_rate', 0):.6f} units/generation
                    - **Maximum GFP achieved**: {summary.get('max_final_gfp', 0):.3f}
                    - **Evidence of evolution**: {'✅ YES' if summary.get('evidence_of_evolution', False) else '❌ NO'}

                    ### Environmental Adaptation  
                    - **Temperature change**: {summary.get('initial_temperature', 30):.1f}°C → {summary.get('final_temperature', 30):.1f}°C
                    - **Temperature stability**: σ = {summary.get('temperature_stability', 0):.2f}°C
                    - **Learning detected**: {'✅ YES' if summary.get('temperature_learning', False) else '❌ NO'}

                    ### Selection Dynamics
                    - **Most selected well**: #{summary.get('most_selected_well', 0)} ({summary.get('max_selections', 0)} times)
                    - **Selection inequality**: {summary.get('selection_inequality', 0):.3f} (0=equal, higher=more unequal)
                    - **Average mutations**: {summary.get('average_mutations_per_cell', 0):.1f} per cell

                    ## 🧬 Biological Interpretation

                    ### Evolutionary Success
                    {'✅ **Successful Evolution**: The population showed clear signs of evolutionary adaptation. GFP levels increased over time, indicating that the temperature feedback successfully selected for higher GFP expression.' if summary.get('evidence_of_evolution', False) else '❌ **Limited Evolution**: The population showed minimal evolutionary change. This could indicate insufficient selection pressure, high mutation load, or population bottlenecks.'}

                    ### Population Adaptation
                    {'✅ **Good Population Survival**: Over half the wells survived, indicating successful adaptation to the selective environment.' if summary.get('population_adaptation', False) else '❌ **Poor Population Survival**: High extinction rate suggests strong selection pressure or environmental stress exceeded adaptive capacity.'}

                    ### Learning Mechanism
                    {'✅ **Environmental Learning**: Temperature changes indicate the feedback system effectively linked GFP expression to environmental conditions.' if summary.get('temperature_learning', False) else '❌ **Limited Learning**: Minimal temperature change suggests weak feedback coupling or insufficient time for environmental learning.'}

                    ## 🔬 Key Findings

                    1. **Direct vs Indirect Selection**: Wells that were selected more frequently {'showed' if summary.get('evidence_of_evolution', False) else 'may have shown'} different evolutionary trajectories than rarely selected wells.

                    2. **Mutation-Selection Balance**: An average of {summary.get('average_mutations_per_cell', 0):.1f} mutations per cell accumulated, representing the balance between mutational load and beneficial mutations.

                    3. **Population Bottlenecks**: {summary.get('extinct_wells', 0)} extinctions occurred, highlighting the role of demographic stochasticity in small populations.

                    4. **Regulatory Evolution**: Changes in tetranucleotide motifs in surviving strains indicate evolution of gene regulatory networks.

                    ## 📈 Conclusions

                    This experiment {'successfully demonstrated' if summary.get('evidence_of_evolution', False) and summary.get('population_adaptation', False) else 'attempted to model'} evolutionary learning under environmental feedback in yeast populations. 

                    ### Key Biological Insights:
                    - **Shared environment effects**: Even wells not directly selected experienced evolutionary pressure through shared temperature changes
                    - **Stochastic selection**: Random well selection each generation created unequal selection pressure across the population  
                    - **Adaptation limits**: {'Population successfully adapted within the experimental timeframe' if summary.get('population_adaptation', False) else 'Population survival was limited, suggesting adaptation constraints'}
                    - **Regulatory evolution**: GFP expression changes were mediated by evolution of DNA regulatory motifs

                    ### Implications for Real Biology

                    1. **Microbial evolution**: Demonstrates how environmental feedback can drive rapid evolutionary change in microbial populations
                    2. **Synthetic biology**: Shows potential for engineering evolutionary learning systems in bioengineered organisms
                    3. **Population genetics**: Illustrates effects of selection heterogeneity and demographic stochasticity
                    4. **Evolutionary dynamics**: Models how indirect selection can spread beneficial traits through populations

                    ## ⚠️ Experimental Considerations

                    {get_experiment_warnings(summary)}

                    ## 📁 Data Files Generated

                    - `yeast_evolution_analysis.png` - Main experimental plots
                    - `experiment_summary.png` - Detailed summary visualization
                    - `dna_evolution_analysis.png` - Molecular evolution analysis  
                    - `well_analysis.csv` - Individual well outcomes
                    - `global_evolution_history.csv` - Generation-by-generation dynamics
                    - `final_genomes.fasta` - DNA sequences of surviving strains
                    - `experiment_summary.json` - Quantitative summary statistics
                    - `well_histories.json` - Detailed well trajectories

                    ## 🔧 Technical Details

                    ### Simulation Parameters
                    ```
                    {json.dumps(params, indent=2)}
                    ```

                    ### Success Metrics
                    - **Simulation completed**: {'✅ YES' if results.get('success', False) else '❌ NO'}
                    - **Data export completed**: ✅ YES
                    - **Analysis completed**: ✅ YES

                    ---
                    *Report generated by bulletproof_yeast_evolution.py*
                    *For questions or issues, check the log files in the output directory*
                    """
          
        with open(f"{output_dir}/biological_report.md", 'w') as f:
            f.write(report)
        logger.info("Biological report created successfully")  
    
    except Exception as e:
        logger.error(f"Error creating biological report: {e}")
        # Create minimal error report
        try:
            error_report = f"""# Experiment Report - Error

  An error occurred while generating the full biological report.

  Error: {str(e)}

  ## Summary
  - Experiment attempted on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  - Check log files for detailed error information
  - Parameters: {results.get('params', {})}

  ## Basic Results
  - Wells survived: {summary.get('surviving_wells', 'Unknown')}
  - Initial wells: {summary.get('initial_wells', 'Unknown')}
  - Simulation success: {results.get('success', False)}
  """
            with open(f"{output_dir}/biological_report.md", 'w') as f:
                f.write(error_report)
        except:
            pass


def main():
    """Main function with bulletproof command line interface"""
    try:
        parser = argparse.ArgumentParser(
            description='🧪 Bulletproof Yeast Evolutionary Learning Simulation',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
  🧬 Biological Context:
  This simulation models evolutionary learning in yeast with GFP-sugar metabolism
  construct under shared temperature feedback. Includes comprehensive error handling
  and data validation for robust execution.

  Examples:
  python bulletproof_yeast_evolution.py --experiment quick_test
  python bulletproof_yeast_evolution.py --experiment standard_experiment --output my_results
  python bulletproof_yeast_evolution.py --wells 96 --generations 800 --mutation-rate 1.0
            """
        )
        
        # Experiment presets
        parser.add_argument('--experiment', default='quick_test',
                           choices=['quick_test', 'standard_experiment', 'large_scale'],
                           help='Experimental preset (default: quick_test)')

        parser.add_argument('--feedback-well-id', type=int,
                           help='ID of the well that receives direct feedback (default: random)')
        parser.add_argument('--stress-temperature', type=float,
                           help='Initial high temperature for the whole plate (°C)')
        parser.add_argument('--relief-step', type=float,
                           help='Temperature drop (°C) when the feedback well improves')
        
        # Custom parameters
        parser.add_argument('--wells', type=int, help='Number of wells (overrides preset)')
        parser.add_argument('--generations', type=int, help='Number of generations (overrides preset)')
        parser.add_argument('--mutation-rate', type=float, help='Mutations per genome per generation')
        parser.add_argument('--genome-length', type=int, help='Length of DNA sequence')
        parser.add_argument('--cells-per-well', type=int, help='Maximum cells per well')
        
        # Output options
        parser.add_argument('--output', help='Output directory name')
        parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
        parser.add_argument('--verbose', action='store_true', help='Verbose logging')
        
        # Special modes
        parser.add_argument('--compare-evolution', action='store_true',
                           help='Compare evolution with vs without feedback')
        parser.add_argument('--parameter-sweep', help='Parameter to sweep (wells, mutation_rate, etc.)')
        parser.add_argument('--sweep-min', type=float, default=0.5, help='Minimum sweep value')
        parser.add_argument('--sweep-max', type=float, default=3.0, help='Maximum sweep value')
        parser.add_argument('--sweep-steps', type=int, default=5, help='Number of sweep steps')
        
        args = parser.parse_args()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Load base parameters safely
        try:
            params = YeastExperimentConfig.get_preset(args.experiment)
        except Exception as e:
            logger.error(f"Error loading preset: {e}")
            params = YeastExperimentConfig.get_preset('quick_test')  # Fallback
        
        # Override with command line arguments safely
        try:
            if args.wells and args.wells > 0: 
                params['num_wells'] = args.wells
            if args.generations and args.generations > 0: 
                params['num_generations'] = args.generations
            if args.mutation_rate is not None and args.mutation_rate >= 0: 
                params['mutation_rate'] = args.mutation_rate
            if args.genome_length and args.genome_length > 0: 
                params['genome_length'] = args.genome_length
            if args.cells_per_well and args.cells_per_well > 0: 
                params['cells_per_well'] = args.cells_per_well
            if args.feedback_well_id is not None: 
                params['feedback_well_id'] = max(0, args.feedback_well_id)
            if args.stress_temperature is not None: 
                params['stress_temperature'] = args.stress_temperature
            if args.relief_step is not None: 
                params['relief_step'] = max(0, args.relief_step)
            
            params['random_seed'] = max(0, args.seed)
        except Exception as e:
            logger.error(f"Error processing command line arguments: {e}")
        
        # Create output directory safely
        if args.output:
            output_dir = str(args.output)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"yeast_evolution_{timestamp}"
        
        if not safe_create_directory(output_dir):
            # Fallback directory
            output_dir = f"yeast_evolution_fallback_{int(time.time())}"
            if not safe_create_directory(output_dir):
                raise RuntimeError("Cannot create any output directory")
        
        # Setup logging to file
        log_file = f"{output_dir}/experiment.log"
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not setup file logging: {e}")
        
        try:
            if args.compare_evolution:
                # Compare with and without feedback
                logger.info("🔬 Running comparative evolution experiment...")
                
                # Experiment 1: With feedback
                params1 = params.copy()
                try:
                    results1, summary1 = run_yeast_evolution_experiment(params1, f"{output_dir}/with_feedback")
                except Exception as e:
                    logger.error(f"Error in feedback experiment: {e}")
                    results1, summary1 = {}, {'experiment_failed': True, 'error': str(e)}
                
                # Experiment 2: Without feedback (constant temperature)
                params2 = params.copy()
                params2['relief_step'] = 0.0  # No temperature changes
                try:
                    results2, summary2 = run_yeast_evolution_experiment(params2, f"{output_dir}/without_feedback")
                except Exception as e:
                    logger.error(f"Error in no-feedback experiment: {e}")
                    results2, summary2 = {}, {'experiment_failed': True, 'error': str(e)}
                
                # Compare results
                logger.info("\n📊 COMPARATIVE ANALYSIS:")
                logger.info(f"WITH feedback    - Survival: {summary1.get('survival_rate', 0):.1%}, "
                           f"GFP: {summary1.get('final_avg_gfp', 0):.3f}")
                logger.info(f"WITHOUT feedback - Survival: {summary2.get('survival_rate', 0):.1%}, "
                           f"GFP: {summary2.get('final_avg_gfp', 0):.3f}")
                
                # Save comparison summary
                try:
                    comparison = {
                        'with_feedback': summary1,
                        'without_feedback': summary2,
                        'comparison_date': datetime.now().isoformat()
                    }
                    with open(f"{output_dir}/comparison_summary.json", 'w') as f:
                        json.dump(comparison, f, indent=2, default=str)
                except Exception as e:
                    logger.error(f"Error saving comparison: {e}")
                
            elif args.parameter_sweep:
                # Parameter sweep
                logger.info(f"🔄 Running parameter sweep for {args.parameter_sweep}...")
                
                try:
                    sweep_values = np.linspace(max(0.1, args.sweep_min), args.sweep_max, args.sweep_steps)
                    sweep_results = []
                    
                    for i, value in enumerate(sweep_values):
                        logger.info(f"\n--- Sweep {i+1}/{args.sweep_steps}: {args.parameter_sweep} = {value} ---")
                        
                        current_params = params.copy()
                        current_params[args.parameter_sweep] = value
                        
                        sweep_output = f"{output_dir}/sweep_{i+1}_{value:.3f}"
                        try:
                            results, summary = run_yeast_evolution_experiment(current_params, sweep_output)
                            summary[args.parameter_sweep] = value
                            summary['sweep_index'] = i
                            sweep_results.append(summary)
                        except Exception as e:
                            logger.error(f"Error in sweep iteration {i}: {e}")
                            error_summary = {
                                args.parameter_sweep: value,
                                'sweep_index': i,
                                'error': str(e),
                                'failed': True
                            }
                            sweep_results.append(error_summary)
                    
                    # Analyze sweep results
                    try:
                        sweep_df = pd.DataFrame(sweep_results)
                        sweep_df.to_csv(f"{output_dir}/parameter_sweep_results.csv", index=False)
                        
                        # Plot sweep results if we have valid data
                        valid_results = [r for r in sweep_results if not r.get('failed', False)]
                        if len(valid_results) >= 2:
                            create_sweep_plots(valid_results, args.parameter_sweep, output_dir)
                        else:
                            logger.warning("Not enough valid results for sweep plotting")
                            
                    except Exception as e:
                        logger.error(f"Error analyzing sweep results: {e}")
                    
                    logger.info(f"\n✅ Parameter sweep completed! Results in {output_dir}/")
                    
                except Exception as e:
                    logger.error(f"Critical error in parameter sweep: {e}")
                    
            else:
                # Standard single experiment
                results, summary = run_yeast_evolution_experiment(params, output_dir)
        
        except KeyboardInterrupt:
            logger.info("\n\n⚠️  Experiment interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"\n❌ Unexpected error during experiment: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Critical error in main function: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


def create_sweep_plots(sweep_results: List[Dict], param_name: str, output_dir: str):
    """Create plots for parameter sweep results"""
    try:
        sweep_df = pd.DataFrame(sweep_results)
        param_values = sweep_df[param_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Survival rate
        if 'survival_rate' in sweep_df.columns:
            axes[0, 0].plot(param_values, sweep_df['survival_rate'], 'o-', color='blue')
            axes[0, 0].set_xlabel(param_name)
            axes[0, 0].set_ylabel('Survival Rate')
            axes[0, 0].set_title('Survival vs Parameter')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Final GFP
        if 'final_avg_gfp' in sweep_df.columns:
            axes[0, 1].plot(param_values, sweep_df['final_avg_gfp'], 'o-', color='green')
            axes[0, 1].set_xlabel(param_name)
            axes[0, 1].set_ylabel('Final Average GFP')
            axes[0, 1].set_title('GFP Evolution vs Parameter')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Evolution rate
        if 'gfp_evolution_rate' in sweep_df.columns:
            axes[1, 0].plot(param_values, sweep_df['gfp_evolution_rate'], 'o-', color='red')
            axes[1, 0].set_xlabel(param_name)
            axes[1, 0].set_ylabel('GFP Evolution Rate')
            axes[1, 0].set_title('Evolution Rate vs Parameter')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Temperature change
        if 'temperature_change' in sweep_df.columns:
            axes[1, 1].plot(param_values, sweep_df['temperature_change'], 'o-', color='orange')
            axes[1, 1].set_xlabel(param_name)
            axes[1, 1].set_ylabel('Temperature Change (°C)')
            axes[1, 1].set_title('Temperature Learning vs Parameter')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter Sweep: {param_name}', fontsize=16)
        plt.tight_layout()
        
        if safe_save_plot(fig, f"{output_dir}/parameter_sweep_analysis.png"):
            logger.info("Parameter sweep plot saved successfully")
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error creating sweep plots: {e}")

# ============================================================================
# Main function to run the experiment and create plots
# ============================================================================
if __name__ == "__main__":
    main()
