#!/usr/bin/env python3
"""
Example script demonstrating the updated particle tagging configuration system.

This script shows how to use the new centralized configuration
instead of hardcoded paths for particle tagging functions.
"""

import os
import sys
import tangos

# Add the darktrace directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from darktrace.tagging.angular_momentum_tagging import angmom_tag_over_full_sim
from darktrace.tagging.tagging_wrapper_func import tag_particles

def main():
    """Main function demonstrating the updated particle tagging system."""
    
    print("=== Particle Tagging Configuration Demo ===")
    print()
    
    # Show current configuration
    print("Current configuration paths:")
    print(f"  Tangos path: {config.get_path('tangos_path')}")
    print(f"  Pynbody path: {config.get_path('pynbody_path')}")
    print(f"  Manual halo num path: {config.get_path('manual_halonum_path')}")
    print()
    
    # Example simulation name (modify as needed)
    sim_name = "Halo1459"
    db_file = f"{sim_name}.db"
    
    print(f"Loading simulation: {sim_name}")
    print(f"Database file: {db_file}")
    print()
    
    try:
        # Initialize tangos database using config
        tangos.init_db(os.path.join(config.get_path("tangos_path"), db_file))
        
        # Get simulation object
        sim_name_dmo = f"{sim_name}_DMO"
        DMOsim = tangos.get_simulation(sim_name_dmo)
        
        print(f"Successfully loaded simulation: {DMOsim.path}")
        print()
        
        # Example 1: Use angular momentum tagging with config paths
        print("Example 1: Angular momentum tagging (using config paths)")
        df_tagged = angmom_tag_over_full_sim(
            DMOsim, 
            halonumber=1, 
            free_param_value=config.get("tagging", "ftag")  # Use config parameter
        )
        
        print(f"Tagged {len(df_tagged)} particles across {len(df_tagged['t'].unique())} timesteps")
        print(f"Stellar mass range: {df_tagged['mstar'].min():.2e} to {df_tagged['mstar'].max():.2e} M_sun")
        print()
        
        # Example 2: Use wrapper function with method selection
        print("Example 2: Using wrapper function with method selection")
        df_tagged_recursive = tag_particles(
            DMOsim,
            tagging_method='angular momentum recursive',
            free_param_val=0.005
        )
        
        print(f"Recursive tagging result: {len(df_tagged_recursive)} particles")
        print()
        
        # Example 3: Show how path resolution works
        print("Example 3: Path resolution demonstration")
        print("When pynbody_path is None -> uses config.get_path('pynbody_path')")
        print("When pynbody_path is provided -> uses the provided path")
        print()
        
        return True
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        return False

def demonstrate_config_access():
    """Demonstrate different ways to access configuration."""
    
    print("=== Configuration Access Examples ===")
    print()
    
    # Access paths
    print("Path access:")
    print(f"  config.get_path('tangos_path'): {config.get_path('tangos_path')}")
    print(f"  config.get_path('pynbody_path'): {config.get_path('pynbody_path')}")
    print()
    
    # Access configuration values
    print("Configuration values:")
    print(f"  config.get('tagging', 'ftag'): {config.get('tagging', 'ftag')}")
    print(f"  config.get('tagging', 'method'): {config.get('tagging', 'method')}")
    print(f"  config.get('darklight', 'n'): {config.get('darklight', 'n')}")
    print()
    
    # Get all paths
    print("All available paths:")
    paths = config.get_all_paths()
    for key, path in paths.items():
        print(f"  {key}: {path}")
    print()

def show_migration_guide():
    """Show migration guide from old to new system."""
    
    print("=== Migration Guide ===")
    print()
    print("Before (hardcoded paths):")
    print("  tangos.init_db('/scratch/dp101/shared/EDGE/tangos/Halo1459.db')")
    print("  simfn = join('/scratch/dp101/shared/EDGE/', 'output_00000')")
    print()
    print("After (config system):")
    print("  tangos.init_db(join(config.get_path('tangos_path'), 'Halo1459.db'))")
    print("  simfn = join(config.get_path('pynbody_path'), 'output_00000')")
    print()

if __name__ == "__main__":
    # Check if config file exists
    config_file = os.path.join(os.path.dirname(__file__), "config", "config.json")
    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found at {config_file}")
        print("Please create the config file with appropriate paths for your system.")
        sys.exit(1)
    
    # Run demonstration
    success = main()
    
    if success:
        demonstrate_config_access()
        show_migration_guide()
        
        print("=== Demo completed successfully! ===")
        print()
        print("To use in your own scripts:")
        print("1. Import: from config import config")
        print("2. Use paths: config.get_path('pynbody_path')")
        print("3. Modify config.json for your environment")
    else:
        print("Demo failed!")
        sys.exit(1)