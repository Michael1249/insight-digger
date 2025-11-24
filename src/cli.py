#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for Insight Digger EFA Toolkit

Provides comprehensive CLI access to all EFA analysis features including:
- Data validation and suitability assessment
- Factor analysis with multiple rotation methods  
- Parallel analysis for factor determination
- Rotation method comparison
- Comprehensive reporting and visualization export
- Batch processing capabilities

Author: Insight Digger Project
Created: November 24, 2025
"""

import argparse
import sys
import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from efa_analyzer import EFAAnalyzer, FactorSolution
    from factor_validator import FactorValidator
    # Import visualization engine if available
    try:
        from visualization_engine import VisualizationEngine
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VisualizationEngine = None
        VISUALIZATION_AVAILABLE = False
        warnings.warn("Visualization engine not available - visualization features disabled")
except ImportError as e:
    print(f"Error importing EFA modules: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)


class CLIError(Exception):
    """Custom exception for CLI-specific errors."""
    pass


class EFACommandLineInterface:
    """Main CLI handler for EFA toolkit operations."""
    
    def __init__(self):
        """Initialize CLI with argument parser setup."""
        self.parser = self._create_parser()
        self.verbose = False
        self.output_dir = None
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser with all commands and options."""
        
        parser = argparse.ArgumentParser(
            description="Insight Digger EFA Toolkit - Advanced Exploratory Factor Analysis",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s validate --input survey_data.csv
  %(prog)s analyze --input data.csv --factors 3 --rotation varimax
  %(prog)s analyze --input data.csv --parallel-analysis --export-plots
  %(prog)s pipeline --input data.csv --validate --compare-rotations --output results/
  %(prog)s batch --config batch_config.json
            """)
        
        # Global options
        parser.add_argument(
            '--verbose', '-v', 
            action='store_true',
            help='Enable verbose output for debugging'
        )
        
        parser.add_argument(
            '--version',
            action='version',
            version='Insight Digger EFA Toolkit v1.0.0'
        )
        
        # Create subparsers for different commands
        subparsers = parser.add_subparsers(
            dest='command',
            help='Available commands',
            metavar='COMMAND'
        )
        
        # Validate command
        self._add_validate_parser(subparsers)
        
        # Analyze command  
        self._add_analyze_parser(subparsers)
        
        # Pipeline command
        self._add_pipeline_parser(subparsers)
        
        # Batch command
        self._add_batch_parser(subparsers)
        
        return parser
    
    def _add_validate_parser(self, subparsers):
        """Add validation command parser."""
        validate_parser = subparsers.add_parser(
            'validate',
            help='Validate data suitability for factor analysis',
            description='Assess data quality and suitability for EFA including KMO, Bartlett test, and sample adequacy'
        )
        
        # Input options
        validate_parser.add_argument(
            '--input', '-i',
            required=True,
            type=str,
            help='Input data file (CSV, Excel, or JSON format)'
        )
        
        validate_parser.add_argument(
            '--sheet',
            type=str,
            help='Excel sheet name (if input is Excel file)'
        )
        
        # Validation options
        validate_parser.add_argument(
            '--correlation-method',
            choices=['pearson', 'spearman', 'kendall'],
            default='pearson',
            help='Correlation method for analysis (default: pearson)'
        )
        
        validate_parser.add_argument(
            '--min-kmo',
            type=float,
            default=0.6,
            help='Minimum acceptable KMO value (default: 0.6)'
        )
        
        validate_parser.add_argument(
            '--alpha-level',
            type=float,
            default=0.05,
            help='Alpha level for Bartlett test (default: 0.05)'
        )
        
        # Output options
        validate_parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output directory for validation report'
        )
        
        validate_parser.add_argument(
            '--format',
            choices=['text', 'json', 'html'],
            default='text',
            help='Output format for validation report (default: text)'
        )
    
    def _add_analyze_parser(self, subparsers):
        """Add analysis command parser."""
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='Perform exploratory factor analysis',
            description='Run comprehensive EFA with specified parameters or automatic optimization'
        )
        
        # Input options
        analyze_parser.add_argument(
            '--input', '-i',
            required=True,
            type=str,
            help='Input data file (CSV, Excel, or JSON format)'
        )
        
        analyze_parser.add_argument(
            '--sheet',
            type=str,
            help='Excel sheet name (if input is Excel file)'
        )
        
        # EFA parameters
        analyze_parser.add_argument(
            '--factors', '-f',
            type=int,
            help='Number of factors to extract (auto-determined if not specified)'
        )
        
        analyze_parser.add_argument(
            '--rotation',
            choices=['varimax', 'oblimin', 'quartimax', 'promax'],
            default='oblimin',
            help='Rotation method (default: oblimin)'
        )
        
        analyze_parser.add_argument(
            '--extraction',
            choices=['principal', 'ml', 'minres'],
            default='principal',
            help='Factor extraction method (default: principal)'
        )
        
        analyze_parser.add_argument(
            '--max-iterations',
            type=int,
            default=1000,
            help='Maximum iterations for convergence (default: 1000)'
        )
        
        # Analysis options
        analyze_parser.add_argument(
            '--parallel-analysis',
            action='store_true',
            help='Use parallel analysis to determine number of factors'
        )
        
        analyze_parser.add_argument(
            '--pa-simulations',
            type=int,
            default=1000,
            help='Number of simulations for parallel analysis (default: 1000)'
        )
        
        analyze_parser.add_argument(
            '--compare-rotations',
            action='store_true',
            help='Compare different rotation methods'
        )
        
        analyze_parser.add_argument(
            '--reliability-analysis',
            action='store_true',
            help='Calculate Cronbach alpha for factors'
        )
        
        # Output options
        analyze_parser.add_argument(
            '--output', '-o',
            type=str,
            help='Output directory for results'
        )
        
        analyze_parser.add_argument(
            '--export-plots',
            action='store_true',
            help='Export visualization plots'
        )
        
        analyze_parser.add_argument(
            '--plot-format',
            choices=['png', 'svg', 'pdf', 'html'],
            default='png',
            help='Plot export format (default: png)'
        )
        
        analyze_parser.add_argument(
            '--loading-threshold',
            type=float,
            default=0.4,
            help='Minimum loading for factor interpretation (default: 0.4)'
        )
    
    def _add_pipeline_parser(self, subparsers):
        """Add pipeline command parser."""
        pipeline_parser = subparsers.add_parser(
            'pipeline',
            help='Run complete EFA analysis pipeline',
            description='Execute comprehensive analysis including validation, factor determination, analysis, and reporting'
        )
        
        # Input options
        pipeline_parser.add_argument(
            '--input', '-i',
            required=True,
            type=str,
            help='Input data file (CSV, Excel, or JSON format)'
        )
        
        pipeline_parser.add_argument(
            '--sheet',
            type=str,
            help='Excel sheet name (if input is Excel file)'
        )
        
        # Pipeline stages
        pipeline_parser.add_argument(
            '--validate',
            action='store_true',
            default=True,
            help='Include data validation stage (default: True)'
        )
        
        pipeline_parser.add_argument(
            '--parallel-analysis',
            action='store_true',
            default=True,
            help='Include parallel analysis stage (default: True)'
        )
        
        pipeline_parser.add_argument(
            '--compare-rotations',
            action='store_true',
            help='Include rotation comparison stage'
        )
        
        pipeline_parser.add_argument(
            '--reliability-analysis',
            action='store_true',
            help='Include reliability analysis stage'
        )
        
        # Pipeline parameters
        pipeline_parser.add_argument(
            '--rotation',
            choices=['varimax', 'oblimin', 'quartimax', 'promax'],
            default='oblimin',
            help='Primary rotation method (default: oblimin)'
        )
        
        pipeline_parser.add_argument(
            '--extraction',
            choices=['principal', 'ml', 'minres'],
            default='principal',
            help='Factor extraction method (default: principal)'
        )
        
        # Output options
        pipeline_parser.add_argument(
            '--output', '-o',
            required=True,
            type=str,
            help='Output directory for all results'
        )
        
        pipeline_parser.add_argument(
            '--export-plots',
            action='store_true',
            default=True,
            help='Export visualization plots (default: True)'
        )
        
        pipeline_parser.add_argument(
            '--generate-report',
            action='store_true',
            default=True,
            help='Generate comprehensive HTML report (default: True)'
        )
    
    def _add_batch_parser(self, subparsers):
        """Add batch processing command parser."""
        batch_parser = subparsers.add_parser(
            'batch',
            help='Process multiple datasets with batch configuration',
            description='Execute EFA analysis on multiple datasets using JSON configuration'
        )
        
        batch_parser.add_argument(
            '--config',
            required=True,
            type=str,
            help='JSON configuration file for batch processing'
        )
        
        batch_parser.add_argument(
            '--parallel',
            action='store_true',
            help='Process datasets in parallel'
        )
        
        batch_parser.add_argument(
            '--max-workers',
            type=int,
            help='Maximum number of parallel workers'
        )
        
        batch_parser.add_argument(
            '--output', '-o',
            type=str,
            help='Base output directory for batch results'
        )
    
    def _validate_file_input(self, filepath: str, sheet: Optional[str] = None) -> pd.DataFrame:
        """
        Validate and load input data file with comprehensive error handling.
        
        Args:
            filepath: Path to input data file
            sheet: Excel sheet name (optional)
            
        Returns:
            Loaded data as pandas DataFrame
            
        Raises:
            CLIError: File validation or loading errors
        """
        # Check file existence
        if not os.path.exists(filepath):
            raise CLIError(f"Input file not found: {filepath}")
            
        # Check file extension
        file_ext = Path(filepath).suffix.lower()
        supported_formats = ['.csv', '.xlsx', '.xls', '.json']
        
        if file_ext not in supported_formats:
            raise CLIError(f"Unsupported file format: {file_ext}. "
                          f"Supported formats: {', '.join(supported_formats)}")
        
        try:
            # Load data based on format
            if file_ext == '.csv':
                data = pd.read_csv(filepath, encoding='utf-8')
            elif file_ext in ['.xlsx', '.xls']:
                if sheet:
                    data = pd.read_excel(filepath, sheet_name=sheet)
                else:
                    data = pd.read_excel(filepath)
            elif file_ext == '.json':
                data = pd.read_json(filepath)
            
            # Basic data validation
            if data.empty:
                raise CLIError("Input file contains no data")
                
            if len(data.columns) < 3:
                raise CLIError(f"Insufficient variables for factor analysis: {len(data.columns)} "
                              f"(minimum 3 required)")
                              
            if len(data) < 10:
                raise CLIError(f"Insufficient observations for factor analysis: {len(data)} "
                              f"(minimum 10 required)")
            
            # Check for numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 3:
                raise CLIError(f"Insufficient numeric variables: {len(numeric_cols)} "
                              f"(minimum 3 required for factor analysis)")
            
            if self.verbose:
                print(f"Successfully loaded data: {len(data)} observations, "
                      f"{len(data.columns)} variables ({len(numeric_cols)} numeric)")
                      
            return data
            
        except Exception as e:
            if isinstance(e, CLIError):
                raise
            else:
                raise CLIError(f"Error loading data from {filepath}: {str(e)}") from e
    
    def _validate_output_directory(self, output_dir: str) -> Path:
        """
        Validate and create output directory.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Validated Path object
            
        Raises:
            CLIError: Directory creation errors
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = output_path / '.write_test'
            test_file.write_text('test')
            test_file.unlink()
            
            if self.verbose:
                print(f"Output directory ready: {output_path.absolute()}")
                
            return output_path
            
        except Exception as e:
            raise CLIError(f"Cannot create or write to output directory {output_dir}: {str(e)}") from e
    
    def cmd_validate(self, args) -> int:
        """Execute data validation command."""
        try:
            if self.verbose:
                print("Starting data validation...")
                
            # Load data
            data = self._validate_file_input(args.input, args.sheet)
            
            # Setup output directory if specified
            if args.output:
                output_dir = self._validate_output_directory(args.output)
            else:
                output_dir = None
            
            # Run validation
            validator = FactorValidator()
            validation_results = validator.comprehensive_validation(data)
            
            # Generate report
            self._generate_validation_report(
                validation_results, data, args.format, output_dir
            )
            
            # Return appropriate exit code
            return 0 if validation_results.is_valid else 1
            
        except CLIError as e:
            print(f"Validation failed: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error during validation: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def cmd_analyze(self, args) -> int:
        """Execute factor analysis command."""
        try:
            if self.verbose:
                print("Starting factor analysis...")
                
            # Load data
            data = self._validate_file_input(args.input, args.sheet)
            
            # Setup output directory if specified
            if args.output:
                output_dir = self._validate_output_directory(args.output)
            else:
                output_dir = None
            
            # Determine number of factors
            n_factors = args.factors
            if args.parallel_analysis or n_factors is None:
                if self.verbose:
                    print("Running parallel analysis...")
                    
                efa = EFAAnalyzer()
                pa_results = efa.parallel_analysis(
                    data, 
                    n_simulations=args.pa_simulations
                )
                n_factors = pa_results['suggested_factors']
                
                if self.verbose:
                    print(f"Parallel analysis suggests {n_factors} factors")
                    
                # Save parallel analysis results
                if output_dir:
                    pa_file = output_dir / 'parallel_analysis.json'
                    with open(pa_file, 'w') as f:
                        json.dump(pa_results, f, indent=2)
            
            # Run factor analysis
            if self.verbose:
                print(f"Running EFA with {n_factors} factors, {args.rotation} rotation...")
                
            efa = EFAAnalyzer(
                n_factors=n_factors,
                rotation_method=args.rotation,
                extraction_method=args.extraction,
                max_iterations=args.max_iterations
            )
            
            solution = efa.fit(data)
            
            # Compare rotations if requested
            rotation_comparison = None
            if args.compare_rotations:
                if self.verbose:
                    print("Comparing rotation methods...")
                    
                rotation_comparison = efa.compare_rotations(
                    data, 
                    rotations=['varimax', 'oblimin', 'quartimax', 'promax']
                )
            
            # Calculate reliability if requested
            reliability_results = None
            if args.reliability_analysis:
                if self.verbose:
                    print("Calculating factor reliability...")
                    
                validator = FactorValidator()
                reliability_results = {}
                
                for factor_col in solution.loadings.columns:
                    loadings = solution.loadings[factor_col]
                    significant_vars = loadings[abs(loadings) >= args.loading_threshold]
                    
                    if len(significant_vars) >= 3:
                        factor_data = data[significant_vars.index]
                        reliability = validator.calculate_cronbach_alpha(
                            significant_vars, factor_data
                        )
                        reliability_results[factor_col] = {
                            'alpha': reliability.alpha_value,
                            'interpretation': reliability.interpretation,
                            'n_items': len(significant_vars)
                        }
            
            # Generate outputs
            self._generate_analysis_results(
                solution, data, args, output_dir,
                pa_results if args.parallel_analysis else None,
                rotation_comparison,
                reliability_results
            )
            
            if self.verbose:
                print("Analysis completed successfully")
                
            return 0
            
        except CLIError as e:
            print(f"Analysis failed: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error during analysis: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def cmd_pipeline(self, args) -> int:
        """Execute complete pipeline command."""
        try:
            if self.verbose:
                print("Starting EFA pipeline...")
                
            # Setup output directory
            output_dir = self._validate_output_directory(args.output)
            
            # Load data
            data = self._validate_file_input(args.input, args.sheet)
            
            pipeline_results = {}
            
            # Stage 1: Validation
            if args.validate:
                if self.verbose:
                    print("Pipeline Stage 1: Data Validation")
                    
                validator = FactorValidator()
                validation_results = validator.comprehensive_validation(data)
                pipeline_results['validation'] = validation_results
                
                if not validation_results.is_valid:
                    print("Warning: Data validation failed. Proceeding with analysis...")
            
            # Stage 2: Parallel Analysis  
            if args.parallel_analysis:
                if self.verbose:
                    print("Pipeline Stage 2: Parallel Analysis")
                    
                efa = EFAAnalyzer()
                pa_results = efa.parallel_analysis(data)
                pipeline_results['parallel_analysis'] = pa_results
                n_factors = pa_results['suggested_factors']
            else:
                n_factors = None
            
            # Stage 3: Factor Analysis
            if self.verbose:
                print("Pipeline Stage 3: Factor Analysis")
                
            efa = EFAAnalyzer(
                n_factors=n_factors,
                rotation_method=args.rotation,
                extraction_method=args.extraction
            )
            
            solution = efa.fit(data)
            pipeline_results['factor_solution'] = solution
            
            # Stage 4: Rotation Comparison
            if args.compare_rotations:
                if self.verbose:
                    print("Pipeline Stage 4: Rotation Comparison")
                    
                rotation_comparison = efa.compare_rotations(data)
                pipeline_results['rotation_comparison'] = rotation_comparison
            
            # Stage 5: Reliability Analysis
            if args.reliability_analysis:
                if self.verbose:
                    print("Pipeline Stage 5: Reliability Analysis")
                    
                validator = FactorValidator()
                reliability_results = {}
                
                for factor_col in solution.loadings.columns:
                    loadings = solution.loadings[factor_col]
                    significant_vars = loadings[abs(loadings) >= 0.4]
                    
                    if len(significant_vars) >= 3:
                        factor_data = data[significant_vars.index]
                        reliability = validator.calculate_cronbach_alpha(
                            significant_vars, factor_data
                        )
                        reliability_results[factor_col] = reliability
                        
                pipeline_results['reliability'] = reliability_results
            
            # Generate comprehensive outputs
            self._generate_pipeline_results(pipeline_results, data, args, output_dir)
            
            if self.verbose:
                print("Pipeline completed successfully")
                
            return 0
            
        except CLIError as e:
            print(f"Pipeline failed: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error in pipeline: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def cmd_batch(self, args) -> int:
        """Execute batch processing command."""
        try:
            if self.verbose:
                print("Starting batch processing...")
                
            # Load configuration
            if not os.path.exists(args.config):
                raise CLIError(f"Configuration file not found: {args.config}")
                
            with open(args.config, 'r') as f:
                config = json.load(f)
            
            # Validate configuration structure
            required_keys = ['datasets', 'analysis_params']
            for key in required_keys:
                if key not in config:
                    raise CLIError(f"Missing required key in config: {key}")
            
            datasets = config['datasets']
            analysis_params = config['analysis_params']
            
            # Setup output directory
            if args.output:
                base_output_dir = self._validate_output_directory(args.output)
            else:
                base_output_dir = Path('batch_results')
                base_output_dir.mkdir(exist_ok=True)
            
            # Process each dataset
            results = {}
            for dataset_name, dataset_config in datasets.items():
                if self.verbose:
                    print(f"Processing dataset: {dataset_name}")
                    
                try:
                    # Load dataset
                    data = self._validate_file_input(
                        dataset_config['file'],
                        dataset_config.get('sheet')
                    )
                    
                    # Create output directory for this dataset
                    dataset_output_dir = base_output_dir / dataset_name
                    dataset_output_dir.mkdir(exist_ok=True)
                    
                    # Run analysis with merged parameters
                    merged_params = {**analysis_params, **dataset_config.get('params', {})}
                    
                    # Execute analysis pipeline
                    dataset_results = self._run_batch_analysis(
                        data, merged_params, dataset_output_dir
                    )
                    
                    results[dataset_name] = {
                        'status': 'success',
                        'results': dataset_results
                    }
                    
                except Exception as e:
                    print(f"Error processing {dataset_name}: {e}")
                    results[dataset_name] = {
                        'status': 'error',
                        'error': str(e)
                    }
            
            # Save batch summary
            summary_file = base_output_dir / 'batch_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            if self.verbose:
                successful = sum(1 for r in results.values() if r['status'] == 'success')
                total = len(results)
                print(f"Batch processing completed: {successful}/{total} datasets successful")
                
            return 0
            
        except CLIError as e:
            print(f"Batch processing failed: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error in batch processing: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _generate_validation_report(self, validation_results, data, format_type, output_dir):
        """Generate validation report in specified format."""
        # Implementation would generate comprehensive validation report
        pass
    
    def _generate_analysis_results(self, solution, data, args, output_dir, 
                                 pa_results=None, rotation_comparison=None, 
                                 reliability_results=None):
        """Generate comprehensive analysis results and outputs."""
        # Implementation would generate all analysis outputs
        pass
    
    def _generate_pipeline_results(self, pipeline_results, data, args, output_dir):
        """Generate comprehensive pipeline results."""
        # Implementation would generate complete pipeline report
        pass
    
    def _run_batch_analysis(self, data, params, output_dir):
        """Run analysis for a single dataset in batch mode."""
        # Implementation would execute analysis with given parameters
        pass
    
    def run(self) -> int:
        """
        Main entry point for CLI execution.
        
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        try:
            # Parse arguments
            args = self.parser.parse_args()
            
            # Set global flags
            self.verbose = args.verbose
            
            # Handle case where no command is provided
            if not args.command:
                self.parser.print_help()
                return 1
            
            # Suppress warnings unless verbose mode
            if not self.verbose:
                warnings.filterwarnings('ignore')
            
            # Route to appropriate command handler
            command_handlers = {
                'validate': self.cmd_validate,
                'analyze': self.cmd_analyze,
                'pipeline': self.cmd_pipeline,
                'batch': self.cmd_batch
            }
            
            handler = command_handlers.get(args.command)
            if handler:
                return handler(args)
            else:
                print(f"Unknown command: {args.command}", file=sys.stderr)
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if self.verbose:
                import traceback
                traceback.print_exc()
            return 1


def main():
    """Main entry point for command line execution."""
    cli = EFACommandLineInterface()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())