# -*- coding: utf-8 -*-
"""
Output Formatting and Validation Module for Insight Digger EFA Toolkit

Provides comprehensive formatting, validation, and export capabilities for EFA results including:
- Standardized result formatting across all output types
- Multi-format export (CSV, JSON, HTML, PDF) with validation
- Professional report generation with statistical summaries
- Interactive visualization export with format validation
- Cross-platform compatibility for file exports
- Memory-efficient processing for large result sets

Author: Insight Digger Project
Created: November 24, 2025
"""

import os
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np


class OutputFormatter:
    """Handles formatting and validation of EFA analysis outputs."""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize output formatter.
        
        Args:
            output_dir: Base directory for outputs (None for current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def validate_output_directory(self) -> bool:
        """
        Validate output directory exists and is writable.
        
        Returns:
            True if directory is valid and writable
            
        Raises:
            OSError: If directory cannot be created or written to
        """
        try:
            # Create directory if it doesn't exist
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = self.output_dir / f".write_test_{self.timestamp}"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
            
        except Exception as e:
            raise OSError(f"Cannot write to output directory {self.output_dir}: {str(e)}") from e
    
    def format_factor_loadings(self, loadings: pd.DataFrame, 
                             threshold: float = 0.4,
                             format_type: str = 'table') -> Union[str, Dict]:
        """
        Format factor loadings with highlighting and interpretation.
        
        Args:
            loadings: Factor loadings matrix
            threshold: Minimum loading for highlighting
            format_type: 'table', 'json', 'html'
            
        Returns:
            Formatted loadings representation
        """
        if format_type == 'table':
            # Create formatted table string
            formatted_lines = []
            
            # Header
            header = f"{'Variable':<20}"
            for col in loadings.columns:
                header += f"{col:>10}"
            header += f"{'Communality':>12}"
            formatted_lines.append(header)
            formatted_lines.append("-" * len(header))
            
            # Rows
            communalities = (loadings ** 2).sum(axis=1)
            for idx, row in loadings.iterrows():
                line = f"{str(idx)[:19]:<20}"
                for val in row:
                    if abs(val) >= threshold:
                        line += f"{val:>10.3f}*"
                    else:
                        line += f"{val:>10.3f} "
                line += f"{communalities[idx]:>11.3f}"
                formatted_lines.append(line)
            
            # Footer
            formatted_lines.append("")
            formatted_lines.append(f"* indicates loading >= {threshold}")
            
            return "\n".join(formatted_lines)
            
        elif format_type == 'json':
            # JSON format with metadata
            result = {
                'loadings': loadings.round(3).to_dict('index'),
                'communalities': (loadings ** 2).sum(axis=1).round(3).to_dict(),
                'threshold': threshold,
                'significant_loadings': {},
                'factor_summary': {}
            }
            
            # Add significant loadings
            for factor in loadings.columns:
                significant = loadings[factor][abs(loadings[factor]) >= threshold]
                result['significant_loadings'][factor] = significant.round(3).to_dict()
                
                # Factor summary
                result['factor_summary'][factor] = {
                    'n_significant_loadings': len(significant),
                    'max_loading': abs(loadings[factor]).max().round(3),
                    'mean_loading': abs(loadings[factor]).mean().round(3)
                }
            
            return result
            
        elif format_type == 'html':
            # HTML table with styling
            html_parts = []
            html_parts.append('<table class="factor-loadings" style="border-collapse: collapse; font-family: monospace;">')
            
            # Header
            html_parts.append('<thead><tr style="background-color: #f0f0f0;">')
            html_parts.append('<th style="border: 1px solid #ddd; padding: 8px;">Variable</th>')
            for col in loadings.columns:
                html_parts.append(f'<th style="border: 1px solid #ddd; padding: 8px;">{col}</th>')
            html_parts.append('<th style="border: 1px solid #ddd; padding: 8px;">Communality</th>')
            html_parts.append('</tr></thead>')
            
            # Rows
            html_parts.append('<tbody>')
            communalities = (loadings ** 2).sum(axis=1)
            for idx, row in loadings.iterrows():
                html_parts.append('<tr>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{idx}</td>')
                
                for val in row:
                    style = "border: 1px solid #ddd; padding: 8px; text-align: center;"
                    if abs(val) >= threshold:
                        style += " background-color: #ffffcc; font-weight: bold;"
                    html_parts.append(f'<td style="{style}">{val:.3f}</td>')
                
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px; text-align: center;">{communalities[idx]:.3f}</td>')
                html_parts.append('</tr>')
            html_parts.append('</tbody></table>')
            
            html_parts.append(f'<p><small>* Highlighted cells indicate loadings >= {threshold}</small></p>')
            
            return "\n".join(html_parts)
        
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
    
    def format_validation_results(self, validation_results, format_type: str = 'text') -> Union[str, Dict]:
        """
        Format validation results with comprehensive statistics.
        
        Args:
            validation_results: ValidationResults object
            format_type: 'text', 'json', 'html'
            
        Returns:
            Formatted validation summary
        """
        if format_type == 'text':
            lines = []
            lines.append("=== DATA VALIDATION RESULTS ===")
            lines.append("")
            
            # Overall status
            status = "✓ VALID" if validation_results.is_valid else "✗ INVALID"
            lines.append(f"Overall Status: {status}")
            lines.append("")
            
            # Statistical measures
            if hasattr(validation_results, 'kmo_score') and validation_results.kmo_score:
                kmo_interpretation = self._interpret_kmo(validation_results.kmo_score)
                lines.append(f"KMO Measure: {validation_results.kmo_score:.3f} ({kmo_interpretation})")
            
            if hasattr(validation_results, 'bartlett_p') and validation_results.bartlett_p is not None:
                bartlett_interpretation = "Significant" if validation_results.bartlett_p < 0.05 else "Non-significant"
                lines.append(f"Bartlett's Test: p = {validation_results.bartlett_p:.6f} ({bartlett_interpretation})")
            
            lines.append("")
            
            # Errors
            if validation_results.errors:
                lines.append("ERRORS:")
                for error in validation_results.errors:
                    lines.append(f"  • {error}")
                lines.append("")
            
            # Warnings
            if validation_results.warnings:
                lines.append("WARNINGS:")
                for warning in validation_results.warnings:
                    lines.append(f"  • {warning}")
                lines.append("")
            
            # Recommendations
            lines.append("RECOMMENDATIONS:")
            if validation_results.is_valid:
                lines.append("  • Data is suitable for factor analysis")
                if hasattr(validation_results, 'kmo_score') and validation_results.kmo_score:
                    if validation_results.kmo_score > 0.8:
                        lines.append("  • Excellent data quality for factor analysis")
                    elif validation_results.kmo_score > 0.7:
                        lines.append("  • Good data quality for factor analysis")
                    else:
                        lines.append("  • Adequate data quality, consider improving correlations")
            else:
                lines.append("  • Address errors before proceeding with factor analysis")
                lines.append("  • Consider data preprocessing or collection of additional data")
            
            return "\n".join(lines)
            
        elif format_type == 'json':
            result = {
                'is_valid': validation_results.is_valid,
                'timestamp': self.timestamp,
                'errors': validation_results.errors,
                'warnings': validation_results.warnings,
                'statistical_measures': {},
                'interpretations': {}
            }
            
            # Add statistical measures if available
            if hasattr(validation_results, 'kmo_score') and validation_results.kmo_score:
                result['statistical_measures']['kmo_score'] = round(validation_results.kmo_score, 3)
                result['interpretations']['kmo'] = self._interpret_kmo(validation_results.kmo_score)
            
            if hasattr(validation_results, 'bartlett_p') and validation_results.bartlett_p is not None:
                result['statistical_measures']['bartlett_p_value'] = validation_results.bartlett_p
                result['interpretations']['bartlett'] = "significant" if validation_results.bartlett_p < 0.05 else "non_significant"
            
            return result
            
        elif format_type == 'html':
            # HTML report with styling
            html_parts = []
            html_parts.append('<div class="validation-report" style="font-family: Arial, sans-serif; max-width: 800px;">')
            html_parts.append('<h2>Data Validation Results</h2>')
            
            # Status badge
            status_color = "#28a745" if validation_results.is_valid else "#dc3545"
            status_text = "VALID" if validation_results.is_valid else "INVALID"
            html_parts.append(f'<div style="background: {status_color}; color: white; padding: 10px; border-radius: 5px; display: inline-block; margin-bottom: 20px;">')
            html_parts.append(f'<strong>Status: {status_text}</strong></div>')
            
            # Statistical measures
            if hasattr(validation_results, 'kmo_score') and validation_results.kmo_score:
                kmo_interpretation = self._interpret_kmo(validation_results.kmo_score)
                html_parts.append('<h3>Statistical Measures</h3>')
                html_parts.append('<table style="border-collapse: collapse; width: 100%; margin-bottom: 20px;">')
                html_parts.append('<tr><td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">KMO Measure</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{validation_results.kmo_score:.3f}</td>')
                html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{kmo_interpretation}</td></tr>')
                
                if hasattr(validation_results, 'bartlett_p') and validation_results.bartlett_p is not None:
                    bartlett_interpretation = "Significant" if validation_results.bartlett_p < 0.05 else "Non-significant"
                    html_parts.append('<tr><td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">Bartlett\'s Test</td>')
                    html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">p = {validation_results.bartlett_p:.6f}</td>')
                    html_parts.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{bartlett_interpretation}</td></tr>')
                
                html_parts.append('</table>')
            
            # Issues section
            if validation_results.errors or validation_results.warnings:
                html_parts.append('<h3>Issues</h3>')
                
                if validation_results.errors:
                    html_parts.append('<h4 style="color: #dc3545;">Errors</h4>')
                    html_parts.append('<ul>')
                    for error in validation_results.errors:
                        html_parts.append(f'<li style="color: #dc3545;">{error}</li>')
                    html_parts.append('</ul>')
                
                if validation_results.warnings:
                    html_parts.append('<h4 style="color: #ffc107;">Warnings</h4>')
                    html_parts.append('<ul>')
                    for warning in validation_results.warnings:
                        html_parts.append(f'<li style="color: #856404;">{warning}</li>')
                    html_parts.append('</ul>')
            
            html_parts.append('</div>')
            
            return "\n".join(html_parts)
        
        else:
            raise ValueError(f"Unsupported format_type: {format_type}")
    
    def export_factor_solution(self, solution, filename: str, 
                             format_type: str = 'csv') -> Path:
        """
        Export factor solution to specified format with validation.
        
        Args:
            solution: FactorSolution object
            filename: Base filename (without extension)
            format_type: 'csv', 'xlsx', 'json'
            
        Returns:
            Path to exported file
            
        Raises:
            ValueError: Invalid format type
            OSError: File writing errors
        """
        # Validate output directory
        self.validate_output_directory()
        
        # Generate timestamped filename
        timestamp_filename = f"{filename}_{self.timestamp}"
        
        if format_type == 'csv':
            # Export as CSV files (multiple files for different components)
            loadings_file = self.output_dir / f"{timestamp_filename}_loadings.csv"
            
            try:
                # Factor loadings
                solution.loadings.to_csv(loadings_file, float_format='%.4f')
                
                # Communalities (if available)
                if hasattr(solution, 'communalities') and solution.communalities is not None:
                    communalities_file = self.output_dir / f"{timestamp_filename}_communalities.csv"
                    solution.communalities.to_csv(communalities_file, header=['Communality'], float_format='%.4f')
                
                # Variance explained
                if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
                    variance_file = self.output_dir / f"{timestamp_filename}_variance.csv"
                    pd.DataFrame({
                        'Factor': solution.variance_explained.index,
                        'Variance_Explained': solution.variance_explained.values,
                        'Cumulative_Variance': solution.variance_explained.cumsum().values
                    }).to_csv(variance_file, index=False, float_format='%.4f')
                
                return loadings_file
                
            except Exception as e:
                raise OSError(f"Error exporting CSV: {str(e)}") from e
                
        elif format_type == 'xlsx':
            # Export as Excel file with multiple sheets
            excel_file = self.output_dir / f"{timestamp_filename}.xlsx"
            
            try:
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    # Factor loadings
                    solution.loadings.to_excel(writer, sheet_name='Factor_Loadings', float_format='%.4f')
                    
                    # Communalities
                    if hasattr(solution, 'communalities') and solution.communalities is not None:
                        solution.communalities.to_excel(writer, sheet_name='Communalities', float_format='%.4f')
                    
                    # Variance explained
                    if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
                        variance_df = pd.DataFrame({
                            'Factor': solution.variance_explained.index,
                            'Variance_Explained': solution.variance_explained.values,
                            'Cumulative_Variance': solution.variance_explained.cumsum().values
                        })
                        variance_df.to_excel(writer, sheet_name='Variance_Explained', index=False, float_format='%.4f')
                    
                    # Summary sheet
                    summary_data = {
                        'Analysis_Timestamp': [self.timestamp],
                        'Number_of_Factors': [solution.loadings.shape[1]],
                        'Number_of_Variables': [solution.loadings.shape[0]],
                        'Rotation_Method': [getattr(solution, 'rotation_method', 'Unknown')],
                        'Extraction_Method': [getattr(solution, 'extraction_method', 'Unknown')]
                    }
                    
                    if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
                        summary_data['Total_Variance_Explained'] = [solution.variance_explained.sum()]
                    
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                
                return excel_file
                
            except Exception as e:
                raise OSError(f"Error exporting Excel: {str(e)}") from e
                
        elif format_type == 'json':
            # Export as JSON with comprehensive structure
            json_file = self.output_dir / f"{timestamp_filename}.json"
            
            try:
                export_data = {
                    'metadata': {
                        'timestamp': self.timestamp,
                        'format_version': '1.0',
                        'analysis_type': 'exploratory_factor_analysis'
                    },
                    'factor_solution': {
                        'loadings': solution.loadings.round(4).to_dict('index'),
                        'number_of_factors': solution.loadings.shape[1],
                        'number_of_variables': solution.loadings.shape[0],
                        'rotation_method': getattr(solution, 'rotation_method', None),
                        'extraction_method': getattr(solution, 'extraction_method', None)
                    }
                }
                
                # Add optional components
                if hasattr(solution, 'communalities') and solution.communalities is not None:
                    export_data['factor_solution']['communalities'] = solution.communalities.round(4).to_dict()
                
                if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
                    export_data['factor_solution']['variance_explained'] = {
                        'by_factor': solution.variance_explained.round(4).to_dict(),
                        'cumulative': solution.variance_explained.cumsum().round(4).to_dict(),
                        'total': round(solution.variance_explained.sum(), 4)
                    }
                
                if hasattr(solution, 'eigenvalues') and solution.eigenvalues is not None:
                    if isinstance(solution.eigenvalues, np.ndarray):
                        export_data['factor_solution']['eigenvalues'] = solution.eigenvalues.tolist()
                    else:
                        export_data['factor_solution']['eigenvalues'] = list(solution.eigenvalues)
                
                # Write JSON with proper formatting
                with open(json_file, 'w') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                return json_file
                
            except Exception as e:
                raise OSError(f"Error exporting JSON: {str(e)}") from e
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}. "
                           f"Supported formats: csv, xlsx, json")
    
    def generate_comprehensive_report(self, solution, validation_results=None, 
                                    parallel_analysis=None, reliability_results=None,
                                    filename: str = "efa_report") -> Path:
        """
        Generate comprehensive HTML report with all analysis results.
        
        Args:
            solution: FactorSolution object
            validation_results: ValidationResults object (optional)
            parallel_analysis: Parallel analysis results (optional)
            reliability_results: Reliability analysis results (optional)
            filename: Base filename for report
            
        Returns:
            Path to generated HTML report
        """
        # Validate output directory
        self.validate_output_directory()
        
        report_file = self.output_dir / f"{filename}_{self.timestamp}.html"
        
        html_content = self._build_comprehensive_html_report(
            solution, validation_results, parallel_analysis, reliability_results
        )
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_file
            
        except Exception as e:
            raise OSError(f"Error generating report: {str(e)}") from e
    
    def _interpret_kmo(self, kmo_score: float) -> str:
        """Interpret KMO score according to standard guidelines."""
        if kmo_score >= 0.9:
            return "Excellent"
        elif kmo_score >= 0.8:
            return "Very Good"
        elif kmo_score >= 0.7:
            return "Good"
        elif kmo_score >= 0.6:
            return "Adequate"
        elif kmo_score >= 0.5:
            return "Poor"
        else:
            return "Unacceptable"
    
    def _build_comprehensive_html_report(self, solution, validation_results, 
                                       parallel_analysis, reliability_results) -> str:
        """Build comprehensive HTML report with all analysis components."""
        
        html_parts = []
        
        # HTML header with styling
        html_parts.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EFA Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
        .section { margin-bottom: 30px; }
        .section h2 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .highlight { background-color: #ffffcc; font-weight: bold; }
        .status-valid { background: #28a745; color: white; padding: 5px 10px; border-radius: 3px; }
        .status-invalid { background: #dc3545; color: white; padding: 5px 10px; border-radius: 3px; }
        .warning { color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 5px; }
        .error { color: #721c24; background-color: #f8d7da; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
        """)
        
        # Report header
        html_parts.append('<div class="header">')
        html_parts.append('<h1>Exploratory Factor Analysis Report</h1>')
        html_parts.append(f'<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')
        html_parts.append(f'<p><strong>Analysis ID:</strong> {self.timestamp}</p>')
        html_parts.append('</div>')
        
        # Executive Summary
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Executive Summary</h2>')
        
        summary_items = []
        summary_items.append(f"Number of factors extracted: {solution.loadings.shape[1]}")
        summary_items.append(f"Number of variables analyzed: {solution.loadings.shape[0]}")
        
        if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
            total_variance = solution.variance_explained.sum()
            summary_items.append(f"Total variance explained: {total_variance:.1%}")
            
        if hasattr(solution, 'rotation_method') and solution.rotation_method:
            summary_items.append(f"Rotation method: {solution.rotation_method}")
            
        html_parts.append('<ul>')
        for item in summary_items:
            html_parts.append(f'<li>{item}</li>')
        html_parts.append('</ul>')
        html_parts.append('</div>')
        
        # Validation Results (if available)
        if validation_results:
            html_parts.append('<div class="section">')
            html_parts.append('<h2>Data Validation Results</h2>')
            html_parts.append(self.format_validation_results(validation_results, 'html'))
            html_parts.append('</div>')
        
        # Factor Loadings
        html_parts.append('<div class="section">')
        html_parts.append('<h2>Factor Loadings Matrix</h2>')
        html_parts.append(self.format_factor_loadings(solution.loadings, format_type='html'))
        html_parts.append('</div>')
        
        # Variance Explained (if available)
        if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
            html_parts.append('<div class="section">')
            html_parts.append('<h2>Variance Explained</h2>')
            
            variance_table = '<table>'
            variance_table += '<tr><th>Factor</th><th>Variance Explained</th><th>Cumulative Variance</th></tr>'
            
            cumulative = 0
            for factor, variance in solution.variance_explained.items():
                cumulative += variance
                variance_table += f'<tr><td>{factor}</td><td>{variance:.1%}</td><td>{cumulative:.1%}</td></tr>'
            
            variance_table += f'<tr style="font-weight: bold;"><td>Total</td><td>{solution.variance_explained.sum():.1%}</td><td>{cumulative:.1%}</td></tr>'
            variance_table += '</table>'
            
            html_parts.append(variance_table)
            html_parts.append('</div>')
        
        # Parallel Analysis (if available)
        if parallel_analysis:
            html_parts.append('<div class="section">')
            html_parts.append('<h2>Parallel Analysis Results</h2>')
            
            pa_info = f"<p>Suggested number of factors: <strong>{parallel_analysis['suggested_factors']}</strong></p>"
            pa_info += f"<p>Based on {parallel_analysis['n_simulations']} simulations at {parallel_analysis['percentile']}th percentile</p>"
            
            html_parts.append(pa_info)
            html_parts.append('</div>')
        
        # Reliability Results (if available)
        if reliability_results:
            html_parts.append('<div class="section">')
            html_parts.append('<h2>Factor Reliability Analysis</h2>')
            
            reliability_table = '<table>'
            reliability_table += '<tr><th>Factor</th><th>Cronbach\'s Alpha</th><th>Items</th><th>Interpretation</th></tr>'
            
            for factor, reliability in reliability_results.items():
                if hasattr(reliability, 'alpha_value'):
                    alpha = reliability.alpha_value
                    interpretation = reliability.interpretation
                    n_items = len(reliability.item_statistics) if hasattr(reliability, 'item_statistics') else 'N/A'
                else:
                    alpha = reliability.get('alpha', 'N/A')
                    interpretation = reliability.get('interpretation', 'N/A')
                    n_items = reliability.get('n_items', 'N/A')
                
                reliability_table += f'<tr><td>{factor}</td><td>{alpha:.3f}</td><td>{n_items}</td><td>{interpretation}</td></tr>'
            
            reliability_table += '</table>'
            html_parts.append(reliability_table)
            html_parts.append('</div>')
        
        # Footer
        html_parts.append('<div class="section">')
        html_parts.append('<hr>')
        html_parts.append('<p><small>Report generated by Insight Digger EFA Toolkit</small></p>')
        html_parts.append('</div>')
        
        html_parts.append('</body></html>')
        
        return '\n'.join(html_parts)


class ResultsValidator:
    """Validates EFA results for consistency and quality."""
    
    @staticmethod
    def validate_factor_solution(solution) -> Dict[str, Any]:
        """
        Validate factor solution for consistency and quality.
        
        Args:
            solution: FactorSolution object
            
        Returns:
            Validation report dictionary
        """
        validation_report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_metrics': {}
        }
        
        try:
            # Check loadings matrix
            if solution.loadings is None:
                validation_report['errors'].append("Factor loadings matrix is missing")
                validation_report['is_valid'] = False
                return validation_report
            
            # Check for NaN values
            if solution.loadings.isnull().any().any():
                validation_report['errors'].append("Factor loadings contain NaN values")
                validation_report['is_valid'] = False
            
            # Check loading ranges
            max_loading = solution.loadings.abs().max().max()
            if max_loading > 1.0:
                validation_report['warnings'].append(f"Unusually high loading detected: {max_loading:.3f}")
            
            # Check communalities (if available)
            if hasattr(solution, 'communalities') and solution.communalities is not None:
                if (solution.communalities < 0).any():
                    validation_report['warnings'].append("Negative communalities detected (Heywood case)")
                
                if (solution.communalities > 1).any():
                    validation_report['warnings'].append("Communalities > 1 detected")
                
                low_communalities = solution.communalities[solution.communalities < 0.25]
                if len(low_communalities) > 0:
                    validation_report['warnings'].append(f"{len(low_communalities)} variables have low communalities (< 0.25)")
            
            # Quality metrics
            validation_report['quality_metrics'] = {
                'n_factors': solution.loadings.shape[1],
                'n_variables': solution.loadings.shape[0],
                'max_loading': round(max_loading, 3),
                'mean_loading': round(solution.loadings.abs().mean().mean(), 3)
            }
            
            if hasattr(solution, 'variance_explained') and solution.variance_explained is not None:
                validation_report['quality_metrics']['total_variance_explained'] = round(solution.variance_explained.sum(), 3)
            
        except Exception as e:
            validation_report['errors'].append(f"Validation error: {str(e)}")
            validation_report['is_valid'] = False
        
        return validation_report
    
    @staticmethod
    def validate_output_files(file_paths: List[Path]) -> Dict[str, Any]:
        """
        Validate that output files were created successfully and have expected content.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            File validation report
        """
        validation_report = {
            'all_files_valid': True,
            'file_status': {},
            'total_files': len(file_paths),
            'valid_files': 0,
            'errors': []
        }
        
        for file_path in file_paths:
            file_validation = {
                'exists': False,
                'readable': False,
                'size_bytes': 0,
                'errors': []
            }
            
            try:
                # Check existence
                if file_path.exists():
                    file_validation['exists'] = True
                    file_validation['size_bytes'] = file_path.stat().st_size
                    
                    # Check readability
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            f.read(100)  # Read first 100 characters
                        file_validation['readable'] = True
                        validation_report['valid_files'] += 1
                        
                    except Exception as e:
                        file_validation['errors'].append(f"File not readable: {str(e)}")
                        validation_report['all_files_valid'] = False
                        
                else:
                    file_validation['errors'].append("File does not exist")
                    validation_report['all_files_valid'] = False
                    
            except Exception as e:
                file_validation['errors'].append(f"Validation error: {str(e)}")
                validation_report['all_files_valid'] = False
            
            validation_report['file_status'][str(file_path)] = file_validation
        
        return validation_report