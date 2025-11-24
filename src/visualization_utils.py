# -*- coding: utf-8 -*-
"""
EFA Visualization Utilities Module

Provides publication-quality visualizations for exploratory factor analysis results.
Includes scree plots, factor loadings heatmaps, biplots, and interactive visualizations
optimized for Jupyter notebook environments.

Author: Insight Digger Project  
Created: November 24, 2025
"""

import warnings
from typing import Optional, Tuple, Dict, List, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import io
import base64

# Try to import scipy components with graceful degradation
try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    SCIPY_CLUSTERING_AVAILABLE = True
except ImportError:
    SCIPY_CLUSTERING_AVAILABLE = False
    warnings.warn("Scipy clustering not available. Variable clustering will be disabled.", UserWarning)

# Try to import interactive visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available - interactive visualizations will be disabled")


class PlotConfig:
    """Configuration container for plot styling."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[float, float] = (10, 8),
                 dpi: int = 100, color_palette: str = 'viridis',
                 publication_ready: bool = False):
        self.style = style
        self.figsize = figsize
        self.dpi = dpi
        self.color_palette = color_palette
        self.publication_ready = publication_ready


class EFAVisualizer:
    """
    Publication-Quality Visualization for EFA Results
    
    Provides comprehensive visualization capabilities for factor analysis including
    scree plots, factor loadings visualizations, biplots, and interactive plots
    optimized for Jupyter notebook environments.
    
    Attributes:
        style (str): Matplotlib style sheet
        publication_ready (bool): Use publication-quality settings
        color_palette (str): Color scheme for plots
        figure_size (Tuple): Default figure size in inches
        dpi (int): Dots per inch for outputs
    """
    
    def __init__(self,
                 style: str = 'seaborn-v0_8',
                 publication_ready: bool = False,
                 color_palette: str = 'viridis',
                 figure_size: Tuple[float, float] = (10, 8),
                 dpi: int = 100):
        """
        Initialize EFA Visualizer.
        
        Args:
            style: Matplotlib style sheet ('seaborn-v0_8', 'ggplot', 'bmh')
            publication_ready: Use publication-quality settings (300 DPI, vector fonts)
            color_palette: Color scheme ('viridis', 'plasma', 'RdBu', 'Set1')
            figure_size: Default figure size in inches (width, height)
            dpi: Dots per inch for raster outputs
            
        Raises:
            ValueError: Invalid style or color palette
        """
        self.style = style
        self.publication_ready = publication_ready
        self.color_palette = color_palette
        self.figure_size = figure_size
        self.dpi = dpi if not publication_ready else 300
        
        # Apply publication settings
        if self.publication_ready:
            self._setup_publication_style()
        
        # Validate settings
        self._validate_settings()
    
    def _setup_publication_style(self):
        """Configure publication-ready plotting parameters."""
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'font.family': 'serif',
            'text.usetex': False  # Avoid LaTeX dependency for Binder
        })
    
    def _validate_settings(self):
        """Validate visualization settings."""
        valid_palettes = ['viridis', 'plasma', 'RdBu', 'Set1', 'tab10']
        if self.color_palette not in valid_palettes:
            warnings.warn(f"Color palette '{self.color_palette}' may not be available")
    
    def plot_scree(self, eigenvalues: np.ndarray, 
                   n_factors: Optional[int] = None,
                   title: str = "Scree Plot",
                   show_kaiser_criterion: bool = True,
                   show_parallel_analysis: bool = False,
                   variance_explained: Optional[np.ndarray] = None,
                   figsize: Optional[Tuple[int, int]] = None,
                   save_path: Optional[str] = None,
                   interactive: bool = False) -> Union[Tuple[Figure, Axes], Any]:
        """
        Create comprehensive scree plot for eigenvalue visualization.
        
        Args:
            eigenvalues: Array of eigenvalues from factor analysis
            n_factors: Number of factors to highlight (optional)
            title: Plot title
            show_kaiser_criterion: Show eigenvalue > 1.0 threshold line
            show_parallel_analysis: Show parallel analysis comparison
            variance_explained: Array of variance explained ratios (optional)
            figsize: Figure size (width, height)
            save_path: Path to save plot (optional)
            interactive: Whether to create interactive plotly version
            
        Returns:
            Tuple of (Figure, Axes) objects or plotly figure
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_scree_plot(eigenvalues, variance_explained, title)
        
        if figsize is None:
            figsize = self.figure_size
        
        # Create matplotlib version with two subplots
        if variance_explained is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None
        
        # Left plot: Eigenvalues
        factors = np.arange(1, len(eigenvalues) + 1)
        ax1.plot(factors, eigenvalues, 'bo-', linewidth=2, markersize=8, label='Eigenvalues')
        
        if show_kaiser_criterion:
            ax1.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='Kaiser Criterion (λ=1)')
        
        # Highlight extracted factors
        if n_factors:
            ax1.axvline(x=n_factors + 0.5, color='g', linestyle=':', alpha=0.7, 
                       label=f'Extracted factors ({n_factors})')
            # Shade extracted factors
            ax1.axvspan(0.5, n_factors + 0.5, alpha=0.2, color='green')
        
        ax1.set_xlabel('Factor Number')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalues by Factor')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add eigenvalue annotations for significant factors
        for i, (factor, eigenval) in enumerate(zip(factors, eigenvalues)):
            if eigenval >= 1.0 or i < 5:  # Annotate significant factors or first 5
                ax1.annotate(f'{eigenval:.2f}', 
                           (factor, eigenval), 
                           textcoords="offset points", 
                           xytext=(0,10), 
                           ha='center', fontsize=9)
        
        # Right plot: Variance explained (if available)
        if variance_explained is not None and ax2 is not None:
            cumulative_var = np.cumsum(variance_explained)
            
            ax2.bar(factors, variance_explained * 100, alpha=0.7, color='skyblue', 
                   label='Individual')
            ax2.plot(factors, cumulative_var * 100, 'ro-', linewidth=2, 
                    label='Cumulative')
            
            # Target lines
            ax2.axhline(y=60, color='g', linestyle='--', alpha=0.7, 
                       label='60% Target')
            if cumulative_var[-1] >= 0.8:
                ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, 
                           label='80% Excellent')
            
            ax2.set_xlabel('Factor Number')
            ax2.set_ylabel('Variance Explained (%)')
            ax2.set_title('Variance Explained by Factor')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 100)
            
            # Annotate cumulative variance for key factors
            for i in range(min(5, len(factors))):
                if i < len(cumulative_var):
                    ax2.annotate(f'{cumulative_var[i]*100:.1f}%', 
                               (factors[i], cumulative_var[i]*100), 
                               textcoords="offset points", 
                               xytext=(0,10), 
                               ha='center', fontsize=9)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        if ax2 is not None:
            return fig, (ax1, ax2)
        else:
            return fig, ax1
    
    def _create_interactive_scree_plot(self, eigenvalues: np.ndarray,
                                      variance_explained: Optional[np.ndarray],
                                      title: str) -> Any:
        """Create interactive scree plot using Plotly."""
        if variance_explained is not None:
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=('Eigenvalues by Factor', 'Variance Explained'),
                               horizontal_spacing=0.1)
        else:
            fig = go.Figure()
        
        factors = np.arange(1, len(eigenvalues) + 1)
        
        # Eigenvalues plot
        fig.add_trace(
            go.Scatter(x=factors, y=eigenvalues, mode='lines+markers',
                      name='Eigenvalues', line=dict(width=3),
                      marker=dict(size=8),
                      hovertemplate='Factor %{x}<br>Eigenvalue: %{y:.3f}<extra></extra>'),
            row=1, col=1 if variance_explained is not None else None
        )
        
        # Kaiser criterion line
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Kaiser Criterion (λ=1)",
                     row=1, col=1 if variance_explained is not None else None)
        
        # Variance explained plot (if available)
        if variance_explained is not None:
            cumulative_var = np.cumsum(variance_explained) * 100
            individual_var = variance_explained * 100
            
            fig.add_trace(
                go.Bar(x=factors, y=individual_var, name='Individual Variance',
                      opacity=0.7, 
                      hovertemplate='Factor %{x}<br>Individual: %{y:.1f}%<extra></extra>'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=factors, y=cumulative_var, mode='lines+markers',
                          name='Cumulative Variance', line=dict(width=3, color='red'),
                          hovertemplate='Factor %{x}<br>Cumulative: %{y:.1f}%<extra></extra>'),
                row=1, col=2
            )
            
            fig.add_hline(y=60, line_dash="dash", line_color="green",
                         annotation_text="60% Target", row=1, col=2)
        
        fig.update_layout(title_text=title, showlegend=True, height=500)
        fig.update_xaxes(title_text="Factor Number")
        fig.update_yaxes(title_text="Eigenvalue", row=1, col=1 if variance_explained is not None else None)
        if variance_explained is not None:
            fig.update_yaxes(title_text="Variance Explained (%)", row=1, col=2)
        
        return fig
    
    def plot_loadings_heatmap(self, loadings: pd.DataFrame,
                             title: str = "Factor Loadings Heatmap",
                             loading_threshold: float = 0.4,
                             cluster_variables: bool = True,
                             figsize: Optional[Tuple[int, int]] = None,
                             save_path: Optional[str] = None,
                             interactive: bool = False) -> Union[Tuple[Figure, Axes], Any]:
        """
        Create comprehensive factor loadings heatmap visualization.
        
        Args:
            loadings: Factor loadings matrix (variables × factors)
            title: Plot title
            loading_threshold: Threshold for highlighting significant loadings
            cluster_variables: Whether to cluster variables by loading patterns
            figsize: Figure size (width, height)
            save_path: Path to save plot (optional)
            interactive: Whether to create interactive plotly version
            
        Returns:
            Tuple of (Figure, Axes) objects or plotly figure
        """
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_heatmap(loadings, title, loading_threshold)
        
        if figsize is None:
            figsize = (max(8, len(loadings.columns) * 2), max(6, len(loadings.index) * 0.4))
        
        # Cluster variables if requested
        if cluster_variables and len(loadings.index) > 3 and SCIPY_CLUSTERING_AVAILABLE:
            try:
                # Calculate distance matrix based on loading patterns
                distances = pdist(np.abs(loadings.values), metric='euclidean')
                linkage_matrix = linkage(distances, method='ward')
                
                # Get optimal ordering
                dendro = dendrogram(linkage_matrix, no_plot=True)
                cluster_order = dendro['leaves']
                loadings_ordered = loadings.iloc[cluster_order]
            except Exception as e:
                warnings.warn(f"Clustering failed: {e}. Using original order.", UserWarning)
                loadings_ordered = loadings
        else:
            loadings_ordered = loadings
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(loadings_ordered.T, annot=True, cmap=cmap, center=0,
                   square=False, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.3f', ax=ax, 
                   annot_kws={'size': 8 if len(loadings.index) > 15 else 10})
        
        # Highlight significant loadings
        for i, factor in enumerate(loadings_ordered.columns):
            for j, variable in enumerate(loadings_ordered.index):
                loading = loadings_ordered.loc[variable, factor]
                if abs(loading) >= loading_threshold:
                    # Add border around significant loadings
                    rect = patches.Rectangle((j, i), 1, 1, linewidth=3,
                                           edgecolor='black', facecolor='none')
                    ax.add_patch(rect)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Variables', fontsize=12)
        ax.set_ylabel('Factors', fontsize=12)
        
        # Add threshold annotation
        ax.text(0.02, 0.98, f'Significant loadings (|loading| ≥ {loading_threshold}) highlighted with black borders',
               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
               verticalalignment='top', fontsize=9)
        
        # Rotate x labels if many variables
        if len(loadings.index) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def _create_interactive_heatmap(self, loadings: pd.DataFrame, 
                                   title: str, threshold: float) -> Any:
        """Create interactive heatmap using Plotly."""
        # Create hover text with significance information
        hover_text = np.empty(loadings.T.shape, dtype=object)
        for i, factor in enumerate(loadings.columns):
            for j, variable in enumerate(loadings.index):
                loading = loadings.loc[variable, factor]
                significance = "Significant" if abs(loading) >= threshold else "Not significant"
                hover_text[i, j] = f"Factor: {factor}<br>Variable: {variable}<br>Loading: {loading:.3f}<br>Status: {significance}"
        
        fig = go.Figure(data=go.Heatmap(
            z=loadings.T.values,
            x=loadings.index,
            y=loadings.columns,
            colorscale='RdBu',
            zmid=0,
            text=loadings.T.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text,
            colorbar=dict(title="Loading Value")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Variables",
            yaxis_title="Factors",
            height=max(400, len(loadings.columns) * 80),
            width=max(600, len(loadings.index) * 40)
        )
        
        return fig
    
    def plot_biplot(self, loadings: pd.DataFrame, 
                   factor_scores: Optional[pd.DataFrame] = None,
                   factor_x: int = 0, factor_y: int = 1,
                   title: str = "EFA Biplot",
                   show_variable_names: bool = True,
                   show_observations: bool = True,
                   highlight_threshold: float = 0.4,
                   max_samples: int = 1000,
                   figsize: Tuple[int, int] = (12, 8),
                   save_path: Optional[str] = None,
                   interactive: bool = False) -> Union[Tuple[Figure, Axes], Any]:
        """
        Create biplot showing variables and observations in factor space.
        
        Args:
            loadings: Factor loadings matrix
            factor_scores: Factor scores for observations (optional)
            factor_x: Index of factor for x-axis
            factor_y: Index of factor for y-axis
            title: Plot title
            show_variable_names: Whether to show variable labels
            show_observations: Whether to show observation points
            highlight_threshold: Threshold for highlighting significant loadings
            max_samples: Maximum number of sample points to plot
            figsize: Figure size (width, height)
            save_path: Path to save plot (optional)
            interactive: Whether to create interactive plotly version
            
        Returns:
            Tuple of (Figure, Axes) objects or plotly figure
            
        Raises:
            ValueError: Invalid factor indices or data mismatch
        """
        # Validate factor indices
        if factor_x >= len(loadings.columns) or factor_y >= len(loadings.columns):
            raise ValueError(f"Factor indices must be < {len(loadings.columns)}")
        
        if interactive and PLOTLY_AVAILABLE:
            return self._create_interactive_biplot(loadings, factor_scores, factor_x, factor_y, title, highlight_threshold)
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot sample points if scores provided
        if factor_scores is not None and show_observations:
            # Sample points if too many
            if len(factor_scores) > max_samples:
                sample_idx = np.random.choice(len(factor_scores), max_samples, replace=False)
                scores_plot = factor_scores.iloc[sample_idx]
            else:
                scores_plot = factor_scores
            
            ax.scatter(scores_plot.iloc[:, factor_x], scores_plot.iloc[:, factor_y], 
                      alpha=0.6, s=30, c='lightblue', edgecolors='darkblue', linewidth=0.5,
                      label=f'Observations (n={len(scores_plot)})')
        
        # Plot variable loadings as vectors
        for i, variable in enumerate(loadings.index):
            x_loading = loadings.iloc[i, factor_x]
            y_loading = loadings.iloc[i, factor_y]
            
            # Determine color based on loading magnitude
            loading_magnitude = np.sqrt(x_loading**2 + y_loading**2)
            is_significant = loading_magnitude >= highlight_threshold
            
            color = 'red' if is_significant else 'gray'
            linewidth = 2 if is_significant else 1
            alpha = 0.8 if is_significant else 0.6
            
            # Draw vector
            ax.arrow(0, 0, x_loading, y_loading, head_width=0.02, head_length=0.02,
                    fc=color, ec=color, linewidth=linewidth, alpha=alpha)
            
            # Add variable label
            if show_variable_names:
                label_offset = 0.05 if is_significant else 0.03
                ax.text(x_loading + label_offset * np.sign(x_loading), 
                       y_loading + label_offset * np.sign(y_loading),
                       variable, fontsize=9 if is_significant else 8,
                       fontweight='bold' if is_significant else 'normal',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                               alpha=0.8 if is_significant else 0.6, edgecolor=color))
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid and axes through origin
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)
        
        # Labels and title
        ax.set_xlabel(f'{loadings.columns[factor_x]} Loadings/Scores', fontsize=12)
        ax.set_ylabel(f'{loadings.columns[factor_y]} Loadings/Scores', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='red', linewidth=2, label=f'Significant (≥{highlight_threshold})'),
            plt.Line2D([0], [0], color='gray', linewidth=1, label=f'Not significant (<{highlight_threshold})')
        ]
        if factor_scores is not None and show_observations:
            legend_elements.insert(0, plt.scatter([], [], c='lightblue', edgecolors='darkblue', 
                                                s=30, label='Observations'))
        
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Add explanation text
        explanation = f"Vectors represent variable loadings.\n{'Points represent observations.' if factor_scores is not None else 'No observation scores provided.'}"
        ax.text(0.02, 0.02, explanation, transform=ax.transAxes, 
               bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
               verticalalignment='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_rotation_comparison(self, comparison_results: Dict[str, Any],
                               metrics_df: Optional[pd.DataFrame] = None,
                               figsize: Tuple[int, int] = (15, 10),
                               save_path: Optional[str] = None) -> Tuple[Figure, Any]:
        """
        Create comprehensive visualization comparing different rotation methods.
        
        Args:
            comparison_results: Results from EFAAnalyzer.compare_rotations()
            metrics_df: Optional pre-computed metrics dataframe
            figsize: Figure size (width, height)
            save_path: Path to save plot (optional)
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        n_rotations = len(comparison_results)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Loadings comparison heatmap
        if len(comparison_results) > 0:
            rotation_names = list(comparison_results.keys())
            first_solution = list(comparison_results.values())[0]
            
            # Create side-by-side heatmaps for first two rotations
            for i, (rotation, solution) in enumerate(list(comparison_results.items())[:2]):
                loadings = solution.loadings
                
                ax = axes[i]
                sns.heatmap(loadings.T, annot=True, cmap='RdBu_r', center=0,
                           square=False, linewidths=0.5, ax=ax, fmt='.2f',
                           annot_kws={'size': 8})
                ax.set_title(f'{rotation.capitalize()} Rotation', fontsize=12, fontweight='bold')
                ax.set_xlabel('Variables')
                ax.set_ylabel('Factors')
        
        # Plot 3: Variance explained comparison
        if metrics_df is not None:
            ax = axes[2]
            bars = ax.bar(metrics_df['Rotation'], metrics_df['Total_Variance_Explained'],
                         color=plt.cm.Set3(np.arange(len(metrics_df))))
            ax.set_title('Total Variance Explained by Rotation', fontweight='bold')
            ax.set_ylabel('Variance Explained')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics_df['Total_Variance_Explained']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Complexity and correlation metrics
        if metrics_df is not None:
            ax = axes[3]
            
            # Create dual y-axis plot
            ax2 = ax.twinx()
            
            # Plot complexity on left axis
            line1 = ax.plot(metrics_df['Rotation'], metrics_df['Average_Complexity'], 
                           'o-', color='blue', linewidth=2, markersize=8, label='Complexity')
            ax.set_ylabel('Average Complexity', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            
            # Plot max correlation on right axis
            line2 = ax2.plot(metrics_df['Rotation'], metrics_df['Max_Factor_Correlation'], 
                            's-', color='red', linewidth=2, markersize=8, label='Max Correlation')
            ax2.set_ylabel('Max Factor Correlation', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add title and legend
            ax.set_title('Rotation Complexity and Factor Correlations', fontweight='bold')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, axes
    
    def plot_parallel_analysis(self, pa_results: Dict[str, Any],
                              title: str = "Parallel Analysis",
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> Tuple[Figure, Axes]:
        """
        Create parallel analysis visualization.
        
        Args:
            pa_results: Results from EFAAnalyzer.parallel_analysis()
            title: Plot title
            figsize: Figure size (width, height)
            save_path: Path to save plot (optional)
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract data
        real_eigenvals = np.array(pa_results['eigenvalues'])
        simulated_eigenvals = np.array(pa_results['simulated_eigenvalues'])
        percentile_eigenvals = np.array(pa_results['percentile_eigenvalues'])
        suggested_factors = pa_results['suggested_factors']
        percentile = pa_results['percentile']
        
        factor_numbers = np.arange(1, len(real_eigenvals) + 1)
        
        # Plot lines
        ax.plot(factor_numbers, real_eigenvals, 'o-', linewidth=2, markersize=8, 
               color='blue', label='Actual Data')
        ax.plot(factor_numbers, simulated_eigenvals, 's--', linewidth=2, markersize=6,
               color='red', label='Random Data (Mean)')
        ax.plot(factor_numbers, percentile_eigenvals, '^--', linewidth=2, markersize=6,
               color='orange', label=f'{percentile}th Percentile')
        
        # Highlight suggested factors
        if suggested_factors > 0:
            ax.axvline(x=suggested_factors + 0.5, color='green', linestyle=':', linewidth=3,
                      alpha=0.7, label=f'Suggested: {suggested_factors} factors')
            
            # Shade the retained factors area
            ax.axvspan(0.5, suggested_factors + 0.5, alpha=0.2, color='green')
        
        # Add horizontal line at eigenvalue = 1 (Kaiser criterion)
        ax.axhline(y=1, color='gray', linestyle='-', alpha=0.5, label='Kaiser Criterion (λ=1)')
        
        # Formatting
        ax.set_xlabel('Factor Number', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Set x-axis to show integer values
        ax.set_xticks(factor_numbers)
        
        # Add text annotation
        ax.text(0.02, 0.98, 
               f'Parallel Analysis Recommendation:\n{suggested_factors} factors to retain\n'
               f'(Based on {pa_results["n_simulations"]} simulations)',
               transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
               verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def create_factor_summary_plot(self, solution: Any,
                                 title: str = "EFA Summary") -> Figure:
        """
        Create comprehensive summary plot with multiple panels.
        
        Args:
            solution: Complete factor analysis solution
            title: Overall plot title
            
        Returns:
            Figure object with multiple subplots
        """
        # TODO: Implement comprehensive summary visualization
        raise NotImplementedError("Summary plot to be implemented in advanced phase")
    
    def save_plot(self, fig: Figure, filename: str, 
                 format: str = 'png', **kwargs) -> str:
        """
        Save plot to file with appropriate settings.
        
        Args:
            fig: Figure object to save
            filename: Output filename (without extension)
            format: Output format ('png', 'pdf', 'svg')
            **kwargs: Additional save parameters
            
        Returns:
            Full path to saved file
            
        Raises:
            IOError: Save operation failures
        """
        # TODO: Implement in Phase 5 (T044)
        raise NotImplementedError("Plot saving to be implemented in T044")
    
    def create_interactive_plot(self, solution: Any,
                              plot_type: str = 'loadings') -> Any:
        """
        Create interactive visualization for Jupyter environments.
        
        Args:
            solution: Factor analysis solution
            plot_type: Type of interactive plot ('loadings', 'biplot', 'scree')
            
        Returns:
            Interactive plot object
        """
        # TODO: Implement in Phase 5 (T047)
        raise NotImplementedError("Interactive plots to be implemented in T047")


# Utility functions for visualization
def quick_scree_plot(eigenvalues: np.ndarray) -> Tuple[Figure, Axes]:
    """
    Quick scree plot with default settings.
    
    Args:
        eigenvalues: Array of eigenvalues
        
    Returns:
        Tuple of (Figure, Axes) objects
    """
    visualizer = EFAVisualizer()
    return visualizer.plot_scree(eigenvalues)


def setup_publication_plots():
    """Configure matplotlib for publication-quality plots."""
    visualizer = EFAVisualizer(publication_ready=True)
    return visualizer


# Custom exceptions for visualization
class VisualizationError(Exception):
    """Base exception for visualization-related errors."""
    pass

class PlotDataError(VisualizationError):
    """Exception for invalid plot data."""
    pass