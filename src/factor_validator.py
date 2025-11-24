# -*- coding: utf-8 -*-
"""
Factor Validator Module

Provides comprehensive statistical validation for factor analysis prerequisites 
and results quality assessment. Includes KMO measure, Bartlett's test, sample 
size adequacy, and reliability analysis.

Author: Insight Digger Project
Created: November 24, 2025
"""

import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

try:
    from factor_analyzer.utils import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False
    calculate_kmo = None
    calculate_bartlett_sphericity = None


class DataAdequacyResult:
    """Container for data adequacy assessment results."""
    
    def __init__(self, is_adequate: bool = True, sample_size: int = 0,
                 n_variables: int = 0, ratio: float = 0.0,
                 warnings: List[str] = None, recommendations: List[str] = None):
        self.is_adequate = is_adequate
        self.sample_size = sample_size
        self.n_variables = n_variables
        self.ratio = ratio
        self.warnings = warnings or []
        self.recommendations = recommendations or []


class FactorabilityResult:
    """Container for factorability assessment results."""
    
    def __init__(self, is_factorable: bool = True, kmo_overall: float = 0.0,
                 kmo_individual: Dict[str, float] = None, 
                 bartlett_statistic: float = 0.0, bartlett_p: float = 1.0,
                 warnings: List[str] = None, details: Dict[str, Any] = None):
        self.is_factorable = is_factorable
        self.kmo_overall = kmo_overall
        self.kmo_individual = kmo_individual or {}
        self.bartlett_statistic = bartlett_statistic
        self.bartlett_p = bartlett_p
        self.warnings = warnings or []
        self.details = details or {}


class ReliabilityResult:
    """Container for reliability analysis results."""
    
    def __init__(self, cronbach_alphas: Dict[str, float] = None,
                 factor_reliabilities: Dict[str, Dict] = None,
                 overall_assessment: str = "Unknown",
                 recommendations: List[str] = None):
        self.cronbach_alphas = cronbach_alphas or {}
        self.factor_reliabilities = factor_reliabilities or {}
        self.overall_assessment = overall_assessment
        self.recommendations = recommendations or []


class FactorValidator:
    """
    Comprehensive Statistical Validation for Factor Analysis
    
    Provides validation methods for assessing data suitability, factorability,
    and solution quality in exploratory factor analysis.
    
    Attributes:
        kmo_threshold (float): Minimum acceptable KMO value
        bartlett_alpha (float): Significance level for Bartlett's test
        min_sample_ratio (float): Minimum observations-to-variables ratio
        loading_threshold (float): Minimum factor loading for significance
    """
    
    def __init__(self,
                 kmo_threshold: float = 0.6,
                 bartlett_alpha: float = 0.05,
                 min_sample_ratio: float = 5.0,
                 loading_threshold: float = 0.40):
        """
        Initialize Factor Validator.
        
        Args:
            kmo_threshold: Minimum acceptable KMO value for analysis
            bartlett_alpha: Significance level for Bartlett's sphericity test
            min_sample_ratio: Minimum observations-to-variables ratio
            loading_threshold: Minimum factor loading for practical significance
            
        Raises:
            ValueError: Invalid threshold values
        """
        self.kmo_threshold = kmo_threshold
        self.bartlett_alpha = bartlett_alpha
        self.min_sample_ratio = min_sample_ratio
        self.loading_threshold = loading_threshold
        
        # Validate thresholds
        self._validate_thresholds()
    
    def _validate_thresholds(self):
        """Validate threshold parameters."""
        if not 0 < self.kmo_threshold <= 1:
            raise ValueError("kmo_threshold must be between 0 and 1")
        if not 0 < self.bartlett_alpha <= 1:
            raise ValueError("bartlett_alpha must be between 0 and 1") 
        if self.min_sample_ratio <= 0:
            raise ValueError("min_sample_ratio must be positive")
        if not 0 < self.loading_threshold <= 1:
            raise ValueError("loading_threshold must be between 0 and 1")
    
    def check_data_adequacy(self, data: pd.DataFrame) -> DataAdequacyResult:
        """
        Comprehensive data quality assessment for factor analysis.
        
        Args:
            data: Input data matrix (observations × variables)
            
        Returns:
            DataAdequacyResult: Complete adequacy assessment
            
        Raises:
            ValueError: Invalid data format or content
            TypeError: Incorrect data type
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
            
        n_obs, n_vars = data.shape
        ratio = n_obs / n_vars if n_vars > 0 else 0
        
        warnings_list = []
        recommendations = []
        
        # Sample size adequacy checks
        is_adequate = True
        
        # Absolute minimum check
        if n_obs < 50:
            warnings_list.append(f"Sample size {n_obs} below absolute minimum of 50")
            is_adequate = False
            
        # Ratio-based check  
        if ratio < self.min_sample_ratio:
            warnings_list.append(f"Sample-to-variable ratio {ratio:.1f} below recommended {self.min_sample_ratio}")
            if ratio < 3.0:
                is_adequate = False
                
        # Variable count check
        if n_vars < 3:
            warnings_list.append("Need at least 3 variables for factor analysis")
            is_adequate = False
            
        # Generate recommendations
        if not is_adequate:
            recommended_n = max(50, int(n_vars * self.min_sample_ratio))
            recommendations.append(f"Increase sample size to at least {recommended_n}")
            
        if n_vars < 6:
            recommendations.append("Consider adding more variables for stable factor structure")
            
        return DataAdequacyResult(
            is_adequate=is_adequate,
            sample_size=n_obs,
            n_variables=n_vars,
            ratio=ratio,
            warnings=warnings_list,
            recommendations=recommendations
        )
    
    def calculate_kmo(self, data: pd.DataFrame, correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Calculate Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy.
        
        KMO measures whether the data is suitable for factor analysis by comparing
        the correlations and partial correlations of the variables.
        
        Args:
            data: Input data matrix (observations x variables)
            correlation_matrix: Pre-calculated correlation matrix (optional)
            
        Returns:
            Dictionary containing KMO results
        """
        if correlation_matrix is None:
            correlation_matrix = data.corr(method='pearson')
        
        try:
            if FACTOR_ANALYZER_AVAILABLE:
                # Use factor_analyzer library implementation
                kmo_all, kmo_model = calculate_kmo(data.dropna())
                
                individual_kmo = {}
                for i, var in enumerate(data.columns):
                    individual_kmo[var] = kmo_all[i]
                
                overall_kmo = kmo_model
                
            else:
                # Manual KMO calculation
                overall_kmo, individual_kmo = self._calculate_kmo_manual(correlation_matrix)
            
            # Interpret results
            is_adequate = overall_kmo >= self.kmo_threshold
            interpretation = self._interpret_kmo(overall_kmo)
            recommendations = self._generate_kmo_recommendations(overall_kmo, individual_kmo)
            
            return {
                'overall_kmo': overall_kmo,
                'individual_kmo': individual_kmo,
                'is_adequate': is_adequate,
                'interpretation': interpretation,
                'recommendations': recommendations,
                'threshold': self.kmo_threshold
            }
            
        except Exception as e:
            warnings.warn(f"KMO calculation failed: {e}")
            
            # Return conservative estimates
            individual_kmo = {var: 0.5 for var in correlation_matrix.index}
            return {
                'overall_kmo': 0.5,
                'individual_kmo': individual_kmo,
                'is_adequate': False,
                'interpretation': "Calculation failed - results unreliable",
                'recommendations': ["Check data quality", "Remove missing values", "Check for constant variables"],
                'threshold': self.kmo_threshold
            }
    
    def _calculate_kmo_manual(self, correlation_matrix: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
        """Manual KMO calculation when factor_analyzer is not available."""
        try:
            # Calculate anti-image correlation matrix
            corr_inv = np.linalg.inv(correlation_matrix.values)
            
            # Calculate partial correlations
            partial_corr = np.zeros_like(corr_inv)
            for i in range(corr_inv.shape[0]):
                for j in range(corr_inv.shape[1]):
                    if i != j:
                        partial_corr[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
            
            # Calculate KMO measures
            corr_matrix = correlation_matrix.values
            
            # Overall KMO
            sum_corr_sq = np.sum(corr_matrix**2) - np.sum(np.diag(corr_matrix)**2)
            sum_partial_sq = np.sum(partial_corr**2)
            overall_kmo = sum_corr_sq / (sum_corr_sq + sum_partial_sq)
            
            # Individual KMO for each variable
            individual_kmo = {}
            for i, var in enumerate(correlation_matrix.index):
                # Sum of squared correlations for variable i
                corr_i_sq = np.sum(corr_matrix[i, :]**2) - corr_matrix[i, i]**2
                # Sum of squared partial correlations for variable i
                partial_i_sq = np.sum(partial_corr[i, :]**2)
                
                if corr_i_sq + partial_i_sq > 0:
                    individual_kmo[var] = corr_i_sq / (corr_i_sq + partial_i_sq)
                else:
                    individual_kmo[var] = 0.0
            
            return overall_kmo, individual_kmo
            
        except np.linalg.LinAlgError:
            # Handle singular correlation matrix
            individual_kmo = {var: 0.5 for var in correlation_matrix.index}
            return 0.5, individual_kmo
    
    def _interpret_kmo(self, kmo_value: float) -> str:
        """Interpret KMO value according to Kaiser's guidelines."""
        if kmo_value >= 0.9:
            return "Marvelous - excellent for factor analysis"
        elif kmo_value >= 0.8:
            return "Meritorious - very good for factor analysis"
        elif kmo_value >= 0.7:
            return "Middling - acceptable for factor analysis"
        elif kmo_value >= 0.6:
            return "Mediocre - marginally acceptable for factor analysis"
        elif kmo_value >= 0.5:
            return "Miserable - poor for factor analysis"
        else:
            return "Unacceptable - factor analysis not recommended"
    
    def _generate_kmo_recommendations(self, overall_kmo: float, individual_kmo: Dict[str, float]) -> List[str]:
        """Generate recommendations based on KMO results."""
        recommendations = []
        
        if overall_kmo < 0.6:
            recommendations.append("Overall KMO < 0.6 - consider collecting more data or removing problematic variables")
        
        if overall_kmo < 0.5:
            recommendations.append("Factor analysis is not recommended with current data")
        
        # Check for problematic individual variables
        low_kmo_vars = [var for var, kmo in individual_kmo.items() if kmo < 0.5]
        if low_kmo_vars:
            recommendations.append(f"Consider removing variables with low individual KMO: {', '.join(low_kmo_vars[:5])}")
        
        very_low_kmo_vars = [var for var, kmo in individual_kmo.items() if kmo < 0.3]
        if very_low_kmo_vars:
            recommendations.append(f"Variables with very low KMO (< 0.3) should be removed: {', '.join(very_low_kmo_vars[:3])}")
        
        if overall_kmo >= 0.8:
            recommendations.append("Excellent sampling adequacy - proceed with factor analysis")
        elif overall_kmo >= 0.6:
            recommendations.append("Acceptable sampling adequacy - factor analysis can proceed with caution")
        
        return recommendations
    
    def calculate_bartlett_test(self, data: pd.DataFrame, correlation_matrix: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform Bartlett's test of sphericity.
        
        Tests the null hypothesis that the correlation matrix is an identity matrix
        (i.e., variables are uncorrelated). Significant results (p < 0.05) indicate
        that factor analysis is appropriate.
        
        Args:
            data: Input data matrix (observations x variables)
            correlation_matrix: Pre-calculated correlation matrix (optional)
            
        Returns:
            Dictionary containing Bartlett's test results
        """
        if correlation_matrix is None:
            correlation_matrix = data.corr(method='pearson')
        
        n_obs, n_vars = data.shape
        
        try:
            if FACTOR_ANALYZER_AVAILABLE:
                # Use factor_analyzer library implementation
                statistic, p_value = calculate_bartlett_sphericity(data.dropna())
                df = n_vars * (n_vars - 1) / 2
                
            else:
                # Manual Bartlett's test calculation
                statistic, p_value, df = self._calculate_bartlett_manual(correlation_matrix, n_obs)
            
            # Interpret results
            is_significant = p_value < self.bartlett_alpha
            interpretation = self._interpret_bartlett(p_value, is_significant)
            recommendations = self._generate_bartlett_recommendations(p_value, is_significant)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'degrees_of_freedom': df,
                'is_significant': is_significant,
                'interpretation': interpretation,
                'recommendations': recommendations,
                'alpha': self.bartlett_alpha
            }
            
        except Exception as e:
            warnings.warn(f"Bartlett's test calculation failed: {e}")
            
            return {
                'statistic': 0.0,
                'p_value': 1.0,
                'degrees_of_freedom': 0,
                'is_significant': False,
                'interpretation': "Test calculation failed",
                'recommendations': ["Check data quality", "Ensure sufficient observations", "Remove constant variables"],
                'alpha': self.bartlett_alpha
            }
    
    def _calculate_bartlett_manual(self, correlation_matrix: pd.DataFrame, n_obs: int) -> Tuple[float, float, int]:
        """Manual Bartlett's test calculation when factor_analyzer is not available."""
        n_vars = correlation_matrix.shape[0]
        
        # Calculate determinant of correlation matrix
        det_corr = np.linalg.det(correlation_matrix.values)
        
        # Bartlett's test statistic
        chi_square = -(n_obs - 1 - (2 * n_vars + 5) / 6) * np.log(det_corr)
        
        # Degrees of freedom
        df = n_vars * (n_vars - 1) / 2
        
        # Calculate p-value using chi-square distribution
        if SCIPY_AVAILABLE:
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi_square, df)
        else:
            # Rough approximation for p-value without scipy
            # For large chi-square values, p-value approaches 0
            if chi_square > 50:  # Arbitrary threshold for "very significant"
                p_value = 0.001
            elif chi_square > 20:
                p_value = 0.01
            elif chi_square > 10:
                p_value = 0.05
            else:
                p_value = 0.5  # Conservative estimate
        
        return chi_square, p_value, int(df)
    
    def _interpret_bartlett(self, p_value: float, is_significant: bool) -> str:
        """Interpret Bartlett's test results."""
        if is_significant:
            if p_value < 0.001:
                return "Highly significant - correlation matrix differs significantly from identity matrix (excellent for factor analysis)"
            elif p_value < 0.01:
                return "Very significant - correlation matrix differs from identity matrix (very good for factor analysis)"
            else:
                return "Significant - correlation matrix differs from identity matrix (good for factor analysis)"
        else:
            return "Not significant - correlation matrix may not differ from identity matrix (poor for factor analysis)"
    
    def _generate_bartlett_recommendations(self, p_value: float, is_significant: bool) -> List[str]:
        """Generate recommendations based on Bartlett's test results."""
        recommendations = []
        
        if is_significant:
            if p_value < 0.001:
                recommendations.append("Excellent - proceed with factor analysis")
            else:
                recommendations.append("Factor analysis is appropriate")
        else:
            recommendations.append("Factor analysis may not be appropriate - variables appear uncorrelated")
            recommendations.append("Consider checking data quality and variable selection")
            recommendations.append("May need to collect more data or use different variables")
        
        if p_value > 0.05:
            recommendations.append("Consider alternative analysis methods (e.g., principal component analysis)")
        
        return recommendations
    
    def check_enhanced_sample_adequacy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Enhanced sample size adequacy assessment with multiple criteria.
        
        Uses several established guidelines:
        - Comrey & Lee (1992): N=50 poor, N=100 poor, N=200 fair, N=300 good, N=500 very good, N>=1000 excellent
        - Hair et al. (2010): Minimum 5:1 ratio, preferably 10:1 or higher
        - Tabachnick & Fidell (2013): N >= 300 for stable solutions
        
        Args:
            data: Input data matrix (observations x variables)
            
        Returns:
            Dictionary with detailed adequacy assessment
        """
        n_obs, n_vars = data.shape
        ratio = n_obs / n_vars if n_vars > 0 else 0
        
        assessments = []
        overall_adequacy = "Excellent"
        recommendations = []
        
        # Comrey & Lee classification
        if n_obs >= 1000:
            comrey_rating = "Excellent"
        elif n_obs >= 500:
            comrey_rating = "Very Good"
        elif n_obs >= 300:
            comrey_rating = "Good"
        elif n_obs >= 200:
            comrey_rating = "Fair"
        elif n_obs >= 100:
            comrey_rating = "Poor"
        elif n_obs >= 50:
            comrey_rating = "Very Poor"
        else:
            comrey_rating = "Unacceptable"
        
        assessments.append(f"Comrey & Lee (1992): {comrey_rating}")
        
        # Ratio-based assessment
        if ratio >= 20:
            ratio_rating = "Excellent"
        elif ratio >= 15:
            ratio_rating = "Very Good"
        elif ratio >= 10:
            ratio_rating = "Good"
        elif ratio >= 5:
            ratio_rating = "Adequate"
        elif ratio >= 3:
            ratio_rating = "Marginal"
        else:
            ratio_rating = "Inadequate"
        
        assessments.append(f"Obs:Var ratio ({ratio:.1f}:1): {ratio_rating}")
        
        # Factor stability assessment
        if n_obs >= 300 and ratio >= 10:
            stability_rating = "High"
        elif n_obs >= 200 and ratio >= 5:
            stability_rating = "Moderate"
        elif n_obs >= 100 and ratio >= 3:
            stability_rating = "Low"
        else:
            stability_rating = "Very Low"
        
        assessments.append(f"Factor stability: {stability_rating}")
        
        # Determine overall adequacy
        if comrey_rating in ["Unacceptable", "Very Poor"] or ratio < 3:
            overall_adequacy = "Inadequate"
        elif comrey_rating == "Poor" or ratio < 5:
            overall_adequacy = "Marginal"
        elif comrey_rating in ["Fair", "Good"] and ratio >= 5:
            overall_adequacy = "Adequate"
        elif comrey_rating in ["Very Good", "Excellent"] and ratio >= 10:
            overall_adequacy = "Excellent"
        else:
            overall_adequacy = "Good"
        
        # Generate specific recommendations
        if n_obs < 50:
            recommendations.append("Critical: Sample size below minimum threshold - analysis not recommended")
        elif n_obs < 100:
            recommendations.append("Warning: Sample size very small - results may be unstable")
        elif n_obs < 200:
            recommendations.append("Caution: Sample size small - interpret results carefully")
        elif n_obs < 300:
            recommendations.append("Note: Adequate sample size for exploratory analysis")
        
        if ratio < 3:
            recommendations.append("Critical: Too few observations per variable - increase sample size")
        elif ratio < 5:
            recommendations.append("Warning: Low observations-to-variables ratio - consider reducing variables")
        elif ratio < 10:
            recommendations.append("Acceptable ratio, but higher would improve stability")
        
        # Specific targets
        if overall_adequacy in ["Inadequate", "Marginal"]:
            target_n = max(300, n_vars * 15)  # Conservative target
            recommendations.append(f"Target sample size: {target_n} observations for stable results")
        
        return {
            'overall_adequacy': overall_adequacy,
            'sample_size': n_obs,
            'n_variables': n_vars,
            'ratio': ratio,
            'assessments': assessments,
            'recommendations': recommendations,
            'comrey_rating': comrey_rating,
            'ratio_rating': ratio_rating,
            'stability_rating': stability_rating
        }
    
    def calculate_cronbach_alpha(self, data: pd.DataFrame, factor_assignments: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Calculate Cronbach's alpha reliability coefficient.
        
        Can calculate for overall data or for specific factor groupings.
        
        Args:
            data: Input data matrix (observations x variables)
            factor_assignments: Dictionary mapping variables to factors (optional)
            
        Returns:
            Dictionary containing reliability results
        """
        results = {}
        
        try:
            if factor_assignments is None:
                # Calculate overall Cronbach's alpha
                alpha_overall = self._calculate_alpha(data)
                results['overall'] = {
                    'alpha': alpha_overall,
                    'interpretation': self._interpret_alpha(alpha_overall),
                    'n_items': data.shape[1]
                }
            else:
                # Calculate alpha for each factor
                results['by_factor'] = {}
                
                # Group variables by factor
                factor_groups = {}
                for variable, factor in factor_assignments.items():
                    if factor not in factor_groups:
                        factor_groups[factor] = []
                    factor_groups[factor].append(variable)
                
                # Calculate alpha for each factor
                for factor_name, variables in factor_groups.items():
                    if len(variables) >= 2:  # Need at least 2 variables for alpha
                        factor_data = data[variables]
                        alpha_factor = self._calculate_alpha(factor_data)
                        
                        results['by_factor'][factor_name] = {
                            'alpha': alpha_factor,
                            'interpretation': self._interpret_alpha(alpha_factor),
                            'n_items': len(variables),
                            'variables': variables
                        }
                    else:
                        results['by_factor'][factor_name] = {
                            'alpha': None,
                            'interpretation': "Cannot calculate - insufficient variables",
                            'n_items': len(variables),
                            'variables': variables
                        }
                
                # Calculate overall average
                valid_alphas = [f['alpha'] for f in results['by_factor'].values() if f['alpha'] is not None]
                if valid_alphas:
                    results['average_alpha'] = sum(valid_alphas) / len(valid_alphas)
                else:
                    results['average_alpha'] = None
            
            # Generate recommendations
            results['recommendations'] = self._generate_alpha_recommendations(results)
            
            return results
            
        except Exception as e:
            warnings.warn(f"Cronbach's alpha calculation failed: {e}")
            
            return {
                'overall': {
                    'alpha': None,
                    'interpretation': "Calculation failed",
                    'n_items': data.shape[1]
                },
                'recommendations': ["Check data quality", "Ensure variables are numeric", "Remove constant variables"]
            }
    
    def _calculate_alpha(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate Cronbach's alpha for a set of variables."""
        # Remove missing values
        clean_data = data.dropna()
        
        if clean_data.shape[0] < 2 or clean_data.shape[1] < 2:
            return None
        
        # Calculate item variances
        item_variances = clean_data.var(axis=0)
        
        # Calculate total scale variance
        scale_scores = clean_data.sum(axis=1)
        scale_variance = scale_scores.var()
        
        # Calculate Cronbach's alpha
        k = clean_data.shape[1]  # Number of items
        sum_item_variances = item_variances.sum()
        
        if scale_variance == 0:
            return 0.0
        
        alpha = (k / (k - 1)) * (1 - (sum_item_variances / scale_variance))
        
        return alpha
    
    def _interpret_alpha(self, alpha: Optional[float]) -> str:
        """Interpret Cronbach's alpha value."""
        if alpha is None:
            return "Cannot calculate"
        
        if alpha >= 0.9:
            return "Excellent reliability"
        elif alpha >= 0.8:
            return "Good reliability"
        elif alpha >= 0.7:
            return "Acceptable reliability"
        elif alpha >= 0.6:
            return "Questionable reliability"
        elif alpha >= 0.5:
            return "Poor reliability"
        else:
            return "Unacceptable reliability"
    
    def _generate_alpha_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on reliability results."""
        recommendations = []
        
        # Check overall alpha
        if 'overall' in results:
            alpha = results['overall']['alpha']
            if alpha is not None:
                if alpha < 0.6:
                    recommendations.append("Overall reliability is poor - consider revising scale or removing items")
                elif alpha < 0.7:
                    recommendations.append("Overall reliability is questionable - consider improvements")
                elif alpha >= 0.8:
                    recommendations.append("Overall reliability is good - scale is internally consistent")
        
        # Check factor-wise alphas
        if 'by_factor' in results:
            low_alpha_factors = []
            for factor_name, factor_info in results['by_factor'].items():
                alpha = factor_info['alpha']
                if alpha is not None and alpha < 0.6:
                    low_alpha_factors.append(factor_name)
            
            if low_alpha_factors:
                recommendations.append(f"Factors with low reliability: {', '.join(low_alpha_factors)}")
                recommendations.append("Consider removing or replacing items in unreliable factors")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Reliability assessment complete - no major concerns identified")
        
        return recommendations
    
    def check_correlation_singularity(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Check correlation matrix for singularity and multicollinearity issues.
        
        A singular correlation matrix indicates perfect multicollinearity and
        can cause problems in factor analysis.
        
        Args:
            correlation_matrix: Correlation matrix to check
            
        Returns:
            Dictionary containing singularity assessment results
        """
        try:
            # Calculate determinant
            det = np.linalg.det(correlation_matrix.values)
            
            # Calculate condition number
            cond_num = np.linalg.cond(correlation_matrix.values)
            
            # Check for perfect correlations
            perfect_corrs = []
            high_corrs = []
            
            n_vars = correlation_matrix.shape[0]
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    var1 = correlation_matrix.index[i]
                    var2 = correlation_matrix.index[j]
                    
                    if corr_val >= 0.95:
                        perfect_corrs.append((var1, var2, corr_val))
                    elif corr_val >= 0.85:
                        high_corrs.append((var1, var2, corr_val))
            
            # Assess singularity
            is_singular = det < 1e-10  # Very small determinant indicates singularity
            is_near_singular = det < 1e-6 or cond_num > 1e12
            
            # Generate interpretation
            if is_singular:
                interpretation = "Matrix is singular - perfect multicollinearity detected"
                severity = "Critical"
            elif is_near_singular:
                interpretation = "Matrix is near-singular - severe multicollinearity present"
                severity = "Warning"
            elif cond_num > 1000:
                interpretation = "Moderate multicollinearity detected"
                severity = "Caution"
            else:
                interpretation = "No serious multicollinearity issues"
                severity = "OK"
            
            # Generate recommendations
            recommendations = self._generate_singularity_recommendations(
                is_singular, is_near_singular, perfect_corrs, high_corrs, cond_num
            )
            
            return {
                'is_singular': is_singular,
                'is_near_singular': is_near_singular,
                'determinant': det,
                'condition_number': cond_num,
                'perfect_correlations': perfect_corrs,
                'high_correlations': high_corrs,
                'interpretation': interpretation,
                'severity': severity,
                'recommendations': recommendations
            }
            
        except Exception as e:
            warnings.warn(f"Correlation matrix singularity check failed: {e}")
            
            return {
                'is_singular': True,  # Conservative assumption
                'is_near_singular': True,
                'determinant': 0.0,
                'condition_number': float('inf'),
                'perfect_correlations': [],
                'high_correlations': [],
                'interpretation': "Check failed - assume problems exist",
                'severity': "Critical",
                'recommendations': ["Manual inspection required", "Check for constant variables", "Remove redundant variables"]
            }
    
    def _generate_singularity_recommendations(self, is_singular: bool, is_near_singular: bool,
                                            perfect_corrs: List[Tuple], high_corrs: List[Tuple],
                                            cond_num: float) -> List[str]:
        """Generate recommendations based on singularity assessment."""
        recommendations = []
        
        if is_singular:
            recommendations.append("Critical: Remove perfectly correlated variables before proceeding")
            recommendations.append("Factor analysis cannot be performed with singular correlation matrix")
        
        if perfect_corrs:
            recommendations.append(f"Remove one variable from each perfect correlation pair:")
            for var1, var2, corr in perfect_corrs[:3]:  # Show first 3
                recommendations.append(f"  - {var1} and {var2} (r = {corr:.3f})")
        
        if is_near_singular:
            recommendations.append("Severe multicollinearity detected - consider variable reduction")
        
        if high_corrs and not is_singular:
            recommendations.append(f"High correlations detected (consider combining variables):")
            for var1, var2, corr in high_corrs[:3]:  # Show first 3
                recommendations.append(f"  - {var1} and {var2} (r = {corr:.3f})")
        
        if cond_num > 1000 and not (is_singular or is_near_singular):
            recommendations.append("Moderate multicollinearity present - monitor factor stability")
        
        if not recommendations:
            recommendations.append("Correlation matrix appears suitable for factor analysis")
        
        return recommendations
    
    def assess_factorability(self, data: pd.DataFrame) -> FactorabilityResult:
        """
        Assess data suitability for factor analysis using KMO and Bartlett's test.
        
        Args:
            data: Input data matrix or correlation matrix
            
        Returns:
            FactorabilityResult: Comprehensive factorability assessment
            
        Raises:
            ValueError: Invalid data for statistical testing
            RuntimeError: Statistical computation errors
        """
        # TODO: Implement in Phase 4 (T025, T026)
        raise NotImplementedError("Factorability assessment to be implemented in T025/T026")
    
    def calculate_reliability(self, loadings: pd.DataFrame, 
                            data: Optional[pd.DataFrame] = None) -> ReliabilityResult:
        """
        Calculate reliability statistics for extracted factors.
        
        Args:
            loadings: Factor loadings matrix
            data: Original data matrix for Cronbach's alpha calculation
            
        Returns:
            ReliabilityResult: Comprehensive reliability assessment
            
        Raises:
            ValueError: Invalid loadings matrix or data mismatch
        """
        # TODO: Implement in Phase 4 (T028)
        raise NotImplementedError("Reliability calculation to be implemented in T028")
    
    def validate_correlation_matrix(self, corr_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate correlation matrix for singularity and other issues.
        
        Args:
            corr_matrix: Correlation matrix to validate
            
        Returns:
            Dictionary containing validation results and warnings
            
        Raises:
            ValueError: Invalid correlation matrix
        """
        # TODO: Implement in Phase 4 (T029)
        raise NotImplementedError("Correlation matrix validation to be implemented in T029")
    
    def generate_validation_report(self, data: pd.DataFrame, 
                                 solution: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for EFA.
        
        Args:
            data: Input data matrix
            solution: Optional factor solution for post-hoc validation
            
        Returns:
            Dictionary containing complete validation assessment
        """
        # TODO: Implement in Phase 4 (T035)
        raise NotImplementedError("Validation reporting to be implemented in T035")


# Utility functions for validation
def quick_factorability_check(data: pd.DataFrame) -> Tuple[bool, str]:
    """
    Quick factorability check with simple pass/fail result.
    
    Args:
        data: Input data matrix
        
    Returns:
        Tuple of (is_suitable, reason)
    """
    validator = FactorValidator()
    try:
        result = validator.assess_factorability(data)
        return result.is_factorable, "Passed standard checks"
    except NotImplementedError:
        return False, "Validation not yet implemented"


def get_sample_size_recommendation(n_variables: int, 
                                 target_ratio: float = 10.0) -> int:
    """
    Calculate recommended sample size for factor analysis.
    
    Args:
        n_variables: Number of variables in analysis
        target_ratio: Target observations-to-variables ratio
        
    Returns:
        Recommended minimum sample size
    """
    conservative_min = max(50, int(n_variables * target_ratio))
    return conservative_min


# Custom exceptions for validation
class ValidationError(Exception):
    """Base exception for validation-related errors."""
    pass

class FactorabilityError(ValidationError):
    """Exception for factorability assessment failures."""
    pass

class SampleSizeError(ValidationError):
    """Exception for insufficient sample size."""
    pass