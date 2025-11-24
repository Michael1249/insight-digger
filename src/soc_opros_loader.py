"""
SocOpros Data Loader Module
Specialized module for loading and processing the 'soc opros' survey data.
Follows Insight Digger Constitution: Requirements-First Development
"""

import pandas as pd
import requests
from typing import Optional, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocOprosLoader:
    """
    Specialized loader for the 'soc opros' Google Sheets survey data.
    Handles dynamic table structure with variable statements and respondents.
    """
    
    def __init__(self):
        """Initialize the SocOpros loader with default configuration."""
        self.sheet_id = "17oJL-hVMqOehHFugKHDJBmGtbWkp7e1y4ccJFnxwapk"
        self.worksheet_gid = "992488085"  # soc opros worksheet
        self.base_url = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        self.data = None
        self.statements = []
        self.respondents = []
        self.responses = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the soc opros data from Google Sheets.
        
        Returns:
            pd.DataFrame: Raw data from the spreadsheet
            
        Raises:
            Exception: If data loading fails
        """
        try:
            url = self.base_url.format(sheet_id=self.sheet_id, gid=self.worksheet_gid)
            logger.info(f"Loading data from: {url}")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Handle encoding properly for Cyrillic text
            from io import StringIO
            
            # Set proper encoding for the response
            response.encoding = 'utf-8'
            text_content = response.text
            
            # Read CSV with first row as header
            self.data = pd.read_csv(StringIO(text_content), header=0)
            
            # If columns are still "Unnamed", use first data row as column names
            if any(col.startswith('Unnamed') for col in self.data.columns):
                # Extract first row as column names
                new_columns = ['statements'] + [str(name) for name in self.data.iloc[0, 1:].values]
                
                # Create new dataframe with proper column names
                self.data.columns = new_columns
                
                # Remove the first row since it's now used as column names
                self.data = self.data.iloc[1:].reset_index(drop=True)
            
            logger.info(f"Data loaded successfully with encoding: {response.encoding} - Shape: {self.data.shape}")
            logger.info(f"Column names: {list(self.data.columns)}")
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise Exception(f"Data loading failed: {e}")
    
    def _fix_encoding(self, text):
        """
        Fix encoding issues with Cyrillic text from Google Sheets CSV export.
        
        Args:
            text (str): Text that might have encoding issues
            
        Returns:
            str: Properly encoded text
        """
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Skip if already looks like proper text
        if all(ord(c) < 128 for c in text if c.isalpha()):
            return text
        
        # Try multiple encoding fixes for Cyrillic text
        try:
            # Method 1: If it looks like UTF-8 was decoded as latin1
            if any(char in text for char in ['Ð', 'Ñ', 'ñ', 'Ð']):
                try:
                    fixed = text.encode('latin1').decode('utf-8')
                    return fixed
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
            
            # Method 2: Try direct UTF-8 decode of bytes representation
            if '\\x' in repr(text):
                try:
                    # Convert escape sequences back to bytes then decode
                    bytes_data = text.encode('latin1')
                    return bytes_data.decode('utf-8')
                except (UnicodeDecodeError, UnicodeEncodeError):
                    pass
                    
        except Exception:
            pass
        
        return text
    
    def get_respondent_names_fixed(self):
        """
        Get properly decoded respondent names.
        
        Returns:
            List[str]: List of respondent names with fixed encoding
        """
        if self.data is None:
            return []
            
        # Try to get names from first data row (row 0 contains headers/names)
        if self.data.shape[0] > 0:
            first_row = self.data.iloc[0, 1:].values  # Skip first column (statements)
            names = []
            
            for name in first_row:
                if name and str(name) != 'nan':
                    fixed_name = self._fix_encoding(str(name))
                    names.append(fixed_name)
            
            return names
        
        return []
    
    def parse_structure(self) -> Dict:
        """
        Parse the survey structure to identify statements and respondents.
        
        Returns:
            Dict: Structure information including statements, respondents, and metadata
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        # First column contains statements (skip header if present)
        statements_column = self.data.iloc[:, 0]
        
        # Filter out empty or invalid statements
        self.statements = [
            stmt for stmt in statements_column.dropna() 
            if isinstance(stmt, str) and stmt.strip() and not stmt.startswith('Unnamed')
        ]
        
        # Respondent columns are the remaining columns (excluding first)
        respondent_columns = self.data.columns[1:]
        
        # Extract respondent names from column headers (now properly set)
        self.respondents = [str(col) for col in respondent_columns if str(col) != 'nan']
            
        # Clean up empty or invalid respondent names
        self.respondents = [name for name in self.respondents if name and name.strip()]
        
        structure_info = {
            'total_statements': len(self.statements),
            'total_respondents': len(self.respondents),
            'data_shape': self.data.shape,
            'statements_preview': self.statements[:3] if self.statements else [],
            'respondents_preview': self.respondents[:3] if self.respondents else []
        }
        
        logger.info(f"Structure parsed - {structure_info['total_statements']} statements, "
                   f"{structure_info['total_respondents']} respondents")
        
        return structure_info
    
    def get_responses_matrix(self) -> pd.DataFrame:
        """
        Extract the responses matrix (statements × respondents).
        
        Returns:
            pd.DataFrame: Clean responses matrix with proper indexing
        """
        if self.data is None:
            raise ValueError("Data must be loaded first using load_data()")
        
        # Parse structure first to get statements and respondents
        self.parse_structure()
        
        # Create responses matrix
        if self.respondents:
            # Use identified respondent names
            response_data = self.data.iloc[:len(self.statements), 1:len(self.respondents)+1]
            self.responses = pd.DataFrame(
                response_data.values,
                index=self.statements,
                columns=self.respondents
            )
        else:
            # Use column indices if names not available
            response_data = self.data.iloc[:len(self.statements), 1:]
            self.responses = pd.DataFrame(
                response_data.values,
                index=self.statements,
                columns=[f"Respondent_{i+1}" for i in range(response_data.shape[1])]
            )
        
        logger.info(f"Responses matrix created - Shape: {self.responses.shape}")
        return self.responses
    
    def get_statements(self) -> List[str]:
        """
        Get the list of survey statements.
        
        Returns:
            List[str]: List of survey statements
        """
        if not self.statements:
            self.parse_structure()
        return self.statements
    
    def get_respondents(self) -> List[str]:
        """
        Get the list of respondent identifiers.
        
        Returns:
            List[str]: List of respondent names/identifiers
        """
        if not self.respondents:
            self.parse_structure()
        return self.respondents
    
    def get_response_summary(self) -> Dict:
        """
        Get a summary of the response data including statistics.
        
        Returns:
            Dict: Summary statistics and information about responses
        """
        if self.responses is None:
            self.get_responses_matrix()
        
        # Analyze response patterns
        unique_responses = set()
        for col in self.responses.columns:
            unique_responses.update(self.responses[col].dropna().astype(str))
        
        # Count response frequencies
        response_counts = {}
        for col in self.responses.columns:
            for response in self.responses[col].dropna().astype(str):
                response_counts[response] = response_counts.get(response, 0) + 1
        
        summary = {
            'total_responses': self.responses.size,
            'non_null_responses': self.responses.count().sum(),
            'unique_response_values': sorted(list(unique_responses)),
            'response_frequencies': response_counts,
            'completion_rate': (self.responses.count().sum() / self.responses.size) * 100,
            'statements_count': len(self.statements),
            'respondents_count': len(self.respondents)
        }
        
        return summary
    
    def export_clean_data(self, format_type: str = 'dataframe') -> pd.DataFrame:
        """
        Export clean, structured data ready for analysis.
        
        Args:
            format_type (str): Export format ('dataframe', 'csv', 'json')
            
        Returns:
            pd.DataFrame: Clean data ready for analysis
        """
        if self.responses is None:
            self.get_responses_matrix()
        
        if format_type == 'dataframe':
            return self.responses
        elif format_type == 'csv':
            return self.responses.to_csv()
        elif format_type == 'json':
            return self.responses.to_json(orient='index')
        else:
            raise ValueError(f"Unsupported format: {format_type}")


def load_soc_opros_data() -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to load soc opros data and structure information.
    
    Returns:
        Tuple[pd.DataFrame, Dict]: (responses_matrix, structure_info)
    """
    loader = SocOprosLoader()
    loader.load_data()
    
    responses = loader.get_responses_matrix()
    structure = loader.parse_structure()
    
    return responses, structure


# Example usage and testing functions
if __name__ == "__main__":
    # Test the loader
    print("Testing SocOpros Loader...")
    
    try:
        loader = SocOprosLoader()
        data = loader.load_data()
        print(f"✅ Data loaded: {data.shape}")
        
        structure = loader.parse_structure()
        print(f"✅ Structure parsed: {structure}")
        
        responses = loader.get_responses_matrix()
        print(f"✅ Responses matrix: {responses.shape}")
        
        summary = loader.get_response_summary()
        print(f"✅ Summary: {summary}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")