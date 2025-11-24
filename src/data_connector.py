"""
Data Connector Module

Handles data ingestion from multiple sources including Google Sheets and CSV files.
Provides unified interface for data loading with fallback options.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Any
import requests
from io import StringIO
import json
import os


class DataConnector:
    """Unified data connector for multiple data sources."""
    
    def __init__(self):
        self.data_cache = {}
        self.source_info = {}
    
    def load_google_sheet(self, 
                         sheet_url_or_id: str, 
                         worksheet_name: Optional[str] = None,
                         credentials_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data from Google Sheets using API.
        
        Args:
            sheet_url_or_id: Google Sheet URL or ID
            worksheet_name: Specific worksheet name (default: first sheet)
            credentials_path: Path to credentials file
            
        Returns:
            pandas.DataFrame or None if loading fails
        """
        try:
            from config.google_sheets_config import GoogleSheetsConfig
            
            config = GoogleSheetsConfig()
            client = config.get_client(credentials_path)
            
            if not client:
                print("❌ Failed to authenticate with Google Sheets API")
                return None
            
            # Extract sheet ID if URL provided
            if 'docs.google.com' in sheet_url_or_id:
                sheet_id = sheet_url_or_id.split('/d/')[1].split('/')[0]
            else:
                sheet_id = sheet_url_or_id
            
            # Open the sheet
            sheet = client.open_by_key(sheet_id)
            
            # Select worksheet
            if worksheet_name:
                worksheet = sheet.worksheet(worksheet_name)
            else:
                worksheet = sheet.get_worksheet(0)  # First sheet
            
            # Get all values and convert to DataFrame
            data = worksheet.get_all_records()
            df = pd.DataFrame(data)
            
            # Store source info
            self.source_info[sheet_id] = {
                'type': 'google_sheets',
                'sheet_title': sheet.title,
                'worksheet_title': worksheet.title,
                'last_updated': pd.Timestamp.now(),
                'rows': len(df),
                'columns': len(df.columns)
            }
            
            print(f"✅ Loaded {len(df)} rows from Google Sheet: {sheet.title} / {worksheet.title}")
            return df
            
        except ImportError:
            print("❌ Google Sheets dependencies not installed")
            print("💡 Install with: pip install gspread google-auth google-auth-oauthlib")
            return None
        except Exception as e:
            print(f"❌ Error loading Google Sheet: {e}")
            print("💡 Trying CSV fallback...")
            return None
    
    def load_csv_file(self, file_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load data from local CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            pandas.DataFrame or None if loading fails
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                print(f"❌ File not found: {file_path}")
                return None
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    
                    # Store source info
                    self.source_info[str(file_path)] = {
                        'type': 'csv_file',
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'last_updated': pd.Timestamp.now(),
                        'rows': len(df),
                        'columns': len(df.columns),
                        'encoding': encoding
                    }
                    
                    print(f"✅ Loaded {len(df)} rows from CSV: {file_path.name}")
                    return df
                    
                except UnicodeDecodeError:
                    continue
            
            print(f"❌ Could not read CSV file with any encoding: {file_path}")
            return None
            
        except Exception as e:
            print(f"❌ Error loading CSV file: {e}")
            return None
    
    def load_google_sheet_csv(self, sheet_id: str, gid: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load Google Sheet data via CSV export (works with public sheets without authentication).
        
        Args:
            sheet_id: Google Sheet ID (from URL)
            gid: Worksheet GID (from URL, optional for first sheet)
            
        Returns:
            pandas.DataFrame or None if loading fails
        """
        try:
            # Construct CSV export URL
            if gid:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            else:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            
            print(f"📥 Loading via CSV export: {csv_url}")
            
            # Download CSV data
            response = requests.get(csv_url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from response
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Store source info
            self.source_info[sheet_id] = {
                'type': 'google_sheets_csv',
                'sheet_id': sheet_id,
                'gid': gid,
                'csv_url': csv_url,
                'last_updated': pd.Timestamp.now(),
                'rows': len(df),
                'columns': len(df.columns)
            }
            
            print(f"✅ Loaded {len(df)} rows via CSV export")
            return df
            
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return None
        """
        Load data from CSV URL.
        
        Args:
            url: URL to CSV file
            
        Returns:
            pandas.DataFrame or None if loading fails
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from string
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Store source info
            self.source_info[url] = {
                'type': 'csv_url',
                'url': url,
                'last_updated': pd.Timestamp.now(),
                'rows': len(df),
                'columns': len(df.columns),
                'response_size': len(response.content)
            }
            
            print(f"✅ Loaded {len(df)} rows from CSV URL: {url}")
            return df
            
        except Exception as e:
            print(f"❌ Error loading CSV from URL: {e}")
            return None
    
    def create_demo_data(self) -> pd.DataFrame:
        """
        Create demo dataset for testing and examples.
        
        Returns:
            pandas.DataFrame with sample data
        """
        np.random.seed(42)  # For reproducible demo data
        
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        categories = ['Sales', 'Marketing', 'Operations', 'Support', 'Development']
        
        data = []
        for date in dates:
            for _ in range(np.random.randint(1, 4)):  # 1-3 entries per day
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Category': np.random.choice(categories),
                    'Value': np.random.randint(100, 2000),
                    'Region': np.random.choice(['North', 'South', 'East', 'West']),
                    'Product': np.random.choice(['Product A', 'Product B', 'Product C']),
                    'Description': f"Sample entry {len(data) + 1}"
                })
        
        df = pd.DataFrame(data)
        
        # Store source info
        self.source_info['demo_data'] = {
            'type': 'demo_data',
            'generated_at': pd.Timestamp.now(),
            'rows': len(df),
            'columns': len(df.columns),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}"
        }
        
        print(f"✅ Generated demo dataset with {len(df)} rows")
        return df
    
    def auto_load(self, 
                  source: str, 
                  worksheet_name: Optional[str] = None,
                  credentials_path: Optional[str] = None) -> pd.DataFrame:
        """
        Automatically detect source type and load data with fallbacks.
        
        Args:
            source: Data source (URL, file path, or 'demo')
            worksheet_name: For Google Sheets, specific worksheet name
            credentials_path: For Google Sheets, credentials file path
            
        Returns:
            pandas.DataFrame (demo data if all sources fail)
        """
        print(f"🔄 Attempting to load data from: {source}")
        
        # Demo data
        if source.lower() == 'demo':
            return self.create_demo_data()
        
        # Google Sheets (URL or ID)
        if 'docs.google.com' in source or len(source.replace('-', '')) == 44:
            df = self.load_google_sheet(source, worksheet_name, credentials_path)
            if df is not None:
                return df
        
        # CSV URL
        if source.startswith('http'):
            df = self.load_csv_url(source)
            if df is not None:
                return df
        
        # Local CSV file
        if Path(source).suffix.lower() == '.csv':
            df = self.load_csv_file(source)
            if df is not None:
                return df
        
        # Fallback to demo data
        print("⚠️  All data loading attempts failed, using demo data")
        return self.create_demo_data()
    
    def get_source_info(self, source_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about loaded data sources.
        
        Args:
            source_key: Specific source to get info for (None for all)
            
        Returns:
            dict: Source information
        """
        if source_key:
            return self.source_info.get(source_key, {})
        return self.source_info
    
    def save_to_cache(self, df: pd.DataFrame, cache_key: str) -> None:
        """Save DataFrame to cache."""
        self.data_cache[cache_key] = df.copy()
        print(f"💾 Saved data to cache: {cache_key}")
    
    def load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache."""
        if cache_key in self.data_cache:
            print(f"📋 Loaded data from cache: {cache_key}")
            return self.data_cache[cache_key].copy()
        return None


# Convenience functions
def quick_load(source: str, **kwargs) -> pd.DataFrame:
    """Quick data loading function."""
    connector = DataConnector()
    return connector.auto_load(source, **kwargs)


def load_demo_data() -> pd.DataFrame:
    """Quick demo data loading."""
    connector = DataConnector()
    return connector.create_demo_data()


if __name__ == "__main__":
    # Test the connector
    connector = DataConnector()
    
    print("🧪 Testing data connector...")
    
    # Test demo data
    demo_df = connector.create_demo_data()
    print(f"Demo data shape: {demo_df.shape}")
    print(f"Columns: {list(demo_df.columns)}")
    
    # Show source info
    print(f"Source info: {connector.get_source_info()}")