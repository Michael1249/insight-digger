"""
Google Sheets API Configuration Module

This module handles authentication and connection setup for Google Sheets API.
Supports both service account and OAuth2 authentication methods.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import gspread
from google.auth import default
from google.auth.exceptions import DefaultCredentialsError
from google.oauth2.service_account import Credentials
from google.oauth2.credentials import Credentials as OAuthCredentials


class GoogleSheetsConfig:
    """Configuration manager for Google Sheets API connections."""
    
    def __init__(self):
        self.credentials = None
        self.client = None
        self.scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]
    
    def setup_credentials(self, credentials_path: Optional[str] = None) -> bool:
        """
        Setup Google Sheets API credentials.
        
        Args:
            credentials_path: Path to service account JSON file
            
        Returns:
            bool: True if credentials setup successful, False otherwise
        """
        try:
            # Method 1: Service Account from file
            if credentials_path and Path(credentials_path).exists():
                self.credentials = Credentials.from_service_account_file(
                    credentials_path, 
                    scopes=self.scopes
                )
                print(f"✅ Loaded service account credentials from {credentials_path}")
                return True
            
            # Method 2: Service Account from environment variable
            creds_env = os.getenv('GOOGLE_SHEETS_CREDENTIALS_JSON')
            if creds_env:
                creds_info = json.loads(creds_env)
                self.credentials = Credentials.from_service_account_info(
                    creds_info,
                    scopes=self.scopes
                )
                print("✅ Loaded service account credentials from environment")
                return True
            
            # Method 3: Default credentials (for development)
            try:
                self.credentials, _ = default(scopes=self.scopes)
                print("✅ Using default Google credentials")
                return True
            except DefaultCredentialsError:
                pass
            
            print("❌ No valid credentials found")
            print("💡 Available options:")
            print("   1. Set GOOGLE_SHEETS_CREDENTIALS_JSON environment variable")
            print("   2. Provide credentials_path to service account file")
            print("   3. Use 'gcloud auth application-default login' for development")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up credentials: {e}")
            return False
    
    def get_client(self, credentials_path: Optional[str] = None) -> Optional[gspread.Client]:
        """
        Get authenticated gspread client.
        
        Args:
            credentials_path: Path to service account JSON file
            
        Returns:
            gspread.Client or None if authentication fails
        """
        if not self.credentials:
            if not self.setup_credentials(credentials_path):
                return None
        
        try:
            self.client = gspread.authorize(self.credentials)
            print("✅ Google Sheets client authenticated successfully")
            return self.client
        except Exception as e:
            print(f"❌ Error creating client: {e}")
            return None
    
    def test_connection(self, sheet_url_or_id: str) -> bool:
        """
        Test connection to a specific Google Sheet.
        
        Args:
            sheet_url_or_id: Google Sheet URL or ID
            
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not self.client:
            print("❌ No authenticated client available")
            return False
        
        try:
            # Extract sheet ID from URL if needed
            if 'docs.google.com' in sheet_url_or_id:
                sheet_id = sheet_url_or_id.split('/d/')[1].split('/')[0]
            else:
                sheet_id = sheet_url_or_id
            
            sheet = self.client.open_by_key(sheet_id)
            worksheets = sheet.worksheets()
            
            print(f"✅ Successfully connected to sheet: {sheet.title}")
            print(f"📋 Found {len(worksheets)} worksheets:")
            for ws in worksheets:
                print(f"   - {ws.title} ({ws.row_count} rows x {ws.col_count} cols)")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def get_demo_config() -> Dict[str, Any]:
    """
    Get demo configuration for testing without real credentials.
    
    Returns:
        dict: Demo configuration settings
    """
    return {
        'demo_mode': True,
        'demo_data': {
            'sheet_name': 'Demo Data',
            'columns': ['Date', 'Category', 'Value', 'Description'],
            'sample_rows': [
                ['2024-01-01', 'Sales', 1000, 'Product A sales'],
                ['2024-01-02', 'Marketing', 500, 'Ad campaign cost'],
                ['2024-01-03', 'Sales', 1200, 'Product B sales'],
                ['2024-01-04', 'Operations', 300, 'Maintenance cost'],
                ['2024-01-05', 'Sales', 800, 'Product A sales'],
            ]
        }
    }


# Convenience function for quick setup
def quick_setup(credentials_path: Optional[str] = None) -> Optional[gspread.Client]:
    """
    Quick setup function for Google Sheets client.
    
    Args:
        credentials_path: Path to service account JSON file
        
    Returns:
        gspread.Client or None
    """
    config = GoogleSheetsConfig()
    return config.get_client(credentials_path)


if __name__ == "__main__":
    # Test the configuration
    config = GoogleSheetsConfig()
    client = config.get_client()
    
    if client:
        print("🎉 Configuration test successful!")
    else:
        print("⚠️  Running in demo mode - see notebooks for sample data")