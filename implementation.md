# Complete Octav Portfolio API Implementation

## Product Requirements Document (PRD)

### 1. Overview
**Product Name**: Wallet Portfolio Retriever using Octav API  
**Purpose**: Retrieve and display comprehensive portfolio data for any wallet address across multiple blockchains and DeFi protocols  
**Target Users**: Developers, analysts, and applications needing wallet portfolio data  

### 2. Functional Requirements

#### 2.1 Core Features
- **Portfolio Retrieval**: Fetch complete portfolio data for any wallet address
- **Multi-Chain Support**: Display assets across all supported blockchains
- **Protocol Integration**: Show DeFi protocol positions and yields
- **Real-time Data**: Access fresh portfolio data with configurable sync options
- **Formatted Output**: Present data in human-readable, hierarchical format

#### 2.2 API Integration Requirements
- **Endpoint**: `https://api.octav.fi/v1/portfolio`
- **Authentication**: Bearer token authentication
- **Method**: GET request with query parameters
- **Response**: JSON format with portfolio data structure

#### 2.3 Configuration Options
- **includeNFTs**: Boolean flag for NFT inclusion
- **includeImages**: Boolean flag for image URLs
- **includeExplorerUrls**: Boolean flag for explorer links
- **waitForSync**: Boolean flag for fresh data retrieval

### 3. Technical Requirements

#### 3.1 Dependencies
```python
requests==2.31.0      # HTTP client for API calls
python-dotenv==1.0.0  # Environment variable management
```

#### 3.2 Error Handling
- API key validation
- Network error handling
- JSON parsing error handling
- HTTP status code validation
- Graceful error messages

#### 3.3 Data Processing
- Portfolio data parsing and validation
- Hierarchical data structure formatting
- Currency and value formatting
- Asset categorization by protocol and chain

### 4. User Experience Requirements
- Clear, formatted output with visual separators
- Hierarchical display of portfolio structure
- Comprehensive asset information (balance, price, value)
- Easy-to-read currency formatting
- Detailed protocol and chain breakdown

---

## Complete Implementation Code

### 1. requirements.txt
```txt
requests==2.31.0
python-dotenv==1.0.0
```

### 2. portfolio_retriever.py
```python
#!/usr/bin/env python3
"""
Wallet Portfolio Retriever using Octav API
Retrieves portfolio data for a given wallet address
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class OctavPortfolioRetriever:
    """Class to handle Octav API portfolio requests"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the portfolio retriever
        
        Args:
            api_key: Octav API key. If not provided, will try to get from environment variable OCTAV_API_KEY
        """
        self.base_url = "https://api.octav.fi"
        self.api_key = api_key or os.getenv('OCTAV_API_KEY')
        
        if not self.api_key:
            raise ValueError("API key is required. Set OCTAV_API_KEY environment variable or pass api_key parameter")
    
    def get_portfolio(self, address: str, include_nfts: bool = False, 
                     include_images: bool = False, include_explorer_urls: bool = False,
                     wait_for_sync: bool = False) -> Dict[str, Any]:
        """
        Retrieve portfolio data for a given wallet address
        
        Args:
            address: Wallet address to query
            include_nfts: Include NFTs in the response
            include_images: Include image links in the response
            include_explorer_urls: Include explorer links in the response
            wait_for_sync: Wait for fresh data (may take longer)
            
        Returns:
            Portfolio data as dictionary
        """
        endpoint = f"{self.base_url}/v1/portfolio"
        
        # Prepare query parameters
        params = {
            'addresses': address,
            'includeNFTs': str(include_nfts).lower(),
            'includeImages': str(include_images).lower(),
            'includeExplorerUrls': str(include_explorer_urls).lower(),
            'waitForSync': str(wait_for_sync).lower()
        }
        
        # Prepare headers
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            print(f"Fetching portfolio for address: {address}")
            response = requests.get(endpoint, params=params, headers=headers)
            
            # Check if request was successful
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            raise

def format_portfolio_data(portfolio_data: list) -> None:
    """
    Format and display portfolio data in a readable way
    
    Args:
        portfolio_data: List of portfolio data from API response
    """
    if not portfolio_data:
        print("No portfolio data found")
        return
    
    for portfolio in portfolio_data:
        print("=" * 60)
        print(f"WALLET PORTFOLIO")
        print("=" * 60)
        print(f"Address: {portfolio.get('address', 'N/A')}")
        print(f"Net Worth: ${portfolio.get('networth', 'N/A')}")
        print(f"Cash Balance: ${portfolio.get('cashBalance', 'N/A')}")
        print(f"Open PnL: ${portfolio.get('openPnl', 'N/A')}")
        print(f"Closed PnL: ${portfolio.get('closedPnl', 'N/A')}")
        print(f"Daily Income: ${portfolio.get('dailyIncome', 'N/A')}")
        print(f"Daily Expense: ${portfolio.get('dailyExpense', 'N/A')}")
        print(f"Fees: ${portfolio.get('feesFiat', 'N/A')}")
        print(f"Last Updated: {portfolio.get('lastUpdated', 'N/A')}")
        
        # Display chains summary
        chains = portfolio.get('chains', {})
        if chains:
            print("\n" + "-" * 40)
            print("CHAINS SUMMARY")
            print("-" * 40)
            for chain_key, chain_data in chains.items():
                print(f"{chain_data.get('name', chain_key)}: ${chain_data.get('value', 'N/A')}")
        
        # Display assets by protocols
        asset_by_protocols = portfolio.get('assetByProtocols', {})
        if asset_by_protocols:
            print("\n" + "-" * 40)
            print("ASSETS BY PROTOCOLS")
            print("-" * 40)
            for protocol_key, protocol_data in asset_by_protocols.items():
                print(f"\n{protocol_data.get('name', protocol_key)}: ${protocol_data.get('value', 'N/A')}")
                
                # Display chains within each protocol
                chains = protocol_data.get('chains', {})
                for chain_key, chain_data in chains.items():
                    print(f"  └─ {chain_data.get('name', chain_key)}: ${chain_data.get('value', 'N/A')}")
                    
                    # Display protocol positions
                    protocol_positions = chain_data.get('protocolPositions', {})
                    for position_key, position_data in protocol_positions.items():
                        print(f"    └─ {position_data.get('name', position_key)}: ${position_data.get('totalValue', 'N/A')}")
                        
                        # Display assets
                        assets = position_data.get('assets', [])
                        for asset in assets:
                            print(f"      └─ {asset.get('name', 'Unknown')} ({asset.get('symbol', 'N/A')}): {asset.get('balance', 'N/A')} @ ${asset.get('price', 'N/A')} = ${asset.get('value', 'N/A')}")

def main():
    """Main function to retrieve and display portfolio data"""
    # Wallet address to query
    wallet_address = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    try:
        # Initialize the portfolio retriever with test API key
        test_api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1pZCI6Ik51dm9sYXJpLkFpICJ9fQ.aP2REhbebJw4wYRhyQzXUBH8HkBm41Zry_UteTdyUR4"
        retriever = OctavPortfolioRetriever(api_key=test_api_key)
        
        # Retrieve portfolio data
        portfolio_data = retriever.get_portfolio(
            address=wallet_address,
            include_nfts=False,
            include_images=False,
            include_explorer_urls=False,
            wait_for_sync=False
        )
        
        # Format and display the data
        format_portfolio_data(portfolio_data)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set your OCTAV_API_KEY environment variable or pass it to the OctavPortfolioRetriever constructor")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

### 3. example_usage.py
```python
#!/usr/bin/env python3
"""
Example usage of the Octav Portfolio Retriever
This script shows how to use the portfolio retriever with an API key
"""

from portfolio_retriever import OctavPortfolioRetriever, format_portfolio_data

def example_with_api_key():
    """Example using API key directly"""
    # Test API key for Octav
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1pZCI6Ik51dm9sYXJpLkFpICJ9fQ.aP2REhbebJw4wYRhyQzXUBH8HkBm41Zry_UteTdyUR4"
    
    # Wallet address to query
    wallet_address = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    try:
        # Initialize with API key
        retriever = OctavPortfolioRetriever(api_key=api_key)
        
        # Get portfolio data with all options enabled
        print("Fetching portfolio with all options enabled...")
        portfolio_data = retriever.get_portfolio(
            address=wallet_address,
            include_nfts=True,
            include_images=True,
            include_explorer_urls=True,
            wait_for_sync=False  # Set to True for fresh data
        )
        
        # Display the data
        format_portfolio_data(portfolio_data)
        
    except Exception as e:
        print(f"Error: {e}")

def example_with_env_variable():
    """Example using environment variable for API key"""
    # Make sure OCTAV_API_KEY is set in your environment
    # export OCTAV_API_KEY="your_api_key_here"
    
    wallet_address = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    try:
        # Initialize without API key (will use environment variable)
        retriever = OctavPortfolioRetriever()
        
        # Get basic portfolio data
        print("Fetching basic portfolio data...")
        portfolio_data = retriever.get_portfolio(
            address=wallet_address,
            include_nfts=False,
            include_images=False,
            include_explorer_urls=False,
            wait_for_sync=False
        )
        
        # Display the data
        format_portfolio_data(portfolio_data)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Octav Portfolio Retriever - Example Usage")
    print("=" * 50)
    
    print("\nTo use this example:")
    print("1. Replace 'your_octav_api_key_here' with your actual API key")
    print("2. Or set the OCTAV_API_KEY environment variable")
    print("3. Run the script")
    
    # Test with the provided API key
    example_with_api_key()
```

### 4. README.md
```markdown
# Wallet Portfolio Retriever

A Python script to retrieve wallet portfolio data using the Octav API.

## Features

- Retrieve portfolio data for any wallet address
- Display formatted portfolio information including:
  - Net worth, cash balance, PnL
  - Assets by protocols and chains
  - Individual token holdings and values
- Configurable options for NFTs, images, and explorer URLs
- Proper error handling and API response validation

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Get your Octav API key:**
   - Sign up at [Octav](https://api-docs.octav.fi/)
   - Get your API key from the dashboard

3. **Set your API key:**
   
   **Option A: Environment variable (recommended)**
   ```bash
   export OCTAV_API_KEY="your_api_key_here"
   ```
   
   **Option B: .env file**
   Create a `.env` file in the project directory:
   ```
   OCTAV_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage
```bash
python portfolio_retriever.py
```

This will retrieve portfolio data for the wallet address: `0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18`

### Custom Usage
You can modify the script to use different wallet addresses or API parameters:

```python
# Initialize with custom API key
retriever = OctavPortfolioRetriever(api_key="your_api_key")

# Get portfolio with custom options
portfolio_data = retriever.get_portfolio(
    address="0x1234...",
    include_nfts=True,
    include_images=True,
    include_explorer_urls=True,
    wait_for_sync=True
)
```

## API Parameters

- `addresses`: Comma-separated list of wallet addresses
- `includeNFTs`: Include NFTs in the response (boolean)
- `includeImages`: Include image links (boolean)
- `includeExplorerUrls`: Include explorer links (boolean)
- `waitForSync`: Wait for fresh data (boolean, may take longer)

## Output Format

The script displays:
- **Wallet Overview**: Address, net worth, cash balance, PnL
- **Chains Summary**: Total value per blockchain
- **Assets by Protocols**: Detailed breakdown of assets organized by DeFi protocols
- **Individual Assets**: Token balances, prices, and values

## Error Handling

The script includes comprehensive error handling for:
- Missing API keys
- Network errors
- API response errors
- JSON parsing errors

## Requirements

- Python 3.7+
- requests
- python-dotenv

## API Documentation

For more information about the Octav API, visit: https://api-docs.octav.fi/getting-started/portfolio
```

---

## Implementation Guide

### Step 1: Environment Setup
1. Create a new directory for your project
2. Copy all the files above into the directory
3. Install dependencies: `pip install -r requirements.txt`

### Step 2: API Key Configuration
1. Get an API key from Octav (https://api-docs.octav.fi/)
2. Set the environment variable: `export OCTAV_API_KEY="your_api_key"`
3. Or modify the script to include your API key directly

### Step 3: Usage
1. Run the main script: `python3 portfolio_retriever.py`
2. Or use the example script: `python3 example_usage.py`
3. Modify the wallet address in the script as needed

### Step 4: Customization
- Change the wallet address in the `main()` function
- Modify the `format_portfolio_data()` function for different output formats
- Add additional error handling or logging as needed
- Extend the `OctavPortfolioRetriever` class with additional methods

### Key Implementation Notes:
- The API uses Bearer token authentication
- All parameters are converted to lowercase strings for the API
- The response is a list of portfolio objects (one per address)
- Error handling includes both network and API-specific errors
- The formatting function creates a hierarchical display of the portfolio data

This implementation provides a complete, production-ready solution for retrieving wallet portfolio data using the Octav API.
