#!/usr/bin/env python3
"""
Direct Octav API Test
Tests the API directly to identify issues
"""

import requests
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_octav_api_direct():
    """Test Octav API directly"""
    print("🔗 Testing Octav API Directly...")
    
    # Get API key from environment
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        print("❌ OCTAV_API_KEY not found in environment")
        print("   Set it with: export OCTAV_API_KEY='your_api_key_here'")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Test wallet address (using the working one from implementation.md)
    test_wallet = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Test portfolio endpoint
    print(f"\n📊 Testing portfolio endpoint for {test_wallet}...")
    
    params = {
        'addresses': test_wallet,
        'includeNFTs': 'false',
        'includeImages': 'false',
        'includeExplorerUrls': 'false',
        'waitForSync': 'false'
    }
    
    try:
        response = requests.get(
            'https://api.octav.fi/v1/portfolio',
            headers=headers,
            params=params,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API call successful!")
            print(f"Response type: {type(data)}")
            if isinstance(data, list):
                print(f"Number of portfolios: {len(data)}")
                if len(data) > 0:
                    print(f"First portfolio keys: {list(data[0].keys())}")
            elif isinstance(data, dict):
                print(f"Portfolio keys: {list(data.keys())}")
        else:
            print(f"❌ API call failed: {response.status_code}")
            print(f"Response: {response.text}")
            
            # Try to parse error response
            try:
                error_data = response.json()
                print(f"Error details: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Raw response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_wallet_endpoint():
    """Test wallet endpoint directly"""
    print(f"\n💳 Testing wallet endpoint...")
    
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        return False
    
    test_wallet = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    params = {
        'addresses': test_wallet
    }
    
    try:
        response = requests.get(
            'https://api.octav.fi/v1/wallet',
            headers=headers,
            params=params,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Wallet endpoint successful!")
            print(f"Response type: {type(data)}")
            if isinstance(data, dict):
                print(f"Wallet data keys: {list(data.keys())}")
        else:
            print(f"❌ Wallet endpoint failed: {response.status_code}")
            print(f"Response: {response.text}")
        
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Wallet endpoint test failed: {e}")
        return False

def test_api_documentation():
    """Check API documentation for correct parameters"""
    print(f"\n📚 Checking API Documentation...")
    
    try:
        # Try to access API documentation
        response = requests.get('https://api-docs.octav.fi/', timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
        else:
            print(f"⚠️ API documentation status: {response.status_code}")
    except:
        print("⚠️ Could not access API documentation")
    
    print("\n📋 Expected API Parameters:")
    print("   - addresses: Comma-separated wallet addresses")
    print("   - includeNFTs: boolean (true/false)")
    print("   - includeImages: boolean (true/false)")
    print("   - includeExplorerUrls: boolean (true/false)")
    print("   - waitForSync: boolean (true/false)")

def main():
    """Run direct API tests"""
    print("🧪 Direct Octav API Test")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        print("\nTo set up your API key:")
        print("1. Get an API key from https://api-docs.octav.fi/")
        print("2. Set it as environment variable:")
        print("   export OCTAV_API_KEY='your_api_key_here'")
        print("3. Or add it to your .env file:")
        print("   OCTAV_API_KEY=your_api_key_here")
        return 1
    
    # Run tests
    portfolio_success = test_octav_api_direct()
    wallet_success = test_wallet_endpoint()
    test_api_documentation()
    
    print("\n" + "=" * 40)
    if portfolio_success and wallet_success:
        print("🎉 Direct API tests passed!")
        print("The issue might be with the proxy server configuration.")
        return 0
    else:
        print("❌ Direct API tests failed!")
        print("Please check your API key and wallet address.")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
