#!/usr/bin/env python3
"""
Simple Octav API Test
Tests different wallet addresses and parameter combinations
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_different_wallets():
    """Test with different wallet addresses"""
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print("🔗 Testing Different Wallet Addresses...")
    
    # Test different wallet addresses
    test_wallets = [
        "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18",  # From implementation.md
        "0x0000000000000000000000000000000000000000",  # Zero address
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"   # UNI token contract
    ]
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    for wallet in test_wallets:
        print(f"\n📊 Testing wallet: {wallet}")
        
        # Try minimal parameters first
        params = {
            'addresses': wallet
        }
        
        try:
            response = requests.get(
                'https://api.octav.fi/v1/portfolio',
                headers=headers,
                params=params,
                timeout=30
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success! Response type: {type(data)}")
                if isinstance(data, list):
                    print(f"   📊 Number of portfolios: {len(data)}")
                elif isinstance(data, dict):
                    print(f"   📊 Keys: {list(data.keys())}")
            else:
                print(f"   ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_parameter_combinations():
    """Test different parameter combinations"""
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print("\n🔧 Testing Parameter Combinations...")
    
    test_wallet = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"  # From implementation.md
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Test different parameter combinations
    test_params = [
        {'addresses': test_wallet},
        {'addresses': test_wallet, 'includeNFTs': 'false'},
        {'addresses': test_wallet, 'includeNFTs': 'false', 'includeImages': 'false'},
        {'addresses': test_wallet, 'includeNFTs': 'false', 'includeImages': 'false', 'includeExplorerUrls': 'false'},
        {'addresses': test_wallet, 'includeNFTs': 'false', 'includeImages': 'false', 'includeExplorerUrls': 'false', 'waitForSync': 'false'}
    ]
    
    for i, params in enumerate(test_params):
        print(f"\n   Test {i+1}: {params}")
        
        try:
            response = requests.get(
                'https://api.octav.fi/v1/portfolio',
                headers=headers,
                params=params,
                timeout=30
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ Success!")
                break
            else:
                print(f"   ❌ Failed: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_api_key_permissions():
    """Test if API key has the right permissions"""
    api_key = os.getenv('OCTAV_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return
    
    print("\n🔑 Testing API Key Permissions...")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # Try to access a simple endpoint or check permissions
    try:
        # Try a simple GET request to see if we get any response
        response = requests.get(
            'https://api.octav.fi/v1/portfolio',
            headers=headers,
            timeout=10
        )
        
        print(f"   Status without parameters: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 400:
            print("   ✅ API key is valid, but missing required parameters")
        elif response.status_code == 401:
            print("   ❌ API key is invalid or expired")
        elif response.status_code == 403:
            print("   ❌ API key doesn't have required permissions")
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

def main():
    """Run all tests"""
    print("🧪 Simple Octav API Test")
    print("=" * 40)
    
    test_different_wallets()
    test_parameter_combinations()
    test_api_key_permissions()
    
    print("\n" + "=" * 40)
    print("📝 Test Complete!")

if __name__ == "__main__":
    main()
