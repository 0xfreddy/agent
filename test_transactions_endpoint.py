#!/usr/bin/env python3
"""
Test script for the new Octav Transactions API endpoint implementation
"""

import requests
import json
from datetime import datetime

def test_transactions_endpoint():
    """Test the new transactions endpoint"""
    
    # Test wallet address
    wallet_address = "0x9ca3fec9003cd8e84cfee35800fd3e05a88eee18"
    
    print("🧪 Testing Octav Transactions API Implementation")
    print("=" * 60)
    
    # Test 1: Basic transaction fetch
    print("\n1. Testing basic transaction fetch...")
    try:
        response = requests.get(
            "http://localhost:3001/api/transactions",
            params={
                "addresses": wallet_address,
                "limit": "10",
                "sort": "DESC"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            print(f"✅ Success! Retrieved {len(transactions)} transactions")
            
            if transactions:
                print(f"   First transaction: {transactions[0].get('type', 'Unknown')} on {transactions[0].get('chain', {}).get('name', 'Unknown')}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Filtered transactions (SWAP only)
    print("\n2. Testing filtered transactions (SWAP only)...")
    try:
        response = requests.get(
            "http://localhost:3001/api/transactions",
            params={
                "addresses": wallet_address,
                "limit": "5",
                "txTypes": "SWAP",
                "sort": "DESC"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            print(f"✅ Success! Retrieved {len(transactions)} SWAP transactions")
            
            if transactions:
                all_swaps = all(tx.get('type') == 'SWAP' for tx in transactions)
                print(f"   All transactions are SWAP: {all_swaps}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Network filtered transactions
    print("\n3. Testing network filtered transactions (Ethereum only)...")
    try:
        response = requests.get(
            "http://localhost:3001/api/transactions",
            params={
                "addresses": wallet_address,
                "limit": "5",
                "networks": "ethereum",
                "sort": "DESC"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            print(f"✅ Success! Retrieved {len(transactions)} Ethereum transactions")
            
            if transactions:
                all_ethereum = all(tx.get('chain', {}).get('key') == 'ethereum' for tx in transactions)
                print(f"   All transactions are on Ethereum: {all_ethereum}")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 4: Spam filtering
    print("\n4. Testing spam filtering...")
    try:
        response = requests.get(
            "http://localhost:3001/api/transactions",
            params={
                "addresses": wallet_address,
                "limit": "10",
                "hideSpam": "true",
                "sort": "DESC"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            transactions = data.get('transactions', [])
            print(f"✅ Success! Retrieved {len(transactions)} non-spam transactions")
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Transaction endpoint testing complete!")

if __name__ == "__main__":
    test_transactions_endpoint()
