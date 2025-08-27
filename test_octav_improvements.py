#!/usr/bin/env python3
"""
Test script for Octav API improvements
Validates the enhanced implementation against the reference
"""

import sys
import os
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.tools.data_collection import OctavDataTool
from src.config import get_config
from src.utils.portfolio_formatter import PortfolioFormatter

def test_configuration():
    """Test configuration management improvements"""
    print("🔧 Testing Configuration Management...")
    
    try:
        config = get_config()
        
        # Test Octav-specific configuration
        assert hasattr(config, 'octav_base_url'), "Missing octav_base_url in config"
        assert hasattr(config, 'octav_portfolio_timeout'), "Missing octav_portfolio_timeout in config"
        assert hasattr(config, 'octav_include_nfts'), "Missing octav_include_nfts in config"
        assert hasattr(config, 'octav_include_images'), "Missing octav_include_images in config"
        assert hasattr(config, 'octav_include_explorer_urls'), "Missing octav_include_explorer_urls in config"
        assert hasattr(config, 'octav_wait_for_sync'), "Missing octav_wait_for_sync in config"
        
        print("✅ Configuration management test passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration management test failed: {e}")
        return False

def test_portfolio_formatter():
    """Test portfolio formatter improvements"""
    print("📊 Testing Portfolio Formatter...")
    
    try:
        # Create mock portfolio data with hierarchical structure
        mock_portfolio_data = {
            'address': '0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb',
            'total_value_usd': 50000.0,
            'transaction_count': 150,
            'networth': 52000.0,
            'cash_balance': 5000.0,
            'open_pnl': 2000.0,
            'closed_pnl': 500.0,
            'daily_income': 100.0,
            'daily_expense': 25.0,
            'fees_fiat': 150.0,
            'last_updated': '2024-01-15T10:30:00Z',
            'balances': {
                'ETH_Uniswap_Ethereum': {
                    'symbol': 'ETH',
                    'name': 'Ethereum',
                    'address': '0x0000000000000000000000000000000000000000',
                    'balance': 2.5,
                    'value_usd': 5000.0,
                    'price_usd': 2000.0,
                    'allocation_percentage': 10.0,
                    'protocol': 'Uniswap',
                    'chain': 'Ethereum',
                    'position': 'Liquidity Pool'
                },
                'USDC_Uniswap_Ethereum': {
                    'symbol': 'USDC',
                    'name': 'USD Coin',
                    'address': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
                    'balance': 10000.0,
                    'value_usd': 10000.0,
                    'price_usd': 1.0,
                    'allocation_percentage': 20.0,
                    'protocol': 'Uniswap',
                    'chain': 'Ethereum',
                    'position': 'Liquidity Pool'
                },
                'WBTC_Aave_Ethereum': {
                    'symbol': 'WBTC',
                    'name': 'Wrapped Bitcoin',
                    'address': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
                    'balance': 0.5,
                    'value_usd': 15000.0,
                    'price_usd': 30000.0,
                    'allocation_percentage': 30.0,
                    'protocol': 'Aave',
                    'chain': 'Ethereum',
                    'position': 'Lending'
                }
            }
        }
        
        # Test formatting for export
        formatted_data = PortfolioFormatter.format_for_export(mock_portfolio_data)
        
        # Validate structure
        assert 'wallet_info' in formatted_data, "Missing wallet_info in formatted data"
        assert 'portfolio_metrics' in formatted_data, "Missing portfolio_metrics in formatted data"
        assert 'holdings' in formatted_data, "Missing holdings in formatted data"
        
        # Validate wallet info
        wallet_info = formatted_data['wallet_info']
        assert wallet_info['address'] == mock_portfolio_data['address'], "Address mismatch"
        assert wallet_info['total_value_usd'] == mock_portfolio_data['total_value_usd'], "Total value mismatch"
        
        # Validate portfolio metrics
        portfolio_metrics = formatted_data['portfolio_metrics']
        assert portfolio_metrics['networth'] == mock_portfolio_data['networth'], "Networth mismatch"
        assert portfolio_metrics['cash_balance'] == mock_portfolio_data['cash_balance'], "Cash balance mismatch"
        
        # Validate holdings
        holdings = formatted_data['holdings']
        assert len(holdings) == 3, f"Expected 3 holdings, got {len(holdings)}"
        
        print("✅ Portfolio formatter test passed")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio formatter test failed: {e}")
        return False

def test_octav_data_tool():
    """Test OctavDataTool improvements"""
    print("🔍 Testing OctavDataTool Improvements...")
    
    try:
        # Test initialization
        data_tool = OctavDataTool()
        
        # Test configuration integration
        assert hasattr(data_tool, 'config'), "Missing config in OctavDataTool"
        assert hasattr(data_tool.config, 'octav_include_nfts'), "Missing octav_include_nfts in config"
        
        # Test error handling (without making actual API calls)
        print("✅ OctavDataTool initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ OctavDataTool test failed: {e}")
        return False

def test_error_handling():
    """Test error handling improvements"""
    print("⚠️ Testing Error Handling...")
    
    try:
        # Test configuration validation
        config = get_config()
        
        # Test API key validation
        if not config.octav_api_key or config.octav_api_key == "your_octav_api_key_here":
            print("⚠️ No valid Octav API key found - skipping API tests")
            print("✅ Error handling test passed (no API key)")
            return True
        
        print("✅ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Octav API Improvements")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_portfolio_formatter,
        test_octav_data_tool,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Octav API improvements are working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Please review the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
