const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(express.json());

// Octav.fi API configuration
const OCTAV_API_KEY = process.env.OCTAV_API_KEY;
const OCTAV_BASE_URL = 'https://api.octav.fi';

// Helper function to make authenticated requests to Octav.fi
async function makeOctavRequest(endpoint, params = {}, method = 'GET') {
    try {
        const url = `${OCTAV_BASE_URL}${endpoint}`;
        
        const config = {
            method: method,
            url: url,
            headers: {
                'Authorization': `Bearer ${OCTAV_API_KEY}`,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            },
            timeout: 30000 // Increased timeout for waitForSync
        };

        if (method === 'GET') {
            config.params = params;
        } else {
            config.data = params;
        }
        
        const response = await axios(config);
        return response.data;
    } catch (error) {
        console.error(`Octav API error for ${endpoint}:`, error.response?.data || error.message);
        if (error.response) {
            console.error(`Status: ${error.response.status}`);
            console.error(`Response data:`, JSON.stringify(error.response.data, null, 2));
        }
        throw error;
    }
}

// Portfolio endpoint - uses /v1/portfolio
app.get('/api/portfolio', async (req, res) => {
    try {
        const { addresses } = req.query;
        
        if (!addresses) {
            return res.status(400).json({ error: 'addresses parameter is required' });
        }
        
        // Get parameters from request or use defaults
        const includeNFTs = req.query.includeNFTs || 'false';
        const includeImages = req.query.includeImages || 'false';
        const includeExplorerUrls = req.query.includeExplorerUrls || 'false';
        const waitForSync = req.query.waitForSync || 'false';
        
        // Octav.fi API expects 'addresses' parameter as a comma-separated string for GET requests
        // Boolean parameters should be sent as strings
        const portfolioData = await makeOctavRequest('/v1/portfolio', {
            addresses: addresses, // Keep as string (can be comma-separated for multiple)
            includeNFTs: includeNFTs,
            includeImages: includeImages,
            includeExplorerUrls: includeExplorerUrls,
            waitForSync: waitForSync
        }, 'GET');
        
        res.json(portfolioData);
    } catch (error) {
        console.error('Portfolio proxy error:', error.message);
        res.status(500).json({ 
            error: 'Failed to fetch portfolio data',
            details: error.response?.data || error.message
        });
    }
});

// Wallet endpoint - uses /v1/wallet for transaction history
app.get('/api/wallet', async (req, res) => {
    try {
        const { addresses } = req.query;
        
        if (!addresses) {
            return res.status(400).json({ error: 'addresses parameter is required' });
        }
        
        // Octav.fi API expects 'addresses' parameter as a comma-separated string
        const walletData = await makeOctavRequest('/v1/wallet', {
            addresses: addresses // Keep as string (can be comma-separated for multiple)
        }, 'GET');
        
        res.json(walletData);
    } catch (error) {
        console.error('Wallet proxy error:', error.message);
        res.status(500).json({ 
            error: 'Failed to fetch wallet data',
            details: error.response?.data || error.message
        });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        timestamp: new Date().toISOString(),
        octav_api_key: OCTAV_API_KEY ? 'configured' : 'missing'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Proxy server running on http://localhost:${PORT}`);
});

module.exports = app;
