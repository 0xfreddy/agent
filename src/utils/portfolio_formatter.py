#!/usr/bin/env python3
"""
Portfolio Data Formatter
Formats and displays portfolio data in a hierarchical, readable format
"""

from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import json

console = Console()

class PortfolioFormatter:
    """Formats portfolio data for display"""
    
    @staticmethod
    def format_portfolio_data(portfolio_data: Dict[str, Any], wallet_address: str = None) -> None:
        """
        Format and display portfolio data in a readable way
        
        Args:
            portfolio_data: Portfolio data dictionary
            wallet_address: Wallet address being analyzed
        """
        if not portfolio_data:
            console.print("[red]No portfolio data found[/red]")
            return
        
        # Display wallet overview
        PortfolioFormatter._display_wallet_overview(portfolio_data, wallet_address)
        
        # Display portfolio metrics
        PortfolioFormatter._display_portfolio_metrics(portfolio_data)
        
        # Display balances by protocol and chain
        PortfolioFormatter._display_balances_by_protocol(portfolio_data)
        
        # Display top holdings
        PortfolioFormatter._display_top_holdings(portfolio_data)
    
    @staticmethod
    def _display_wallet_overview(portfolio_data: Dict[str, Any], wallet_address: str = None):
        """Display wallet overview information"""
        address = wallet_address or portfolio_data.get('address', 'N/A')
        
        overview_panel = Panel(
            f"[bold cyan]Wallet Portfolio Analysis[/bold cyan]\n"
            f"[dim]Address: {address}[/dim]\n"
            f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            style="bold blue"
        )
        console.print(overview_panel)
        console.print()
    
    @staticmethod
    def _display_portfolio_metrics(portfolio_data: Dict[str, Any]):
        """Display comprehensive portfolio metrics"""
        metrics_table = Table(title="Portfolio Overview", show_header=True)
        metrics_table.add_column("Metric", style="cyan", width=20)
        metrics_table.add_column("Value", style="white", width=25)
        metrics_table.add_column("Description", style="dim", width=40)
        
        # Core metrics
        total_value = portfolio_data.get('total_value_usd', 0)
        networth = portfolio_data.get('networth', total_value)
        cash_balance = portfolio_data.get('cash_balance', 0)
        open_pnl = portfolio_data.get('open_pnl', 0)
        closed_pnl = portfolio_data.get('closed_pnl', 0)
        daily_income = portfolio_data.get('daily_income', 0)
        daily_expense = portfolio_data.get('daily_expense', 0)
        fees = portfolio_data.get('fees_fiat', 0)
        transaction_count = portfolio_data.get('transaction_count', 0)
        
        metrics_table.add_row(
            "Total Value",
            f"${total_value:,.2f}",
            "Total portfolio value in USD"
        )
        metrics_table.add_row(
            "Net Worth",
            f"${networth:,.2f}",
            "Net worth including PnL"
        )
        metrics_table.add_row(
            "Cash Balance",
            f"${cash_balance:,.2f}",
            "Available cash/stables"
        )
        metrics_table.add_row(
            "Open PnL",
            f"${open_pnl:+,.2f}",
            "Unrealized profit/loss"
        )
        metrics_table.add_row(
            "Closed PnL",
            f"${closed_pnl:+,.2f}",
            "Realized profit/loss"
        )
        metrics_table.add_row(
            "Daily Income",
            f"${daily_income:+,.2f}",
            "Daily income from positions"
        )
        metrics_table.add_row(
            "Daily Expense",
            f"${daily_expense:+,.2f}",
            "Daily expenses/fees"
        )
        metrics_table.add_row(
            "Total Fees",
            f"${fees:,.2f}",
            "Cumulative fees paid"
        )
        metrics_table.add_row(
            "Transactions",
            f"{transaction_count:,}",
            "Total transaction count"
        )
        
        console.print(metrics_table)
        console.print()
    
    @staticmethod
    def _display_balances_by_protocol(portfolio_data: Dict[str, Any]):
        """Display balances organized by protocol and chain"""
        balances = portfolio_data.get('balances', {})
        
        if not balances:
            console.print("[yellow]No balance data available[/yellow]")
            return
        
        # Group balances by protocol
        protocol_groups = {}
        for asset_key, asset_data in balances.items():
            if asset_key.startswith('_'):  # Skip special keys
                continue
            
            protocol = asset_data.get('protocol', 'Unknown')
            chain = asset_data.get('chain', 'Unknown')
            
            if protocol not in protocol_groups:
                protocol_groups[protocol] = {}
            
            if chain not in protocol_groups[protocol]:
                protocol_groups[protocol][chain] = []
            
            protocol_groups[protocol][chain].append(asset_data)
        
        if not protocol_groups:
            console.print("[yellow]No protocol data available[/yellow]")
            return
        
        # Display protocol breakdown
        protocol_table = Table(title="Assets by Protocols", show_header=True)
        protocol_table.add_column("Protocol", style="cyan", width=20)
        protocol_table.add_column("Chain", style="blue", width=15)
        protocol_table.add_column("Assets", style="white", width=30)
        protocol_table.add_column("Value", style="green", width=15)
        
        for protocol_name, chains in protocol_groups.items():
            protocol_total = 0
            first_row = True
            
            for chain_name, assets in chains.items():
                chain_total = sum(asset.get('value_usd', 0) for asset in assets)
                protocol_total += chain_total
                
                # Format asset list
                asset_list = []
                for asset in assets[:3]:  # Show top 3 assets
                    symbol = asset.get('symbol', 'Unknown')
                    value = asset.get('value_usd', 0)
                    asset_list.append(f"{symbol}: ${value:,.0f}")
                
                if len(assets) > 3:
                    asset_list.append(f"... and {len(assets) - 3} more")
                
                asset_text = "\n".join(asset_list)
                
                protocol_table.add_row(
                    protocol_name if first_row else "",
                    chain_name,
                    asset_text,
                    f"${chain_total:,.0f}"
                )
                first_row = False
            
            # Add protocol total row
            if len(chains) > 1:
                protocol_table.add_row(
                    f"[bold]{protocol_name} Total[/bold]",
                    "",
                    "",
                    f"[bold]${protocol_total:,.0f}[/bold]"
                )
        
        console.print(protocol_table)
        console.print()
    
    @staticmethod
    def _display_top_holdings(portfolio_data: Dict[str, Any]):
        """Display top holdings in a table format"""
        balances = portfolio_data.get('balances', {})
        
        if not balances:
            return
        
        # Filter out special keys and sort by value
        asset_balances = {
            k: v for k, v in balances.items() 
            if not k.startswith('_') and v.get('value_usd', 0) > 0
        }
        
        if not asset_balances:
            return
        
        sorted_balances = sorted(
            asset_balances.items(),
            key=lambda x: x[1].get('value_usd', 0),
            reverse=True
        )[:10]  # Top 10 holdings
        
        holdings_table = Table(title="Top Holdings", show_header=True)
        holdings_table.add_column("Rank", style="cyan", width=5)
        holdings_table.add_column("Token", style="white", width=15)
        holdings_table.add_column("Balance", style="white", width=15)
        holdings_table.add_column("Price", style="yellow", width=12)
        holdings_table.add_column("Value (USD)", style="green", width=15)
        holdings_table.add_column("Allocation", style="blue", width=12)
        holdings_table.add_column("Protocol", style="dim", width=15)
        
        total_value = sum(b.get('value_usd', 0) for b in asset_balances.values())
        
        for rank, (asset_key, asset_data) in enumerate(sorted_balances, 1):
            symbol = asset_data.get('symbol', 'Unknown')
            balance = asset_data.get('balance', 0)
            price = asset_data.get('price_usd', 0)
            value = asset_data.get('value_usd', 0)
            allocation = (value / total_value * 100) if total_value > 0 else 0
            protocol = asset_data.get('protocol', 'Unknown')
            
            holdings_table.add_row(
                str(rank),
                symbol,
                f"{balance:.4f}",
                f"${price:,.2f}",
                f"${value:,.2f}",
                f"{allocation:.1f}%",
                protocol
            )
        
        console.print(holdings_table)
        console.print()
    
    @staticmethod
    def format_for_export(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format portfolio data for export (JSON/CSV)
        
        Args:
            portfolio_data: Raw portfolio data
            
        Returns:
            Formatted data suitable for export
        """
        formatted_data = {
            'wallet_info': {
                'address': portfolio_data.get('address', 'N/A'),
                'analysis_timestamp': datetime.now().isoformat(),
                'total_value_usd': portfolio_data.get('total_value_usd', 0),
                'transaction_count': portfolio_data.get('transaction_count', 0)
            },
            'portfolio_metrics': {
                'networth': portfolio_data.get('networth', 0),
                'cash_balance': portfolio_data.get('cash_balance', 0),
                'open_pnl': portfolio_data.get('open_pnl', 0),
                'closed_pnl': portfolio_data.get('closed_pnl', 0),
                'daily_income': portfolio_data.get('daily_income', 0),
                'daily_expense': portfolio_data.get('daily_expense', 0),
                'fees_fiat': portfolio_data.get('fees_fiat', 0),
                'last_updated': portfolio_data.get('last_updated', 'N/A')
            },
            'holdings': []
        }
        
        # Format holdings
        balances = portfolio_data.get('balances', {})
        for asset_key, asset_data in balances.items():
            if not asset_key.startswith('_'):
                formatted_data['holdings'].append({
                    'symbol': asset_data.get('symbol', 'Unknown'),
                    'name': asset_data.get('name', asset_data.get('symbol', 'Unknown')),
                    'address': asset_data.get('address', ''),
                    'balance': asset_data.get('balance', 0),
                    'price_usd': asset_data.get('price_usd', 0),
                    'value_usd': asset_data.get('value_usd', 0),
                    'allocation_percentage': asset_data.get('allocation_percentage', 0),
                    'protocol': asset_data.get('protocol', 'Unknown'),
                    'chain': asset_data.get('chain', 'Unknown'),
                    'position': asset_data.get('position', 'Unknown')
                })
        
        return formatted_data
    
    @staticmethod
    def format_protocol_risk_analysis(risk_analysis: Dict[str, Any]) -> None:
        """Format and display DefiLlama protocol risk analysis"""
        if not risk_analysis:
            console.print("[yellow]No protocol risk analysis available[/yellow]")
            return
        
        console.print(f"\n[bold red]🔴 PROTOCOL RISK ANALYSIS[/bold red]")
        console.print("=" * 80)
        
        # Display overall wallet protocol risk
        wallet_risk = risk_analysis.get('wallet_protocol_risk', 0)
        risk_level = risk_analysis.get('risk_level', 'unknown')
        
        risk_color = "red" if risk_level == "high" else "yellow" if risk_level == "medium" else "green"
        
        overall_risk_panel = Panel(
            f"[bold]Overall Protocol Risk: {wallet_risk:.1%}[/bold]\n"
            f"[bold]Risk Level: {risk_level.upper()}[/bold]",
            style=risk_color
        )
        console.print(overall_risk_panel)
        console.print()
        
        # Display high-risk protocols
        high_risk_protocols = risk_analysis.get('high_risk_protocols', [])
        if high_risk_protocols:
            high_risk_table = Table(title="🚨 High-Risk Protocols", show_header=True, header_style="bold red")
            high_risk_table.add_column("Protocol", style="red")
            high_risk_table.add_column("Risk Level", style="red")
            high_risk_table.add_column("Recommendation", style="yellow")
            
            for protocol in high_risk_protocols:
                protocol_data = risk_analysis.get('protocol_risks', {}).get(protocol, {})
                risk_level = protocol_data.get('risk_level', 'high')
                recommendations = protocol_data.get('recommendations', ['Consider exiting position'])
                
                high_risk_table.add_row(
                    protocol.upper(),
                    risk_level.upper(),
                    recommendations[0] if recommendations else "Consider exiting position"
                )
            
            console.print(high_risk_table)
            console.print()
        
        # Display detailed protocol risks
        protocol_risks = risk_analysis.get('protocol_risks', {})
        if protocol_risks:
            protocol_table = Table(title="Protocol Risk Breakdown", show_header=True, header_style="bold magenta")
            protocol_table.add_column("Protocol", style="cyan")
            protocol_table.add_column("Risk Score", style="white")
            protocol_table.add_column("Risk Level", style="white")
            protocol_table.add_column("TVL Risk", style="yellow")
            protocol_table.add_column("Hack Risk", style="red")
            protocol_table.add_column("Yield Risk", style="blue")
            
            for protocol, data in protocol_risks.items():
                risk_factors = data.get('risk_factors', {})
                
                protocol_table.add_row(
                    protocol.upper(),
                    f"{data.get('total_risk_score', 0):.1%}",
                    data.get('risk_level', 'unknown').upper(),
                    f"{risk_factors.get('tvl_risk', 0):.1%}",
                    f"{risk_factors.get('hack_risk', 0):.1%}",
                    f"{risk_factors.get('yield_risk', 0):.1%}"
                )
            
            console.print(protocol_table)
            console.print()
        
        # Display recommendations
        recommendations = risk_analysis.get('recommendations', [])
        if recommendations:
            rec_table = Table(title="Risk Recommendations", show_header=True, header_style="bold green")
            rec_table.add_column("Recommendation", style="green")
            
            for rec in recommendations:
                rec_table.add_row(rec)
            
            console.print(rec_table)
            console.print()
    
    @staticmethod
    def format_transactions(transactions: List[Dict], max_display: int = 10) -> None:
        """Format and display transaction data"""
        if not transactions:
            console.print("[yellow]No transactions found[/yellow]")
            return
        
        console.print(f"\n[bold blue]📊 TRANSACTIONS ({len(transactions)} total)[/bold blue]")
        console.print("=" * 80)
        
        # Display transaction summary
        tx_types = {}
        chains = {}
        protocols = {}
        total_value = 0
        total_fees = 0
        
        for tx in transactions:
            tx_type = tx.get('type', 'UNKNOWN')
            tx_types[tx_type] = tx_types.get(tx_type, 0) + 1
            
            chain_name = tx.get('chain', {}).get('name', 'Unknown')
            chains[chain_name] = chains.get(chain_name, 0) + 1
            
            protocol_name = tx.get('protocol', {}).get('name', 'Unknown')
            protocols[protocol_name] = protocols.get(protocol_name, 0) + 1
            
            total_value += tx.get('value_fiat', 0)
            total_fees += tx.get('fees_fiat', 0)
        
        # Create summary table
        summary_table = Table(title="Transaction Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Transactions", str(len(transactions)))
        summary_table.add_row("Total Value", f"${total_value:,.2f}")
        summary_table.add_row("Total Fees", f"${total_fees:,.2f}")
        summary_table.add_row("Avg Value per TX", f"${total_value/len(transactions):,.2f}" if transactions else "$0.00")
        
        console.print(summary_table)
        console.print()
        
        # Display transaction types breakdown
        if tx_types:
            type_table = Table(title="Transaction Types", show_header=True, header_style="bold magenta")
            type_table.add_column("Type", style="cyan")
            type_table.add_column("Count", style="green")
            type_table.add_column("Percentage", style="yellow")
            
            for tx_type, count in sorted(tx_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(transactions)) * 100
                type_table.add_row(tx_type, str(count), f"{percentage:.1f}%")
            
            console.print(type_table)
            console.print()
        
        # Display recent transactions
        recent_txs = transactions[:max_display]
        if recent_txs:
            tx_table = Table(title=f"Recent Transactions (showing {len(recent_txs)})", show_header=True, header_style="bold magenta")
            tx_table.add_column("Date", style="cyan")
            tx_table.add_column("Type", style="green")
            tx_table.add_column("Chain", style="yellow")
            tx_table.add_column("Protocol", style="blue")
            tx_table.add_column("Value", style="red")
            tx_table.add_column("Fees", style="magenta")
            
            for tx in recent_txs:
                timestamp = tx.get('timestamp', datetime.now())
                if isinstance(timestamp, datetime):
                    date_str = timestamp.strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = str(timestamp)
                
                tx_type = tx.get('type', 'UNKNOWN')
                chain = tx.get('chain', {}).get('name', 'Unknown')
                protocol = tx.get('protocol', {}).get('name', 'Unknown')
                value = f"${tx.get('value_fiat', 0):,.2f}"
                fees = f"${tx.get('fees_fiat', 0):,.2f}"
                
                tx_table.add_row(date_str, tx_type, chain, protocol, value, fees)
            
            console.print(tx_table)
            console.print()
        
        # Display asset flow summary
        asset_inflows = {}
        asset_outflows = {}
        
        for tx in transactions:
            # Track assets received
            for asset in tx.get('assets_in', []):
                symbol = asset.get('symbol', 'Unknown')
                value = asset.get('value', 0)
                asset_inflows[symbol] = asset_inflows.get(symbol, 0) + value
            
            # Track assets sent
            for asset in tx.get('assets_out', []):
                symbol = asset.get('symbol', 'Unknown')
                value = asset.get('value', 0)
                asset_outflows[symbol] = asset_outflows.get(symbol, 0) + value
        
        if asset_inflows or asset_outflows:
            asset_table = Table(title="Asset Flow Summary", show_header=True, header_style="bold magenta")
            asset_table.add_column("Asset", style="cyan")
            asset_table.add_column("Inflow", style="green")
            asset_table.add_column("Outflow", style="red")
            asset_table.add_column("Net Flow", style="yellow")
            
            all_assets = set(asset_inflows.keys()) | set(asset_outflows.keys())
            
            for asset in sorted(all_assets):
                inflow = asset_inflows.get(asset, 0)
                outflow = asset_outflows.get(asset, 0)
                net_flow = inflow - outflow
                
                inflow_str = f"${inflow:,.2f}" if inflow > 0 else "-"
                outflow_str = f"${outflow:,.2f}" if outflow > 0 else "-"
                net_flow_str = f"${net_flow:,.2f}"
                
                asset_table.add_row(asset, inflow_str, outflow_str, net_flow_str)
            
            console.print(asset_table)
            console.print()
