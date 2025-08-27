#!/usr/bin/env python3

import click
from rich.console import Console
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
import json
import csv
from datetime import datetime
import sys
import os
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config, Config
from src.models.schemas import Mood
from src.agents.recommendation_agent import RecommendationAgent
from src.utils.portfolio_formatter import PortfolioFormatter

console = Console()

class CryptoRecommendationCLI:
    def __init__(self):
        self.config = get_config()
        self.console = console
        self.agent = RecommendationAgent()
    
    def analyze_wallet(self, wallet_address: str, mood: str = 'balanced') -> Dict[str, Any]:
        self.console.print(f"\n[bold blue]🔍 Analyzing wallet: {wallet_address}[/bold blue]")
        self.console.print(f"[dim]Mood setting: {mood}[/dim]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task("[cyan]Analyzing wallet...", total=3)
            
            # Step 1: Full Analysis
            progress.update(task, description="[cyan]Running comprehensive analysis...")
            analysis_results = self.agent.analyze(wallet_address, mood)
            progress.update(task, advance=1)
            
            # Step 2: Generate Recommendation
            progress.update(task, description="[cyan]Generating recommendation...")
            recommendation = self.agent.generate_recommendation(analysis_results, mood)
            progress.update(task, advance=1)
            
            # Step 3: Compile Results
            progress.update(task, description="[cyan]Compiling results...")
            results = {
                'analysis': analysis_results,
                'recommendation': recommendation,
                'explanation': self.agent.explain_recommendation(recommendation)
            }
            progress.update(task, advance=1)
            
            return results
    
    def display_results(self, results: Dict[str, Any]):
        """Display analysis results in a formatted way"""
        analysis = results['analysis']
        recommendation = results['recommendation']
        explanation = results['explanation']
        
        wallet_data = analysis['wallet_data']
        
        # Title
        title_panel = Panel(
            f"[bold cyan]Crypto Wallet Analysis Report[/bold cyan]\n"
            f"[dim]Address: {wallet_data['address']}[/dim]\n"
            f"[dim]Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
            style="bold blue"
        )
        self.console.print(title_panel)
        
        # Use the new portfolio formatter for comprehensive display
        PortfolioFormatter.format_portfolio_data(wallet_data, wallet_data['address'])
        
        # Add risk score to the display
        risk_panel = Panel(
            f"[bold]Risk Assessment Score: {recommendation['risk_score']:.1f}/100[/bold]",
            style="red" if recommendation['risk_score'] > 70 else "yellow" if recommendation['risk_score'] > 40 else "green"
        )
        self.console.print(risk_panel)
        self.console.print()
        
        # Risk Assessment
        risk_metrics = analysis['risk_metrics']
        risk_table = Table(title="Risk Assessment", show_header=True)
        risk_table.add_column("Risk Type", style="cyan")
        risk_table.add_column("Score", style="white")
        risk_table.add_column("Level", style="white")
        
        risk_table.add_row(
            "Volatility",
            f"{risk_metrics['volatility']['volatility_percentage']:.2f}%",
            risk_metrics['volatility']['risk_level']
        )
        risk_table.add_row(
            "Concentration",
            f"{risk_metrics['concentration']['hhi_score']:.2%}",
            risk_metrics['concentration']['risk_level']
        )
        risk_table.add_row(
            "VaR (95%)",
            f"${risk_metrics['var']['var_95_dollar']:.2f}",
            risk_metrics['var']['risk_assessment']
        )
        
        self.console.print(risk_table)
        self.console.print()
        
        # Recommendation
        try:
            rec_panel = Panel(
                explanation,
                style="green"
            )
            self.console.print(rec_panel)
        except Exception as e:
            self.console.print(f"[red]Error displaying recommendation: {e}[/red]")
            # Fallback display
            self.console.print(Panel(
                f"Action: {recommendation['action']}\n"
                f"Confidence: {recommendation['confidence']:.1%}\n"
                f"Risk Score: {recommendation['risk_score']:.1f}/100",
                style="green"
            ))
        
        # Top holdings are now displayed by the PortfolioFormatter
        # Additional analysis can be added here if needed
    
    def export_results(self, results: Dict[str, Any], format: str, output_path: str):
        if format == 'json':
            # Use the new formatter for better structure
            formatted_data = PortfolioFormatter.format_for_export(results['analysis']['wallet_data'])
            formatted_data['recommendation'] = results['recommendation']
            formatted_data['explanation'] = results['explanation']
            
            with open(output_path, 'w') as f:
                json.dump(formatted_data, f, indent=2, default=str)
            self.console.print(f"[green]✅ Results exported to {output_path}[/green]")
        
        elif format == 'csv':
            # Use the new formatter for better structure
            formatted_data = PortfolioFormatter.format_for_export(results['analysis']['wallet_data'])
            flat_data = self._flatten_dict(formatted_data)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Key', 'Value'])
                for key, value in flat_data.items():
                    writer.writerow([key, value])
            
            self.console.print(f"[green]✅ Results exported to {output_path}[/green]")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

@click.group()
def cli():
    """Crypto Wallet Recommendation Agent"""
    pass

@cli.command()
@click.argument('wallet_address')
@click.option('--mood', default='balanced', type=click.Choice(['degen', 'balanced', 'saver']))
@click.option('--export', type=click.Choice(['json', 'csv']), help='Export format')
@click.option('--output', default='results.json', help='Output file path')
@click.option('--include-nfts', is_flag=True, help='Include NFTs in portfolio data')
@click.option('--include-images', is_flag=True, help='Include image URLs in portfolio data')
@click.option('--include-explorer-urls', is_flag=True, help='Include explorer links in portfolio data')
@click.option('--wait-for-sync', is_flag=True, help='Wait for fresh data (may take longer)')
def analyze(wallet_address: str, mood: str, export: Optional[str], output: str, 
           include_nfts: bool, include_images: bool, include_explorer_urls: bool, wait_for_sync: bool):
    """Analyze a wallet and generate recommendations"""
    
    cli_app = CryptoRecommendationCLI()
    
    # Update configuration with CLI parameters
    if include_nfts or include_images or include_explorer_urls or wait_for_sync:
        cli_app.config.octav_include_nfts = include_nfts
        cli_app.config.octav_include_images = include_images
        cli_app.config.octav_include_explorer_urls = include_explorer_urls
        cli_app.config.octav_wait_for_sync = wait_for_sync
    
    try:
        results = cli_app.analyze_wallet(wallet_address, mood)
        
        cli_app.display_results(results)
        
        if export:
            cli_app.export_results(results, export, output)
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)

@cli.command()
@click.argument('file_path')
@click.option('--mood', default='balanced', type=click.Choice(['degen', 'balanced', 'saver']))
def batch(file_path: str, mood: str):
    """Process multiple wallets from a file"""
    
    cli_app = CryptoRecommendationCLI()
    
    try:
        with open(file_path, 'r') as f:
            wallets = [line.strip() for line in f if line.strip()]
        
        console.print(f"[bold]Processing {len(wallets)} wallets...[/bold]\n")
        
        results = []
        for wallet in track(wallets, description="Analyzing wallets..."):
            try:
                result = cli_app.analyze_wallet(wallet, mood)
                results.append(result)
                console.print(f"[green]✓[/green] {wallet}")
            except Exception as e:
                console.print(f"[red]✗[/red] {wallet}: {e}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"batch_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ Batch processing complete. Results saved to {output_file}[/green]")
    
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        sys.exit(1)



if __name__ == "__main__":
    cli()