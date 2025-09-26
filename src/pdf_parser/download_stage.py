#!/usr/bin/env python3
"""
SEC Filings Download Stage for DVC Pipeline
============================================

This script downloads SEC filings and converts them to PDF format using sec-api.
It also downloads corresponding XBRL data for cross-validation purposes.

"""

import os
import sys
import argparse
import json
import time
import zipfile
import io
import re
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import requests
import pandas as pd

# Try to import dotenv (optional)
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Try to import sec-api (optional but recommended)
try:
    from sec_api import QueryApi, PdfGeneratorApi, XbrlApi
    SEC_API_AVAILABLE = True
except ImportError:
    SEC_API_AVAILABLE = False
    print(" sec-api not available. Install with: pip install sec-api")


class SECDownloader:
    """Handles downloading SEC filings and XBRL data."""
    
    def __init__(self, api_key: Optional[str] = None, user_agent: Optional[str] = None):
        """Initialize the SEC downloader with API credentials."""
        self.api_key = api_key
        self.user_agent = user_agent or self._get_default_user_agent()
        
        # Initialize APIs if available
        if SEC_API_AVAILABLE and api_key:
            self.query_api = QueryApi(api_key=api_key)
            self.pdf_generator = PdfGeneratorApi(api_key=api_key)
            self.xbrl_api = XbrlApi(api_key=api_key)
        else:
            self.query_api = None
            self.pdf_generator = None
            self.xbrl_api = None
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Request configuration for direct SEC access
        self.request_kwargs = {
            "headers": {"User-Agent": self._ascii_http_header(self.user_agent)},
            "timeout": 30
        }
    
    def _get_default_user_agent(self) -> str:
        """Get default User-Agent string for SEC requests."""
        return os.environ.get(
            "SEC_USER_AGENT",
            "PDF Parser pdf-parser@example.com - academic research"
        )
    
    def _ascii_http_header(self, s: str) -> str:
        """Convert Unicode to safe ASCII for HTTP headers."""
        if not s:
            return ""
        s = s.replace("\u2013", "-").replace("\u2014", "-")
        s = unicodedata.normalize("NFKD", s)
        return "".join(ch if ord(ch) < 256 else "-" for ch in s)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the downloader."""
        logger = logging.getLogger("sec_downloader")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def search_filings(self, ticker: str, form_type: str, num_filings: int = 3) -> List[Dict]:
        """Search for SEC filings using sec-api or fallback methods."""
        self.logger.info(f"🔍 Searching for {ticker} {form_type} filings...")
        
        if not self.query_api:
            raise ValueError("SEC API not available. Please install sec-api and provide API key.")
        
        # Calculate date range (last 3 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3*365)
        
        query = {
            "query": f"ticker:{ticker} AND formType:\"{form_type}\" AND filedAt:[{start_date.strftime('%Y-%m-%d')} TO {end_date.strftime('%Y-%m-%d')}]",
            "from": "0",
            "size": str(num_filings),
            "sort": [{"filedAt": {"order": "desc"}}]
        }
        
        try:
            response = self.query_api.get_filings(query)
            filings = response.get("filings", [])
            
            self.logger.info(f"📋 Found {len(filings)} {form_type} filings for {ticker}")
            
            if not filings:
                self.logger.warning("❌ No filings found. Check ticker symbol or date range.")
                return []
            
            # Log filing details
            for i, filing in enumerate(filings):
                filed_date = filing["filedAt"][:10]
                accession_no = filing["accessionNo"]
                self.logger.info(f"  {i+1}. Filed: {filed_date} | Accession: {accession_no}")
            
            return filings
            
        except Exception as e:
            self.logger.error(f"❌ Error searching filings: {e}")
            raise
    
    def download_pdfs(self, filings: List[Dict], ticker: str, output_dir: str) -> Dict[str, any]:
        """Download and convert SEC filings to PDF format."""
        pdf_dir = Path(output_dir) / ticker / "10-K" / "PDFs"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🔄 Converting {len(filings)} SEC filings to PDF format...")
        self.logger.info(f"📂 Saving PDFs to: {pdf_dir}")
        
        if not self.pdf_generator:
            raise ValueError("PDF Generator API not available.")
        
        results = {
            "total_filings": len(filings),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "pdf_files": [],
            "total_size_mb": 0.0
        }
        
        for i, filing in enumerate(filings):
            try:
                # Extract filing information
                filed_date = filing["filedAt"][:10].replace("-", "")
                accession_no = filing["accessionNo"].replace("-", "")
                form_type = filing["formType"]
                
                # Create filename
                pdf_filename = f"{ticker}_{form_type}_{filed_date}_{accession_no}.pdf"
                pdf_path = pdf_dir / pdf_filename
                
                # Skip if file already exists
                if pdf_path.exists():
                    file_size = pdf_path.stat().st_size / (1024 * 1024)
                    self.logger.info(f"  ⏩ Skipping {pdf_filename} (already exists, {file_size:.1f} MB)")
                    results["successful_downloads"] += 1
                    results["pdf_files"].append(str(pdf_path))
                    results["total_size_mb"] += file_size
                    continue
                
                self.logger.info(f"  🔄 Processing filing {i+1}/{len(filings)}: {filed_date}")
                
                # Get the filing URL
                filing_url = filing["linkToFilingDetails"]
                
                # Generate PDF using sec-api
                self.logger.info(f"    📄 Generating PDF from: {filing_url}")
                
                # Add delay to respect rate limits
                if i > 0:
                    time.sleep(1)
                
                pdf_content = self.pdf_generator.get_pdf(filing_url)
                
                # Save PDF to file
                with open(pdf_path, 'wb') as pdf_file:
                    pdf_file.write(pdf_content)
                
                # Check file size
                file_size = pdf_path.stat().st_size / (1024 * 1024)
                
                if file_size > 0.1:  # At least 100KB
                    self.logger.info(f"    ✅ Successfully saved: {pdf_filename} ({file_size:.1f} MB)")
                    results["successful_downloads"] += 1
                    results["pdf_files"].append(str(pdf_path))
                    results["total_size_mb"] += file_size
                else:
                    self.logger.warning(f"    ⚠️  Small file size for {pdf_filename} ({file_size:.1f} MB)")
                    results["successful_downloads"] += 1  # Still count as success
                    results["pdf_files"].append(str(pdf_path))
                    results["total_size_mb"] += file_size
                
            except Exception as e:
                self.logger.error(f"     Error processing filing {i+1}: {e}")
                results["failed_downloads"] += 1
                continue
        
        # Generate summary
        success_rate = (results["successful_downloads"] / results["total_filings"]) * 100
        self.logger.info(f"\n📊 PDF Download Summary:")
        self.logger.info(f"  Total filings: {results['total_filings']}")
        self.logger.info(f"  Successful downloads: {results['successful_downloads']}")
        self.logger.info(f"  Failed downloads: {results['failed_downloads']}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        self.logger.info(f"  Total size: {results['total_size_mb']:.1f} MB")
        
        return results
    
    def download_xbrl_data(self, filings: List[Dict], ticker: str, output_dir: str) -> Dict[str, any]:
        """Download XBRL data for the filings."""
        xbrl_dir = Path(output_dir) / ticker / "10-K" / "XBRL"
        xbrl_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📦 Downloading XBRL artifacts from {len(filings)} filings...")
        self.logger.info(f"📂 Output directory: {xbrl_dir}")
        
        results = {
            "total_filings": len(filings),
            "successful_downloads": 0,
            "failed_downloads": 0,
            "xbrl_directories": []
        }
        
        for i, filing in enumerate(filings):
            try:
                filed_date = filing["filedAt"][:10].replace("-", "")
                accession_no = filing["accessionNo"]
                form_type = filing.get("formType", "UNKNOWN")
                cik = filing.get("cik", "0000789019")  # Default to MSFT CIK
                
                filing_dir = xbrl_dir / f"{ticker}_{form_type}_{filed_date}_xbrl"
                
                # Skip if already exists and has files
                if filing_dir.exists() and any(filing_dir.glob("*")):
                    self.logger.info(f"  ⏩ [{i+1}/{len(filings)}] {form_type} {filed_date} already present")
                    results["successful_downloads"] += 1
                    results["xbrl_directories"].append(str(filing_dir))
                    continue
                
                self.logger.info(f"  📥 [{i+1}/{len(filings)}] {form_type} {filed_date} → {filing_dir}")
                
                # Download XBRL files
                xbrl_result = self._download_xbrl_zip_or_files(cik, accession_no, str(filing_dir))
                
                if xbrl_result["saved"]:
                    self.logger.info(f"    Saved {len(xbrl_result['saved'])} file(s) via {xbrl_result['mode']}")
                    results["successful_downloads"] += 1
                    results["xbrl_directories"].append(str(filing_dir))
                else:
                    self.logger.warning(f"    ⚠️  No XBRL files saved for {form_type} {filed_date}")
                    results["failed_downloads"] += 1
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"     Error downloading XBRL for filing {i+1}: {e}")
                results["failed_downloads"] += 1
                continue
        
        # Summary
        success_rate = (results["successful_downloads"] / results["total_filings"]) * 100
        self.logger.info(f"\n📊 XBRL Download Summary:")
        self.logger.info(f"  Total filings: {results['total_filings']}")
        self.logger.info(f"  Successful downloads: {results['successful_downloads']}")
        self.logger.info(f"  Failed downloads: {results['failed_downloads']}")
        self.logger.info(f"  Success rate: {success_rate:.1f}%")
        
        return results
    
    def _download_xbrl_zip_or_files(self, cik: str, accession_no: str, dest_dir: str) -> Dict[str, any]:
        """Download XBRL files either as ZIP or individual files."""
        os.makedirs(dest_dir, exist_ok=True)
        saved = []
        
        # Helper functions
        def _clean_cik(cik_str: str) -> str:
            return str(int(str(cik_str).strip()))
        
        def _clean_accession(acc: str) -> str:
            return str(acc).replace("-", "").strip()
        
        def submission_base_url(cik_str: str, accession_no: str) -> str:
            return f"https://www.sec.gov/Archives/edgar/data/{_clean_cik(cik_str)}/{_clean_accession(accession_no)}"
        
        def xbrl_zip_url(cik_str: str, accession_no: str) -> str:
            return f"{submission_base_url(cik_str, accession_no)}/{accession_no}-xbrl.zip"
        
        # Try ZIP first
        zip_url = xbrl_zip_url(cik, accession_no)
        try:
            response = requests.get(zip_url, **self.request_kwargs)
            if response.status_code == 200 and response.content:
                zip_path = os.path.join(dest_dir, f"{_clean_accession(accession_no)}-xbrl.zip")
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                
                # Extract contents
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(dest_dir)
                    saved.extend([str(Path(dest_dir) / name) for name in zf.namelist()])
                
                saved.append(zip_path)
                return {"mode": "zip", "saved": saved}
        
        except Exception as e:
            self.logger.debug(f"ZIP download failed: {e}")
        
        # Fallback to individual files
        index_url = f"{submission_base_url(cik, accession_no)}/index.json"
        try:
            response = requests.get(index_url, **self.request_kwargs)
            if response.status_code != 200:
                raise RuntimeError(f"index.json not accessible (status {response.status_code})")
            
            data = response.json()
            items = (data.get("directory") or {}).get("item") or []
            
            # XBRL file patterns
            xbrl_patterns = [
                r".*?-ins\.xml$", r".*?-pre\.xml$", r".*?-cal\.xml$",
                r".*?-def\.xml$", r".*?-lab\.xml$", r".*?\.xsd$",
                r".*?\.(?:htm|html)$"
            ]
            xbrl_regexes = [re.compile(p, re.IGNORECASE) for p in xbrl_patterns]
            
            def looks_like_xbrl(filename: str) -> bool:
                return any(rx.search(filename) for rx in xbrl_regexes)
            
            base = submission_base_url(cik, accession_no)
            for item in items:
                name = item.get("name")
                if not name or not looks_like_xbrl(name):
                    continue
                
                file_url = f"{base}/{name}"
                out_path = os.path.join(dest_dir, name)
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                
                file_response = requests.get(file_url, **self.request_kwargs)
                if file_response.status_code == 200 and file_response.content:
                    with open(out_path, "wb") as f:
                        f.write(file_response.content)
                    saved.append(out_path)
                
                time.sleep(0.2)  # Rate limiting
            
            return {"mode": "files", "saved": saved}
        
        except Exception as e:
            self.logger.error(f"Failed to download XBRL files: {e}")
            return {"mode": "failed", "saved": []}
    
    def generate_metadata(self, pdf_results: Dict, xbrl_results: Dict, output_dir: str, ticker: str) -> str:
        """Generate metadata file for the download stage."""
        metadata = {
            "download_info": {
                "timestamp": datetime.now().isoformat(),
                "ticker": ticker,
                "downloader_version": "1.0.0",
                "user_agent": self.user_agent
            },
            "pdf_download": pdf_results,
            "xbrl_download": xbrl_results,
            "summary": {
                "total_filings_processed": pdf_results["total_filings"],
                "pdf_success_rate": (pdf_results["successful_downloads"] / pdf_results["total_filings"]) * 100,
                "xbrl_success_rate": (xbrl_results["successful_downloads"] / xbrl_results["total_filings"]) * 100,
                "total_pdf_size_mb": pdf_results["total_size_mb"]
            }
        }
        
        metadata_path = Path(output_dir) / "download_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"📊 Metadata saved to: {metadata_path}")
        return str(metadata_path)


def main():
    """Main function for the download stage."""
    parser = argparse.ArgumentParser(
        description="Download SEC filings and convert to PDF format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol (e.g., MSFT)")
    parser.add_argument("--form-type", default="10-K", help="SEC form type (default: 10-K)")
    parser.add_argument("--num-filings", type=int, default=3, help="Number of filings to download (default: 3)")
    parser.add_argument("--output-dir", default="data/raw", required=True, help="Output directory for downloaded files")
    parser.add_argument("--api-key", help="SEC API key (or set SEC_API_KEY env var)")
    parser.add_argument("--user-agent", help="User-Agent string for SEC requests")
    parser.add_argument("--skip-xbrl", action="store_true", help="Skip XBRL download")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Load environment variables
    if DOTENV_AVAILABLE:
        load_dotenv()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger("sec_downloader").setLevel(logging.DEBUG)
    
    # Get API key
    api_key = args.api_key or os.getenv("SEC_API_KEY")
    if not api_key:
        print("SEC API key required. Set SEC_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    if not SEC_API_AVAILABLE:
        print("sec-api package not available. Install with: pip install sec-api")
        sys.exit(1)
    
    try:
        # Initialize downloader
        downloader = SECDownloader(api_key=api_key, user_agent=args.user_agent)
        
        # Search for filings
        filings = downloader.search_filings(args.ticker, args.form_type, args.num_filings)
        
        if not filings:
            print(" No filings found. Exiting.")
            sys.exit(1)
        
        # Download PDFs
        pdf_results = downloader.download_pdfs(filings, args.ticker, args.output_dir)
        
        # Download XBRL data (unless skipped)
        if args.skip_xbrl:
            xbrl_results = {"total_filings": 0, "successful_downloads": 0, "failed_downloads": 0, "xbrl_directories": []}
            print("⏩ Skipping XBRL download as requested")
        else:
            xbrl_results = downloader.download_xbrl_data(filings, args.ticker, args.output_dir)
        
        # Generate metadata
        metadata_path = downloader.generate_metadata(pdf_results, xbrl_results, args.output_dir, args.ticker)
        
        # Final summary
        print(f"\n Download Stage Complete!")
        print(f"  PDFs: {pdf_results['successful_downloads']}/{pdf_results['total_filings']} successful")
        if not args.skip_xbrl:
            print(f"  XBRL: {xbrl_results['successful_downloads']}/{xbrl_results['total_filings']} successful")
        print(f"  Metadata: {metadata_path}")
        print(f"  Ready for next pipeline stage! ")
        
    except Exception as e:
        print(f" Download stage failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
