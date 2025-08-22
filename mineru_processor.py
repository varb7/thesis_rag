#!/usr/bin/env python3
"""
Enhanced MinerU PDF to Markdown Processor
Robust batch processing with error handling, progress tracking, and manifest generation
"""

import os
import sys
import glob
import json
import csv
import hashlib
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# Import MinerU modules
try:
    from mineru.cli.common import do_parse, read_fn
    from mineru.utils.enum_class import MakeMode
    from mineru.data.data_reader_writer import FileBasedDataWriter
except ImportError as e:
    print(f"❌ MinerU not installed. Install with: pip install mineru")
    print(f"Error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mineru_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MinerUProcessor:
    """Enhanced MinerU processor with robust error handling and batch processing"""
    
    def __init__(self, output_root: str = "./mineru_out"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(exist_ok=True)
        
        # Default settings
        self.default_settings = {
            "language": "en",
            "backend": "pipeline", 
            "parse_method": "auto",
            "formula_enable": True,
            "table_enable": True,
            "include_images": True,
            "start_page": 0,
            "end_page": None,
            "markdown_mode": "mm_markdown"
        }
        
        logger.info(f"Initialized MinerU processor with output root: {self.output_root}")
    
    def setup_mineru_environment(self):
        """Setup MinerU environment and validate installation"""
        try:
            # Test basic MinerU functionality
            logger.info("Setting up MinerU environment...")
            
            # Validate MinerU installation
            if not hasattr(MakeMode, 'MM_MD'):
                raise ImportError("MinerU MakeMode not properly imported")
            
            logger.info("MinerU environment setup complete!")
            logger.info("Models will be downloaded automatically on first use.")
            return True
            
        except Exception as e:
            logger.error(f"MinerU environment setup failed: {e}")
            return False
    
    def _pdf_id_from_path(self, pdf_path: str) -> str:
        """Generate stable ID for manifest (filename stem + short hash of abs path)"""
        p = Path(pdf_path)
        h = hashlib.md5(str(p.resolve()).encode("utf-8")).hexdigest()[:8]
        return f"{p.stem}-{h}"
    
    def _is_already_parsed(self, parse_dir: str, pdf_stem: str) -> bool:
        """Check if PDF is already processed (markdown exists)"""
        md_path = Path(parse_dir) / f"{pdf_stem}.md"
        content_list_path = Path(parse_dir) / f"{pdf_stem}_content_list.json"
        
        # Check both markdown and content list exist
        return md_path.exists() and content_list_path.exists()
    
    def _validate_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """Validate PDF file before processing"""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            return False, f"File not found: {pdf_path}"
        
        if not pdf_path.suffix.lower() == '.pdf':
            return False, f"Not a PDF file: {pdf_path}"
        
        # Check file size (skip very small or very large files)
        file_size = pdf_path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            return False, f"File too small: {file_size} bytes"
        if file_size > 500 * 1024 * 1024:  # More than 500MB
            return False, f"File too large: {file_size / (1024*1024):.1f} MB"
        
        return True, "Valid PDF"
    
    def extract_pdf_to_markdown(
        self,
        pdf_path: str,
        output_dir: str,
        **kwargs
    ) -> Dict:
        """
        Extract PDF content to markdown with images using MinerU.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory for results
            **kwargs: Processing options (language, backend, etc.)
        
        Returns:
            Dictionary containing paths to generated files
        """
        try:
            # Validate PDF
            is_valid, message = self._validate_pdf(pdf_path)
            if not is_valid:
                return {"error": message}
            
            # Merge with default settings
            settings = {**self.default_settings, **kwargs}
            
            # Prepare file lists for MinerU
            pdf_file_names = [Path(pdf_path).stem]
            pdf_bytes_list = [read_fn(pdf_path)]
            lang_list = [settings["language"]]
            
            # Set markdown mode
            if settings["markdown_mode"] == "mm_markdown":
                md_mode = MakeMode.MM_MD
            elif settings["markdown_mode"] == "nlp_markdown":
                md_mode = MakeMode.NLP_MD
            else:
                raise ValueError(f"Invalid markdown_mode: {settings['markdown_mode']}")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Processing PDF: {pdf_path}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"Settings: {settings}")
            
            # Process the PDF with MinerU
            do_parse(
                output_dir=output_dir,
                pdf_file_names=pdf_file_names,
                pdf_bytes_list=pdf_bytes_list,
                p_lang_list=lang_list,
                backend=settings["backend"],
                parse_method=settings["parse_method"],
                formula_enable=settings["formula_enable"],
                table_enable=settings["table_enable"],
                f_dump_md=True,
                f_dump_middle_json=True,
                f_dump_model_output=True,
                f_dump_content_list=True,
                f_dump_orig_pdf=True,
                f_draw_layout_bbox=True,
                f_draw_span_bbox=True,
                f_make_md_mode=md_mode,
                start_page_id=settings["start_page"],
                end_page_id=settings["end_page"]
            )
            
            # Get the output paths
            pdf_name = pdf_file_names[0]
            parse_dir = os.path.join(output_dir, pdf_name, settings["parse_method"])
            
            result_files = {
                "markdown": os.path.join(parse_dir, f"{pdf_name}.md"),
                "middle_json": os.path.join(parse_dir, f"{pdf_name}_middle.json"),
                "model_output": os.path.join(parse_dir, f"{pdf_name}_model.json"),
                "content_list": os.path.join(parse_dir, f"{pdf_name}_content_list.json"),
                "original_pdf": os.path.join(parse_dir, f"{pdf_name}_origin.pdf"),
                "layout_bbox": os.path.join(parse_dir, f"{pdf_name}_layout.pdf"),
                "span_bbox": os.path.join(parse_dir, f"{pdf_name}_span.pdf"),
                "images_dir": os.path.join(parse_dir, "images")
            }
            
            # Check which files were actually created
            existing_files = {}
            for key, path in result_files.items():
                if os.path.exists(path):
                    existing_files[key] = path
                else:
                    existing_files[key] = None
            
            return existing_files
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {"error": str(e)}
    
    def process_one_pdf(
        self,
        pdf_path: str,
        language: str = "en",
        backend: str = "pipeline",
        parse_method: str = "auto",
        force: bool = False,
        **kwargs
    ) -> Dict:
        """
        Process a single PDF with enhanced error handling and manifest generation.
        
        Args:
            pdf_path: Path to PDF file
            language: Language code
            backend: MinerU backend
            parse_method: Parsing method
            force: Force reprocessing even if already done
            **kwargs: Additional processing options
        
        Returns:
            Manifest record for the processed PDF
        """
        try:
            pdf_path = str(Path(pdf_path).resolve())
            pdf_id = self._pdf_id_from_path(pdf_path)
            out_dir = os.path.join(self.output_root, pdf_id)
            
            # Create output directory
            os.makedirs(out_dir, exist_ok=True)
            
            # Where MinerU will write: out_dir/<stem>/<parse_method>/
            pdf_stem = Path(pdf_path).stem
            parse_dir = os.path.join(out_dir, pdf_stem, parse_method)
            
            # Check if already processed
            if (not force) and self._is_already_parsed(parse_dir, pdf_stem):
                logger.info(f"Skipping {pdf_path} - already processed")
                
                # Reconstruct result_files from existing files
                result_files = {
                    "markdown": os.path.join(parse_dir, f"{pdf_stem}.md"),
                    "middle_json": os.path.join(parse_dir, f"{pdf_stem}_middle.json"),
                    "model_output": os.path.join(parse_dir, f"{pdf_stem}_model.json"),
                    "content_list": os.path.join(parse_dir, f"{pdf_stem}_content_list.json"),
                    "original_pdf": os.path.join(parse_dir, f"{pdf_stem}_origin.pdf"),
                    "layout_bbox": os.path.join(parse_dir, f"{pdf_stem}_layout.pdf"),
                    "span_bbox": os.path.join(parse_dir, f"{pdf_stem}_span.pdf"),
                    "images_dir": os.path.join(parse_dir, "images"),
                }
            else:
                # Process the PDF
                result_files = self.extract_pdf_to_markdown(
                    pdf_path=pdf_path,
                    output_dir=out_dir,
                    language=language,
                    backend=backend,
                    parse_method=parse_method,
                    **kwargs
                )
            
            # Check for errors
            if "error" in result_files:
                status = "error"
                error_msg = result_files["error"]
            elif result_files.get("markdown") and result_files.get("content_list"):
                status = "ok"
                error_msg = None
            else:
                status = "partial"
                error_msg = "Some files missing"
            
            # Build manifest row
            manifest_row = {
                "pdf_id": pdf_id,
                "pdf_path": pdf_path,
                "output_root": out_dir,
                "parse_dir": parse_dir,
                "language": language,
                "backend": backend,
                "parse_method": parse_method,
                "markdown": result_files.get("markdown"),
                "middle_json": result_files.get("middle_json"),
                "model_output": result_files.get("model_output"),
                "content_list": result_files.get("content_list"),
                "original_pdf": result_files.get("original_pdf"),
                "layout_bbox": result_files.get("layout_bbox"),
                "span_bbox": result_files.get("span_bbox"),
                "images_dir": result_files.get("images_dir"),
                "status": status,
                "error": error_msg,
                "ts": int(time.time())
            }
            
            return manifest_row
            
        except Exception as e:
            logger.error(f"Unexpected error processing {pdf_path}: {e}")
            return {
                "pdf_id": self._pdf_id_from_path(pdf_path),
                "pdf_path": pdf_path,
                "status": "error",
                "error": str(e),
                "ts": int(time.time())
            }
    
    def find_pdfs(self, input_dir: str, recursive: bool = True) -> List[str]:
        """Find all PDF files in the input directory"""
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdfs = [str(p) for p in Path(input_dir).glob(pattern)]
        logger.info(f"Found {len(pdfs)} PDF files in {input_dir}")
        return sorted(pdfs)
    
    def batch_process_dir(
        self,
        input_dir: str,
        language: str = "en",
        backend: str = "pipeline",
        parse_method: str = "auto",
        recursive: bool = True,
        max_workers: int = 1,
        force: bool = False,
        **kwargs
    ) -> List[Dict]:
        """
        Process all PDFs in a folder with enhanced error handling.
        
        Args:
            input_dir: Directory containing PDFs
            language: Language code for processing
            backend: MinerU backend
            parse_method: Parsing method
            recursive: Search subdirectories
            max_workers: Number of parallel workers
            force: Force reprocessing
            **kwargs: Additional processing options
        
        Returns:
            List of manifest records for all processed PDFs
        """
        # Find PDFs
        pdfs = self.find_pdfs(input_dir, recursive=recursive)
        if not pdfs:
            logger.warning(f"No PDFs found under: {input_dir}")
            return []
        
        logger.info(f"Starting batch processing of {len(pdfs)} PDFs")
        logger.info(f"Output root: {self.output_root}")
        logger.info(f"Max workers: {max_workers}")
        
        results: List[Dict] = []
        
        # Process PDFs with progress tracking
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for pdf in pdfs:
                futures.append(ex.submit(
                    self.process_one_pdf,
                    pdf, language, backend, parse_method, force, **kwargs
                ))
            
            # Process results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDFs"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Future execution failed: {e}")
                    results.append({
                        "status": "error",
                        "error": f"Future execution failed: {e}",
                        "ts": int(time.time())
                    })
        
        # Write manifest files
        self._write_manifests(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _write_manifests(self, results: List[Dict]):
        """Write manifest files in multiple formats"""
        try:
            # JSONL manifest
            manifest_jsonl = self.output_root / "manifest.jsonl"
            with open(manifest_jsonl, "w", encoding="utf-8") as f:
                for row in results:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
            # CSV manifest
            manifest_csv = self.output_root / "manifest.csv"
            if results:
                fieldnames = sorted(set().union(*[row.keys() for row in results if isinstance(row, dict)]))
                with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in results:
                        writer.writerow(row)
            
            logger.info(f"Manifests written:")
            logger.info(f"  - {manifest_jsonl}")
            logger.info(f"  - {manifest_csv}")
            
        except Exception as e:
            logger.error(f"Error writing manifests: {e}")
    
    def _print_summary(self, results: List[Dict]):
        """Print processing summary"""
        total = len(results)
        ok = sum(1 for r in results if r.get("status") == "ok")
        partial = sum(1 for r in results if r.get("status") == "partial")
        errors = sum(1 for r in results if r.get("status") == "error")
        
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total PDFs: {total}")
        print(f"Successful: {ok}")
        print(f"Partial: {partial}")
        print(f"Errors: {errors}")
        print(f"Success Rate: {(ok/total)*100:.1f}%" if total > 0 else "N/A")
        
        if errors > 0:
            print(f"\nError details:")
            for result in results:
                if result.get("status") == "error":
                    print(f"  - {Path(result.get('pdf_path', 'Unknown')).name}: {result.get('error', 'Unknown error')}")
    
    def validate_outputs(self, pdf_id: str) -> Dict:
        """Validate the outputs for a specific PDF"""
        pdf_dir = self.output_root / pdf_id
        
        if not pdf_dir.exists():
            return {"valid": False, "error": "PDF directory not found"}
        
        # Find the actual parse directory
        parse_dirs = list(pdf_dir.rglob("*_content_list.json"))
        if not parse_dirs:
            return {"valid": False, "error": "No content list found"}
        
        content_list_path = parse_dirs[0]
        parse_dir = content_list_path.parent
        
        # Check required files
        required_files = [
            "*.md",  # Markdown
            "*_content_list.json",  # Content list
            "*_middle.json",  # Middle JSON
        ]
        
        missing_files = []
        for pattern in required_files:
            if not list(parse_dir.glob(pattern)):
                missing_files.append(pattern)
        
        if missing_files:
            return {
                "valid": False,
                "error": f"Missing required files: {missing_files}",
                "parse_dir": str(parse_dir)
            }
        
        return {
            "valid": True,
            "parse_dir": str(parse_dir),
            "files": [f.name for f in parse_dir.iterdir() if f.is_file()]
        }

# Convenience functions for backward compatibility
def setup_mineru_environment():
    """Setup MinerU environment (backward compatibility)"""
    processor = MinerUProcessor()
    return processor.setup_mineru_environment()

def batch_process_dir(
    input_dir: str,
    output_root: str = "./mineru_out",
    language: str = "en",
    backend: str = "pipeline",
    parse_method: str = "auto",
    recursive: bool = True,
    max_workers: int = 1,
    force: bool = False,
    **kwargs
) -> List[Dict]:
    """Batch process PDFs (backward compatibility)"""
    processor = MinerUProcessor(output_root)
    return processor.batch_process_dir(
        input_dir, language, backend, parse_method, 
        recursive, max_workers, force, **kwargs
    )

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = MinerUProcessor("./mineru_out")
    
    # Setup environment
    if not processor.setup_mineru_environment():
        print("Failed to setup MinerU environment")
        sys.exit(1)
    
    # Process PDFs
    results = processor.batch_process_dir(
        input_dir="./pdfs",  # Change to your PDF folder
        language="en",
        backend="pipeline",
        parse_method="auto",
        recursive=True,
        max_workers=1,  # Keep 1 for GPU safety
        force=False
    )
    
    print(f"\n🎉 Processing complete! Check {processor.output_root} for results.")