#!/usr/bin/env python3
"""
Test script to demonstrate file processing capabilities.
This script shows how different file formats are processed.
"""

import os
import tempfile
from analyzer.file_processor import FileProcessor

def create_test_files():
    """Create test files for demonstration."""
    processor = FileProcessor()
    
    # Create test text file
    text_content = """This is a great product!
I love the quality and service.
The shipping was fast and reliable.
Not satisfied with the purchase.
Excellent customer support.
The item arrived broken.
Fantastic experience overall.
"""
    
    # Create test CSV file
    csv_content = """id,comment,rating
1,"I absolutely love this product! It exceeded all my expectations.",5
2,"Terrible experience. The item arrived broken and late.",1
3,"It was okay, nothing special but not bad either.",3
4,"Fantastic service, will definitely come back!",5
5,"I'm disappointed. The quality is not worth the price.",2
6,"Superb quality and fast shipping — highly recommended!",5
"""
    
    # Create test TSV file
    tsv_content = """id\tcomment\trating
1\tI absolutely love this product! It exceeded all my expectations.\t5
2\tTerrible experience. The item arrived broken and late.\t1
3\tIt was okay, nothing special but not bad either.\t3
4\tFantastic service, will definitely come back!\t5
5\tI'm disappointed. The quality is not worth the price.\t2
6\tSuperb quality and fast shipping — highly recommended!\t5
"""
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(text_content)
        text_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_file = f.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        f.write(tsv_content)
        tsv_file = f.name
    
    return text_file, csv_file, tsv_file

def test_file_processing():
    """Test the file processing functionality."""
    processor = FileProcessor()
    
    # Create test files
    text_file, csv_file, tsv_file = create_test_files()
    
    try:
        print("=== File Processing Test ===\n")
        
        # Test text file processing
        print("1. Processing text file:")
        comments = processor.process_file(text_file)
        print(f"   Found {len(comments)} comments:")
        for i, comment in enumerate(comments, 1):
            print(f"   {i}. {comment}")
        print()
        
        # Test CSV file processing (default: first column)
        print("2. Processing CSV file (first column):")
        comments = processor.process_file(csv_file)
        print(f"   Found {len(comments)} comments:")
        for i, comment in enumerate(comments, 1):
            print(f"   {i}. {comment}")
        print()
        
        # Test CSV file processing (second column)
        print("3. Processing CSV file (second column):")
        comments = processor.process_file(csv_file, comment_column=1)
        print(f"   Found {len(comments)} comments:")
        for i, comment in enumerate(comments, 1):
            print(f"   {i}. {comment}")
        print()
        
        # Test CSV file processing (no header)
        print("4. Processing CSV file (no header):")
        comments = processor.process_file(csv_file, has_header=False)
        print(f"   Found {len(comments)} comments:")
        for i, comment in enumerate(comments, 1):
            print(f"   {i}. {comment}")
        print()
        
        # Test TSV file processing
        print("5. Processing TSV file:")
        comments = processor.process_file(tsv_file)
        print(f"   Found {len(comments)} comments:")
        for i, comment in enumerate(comments, 1):
            print(f"   {i}. {comment}")
        print()
        
        # Test file validation
        print("6. File validation:")
        for file_path in [text_file, csv_file, tsv_file]:
            try:
                file_info = processor.validate_file(file_path)
                print(f"   {os.path.basename(file_path)}: {file_info}")
            except Exception as e:
                print(f"   {os.path.basename(file_path)}: Error - {e}")
        print()
        
        # Test encoding detection
        print("7. Encoding detection:")
        for file_path in [text_file, csv_file, tsv_file]:
            encoding = processor.detect_encoding(file_path)
            print(f"   {os.path.basename(file_path)}: {encoding}")
        
    finally:
        # Clean up temporary files
        for file_path in [text_file, csv_file, tsv_file]:
            try:
                os.unlink(file_path)
            except:
                pass

if __name__ == "__main__":
    test_file_processing() 