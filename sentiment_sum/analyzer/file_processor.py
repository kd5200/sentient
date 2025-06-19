import csv
import os
import chardet
from typing import List, Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FileProcessor:
    """Enhanced file processor for CSV and text files."""
    
    def __init__(self):
        self.supported_formats = ['.txt', '.csv', '.tsv']
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence})")
                
                # Fallback to UTF-8 if confidence is low
                if confidence < 0.7:
                    logger.warning(f"Low confidence in encoding detection, using UTF-8")
                    return 'utf-8'
                
                return encoding or 'utf-8'
        except Exception as e:
            logger.error(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    def process_text_file(self, file_path: str) -> List[str]:
        """Process text file with improved handling."""
        try:
            encoding = self.detect_encoding(file_path)
            
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                lines = f.readlines()
            
            # Process lines
            comments = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    # Remove common comment markers
                    if line.startswith(('#', '//', '--', '/*')):
                        continue
                    comments.append(line)
            
            logger.info(f"Processed {len(comments)} comments from text file")
            return comments
            
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise
    
    def process_csv_file(self, file_path: str, comment_column: Optional[int] = None, 
                        has_header: bool = True, delimiter: str = ',') -> List[str]:
        """Process CSV file with flexible column selection."""
        try:
            encoding = self.detect_encoding(file_path)
            
            # Try using pandas first for better CSV handling
            try:
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                
                # If no specific column is specified, try to find the comment column
                if comment_column is None:
                    comment_column = self._find_comment_column(df)
                
                if comment_column is not None and comment_column < len(df.columns):
                    comments = df.iloc[:, comment_column].dropna().astype(str).tolist()
                else:
                    # Fallback to first column
                    comments = df.iloc[:, 0].dropna().astype(str).tolist()
                
                logger.info(f"Processed {len(comments)} comments from CSV file using pandas")
                return comments
                
            except Exception as pandas_error:
                logger.warning(f"Pandas processing failed, falling back to csv module: {pandas_error}")
                return self._process_csv_fallback(file_path, encoding, comment_column, has_header, delimiter)
                
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise
    
    def _find_comment_column(self, df: pd.DataFrame) -> Optional[int]:
        """Automatically find the column most likely to contain comments."""
        # Look for columns with text-like content
        text_columns = []
        
        for i, col in enumerate(df.columns):
            # Check if column contains mostly text
            sample_values = df[col].dropna().astype(str).head(10)
            avg_length = sample_values.str.len().mean()
            
            # If average length > 10 characters, likely contains comments
            if avg_length > 10:
                text_columns.append((i, avg_length))
        
        if text_columns:
            # Return the column with the longest average text
            return max(text_columns, key=lambda x: x[1])[0]
        
        return 0  # Default to first column
    
    def _process_csv_fallback(self, file_path: str, encoding: str, 
                            comment_column: Optional[int], has_header: bool, 
                            delimiter: str) -> List[str]:
        """Fallback CSV processing using the csv module."""
        comments = []
        
        with open(file_path, 'r', encoding=encoding, errors='replace') as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            # Skip header if specified
            if has_header:
                next(reader, None)
            
            for row_num, row in enumerate(reader, 1):
                if not row:  # Skip empty rows
                    continue
                
                # Use specified column or first column
                col_index = comment_column if comment_column is not None else 0
                
                if col_index < len(row):
                    comment = row[col_index].strip()
                    if comment:  # Skip empty comments
                        comments.append(comment)
                else:
                    logger.warning(f"Row {row_num}: Column {col_index} not found")
        
        return comments
    
    def process_file(self, file_path: str, **kwargs) -> List[str]:
        """Main method to process any supported file type."""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        if file_ext == '.txt':
            return self.process_text_file(file_path)
        elif file_ext in ['.csv', '.tsv']:
            delimiter = '\t' if file_ext == '.tsv' else kwargs.get('delimiter', ',')
            comment_column = kwargs.get('comment_column')
            has_header = kwargs.get('has_header', True)
            return self.process_csv_file(file_path, comment_column, has_header, delimiter)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """Validate file and return metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Check file size (limit to 10MB)
        if file_size > 10 * 1024 * 1024:
            raise ValueError("File too large (max 10MB)")
        
        # Check if file is empty
        if file_size == 0:
            raise ValueError("File is empty")
        
        return {
            'file_path': file_path,
            'file_size': file_size,
            'file_extension': file_ext,
            'is_supported': file_ext in self.supported_formats
        } 