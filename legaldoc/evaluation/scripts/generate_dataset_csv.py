import os
import csv
import pandas as pd
from pathlib import Path

def generate_dataset_csv(dataset_root, output_csv):
    """
    Generate dataset manifest CSV from folder structure
    
    Expected folder structure (as shown in your image):
    dataset/
    ├── legal/           # All files here labeled as 'legal'
    │   ├── contracts/
    │   ├── agreements/
    │   └── policies/
    ├── non_legal/       # All files here labeled as 'not_legal'  
    │   ├── marketing/
    │   ├── content/
    │   └── academic/
    └── edge_cases/      # All files here labeled as 'not_legal' (borderline cases)
    """
    
    dataset_entries = []
    
    # Define folder mappings
    folder_mappings = {
        'legal': {
            'label': 'legal',
            'subcategories': {
                'contracts': 'contract',
                'policies': 'policy', 
                'court_docs': 'court_document',
                'agreements': 'agreement',
                'ndas': 'nda',
                'legal_docs': 'legal_document',
                'default': 'legal_document'
            }
        },
        'non_legal': {
            'label': 'not_legal',
            'subcategories': {
                'marketing': 'marketing',
                'content': 'content',
                'academic': 'academic',
                'manuals': 'manual',
                'recipes': 'content',
                'stories': 'content',
                'news': 'news_article',
                'default': 'non_legal_document'
            }
        },
        'edge_cases': {
            'label': 'not_legal',
            'subcategories': {
                'quasi_legal': 'quasi_legal',
                'scanned': 'scanned_document',
                'multilingual': 'multilingual',
                'fake_legal': 'quasi_legal',
                'borderline': 'edge_case',
                'default': 'edge_case'
            }
        }
    }
    
    # Supported file extensions (INCLUDING CSV)
    supported_extensions = {
        '.pdf', '.docx', '.doc', '.txt', '.csv', 
        '.jpg', '.jpeg', '.png', '.rtf', '.odt'
    }
    
    for main_folder, config in folder_mappings.items():
        main_folder_path = Path(dataset_root) / main_folder
        
        if not main_folder_path.exists():
            print(f"Warning: Folder {main_folder_path} does not exist")
            continue
            
        print(f"Scanning folder: {main_folder_path}")
        
        # Walk through all files in the main folder and subfolders
        for file_path in main_folder_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                # Skip hidden files and system files
                if file_path.name.startswith('.') or file_path.name.startswith('~'):
                    continue
                    
                # Determine category based on parent folder
                parent_folder = file_path.parent.name
                
                # Get category from subcategories or use default
                category = config['subcategories'].get(parent_folder, 
                          config['subcategories']['default'])
                
                # Create relative path from dataset root
                try:
                    relative_path = file_path.relative_to(Path(dataset_root).parent)
                except ValueError:
                    # Fallback if relative path calculation fails
                    relative_path = file_path.relative_to(Path(dataset_root))
                    relative_path = Path('dataset') / relative_path
                
                # Generate notes based on file and folder
                notes = generate_notes(file_path, main_folder, parent_folder)
                
                dataset_entries.append({
                    'file_path': str(relative_path).replace('\\', '/'),  # Use forward slashes
                    'label': config['label'],
                    'category': category,
                    'notes': notes
                })
                
                print(f"  ✅ Added: {relative_path}")
    
    # Sort entries by file path for consistency
    dataset_entries.sort(key=lambda x: x['file_path'])
    
    # Write to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_path', 'label', 'category', 'notes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset_entries)
    
    return dataset_entries

def generate_notes(file_path, main_folder, parent_folder):
    """Generate descriptive notes for the file"""
    filename = file_path.stem.lower()
    file_ext = file_path.suffix.lower()
    
    # Generate notes based on filename patterns and folder structure
    notes_parts = []
    
    # Add folder context
    if parent_folder != main_folder:
        notes_parts.append(f"{parent_folder.replace('_', ' ').title()} document")
    
    # Add file type context based on filename
    if 'contract' in filename or 'agreement' in filename:
        notes_parts.append("Contract document")
    elif 'nda' in filename or 'non_disclosure' in filename:
        notes_parts.append("Non-disclosure agreement")
    elif 'policy' in filename or 'terms' in filename:
        notes_parts.append("Policy document")
    elif 'invoice' in filename or 'bill' in filename:
        notes_parts.append("Invoice document")
    elif 'license' in filename:
        notes_parts.append("License agreement")
    elif 'employment' in filename:
        notes_parts.append("Employment document")
    elif 'service' in filename:
        notes_parts.append("Service agreement")
    elif 'purchase' in filename or 'sale' in filename:
        notes_parts.append("Purchase/sale document")
    elif 'recipe' in filename:
        notes_parts.append("Recipe content")
    elif 'story' in filename or 'novel' in filename:
        notes_parts.append("Story content")
    elif 'manual' in filename or 'guide' in filename:
        notes_parts.append("Manual/guide document")
    elif 'marketing' in filename or 'brochure' in filename:
        notes_parts.append("Marketing material")
    elif 'news' in filename or 'article' in filename:
        notes_parts.append("News article")
    elif 'research' in filename or 'paper' in filename:
        notes_parts.append("Research paper")
    elif 'scanned' in filename or 'scan' in filename:
        notes_parts.append("Scanned document")
    elif 'cuad' in filename:
        notes_parts.append("CUAD legal contract")
    
    # Add file extension info
    if file_ext == '.csv':
        notes_parts.append("CSV data file")
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        notes_parts.append("Image document")
    elif file_ext == '.pdf':
        notes_parts.append("PDF document")
    elif file_ext in ['.docx', '.doc']:
        notes_parts.append("Word document")
    elif file_ext == '.txt':
        notes_parts.append("Text document")
    elif file_ext == '.rtf':
        notes_parts.append("Rich text document")
    elif file_ext == '.odt':
        notes_parts.append("OpenOffice document")
    
    return "; ".join(notes_parts) if notes_parts else f"Document from {main_folder}"

def print_dataset_summary(entries):
    """Print summary statistics of the dataset"""
    if not entries:
        print("No entries to summarize")
        return
        
    df = pd.DataFrame(entries)
    
    print("\n" + "="*60)
    print("📊 DATASET SUMMARY")
    print("="*60)
    
    print(f"Total files: {len(entries)}")
    
    print(f"\n📋 Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(entries)) * 100
        print(f"  {label}: {count} files ({percentage:.1f}%)")
    
    print(f"\n📂 Category distribution:")
    category_counts = df['category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} files")
    
    print(f"\n📄 File types:")
    file_extensions = [Path(entry['file_path']).suffix.lower() for entry in entries]
    ext_counts = pd.Series(file_extensions).value_counts()
    for ext, count in ext_counts.items():
        print(f"  {ext}: {count} files")
    
    # Check balance
    legal_count = len(df[df['label'] == 'legal'])
    non_legal_count = len(df[df['label'] == 'not_legal'])
    balance_ratio = min(legal_count, non_legal_count) / max(legal_count, non_legal_count) if max(legal_count, non_legal_count) > 0 else 0
    
    print(f"\n⚖️ Dataset Balance:")
    print(f"  Legal documents: {legal_count}")
    print(f"  Non-legal documents: {non_legal_count}")
    print(f"  Balance ratio: {balance_ratio:.2f} {'✅ Well balanced' if balance_ratio > 0.7 else '⚠️ Imbalanced - consider balancing'}")

def validate_dataset_structure(dataset_root):
    """Validate the dataset folder structure"""
    dataset_path = Path(dataset_root)
    
    print(f"🔍 VALIDATING DATASET STRUCTURE")
    print(f"Root path: {dataset_path.absolute()}")
    print("-" * 50)
    
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset root folder does not exist!")
        print(f"Expected: {dataset_path.absolute()}")
        return False
    
    print(f"✅ Dataset root exists")
    
    # Check expected main folders
    expected_folders = ['legal', 'non_legal', 'edge_cases']
    found_folders = []
    total_files = 0
    
    for folder in expected_folders:
        folder_path = dataset_path / folder
        if folder_path.exists():
            files = list(folder_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_files += file_count
            print(f"✅ {folder}/ exists with {file_count} files")
            found_folders.append(folder)
            
            # Show subfolder structure
            subfolders = [f for f in folder_path.iterdir() if f.is_dir()]
            if subfolders:
                print(f"   📁 Subfolders: {[sf.name for sf in subfolders]}")
        else:
            print(f"⚠️ {folder}/ does not exist (optional)")
    
    print(f"\n📊 Total files found: {total_files}")
    
    if total_files == 0:
        print(f"\n❌ No files found in dataset!")
        print(f"Please add documents to the folder structure")
        return False
    
    print(f"✅ Dataset structure is valid")
    return True

if __name__ == "__main__":
    # Configuration
    dataset_root = "../dataset"  # Adjust path to your dataset folder
    output_csv = "../dataset_manifest.csv"
    
    print("🎯 Legal Document Dataset CSV Generator (Enhanced)")
    print("=" * 60)
    
    try:
        # Validate structure first
        if not validate_dataset_structure(dataset_root):
            exit(1)
        
        print(f"\n📝 Generating CSV manifest...")
        entries = generate_dataset_csv(dataset_root, output_csv)
        
        if entries:
            print(f"\n✅ Successfully generated {output_csv}")
            print_dataset_summary(entries)
            
            # Show first few entries as preview
            print(f"\n📋 First 5 entries preview:")
            df = pd.DataFrame(entries)
            print(df.head().to_string(index=False))
            
            # Save a backup copy with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_csv = f"../dataset_manifest_backup_{timestamp}.csv"
            df.to_csv(backup_csv, index=False)
            print(f"\n💾 Backup saved: {backup_csv}")
            
        else:
            print(f"\n❌ No valid files found!")
            print(f"Make sure your documents are in the correct folders")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
