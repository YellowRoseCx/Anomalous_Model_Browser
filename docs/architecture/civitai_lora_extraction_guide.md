# Civitai LoRA Metadata & Trigger Word Scraping: Architectural Blueprint & Extraction Guide

This document provides a comprehensive architectural map and step-by-step extraction guide for porting the Civitai LoRA metadata scanning, SHA-256 hash lookup, trigger word extraction, and header fingerprinting system from **Anomalous Model Browser** into another project or standalone microservice.

---

## 1. System Map & Component Overview

The LoRA metadata scanning capability consists of 5 main files in this repository:

| File Path | Component Name | Role & Functionality |
| :--- | :--- | :--- |
| `scraper.py` | **Scraper Engine & CLI** | Core standalone runner that calculates file SHA-256 hashes, queries Civitai endpoints, parses trigger words (`trainedWords`), fetches previews, and handles `.info` sidecar persistence and local fallback. |
| `model_identity.py` | **File Identity Evidence** | Validates SHA-256 digests and pairs them with file size and modification time (`mtime_ns`) to bind metadata to physical files reliably. |
| `api/scanner.py` | **Async API Controller** | Background process orchestration, status polling (`/anomalous/scan_status`), lock management (`.scan_in_progress`), and sub-process execution. |
| `api/metadata.py` | **Sidecar Parser & Reader** | Reads saved `.info` / `.civitai.info` files, extracts `trainedWords`, `baseModel`, `civitai_url`, and resolves proper file entries based on byte size matching. |
| `api/model_catalog.py` | **Catalog Integrator** | Combines directory discovery with `metadata.get_metadata()` to serve model information to the frontend UI. |

---

## 2. End-to-End Execution & Data Flow

```text
[Local .safetensors File]
       │
       ▼
1. Calculate SHA-256 Digest (4MB Chunks)
       │
       ▼
2. HTTP GET https://civitai.com/api/v1/model-versions/by-hash/{SHA256}
       ├──────► [Success (200 OK)] ────────┐
       │                                   ▼
       └──────► [Fail / 404 / Offline] ──► 3. Header Tensor Fingerprinting Fallback
                                           (Safetensors JSON header inspection)
                                                   │
                                                   ▼
                                           4. Extract Data Payload:
                                              - Model Name & Version Name
                                              - Base Model (e.g. SDXL, SD 1.5, Flux.1 D)
                                              - trainedWords (Trigger Words list)
                                              - Primary Preview Media URL
                                                   │
                                                   ▼
                                           5. HTTP GET https://civitai.com/api/v1/models/{modelId}
                                              (Optional: Fetch full HTML description)
                                                   │
                                                   ▼
                                           6. Write Sidecar (<filename>.info JSON)
                                              Write Cover (<filename>.preview.png / .jpg)
```

---

## 3. Detailed Technical Blueprint

### A. SHA-256 Digest Calculation (`scraper.py`)
Civitai matches model files strictly using full-file SHA-256 hashes:

```python
import hashlib

def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096 * 1024), b""): # 4MB chunks
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
```

---

### B. Civitai API Endpoint Specs & Payload Mapping

#### 1. By-Hash Lookup
- **URL**: `GET https://civitai.com/api/v1/model-versions/by-hash/{hash}`
- **Headers**:
  - `User-Agent: Mozilla/5.0 ...`
  - `Authorization: Bearer <CIVITAI_API_KEY>` (Optional, needed for NSFW/restricted models)

**Response Structure & Data Mapping**:
```json
{
  "id": 123456,                // Version ID
  "modelId": 98765,            // Main Model ID
  "name": "v1.0",              // Version Name
  "baseModel": "SD 1.5",       // Base Model Architecture
  "trainedWords": [            // <--- TRIGGER WORDS
    "anime style",
    "masterpiece",
    "1girl"
  ],
  "model": {
    "name": "My Custom LoRA",  // Model Name
    "type": "LORA"             // Model Type (LORA / Checkpoint)
  },
  "images": [
    {
      "url": "https://image.civitai.com/xG123/.../cover.jpeg",
      "nsfwLevel": 1
    }
  ],
  "files": [
    {
      "name": "my_lora.safetensors",
      "sizeKB": 36864.0,
      "hashes": {
        "SHA256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      }
    }
  ]
}
```

#### 2. Full Model Description Fetch (Optional)
- **URL**: `GET https://civitai.com/api/v1/models/{modelId}`
- **Field Extracted**: `response["description"]` (HTML string describing usage and details).

---

### C. Safetensors Header Fingerprinting (Offline Fallback)
When Civitai lookup fails (404 or no internet connection), the base model architecture can be inferred by reading the 8-byte uint64 header size and inspecting tensor keys:

```python
import struct
import json

def infer_base_model_from_header(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) < 8: return 'Unknown'
            header_size = struct.unpack('<Q', header_size_bytes)[0]
            if header_size > 100 * 1024 * 1024: return 'Unknown'

            header_json = json.loads(f.read(header_size).decode('utf-8'))

            # 1. Metadata check
            metadata = header_json.get('__metadata__', {})
            arch = metadata.get('modelspec.architecture', '')
            if 'stable-diffusion-xl' in arch.lower(): return 'SDXL'
            if 'flux' in arch.lower(): return 'Flux.1 D'
            if 'sd3' in arch.lower(): return 'SD3'

            # 2. Tensor Key Fingerprinting
            keys_str = " ".join(list(header_json.keys())[:500])
            if 'double_blocks.0.img_attn' in keys_str or 'img_in.weight' in keys_str: return 'Flux.1 D'
            if 'joint_blocks.0.x_block' in keys_str: return 'SD3'
            if 'conditioner.embedders.1.model' in keys_str: return 'SDXL'
            if 'cond_stage_model.transformer.text_model' in keys_str: return 'SD 1.5'

            return 'Unknown'
    except Exception:
        return 'Unknown'
```

---

### D. Standard `.info` Sidecar JSON Spec

To store metadata locally alongside `my_lora.safetensors`, write `my_lora.info`:

```json
{
  "id": 123456,
  "modelId": 98765,
  "name": "v1.0",
  "baseModel": "SD 1.5",
  "trainedWords": [
    "trigger_word_1",
    "trigger_word_2"
  ],
  "description": "<p>Model description HTML</p>",
  "model": {
    "name": "My Custom LoRA Name",
    "type": "LORA"
  },
  "files": [
    {
      "hashes": {
        "SHA256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
      }
    }
  ],
  "anomalous_file_identity": {
    "algorithm": "sha256",
    "scope": "file",
    "value": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "size": 37748736,
    "mtime_ns": 1700000000000000000,
    "source": "computed"
  }
}
```

---

## 4. Standalone Extraction Blueprint

To port this logic into another project, you only need to extract a single self-contained Python module (e.g. `civitai_lora_scanner.py`):

```python
import os
import json
import hashlib
import urllib.request
import urllib.error
from typing import Optional, Dict, List

class CivitaiLoraScanner:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def calculate_sha256(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096 * 1024), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def fetch_version_by_hash(self, file_hash: str) -> Optional[Dict]:
        url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
        headers = {"User-Agent": "Mozilla/5.0"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def scan_lora(self, file_path: str) -> Dict:
        file_hash = self.calculate_sha256(file_path)
        data = self.fetch_version_by_hash(file_hash)

        if not data:
            return {
                "file_path": file_path,
                "sha256": file_hash,
                "found_on_civitai": False,
                "trainedWords": [],
                "baseModel": "Unknown"
            }

        return {
            "file_path": file_path,
            "sha256": file_hash,
            "found_on_civitai": True,
            "model_name": data.get("model", {}).get("name", ""),
            "version_name": data.get("name", ""),
            "baseModel": data.get("baseModel", ""),
            "trainedWords": data.get("trainedWords", []),
            "civitai_model_id": data.get("modelId"),
            "civitai_version_id": data.get("id"),
            "cover_image_url": data["images"][0]["url"] if data.get("images") else None
        }

# Usage example:
# scanner = CivitaiLoraScanner(api_key="YOUR_KEY")
# result = scanner.scan_lora("path/to/my_lora.safetensors")
# print("Trigger Words:", result["trainedWords"])
```
