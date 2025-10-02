"""
Document Security Service

Comprehensive protection against malicious document uploads including:
- File type validation and content verification
- Malware scanning and virus detection
- Content filtering and moderation
- Rate limiting and abuse prevention
- Quarantine system for suspicious files
"""

import os
import re
import hashlib
import mimetypes
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import magic
import tempfile
from datetime import datetime, timedelta
import asyncio
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)

class DocumentSecurityService:
    """Comprehensive document security and validation service"""

    # Allowed file extensions (whitelist approach)
    ALLOWED_EXTENSIONS = {
        # Documents
        '.pdf', '.doc', '.docx', '.txt', '.md', '.rtf',
        # Spreadsheets
        '.xls', '.xlsx', '.csv',
        # Presentations
        '.ppt', '.pptx',
        # Images (limited)
        '.jpg', '.jpeg', '.png', '.gif', '.bmp',
        # Plain text
        '.json', '.xml', '.html', '.htm'
    }

    # Allowed MIME types (secondary validation)
    ALLOWED_MIME_TYPES = {
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain',
        'text/markdown',
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-powerpoint',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'image/jpeg',
        'image/png',
        'image/gif',
        'image/bmp',
        'application/json',
        'text/xml',
        'text/html'
    }

    # Dangerous file signatures (magic bytes)
    DANGEROUS_SIGNATURES = {
        b'\x4d\x5a': 'PE executable',
        b'\x7f\x45\x4c\x46': 'ELF executable',
        b'\xca\xfe\xba\xbe': 'Java class file',
        b'\xfe\xed\xfa\xce': 'Mach-O executable',
        b'\xfe\xed\xfa\xcf': 'Mach-O executable',
        b'\x50\x4b\x03\x04': 'ZIP (check for embedded executables)',
        b'\x1f\x8b\x08': 'GZIP (check for embedded content)',
        b'#!/bin/': 'Shell script',
        b'#!/usr/bin/': 'Shell script',
        b'<script': 'HTML script tag',
        b'javascript:': 'JavaScript URI'
    }

    # Suspicious keywords in content
    SUSPICIOUS_KEYWORDS = [
        # Malware/virus indicators
        'trojan', 'backdoor', 'keylogger', 'rootkit', 'botnet',
        # Script injection
        '<script>', 'javascript:', 'eval(', 'document.cookie',
        # SQL injection
        'union select', 'drop table', '; delete from', 'xp_cmdshell',
        # Command injection
        '$(', '`', '&&', '||', ';rm ', ';del ',
        # XSS indicators
        'alert(', 'confirm(', 'prompt(', 'window.location',
        # Social engineering
        'download now', 'urgent action required', 'verify your account',
        # Crypto miners
        'crypto', 'mining', 'hashrate', 'blockchain'
    ]

    # File size limits by type (bytes)
    SIZE_LIMITS = {
        'text/plain': 10 * 1024 * 1024,      # 10MB for text
        'application/pdf': 50 * 1024 * 1024,  # 50MB for PDFs
        'application/msword': 25 * 1024 * 1024,  # 25MB for Word docs
        'image/jpeg': 20 * 1024 * 1024,      # 20MB for images
        'image/png': 20 * 1024 * 1024,       # 20MB for images
        'default': 30 * 1024 * 1024          # 30MB default
    }

    def __init__(self):
        self.quarantine_dir = Path(tempfile.gettempdir()) / "document_quarantine"
        self.quarantine_dir.mkdir(exist_ok=True)

        # Rate limiting storage (in production, use Redis)
        self.upload_counts = {}
        self.blocked_hashes = set()

    async def validate_upload(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        user_id: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Comprehensive upload validation

        Returns:
            (is_safe: bool, details: dict)
        """
        validation_result = {
            'safe': True,
            'issues': [],
            'warnings': [],
            'file_info': {},
            'security_score': 100,
            'quarantined': False
        }

        try:
            # 1. Basic file validation
            basic_check = await self._validate_basic_file_properties(
                file_content, filename, content_type
            )
            validation_result.update(basic_check)

            if not basic_check['safe']:
                return False, validation_result

            # 2. Rate limiting check
            rate_check = await self._check_rate_limiting(user_id)
            if not rate_check['safe']:
                validation_result.update(rate_check)
                return False, validation_result

            # 3. File signature and magic byte validation
            signature_check = await self._validate_file_signature(file_content, filename)
            validation_result['security_score'] -= signature_check.get('risk_score', 0)
            validation_result['warnings'].extend(signature_check.get('warnings', []))

            if not signature_check['safe']:
                validation_result['safe'] = False
                validation_result['issues'].extend(signature_check['issues'])

            # 4. Content scanning for suspicious patterns
            content_check = await self._scan_content_for_threats(file_content, filename)
            validation_result['security_score'] -= content_check.get('risk_score', 0)
            validation_result['warnings'].extend(content_check.get('warnings', []))

            if not content_check['safe']:
                validation_result['safe'] = False
                validation_result['issues'].extend(content_check['issues'])

            # 5. Hash-based blacklist check
            hash_check = await self._check_file_hash_reputation(file_content)
            if not hash_check['safe']:
                validation_result['safe'] = False
                validation_result['issues'].extend(hash_check['issues'])

            # 6. Advanced threat detection
            advanced_check = await self._advanced_threat_detection(file_content, filename)
            validation_result['security_score'] -= advanced_check.get('risk_score', 0)

            # 7. Final security score assessment
            if validation_result['security_score'] < 70:
                validation_result['safe'] = False
                validation_result['issues'].append(f"Low security score: {validation_result['security_score']}")
            elif validation_result['security_score'] < 85:
                validation_result['warnings'].append(f"Medium risk file (score: {validation_result['security_score']})")

            # 8. Quarantine if suspicious
            if not validation_result['safe'] or validation_result['security_score'] < 60:
                await self._quarantine_file(file_content, filename, validation_result)
                validation_result['quarantined'] = True

            return validation_result['safe'], validation_result

        except Exception as e:
            logger.error(f"Error during upload validation: {e}")
            return False, {
                'safe': False,
                'issues': ['Validation system error'],
                'error': str(e)
            }

    async def _validate_basic_file_properties(
        self,
        file_content: bytes,
        filename: str,
        content_type: str
    ) -> Dict[str, Any]:
        """Basic file property validation"""
        result = {'safe': True, 'issues': [], 'warnings': [], 'file_info': {}}

        # File size validation
        file_size = len(file_content)
        max_size = self.SIZE_LIMITS.get(content_type, self.SIZE_LIMITS['default'])

        if file_size > max_size:
            result['safe'] = False
            result['issues'].append(f"File too large: {file_size} bytes (max: {max_size})")

        if file_size == 0:
            result['safe'] = False
            result['issues'].append("Empty file")

        # Filename validation
        if not filename or len(filename) > 255:
            result['safe'] = False
            result['issues'].append("Invalid filename")

        # Extension validation
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.ALLOWED_EXTENSIONS:
            result['safe'] = False
            result['issues'].append(f"Disallowed file extension: {file_ext}")

        # MIME type validation
        if content_type not in self.ALLOWED_MIME_TYPES:
            result['warnings'].append(f"Unusual MIME type: {content_type}")

        # Detect MIME type from content
        try:
            detected_mime = magic.from_buffer(file_content[:2048], mime=True)
            if detected_mime != content_type:
                result['warnings'].append(f"MIME mismatch: declared={content_type}, detected={detected_mime}")
        except:
            result['warnings'].append("Could not detect MIME type from content")

        # Suspicious filename patterns
        suspicious_patterns = [
            r'\.exe\.', r'\.scr\.', r'\.com\.', r'\.bat\.', r'\.cmd\.',
            r'double\s+click', r'click\s+here', r'urgent', r'important'
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, filename, re.IGNORECASE):
                result['warnings'].append(f"Suspicious filename pattern: {pattern}")

        result['file_info'] = {
            'size': file_size,
            'extension': file_ext,
            'mime_type': content_type,
            'filename': filename
        }

        return result

    async def _check_rate_limiting(self, user_id: int) -> Dict[str, Any]:
        """Check upload rate limiting per user"""
        current_time = datetime.utcnow()
        hour_ago = current_time - timedelta(hours=1)

        # Clean old entries
        if user_id in self.upload_counts:
            self.upload_counts[user_id] = [
                timestamp for timestamp in self.upload_counts[user_id]
                if timestamp > hour_ago
            ]
        else:
            self.upload_counts[user_id] = []

        # Check limits
        uploads_last_hour = len(self.upload_counts[user_id])
        max_uploads_per_hour = getattr(settings, 'MAX_UPLOADS_PER_HOUR', 50)

        if uploads_last_hour >= max_uploads_per_hour:
            return {
                'safe': False,
                'issues': [f"Rate limit exceeded: {uploads_last_hour} uploads in last hour (max: {max_uploads_per_hour})"]
            }

        # Record this upload
        self.upload_counts[user_id].append(current_time)

        return {'safe': True, 'uploads_count': uploads_last_hour}

    async def _validate_file_signature(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Validate file signatures and magic bytes"""
        result = {'safe': True, 'issues': [], 'warnings': [], 'risk_score': 0}

        if len(file_content) < 16:
            result['warnings'].append("File too small for signature analysis")
            return result

        # Check for dangerous signatures
        header = file_content[:512]  # Check first 512 bytes

        for signature, description in self.DANGEROUS_SIGNATURES.items():
            if signature in header:
                if 'executable' in description.lower():
                    result['safe'] = False
                    result['issues'].append(f"Executable file detected: {description}")
                elif 'script' in description.lower():
                    result['safe'] = False
                    result['issues'].append(f"Script file detected: {description}")
                else:
                    result['warnings'].append(f"Suspicious signature: {description}")
                    result['risk_score'] += 20

        # Check for embedded content in archives
        if header.startswith(b'\x50\x4b\x03\x04'):  # ZIP signature
            result['warnings'].append("ZIP archive detected - contents not scanned")
            result['risk_score'] += 10

        # PDF-specific checks
        if header.startswith(b'%PDF'):
            # Check for JavaScript in PDF
            if b'/JS' in file_content or b'/JavaScript' in file_content:
                result['safe'] = False
                result['issues'].append("PDF contains JavaScript")
                result['risk_score'] += 30

            # Check for form fields
            if b'/AcroForm' in file_content:
                result['warnings'].append("PDF contains forms")
                result['risk_score'] += 10

        return result

    async def _scan_content_for_threats(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Scan file content for suspicious patterns"""
        result = {'safe': True, 'issues': [], 'warnings': [], 'risk_score': 0}

        try:
            # Convert to text for pattern matching
            try:
                content_str = file_content.decode('utf-8', errors='ignore')
            except:
                content_str = str(file_content)

            content_lower = content_str.lower()

            # Check for suspicious keywords
            found_keywords = []
            for keyword in self.SUSPICIOUS_KEYWORDS:
                if keyword in content_lower:
                    found_keywords.append(keyword)

            if found_keywords:
                # Check for dangerous keywords that should always block
                dangerous_keywords = ['trojan', 'malware', 'keylogger', 'virus', 'backdoor', 'rootkit', 'spyware']
                dangerous_found = [kw for kw in found_keywords if kw in dangerous_keywords]

                if dangerous_found or len(found_keywords) > 2:
                    result['safe'] = False
                    result['issues'].append(f"Dangerous keywords detected: {found_keywords[:5]}")
                    result['risk_score'] += 30
                elif len(found_keywords) > 1:
                    result['safe'] = False
                    result['issues'].append(f"Multiple suspicious keywords found: {found_keywords}")
                    result['risk_score'] += 20
                else:
                    result['warnings'].append(f"Suspicious keywords: {found_keywords}")
                    result['risk_score'] += len(found_keywords) * 10

            # Check for URLs and external references
            url_patterns = [
                r'https?://[^\s<>"\']+',
                r'ftp://[^\s<>"\']+',
                r'file://[^\s<>"\']+',
                r'javascript:[^\s<>"\']+',
                r'data:[^\s<>"\']+',
            ]

            urls_found = []
            for pattern in url_patterns:
                matches = re.findall(pattern, content_str, re.IGNORECASE)
                urls_found.extend(matches)

            if urls_found:
                if len(urls_found) > 10:
                    result['warnings'].append(f"Many URLs found ({len(urls_found)})")
                    result['risk_score'] += 15
                else:
                    result['warnings'].append(f"URLs found: {len(urls_found)}")
                    result['risk_score'] += 5

            # Check for encoded content
            suspicious_encoding = [
                r'base64,', r'%[0-9a-f]{2}', r'\\x[0-9a-f]{2}',
                r'\\u[0-9a-f]{4}', r'&#x[0-9a-f]+;'
            ]

            for pattern in suspicious_encoding:
                if re.search(pattern, content_str, re.IGNORECASE):
                    result['warnings'].append(f"Encoded content detected: {pattern}")
                    result['risk_score'] += 10

            # Check for excessive special characters (potential obfuscation)
            special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', content_str)) / max(len(content_str), 1)
            if special_char_ratio > 0.3:
                result['warnings'].append(f"High special character ratio: {special_char_ratio:.2f}")
                result['risk_score'] += 15

        except Exception as e:
            result['warnings'].append(f"Content scanning error: {str(e)}")

        return result

    async def _check_file_hash_reputation(self, file_content: bytes) -> Dict[str, Any]:
        """Check file hash against known malicious files"""
        result = {'safe': True, 'issues': [], 'warnings': []}

        # Calculate file hashes
        md5_hash = hashlib.md5(file_content).hexdigest()
        sha256_hash = hashlib.sha256(file_content).hexdigest()

        # Check against local blacklist
        if md5_hash in self.blocked_hashes or sha256_hash in self.blocked_hashes:
            result['safe'] = False
            result['issues'].append("File matches known malicious hash")

        # In production, integrate with threat intelligence APIs
        # Examples: VirusTotal, Hybrid Analysis, etc.

        return result

    async def _advanced_threat_detection(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """Advanced threat detection techniques"""
        result = {'risk_score': 0, 'warnings': []}

        # Entropy analysis (high entropy may indicate encryption/compression/obfuscation)
        if len(file_content) > 1024:
            sample = file_content[:1024]
            entropy = self._calculate_entropy(sample)

            if entropy > 7.5:  # High entropy threshold
                result['warnings'].append(f"High entropy content: {entropy:.2f}")
                result['risk_score'] += 15

        # File structure analysis
        if filename.endswith('.pdf'):
            # Check for PDF structure anomalies
            if not file_content.startswith(b'%PDF'):
                result['warnings'].append("PDF file without proper header")
                result['risk_score'] += 20

        # Check for polyglot files (files that are valid in multiple formats)
        if self._detect_polyglot(file_content):
            result['warnings'].append("Potential polyglot file detected")
            result['risk_score'] += 25

        return result

    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data"""
        if not data:
            return 0

        entropy = 0
        for i in range(256):
            freq = data.count(i)
            if freq > 0:
                freq = float(freq) / len(data)
                entropy -= freq * (freq.bit_length() - 1)

        return entropy

    def _detect_polyglot(self, file_content: bytes) -> bool:
        """Detect potential polyglot files"""
        # Check for multiple file signatures in the same file
        signatures_found = 0

        common_signatures = [
            b'%PDF', b'\x89PNG', b'\xff\xd8\xff', b'PK\x03\x04',
            b'GIF8', b'BM', b'\x00\x00\x01\x00'
        ]

        for sig in common_signatures:
            if sig in file_content[:1024]:
                signatures_found += 1

        return signatures_found > 1

    async def _quarantine_file(
        self,
        file_content: bytes,
        filename: str,
        validation_result: Dict[str, Any]
    ) -> None:
        """Quarantine suspicious files for analysis"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
            quarantine_filename = f"{timestamp}_{safe_filename}"
            quarantine_path = self.quarantine_dir / quarantine_filename

            # Save file
            with open(quarantine_path, 'wb') as f:
                f.write(file_content)

            # Save analysis report
            report_path = quarantine_path.with_suffix('.report.json')
            import json
            with open(report_path, 'w') as f:
                json.dump({
                    'original_filename': filename,
                    'quarantine_time': timestamp,
                    'validation_result': validation_result,
                    'file_size': len(file_content),
                    'md5': hashlib.md5(file_content).hexdigest(),
                    'sha256': hashlib.sha256(file_content).hexdigest()
                }, f, indent=2)

            logger.warning(f"File quarantined: {filename} -> {quarantine_filename}")

        except Exception as e:
            logger.error(f"Failed to quarantine file {filename}: {e}")

    async def add_to_blacklist(self, file_hash: str) -> None:
        """Add file hash to blacklist"""
        self.blocked_hashes.add(file_hash)
        # In production, persist to database

    async def get_quarantine_files(self) -> List[Dict[str, Any]]:
        """Get list of quarantined files"""
        quarantine_files = []

        try:
            for file_path in self.quarantine_dir.glob("*.report.json"):
                with open(file_path, 'r') as f:
                    import json
                    report = json.load(f)
                    quarantine_files.append(report)
        except Exception as e:
            logger.error(f"Error reading quarantine files: {e}")

        return quarantine_files

    async def clean_quarantine(self, days_old: int = 30) -> None:
        """Clean old quarantine files"""
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        try:
            for file_path in self.quarantine_dir.glob("*"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    logger.info(f"Cleaned quarantine file: {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning quarantine: {e}")

# Global instance
document_security = DocumentSecurityService()