# This file contains the replacement code for multipart parsing

import io
import re
from email import message_from_bytes

def parse_multipart_data(content_type_header, request_body):
    """Parse multipart form data using email module"""
    # Extract boundary from Content-Type header
    boundary_match = re.search(r'boundary=([^;\s]+)', content_type_header)
    if not boundary_match:
        raise ValueError('No boundary found in Content-Type header')
    
    boundary = boundary_match.group(1).strip('"')
    
    # Construct a valid email message by adding headers
    full_message = b'MIME-Version: 1.0\r\nContent-Type: ' + content_type_header.encode() + b'\r\n\r\n' + request_body
    
    msg = message_from_bytes(full_message)
    
    # Extract the image file from the message
    image_data = None
    for part in msg.iter_parts():
        if part.get_filename():
            image_data = part.get_payload(decode=True)
            break
    
    if not image_data:
        raise ValueError('No image file found in multipart data')
    
    return image_data

# Create a simple object to hold file data that mimics FieldStorage fileitem
class FileItem:
    def __init__(self, file_data):
        self.file = io.BytesIO(file_data)
    
    def read(self):
        return self.file.read()
