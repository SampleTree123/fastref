"""Compatibility import for the unchanged, hash-recorded legacy cache encoder.

New callers use encoding_helpers. There is no old smoke-test entry point here.
"""
from encoding_helpers import ROOT, atomic_json, sha_file, shape_check, tensor_info, weight_paths
