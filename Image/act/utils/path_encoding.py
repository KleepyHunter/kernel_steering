# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from urllib.parse import quote, unquote


def encode_path_component(value: str) -> str:
    return quote(str(value), safe="")


def decode_path_component(value: str) -> str:
    return unquote(str(value))
