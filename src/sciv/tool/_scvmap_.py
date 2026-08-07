# -*- coding: UTF-8 -*-

import requests
from requests import Response

from ..util import scvmap_url


def get_result_data(resp: Response):
    json_data = resp.json()

    if json_data["status"]:
        return json_data["data"]

    raise ValueError(json_data["message"])


def request_get_data(path: str, **kwargs):
    response = requests.get(f"{scvmap_url}/{path}", **kwargs)
    return get_result_data(response)


def request_post_data(path: str, json: dict = None, **kwargs):
    response = requests.post(f"{scvmap_url}/{path}", json=json, **kwargs)
    return get_result_data(response)
