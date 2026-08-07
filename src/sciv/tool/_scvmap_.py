# -*- coding: UTF-8 -*-

import requests
from requests import Response


def get_result_data(resp: Response):
    json_data = resp.json()

    if json_data["status"]:
        return json_data["data"]

    raise ValueError(json_data["message"])


def request_get_data(url: str, **kwargs):
    response = requests.get(url, **kwargs)
    return get_result_data(response)


def request_post_data(url: str, json: dict = None, **kwargs):
    response = requests.post(url, json=json, **kwargs)
    return get_result_data(response)
