# -*- coding: UTF-8 -*-

import requests
from requests import Response

from .. import util as ul

from ..util import scvmap_url

__name__: str = "tool_scvmap"

log = ul.log(__name__)


def get_result_data(resp: Response):
    json_data = resp.json()

    if json_data["status"]:
        return json_data["data"]

    raise ValueError(json_data["message"])


def request_get_data(path: str, **kwargs):
    log.info(f"Request {scvmap_url}/{path}")
    response = requests.get(f"{scvmap_url}/{path}", **kwargs)
    return get_result_data(response)


def request_post_data(path: str, json: dict = None, **kwargs):
    log.info(f"Request {scvmap_url}/{path}")
    response = requests.post(f"{scvmap_url}/{path}", json=json, **kwargs)
    return get_result_data(response)
