import json
import os
from typing import Any

import requests
from google.ai.generativelanguage_v1beta.types import (
    GenerateContentRequest,
    GenerateContentResponse,
)
from google.protobuf.json_format import MessageToDict


class CustomClient:
    def generate_content(
        self, request: GenerateContentRequest, metadata: Any
    ) -> GenerateContentResponse:
        request_dict = MessageToDict(request._pb, preserving_proto_field_name=True)

        tools_json_str = json.dumps(request_dict.get("tools"))
        tools_dict = json.loads(tools_json_str.replace('"type_"', '"type"'))

        payload = {
            "contents": request_dict.get("contents"),
            "tools": tools_dict,
        }
        response = requests.post(
            os.getenv("DEWLAP_QUERY_DEV_URL"),
            headers={
                "content-type": "application/json",
            },
            data=json.dumps(
                {
                    "user_id": os.getenv("ROOT_USER_ID"),
                    "payload": payload,
                }
            ),
        )
        response_json_str = json.dumps(response.json())
        response_dict = json.loads(
            response_json_str.replace('"functionCall"', '"function_call"')
            .replace('"finishReason"', '"finish_reason"')
            .replace('"avgLogprobs"', '"avg_logprobs"')
        )
        response_object = GenerateContentResponse(
            candidates=response_dict.get("candidates", [])
        )
        return response_object


class CustomAsyncClient:
    async def generate_content(
        self, request: GenerateContentRequest, metadata: Any
    ) -> GenerateContentResponse:
        request_dict = MessageToDict(request._pb, preserving_proto_field_name=True)

        tools_json_str = json.dumps(request_dict.get("tools"))
        tools_dict = json.loads(tools_json_str.replace('"type_"', '"type"'))

        payload = {
            "contents": request_dict.get("contents"),
            "tools": tools_dict,
        }
        response = requests.post(
            os.getenv("DEWLAP_QUERY_DEV_URL"),
            headers={
                "content-type": "application/json",
            },
            data=json.dumps(
                {
                    "user_id": os.getenv("ROOT_USER_ID"),
                    "payload": payload,
                }
            ),
        )
        response_json_str = json.dumps(response.json())
        response_dict = json.loads(
            response_json_str.replace('"functionCall"', '"function_call"')
            .replace('"finishReason"', '"finish_reason"')
            .replace('"avgLogprobs"', '"avg_logprobs"')
        )
        response_object = GenerateContentResponse(
            candidates=response_dict.get("candidates", [])
        )
        return response_object
