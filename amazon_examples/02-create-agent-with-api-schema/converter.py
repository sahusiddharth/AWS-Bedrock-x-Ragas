from typing import List, Any, Union
from ragas.messages import HumanMessage, AIMessage, ToolMessage, ToolCall
import json
import ast  # To safely evaluate list-like strings into actual lists


def convert_data(data):
    result_dict = {}
    if not data:
        return result_dict

    for item in data:
        key = item.get("name")
        value = item.get("value")
        value_type = item.get("type")

        # Convert value based on its type
        if value_type == "integer":
            result_dict[key] = int(value)
        elif value_type == "float":
            result_dict[key] = float(value)
        elif value_type == "boolean":
            # Convert 'true'/'false' string to a boolean
            result_dict[key] = value.lower() == "true"
        elif value_type == "string":
            result_dict[key] = value  # Keep it as a string
        elif value_type == "list":
            # Safely convert the string representation of a list to an actual list
            result_dict[key] = ast.literal_eval(value)
        elif value_type == "none":
            # If the type is 'none', convert the value to None (useful for 'null' values)
            result_dict[key] = None
        else:
            # If the type is unknown, leave the value unchanged or raise an error
            raise ValueError(f"Unsupported type: {value_type}")

    return result_dict


# TODO: Set metadata to trace
def convert_to_ragas_messages(
    traces: List[Any],
) -> List[Union[HumanMessage, AIMessage, ToolMessage, ToolCall]]:
    ragas_messages = []
    prev_trace = None
    for index, trace in enumerate(traces):
        if "orchestrationTrace" in trace["trace"]:
            if "modelInvocationInput" in trace["trace"]["orchestrationTrace"]:
                if prev_trace is None:
                    message_list = json.loads(
                        trace["trace"]["orchestrationTrace"]["modelInvocationInput"]["text"]
                    )["messages"]
                    if len(message_list) > 1 and message_list[-2]["role"] == "user":
                        last_user_message = message_list[-2]
                        ragas_messages.append(
                            HumanMessage(content=last_user_message["content"])
                        )
            elif "modelInvocationOutput" in trace["trace"]["orchestrationTrace"]:
                if "modelInvocationInput" in prev_trace:
                    pass
            elif "rationale" in trace["trace"]["orchestrationTrace"]:
                if (
                    index + 1 < len(traces)
                    and "observation" in traces[index + 1]["trace"]["orchestrationTrace"]
                ):
                    pass
                else:
                    ai_message = trace["trace"]["orchestrationTrace"]["rationale"]["text"]
                    ragas_messages.append(AIMessage(content=ai_message))
            elif "invocationInput" in trace["trace"]["orchestrationTrace"]:
                tool_calls = []
                if (
                    "actionGroupInvocationInput"
                    in trace["trace"]["orchestrationTrace"]["invocationInput"]
                ):
                    if (
                        "function"
                        in trace["trace"]["orchestrationTrace"]["invocationInput"][
                            "actionGroupInvocationInput"
                        ]
                    ):
                        function_name = trace["trace"]["orchestrationTrace"][
                            "invocationInput"
                        ]["actionGroupInvocationInput"].get("function")

                    elif (
                        "apiPath"
                        in trace["trace"]["orchestrationTrace"]["invocationInput"][
                            "actionGroupInvocationInput"
                        ]
                    ):
                        function_name = trace["trace"]["orchestrationTrace"][
                            "invocationInput"
                        ]["actionGroupInvocationInput"].get("apiPath")
                        verb = trace["trace"]["orchestrationTrace"]["invocationInput"][
                            "actionGroupInvocationInput"
                        ].get("verb")
                        function_name = f"{function_name}.{verb}"

                    parameters = trace["trace"]["orchestrationTrace"]["invocationInput"][
                        "actionGroupInvocationInput"
                    ].get("parameters")
                    arguements = convert_data(parameters)
                    tool_calls.append(ToolCall(name=function_name, args=arguements))
                    ragas_messages[-1].tool_calls = tool_calls
                elif (
                    "knowledgeBaseLookupInput"
                    in trace["trace"]["orchestrationTrace"]["invocationInput"]
                ):
                    if index == 0:
                        human_message = trace["trace"]["orchestrationTrace"][
                            "invocationInput"
                        ]["knowledgeBaseLookupInput"]["text"]
                        ragas_messages.append(HumanMessage(content=human_message))
                    else:
                        # {toolUse={input={searchQuery=...}, name=GET__x_amz_knowledgebase_YBGVFNRXSS__Search}}
                        # {'invocationInput': {'invocationType': 'KNOWLEDGE_BASE', 'knowledgeBaseLookupInput': {'knowledgeBaseId': 'YBGVFNRXSS', 'text': 'components of a Bedrock Guardrail'}, 'traceId': '54620437-3521-4fa3-9578-b024b4fce799-0'}}
                        function_name = f"GET__x_amz_knowledgebase_{trace['trace']['orchestrationTrace']['invocationInput']['knowledgeBaseLookupInput']['knowledgeBaseId']}__Search"
                        arguements = {
                            "searchQuery": trace["trace"]["orchestrationTrace"][
                                "invocationInput"
                            ]["knowledgeBaseLookupInput"]["text"]
                        }
                        tool_calls.append(ToolCall(name=function_name, args=arguements))
                        ragas_messages[-1].tool_calls = tool_calls
                elif "codeInterpreterInvocationInput" in trace["trace"]["orchestrationTrace"]["invocationInput"]:
                    # TODO: complete this codeInterpreterInvocationInput
                    function_name = ""
                    arguements = ""
                    tool_calls.append(ToolCall(name=function_name, args=arguements))
                    ragas_messages[-1].tool_calls = tool_calls
            elif "observation" in trace["trace"]["orchestrationTrace"]:
                if "finalResponse" in trace["trace"]["orchestrationTrace"]["observation"]:
                    ai_message = trace["trace"]["orchestrationTrace"]["observation"][
                        "finalResponse"
                    ].get("text")
                    ragas_messages.append(AIMessage(content=ai_message))
                elif (
                    "actionGroupInvocationOutput"
                    in trace["trace"]["orchestrationTrace"]["observation"]
                ):
                    tool_message = trace["trace"]["orchestrationTrace"]["observation"][
                        "actionGroupInvocationOutput"
                    ].get("text")
                    ragas_messages.append(ToolMessage(content=tool_message))
                elif (
                    "knowledgeBaseLookupOutput"
                    in trace["trace"]["orchestrationTrace"]["observation"]
                ):
                    tool_message = ""
                    for context in trace["trace"]["orchestrationTrace"]["observation"][
                        "knowledgeBaseLookupOutput"
                    ]["retrievedReferences"]:
                        tool_message += context.get("content").get("text") + "\n\n"
                    ragas_messages.append(ToolMessage(content=tool_message))
                elif (
                    "codeInterpreterInvocationOutput"
                    in trace["trace"]["orchestrationTrace"]["observation"]
                ):
                    tool_message = trace["trace"]["orchestrationTrace"]["observation"]["codeInterpreterInvocationOutput"].get("executionOutput")
                    ragas_messages.append(ToolMessage(content=tool_message))
            prev_trace = trace["trace"]["orchestrationTrace"]
        elif "guardrailTrace" in trace["trace"]:
            pass
    return ragas_messages
